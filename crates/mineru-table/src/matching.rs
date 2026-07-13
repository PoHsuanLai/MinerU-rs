//! Cell-bbox → OCR matching and HTML assembly for the wireless (SLANet) path.
//!
//! Port of `slanet_plus/matcher.py`'s `TableMatch`. Given the structure token
//! stream the SLANet decoder emits, the per-`<td>` cell bounding boxes it
//! regresses, and the OCR spans for the same crop, this assigns each OCR span to
//! its best-fitting cell and splices the recognized text back into the HTML
//! skeleton described by the token stream.
//!
//! The numpy vectorized helpers in Python (`_pairwise_iou_and_distance`,
//! `_select_best_cell_indices`) are re-expressed as straightforward loops here;
//! the tables are small enough that the O(n·m) scan is fine and the result is
//! bit-for-bit the same selection rule (minimize `1 - IoU`, tie-break on the
//! Python `distance` score).

use mineru_types::{BBox, Html};

use crate::ocr::OcrSpan;

/// The IoU floor from Python (`min_iou = 0.1**8`): a span whose best cell IoU is
/// below this is left unassigned.
const MIN_IOU: f64 = 1e-8;

/// Matches OCR spans to predicted cells and renders the final table HTML.
///
/// Mirrors `TableMatch(filter_ocr_result=True)`.
#[derive(Debug, Clone)]
pub struct TableMatch {
    /// When true, drop OCR spans that sit entirely above the topmost cell (the
    /// Python `_filter_ocr_result` step).
    pub filter_ocr_result: bool,
}

impl Default for TableMatch {
    fn default() -> Self {
        Self {
            filter_ocr_result: true,
        }
    }
}

impl TableMatch {
    /// Runs the full match + assemble pipeline.
    ///
    /// * `structures` — the structure token stream (e.g. `<html>`, `<td></td>`,
    ///   `<td`, ` colspan="2"`, `>`, `</td>`, `<tr>`, ...), already wrapped in the
    ///   `<html><body><table> ... </table></body></html>` frame.
    /// * `cell_bboxes` — one axis-aligned box per emitted `<td>` token, in the
    ///   same order the tokens appear.
    /// * `spans` — OCR detections for the crop.
    pub fn run(&self, structures: &[String], cell_bboxes: &[BBox], spans: &[OcrSpan]) -> Html {
        let (spans, cell_bboxes) = if self.filter_ocr_result {
            filter_ocr_result(cell_bboxes, spans)
        } else {
            (spans.to_vec(), cell_bboxes.to_vec())
        };
        let matched = match_result(&spans, &cell_bboxes);
        Html(get_pred_html(structures, &matched, &spans))
    }
}

/// The Python `distance(box1, box2)` metric used to break IoU ties.
fn distance(a: &BBox, b: &BBox) -> f64 {
    let (ax0, ay0, ax1, ay1) = (a.x0 as f64, a.y0 as f64, a.x1 as f64, a.y1 as f64);
    let (bx0, by0, bx1, by1) = (b.x0 as f64, b.y0 as f64, b.x1 as f64, b.y1 as f64);
    let dis = (bx0 - ax0).abs() + (by0 - ay0).abs() + (bx1 - ax1).abs() + (by1 - ay1).abs();
    let dis_2 = (bx0 - ax0).abs() + (by0 - ay0).abs();
    let dis_3 = (bx1 - ax1).abs() + (by1 - ay1).abs();
    dis + dis_2.min(dis_3)
}

/// IoU with the `sum_area - intersect` union convention used by the matcher.
fn iou(a: &BBox, b: &BBox) -> f64 {
    let left = (a.y0 as f64).max(b.y0 as f64);
    let right = (a.y1 as f64).min(b.y1 as f64);
    let top = (a.x0 as f64).max(b.x0 as f64);
    let bottom = (a.x1 as f64).min(b.x1 as f64);
    if left >= right || top >= bottom {
        return 0.0;
    }
    let intersect = (right - left) * (bottom - top);
    let sum = (a.x1 as f64 - a.x0 as f64) * (a.y1 as f64 - a.y0 as f64)
        + (b.x1 as f64 - b.x0 as f64) * (b.y1 as f64 - b.y0 as f64);
    let union = sum - intersect;
    if union == 0.0 {
        0.0
    } else {
        intersect / union
    }
}

/// Assigns each OCR span to its best cell. Returns a map `cell_index -> [span
/// indices]`, preserving the Python selection and threshold semantics.
fn match_result(spans: &[OcrSpan], cell_bboxes: &[BBox]) -> Vec<(usize, Vec<usize>)> {
    // Insertion-ordered map keyed by cell index (small n, so a Vec is fine and
    // keeps ordering deterministic like Python dict insertion order).
    let mut matched: Vec<(usize, Vec<usize>)> = Vec::new();
    if spans.is_empty() || cell_bboxes.is_empty() {
        return matched;
    }

    for (span_idx, span) in spans.iter().enumerate() {
        // Find the cell minimizing (1 - IoU), tie-broken by distance.
        let mut best_cell = 0usize;
        let mut best_inv_iou = f64::INFINITY;
        let mut best_distance = f64::INFINITY;
        for (cell_idx, cell) in cell_bboxes.iter().enumerate() {
            let inv_iou = 1.0 - iou(&span.bbox, cell);
            let dist = distance(cell, &span.bbox);
            if inv_iou < best_inv_iou || (inv_iou == best_inv_iou && dist < best_distance) {
                best_inv_iou = inv_iou;
                best_distance = dist;
                best_cell = cell_idx;
            }
        }
        // Python: skip when best_inv_iou >= 1 - min_iou.
        if best_inv_iou >= 1.0 - MIN_IOU {
            continue;
        }
        match matched.iter_mut().find(|(c, _)| *c == best_cell) {
            Some((_, v)) => v.push(span_idx),
            None => matched.push((best_cell, vec![span_idx])),
        }
    }
    matched
}

/// Splices matched OCR text into the token stream, mirroring
/// `TableMatch.get_pred_html`.
fn get_pred_html(
    structures: &[String],
    matched: &[(usize, Vec<usize>)],
    spans: &[OcrSpan],
) -> String {
    let lookup = |cell: usize| matched.iter().find(|(c, _)| *c == cell).map(|(_, v)| v);

    let mut out = String::new();
    let mut td_index = 0usize;

    for tag in structures {
        if !tag.contains("</td>") {
            out.push_str(tag);
            continue;
        }

        // A `<td></td>` token opens with `<td>` here and closes below.
        if tag == "<td></td>" {
            out.push_str("<td>");
        }

        if let Some(span_indices) = lookup(td_index) {
            let mut b_with = false;
            if let Some(&first) = span_indices.first() {
                if spans[first].text.contains("<b>") && span_indices.len() > 1 {
                    b_with = true;
                    out.push_str("<b>");
                }
            }

            for (i, &si) in span_indices.iter().enumerate() {
                let mut content = spans[si].text.clone();
                if span_indices.len() > 1 {
                    if content.is_empty() {
                        continue;
                    }
                    if content.starts_with(' ') {
                        content.remove(0);
                    }
                    if content.contains("<b>") {
                        content = content.chars().skip(3).collect();
                    }
                    if content.contains("</b>") {
                        let n = content.chars().count();
                        content = content.chars().take(n.saturating_sub(4)).collect();
                    }
                    if content.is_empty() {
                        continue;
                    }
                    if i != span_indices.len() - 1 && !content.ends_with(' ') {
                        content.push(' ');
                    }
                }
                out.push_str(&content);
            }

            if b_with {
                out.push_str("</b>");
            }
        }

        if tag == "<td></td>" {
            out.push_str("</td>");
        } else {
            out.push_str(tag);
        }

        td_index += 1;
    }

    // Filter out thead/tbody wrapper tokens (Python does the same).
    for f in ["<thead>", "</thead>", "<tbody>", "</tbody>"] {
        out = out.replace(f, "");
    }
    out
}

/// Drops OCR spans lying entirely above the topmost predicted cell.
///
/// Port of `_filter_ocr_result`: `y1` is the minimum y over all cell boxes; a
/// span whose maximum y is below that is discarded.
fn filter_ocr_result(cell_bboxes: &[BBox], spans: &[OcrSpan]) -> (Vec<OcrSpan>, Vec<BBox>) {
    if cell_bboxes.is_empty() {
        return (spans.to_vec(), cell_bboxes.to_vec());
    }
    let y1 = cell_bboxes
        .iter()
        .flat_map(|b| [b.y0, b.y1])
        .fold(f32::INFINITY, f32::min);
    let kept = spans
        .iter()
        .filter(|s| s.bbox.y1 >= y1)
        .cloned()
        .collect();
    (kept, cell_bboxes.to_vec())
}

/// A single cell's logical position: `[row_start, row_end, col_start, col_end]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicPoint {
    /// Inclusive first row index.
    pub row_start: usize,
    /// Inclusive last row index (accounts for rowspan).
    pub row_end: usize,
    /// Inclusive first column index.
    pub col_start: usize,
    /// Inclusive last column index (accounts for colspan).
    pub col_end: usize,
}

/// Decodes each cell's logical grid position from the structure token stream.
///
/// Port of `TableMatch.decode_logic_points`. Walks the tokens, tracking the
/// current row/column and honouring `rowspan`/`colspan` attributes, skipping
/// grid positions already occupied by an earlier spanning cell.
pub fn decode_logic_points(structures: &[String]) -> Vec<LogicPoint> {
    let mut logic_points = Vec::new();
    let mut current_row = 0usize;
    let mut current_col = 0usize;
    // Occupied grid cells, tracked as a set of (row, col).
    let mut occupied = std::collections::HashSet::new();

    let mut i = 0usize;
    while i < structures.len() {
        let token = &structures[i];
        if token == "<tr>" {
            current_col = 0;
        } else if token == "</tr>" {
            current_row += 1;
        } else if token.starts_with("<td") {
            let mut colspan = 1usize;
            let mut rowspan = 1usize;
            let mut j = i;
            if token != "<td></td>" {
                j += 1;
                while j < structures.len() && !structures[j].starts_with('>') {
                    if let Some(v) = parse_span(&structures[j], "colspan=") {
                        colspan = v;
                    } else if let Some(v) = parse_span(&structures[j], "rowspan=") {
                        rowspan = v;
                    }
                    j += 1;
                }
            }
            i = j;

            while occupied.contains(&(current_row, current_col)) {
                current_col += 1;
            }

            let r_start = current_row;
            let r_end = current_row + rowspan - 1;
            let c_start = current_col;
            let c_end = current_col + colspan - 1;

            logic_points.push(LogicPoint {
                row_start: r_start,
                row_end: r_end,
                col_start: c_start,
                col_end: c_end,
            });

            for r in r_start..=r_end {
                for c in c_start..=c_end {
                    occupied.insert((r, c));
                }
            }
            current_col += colspan;
        }
        i += 1;
    }
    logic_points
}

/// Extracts the integer following `key` (e.g. `colspan="2"`) from a token.
fn parse_span(token: &str, key: &str) -> Option<usize> {
    let rest = token.split(key).nth(1)?;
    let trimmed = rest.trim_matches(|c| c == '"' || c == '\'' || c == ' ' || c == '>');
    trimmed.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn matches_single_span_to_overlapping_cell() {
        let structures = s(&["<table>", "<tr>", "<td></td>", "<td></td>", "</tr>", "</table>"]);
        // Two side-by-side cells.
        let cells = vec![
            BBox::new(0.0, 0.0, 10.0, 10.0),
            BBox::new(10.0, 0.0, 20.0, 10.0),
        ];
        // One span squarely inside the second cell.
        let spans = vec![OcrSpan::new(BBox::new(11.0, 1.0, 19.0, 9.0), "hi", 0.9)];
        let m = TableMatch {
            filter_ocr_result: false,
        };
        let html = m.run(&structures, &cells, &spans);
        assert_eq!(html.0, "<table><tr><td></td><td>hi</td></tr></table>");
    }

    #[test]
    fn empty_cell_left_empty() {
        let structures = s(&["<td></td>"]);
        let cells = vec![BBox::new(0.0, 0.0, 10.0, 10.0)];
        let spans: Vec<OcrSpan> = vec![];
        let m = TableMatch {
            filter_ocr_result: false,
        };
        assert_eq!(m.run(&structures, &cells, &spans).0, "<td></td>");
    }

    #[test]
    fn strips_thead_tbody_wrappers() {
        let structures = s(&["<thead>", "<td></td>", "</thead>", "<tbody>", "</tbody>"]);
        let cells = vec![BBox::new(0.0, 0.0, 10.0, 10.0)];
        let spans = vec![OcrSpan::new(BBox::new(1.0, 1.0, 9.0, 9.0), "x", 0.9)];
        let m = TableMatch {
            filter_ocr_result: false,
        };
        assert_eq!(m.run(&structures, &cells, &spans).0, "<td>x</td>");
    }

    #[test]
    fn joins_multiple_spans_in_one_cell_with_spaces() {
        let structures = s(&["<td></td>"]);
        let cells = vec![BBox::new(0.0, 0.0, 100.0, 10.0)];
        let spans = vec![
            OcrSpan::new(BBox::new(0.0, 0.0, 40.0, 10.0), "foo", 0.9),
            OcrSpan::new(BBox::new(50.0, 0.0, 90.0, 10.0), "bar", 0.9),
        ];
        let m = TableMatch {
            filter_ocr_result: false,
        };
        // Both land in the single cell; first gets a trailing space appended.
        assert_eq!(m.run(&structures, &cells, &spans).0, "<td>foo bar</td>");
    }

    #[test]
    fn decode_logic_points_simple_grid() {
        // 2x2 grid.
        let structures = s(&[
            "<tr>", "<td></td>", "<td></td>", "</tr>", "<tr>", "<td></td>", "<td></td>", "</tr>",
        ]);
        let pts = decode_logic_points(&structures);
        assert_eq!(pts.len(), 4);
        assert_eq!(
            pts[0],
            LogicPoint {
                row_start: 0,
                row_end: 0,
                col_start: 0,
                col_end: 0
            }
        );
        assert_eq!(
            pts[3],
            LogicPoint {
                row_start: 1,
                row_end: 1,
                col_start: 1,
                col_end: 1
            }
        );
    }

    #[test]
    fn decode_logic_points_honours_colspan() {
        // First cell spans two columns, second is a normal cell.
        let structures = s(&[
            "<tr>",
            "<td",
            " colspan=\"2\"",
            ">",
            "</td>",
            "<td></td>",
            "</tr>",
            "<tr>",
            "<td></td>",
            "<td></td>",
            "<td></td>",
            "</tr>",
        ]);
        let pts = decode_logic_points(&structures);
        // Cell 0: row0 cols 0..1 (colspan 2).
        assert_eq!(
            pts[0],
            LogicPoint {
                row_start: 0,
                row_end: 0,
                col_start: 0,
                col_end: 1
            }
        );
        // Cell 1: row0 col2.
        assert_eq!(
            pts[1],
            LogicPoint {
                row_start: 0,
                row_end: 0,
                col_start: 2,
                col_end: 2
            }
        );
    }

    #[test]
    fn decode_logic_points_honours_rowspan() {
        let structures = s(&[
            "<tr>",
            "<td",
            " rowspan=\"2\"",
            ">",
            "</td>",
            "<td></td>",
            "</tr>",
            "<tr>",
            "<td></td>",
            "</tr>",
        ]);
        let pts = decode_logic_points(&structures);
        // Cell 0 spans rows 0..1 in col 0.
        assert_eq!(
            pts[0],
            LogicPoint {
                row_start: 0,
                row_end: 1,
                col_start: 0,
                col_end: 0
            }
        );
        // Cell 2 (second row's only td) must skip col 0 (occupied) -> col 1.
        assert_eq!(
            pts[2],
            LogicPoint {
                row_start: 1,
                row_end: 1,
                col_start: 1,
                col_end: 1
            }
        );
    }

    #[test]
    fn filter_drops_spans_above_all_cells() {
        let cells = vec![BBox::new(0.0, 100.0, 50.0, 150.0)];
        let spans = vec![
            OcrSpan::new(BBox::new(0.0, 0.0, 10.0, 10.0), "above", 0.9),
            OcrSpan::new(BBox::new(0.0, 110.0, 10.0, 120.0), "inside", 0.9),
        ];
        let (kept, _) = filter_ocr_result(&cells, &spans);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].text, "inside");
    }
}
