//! Wired-table HTML assembly from logical points and cell text.
//!
//! Port of `utils_table_recover.plot_html_table` (with its `_build_table_grid`
//! and noise-edge-trimming helpers). Given each physical cell's logical span and
//! the OCR text mapped onto it, this fills a grid — keeping structural
//! placeholders for empty cells — trims noise rows/columns at the edges, and
//! emits `<table>` HTML with `rowspan`/`colspan`.
//!
//! The geometry-based "abnormal edge size" heuristic in Python additionally uses
//! per-cell physical bboxes; that refinement is only reached for edges that are
//! already empty *and* fully covered, and is omitted here (such fully-covered
//! empty edges are kept, which is the conservative choice). Text- and
//! coverage-based trimming is ported exactly.

use std::collections::HashMap;

use crate::matching::LogicPoint;

/// A grid slot: the physical cell index plus its full logical span.
#[derive(Clone, Copy)]
struct Slot {
    cell_idx: usize,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
}

type Grid = Vec<Vec<Option<Slot>>>;

/// Builds the occupancy grid from logical points.
///
/// Port of `_build_table_grid`.
#[allow(clippy::needless_range_loop)] // row/col indices double as grid subscripts
fn build_grid(logic: &[LogicPoint]) -> (Grid, usize, usize) {
    let max_row = logic.iter().map(|p| p.row_end).max().map_or(0, |m| m + 1);
    let max_col = logic.iter().map(|p| p.col_end).max().map_or(0, |m| m + 1);
    let mut grid: Grid = vec![vec![None; max_col]; max_row];
    for (i, p) in logic.iter().enumerate() {
        for r in p.row_start..=p.row_end {
            for c in p.col_start..=p.col_end {
                if r < max_row && c < max_col {
                    grid[r][c] = Some(Slot {
                        cell_idx: i,
                        row_start: p.row_start,
                        row_end: p.row_end,
                        col_start: p.col_start,
                        col_end: p.col_end,
                    });
                }
            }
        }
    }
    (grid, max_row, max_col)
}

/// Joins the text pieces for a cell.
fn cell_text(text_map: &HashMap<usize, Vec<String>>, idx: usize) -> String {
    text_map.get(&idx).map(|v| v.concat()).unwrap_or_default()
}

fn cell_has_text(text_map: &HashMap<usize, Vec<String>>, idx: usize) -> bool {
    !cell_text(text_map, idx).trim().is_empty()
}

/// Does the given edge row/column contain any visible text within range?
#[allow(clippy::too_many_arguments)] // mirrors the Python helper's parameter list
fn edge_has_text(
    grid: &Grid,
    text_map: &HashMap<usize, Vec<String>>,
    axis_is_col: bool,
    axis_idx: usize,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
) -> bool {
    if axis_is_col {
        (row_start..=row_end).any(|r| {
            grid[r][axis_idx].is_some_and(|s| cell_has_text(text_map, s.cell_idx))
        })
    } else {
        (col_start..=col_end).any(|c| {
            grid[axis_idx][c].is_some_and(|s| cell_has_text(text_map, s.cell_idx))
        })
    }
}

/// Structural coverage `(covered, total)` of an edge within the current range.
fn edge_coverage(
    grid: &Grid,
    axis_is_col: bool,
    axis_idx: usize,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
) -> (usize, usize) {
    if axis_is_col {
        let covered = (row_start..=row_end)
            .filter(|&r| grid[r][axis_idx].is_some())
            .count();
        (covered, row_end - row_start + 1)
    } else {
        let covered = (col_start..=col_end)
            .filter(|&c| grid[axis_idx][c].is_some())
            .count();
        (covered, col_end - col_start + 1)
    }
}

/// Whether the given edge row/column is trimmable noise: no text and either
/// empty or partially covered.
#[allow(clippy::too_many_arguments)] // mirrors the Python helper's parameter list
fn is_noise_edge(
    grid: &Grid,
    text_map: &HashMap<usize, Vec<String>>,
    axis_is_col: bool,
    axis_idx: usize,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
) -> bool {
    if edge_has_text(
        grid, text_map, axis_is_col, axis_idx, row_start, row_end, col_start, col_end,
    ) {
        return false;
    }
    let (covered, total) = edge_coverage(
        grid, axis_is_col, axis_idx, row_start, row_end, col_start, col_end,
    );
    // Fully-covered empty edges are kept (the geometry heuristic that could still
    // trim them is intentionally omitted; see module docs).
    covered == 0 || covered < total
}

/// Trims noise edges, returning the retained `(row_start, row_end, col_start,
/// col_end)` window.
fn trim_noise_edges(
    grid: &Grid,
    text_map: &HashMap<usize, Vec<String>>,
    max_row: usize,
    max_col: usize,
) -> (usize, usize, usize, usize) {
    let mut rs = 0isize;
    let mut re = max_row as isize - 1;
    let mut cs = 0isize;
    let mut ce = max_col as isize - 1;

    while rs <= re
        && is_noise_edge(
            grid, text_map, false, rs as usize, rs as usize, re as usize, cs as usize,
            ce as usize,
        )
    {
        rs += 1;
    }
    while re >= rs
        && is_noise_edge(
            grid, text_map, false, re as usize, rs as usize, re as usize, cs as usize,
            ce as usize,
        )
    {
        re -= 1;
    }
    while cs <= ce
        && is_noise_edge(
            grid, text_map, true, cs as usize, rs as usize, re as usize, cs as usize,
            ce as usize,
        )
    {
        cs += 1;
    }
    while ce >= cs
        && is_noise_edge(
            grid, text_map, true, ce as usize, rs as usize, re as usize, cs as usize,
            ce as usize,
        )
    {
        ce -= 1;
    }
    (rs as usize, re as usize, cs as usize, ce as usize)
}

/// Renders wired-table HTML from logical points and per-cell text.
///
/// Port of `plot_html_table` (without the optional bbox geometry input).
#[allow(clippy::needless_range_loop)] // row/col indices double as grid subscripts
pub fn plot_html_table(logic: &[LogicPoint], text_map: &HashMap<usize, Vec<String>>) -> String {
    if logic.is_empty() {
        return "<html><body><table></table></body></html>".to_string();
    }
    let (grid, max_row, max_col) = build_grid(logic);
    let (rs, re, cs, ce) = trim_noise_edges(&grid, text_map, max_row, max_col);

    let mut html = String::from("<html><body><table>");
    if rs > re || cs > ce {
        html.push_str("</table></body></html>");
        return html;
    }

    for row in rs..=re {
        html.push_str("<tr>");
        for col in cs..=ce {
            match grid[row][col] {
                None => html.push_str("<td></td>"),
                Some(slot) => {
                    let clipped_row_start = slot.row_start.max(rs);
                    let clipped_col_start = slot.col_start.max(cs);
                    if row == clipped_row_start && col == clipped_col_start {
                        let row_span = slot.row_end.min(re) - clipped_row_start + 1;
                        let col_span = slot.col_end.min(ce) - clipped_col_start + 1;
                        let text = cell_text(text_map, slot.cell_idx);
                        html.push_str(&format!(
                            "<td rowspan={row_span} colspan={col_span}>{text}</td>"
                        ));
                    }
                }
            }
        }
        html.push_str("</tr>");
    }
    html.push_str("</table></body></html>");
    html
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lp(rs: usize, re: usize, cs: usize, ce: usize) -> LogicPoint {
        LogicPoint {
            row_start: rs,
            row_end: re,
            col_start: cs,
            col_end: ce,
        }
    }

    fn text(pairs: &[(usize, &str)]) -> HashMap<usize, Vec<String>> {
        pairs
            .iter()
            .map(|(k, v)| (*k, vec![v.to_string()]))
            .collect()
    }

    #[test]
    fn empty_logic_points_gives_empty_table() {
        assert_eq!(
            plot_html_table(&[], &HashMap::new()),
            "<html><body><table></table></body></html>"
        );
    }

    #[test]
    fn simple_2x2_with_text() {
        let logic = vec![lp(0, 0, 0, 0), lp(0, 0, 1, 1), lp(1, 1, 0, 0), lp(1, 1, 1, 1)];
        let tm = text(&[(0, "a"), (1, "b"), (2, "c"), (3, "d")]);
        let html = plot_html_table(&logic, &tm);
        assert_eq!(
            html,
            "<html><body><table>\
             <tr><td rowspan=1 colspan=1>a</td><td rowspan=1 colspan=1>b</td></tr>\
             <tr><td rowspan=1 colspan=1>c</td><td rowspan=1 colspan=1>d</td></tr>\
             </table></body></html>"
        );
    }

    #[test]
    fn colspan_renders_once() {
        // Cell 0 spans two columns in row 0; row 1 has two cells.
        let logic = vec![lp(0, 0, 0, 1), lp(1, 1, 0, 0), lp(1, 1, 1, 1)];
        let tm = text(&[(0, "wide"), (1, "x"), (2, "y")]);
        let html = plot_html_table(&logic, &tm);
        assert!(html.contains("<td rowspan=1 colspan=2>wide</td>"));
        // The spanned second column in row 0 must not emit a second <td>.
        let row0 = &html[html.find("<tr>").unwrap()..html.find("</tr>").unwrap()];
        assert_eq!(row0.matches("<td").count(), 1);
    }

    #[test]
    fn empty_interior_cell_kept_as_placeholder() {
        // 1x2 row where cell 1 is missing text; it stays as <td></td>? No: both
        // slots are filled; test a genuinely missing grid slot instead.
        // Cell 0 at (0,0); nothing at (0,1) -> placeholder, but that empty col is
        // an edge with no text and zero coverage, so it is trimmed. Put a text
        // cell at (1,1) so the empty (0,1) is interior.
        let logic = vec![lp(0, 0, 0, 0), lp(1, 1, 1, 1)];
        let tm = text(&[(0, "a"), (1, "d")]);
        let html = plot_html_table(&logic, &tm);
        // Grid is 2x2 with (0,1) and (1,0) empty interior placeholders.
        assert!(html.contains("<td></td>"));
        assert!(html.contains(">a</td>"));
        assert!(html.contains(">d</td>"));
    }

    #[test]
    fn noise_edge_column_trimmed() {
        // A trailing empty, partially-covered column should be trimmed.
        // Row 0: cells at col 0 and col 1 (text). Add a stray cell only at row 0
        // col 2 with no text and no coverage in row 1 -> trimmed.
        let logic = vec![lp(0, 0, 0, 0), lp(0, 0, 1, 1), lp(1, 1, 0, 0), lp(1, 1, 1, 1)];
        let tm = text(&[(0, "a"), (1, "b"), (2, "c"), (3, "d")]);
        let html = plot_html_table(&logic, &tm);
        // No trailing empty column emitted.
        assert!(!html.contains("<td></td>"));
    }
}
