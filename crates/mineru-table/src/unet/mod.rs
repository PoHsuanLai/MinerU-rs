//! Wired-table recognition (UNet line segmentation + line recovery).
//!
//! The Python path is: UNet segments the table's ruling lines into a 3-class mask
//! (background / horizontal / vertical), classical morphology + connected
//! components turn that mask into cell quadrilaterals, [`recover`] infers the
//! logical grid, OCR text is matched onto the cells, and [`plot_html_table`]
//! emits the HTML.
//!
//! ## Status
//!
//! - The UNet itself is a plain conv/convtranspose/concat/sigmoid encoder-decoder
//!   and imports cleanly via `burn-onnx` codegen behind the `onnx-import`
//!   feature; [`UnetModel`] wraps the generated segmentation network and reports
//!   [`crate::error::Error::ModelUnavailable`] when built without it.
//! - The mask → polygon line-extraction stage (morphology, 8-connectivity
//!   labeling, min-area-rect) is ported in [`extract`]; it reuses the hoisted
//!   `cv2.minAreaRect` port and hand-rolls only the anisotropic 1-D line-kernel
//!   close that isotropic `imageproc` morphology cannot express.
//! - [`recover`] (grid inference) and [`plot_html_table`] (HTML assembly) are
//!   fully ported and unit-tested, and [`recover_and_render`] runs them together
//!   given cell polygons + matched text — the whole post-segmentation pipeline.

pub mod extract;
pub mod model;
pub mod postprocess;
pub mod recover;

pub use postprocess::plot_html_table;
pub use recover::{recover, LogicPoint, Poly};

use std::collections::HashMap;

use image::RgbImage;
use mineru_types::Html;

use crate::error::Result;
use crate::ocr::OcrSpan;

/// Default row grouping threshold (pixels), from Python `row_threshold=10`.
pub const ROW_THRESHOLD: f32 = 10.0;
/// Default column reconciliation threshold (pixels), from `col_threshold=15`.
pub const COL_THRESHOLD: f32 = 15.0;

/// Runs grid recovery + HTML rendering given cell polygons and per-cell text.
///
/// This is the post-segmentation half of the wired pipeline, independent of the
/// neural network: it takes the recovered cell quadrilaterals and a
/// `cell_index -> text pieces` map (as produced by matching OCR onto cells) and
/// returns the table HTML.
pub fn recover_and_render(
    polygons: &[Poly],
    text_map: &HashMap<usize, Vec<String>>,
    rows_thresh: f32,
    col_thresh: f32,
) -> Html {
    let logic = recover(polygons, rows_thresh, col_thresh);
    Html(plot_html_table(&logic, text_map))
}

/// Recognizes a wired-table crop into HTML.
///
/// Requires a loaded [`model::UnetModel`]; without weights it returns
/// [`crate::error::Error::ModelUnavailable`]. The recovery and rendering steps it
/// would call are exercised directly via [`recover_and_render`] in tests.
pub fn recognize_wired(model: &model::UnetModel, img: &RgbImage, spans: &[OcrSpan]) -> Result<Html> {
    let polygons = model.segment_cells(img)?;
    // Match OCR spans onto recovered cells (axis-aligned overlap), then render.
    let text_map = match_spans_to_cells(&polygons, spans);
    Ok(recover_and_render(
        &polygons,
        &text_map,
        ROW_THRESHOLD,
        COL_THRESHOLD,
    ))
}

/// Assigns OCR spans to their most-overlapping cell polygon (axis-aligned),
/// producing the `cell_index -> [text]` map the renderer consumes.
///
/// This mirrors the intent of `match_ocr_cell` at a coarse grain (best overlap
/// ratio wins); the full Python cross-cell splitting heuristics are not ported.
fn match_spans_to_cells(polygons: &[Poly], spans: &[OcrSpan]) -> HashMap<usize, Vec<String>> {
    let mut map: HashMap<usize, Vec<String>> = HashMap::new();
    for span in spans {
        let mut best: Option<(usize, f32)> = None;
        for (i, poly) in polygons.iter().enumerate() {
            let (x0, y0, x1, y1) = poly_extent(poly);
            let ix0 = span.bbox.x0.max(x0);
            let iy0 = span.bbox.y0.max(y0);
            let ix1 = span.bbox.x1.min(x1);
            let iy1 = span.bbox.y1.min(y1);
            if ix1 <= ix0 || iy1 <= iy0 {
                continue;
            }
            let inter = (ix1 - ix0) * (iy1 - iy0);
            let span_area = span.bbox.area().max(1e-6);
            let ratio = inter / span_area;
            if best.is_none_or(|(_, r)| ratio > r) {
                best = Some((i, ratio));
            }
        }
        if let Some((idx, _)) = best {
            map.entry(idx).or_default().push(span.text.clone());
        }
    }
    map
}

/// Axis-aligned extent `(x0, y0, x1, y1)` of a quadrilateral.
fn poly_extent(poly: &Poly) -> (f32, f32, f32, f32) {
    let xs = [poly[0][0], poly[1][0], poly[2][0], poly[3][0]];
    let ys = [poly[0][1], poly[1][1], poly[2][1], poly[3][1]];
    (
        xs.iter().copied().fold(f32::INFINITY, f32::min),
        ys.iter().copied().fold(f32::INFINITY, f32::min),
        xs.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        ys.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mineru_types::BBox;

    fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Poly {
        [[x0, y0], [x0, y1], [x1, y1], [x1, y0]]
    }

    #[test]
    fn recover_and_render_2x2() {
        let polys = vec![
            rect(0.0, 0.0, 10.0, 20.0),
            rect(10.0, 0.0, 20.0, 20.0),
            rect(0.0, 20.0, 10.0, 40.0),
            rect(10.0, 20.0, 20.0, 40.0),
        ];
        let mut tm = HashMap::new();
        for (i, t) in ["a", "b", "c", "d"].iter().enumerate() {
            tm.insert(i, vec![t.to_string()]);
        }
        let html = recover_and_render(&polys, &tm, ROW_THRESHOLD, COL_THRESHOLD);
        assert!(html.0.contains(">a</td>"));
        assert!(html.0.contains(">d</td>"));
    }

    #[test]
    fn match_spans_picks_overlapping_cell() {
        let polys = vec![rect(0.0, 0.0, 10.0, 10.0), rect(10.0, 0.0, 20.0, 10.0)];
        let spans = vec![OcrSpan::new(BBox::new(11.0, 1.0, 19.0, 9.0), "hi", 0.9)];
        let map = match_spans_to_cells(&polys, &spans);
        assert_eq!(map.get(&1).map(|v| v.concat()), Some("hi".to_string()));
        assert!(!map.contains_key(&0));
    }
}
