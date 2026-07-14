//! End-to-end integration tests that exercise the *real* generated neural
//! networks (LCNet classifier + UNet segmenter).
//!
//! The models are always compiled in; running them triggers a runtime weight
//! fetch (or reuses the cache under `MINERU_MODELS_DIR`). They are `#[ignore]`d
//! because the ndarray CPU forward pass is slow. Run with:
//!
//! ```text
//! MINERU_MODELS_DIR=/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0 \
//!   cargo test -p mineru-table --test real_models -- --ignored --nocapture
//! ```

use image::{Rgb, RgbImage};

/// Builds a synthetic RGB image with a simple grid pattern (loosely table-like).
fn synthetic_table(w: u32, h: u32) -> RgbImage {
    let mut img = RgbImage::from_pixel(w, h, Rgb([255, 255, 255]));
    // Draw a few horizontal and vertical black lines to look table-ish.
    for y in (0..h).step_by((h / 4).max(1) as usize) {
        for x in 0..w {
            img.put_pixel(x, y.min(h - 1), Rgb([0, 0, 0]));
        }
    }
    for x in (0..w).step_by((w / 4).max(1) as usize) {
        for y in 0..h {
            img.put_pixel(x.min(w - 1), y, Rgb([0, 0, 0]));
        }
    }
    img
}

#[test]
#[ignore = "requires MINERU_MODELS_DIR or network for the weight fetch; slow ndarray forward"]
fn classify_runs_real_lcnet_forward() {
    use mineru_table::cls::classify;

    let img = synthetic_table(300, 300);
    let res = classify(&img);

    // Must NOT be ModelUnavailable — the real forward pass must run.
    let c = res.expect("classify should return Ok with the generated model wired");
    println!("classification: {c:?}");

    // A valid class (Wired or Wireless) and a plausible score.
    assert!(
        matches!(
            c.class,
            mineru_table::cls::TableClass::Wired | mineru_table::cls::TableClass::Wireless
        ),
        "unexpected class {:?}",
        c.class
    );
    // Raw logits are unbounded, but the winning logit for a 2-class head sits in a
    // sane finite range; assert it is finite and not absurd.
    assert!(c.score.is_finite(), "score must be finite, got {}", c.score);
    assert!(
        c.score > -50.0 && c.score < 50.0,
        "score out of plausible range: {}",
        c.score
    );
}

/// Exercises the UNet forward pass directly and asserts a sane 3-class mask.
#[test]
#[ignore = "requires MINERU_MODELS_DIR or network for the weight fetch; slow ndarray forward"]
fn unet_forward_produces_mask() {
    use mineru_table::unet::model::UnetModel;

    let img = synthetic_table(256, 256);
    let model = UnetModel::loaded();
    let mask = model
        .segment_mask(&img)
        .expect("segment_mask should run the real UNet forward");

    println!(
        "unet mask: {}x{}, {} px",
        mask.width,
        mask.height,
        mask.classes.len()
    );
    assert_eq!(mask.classes.len(), mask.width * mask.height);
    assert!(mask.width > 0 && mask.height > 0);
    // 3-class segmentation: every pixel is background/horizontal/vertical.
    assert!(
        mask.classes.iter().all(|&c| (0..=2).contains(&c)),
        "mask classes must be in 0..=2"
    );
    let (h_px, v_px) = mask.classes.iter().fold((0usize, 0usize), |(h, v), &c| match c {
        1 => (h + 1, v),
        2 => (h, v + 1),
        _ => (h, v),
    });
    println!("mask line pixels: horizontal={h_px} vertical={v_px}");
}

/// Full wired-table path against the real UNet: `segment_cells` (forward pass +
/// mask → polygon extraction, including the ported line endpoint-adjustment) →
/// grid recovery → HTML. On a clean 1024² 4×4 ruled grid this must recover exactly
/// 4 rows × 4 cols with unit rowspans/colspans — the endpoint adjustment
/// (`adjust_lines`/`final_adjust_lines`) is what stops per-column y-jitter from
/// over-segmenting rows (pre-fix this recovered 13 rows).
#[test]
#[ignore = "requires MINERU_MODELS_DIR or network for the weight fetch; slow ndarray forward"]
fn unet_segment_cells_recovers_grid_html() {
    use mineru_table::unet::model::UnetModel;
    use mineru_table::unet::{plot_html_table, recover, COL_THRESHOLD, ROW_THRESHOLD};
    use std::collections::HashMap;

    // A clear 4x4 ruled grid so the segmenter has strong, unambiguous rulings.
    let img = synthetic_table(1024, 1024);
    let model = UnetModel::loaded();
    let polys = model
        .segment_cells(&img)
        .expect("segment_cells should run end-to-end with the generated model");

    println!("segment_cells produced {} cell polygons", polys.len());
    for (i, p) in polys.iter().take(8).enumerate() {
        println!("  cell {i}: {p:?}");
    }
    // A clean 4x4 grid should recover exactly 16 interior cells.
    assert_eq!(polys.len(), 16, "expected 16 cells on a 4x4 grid");

    let logic = recover(&polys, ROW_THRESHOLD, COL_THRESHOLD);
    assert_eq!(logic.len(), polys.len());
    let max_row = logic.iter().map(|l| l.row_end).max().unwrap_or(0) + 1;
    let max_col = logic.iter().map(|l| l.col_end).max().unwrap_or(0) + 1;
    println!("recovered grid: {max_row} rows x {max_col} cols");

    // The gate: a 4x4 grid must recover as exactly 4 rows x 4 cols (was 13x4).
    assert_eq!(max_row, 4, "expected 4 rows, got {max_row}");
    assert_eq!(max_col, 4, "expected 4 cols, got {max_col}");
    // Every cell must be a unit 1x1 span (no spurious rowspan/colspan merging).
    for l in &logic {
        assert_eq!(l.row_start, l.row_end, "spurious rowspan: {l:?}");
        assert_eq!(l.col_start, l.col_end, "spurious colspan: {l:?}");
    }

    let text: HashMap<usize, Vec<String>> = HashMap::new();
    let html = plot_html_table(&logic, &text);
    println!(
        "html (first 400 chars): {}",
        &html[..html.len().min(400)]
    );
    assert!(html.starts_with("<html>"), "html: {html}");
    assert!(html.contains("<table>"));
    assert!(html.contains("</table>"));
    // No spurious spans in the rendered HTML either.
    assert!(
        !html.contains("rowspan=8") && !html.contains("colspan=8"),
        "unexpected large span in html: {html}"
    );
    // Exactly 16 unit cells rendered.
    let unit_cells = html.matches("<td rowspan=1 colspan=1>").count();
    assert_eq!(unit_cells, 16, "expected 16 unit <td> cells, html: {html}");
    // And exactly 4 table rows.
    assert_eq!(html.matches("<tr>").count(), 4, "expected 4 <tr> rows");
}
