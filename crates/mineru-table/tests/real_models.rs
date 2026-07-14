//! End-to-end integration tests that exercise the *real* generated neural
//! networks (LCNet classifier + UNet segmenter).
//!
//! These require the `onnx-import` feature AND `MINERU_MODELS_DIR` set to a
//! PDF-Extract-Kit checkout at build time (so build.rs can codegen the models).
//! They are `#[ignore]`d because the ndarray CPU forward pass is slow. Run with:
//!
//! ```text
//! MINERU_MODELS_DIR=/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0 \
//!   cargo test -p mineru-table --features onnx-import --test real_models -- --ignored --nocapture
//! ```

#![cfg(lcnet_generated)]

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
#[ignore = "requires onnx-import feature + MINERU_MODELS_DIR; slow ndarray forward"]
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
#[cfg(unet_generated)]
#[test]
#[ignore = "requires onnx-import feature + MINERU_MODELS_DIR; slow ndarray forward"]
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
/// mask → polygon extraction) → grid recovery → HTML. Asserts a plausible cell
/// count and a well-formed `<table>` with a sane rough row/col count.
#[cfg(unet_generated)]
#[test]
#[ignore = "requires onnx-import feature + MINERU_MODELS_DIR; slow ndarray forward"]
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
    // A ruled grid must yield at least a handful of cells (not zero, not absurd).
    assert!(
        (1..=400).contains(&polys.len()),
        "implausible cell count {}",
        polys.len()
    );

    let logic = recover(&polys, ROW_THRESHOLD, COL_THRESHOLD);
    assert_eq!(logic.len(), polys.len());
    let max_row = logic.iter().map(|l| l.row_end).max().unwrap_or(0) + 1;
    let max_col = logic.iter().map(|l| l.col_end).max().unwrap_or(0) + 1;
    println!("recovered grid: {max_row} rows x {max_col} cols");

    let text: HashMap<usize, Vec<String>> = HashMap::new();
    let html = plot_html_table(&logic, &text);
    println!(
        "html (first 400 chars): {}",
        &html[..html.len().min(400)]
    );
    assert!(html.starts_with("<html>"), "html: {html}");
    assert!(html.contains("<table>"));
    assert!(html.contains("</table>"));
    // A grid should recover more than a single 1x1 cell.
    assert!(max_row >= 2 || max_col >= 2, "grid collapsed to a single cell");
}
