//! End-to-end integration test for the hand-ported SLANet-plus recognizer.
//!
//! Unlike the LCNet/UNet tests (which need the `onnx-import` codegen feature),
//! SLANet-plus loads its weights at runtime from a converted `.safetensors` that
//! sits next to the `.onnx`, so this test needs no cargo feature — only the model
//! files on disk. It is `#[ignore]`d because the CPU forward pass is slow.
//!
//! Run with:
//!
//! ```text
//! SLANET_ONNX=/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0/models/TabRec/SlanetPlus/slanet-plus.onnx \
//!   cargo test -p mineru-table --test slanet_real -- --ignored --nocapture
//! ```

use image::{Rgb, RgbImage};
use mineru_table::slanet::model::SlaNet;
use mineru_table::slanet::preprocess::preprocess;
use mineru_table::slanet::{build_vocab, decode::decode};

/// Default model path (overridable via `SLANET_ONNX`).
const DEFAULT_ONNX: &str =
    "/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0/models/TabRec/SlanetPlus/slanet-plus.onnx";

/// Builds a synthetic 4×4 grid image, table-like enough to exercise the decoder.
fn synthetic_table(w: u32, h: u32) -> RgbImage {
    let mut img = RgbImage::from_pixel(w, h, Rgb([255, 255, 255]));
    let sh = (h / 4).max(1);
    let sw = (w / 4).max(1);
    let mut y = 0;
    while y < h {
        for x in 0..w {
            img.put_pixel(x, y.min(h - 1), Rgb([0, 0, 0]));
        }
        y += sh;
    }
    let mut x = 0;
    while x < w {
        for yy in 0..h {
            img.put_pixel(x.min(w - 1), yy, Rgb([0, 0, 0]));
        }
        x += sw;
    }
    img
}

#[test]
#[ignore = "requires the SLANet-plus model files on disk; slow ndarray forward"]
fn slanet_runs_real_forward_and_decodes() {
    let onnx = std::env::var("SLANET_ONNX").unwrap_or_else(|_| DEFAULT_ONNX.to_string());

    let model = SlaNet::load(&onnx).expect("SlaNet::load should succeed");
    let img = synthetic_table(488, 488);
    let pre = preprocess(&img);

    let out = model
        .forward(&pre)
        .expect("forward must run with the real weights (not ModelUnavailable)");

    println!(
        "SLANet forward: len={} num_classes={} structure_probs.len={} loc_preds.len={}",
        out.len,
        out.num_classes,
        out.structure_probs.len(),
        out.loc_preds.len()
    );

    // Shape sanity: the decoder contract must hold.
    assert!(out.len > 0, "decoded length must be positive");
    assert_eq!(out.num_classes, 50, "SLANet-plus has 50 structure classes");
    assert_eq!(
        out.structure_probs.len(),
        out.len * out.num_classes,
        "structure_probs must be [L, C] row-major"
    );
    assert_eq!(
        out.loc_preds.len(),
        out.len * 4,
        "loc_preds must be [L, 4] (quad corners reduced to axis-aligned boxes)"
    );
    // Every structure row should be a probability distribution (softmaxed).
    for step in 0..out.len {
        let row = &out.structure_probs[step * out.num_classes..(step + 1) * out.num_classes];
        let sum: f32 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-2,
            "row {step} probs should sum to ~1, got {sum}"
        );
    }

    // Decode must yield a non-empty token stream wrapped in the HTML frame.
    let vocab = build_vocab();
    let res = decode(&out.as_preds(), &vocab, pre.orig_w, pre.orig_h);
    println!("decoded tokens: {:?}", res.tokens);
    println!(
        "cell boxes: {}, score: {:.3}",
        res.cell_bboxes.len(),
        res.score
    );

    assert!(res.tokens.len() > 6, "expected structure tokens, not just the frame");
    assert_eq!(res.tokens.first().map(String::as_str), Some("<html>"));
    assert_eq!(res.tokens.get(1).map(String::as_str), Some("<body>"));
    assert_eq!(res.tokens.get(2).map(String::as_str), Some("<table>"));
    assert_eq!(res.tokens.last().map(String::as_str), Some("</html>"));
    // A grid image should decode at least one row and one cell.
    assert!(
        res.tokens.iter().any(|t| t == "<tr>"),
        "expected at least one <tr>"
    );
    assert!(
        res.tokens.iter().any(|t| t == "<td></td>"),
        "expected at least one <td></td>"
    );
    assert!(!res.cell_bboxes.is_empty(), "expected at least one cell box");
}
