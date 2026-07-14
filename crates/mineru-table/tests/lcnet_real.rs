//! Hard numeric parity gate for the burn-onnx-generated PP-LCNet table
//! classifier.
//!
//! This is the un-fakeable proof that the generated Burn forward reproduces the
//! real `PP-LCNet_x1_0_table_cls.onnx` numerically — not just that `classify`
//! returns *a* class. It preprocesses the SAME deterministic 300×300 synthetic
//! grid that `tests/reference/py_ref_lcnet.py` fed to `onnxruntime`, runs the
//! generated forward, and asserts the 2-class output vector matches the committed
//! `lcnet_logits.bin` reference to a tight tolerance.
//!
//! Note the output is post-*softmax* probabilities (the ONNX graph, and thus the
//! generated forward, ends in a `Softmax`), so the values live in `[0, 1]`; the
//! argmax must also agree.
//!
//! Measured on-disk (release, ndarray CPU forward): max-abs diff ≈ 3.1e-5, so the
//! gate asserts `< 1e-4`. That residual is ordinary float32 accumulation drift
//! between onnxruntime and Burn's ndarray backend across the 32-conv CNN — still
//! tight enough (and the argmax must agree) to catch any structural divergence in
//! the port.
//!
//! The `.bin`/`.shape` dumps are gitignored (regenerate with the venv:
//! `python tests/reference/py_ref_lcnet.py`). This test is `#[ignore]`d because
//! it triggers a runtime weight fetch (or reuses the cache under
//! `MINERU_MODELS_DIR`) and the ndarray CPU forward is slow. The models are
//! always compiled now, so no cargo feature is needed. Run with:
//!
//! ```text
//! MINERU_MODELS_DIR=/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0 \
//!   cargo test -p mineru-table --release \
//!   --test lcnet_real -- --ignored --nocapture
//! ```

use image::{Rgb, RgbImage};

/// Number of output classes (wired / wireless).
const NUM_CLASSES: usize = 2;

/// Builds the deterministic 300×300 grid the reference dumper uses (white with a
/// black ruling every quarter across width and height).
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

/// Loads a committed little-endian f32 reference dump, or returns `None` (with a
/// note) when the regenerable `.bin` is absent so the test can skip cleanly.
fn load_ref(name: &str) -> Option<Vec<f32>> {
    let path = format!("{}/tests/reference/{name}.bin", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&path).ok()?;
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// Index of the maximum element (first on ties).
fn argmax(row: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best {
            best = v;
            best_idx = i;
        }
    }
    best_idx
}

#[test]
#[ignore = "requires the PP-LCNet model files + reference dump on disk; slow ndarray forward"]
fn lcnet_matches_onnx_reference() {
    let reference = match load_ref("lcnet_logits") {
        Some(v) => v,
        None => {
            eprintln!(
                "SKIP: tests/reference/lcnet_logits.bin missing; \
                 regenerate with `python tests/reference/py_ref_lcnet.py`"
            );
            return;
        }
    };
    assert_eq!(
        reference.len(),
        NUM_CLASSES,
        "reference must be a 2-class vector"
    );

    // The SAME input the Python dumper preprocessed and fed to onnxruntime: the
    // nearest-neighbor resize in `preprocess` is byte-identical across both sides.
    let img = synthetic_table(300, 300);
    let input = mineru_table::cls::preprocess(&img).expect("preprocess should succeed");

    let out = mineru_table::cls::debug_forward::<mineru_burn_common::backend::Cpu>(input)
        .expect("LCNet forward must run with real weights");
    assert_eq!(out.len(), NUM_CLASSES, "forward must yield a 2-class vector");
    println!("rust lcnet output = {out:?}");
    println!("onnx reference    = {reference:?}");

    // 1) Argmax parity.
    assert_eq!(
        argmax(&out),
        argmax(&reference),
        "argmax class disagrees: rust={out:?} onnx={reference:?}"
    );

    // 2) Output-vector parity (< 1e-4 max-abs; measured ≈ 3.1e-5).
    let mut max_abs = 0.0f32;
    for (r, o) in out.iter().zip(reference.iter()) {
        max_abs = max_abs.max((r - o).abs());
    }
    println!("lcnet output max-abs diff = {max_abs:.3e}");
    assert!(
        max_abs < 1e-4,
        "lcnet output max-abs diff {max_abs:.3e} exceeds 1e-4"
    );
}
