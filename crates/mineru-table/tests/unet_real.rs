//! Hard numeric parity gate for the burn-onnx-generated UNet line segmenter.
//!
//! This is the un-fakeable proof that the generated Burn forward reproduces the
//! real `unet.onnx` segmentation — not just that `segment_mask` returns a
//! sanely-shaped mask. Both the ONNX graph and the generated Rust `forward` end
//! in the same `... ReduceMax, Sub, Exp, ReduceSum, Div, ArgMax, Unsqueeze`, so
//! each emits an **argmaxed** int class mask (`0` bg / `1` horizontal / `2`
//! vertical) and there is no pre-argmax logit volume exposed on the Rust side.
//! The gate therefore asserts per-pixel agreement between the two masks.
//!
//! ## What this gate does NOT cover
//!
//! It proves the **forward** matches ONNX, and nothing about **preprocess**: both
//! sides are fed the same dumped buffer, so any error in producing that buffer is
//! shared and cancels. It passed at 100% while preprocess squashed the aspect
//! ratio and skipped mean/std normalization entirely — the input was wrong, and
//! both sides were wrong identically. Preprocess is pinned by unit tests against
//! the reference's constants instead; this file's fixture is only kept non-square
//! so a shape bug cannot hide here too.
//!
//! ## Why this test runs in two phases
//!
//! The Rust preprocess resizes with the `image` crate's separable **Triangle**
//! filter, which is not byte-identical to any stock numpy/PIL/cv2 resize. To feed
//! `onnxruntime` the IDENTICAL tensor the Burn forward consumes, this test dumps
//! its preprocessed input to `unet_input.bin`/`.shape` on first run, then SKIPs
//! (there is no reference mask yet). You then run the Python dumper on that exact
//! tensor, and re-run this test to assert parity:
//!
//! ```text
//! # 1. dump the input (test SKIPs):
//! MINERU_MODELS_DIR=/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0 \
//!   cargo test -p mineru-table --release \
//!   --test unet_real -- --ignored --nocapture --test-threads=1
//! # 2. produce the ONNX reference mask from that input:
//! python tests/reference/py_ref_unet.py
//! # 3. re-run: now asserts per-pixel parity:
//! MINERU_MODELS_DIR=/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0 \
//!   cargo test -p mineru-table --release \
//!   --test unet_real -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Measured on-disk (release, CPU forward): the two masks agree on
//! **100.0%** of the 1024² pixels, so the gate asserts exact per-pixel equality.
//!
//! The `.bin`/`.shape` dumps are gitignored (regenerable). This test is
//! `#[ignore]`d because it triggers a runtime weight fetch (or reuses the cache
//! under `MINERU_MODELS_DIR`) and the CPU forward over a 1024² image is
//! slow (minutes in debug — use `--release`). The models are always compiled, so
//! no cargo feature is needed.

use image::{Rgb, RgbImage};

use mineru_burn_common::backend::Cpu;
use mineru_table::unet::model::UnetModel;

/// The synthetic-image side (a clean square grid, upscaled to 1024 by preprocess).
/// Deliberately non-square (see `unet_matches_onnx_reference`): a square fixture
/// cannot see an aspect-ratio bug in preprocess.
const IMG_W: u32 = 320;
const IMG_H: u32 = 256;

/// Builds the deterministic grid the reference pipeline uses (white with a black
/// ruling every quarter across width and height).
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

fn ref_path(name: &str, ext: &str) -> String {
    format!(
        "{}/tests/reference/{name}.{ext}",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// Writes a little-endian f32 `.bin` + comma-separated `.shape` dump.
fn dump_f32(name: &str, data: &[f32], shape: &[usize]) {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(ref_path(name, "bin"), bytes).expect("write input bin");
    let shape_str = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    std::fs::write(ref_path(name, "shape"), shape_str).expect("write input shape");
}

/// Loads a committed little-endian i32 reference dump, or `None` if absent.
fn load_i32(name: &str) -> Option<Vec<i32>> {
    let bytes = std::fs::read(ref_path(name, "bin")).ok()?;
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

#[test]
#[ignore = "requires the UNet model files + reference dump on disk; slow CPU forward"]
fn unet_matches_onnx_reference() {
    // Non-square on purpose: preprocess fits the long edge and keeps the aspect
    // ratio, so a square fixture would make a squashing bug invisible here.
    let img = synthetic_table(IMG_W, IMG_H);

    // Always (re)dump the exact preprocessed tensor so the Python dumper can feed
    // onnxruntime the identical bytes the Burn forward sees.
    let (input, w, h) = UnetModel::<Cpu>::debug_preprocess(&img);
    let (wu, hu) = (w as usize, h as usize);
    assert_eq!(input.len(), 3 * wu * hu, "preprocess must be [3,H,W]");
    let side = mineru_table::unet::model::INPUT_SIDE;
    assert_eq!(w.max(h), side, "the long edge must land on INPUT_SIDE");
    assert!(
        w != h,
        "fixture must stay non-square through preprocess, got {w}x{h}"
    );
    dump_f32("unet_input", &input, &[1, 3, hu, wu]);
    println!("dumped unet_input.bin ({} f32, [1,3,{hu},{wu}])", input.len());

    let reference = match load_i32("unet_mask") {
        Some(v) => v,
        None => {
            eprintln!(
                "SKIP: tests/reference/unet_mask.bin missing. Just dumped unet_input.bin; \
                 now run `python tests/reference/py_ref_unet.py` (in the onnxruntime venv) \
                 to produce the reference mask, then re-run this test."
            );
            return;
        }
    };

    let model = UnetModel::<Cpu>::loaded();
    let mask = model
        .debug_segment_from_input(input, w, h)
        .expect("UNet forward must run with real weights");
    println!("rust mask: {}x{} ({} px)", mask.width, mask.height, mask.classes.len());

    assert_eq!(
        reference.len(),
        mask.classes.len(),
        "reference mask has {} px, rust has {}",
        reference.len(),
        mask.classes.len()
    );

    // Per-pixel agreement between the two argmaxed masks.
    let total = mask.classes.len();
    let mut mismatches = 0usize;
    for (r, o) in mask.classes.iter().zip(reference.iter()) {
        if *r != i64::from(*o) {
            mismatches += 1;
        }
    }
    let agree = (total - mismatches) as f64 / total as f64 * 100.0;
    println!("per-pixel agreement = {agree:.4}% ({mismatches} mismatched of {total})");

    // Measured: 100.0% agreement, so require exact equality. (The task permits a
    // ≥99.9% floor if sub-pixel resize made a handful of boundary pixels flip;
    // it does not here because both sides consume the identical input tensor.)
    assert_eq!(
        mismatches, 0,
        "unet mask disagrees on {mismatches}/{total} pixels ({agree:.4}% agree)"
    );
}
