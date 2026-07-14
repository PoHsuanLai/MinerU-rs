//! Numerical parity test against the PyTorch reference (UniMerNet).
//!
//! `#[ignore]`d by default. It proves the Rust UniMerNet forward — the Swin
//! encoder and the first MBart decoder step — produces the SAME output as the
//! Python reference on the SAME deterministic input, within a tight fp32
//! tolerance.
//!
//! Run (first generate the reference dumps with the committed Python script,
//! which writes them next to itself):
//!
//! ```text
//! FORMULA_WEIGHTS=/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0/models/MFR/unimernet_hf_small_2503 \
//!   python3 crates/mineru-formula/tests/reference/py_ref_formula.py
//!
//! FORMULA_WEIGHTS=/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0/models/MFR/unimernet_hf_small_2503 \
//! FORMULA_REF_DIR=crates/mineru-formula/tests/reference \
//!   cargo test -p mineru-formula --test parity -- --ignored --nocapture
//! ```
//!
//! `FORMULA_WEIGHTS` points at the checkpoint DIRECTORY (containing
//! `model.safetensors` + `tokenizer.json`); `FORMULA_REF_DIR` at the Python dump
//! dir. The dump dir must contain the `.bin` + `.shape` pairs the script writes
//! (`input`, `swin_embed`, `swin_stage_0..3`, `encoder_out`, `decoder_logits`).
//! The dumps are regenerable and intentionally not committed.
//!
//! Methodology (mirrors the proven `mineru-ocr-det` template):
//! - The UniMerNet image processor targets `[H=192, W=672]` and yields ONE
//!   grayscale channel normalised by `(gray - 0.7931*255)/(0.1738*255)`, repeated
//!   to 3 channels. We build the input tensor DIRECTLY at the target size (the
//!   byte-identical formula the Python side uses), so resize/crop/pad are identity
//!   and only model math is compared.
//! - We diff in stages: patch-embed, each Swin stage, the encoder memory, then the
//!   FIRST decoder-step logits for the BOS token (a deterministic quantity). We do
//!   NOT diff a sampled sequence, which diverges chaotically.

use std::path::{Path, PathBuf};

use burn::tensor::{Tensor, TensorData};
use image::{Rgb, RgbImage};
use mineru_burn_common::backend::{cpu_device, Cpu};
use mineru_burn_common::weights::Coverage;
use mineru_formula::FormulaRecognizer;

const IMG_H: usize = 192;
const IMG_W: usize = 672;
const NORM_MEAN: f32 = 0.7931;
const NORM_STD: f32 = 0.1738;
const BOS: u32 = 0;

/// The deterministic grayscale pattern (0..255) the Python reference builds:
/// `gray[y,x] = (x*173 + y*149) % 256`.
fn make_gray() -> Vec<f32> {
    let mut g = vec![0.0_f32; IMG_H * IMG_W];
    for y in 0..IMG_H {
        for x in 0..IMG_W {
            g[y * IMG_W + x] = ((x * 173 + y * 149) % 256) as f32;
        }
    }
    g
}

/// Build the `[1, 3, 192, 672]` pixel tensor directly, matching the Python
/// `make_input`: normalise the gray pattern, then repeat to 3 channels.
fn make_input() -> Tensor<Cpu, 4> {
    let device = cpu_device();
    let gray = make_gray();
    let mean = NORM_MEAN * 255.0;
    let std = NORM_STD * 255.0;
    let plane: Vec<f32> = gray.iter().map(|&p| (p - mean) / std).collect();
    let mut data = Vec::with_capacity(3 * IMG_H * IMG_W);
    for _ in 0..3 {
        data.extend_from_slice(&plane);
    }
    Tensor::<Cpu, 1>::from_data(TensorData::new(data, [3 * IMG_H * IMG_W]), &device)
        .reshape([1, 3, IMG_H, IMG_W])
}

/// An RGB image whose luma equals the gray pattern, so the Rust preprocess path
/// (which grayscales internally) can be diffed against the Python input.
///
/// We set R=G=B=gray; `into_luma8` then reproduces the same gray. The pattern
/// spans the full frame (dark pixels touch all edges) so margin-crop is identity,
/// and the frame is already exactly the target size so resize/pad are identity.
fn make_rgb_image() -> RgbImage {
    let gray = make_gray();
    let mut img = RgbImage::new(IMG_W as u32, IMG_H as u32);
    for y in 0..IMG_H {
        for x in 0..IMG_W {
            let v = gray[y * IMG_W + x] as u8;
            img.put_pixel(x as u32, y as u32, Rgb([v, v, v]));
        }
    }
    img
}

/// Read a Python-dumped `<name>.bin` (little-endian f32) + `<name>.shape`.
fn read_ref(dir: &Path, name: &str) -> (Vec<f32>, Vec<usize>) {
    let bin = std::fs::read(dir.join(format!("{name}.bin")))
        .unwrap_or_else(|e| panic!("read {name}.bin: {e}"));
    let shape_s = std::fs::read_to_string(dir.join(format!("{name}.shape")))
        .unwrap_or_else(|e| panic!("read {name}.shape: {e}"));
    let shape: Vec<usize> = shape_s
        .trim()
        .split(',')
        .map(|s| s.parse().expect("shape dim"))
        .collect();
    let vals: Vec<f32> = bin
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let n: usize = shape.iter().product();
    assert_eq!(vals.len(), n, "{name}: bin len {} != shape prod {n}", vals.len());
    (vals, shape)
}

/// Max-abs and mean-abs elementwise diff between two equal-length vectors.
fn diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_abs += d as f64;
    }
    (max_abs, (sum_abs / a.len() as f64) as f32)
}

fn to_vec<const D: usize>(t: Tensor<Cpu, D>) -> Vec<f32> {
    t.into_data().into_vec::<f32>().expect("f32")
}

#[test]
#[ignore = "requires real weights + Python reference dump; run with --ignored"]
fn forward_matches_pytorch_reference() {
    let weights_dir = PathBuf::from(
        std::env::var("FORMULA_WEIGHTS").expect("set FORMULA_WEIGHTS to the checkpoint dir"),
    );
    let ref_dir = PathBuf::from(
        std::env::var("FORMULA_REF_DIR").expect("set FORMULA_REF_DIR to the Python dump dir"),
    );

    let recognizer = FormulaRecognizer::<Cpu>::from_pretrained(&weights_dir, Coverage::Strict)
        .expect("load formula weights");

    // ---- 1. Preprocessing path check: Rust preprocess vs Python input ----
    let (ref_input, ref_input_shape) = read_ref(&ref_dir, "input");
    let rust_pre = recognizer
        .preprocess_pixels(&make_rgb_image())
        .expect("preprocess_pixels");
    let (pre_max, pre_mean) = diff(&to_vec(rust_pre), &ref_input);
    println!(
        "[preprocess] shape {:?}  max-abs={:.3e}  mean-abs={:.3e}",
        ref_input_shape, pre_max, pre_mean
    );

    // ---- 2. Model math: feed the identical input tensor, diff each stage ----
    let input = make_input();
    // Sanity: the directly-built input matches the Python-dumped input exactly.
    let (in_max, _) = diff(&to_vec(input.clone()), &ref_input);
    println!("[input] direct-build vs python max-abs={in_max:.3e}");

    let (embed, stages) = recognizer.encode_stages(input);

    let (embed_ref, embed_shape) = read_ref(&ref_dir, "swin_embed");
    let (embed_mx, embed_mn) = diff(&to_vec(embed), &embed_ref);
    println!(
        "[swin_embed] shape {:?}  max-abs={:.3e}  mean-abs={:.3e}",
        embed_shape, embed_mx, embed_mn
    );

    let mut worst = embed_mx;
    for (i, stage) in stages.iter().enumerate() {
        let (rv, rs) = read_ref(&ref_dir, &format!("swin_stage_{i}"));
        assert_eq!(rs, stage.dims().to_vec(), "swin_stage_{i} shape mismatch");
        let (mx, mn) = diff(&to_vec(stage.clone()), &rv);
        worst = worst.max(mx);
        println!(
            "[swin_stage_{i}] shape {:?}  max-abs={:.3e}  mean-abs={:.3e}",
            stage.dims(),
            mx,
            mn
        );
    }

    let encoder_out = stages.last().expect("has stages").clone();
    let (enc_ref, enc_shape) = read_ref(&ref_dir, "encoder_out");
    assert_eq!(enc_shape, encoder_out.dims().to_vec(), "encoder_out shape mismatch");
    let (enc_mx, enc_mn) = diff(&to_vec(encoder_out.clone()), &enc_ref);
    println!(
        "[encoder_out] shape {:?}  max-abs={:.3e}  mean-abs={:.3e}",
        enc_shape, enc_mx, enc_mn
    );

    // ---- 3. First decoder step (BOS token) logits ----
    let logits = recognizer.decoder_step_logits(&[BOS], encoder_out);
    let (logits_ref, logits_shape) = read_ref(&ref_dir, "decoder_logits");
    let (log_mx, log_mn) = diff(&to_vec(logits), &logits_ref);
    println!(
        "[decoder_logits] shape {:?}  max-abs={:.3e}  mean-abs={:.3e}",
        logits_shape, log_mx, log_mn
    );

    println!(
        "\nVERDICT: encoder_out max-abs={enc_mx:.3e} mean-abs={enc_mn:.3e}; \
         first-step logits max-abs={log_mx:.3e} mean-abs={log_mn:.3e}; \
         worst swin stage max-abs={worst:.3e}; preprocess max-abs={pre_max:.3e}"
    );

    // Tight fp32 tolerance for a faithful port. Attention stacks accumulate error,
    // so the encoder bar is a touch looser than a single conv/BN layer.
    let enc_tol = 1e-3_f32;
    let log_tol = 2e-3_f32;
    assert!(
        enc_mx < enc_tol,
        "encoder_out diverges: max-abs {enc_mx:.3e} >= tol {enc_tol:.1e}"
    );
    assert!(
        log_mx < log_tol,
        "first-step logits diverge: max-abs {log_mx:.3e} >= tol {log_tol:.1e}"
    );
}
