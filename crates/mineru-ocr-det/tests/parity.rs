//! Numerical parity test against the PyTorch reference.
//!
//! `#[ignore]`d by default. It proves the Rust DBNet forward pass produces the
//! SAME output as the Python `pytorchocr` reference on the SAME deterministic
//! input, within a tight fp32 tolerance.
//!
//! Run (first generate the reference dumps with the committed Python script,
//! which writes them next to itself):
//!
//! ```text
//! python3 crates/mineru-ocr-det/tests/reference/py_ref_det.py
//! DET_WEIGHTS=/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0/models/OCR/paddleocr_torch/ch_PP-OCRv6_small_det_infer.safetensors \
//! DET_REF_DIR=crates/mineru-ocr-det/tests/reference \
//! cargo test -p mineru-ocr-det --test parity -- --ignored --nocapture
//! ```
//!
//! The reference dir must contain the `.bin` + `.shape` dumps produced by the
//! Python script (`input`, `backbone_0..3`, `neck`, `maps`); the script reads its
//! checkpoint from `$DET_WEIGHTS` and dumps into its own directory. The dumps are
//! regenerable and intentionally not committed.
//!
//! Methodology: the fixed input is a deterministic 320x320 RGB gradient (a
//! multiple of 32 in both dims, so the Python `DetResizeForTest` is an identity
//! resize -- this isolates conv/BN math from resize-interpolation differences).
//! The input tensor is built here with the exact same normalisation the Python
//! side used, so preprocessing is not a variable in the model-math comparison;
//! separately we also diff the Rust `preprocess_at` (resize+normalise) output
//! against the Python input to validate the preprocessing path itself.

use std::path::{Path, PathBuf};

use burn::tensor::{Tensor, TensorData};
use image::{Rgb, RgbImage};
use mineru_burn_common::backend::{cpu_device, Cpu, CpuDevice};
use mineru_burn_common::preprocess::Size;
use mineru_ocr_det::{DetConfig, TextDetector};

const H: u32 = 320;
const W: u32 = 320;
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// The exact same deterministic RGB gradient the Python reference builds.
fn make_image() -> RgbImage {
    let mut img = RgbImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let r = (x * 255 / (W - 1)) as u8;
            let g = (y * 255 / (H - 1)) as u8;
            let b = ((x + y) * 255 / (H + W - 2)) as u8;
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    img
}

/// Build the NCHW input tensor directly (no resize), matching Python `preprocess`.
fn make_input(device: &CpuDevice) -> Tensor<Cpu, 4> {
    let img = make_image();
    let (h, w) = (H as usize, W as usize);
    let mut data = vec![0.0_f32; 3 * h * w];
    for c in 0..3 {
        let plane = &mut data[c * h * w..(c + 1) * h * w];
        for (i, px) in img.pixels().enumerate() {
            let v = px.0[c] as f32 / 255.0;
            plane[i] = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
        }
    }
    Tensor::<Cpu, 1>::from_data(TensorData::new(data, [3 * h * w]), device).reshape([1, 3, h, w])
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
    assert_eq!(bin.len() % 4, 0, "{name}.bin not f32-aligned");
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

#[test]
#[ignore = "requires real weights + Python reference dump; run with --ignored"]
fn forward_matches_pytorch_reference() {
    let weights = PathBuf::from(
        std::env::var("DET_WEIGHTS").expect("set DET_WEIGHTS to the safetensors path"),
    );
    let ref_dir = PathBuf::from(
        std::env::var("DET_REF_DIR").expect("set DET_REF_DIR to the Python dump dir"),
    );

    let device = cpu_device();
    let mut det = TextDetector::<Cpu>::new(DetConfig::default(), device);
    det.load_weights(&weights).expect("load det weights");

    // ---- 1. Preprocessing path check: Rust resize+normalise vs Python input ----
    let (ref_input, ref_input_shape) = read_ref(&ref_dir, "input");
    let rust_pre = det
        .preprocess_at(&make_image(), Size::new(W, H))
        .expect("preprocess_at");
    let rust_pre_v = rust_pre.into_data().into_vec::<f32>().expect("f32");
    let (pre_max, pre_mean) = diff(&rust_pre_v, &ref_input);
    println!(
        "[preprocess] shape {:?}  max-abs={:.3e}  mean-abs={:.3e}",
        ref_input_shape, pre_max, pre_mean
    );

    // ---- 2. Model math: feed the identical input tensor, diff each stage ----
    let input = make_input(&device);
    let (backbone, neck, maps) = det.forward_stages(input);

    let mut worst = 0.0_f32;
    for (i, (v, shape)) in backbone.iter().enumerate() {
        let (rv, rs) = read_ref(&ref_dir, &format!("backbone_{i}"));
        assert_eq!(&rs, &shape.to_vec(), "backbone_{i} shape mismatch");
        let (mx, mn) = diff(v, &rv);
        worst = worst.max(mx);
        println!(
            "[backbone_{i}] shape {:?}  max-abs={:.3e}  mean-abs={:.3e}",
            shape, mx, mn
        );
    }

    let (neck_v, neck_shape) = neck;
    let (neck_ref, neck_ref_shape) = read_ref(&ref_dir, "neck");
    assert_eq!(neck_ref_shape, neck_shape.to_vec(), "neck shape mismatch");
    let (neck_mx, neck_mn) = diff(&neck_v, &neck_ref);
    worst = worst.max(neck_mx);
    println!(
        "[neck] shape {:?}  max-abs={:.3e}  mean-abs={:.3e}",
        neck_shape, neck_mx, neck_mn
    );

    let (maps_v, maps_shape) = maps;
    let (maps_ref, maps_ref_shape) = read_ref(&ref_dir, "maps");
    assert_eq!(maps_ref_shape, maps_shape.to_vec(), "maps shape mismatch");
    let (maps_mx, maps_mn) = diff(&maps_v, &maps_ref);
    println!(
        "[maps] shape {:?}  max-abs={:.3e}  mean-abs={:.3e}",
        maps_shape, maps_mx, maps_mn
    );

    println!(
        "\nVERDICT: final maps max-abs={:.3e} mean-abs={:.3e}; worst intermediate max-abs={:.3e}; preprocess max-abs={:.3e}",
        maps_mx, maps_mn, worst, pre_max
    );

    // Tight fp32 tolerance for a faithful conv/BN port.
    let tol = 1e-3_f32;
    assert!(
        maps_mx < tol,
        "final maps diverge: max-abs {maps_mx:.3e} >= tol {tol:.1e}"
    );
}
