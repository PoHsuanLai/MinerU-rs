//! Numerical parity test against the PyTorch reference.
//!
//! `#[ignore]`d by default. It proves the Rust SVTR_LCNet + CTC forward pass
//! produces the SAME output as the Python `pytorchocr` reference on the SAME
//! deterministic input, within a tight fp32 tolerance.
//!
//! Run (first generate the reference dumps with the committed Python script,
//! which writes them next to itself):
//!
//! ```text
//! python3 crates/mineru-ocr-rec/tests/reference/py_ref_rec.py
//! REC_WEIGHTS=/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0/models/OCR/paddleocr_torch/ch_PP-OCRv6_small_rec_infer.safetensors \
//! REC_REF_DIR=crates/mineru-ocr-rec/tests/reference \
//! cargo test -p mineru-ocr-rec --test parity -- --ignored --nocapture
//! ```
//!
//! The reference dir must contain the `.bin` + `.shape` dumps produced by the
//! Python script (`input`, `backbone_0..3`, `backbone_pooled`, `neck`, `logits`);
//! the script reads its checkpoint from `$REC_WEIGHTS` and dumps into its own
//! directory. The dumps are regenerable and intentionally not committed.
//!
//! Methodology: the fixed input is a deterministic 320x48 RGB gradient. The rec
//! recognizer resizes crops to height 48 and width `ceil(48 * aspect)` capped at the
//! padded canvas; a 320x48 crop has aspect 320/48 so it resizes to exactly 320x48 --
//! an IDENTITY resize (no cv2 interpolation, no right-padding). That isolates
//! conv/BN/attention/CTC-head math from resize-filter differences, which is the point
//! of THIS parity target.
//!
//! The input tensor is built here with the exact same normalisation the Python side
//! used (BGR channel order, `x/127.5 - 1`), so preprocessing is not a variable in the
//! model-math comparison; separately we diff the Rust `preprocess_at` output against
//! the Python input to validate the preprocessing path itself.

use std::path::{Path, PathBuf};

use burn::tensor::{Tensor, TensorData};
use image::{Rgb, RgbImage};
use mineru_burn_common::backend::{cpu_device, Cpu, CpuDevice};
use mineru_ocr_rec::{CharDict, RecConfig, TextRecognizer};

const H: u32 = 48;
const W: u32 = 320;

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

/// Build the NCHW input tensor directly (no resize), matching Python `preprocess`:
/// BGR channel order, `x/127.5 - 1`.
fn make_input(device: &CpuDevice) -> Tensor<Cpu, 4> {
    let img = make_image();
    let (h, w) = (H as usize, W as usize);
    let mut data = vec![0.0_f32; 3 * h * w];
    // RGB channel c -> BGR plane index: R(0)->2, G(1)->1, B(2)->0.
    let bgr_plane = [2usize, 1, 0];
    for (rgb_c, &plane_c) in bgr_plane.iter().enumerate() {
        let plane = &mut data[plane_c * h * w..(plane_c + 1) * h * w];
        for (i, px) in img.pixels().enumerate() {
            plane[i] = px.0[rgb_c] as f32 / 127.5 - 1.0;
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
        std::env::var("REC_WEIGHTS").expect("set REC_WEIGHTS to the safetensors path"),
    );
    let ref_dir = PathBuf::from(
        std::env::var("REC_REF_DIR").expect("set REC_REF_DIR to the Python dump dir"),
    );

    let device = cpu_device();
    // The embedded ppocrv6 dict (+ blank + space) sizes the CTC head to 18710 classes,
    // matching the reference `out_channels_list.CTCLabelDecode`.
    let dict = CharDict::ppocrv6(true).expect("load embedded ppocrv6 dict");
    assert_eq!(dict.num_classes(), 18710, "dict must size the head to 18710 classes");
    let mut rec = TextRecognizer::<Cpu>::new(dict, RecConfig::default(), device);
    rec.load_weights(&weights).expect("load rec weights");

    // ---- 1. Preprocessing path check: Rust preprocess vs Python input ----
    let (ref_input, ref_input_shape) = read_ref(&ref_dir, "input");
    let rust_pre = rec.preprocess_at(&make_image()).expect("preprocess_at");
    let rust_pre_v = rust_pre.into_data().into_vec::<f32>().expect("f32");
    let (pre_max, pre_mean) = diff(&rust_pre_v, &ref_input);
    println!(
        "[preprocess] shape {:?}  max-abs={:.3e}  mean-abs={:.3e}",
        ref_input_shape, pre_max, pre_mean
    );

    // ---- 2. Model math: feed the identical input tensor, diff each stage ----
    let input = make_input(&device);
    let (stages, pooled, neck, logits) = rec.forward_stages(input).expect("forward_stages");

    let mut worst = 0.0_f32;
    for (i, (v, shape)) in stages.iter().enumerate() {
        let (rv, rs) = read_ref(&ref_dir, &format!("backbone_{i}"));
        assert_eq!(&rs, shape, "backbone_{i} shape mismatch");
        let (mx, mn) = diff(v, &rv);
        worst = worst.max(mx);
        println!(
            "[backbone_{i}] shape {:?}  max-abs={:.3e}  mean-abs={:.3e}",
            shape, mx, mn
        );
    }

    let (pooled_v, pooled_shape) = pooled;
    let (pooled_ref, pooled_ref_shape) = read_ref(&ref_dir, "backbone_pooled");
    assert_eq!(pooled_ref_shape, pooled_shape, "backbone_pooled shape mismatch");
    let (pooled_mx, pooled_mn) = diff(&pooled_v, &pooled_ref);
    worst = worst.max(pooled_mx);
    println!(
        "[backbone_pooled] shape {:?}  max-abs={:.3e}  mean-abs={:.3e}",
        pooled_shape, pooled_mx, pooled_mn
    );

    let (neck_v, neck_shape) = neck;
    let (neck_ref, neck_ref_shape) = read_ref(&ref_dir, "neck");
    assert_eq!(neck_ref_shape, neck_shape, "neck shape mismatch");
    let (neck_mx, neck_mn) = diff(&neck_v, &neck_ref);
    worst = worst.max(neck_mx);
    println!(
        "[neck] shape {:?}  max-abs={:.3e}  mean-abs={:.3e}",
        neck_shape, neck_mx, neck_mn
    );

    let (logits_v, logits_shape) = logits;
    let (logits_ref, logits_ref_shape) = read_ref(&ref_dir, "logits");
    assert_eq!(logits_ref_shape, logits_shape, "logits shape mismatch");
    let (logits_mx, logits_mn) = diff(&logits_v, &logits_ref);
    println!(
        "[logits] shape {:?}  max-abs={:.3e}  mean-abs={:.3e}",
        logits_shape, logits_mx, logits_mn
    );

    println!(
        "\nVERDICT: final logits max-abs={:.3e} mean-abs={:.3e}; worst intermediate max-abs={:.3e}; preprocess max-abs={:.3e}",
        logits_mx, logits_mn, worst, pre_max
    );

    // Tight fp32 tolerance for a faithful conv/BN/attention/CTC-head port.
    let tol = 1e-3_f32;
    assert!(
        logits_mx < tol,
        "final logits diverge: max-abs {logits_mx:.3e} >= tol {tol:.1e}"
    );
}
