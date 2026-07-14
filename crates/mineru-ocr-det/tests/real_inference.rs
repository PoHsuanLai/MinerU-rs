//! Real-weight detection smoke test.
//!
//! `#[ignore]`d by default so `cargo test` stays offline and fast. Run with:
//!
//! ```text
//! cargo test -p mineru-ocr-det --test real_inference -- --ignored
//! ```
//!
//! Downloads the PP-OCRv6 small-det safetensors from the PDF-Extract-Kit repo into
//! `/Volumes/Archive/mineru/models`, loads them, and runs the detector on a blank
//! image — verifying the whole pipeline (weight load + forward + post-process) wires
//! up without panicking.

use std::path::PathBuf;

use image::RgbImage;
use mineru_burn_common::backend::{cpu_device, Cpu};
use mineru_ocr_det::{DetConfig, TextDetector};

/// Repo + path for the PP-OCRv6 small-det checkpoint.
const HF_REPO: &str = "opendatalab/PDF-Extract-Kit-1.0";
const DET_REL_PATH: &str =
    "models/OCR/paddleocr_torch/ch_PP-OCRv6_small_det_infer.safetensors";
/// hf-hub download cache dir. Set `MINERU_MODELS_DIR` before running this
/// `#[ignore]`d test; there is no baked-in machine path.
fn cache_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("MINERU_MODELS_DIR")
            .expect("set MINERU_MODELS_DIR to a model cache directory"),
    )
}

fn download_weights() -> PathBuf {
    use hf_hub::api::sync::ApiBuilder;
    let cache = cache_dir();
    std::fs::create_dir_all(&cache).expect("create model cache dir");
    let api = ApiBuilder::new()
        .with_cache_dir(cache)
        .build()
        .expect("hf-hub api");
    api.model(HF_REPO.to_string())
        .get(DET_REL_PATH)
        .expect("download det weights")
}

#[test]
#[ignore = "downloads real weights; run with --ignored"]
fn loads_weights_and_detects_without_panic() {
    let weights = download_weights();
    let device = cpu_device();
    let mut det = TextDetector::<Cpu>::new(DetConfig::default(), device);
    det.load_weights(&weights).expect("load det weights");

    // A blank image should simply yield zero (or few) boxes, never a panic.
    let image = RgbImage::new(320, 128);
    let boxes = det.detect(&image).expect("detection runs");
    // No assertion on count — a blank page legitimately has no text lines.
    println!("detected {} boxes on blank image", boxes.len());
}
