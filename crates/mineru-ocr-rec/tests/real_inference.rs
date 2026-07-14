//! Real-weight recognition smoke test.
//!
//! `#[ignore]`d by default so `cargo test` stays offline and fast. Run with:
//!
//! ```text
//! cargo test -p mineru-ocr-rec --test real_inference -- --ignored
//! ```
//!
//! Downloads the PP-OCRv6 small-rec safetensors + dictionary from the
//! PDF-Extract-Kit repo into `/Volumes/Archive/mineru/models`, loads them, and runs
//! recognition on a blank crop — verifying weight load + forward + CTC decode wire up
//! without panicking.

use std::path::PathBuf;

use image::RgbImage;
use mineru_burn_common::backend::{cpu_device, Cpu};
use mineru_ocr_rec::{CharDict, RecConfig, TextRecognizer};

const HF_REPO: &str = "opendatalab/PDF-Extract-Kit-1.0";
const REC_REL_PATH: &str =
    "models/OCR/paddleocr_torch/ch_PP-OCRv6_small_rec_infer.safetensors";
/// hf-hub download cache dir. Set `MINERU_MODELS_DIR` before running this
/// `#[ignore]`d test; there is no baked-in machine path.
fn cache_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("MINERU_MODELS_DIR")
            .expect("set MINERU_MODELS_DIR to a model cache directory"),
    )
}

fn download(rel_path: &str) -> PathBuf {
    use hf_hub::api::sync::ApiBuilder;
    let cache = cache_dir();
    std::fs::create_dir_all(&cache).expect("create model cache dir");
    let api = ApiBuilder::new()
        .with_cache_dir(cache)
        .build()
        .expect("hf-hub api");
    api.model(HF_REPO.to_string())
        .get(rel_path)
        .expect("download file")
}

#[test]
#[ignore = "downloads real weights; run with --ignored"]
fn loads_weights_and_recognizes_without_panic() {
    let weights = download(REC_REL_PATH);

    // The v6 charset is bundled in the crate, so the test uses the embedded
    // dictionary rather than an external machine-specific path.
    let dict = CharDict::ppocrv6(true).expect("load embedded char dict");

    let device = cpu_device();
    let mut rec = TextRecognizer::<Cpu>::new(dict, RecConfig::default(), device);
    rec.load_weights(&weights).expect("load rec weights");

    // A blank crop should decode to (usually empty) text with a valid score.
    let crop = RgbImage::new(160, 48);
    let (text, score) = rec.recognize(&crop).expect("recognition runs");
    println!("recognized {text:?} with score {score}");
    assert!((0.0..=1.0).contains(&score));
}
