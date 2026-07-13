//! Real-weight load test, gated behind `#[ignore]` so the default test run never
//! needs the multi-hundred-megabyte checkpoint.
//!
//! Run with the checkpoint present:
//!
//! ```sh
//! MINERU_OCR_DET_WEIGHTS=/path/to/ch_PP-OCRv6_small_det_infer.safetensors \
//!     cargo test -p mineru-ocr-det --test real_weights -- --ignored --nocapture
//! ```
//!
//! A successful load under [`Coverage::Strict`] proves the key remap is complete:
//! every one of the checkpoint's tensors matched a Burn field (no `UnmappedKeys`),
//! and no tensor failed to apply (no shape mismatch).

use std::path::PathBuf;

use mineru_burn_common::backend::{cpu_device, Cpu};
use mineru_ocr_det::{DetConfig, TextDetector};

/// Resolves the weights path from `MINERU_OCR_DET_WEIGHTS`, skipping (returning
/// `None`) when unset so the test is a no-op if someone forgets to point at it.
fn weights_path() -> Option<PathBuf> {
    std::env::var_os("MINERU_OCR_DET_WEIGHTS").map(PathBuf::from)
}

#[test]
#[ignore = "requires the PP-OCRv6 small-det safetensors checkpoint"]
fn loads_all_keys_consumed() {
    let Some(path) = weights_path() else {
        eprintln!("MINERU_OCR_DET_WEIGHTS not set; skipping");
        return;
    };
    let device = cpu_device();
    let mut det = TextDetector::<Cpu>::new(DetConfig::default(), device);
    // Strict coverage: an `Ok` here means zero unmapped keys.
    let res = det.load_weights(&path);
    assert!(res.is_ok(), "weight load failed: {:?}", res.err());
}
