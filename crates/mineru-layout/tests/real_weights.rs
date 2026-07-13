//! Real-weight integration tests, gated behind `#[ignore]` so the default test
//! run never needs the multi-gigabyte checkpoint or a network download.
//!
//! Run with the checkpoint present:
//!
//! ```sh
//! MINERU_LAYOUT_WEIGHTS=/path/to/PP-DocLayoutV2/model.safetensors \
//!     cargo test -p mineru-layout --test real_weights -- --ignored --nocapture
//! ```
//!
//! The download (via `hf-hub` or the MinerU model downloader) should place the
//! safetensors under `/Volumes/Archive/mineru/models/.../PP-DocLayoutV2/`.

use std::path::PathBuf;

use image::{Rgb, RgbImage};
use mineru_layout::LayoutModel;

/// Resolves the weights path from `MINERU_LAYOUT_WEIGHTS`, skipping (returning
/// `None`) when unset so the test is a no-op if someone forgets `--ignored`.
fn weights_path() -> Option<PathBuf> {
    std::env::var_os("MINERU_LAYOUT_WEIGHTS").map(PathBuf::from)
}

#[test]
#[ignore = "requires the PP-DocLayoutV2 safetensors checkpoint"]
fn loads_all_keys_consumed() {
    let Some(path) = weights_path() else {
        eprintln!("MINERU_LAYOUT_WEIGHTS not set; skipping");
        return;
    };
    // A successful load under `Coverage::Strict` proves the key remap is complete:
    // every source tensor matched a Burn field (no `UnmappedKeys`), and no tensor
    // failed to apply (no shape mismatch).
    let model = LayoutModel::from_safetensors(&path);
    assert!(model.is_ok(), "weight load failed: {:?}", model.err());
}

#[test]
#[ignore = "requires the PP-DocLayoutV2 safetensors checkpoint"]
fn runs_forward_on_blank_page() {
    let Some(path) = weights_path() else {
        eprintln!("MINERU_LAYOUT_WEIGHTS not set; skipping");
        return;
    };
    let model = LayoutModel::from_safetensors(&path).expect("load weights");
    // A blank white page: the point is that the whole forward pass (backbone →
    // encoder → decoder → reading order → postprocess) runs end-to-end without
    // panicking and yields a (possibly empty) detection list.
    let page = RgbImage::from_pixel(1000, 1400, Rgb([255, 255, 255]));
    let dets = model.detect(&page).expect("forward pass");
    eprintln!("blank-page detections: {}", dets.len());
    // Reading-order ranks must be a valid 0..n permutation prefix.
    for (i, d) in dets.iter().enumerate() {
        assert_eq!(d.order, i, "detections should be emitted in reading order");
        assert!(d.score >= 0.0 && d.score <= 1.0);
    }
}
