//! Real-weight load test under strict coverage, gated behind `#[ignore]` so the
//! default test run never needs the multi-gigabyte checkpoint.
//!
//! Run with the checkpoint present:
//!
//! ```sh
//! MINERU_OCR_REC_WEIGHTS=/path/to/ch_PP-OCRv6_small_rec_infer.safetensors \
//!     cargo test -p mineru-ocr-rec --test real_weights -- --ignored --nocapture
//! ```
//!
//! A successful load under `Coverage::Strict` proves the key remap is complete:
//! every source tensor (`model.backbone.*` and `head.*`) matched a Burn field (no
//! `UnmappedKeys`) and applied without a shape mismatch.

use std::path::PathBuf;

use mineru_burn_common::backend::{cpu_device, Cpu};
use mineru_ocr_rec::{CharDict, RecConfig, TextRecognizer};

// The PP-OCRv6 CTC output size: 18709 dictionary entries + 1 blank = 18710,
// matching `head.head.weight [18710, 120]` in the checkpoint. The bundled
// `CharDict::ppocrv6` with `add_space = true` must yield this count.

fn weights_path() -> Option<PathBuf> {
    std::env::var_os("MINERU_OCR_REC_WEIGHTS").map(PathBuf::from)
}

#[test]
#[ignore = "requires the PP-OCRv6 small-rec safetensors checkpoint"]
fn loads_all_keys_consumed() {
    let Some(path) = weights_path() else {
        eprintln!("MINERU_OCR_REC_WEIGHTS not set; skipping");
        return;
    };

    let dict = CharDict::ppocrv6(true).expect("load embedded char dict");
    let device = cpu_device();
    let mut rec = TextRecognizer::<Cpu>::new(dict, RecConfig::default(), device);

    // Loading uses `Coverage::Strict` internally: an `Ok` here means every source
    // key landed in a real field with a matching shape (zero unmapped keys).
    let res = rec.load_weights(&path);
    assert!(res.is_ok(), "strict weight load failed: {:?}", res.err());
}
