//! Real-weight load test under strict coverage.
//!
//! `#[ignore]`d by default so CI stays offline-clean. To run it, point
//! `MINERU_FORMULA_WEIGHTS` at the checkpoint's `model.safetensors` and pass
//! `--ignored`:
//!
//! ```text
//! MINERU_FORMULA_WEIGHTS=/path/to/unimernet_hf_small_2503/model.safetensors \
//!   cargo test -p mineru-formula --test real_weights -- --ignored --nocapture
//! ```
//!
//! The bar is that the shared loader consumes **every** source key under
//! [`Coverage::Strict`] — i.e. every real weight tensor lands in a real module
//! field, and the only skipped keys are the documented training-only / recomputed
//! buffers in [`mineru_formula::weights::IGNORED_KEYS`]. Any mismatch surfaces as
//! [`Error::UnmappedKeys`] with the exact leftover key list.

use mineru_burn_common::backend::{cpu_device, Cpu};
use mineru_burn_common::weights::{load_weights_ignoring, Coverage};
use mineru_formula::weights::{build_remap, IGNORED_KEYS};
use mineru_formula::{UniMerNet, UniMerNetConfig};

/// Loads the real `unimernet_hf_small_2503` weights under `Coverage::Strict` and
/// asserts zero unmapped source keys.
#[test]
#[ignore = "requires MINERU_FORMULA_WEIGHTS pointing at the unimernet_hf_small_2503 model.safetensors"]
fn real_weights_load_strict_zero_unmapped() {
    let path = std::env::var("MINERU_FORMULA_WEIGHTS")
        .expect("set MINERU_FORMULA_WEIGHTS to the checkpoint model.safetensors path");

    let device = cpu_device();
    let config = UniMerNetConfig::small_2503();
    let mut model = UniMerNet::<Cpu>::new(&config, &device);
    let remap = build_remap().expect("remap builds");

    // Strict coverage: this returns Err(UnmappedKeys { .. }) listing any source key
    // that did not land in a field (other than the documented ignored buffers).
    let result = load_weights_ignoring(
        &mut model,
        &path,
        &remap,
        Coverage::Strict,
        IGNORED_KEYS,
    );

    match &result {
        Ok(()) => {
            println!("real-weight load OK: every source key consumed under Coverage::Strict");
        }
        Err(e) => {
            println!("real-weight load FAILED under Coverage::Strict:\n{e}");
        }
    }
    result.expect("strict real-weight load must leave zero unmapped keys");
}
