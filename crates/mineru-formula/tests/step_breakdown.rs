//! Diagnostic probe: a real decode step costs ~59 ms, but every op it performs —
//! projections, FFN, attention matmuls, softmax, the `swap_dims`/`reshape` shape
//! work, and the `lm_head` — sums to ~12 ms at the same shapes (`op_profile`).
//! This finds the missing ~47 ms by timing the real path with real weights.
//!
//! ```text
//! MINERU_FORMULA_WEIGHTS=/path/to/unimernet_hf_small_2503/model.safetensors \
//!   cargo test -p mineru-formula --release --test step_breakdown -- --ignored --nocapture
//! ```

use mineru_burn_common::backend::Cpu;
use std::time::Instant;

#[test]
#[ignore = "diagnostic; needs the checkpoint and a slow CPU forward"]
fn where_does_a_step_go() {
    use burn::tensor::{Int, Tensor, TensorData};
    use mineru_burn_common::backend::cpu_device;
    use mineru_burn_common::weights::{load_weights_ignoring, Coverage};
    use mineru_formula::weights::{build_remap, IGNORED_KEYS};
    use mineru_formula::{UniMerNet, UniMerNetConfig};

    let Ok(path) = std::env::var("MINERU_FORMULA_WEIGHTS") else {
        eprintln!("set MINERU_FORMULA_WEIGHTS to the checkpoint model.safetensors");
        return;
    };

    let dev = cpu_device();
    let cfg = UniMerNetConfig::small_2503();
    let mut model = UniMerNet::<Cpu>::new(&cfg, &dev);
    let Ok(remap) = build_remap() else {
        eprintln!("remap failed to build");
        return;
    };
    if let Err(e) = load_weights_ignoring(&mut model, &path, &remap, Coverage::Strict, IGNORED_KEYS)
    {
        eprintln!("load failed: {e}");
        return;
    }

    // A realistic encoder grid: 126 visual tokens of width d_model.
    let enc: Tensor<Cpu, 3> = Tensor::random(
        [1, 126, cfg.decoder.d_model],
        burn::tensor::Distribution::Default,
        &dev,
    );

    // Cross-attention K/V are computed once here, as in a real decode.
    let t = Instant::now();
    let mut cache = model.init_decode_cache(enc);
    println!("init_decode_cache (cross K/V once): {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);

    let token: Tensor<Cpu, 2, Int> =
        Tensor::from_data(TensorData::new(vec![cfg.decoder.bos_token_id as i64], [1, 1]), &dev);

    // Warm up, then time steps at a realistic cache depth.
    for p in 0..5 {
        let _ = model.decode_step(token.clone(), p, &mut cache);
    }

    let n = 40;
    let t = Instant::now();
    for p in 5..(5 + n) {
        let logits = model.decode_step(token.clone(), p, &mut cache);
        let _ = logits.into_data();
    }
    let per_step = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
    println!("\nfull decode_step (real weights):    {per_step:.2} ms/step");
    println!("op_profile's sum of the same ops:    ~12.1 ms");
    println!("unaccounted:                         {:.1} ms ({:.0}%)", per_step - 12.1, (per_step - 12.1) / per_step * 100.0);

    // The decoder is 8 identical layers; the step also does an embedding lookup, a
    // final layer-norm and the lm_head. Timing the layer stack alone against the
    // whole step splits "inside the layers" from "around them".
    println!("\n(if the gap is inside the layers, it is per-op overhead Burn adds");
    println!(" beyond the raw kernels; if outside, it is the embed/lm_head path)");
}
