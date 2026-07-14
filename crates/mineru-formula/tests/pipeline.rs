//! Integration tests for the formula-recognition pipeline.
//!
//! Two kinds:
//! - **fast, weight-free** tests that build the model with random weights and run
//!   the real Swin + MBart forward passes end-to-end, asserting shapes and that
//!   nothing panics — this exercises every module's `forward` for real (not just
//!   construction);
//! - a `#[ignore]`d **real-weights** test that only runs when the checkpoint has
//!   been downloaded to `MINERU_FORMULA_MODEL_DIR`, so CI stays offline-clean.

use burn::tensor::{Int, Tensor, TensorData};
use mineru_burn_common::backend::{cpu_device, Cpu};
use mineru_formula::config::UniMerNetConfig;
use mineru_formula::model::UniMerNet;

/// Runs the encoder on a random image tensor and checks the visual-token shape.
///
/// With `[192, 672]` input, the stem downsamples by 4 → `[48, 168]`, then three
/// patch-merges halve each spatial dim: `48→24→12→6` and `168→84→42→21`, giving
/// `6 * 21 = 126` tokens of width `768`.
#[test]
fn encoder_produces_expected_token_grid() {
    let device = cpu_device();
    let cfg = UniMerNetConfig::small_2503();
    let model = UniMerNet::<Cpu>::new(&cfg, &device);

    // A small but real-shaped input: [1, 3, 192, 672].
    let (h, w) = (192usize, 672usize);
    let pixels: Tensor<Cpu, 4> = Tensor::zeros([1, 3, h, w], &device);
    let tokens = model.encode(pixels);

    let dims = tokens.dims();
    assert_eq!(dims[0], 1, "batch");
    assert_eq!(dims[2], cfg.encoder.hidden_size(), "hidden size 768");
    // 6 * 21 == 126 tokens after /4 stem + three /2 merges.
    assert_eq!(dims[1], 126, "token count");
}

/// Runs one decoder step over the encoder grid and checks the logits shape.
#[test]
fn decoder_step_produces_vocab_logits() {
    let device = cpu_device();
    let cfg = UniMerNetConfig::small_2503();
    let model = UniMerNet::<Cpu>::new(&cfg, &device);

    // Fake a tiny encoder grid [1, L, d] rather than run the full encoder.
    let l = 8usize;
    let d = cfg.decoder.d_model;
    let enc: Tensor<Cpu, 3> = Tensor::zeros([1, l, d], &device);

    // A length-3 prefix starting from BOS.
    let ids: Tensor<Cpu, 2, Int> = Tensor::from_data(
        TensorData::new(vec![cfg.decoder.bos_token_id as i64, 5, 7], [1, 3]),
        &device,
    );
    let logits = model.decode(ids, enc);
    let dims = logits.dims();
    assert_eq!(dims, [1, 3, cfg.decoder.vocab_size]);
}

/// End-to-end greedy generation over random weights: proves the full loop
/// (encode once → decode per step → argmax → stop) runs without panicking and
/// terminates within the cap. Output text is meaningless (random weights), so we
/// only assert it terminates and returns a (possibly empty) string.
#[test]
fn greedy_loop_runs_end_to_end_with_random_weights() {
    use mineru_formula::generate::{greedy_decode, DecodeStep};

    let device = cpu_device();
    let cfg = UniMerNetConfig::small_2503();
    let model = UniMerNet::<Cpu>::new(&cfg, &device);

    let enc: Tensor<Cpu, 3> = Tensor::zeros([1, 8, cfg.decoder.d_model], &device);

    struct Step<'a> {
        model: &'a UniMerNet<Cpu>,
        cache: mineru_formula::mbart::DecoderCache<Cpu>,
        device: burn::backend::ndarray::NdArrayDevice,
        vocab: usize,
    }
    impl DecodeStep for Step<'_> {
        fn step(&mut self, token: u32, position: usize) -> u32 {
            let input: Tensor<Cpu, 2, Int> =
                Tensor::from_data(TensorData::new(vec![token as i64], [1, 1]), &self.device);
            let logits = self.model.decode_step(input, position, &mut self.cache);
            let idx = logits.reshape([self.vocab]).argmax(0);
            idx.into_data().to_vec::<i64>().expect("argmax to vec")[0] as u32
        }
    }

    let cache = model.init_decode_cache(enc);
    let mut step = Step {
        model: &model,
        cache,
        device,
        vocab: cfg.decoder.vocab_size,
    };
    // Small cap so the test is fast; the loop must terminate within it.
    let decoded = greedy_decode(
        &mut step,
        cfg.decoder.bos_token_id as u32,
        cfg.decoder.eos_token_id as u32,
        6,
    );
    assert!(decoded.tokens.len() <= 6);
}

/// KV-cache parity gate (weight-free, deterministic).
///
/// Runs the SAME in-memory model two ways over the same fake encoder grid and the
/// same greedy driving order, and asserts the generated token sequences are
/// **identical**:
/// - REFERENCE: the non-cached path — at each step rebuild the full prefix and run
///   [`UniMerNet::decode`], argmaxing the last position (this is exactly the old
///   pre-cache decode loop, kept here as the oracle).
/// - CACHED: the new [`UniMerNet::init_decode_cache`] + [`UniMerNet::decode_step`]
///   incremental path.
///
/// Because both share one model instance, any divergence is purely the cache logic —
/// not weight randomness. Gating on the token *sequence* (not shapes/logits) is the
/// only sound correctness check for an autoregressive decoder. We drive many steps so
/// the growing self-attention cache is exercised well past length 1.
#[test]
fn kv_cache_matches_non_cached_token_sequence() {
    use mineru_formula::mbart::DecoderCache;

    let device = cpu_device();
    let cfg = UniMerNetConfig::small_2503();
    let model = UniMerNet::<Cpu>::new(&cfg, &device);

    // A non-trivial fake encoder grid so cross-attention actually varies.
    let (l, d) = (10usize, cfg.decoder.d_model);
    let n = (l * d) as i64;
    let enc_data: Vec<f32> = (0..n).map(|i| ((i % 37) as f32 - 18.0) * 0.01).collect();
    let enc: Tensor<Cpu, 3> =
        Tensor::from_data(TensorData::new(enc_data, [1, l, d]), &device);

    let vocab = cfg.decoder.vocab_size;
    let start = cfg.decoder.bos_token_id as u32;
    // Force a fixed number of steps regardless of what EOS the random weights emit,
    // so the two paths are compared over a long, identical horizon.
    let steps = 24usize;

    // REFERENCE: non-cached, full-prefix rebuild each step.
    let reference: Vec<u32> = {
        let mut ids = vec![start];
        let mut out = Vec::new();
        for _ in 0..steps {
            let t = ids.len();
            let data: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
            let input: Tensor<Cpu, 2, Int> =
                Tensor::from_data(TensorData::new(data, [1, t]), &device);
            let logits = model.decode(input, enc.clone());
            let idx = logits.narrow(1, t - 1, 1).reshape([vocab]).argmax(0);
            let next = idx.into_data().to_vec::<i64>().expect("argmax to vec")[0] as u32;
            out.push(next);
            ids.push(next);
        }
        out
    };

    // CACHED: incremental path.
    let cached: Vec<u32> = {
        let mut cache: DecoderCache<Cpu> = model.init_decode_cache(enc.clone());
        let mut out = Vec::new();
        let mut token = start;
        for position in 0..steps {
            let input: Tensor<Cpu, 2, Int> =
                Tensor::from_data(TensorData::new(vec![token as i64], [1, 1]), &device);
            let logits = model.decode_step(input, position, &mut cache);
            let idx = logits.reshape([vocab]).argmax(0);
            let next = idx.into_data().to_vec::<i64>().expect("argmax to vec")[0] as u32;
            out.push(next);
            token = next;
        }
        out
    };

    assert_eq!(
        reference, cached,
        "KV-cached decode diverged from the non-cached reference token sequence"
    );
}

/// Real-weights smoke test. Ignored by default: set `MINERU_FORMULA_MODEL_DIR` to
/// a directory containing `model.safetensors` + `tokenizer.json` and run with
/// `--ignored` to exercise the actual load + a real prediction.
///
/// This is where any weight-key mismatch surfaces (start under `Coverage::Lenient`
/// and read the reported unmapped keys — see `mineru_formula::weights`).
#[test]
#[ignore = "requires the unimernet_hf_small_2503 checkpoint on disk"]
fn real_weights_load_and_predict() {
    use mineru_burn_common::weights::Coverage;
    use mineru_formula::FormulaRecognizer;

    let dir = std::env::var("MINERU_FORMULA_MODEL_DIR")
        .expect("set MINERU_FORMULA_MODEL_DIR to the checkpoint directory");

    // Strict: the remap is verified (see tests/real_weights.rs — every source key
    // is consumed), so a real load must leave zero unmapped keys here too.
    let recognizer = FormulaRecognizer::<Cpu>::from_pretrained(&dir, Coverage::Strict)
        .expect("load recognizer");

    // A blank white image is enough to prove the forward path runs end to end.
    let img = image::RgbImage::from_pixel(200, 60, image::Rgb([255, 255, 255]));
    let latex = recognizer.predict(&img).expect("predict");
    // Random-vs-real aside, this must at least return without panic.
    let _ = latex.0;
}

/// Real-weights KV-cache parity gate. Ignored by default.
///
/// Loads the real `unimernet_hf_small_2503` checkpoint (set `MINERU_FORMULA_MODEL_DIR`)
/// and asserts the cached greedy decode returns a **byte-identical** token sequence to
/// the slow non-cached reference decode, over a genuine formula crop. This is the
/// production correctness gate for the KV cache: on real weights, with a real image,
/// the two decode paths must agree token-for-token.
///
/// Point `MINERU_FORMULA_CROP` at a **raw RGB8** crop dump: 8-byte little-endian
/// header (`u32` width, `u32` height) followed by `width*height*3` RGB bytes. This
/// sidesteps the `image` crate's disabled file-format decoders (the production path
/// receives already-decoded `RgbImage` buffers). Without the env var, a synthetic
/// image is used (still valid — both paths see the same input).
#[test]
#[ignore = "requires the unimernet_hf_small_2503 checkpoint on disk"]
fn real_weights_kv_cache_token_parity() {
    use mineru_burn_common::weights::Coverage;
    use mineru_formula::FormulaRecognizer;

    let dir = std::env::var("MINERU_FORMULA_MODEL_DIR")
        .expect("set MINERU_FORMULA_MODEL_DIR to the checkpoint directory");

    let recognizer = FormulaRecognizer::<Cpu>::from_pretrained(&dir, Coverage::Strict)
        .expect("load recognizer");

    let img = match std::env::var("MINERU_FORMULA_CROP") {
        Ok(path) => {
            let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read crop {path}: {e}"));
            let w = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let h = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
            let pixels = bytes[8..].to_vec();
            assert_eq!(
                pixels.len(),
                (w * h * 3) as usize,
                "raw RGB payload does not match {w}x{h}"
            );
            image::RgbImage::from_raw(w, h, pixels).expect("build RgbImage from raw")
        }
        Err(_) => image::RgbImage::from_pixel(320, 64, image::Rgb([255, 255, 255])),
    };

    let cached = recognizer
        .predict_token_ids(&img)
        .expect("cached decode");
    let reference = recognizer
        .reference_token_ids_noncache(&img)
        .expect("non-cached reference decode");

    assert_eq!(
        reference.len(),
        cached.len(),
        "token count differs: reference {} vs cached {}",
        reference.len(),
        cached.len()
    );
    assert_eq!(
        reference, cached,
        "cached decode diverged from the non-cached reference on real weights"
    );
    println!(
        "real-weights KV-cache parity OK: {} tokens byte-identical",
        cached.len()
    );
}

/// Real-weights timing + LaTeX sanity. Ignored by default. Times the cached path
/// against the non-cached reference on the same real crop and prints both the
/// speedup and the decoded LaTeX. Same env vars as the parity test.
#[test]
#[ignore = "requires the unimernet_hf_small_2503 checkpoint on disk"]
fn real_weights_kv_cache_speedup() {
    use mineru_burn_common::weights::Coverage;
    use mineru_formula::FormulaRecognizer;
    use std::time::Instant;

    let dir = std::env::var("MINERU_FORMULA_MODEL_DIR")
        .expect("set MINERU_FORMULA_MODEL_DIR to the checkpoint directory");
    let recognizer = FormulaRecognizer::<Cpu>::from_pretrained(&dir, Coverage::Strict)
        .expect("load recognizer");

    let img = match std::env::var("MINERU_FORMULA_CROP") {
        Ok(path) => {
            let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read crop {path}: {e}"));
            let w = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let h = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
            image::RgbImage::from_raw(w, h, bytes[8..].to_vec()).expect("build RgbImage")
        }
        Err(_) => image::RgbImage::from_pixel(320, 64, image::Rgb([255, 255, 255])),
    };

    // Warm once (weight tensors, allocator) so timing reflects steady state.
    let _ = recognizer.predict_token_ids(&img).expect("warm");

    let t0 = Instant::now();
    let noncache = recognizer
        .reference_token_ids_noncache(&img)
        .expect("non-cached");
    let dt_noncache = t0.elapsed();

    let t1 = Instant::now();
    let cached = recognizer.predict_token_ids(&img).expect("cached");
    let dt_cached = t1.elapsed();

    let latex = recognizer.predict(&img).expect("predict latex");

    assert_eq!(noncache, cached, "timing run diverged in tokens");
    println!(
        "tokens={} | non-cached {:?} | cached {:?} | speedup {:.2}x",
        cached.len(),
        dt_noncache,
        dt_cached,
        dt_noncache.as_secs_f64() / dt_cached.as_secs_f64()
    );
    println!("LaTeX: {}", latex.0);
}
