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

/// Pins the reason this crate has **no** weight-free decoder gate.
///
/// `PtLinear::init` zeroes its weights — sound for inference, where loading overwrites
/// them immediately, but it means an unloaded [`UniMerNet`] is a zero map: every input
/// produces identically zero logits. So a weight-free test of the decoder compares
/// `[0.0, …]` against `[0.0, …]` and passes with the code under test arbitrarily
/// broken. Two such tests (KV-cache parity and batch lane-independence) were removed
/// after a mutation test — corrupting the cache's position offset — failed to make
/// either one fail; both are covered on real weights by
/// [`real_weights_kv_cache_token_parity`] and [`real_weights_batch_token_parity`].
///
/// This test exists so that trap cannot re-establish itself silently. If init ever
/// produces live weights, `spread` goes non-zero and this fails loudly — at which
/// point a weight-free decoder gate becomes worth writing, and this test should be
/// replaced by one.
#[test]
fn unloaded_model_emits_zero_logits_so_weight_free_gates_are_vacuous() {
    let device = cpu_device();
    let cfg = UniMerNetConfig::small_2503();
    let model = UniMerNet::<Cpu>::new(&cfg, &device);

    let (l, d) = (10usize, cfg.decoder.d_model);
    let data: Vec<f32> = (0..(l * d)).map(|i| (i as f32 * 0.031).sin()).collect();
    let enc: Tensor<Cpu, 3> = Tensor::from_data(TensorData::new(data, [1, l, d]), &device);

    let ids: Tensor<Cpu, 2, Int> =
        Tensor::from_data(TensorData::new(vec![cfg.decoder.bos_token_id as i64], [1, 1]), &device);
    let logits = model.decode(ids, enc);
    let v = logits.into_data().to_vec::<f32>().expect("logits to vec");

    let spread = v.iter().cloned().fold(f32::MIN, f32::max)
        - v.iter().cloned().fold(f32::MAX, f32::min);
    assert_eq!(
        spread, 0.0,
        "an unloaded model now emits live logits (spread {spread}); a weight-free \
         decoder gate is finally worth writing — add one and delete this test"
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

/// Decodes one raw-RGB8 crop dump: 8-byte little-endian header (`u32` width, `u32`
/// height) then `width*height*3` RGB bytes. Same convention as `MINERU_FORMULA_CROP`;
/// see that test for why file-format decoding is sidestepped.
#[cfg(test)]
fn read_raw_crop(path: &std::path::Path) -> image::RgbImage {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read crop {path:?}: {e}"));
    assert!(bytes.len() > 8, "crop {path:?} is too short to hold a header");
    let w = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let h = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let pixels = bytes[8..].to_vec();
    assert_eq!(
        pixels.len(),
        (w * h * 3) as usize,
        "raw RGB payload of {path:?} does not match {w}x{h}"
    );
    image::RgbImage::from_raw(w, h, pixels).expect("build RgbImage from raw")
}

/// Loads every `*.bin` raw-RGB8 crop in `MINERU_FORMULA_CROP_DIR`, sorted by filename
/// for a deterministic corpus. Same dump format as `MINERU_FORMULA_CROP`, extended to
/// a directory so the batch gate can exercise >16 real crops.
#[cfg(test)]
fn load_crop_corpus() -> Vec<image::RgbImage> {
    let dir = std::env::var("MINERU_FORMULA_CROP_DIR")
        .expect("set MINERU_FORMULA_CROP_DIR to a directory of raw-RGB8 crop dumps");
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read dir {dir}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .collect();
    paths.sort();
    paths.iter().map(|p| read_raw_crop(p)).collect()
}

/// **Real-weights batch parity gate.** Ignored by default.
///
/// The correctness contract for batched formula recognition: decoding N real crops as
/// batches must produce **byte-identical token ids** to decoding each one alone. Set
/// `MINERU_FORMULA_MODEL_DIR` to the checkpoint and `MINERU_FORMULA_CROP_DIR` to a
/// directory of raw-RGB8 crop dumps (see [`read_raw_crop`]).
///
/// Why token ids and not LaTeX or logits: LaTeX hides a divergence that re-converges
/// (two different token paths detokenising to the same string), and "logits are close"
/// hides an argmax flip that sends an autoregressive decoder down a different branch
/// entirely. Only the emitted id sequence is a sound gate.
///
/// The differing-lengths precondition is load-bearing: if every crop decoded to the
/// same length, ragged EOS — the exact thing the done-mask implements — would never be
/// exercised and this test would pass with the mask completely broken.
#[test]
#[ignore = "requires the unimernet_hf_small_2503 checkpoint and a crop corpus on disk"]
fn real_weights_batch_token_parity() {
    use mineru_burn_common::weights::Coverage;
    use mineru_formula::FormulaRecognizer;
    use std::collections::HashSet;

    let dir = std::env::var("MINERU_FORMULA_MODEL_DIR")
        .expect("set MINERU_FORMULA_MODEL_DIR to the checkpoint directory");
    let recognizer = FormulaRecognizer::<Cpu>::from_pretrained(&dir, Coverage::Strict)
        .expect("load recognizer");

    let images = load_crop_corpus();
    // >16 crops so decoding spans multiple batches: a chunk-boundary or reorder bug
    // is invisible in a single full batch.
    assert!(
        images.len() > 16,
        "need >16 crops to cross the default batch_size=16 boundary, got {}",
        images.len()
    );

    let t_sequential = std::time::Instant::now();
    let sequential: Vec<Vec<u32>> = images
        .iter()
        .map(|img| recognizer.predict_token_ids(img).expect("sequential decode"))
        .collect();
    let t_sequential = t_sequential.elapsed();

    // Precondition: without ragged lengths the done-mask is never exercised.
    let lengths: HashSet<usize> = sequential.iter().map(Vec::len).collect();
    assert!(
        lengths.len() > 1,
        "corpus does not exercise ragged EOS: all {} crops decode to the same length",
        sequential.len()
    );

    let t_batched = std::time::Instant::now();
    let batched = recognizer
        .predict_token_ids_batch(&images)
        .expect("batched decode");
    let t_batched = t_batched.elapsed();

    assert_eq!(
        batched.len(),
        sequential.len(),
        "batched decode returned {} lanes for {} inputs",
        batched.len(),
        sequential.len()
    );
    for (i, (want, got)) in sequential.iter().zip(batched.iter()).enumerate() {
        assert_eq!(
            want, got,
            "crop {i} diverged: sequential {} tokens vs batched {} tokens",
            want.len(),
            got.len()
        );
    }
    assert_eq!(sequential, batched, "batched decode diverged from sequential");

    // Reported, never asserted: a timing threshold would make this flake on a busy
    // machine, and the point of this test is parity. The ratio is here so the win is
    // reproducible from a test run rather than an ad-hoc benchmark.
    println!(
        "batched {:.1}s vs sequential {:.1}s over {} crops ({:.2}x)",
        t_batched.as_secs_f64(),
        t_sequential.as_secs_f64(),
        images.len(),
        t_sequential.as_secs_f64() / t_batched.as_secs_f64().max(f64::MIN_POSITIVE),
    );
    println!(
        "real-weights batch parity OK: {} crops byte-identical, lengths {:?}",
        sequential.len(),
        {
            let mut l: Vec<usize> = lengths.into_iter().collect();
            l.sort_unstable();
            l
        }
    );
}

/// Real-weights batch-of-one check. Ignored by default. The degenerate batch must
/// agree with the scalar entry point — this isolates the stacking/argmax plumbing
/// (`[1,1,H,W]` build, `argmax(1)`, single readback) from any ragged-EOS logic.
#[test]
#[ignore = "requires the unimernet_hf_small_2503 checkpoint on disk"]
fn batch_of_one_matches_predict() {
    use mineru_burn_common::weights::Coverage;
    use mineru_formula::FormulaRecognizer;

    let dir = std::env::var("MINERU_FORMULA_MODEL_DIR")
        .expect("set MINERU_FORMULA_MODEL_DIR to the checkpoint directory");
    let recognizer = FormulaRecognizer::<Cpu>::from_pretrained(&dir, Coverage::Strict)
        .expect("load recognizer");

    let img = match std::env::var("MINERU_FORMULA_CROP") {
        Ok(path) => read_raw_crop(std::path::Path::new(&path)),
        Err(_) => image::RgbImage::from_pixel(320, 64, image::Rgb([255, 255, 255])),
    };

    let scalar = recognizer.predict_token_ids(&img).expect("scalar decode");
    let batched = recognizer
        .predict_token_ids_batch(std::slice::from_ref(&img))
        .expect("batch-of-one decode");
    assert_eq!(batched.len(), 1);
    assert_eq!(scalar, batched[0], "batch-of-one diverged from predict");

    let latex_scalar = recognizer.predict(&img).expect("scalar predict");
    let latex_batched = recognizer
        .predict_batch(std::slice::from_ref(&img))
        .expect("batch predict");
    assert_eq!(latex_batched.len(), 1);
    assert_eq!(
        Some(latex_scalar.0.clone()),
        latex_batched[0].as_ref().map(|l| l.0.clone()),
        "batch-of-one LaTeX diverged from predict"
    );
    println!("batch-of-one OK: {} tokens | {}", scalar.len(), latex_scalar.0);
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
