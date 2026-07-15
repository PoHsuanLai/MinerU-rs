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

/// Batch lane-independence gate (weight-free, deterministic, runs in CI).
///
/// Drives `decode_step` with an N-lane batch and, separately, with N one-lane decodes
/// over the same per-lane encoder grids and the same fed tokens, and asserts each
/// lane's **logits are bit-identical** between the two. This is the cross-talk check:
/// if batching ever let row `i` see row `j`'s keys/values — a mask that mixes instead
/// of broadcasting, a reshape that folds the batch into another dim — a batched lane's
/// logits would drift from its solo run here, on every commit, with no checkpoint.
///
/// # Why logits and not token ids here
/// A freshly-initialised [`UniMerNet`] emits **identically zero** logits (the untrained
/// LM head annihilates the signal), so with random weights every lane argmaxes to token
/// 0 no matter what it was fed. A token-sequence comparison would therefore be vacuous
/// — all-zeros vs all-zeros — and would pass with batching totally broken. The hidden
/// state *is* lane-dependent, so the pre-argmax logits are the live signal at this
/// weight regime. This does not weaken the gate: exact equality on the full logits row
/// is strictly stronger than equality of its argmax. Token-id parity on real weights,
/// where the LM head is meaningful, is covered by `real_weights_batch_token_parity`.
///
/// Distinct per-lane hidden states are asserted as a precondition below: with the
/// lanes indistinguishable, cross-talk would be invisible.
#[test]
fn batched_decode_step_matches_per_lane_decodes() {
    let device = cpu_device();
    let cfg = UniMerNetConfig::small_2503();
    let model = UniMerNet::<Cpu>::new(&cfg, &device);

    let (lanes, l, d) = (4usize, 10usize, cfg.decoder.d_model);
    let steps = 12usize;

    // One clearly different encoder grid per lane.
    let lane_grid = |lane: usize| -> Tensor<Cpu, 3> {
        let phase = lane as f32 * 1.7;
        let scale = 0.05 * (1.0 + lane as f32);
        let data: Vec<f32> = (0..(l * d))
            .map(|i| ((i as f32 * 0.031 + phase).sin() + phase.cos()) * scale)
            .collect();
        Tensor::from_data(TensorData::new(data, [1, l, d]), &device)
    };
    // A distinct token stream per lane, so the self-attention cache (not just the
    // cross-attention grid) differs across rows too.
    let lane_token = |lane: usize, position: usize| -> i64 { ((lane * 7 + position * 3) % 50) as i64 };

    // Cross-attention K/V are what carry the encoder grid into the decoder; if the
    // grids collapsed to the same values the lanes would be trivially equal.
    let mut grid_sigs: Vec<Vec<u32>> = Vec::new();
    for lane in 0..lanes {
        let v = lane_grid(lane).into_data().to_vec::<f32>().expect("grid to vec");
        grid_sigs.push(v.iter().map(|x| x.to_bits()).collect());
    }
    let distinct_grids: std::collections::HashSet<_> = grid_sigs.iter().collect();
    assert_eq!(
        distinct_grids.len(),
        lanes,
        "per-lane encoder grids are not distinct; the test cannot see cross-talk"
    );

    // SOLO: each lane decoded on its own, batch dim 1. Collect every step's logits row.
    let solo: Vec<Vec<Vec<f32>>> = (0..lanes)
        .map(|lane| {
            let mut cache = model.init_decode_cache(lane_grid(lane));
            let mut rows = Vec::new();
            for position in 0..steps {
                let input: Tensor<Cpu, 2, Int> = Tensor::from_data(
                    TensorData::new(vec![lane_token(lane, position)], [1, 1]),
                    &device,
                );
                let logits = model.decode_step(input, position, &mut cache);
                rows.push(logits.into_data().to_vec::<f32>().expect("logits to vec"));
            }
            rows
        })
        .collect();

    // BATCHED: all lanes at once, one shared cache, same grids and same fed tokens.
    let batched: Vec<Vec<Vec<f32>>> = {
        let grids: Vec<Tensor<Cpu, 3>> = (0..lanes).map(lane_grid).collect();
        let enc = Tensor::cat(grids, 0); // [lanes, l, d]
        let mut cache = model.init_decode_cache(enc);
        let mut out: Vec<Vec<Vec<f32>>> = vec![Vec::new(); lanes];
        for position in 0..steps {
            let data: Vec<i64> = (0..lanes).map(|lane| lane_token(lane, position)).collect();
            let input: Tensor<Cpu, 2, Int> =
                Tensor::from_data(TensorData::new(data, [lanes, 1]), &device);
            let logits = model.decode_step(input, position, &mut cache); // [lanes, vocab]
            let flat = logits.into_data().to_vec::<f32>().expect("logits to vec");
            let vocab = flat.len() / lanes;
            for (lane, slot) in out.iter_mut().enumerate() {
                slot.push(flat[lane * vocab..(lane + 1) * vocab].to_vec());
            }
        }
        out
    };

    for lane in 0..lanes {
        assert_eq!(
            solo[lane], batched[lane],
            "lane {lane} diverged when batched: batching leaked state across rows"
        );
    }
}

/// Documents the weight-regime trap that shapes the CI batch test above: a freshly
/// initialised [`UniMerNet`] emits **identically zero** logits, so ANY token-level
/// assertion over random weights is vacuous (every lane argmaxes to token 0 regardless
/// of input). Pinning it here means that if Burn's init ever changes to produce live
/// logits, this test fails loudly and the CI gate above can be upgraded from logits to
/// token ids — rather than the trap silently persisting.
#[test]
fn random_weights_produce_degenerate_logits() {
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
        "random-weights logits are no longer degenerate (spread {spread}); the CI batch \
         test can now assert token ids instead of logits"
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
