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
        enc: Tensor<Cpu, 3>,
        device: burn::backend::ndarray::NdArrayDevice,
        vocab: usize,
    }
    impl DecodeStep for Step<'_> {
        fn step(&mut self, ids: &[u32]) -> Vec<f32> {
            let t = ids.len();
            let data: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
            let input: Tensor<Cpu, 2, Int> =
                Tensor::from_data(TensorData::new(data, [1, t]), &self.device);
            let logits = self.model.decode(input, self.enc.clone());
            logits
                .narrow(1, t - 1, 1)
                .reshape([self.vocab])
                .into_data()
                .to_vec()
                .expect("logits to vec")
        }
    }

    let mut step = Step {
        model: &model,
        enc,
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
