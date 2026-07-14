//! Top-level model: the Swin encoder + MBart decoder pair, and the public
//! [`FormulaRecognizer`] entry point.
//!
//! [`UniMerNet`] is the Burn [`Module`] graph (what weights load into).
//! [`FormulaRecognizer`] owns a `UniMerNet`, the tokenizer, and the config, and
//! exposes [`FormulaRecognizer::predict`]: image → LaTeX.
//!
//! The generation strategy is the KV-cached greedy loop in [`crate::generate`].
//! We bridge to it by encoding the image **once**, precomputing the decoder's
//! cross-attention K/V from that fixed encoder grid, and then, for each decode step,
//! running the decoder over just the one new token (reusing cached K/V) and
//! returning that position's logits (a [`DecodeStep`] impl).

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use mineru_burn_common::backend::Cpu;
use mineru_burn_common::weights::Coverage;
use mineru_types::Latex;

use crate::config::UniMerNetConfig;
use crate::error::{Error, Result};
use crate::generate::{greedy_decode, DecodeStep};
use crate::latex_cleanup::latex_rm_whitespace;
use crate::mbart::{DecoderCache, MBartDecoder};
use crate::preprocess::{self, PreprocessedImage};
use crate::swin::SwinEncoder;
use crate::tokenizer::LatexTokenizer;

/// The UniMerNet vision-encoder-decoder Burn module.
///
/// Field names (`encoder`, `decoder`) are the anchors the weight remap in
/// [`crate::weights`] targets — do not rename without updating the rules.
#[derive(Module, Debug)]
pub struct UniMerNet<B: Backend> {
    /// Swin vision encoder.
    pub encoder: SwinEncoder<B>,
    /// MBart text decoder (+ LM head).
    pub decoder: MBartDecoder<B>,
}

impl<B: Backend> UniMerNet<B> {
    /// Builds the model with freshly-initialised weights from the config.
    pub fn new(cfg: &UniMerNetConfig, device: &B::Device) -> Self {
        Self {
            encoder: SwinEncoder::new(&cfg.encoder, device),
            decoder: MBartDecoder::new(&cfg.decoder, device),
        }
    }

    /// Encodes an image tensor `[B, 3, H, W]` into visual tokens `[B, L, d]`.
    pub fn encode(&self, pixel_values: Tensor<B, 4>) -> Tensor<B, 3> {
        self.encoder.forward(pixel_values)
    }

    /// Runs the decoder over `input_ids` `[B, T]` given the encoder grid,
    /// returning logits `[B, T, vocab]`.
    ///
    /// This is the non-cached path (recomputes the full prefix). For autoregressive
    /// generation use [`UniMerNet::init_decode_cache`] + [`UniMerNet::decode_step`].
    pub fn decode(
        &self,
        input_ids: Tensor<B, 2, Int>,
        encoder_hidden: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.decoder.forward(input_ids, encoder_hidden)
    }

    /// Precomputes the decoder KV cache (fixed cross-attention K/V) from the encoder
    /// grid. Call once before the greedy loop; see [`UniMerNet::decode_step`].
    pub fn init_decode_cache(&self, encoder_hidden: Tensor<B, 3>) -> DecoderCache<B> {
        self.decoder.init_cache(encoder_hidden)
    }

    /// Runs one incremental decode step for a single new `token` at `position`,
    /// advancing `cache`. Returns logits `[B, vocab]` for the next token.
    pub fn decode_step(
        &self,
        token: Tensor<B, 2, Int>,
        position: usize,
        cache: &mut DecoderCache<B>,
    ) -> Tensor<B, 2> {
        self.decoder.step(token, position, cache)
    }
}

/// A [`DecodeStep`] that holds the encoded image plus a running decoder KV cache and
/// advances one token per step.
///
/// KV-cached: the cross-attention K/V (encoder-derived, fixed) are computed once at
/// construction; each [`DecodeStep::step`] feeds only the one new token, appends its
/// self-attention K/V, and returns the logits at that single position. This is the
/// `O(T)` path that replaced the old `O(T²)` non-cached loop — see [`crate::mbart`].
struct ModelStep<'a, B: Backend> {
    model: &'a UniMerNet<B>,
    cache: DecoderCache<B>,
    device: B::Device,
    vocab_size: usize,
}

impl<B: Backend> DecodeStep for ModelStep<'_, B> {
    fn step(&mut self, token: u32, position: usize) -> u32 {
        let input_ids: Tensor<B, 2, Int> =
            Tensor::from_data(TensorData::new(vec![token as i64], [1, 1]), &self.device);
        // [1, vocab] logits for the next token, computed incrementally.
        let logits = self.model.decode_step(input_ids, position, &mut self.cache);
        let last = logits.reshape([self.vocab_size]);
        // Argmax ON-DEVICE: only the single chosen id crosses to the host, not the
        // whole `vocab`-wide row — avoids a per-token device→host copy that would
        // stall the decode loop on GPU backends. Burn's `argmax` breaks ties toward
        // the lower index, matching the host `argmax`/`torch.argmax` (verified by the
        // formula parity gate producing byte-identical LaTeX after this change).
        let idx = last.argmax(0); // [1], Int
        mineru_burn_common::int_to_vec_i64(idx)
            .first()
            .copied()
            .unwrap_or(0) as u32
    }
}

/// The public formula-recognition entry point.
///
/// Owns the model, tokenizer, and config, and turns a cropped formula image into a
/// [`Latex`] string. Parameterised over the Burn backend `B`; [`Cpu`] is the
/// default via [`FormulaRecognizer::<Cpu>`].
pub struct FormulaRecognizer<B: Backend> {
    model: UniMerNet<B>,
    tokenizer: LatexTokenizer,
    config: UniMerNetConfig,
    device: B::Device,
}

impl<B: Backend> FormulaRecognizer<B> {
    /// Builds a recognizer from an in-memory model + tokenizer + config.
    ///
    /// Use this when you have already constructed and weight-loaded a [`UniMerNet`]
    /// (e.g. in tests, or after a custom load). For the common on-disk case use
    /// [`FormulaRecognizer::from_pretrained`].
    pub fn new(
        model: UniMerNet<B>,
        tokenizer: LatexTokenizer,
        config: UniMerNetConfig,
        device: B::Device,
    ) -> Self {
        Self {
            model,
            tokenizer,
            config,
            device,
        }
    }

    /// Recognizes the LaTeX of a single cropped formula image.
    ///
    /// Pipeline: [`preprocess`] → repeat gray channel to 3 → [`UniMerNet::encode`]
    /// → greedy [`greedy_decode`] → tokenizer decode → [`latex_rm_whitespace`].
    ///
    /// # Errors
    /// Returns [`Error::Image`] on an empty/undecodable image or
    /// [`Error::Tokenizer`] on a decode failure.
    pub fn predict(&self, image: &image::RgbImage) -> Result<Latex> {
        let pre = preprocess::preprocess(image, preprocess::DEFAULT_TARGET)?;
        let pixel_values = self.to_pixel_values(&pre);

        let encoder_hidden = self.model.encode(pixel_values);

        let start = self.config.decoder.bos_token_id as u32;
        let eos = self.config.decoder.eos_token_id as u32;
        // Precompute the fixed cross-attention K/V once from the encoder grid; each
        // step then reuses it and extends only the self-attention cache.
        let cache = self.model.init_decode_cache(encoder_hidden);
        let mut step = ModelStep {
            model: &self.model,
            cache,
            device: self.device.clone(),
            vocab_size: self.config.decoder.vocab_size,
        };
        let decoded = greedy_decode(&mut step, start, eos, self.config.max_new_tokens);

        let raw = self.tokenizer.decode(&decoded.tokens)?;
        let cleaned = latex_rm_whitespace(&raw);
        Ok(Latex(cleaned))
    }

    /// Parity hook: the raw generated token ids from the cached greedy decode.
    ///
    /// Exactly what [`FormulaRecognizer::predict`] produces before tokenizer decode
    /// (BOS dropped, EOS excluded). The KV-cache parity test compares this against a
    /// slow non-cached reference decode to prove the sequences are byte-identical.
    ///
    /// # Errors
    /// Returns [`Error::Image`] on an empty/undecodable image.
    #[doc(hidden)]
    pub fn predict_token_ids(&self, image: &image::RgbImage) -> Result<Vec<u32>> {
        let pre = preprocess::preprocess(image, preprocess::DEFAULT_TARGET)?;
        let pixel_values = self.to_pixel_values(&pre);
        let encoder_hidden = self.model.encode(pixel_values);

        let start = self.config.decoder.bos_token_id as u32;
        let eos = self.config.decoder.eos_token_id as u32;
        let cache = self.model.init_decode_cache(encoder_hidden);
        let mut step = ModelStep {
            model: &self.model,
            cache,
            device: self.device.clone(),
            vocab_size: self.config.decoder.vocab_size,
        };
        Ok(greedy_decode(&mut step, start, eos, self.config.max_new_tokens).tokens)
    }

    /// Parity oracle: greedy decode via the **non-cached** full-prefix decoder.
    ///
    /// A deliberately slow (`O(T²)`) reference — at each step it re-runs the whole
    /// decoder over the growing prefix and argmaxes the last position. Kept only as
    /// the correctness oracle for the KV-cache parity test; production uses the cached
    /// [`FormulaRecognizer::predict_token_ids`]. The two must return byte-identical
    /// token sequences.
    ///
    /// # Errors
    /// Returns [`Error::Image`] on an empty/undecodable image.
    #[doc(hidden)]
    pub fn reference_token_ids_noncache(&self, image: &image::RgbImage) -> Result<Vec<u32>> {
        let pre = preprocess::preprocess(image, preprocess::DEFAULT_TARGET)?;
        let pixel_values = self.to_pixel_values(&pre);
        let encoder_hidden = self.model.encode(pixel_values);

        let start = self.config.decoder.bos_token_id as u32;
        let eos = self.config.decoder.eos_token_id as u32;
        let vocab = self.config.decoder.vocab_size;

        let mut ids: Vec<u32> = vec![start];
        let mut out: Vec<u32> = Vec::new();
        for _ in 0..self.config.max_new_tokens {
            let t = ids.len();
            let data: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
            let input_ids: Tensor<B, 2, Int> =
                Tensor::from_data(TensorData::new(data, [1, t]), &self.device);
            let logits = self.model.decode(input_ids, encoder_hidden.clone()); // [1, T, vocab]
            let last = logits.narrow(1, t - 1, 1).reshape([vocab]);
            let idx = last.argmax(0);
            let next = mineru_burn_common::int_to_vec_i64(idx)
                .first()
                .copied()
                .unwrap_or(0) as u32;
            if next == eos {
                break;
            }
            out.push(next);
            ids.push(next);
        }
        Ok(out)
    }

    /// Parity hook: run preprocessing and return the `[1, 3, H, W]` pixel tensor.
    ///
    /// Exposes exactly what [`FormulaRecognizer::predict`] feeds the encoder (margin
    /// crop → resize → pad → grayscale-normalise → repeat to 3 channels), so the
    /// parity test can diff the Rust preprocess path against the Python input.
    ///
    /// # Errors
    /// Returns [`Error::Image`] on an empty/undecodable image.
    #[doc(hidden)]
    pub fn preprocess_pixels(&self, image: &image::RgbImage) -> Result<Tensor<B, 4>> {
        let pre = preprocess::preprocess(image, preprocess::DEFAULT_TARGET)?;
        Ok(self.to_pixel_values(&pre))
    }

    /// Parity hook: Swin encoder forward exposing per-stage activations.
    ///
    /// See [`SwinEncoder::forward_stages`]. Used only by the parity test.
    #[doc(hidden)]
    pub fn encode_stages(&self, pixel_values: Tensor<B, 4>) -> (Tensor<B, 3>, Vec<Tensor<B, 3>>) {
        self.model.encoder.forward_stages(pixel_values)
    }

    /// Parity hook: first-step decoder logits `[1, vocab]` for a fixed prefix.
    ///
    /// Runs the decoder over `input_ids` `[1, T]` given `encoder_hidden`, and returns
    /// the logits at the last position — the deterministic quantity the parity test
    /// compares (feed just the BOS token for the true first step).
    #[doc(hidden)]
    pub fn decoder_step_logits(
        &self,
        input_ids: &[u32],
        encoder_hidden: Tensor<B, 3>,
    ) -> Tensor<B, 2> {
        let t = input_ids.len();
        let data: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
        let ids: Tensor<B, 2, Int> = Tensor::from_data(TensorData::new(data, [1, t]), &self.device);
        let logits = self.model.decode(ids, encoder_hidden); // [1, T, vocab]
        let vocab = self.config.decoder.vocab_size;
        logits.narrow(1, t - 1, 1).reshape([1, vocab])
    }

    /// Turns the single normalised channel into a `[1, 3, H, W]` pixel tensor by
    /// repeating the channel three times (mirroring `pixel_values.repeat(1,3,1,1)`).
    fn to_pixel_values(&self, pre: &PreprocessedImage) -> Tensor<B, 4> {
        let (h, w) = (pre.height, pre.width);
        let plane: Tensor<B, 3> =
            Tensor::from_data(TensorData::new(pre.data.clone(), [1, h, w]), &self.device);
        // [1, 1, H, W] -> repeat to [1, 3, H, W]
        plane.reshape([1, 1, h, w]).repeat_dim(1, 3)
    }
}

impl<B: Backend> FormulaRecognizer<B> {
    /// Loads a recognizer from a checkpoint directory onto an explicit device.
    ///
    /// Backend-generic form of [`FormulaRecognizer::from_pretrained`]: identical
    /// loading, but on the caller's `device`/backend `B` (e.g. the wgpu GPU).
    ///
    /// Expects `dir` to contain `model.safetensors` and `tokenizer.json`. Weight
    /// loading uses the shared harness with the [`crate::weights::build_remap`]
    /// rules; `coverage` controls how unmapped keys are treated.
    ///
    /// # Errors
    /// Returns an error if the tokenizer or weights fail to load. See
    /// [`crate::weights`] for the key-mismatch caveats.
    pub fn from_pretrained_on(
        dir: impl AsRef<std::path::Path>,
        coverage: Coverage,
        device: B::Device,
    ) -> Result<Self> {
        use mineru_burn_common::weights::load_weights_ignoring;

        let dir = dir.as_ref();
        let config = UniMerNetConfig::small_2503();

        let mut model = UniMerNet::<B>::new(&config, &device);
        let remap = crate::weights::build_remap()?;
        let weights_path = dir.join("model.safetensors");
        // `IGNORED_KEYS` are training-only / recomputed buffers with no inference
        // field (relative_position_index, num_batches_tracked); see `weights`.
        load_weights_ignoring(
            &mut model,
            &weights_path,
            &remap,
            coverage,
            crate::weights::IGNORED_KEYS,
        )?;

        let tokenizer = LatexTokenizer::from_file(dir.join("tokenizer.json"))?;
        if tokenizer.vocab_size() != config.decoder.vocab_size {
            return Err(Error::Config(format!(
                "tokenizer vocab {} != decoder vocab {}",
                tokenizer.vocab_size(),
                config.decoder.vocab_size
            )));
        }

        Ok(Self::new(model, tokenizer, config, device))
    }
}

impl FormulaRecognizer<Cpu> {
    /// Loads a recognizer from a checkpoint directory on the CPU backend.
    ///
    /// Expects `dir` to contain `model.safetensors` and `tokenizer.json`. Weight
    /// loading uses the shared harness with the [`crate::weights::build_remap`]
    /// rules; `coverage` controls how unmapped keys are treated (start with
    /// [`Coverage::Lenient`] until the remap is verified against the real file).
    ///
    /// # Errors
    /// Returns an error if the tokenizer or weights fail to load. See
    /// [`crate::weights`] for the key-mismatch caveats.
    pub fn from_pretrained(
        dir: impl AsRef<std::path::Path>,
        coverage: Coverage,
    ) -> Result<Self> {
        use mineru_burn_common::backend::cpu_device;

        Self::from_pretrained_on(dir, coverage, cpu_device())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_builds_on_cpu() {
        // Constructing the full graph exercises every module's `new` and confirms
        // the dims line up (dim doubling, head divisibility, etc.) without weights.
        let device = Default::default();
        let cfg = UniMerNetConfig::small_2503();
        let _model = UniMerNet::<Cpu>::new(&cfg, &device);
    }
}
