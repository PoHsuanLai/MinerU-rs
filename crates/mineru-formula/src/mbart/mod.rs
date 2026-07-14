//! MBart autoregressive decoder + LM head (`UnimerMBartForCausalLM`).
//!
//! Port of the decoder path of `unimer_mbart/modeling_unimer_mbart.py`:
//! `UnimerMBartDecoder` (embeddings + layers + final norm) wrapped by
//! `UnimerMBartForCausalLM` (which adds the `lm_head`). In the checkpoint the
//! decoder lives under `UnimerMBartDecoderWrapper.decoder`, so field names are
//! chosen to match (see [`crate::weights`] for the key remap).
//!
//! Forward (non-cached): given decoder token ids `[B, T]` and the encoder grid
//! `[B, L, d]`, produce logits `[B, T, vocab]`.
//!
//! Embedding pipeline (matches the Python):
//! ```text
//! e = embed_tokens(ids) * sqrt(d_model)          # scaled word embedding
//! p = embed_positions[pos + OFFSET]              # learned, offset by 2
//! h = layernorm_embedding(e + p)
//! for layer in layers: h = layer(h, enc, causal_mask)
//! h = layer_norm(h)                              # final norm (MBart post-norm on output)
//! logits = lm_head(h)
//! ```
//!
//! # Incremental (KV-cached) decode
//! [`MBartDecoder::forward`] is the reference path: it recomputes the whole prefix
//! every call (`O(T²)` across a greedy decode). For autoregressive generation
//! [`MBartDecoder::init_cache`] + [`MBartDecoder::step`] give the `O(T)` path: the
//! cross-attention K/V (encoder-derived, fixed) are computed once into a
//! [`DecoderCache`], and each step embeds only the one new token at its position,
//! attends over the cached self- and cross-K/V, and emits logits for that single
//! position. The output is arithmetically identical to slicing the last position out
//! of the non-cached forward — the parity gate in [`crate::generate`] pins this.

pub mod attention;
pub mod layer;

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use mineru_burn_common::nn::{PtLayerNorm, PtLinear};

use crate::config::MBartConfig;
use layer::{LayerCache, MBartDecoderLayer};

/// The MBart decoder with an LM head.
#[derive(Module, Debug)]
pub struct MBartDecoder<B: Backend> {
    embed_tokens: Embedding<B>,
    embed_positions: Embedding<B>,
    layernorm_embedding: PtLayerNorm<B>,
    layers: Vec<MBartDecoderLayer<B>>,
    layer_norm: PtLayerNorm<B>,
    lm_head: PtLinear<B>,
    embed_scale: f64,
    position_offset: usize,
}

impl<B: Backend> MBartDecoder<B> {
    /// Builds the decoder from the config.
    pub fn new(cfg: &MBartConfig, device: &B::Device) -> Self {
        let ln = || PtLayerNorm::init(cfg.d_model, cfg.layer_norm_eps, device);
        let embed_scale = if cfg.scale_embedding {
            (cfg.d_model as f64).sqrt()
        } else {
            1.0
        };
        Self {
            embed_tokens: EmbeddingConfig::new(cfg.vocab_size, cfg.d_model).init(device),
            embed_positions: EmbeddingConfig::new(
                cfg.max_position_embeddings + MBartConfig::POSITION_OFFSET,
                cfg.d_model,
            )
            .init(device),
            layernorm_embedding: ln(),
            layers: (0..cfg.decoder_layers)
                .map(|_| MBartDecoderLayer::new(cfg, device))
                .collect(),
            layer_norm: ln(),
            lm_head: PtLinear::init(cfg.d_model, cfg.vocab_size, false, device),
            embed_scale,
            position_offset: MBartConfig::POSITION_OFFSET,
        }
    }

    /// Runs the decoder and LM head.
    ///
    /// - `input_ids`: `[B, T]` decoder token ids.
    /// - `encoder_hidden`: `[B, L, d]` visual tokens.
    ///
    /// Returns logits `[B, T, vocab]`.
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        encoder_hidden: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let device = input_ids.device();
        let [b, t] = input_ids.dims();

        // Scaled word embeddings.
        let mut hidden = self.embed_tokens.forward(input_ids).mul_scalar(self.embed_scale);

        // Learned positional embeddings at positions [offset, offset+T).
        let positions: Vec<i64> = (0..t)
            .map(|p| (p + self.position_offset) as i64)
            .collect();
        let pos_ids: Tensor<B, 2, Int> =
            Tensor::from_data(TensorData::new(positions, [1, t]), &device).repeat_dim(0, b);
        let pos = self.embed_positions.forward(pos_ids);
        hidden = hidden + pos;

        hidden = self.layernorm_embedding.forward(hidden);

        let causal = causal_mask::<B>(t, &device);
        for l in &self.layers {
            hidden = l.forward(hidden, encoder_hidden.clone(), causal.clone());
        }
        hidden = self.layer_norm.forward(hidden);

        self.lm_head.forward(hidden)
    }

    /// Builds the incremental-decode cache: precomputes every layer's cross-attention
    /// K/V from the fixed encoder grid and seeds empty self-attention caches.
    ///
    /// Call once per image before the greedy loop; then drive [`MBartDecoder::step`]
    /// once per generated token, feeding the same [`DecoderCache`] back in each time.
    pub fn init_cache(&self, encoder_hidden: Tensor<B, 3>) -> DecoderCache<B> {
        let layers = self
            .layers
            .iter()
            .map(|l| LayerCache::new(l.cross_kv(encoder_hidden.clone())))
            .collect();
        DecoderCache { layers }
    }

    /// Runs one incremental decode step for a single new token.
    ///
    /// - `token`: the id emitted at the previous step (or BOS for the first step).
    /// - `position`: this token's 0-based position in the sequence (the running step
    ///   count); the learned positional embedding is looked up at `position + OFFSET`.
    /// - `cache`: the [`DecoderCache`] from [`MBartDecoder::init_cache`], extended in
    ///   place.
    ///
    /// Returns logits `[B, vocab]` for the next token — the same row the non-cached
    /// [`MBartDecoder::forward`] produces at the last position, but computed in `O(1)`
    /// decoder work for this step instead of `O(T)`.
    pub fn step(
        &self,
        token: Tensor<B, 2, Int>,
        position: usize,
        cache: &mut DecoderCache<B>,
    ) -> Tensor<B, 2> {
        let device = token.device();
        let [b, _one] = token.dims();

        // Scaled word embedding for the single new token: [B, 1, d].
        let mut hidden = self.embed_tokens.forward(token).mul_scalar(self.embed_scale);

        // Learned positional embedding at this one position (offset by 2).
        let pos_ids: Tensor<B, 2, Int> = Tensor::from_data(
            TensorData::new(vec![(position + self.position_offset) as i64], [1, 1]),
            &device,
        )
        .repeat_dim(0, b);
        hidden = hidden + self.embed_positions.forward(pos_ids);

        hidden = self.layernorm_embedding.forward(hidden);

        for (l, lc) in self.layers.iter().zip(cache.layers.iter_mut()) {
            hidden = l.step(hidden, lc);
        }
        hidden = self.layer_norm.forward(hidden);

        // [B, 1, vocab] -> [B, vocab].
        let logits = self.lm_head.forward(hidden);
        let vocab = logits.dims()[2];
        logits.reshape([b, vocab])
    }
}

/// Cross-step state for KV-cached decoding: one [`LayerCache`] per decoder layer.
///
/// Created by [`MBartDecoder::init_cache`] and advanced by [`MBartDecoder::step`].
/// Holding the growing self-attention K/V and the fixed cross-attention K/V here is
/// what turns the `O(T²)` non-cached loop into an `O(T)` one.
#[derive(Debug, Clone)]
pub struct DecoderCache<B: Backend> {
    /// Per-layer caches, in decoder-layer order.
    layers: Vec<LayerCache<B>>,
}

/// Builds an additive causal mask `[t, t]`: `0` on/below the diagonal, a large
/// negative value above (future positions).
///
/// Uses `-1e9` rather than `-inf` to stay finite under softmax on all backends.
pub fn causal_mask<B: Backend>(t: usize, device: &B::Device) -> Tensor<B, 2> {
    // Burn's triangular masks use inverted semantics: the mask is `true` at the
    // positions to *fill*. `tril_mask(offset=0)` is `false` on the lower triangle
    // and diagonal (kept) and `true` strictly above the diagonal — exactly the
    // future positions we forbid. Fill those with a large negative; the rest stay 0.
    let forbid = Tensor::<B, 2, burn::tensor::Bool>::tril_mask([t, t], 0, device);
    Tensor::<B, 2>::zeros([t, t], device).mask_fill(forbid, -1.0e9)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use mineru_burn_common::backend::Cpu;

    #[test]
    fn causal_mask_blocks_future() {
        let device = Default::default();
        let mask = causal_mask::<Cpu>(3, &device);
        let data: Vec<f32> = mask
            .into_data()
            .to_vec()
            .expect("mask data should convert to Vec<f32>");
        // Row 0 attends only to col 0: cols 1,2 blocked.
        assert_eq!(data[0], 0.0);
        assert!(data[1] < -1.0e8);
        assert!(data[2] < -1.0e8);
        // Row 2 attends to all: no blocking.
        assert_eq!(data[6], 0.0);
        assert_eq!(data[7], 0.0);
        assert_eq!(data[8], 0.0);
    }

    #[test]
    fn position_ids_include_offset() {
        // Sanity: OFFSET is 2, so a length-3 sequence maps to positions [2,3,4].
        let _ = TensorData::new(vec![2i64, 3, 4], [1, 3]);
        assert_eq!(MBartConfig::POSITION_OFFSET, 2);
    }
}
