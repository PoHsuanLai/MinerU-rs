//! One MBart decoder layer (`UnimerMBartDecoderLayer`).
//!
//! Port of `UnimerMBartDecoderLayer`. MBart is **pre-norm**: each sub-block
//! layer-norms its input, applies the sublayer, and adds the residual.
//!
//! ```text
//! h = h + self_attn(   self_attn_ln(h),          causal mask )
//! h = h + cross_attn(  encoder_attn_ln(h), enc,  no mask     )
//! h = h + fc2(gelu(fc1( final_ln(h) )))
//! ```

use burn::module::Module;
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use mineru_burn_common::nn::{PtLayerNorm, PtLinear};

use crate::config::MBartConfig;
use crate::mbart::attention::{KvCache, MBartAttention};

/// A single decoder layer: causal self-attn + cross-attn + FFN, all pre-norm.
#[derive(Module, Debug)]
pub struct MBartDecoderLayer<B: Backend> {
    self_attn: MBartAttention<B>,
    self_attn_layer_norm: PtLayerNorm<B>,
    encoder_attn: MBartAttention<B>,
    encoder_attn_layer_norm: PtLayerNorm<B>,
    fc1: PtLinear<B>,
    fc2: PtLinear<B>,
    final_layer_norm: PtLayerNorm<B>,
}

impl<B: Backend> MBartDecoderLayer<B> {
    /// Builds a decoder layer from the config.
    pub fn new(cfg: &MBartConfig, device: &B::Device) -> Self {
        let ln = || PtLayerNorm::init(cfg.d_model, cfg.layer_norm_eps, device);
        Self {
            self_attn: MBartAttention::new(cfg, device),
            self_attn_layer_norm: ln(),
            encoder_attn: MBartAttention::new(cfg, device),
            encoder_attn_layer_norm: ln(),
            fc1: PtLinear::init(cfg.d_model, cfg.decoder_ffn_dim, true, device),
            fc2: PtLinear::init(cfg.decoder_ffn_dim, cfg.d_model, true, device),
            final_layer_norm: ln(),
        }
    }

    /// Runs the layer.
    ///
    /// - `hidden`: `[B, tgt, d_model]`.
    /// - `encoder_hidden`: `[B, src, d_model]` visual tokens for cross-attention.
    /// - `causal_mask`: additive `[tgt, tgt]` mask for the self-attention.
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        encoder_hidden: Tensor<B, 3>,
        causal_mask: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        // Self-attention (causal).
        let residual = hidden.clone();
        let x = self.self_attn_layer_norm.forward(hidden);
        let x = self.self_attn.forward(x, None, Some(causal_mask));
        let hidden = residual + x;

        // Cross-attention over the encoder grid (no mask).
        let residual = hidden.clone();
        let x = self.encoder_attn_layer_norm.forward(hidden);
        let x = self.encoder_attn.forward(x, Some(encoder_hidden), None);
        let hidden = residual + x;

        // Feed-forward.
        let residual = hidden.clone();
        let x = self.final_layer_norm.forward(hidden);
        let x = self.fc2.forward(gelu(self.fc1.forward(x)));
        residual + x
    }

    /// Precomputes this layer's cross-attention K/V from the (fixed) encoder grid.
    ///
    /// Called once at the start of a cached decode; the result is stored in
    /// [`LayerCache::cross`] and reused every step by [`MBartDecoderLayer::step`].
    pub fn cross_kv(&self, encoder_hidden: Tensor<B, 3>) -> KvCache<B> {
        self.encoder_attn.cross_kv(encoder_hidden)
    }

    /// Runs the layer for one incremental decode step.
    ///
    /// - `hidden`: the single new token `[B, 1, d_model]`.
    /// - `cache`: this layer's running self-attention K/V cache plus the precomputed
    ///   cross-attention K/V. The self-attention cache is extended in place.
    ///
    /// Produces the same output as [`MBartDecoderLayer::forward`] would at the last
    /// position — the pre-norm structure and sublayer maths are identical; only K/V
    /// reuse differs.
    pub fn step(&self, hidden: Tensor<B, 3>, cache: &mut LayerCache<B>) -> Tensor<B, 3> {
        // Self-attention (causal, via the running K/V cache).
        let residual = hidden.clone();
        let x = self.self_attn_layer_norm.forward(hidden);
        let x = self.self_attn.forward_self_cached(x, &mut cache.self_attn);
        let hidden = residual + x;

        // Cross-attention over the (cached) encoder grid.
        let residual = hidden.clone();
        let x = self.encoder_attn_layer_norm.forward(hidden);
        let x = self.encoder_attn.forward_cross_cached(x, &cache.cross);
        let hidden = residual + x;

        // Feed-forward.
        let residual = hidden.clone();
        let x = self.final_layer_norm.forward(hidden);
        let x = self.fc2.forward(gelu(self.fc1.forward(x)));
        residual + x
    }
}

/// Per-layer state carried across incremental decode steps.
///
/// Holds the growing self-attention K/V cache (extended one row per step) and the
/// fixed cross-attention K/V (encoder-derived, computed once). One of these exists
/// per decoder layer inside [`crate::mbart::DecoderCache`].
#[derive(Debug, Clone)]
pub struct LayerCache<B: Backend> {
    /// Running self-attention K/V; `None` until the first step populates it.
    self_attn: Option<KvCache<B>>,
    /// Fixed cross-attention K/V from the encoder grid.
    cross: KvCache<B>,
}

impl<B: Backend> LayerCache<B> {
    /// Creates the layer cache with cross-attention K/V precomputed from the encoder
    /// grid and an empty self-attention cache.
    pub fn new(cross: KvCache<B>) -> Self {
        Self {
            self_attn: None,
            cross,
        }
    }
}
