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
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::config::MBartConfig;
use crate::mbart::attention::MBartAttention;

/// A single decoder layer: causal self-attn + cross-attn + FFN, all pre-norm.
#[derive(Module, Debug)]
pub struct MBartDecoderLayer<B: Backend> {
    self_attn: MBartAttention<B>,
    self_attn_layer_norm: LayerNorm<B>,
    encoder_attn: MBartAttention<B>,
    encoder_attn_layer_norm: LayerNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    final_layer_norm: LayerNorm<B>,
}

impl<B: Backend> MBartDecoderLayer<B> {
    /// Builds a decoder layer from the config.
    pub fn new(cfg: &MBartConfig, device: &B::Device) -> Self {
        let ln = || {
            LayerNormConfig::new(cfg.d_model)
                .with_epsilon(cfg.layer_norm_eps)
                .init(device)
        };
        Self {
            self_attn: MBartAttention::new(cfg, device),
            self_attn_layer_norm: ln(),
            encoder_attn: MBartAttention::new(cfg, device),
            encoder_attn_layer_norm: ln(),
            fc1: LinearConfig::new(cfg.d_model, cfg.decoder_ffn_dim).init(device),
            fc2: LinearConfig::new(cfg.decoder_ffn_dim, cfg.d_model).init(device),
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
}
