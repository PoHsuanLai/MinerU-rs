//! MBart *squeeze* multi-head attention (`UnimerMBartAttention`).
//!
//! Port of `UnimerMBartAttention` from `unimer_mbart/modeling_unimer_mbart.py`
//! (the eager path). UniMerNet's twist over vanilla MBart attention is
//! **squeeze**: query and key are projected to `d_model / qk_squeeze` (here
//! `768/2 = 384`) while value and output stay full width (`768`). This shrinks the
//! `Q·Kᵀ` matmul. Head count is unchanged, so the squeezed per-head dim is
//! `squeeze_dim / heads = 24` and the value per-head dim is `d_model / heads = 48`.
//!
//! Used both as **causal self-attention** (over decoder tokens) and as
//! **cross-attention** (query = decoder tokens, key/value = encoder grid). The
//! caller supplies the additive mask: a causal `[tgt, tgt]` mask for self-attn, or
//! `None` for cross-attn (the encoder grid is fully visible).
//!
//! This is the **non-cached** form: every step recomputes K/V over the full
//! prefix. A KV cache is the documented optimization (see [`crate::generate`]).

use burn::module::Module;
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use mineru_burn_common::nn::PtLinear;

use crate::config::MBartConfig;

/// Squeeze multi-head attention.
#[derive(Module, Debug)]
pub struct MBartAttention<B: Backend> {
    q_proj: PtLinear<B>,
    k_proj: PtLinear<B>,
    v_proj: PtLinear<B>,
    out_proj: PtLinear<B>,
    num_heads: usize,
    squeeze_head_dim: usize,
    head_dim: usize,
    scaling: f64,
}

impl<B: Backend> MBartAttention<B> {
    /// Builds an attention module from the decoder config.
    pub fn new(cfg: &MBartConfig, device: &B::Device) -> Self {
        let d = cfg.d_model;
        let sq = cfg.squeeze_dim();
        Self {
            q_proj: PtLinear::init(d, sq, true, device),
            k_proj: PtLinear::init(d, sq, true, device),
            v_proj: PtLinear::init(d, d, true, device),
            out_proj: PtLinear::init(d, d, true, device),
            num_heads: cfg.decoder_attention_heads,
            squeeze_head_dim: cfg.squeeze_head_dim(),
            head_dim: cfg.head_dim(),
            scaling: (cfg.squeeze_head_dim() as f64).powf(-0.5),
        }
    }

    fn shape_qk(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, n, _] = x.dims();
        x.reshape([b, n, self.num_heads, self.squeeze_head_dim])
            .swap_dims(1, 2)
    }

    fn shape_v(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, n, _] = x.dims();
        x.reshape([b, n, self.num_heads, self.head_dim])
            .swap_dims(1, 2)
    }

    /// Runs attention.
    ///
    /// - `hidden`: `[B, tgt, d_model]` (the query source).
    /// - `key_value`: `Some([B, src, d_model])` for cross-attention, or `None` to
    ///   use `hidden` as the key/value source (self-attention).
    /// - `attn_mask`: optional additive `[tgt, src]` mask (e.g. causal), broadcast
    ///   over batch and heads.
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        key_value: Option<Tensor<B, 3>>,
        attn_mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let [b, tgt, d] = hidden.dims();
        let kv = key_value.unwrap_or_else(|| hidden.clone());

        let q = self.shape_qk(self.q_proj.forward(hidden)).mul_scalar(self.scaling);
        let k = self.shape_qk(self.k_proj.forward(kv.clone()));
        let v = self.shape_v(self.v_proj.forward(kv));

        // [B, heads, tgt, src]
        let mut scores = q.matmul(k.swap_dims(2, 3));
        if let Some(mask) = attn_mask {
            let src = scores.dims()[3];
            let mask = mask.reshape([1, 1, tgt, src]);
            scores = scores + mask;
        }
        let probs = softmax(scores, 3);
        // ctx = probs @ v, written as `(vᵀ @ probsᵀ)ᵀ`: `v` (from `shape_v`) is a
        // batch-permuted (`swap_dims(1, 2)`) view, and Burn 0.21's wgpu matmul reads
        // a batch-permuted right-hand operand with wrong strides. Keeping it on the
        // left with a last-two-dim transpose on the right avoids the bug (no-op on
        // CPU). See the matching note in mineru-layout's `encoder.rs`.
        let ctx = v.swap_dims(2, 3).matmul(probs.swap_dims(2, 3)).swap_dims(2, 3); // [B, heads, tgt, head_dim]
        let ctx = ctx.swap_dims(1, 2).reshape([b, tgt, d]);
        self.out_proj.forward(ctx)
    }
}
