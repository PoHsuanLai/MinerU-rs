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
//! Two paths coexist:
//! - the **non-cached** [`MBartAttention::forward`]: every step recomputes K/V over
//!   the full prefix (`O(T²)` across a decode). This is the reference path and the
//!   one the parity test pins against.
//! - the **incremental** [`MBartAttention::forward_self_cached`] /
//!   [`MBartAttention::cross_kv`] / [`MBartAttention::forward_cross_cached`]: used by
//!   the KV-cache decode loop. Self-attention appends the new token's K/V to a
//!   running cache and computes Q for only the 1 new position; cross-attention's K/V
//!   depend solely on the (fixed) encoder grid, so they are computed once and reused
//!   every step. Both produce output arithmetically identical to the non-cached path
//!   for the last position (see [`crate::generate`] for the correctness gate).

use burn::module::Module;
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use mineru_burn_common::nn::PtLinear;

use crate::config::MBartConfig;

/// Cached key/value tensors for one attention module during incremental decode.
///
/// Shapes are `[B, heads, len, squeeze_head_dim]` for `k` and
/// `[B, heads, len, head_dim]` for `v` — exactly the shaped-and-permuted layout the
/// non-cached path builds before the score matmul, so a cached step can concatenate
/// the new token's rows and proceed without any reshaping. For self-attention `len`
/// grows by one per step; for cross-attention it is the fixed encoder length.
#[derive(Debug, Clone)]
pub struct KvCache<B: Backend> {
    /// Keys `[B, heads, len, squeeze_head_dim]`.
    k: Tensor<B, 4>,
    /// Values `[B, heads, len, head_dim]`.
    v: Tensor<B, 4>,
}

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

        self.attend(q, k, v, [b, tgt, d], attn_mask)
    }

    /// Core scaled-dot-product attention given already-shaped `q`/`k`/`v`.
    ///
    /// `q`: `[B, heads, tgt, squeeze_head_dim]` (pre-scaled), `k`:
    /// `[B, heads, src, squeeze_head_dim]`, `v`: `[B, heads, src, head_dim]`. Returns
    /// the projected output `[B, tgt, d_model]`. Shared verbatim by the non-cached
    /// [`MBartAttention::forward`] and the incremental cached paths so all three are
    /// arithmetically identical.
    fn attend(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        out_dims: [usize; 3],
        attn_mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let [b, tgt, d] = out_dims;
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

    /// Incremental **causal self-attention** for one decode step.
    ///
    /// `hidden` is the single new token `[B, 1, d_model]`. Its K/V rows are appended
    /// to `cache` (created on the first step), and the new token's Q attends over the
    /// full cached K/V. No causal mask is needed: the one query row is the newest
    /// position, which legitimately attends to every cached key (all are at or before
    /// it). Returns the attention output `[B, 1, d_model]`.
    pub fn forward_self_cached(
        &self,
        hidden: Tensor<B, 3>,
        cache: &mut Option<KvCache<B>>,
    ) -> Tensor<B, 3> {
        let [b, tgt, d] = hidden.dims();

        let q = self.shape_qk(self.q_proj.forward(hidden.clone())).mul_scalar(self.scaling);
        let k_new = self.shape_qk(self.k_proj.forward(hidden.clone()));
        let v_new = self.shape_v(self.v_proj.forward(hidden));

        let (k, v) = match cache.take() {
            Some(prev) => (
                Tensor::cat(vec![prev.k, k_new], 2),
                Tensor::cat(vec![prev.v, v_new], 2),
            ),
            None => (k_new, v_new),
        };
        *cache = Some(KvCache {
            k: k.clone(),
            v: v.clone(),
        });

        self.attend(q, k, v, [b, tgt, d], None)
    }

    /// Precomputes the **cross-attention** key/value tensors from the fixed encoder
    /// grid `encoder_hidden` `[B, src, d_model]`.
    ///
    /// These depend only on the encoder output, which is constant across the whole
    /// decode, so this is called once and the returned [`KvCache`] is reused by
    /// [`MBartAttention::forward_cross_cached`] every step.
    pub fn cross_kv(&self, encoder_hidden: Tensor<B, 3>) -> KvCache<B> {
        let k = self.shape_qk(self.k_proj.forward(encoder_hidden.clone()));
        let v = self.shape_v(self.v_proj.forward(encoder_hidden));
        KvCache { k, v }
    }

    /// Incremental **cross-attention** for one decode step using cached encoder K/V.
    ///
    /// `hidden` is the single new decoder token `[B, 1, d_model]`; `cache` holds the
    /// encoder-derived K/V from [`MBartAttention::cross_kv`]. Returns the attention
    /// output `[B, 1, d_model]`.
    pub fn forward_cross_cached(
        &self,
        hidden: Tensor<B, 3>,
        cache: &KvCache<B>,
    ) -> Tensor<B, 3> {
        let [b, tgt, d] = hidden.dims();
        let q = self.shape_qk(self.q_proj.forward(hidden)).mul_scalar(self.scaling);
        self.attend(q, cache.k.clone(), cache.v.clone(), [b, tgt, d], None)
    }
}
