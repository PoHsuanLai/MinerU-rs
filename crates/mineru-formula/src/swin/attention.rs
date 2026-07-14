//! Swin windowed multi-head self-attention with a relative-position bias.
//!
//! Port of `UnimerSwinSelfAttention` + `UnimerSwinSelfOutput` +
//! `UnimerSwinAttention` from `unimer_swin/modeling_unimer_swin.py`.
//!
//! Attention runs *inside each window* (the caller partitions the feature map into
//! `window_size × window_size` windows). A learned bias, indexed by the relative
//! position of each query/key pair within the window, is added to the scores. The
//! bias table has shape `[(2*W-1)*(2*W-1), num_heads]` and the fixed index buffer
//! `relative_position_index` (shape `[W*W, W*W]`) selects rows per pair — both are
//! reproduced exactly so the checkpoint's `relative_position_bias_table` loads.

use burn::module::{Module, Param};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use mineru_burn_common::nn::PtLinear;

use crate::config::SwinConfig;

/// Precomputes the flattened `relative_position_index` for a square window.
///
/// Mirrors the meshgrid construction in `UnimerSwinSelfAttention.__init__`:
/// for a window of side `w`, returns a `w*w * w*w` vector of row indices into the
/// bias table, in row-major `[query, key]` order.
pub fn relative_position_index(window: usize) -> Vec<i64> {
    let n = window * window;
    // coords[i] = (row, col) for flattened position i (row-major over the window).
    let coords: Vec<(i64, i64)> = (0..n)
        .map(|i| ((i / window) as i64, (i % window) as i64))
        .collect();
    let w = window as i64;
    let mut idx = Vec::with_capacity(n * n);
    for &(qr, qc) in &coords {
        for &(kr, kc) in &coords {
            // relative_coords = q - k, then shifted into [0, 2W-2] and linearised.
            let mut rh = qr - kr + (w - 1);
            let rw = qc - kc + (w - 1);
            rh *= 2 * w - 1;
            idx.push(rh + rw);
        }
    }
    idx
}

/// Windowed multi-head self-attention for one Swin stage.
#[derive(Module, Debug)]
pub struct WindowAttention<B: Backend> {
    query: PtLinear<B>,
    key: PtLinear<B>,
    value: PtLinear<B>,
    /// The `UnimerSwinSelfOutput.dense` projection applied after attention.
    output: PtLinear<B>,
    /// Learned bias table, `[(2W-1)^2, num_heads]`. Loaded from the checkpoint.
    relative_position_bias_table: Param<Tensor<B, 2>>,
    num_heads: usize,
    head_dim: usize,
    window_size: usize,
}

impl<B: Backend> WindowAttention<B> {
    /// Builds the attention module for a stage of dimension `dim` with `num_heads`.
    pub fn new(cfg: &SwinConfig, dim: usize, num_heads: usize, device: &B::Device) -> Self {
        let window = cfg.window_size;
        let table_rows = (2 * window - 1) * (2 * window - 1);
        let linear = |d_in: usize, d_out: usize, bias: bool| {
            PtLinear::init(d_in, d_out, bias, device)
        };
        Self {
            query: linear(dim, dim, cfg.qkv_bias),
            key: linear(dim, dim, cfg.qkv_bias),
            value: linear(dim, dim, cfg.qkv_bias),
            output: linear(dim, dim, true),
            relative_position_bias_table: Param::from_tensor(Tensor::zeros(
                [table_rows, num_heads],
                device,
            )),
            num_heads,
            head_dim: dim / num_heads,
            window_size: window,
        }
    }

    /// Splits `[B, N, dim]` into heads `[B, heads, N, head_dim]`.
    fn to_heads(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, n, _] = x.dims();
        x.reshape([b, n, self.num_heads, self.head_dim])
            .swap_dims(1, 2)
    }

    /// Builds the `[num_heads, N, N]` relative-position bias from the table.
    fn relative_position_bias(&self, device: &B::Device) -> Tensor<B, 3> {
        let n = self.window_size * self.window_size;
        let index = relative_position_index(self.window_size);
        let index: Tensor<B, 1, Int> =
            Tensor::from_data(TensorData::new(index, [n * n]), device);
        // Select bias rows per (query,key) pair, then reshape to [N, N, heads].
        let selected = self.relative_position_bias_table.val().select(0, index); // [N*N, heads]
        let heads = self.num_heads;
        selected
            .reshape([n, n, heads])
            .permute([2, 0, 1]) // -> [heads, N, N]
    }

    /// Runs attention over windowed input `[num_windows*B, N, dim]`.
    ///
    /// `attn_mask` is the shifted-window mask; in `unimernet_hf_small_2503` it is
    /// **always `None`** (all blocks are W-MSA, `shift_size == 0`), so the masked
    /// path is a faithful no-op here — see `swin::layer` module docs. The argument
    /// is kept for structural fidelity and future shifted-window support.
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        attn_mask: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        debug_assert!(
            attn_mask.is_none(),
            "shifted-window attention mask is unused by unimernet_hf_small_2503"
        );
        let _ = attn_mask;
        let device = hidden.device();
        let [bw, n, dim] = hidden.dims();

        let q = self.to_heads(self.query.forward(hidden.clone()));
        let k = self.to_heads(self.key.forward(hidden.clone()));
        let v = self.to_heads(self.value.forward(hidden));

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        // [bw, heads, N, N]
        let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(scale);

        // Add relative position bias, broadcast over the batch-window dim.
        let bias = self.relative_position_bias(&device).unsqueeze::<4>(); // [1, heads, N, N]
        let scores = scores + bias;

        let probs = softmax(scores, 3);
        // context = probs @ v, written as `(vᵀ @ probsᵀ)ᵀ`: `v` (from `to_heads`)
        // is a batch-permuted view, and Burn 0.21's wgpu matmul reads a batch-
        // permuted right-hand operand with wrong strides. Keeping it on the left
        // with a last-two-dim transpose on the right avoids the bug (no-op on CPU).
        // See the matching note in mineru-layout's `encoder.rs`.
        let context = v.swap_dims(2, 3).matmul(probs.swap_dims(2, 3)).swap_dims(2, 3); // [bw, heads, N, head_dim]
        let context = context
            .swap_dims(1, 2)
            .reshape([bw, n, dim]);
        self.output.forward(context)
    }
}
