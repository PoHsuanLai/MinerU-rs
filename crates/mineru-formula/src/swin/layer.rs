//! One Swin block: `ConvEnhance` + windowed attention + `ConvEnhance` + FFN.
//!
//! Port of `ConvEnhance`, `UnimerSwinIntermediate`, `UnimerSwinOutput`, and
//! `UnimerSwinLayer` from `unimer_swin/modeling_unimer_swin.py`.
//!
//! # Shift is always zero
//! In `UnimerSwinStage.__init__` every block is built with `shift_size=0`, and
//! `set_shift_and_window_size` only ever *lowers* the shift to zero (never raises
//! it). So this model uses **plain W-MSA everywhere** — there is no cyclic
//! `torch.roll` and `get_attn_mask` always returns `None`. We therefore omit the
//! shifted-window machinery entirely; this is a faithful specialization, not a
//! simplification that changes results. (If a future checkpoint enabled shifts,
//! the roll + mask would need to be added back — flagged in the crate notes.)
//!
//! # Window size vs input resolution
//! `set_shift_and_window_size` also clamps the window to `min(H, W)` when the
//! feature map is smaller than the window. We reproduce that per-forward.

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use mineru_burn_common::nn::{PtLayerNorm, PtLinear};

use crate::config::SwinConfig;
use crate::swin::attention::WindowAttention;

/// Depth-wise conv that injects positional information, added as a residual.
///
/// `x + act(depthwise_conv(x))`, with `x` reshaped from `[B, N, C]` to `[B, C, H, W]`.
#[derive(Module, Debug)]
pub struct ConvEnhance<B: Backend> {
    proj: Conv2d<B>,
}

impl<B: Backend> ConvEnhance<B> {
    /// Builds a depth-wise 3×3 `ConvEnhance` for channel dim `dim`.
    pub fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            proj: Conv2dConfig::new([dim, dim], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_groups(dim)
                .init(device),
        }
    }

    /// `[B, H*W, C]` with spatial size `(H, W)` -> `[B, H*W, C]`.
    pub fn forward(&self, x: Tensor<B, 3>, h: usize, w: usize) -> Tensor<B, 3> {
        let [b, _n, c] = x.dims();
        let feat = x.clone().swap_dims(1, 2).reshape([b, c, h, w]);
        let feat = gelu(self.proj.forward(feat));
        let feat = feat.reshape([b, c, h * w]).swap_dims(1, 2);
        x + feat
    }
}

/// Partitions `[B, H, W, C]` into windows `[num_windows*B, win, win, C]`.
fn window_partition<B: Backend>(x: Tensor<B, 4>, win: usize) -> Tensor<B, 4> {
    let [b, h, w, c] = x.dims();
    x.reshape([b, h / win, win, w / win, win, c])
        .permute([0, 1, 3, 2, 4, 5])
        .reshape([b * (h / win) * (w / win), win, win, c])
}

/// Reverses [`window_partition`] back to `[B, H, W, C]`.
fn window_reverse<B: Backend>(
    windows: Tensor<B, 4>,
    win: usize,
    h: usize,
    w: usize,
) -> Tensor<B, 4> {
    let c = windows.dims()[3];
    let b = windows.dims()[0] / ((h / win) * (w / win));
    windows
        .reshape([b, h / win, w / win, win, win, c])
        .permute([0, 1, 3, 2, 4, 5])
        .reshape([b, h, w, c])
}

/// A single Swin transformer block.
#[derive(Module, Debug)]
pub struct SwinLayer<B: Backend> {
    layernorm_before: PtLayerNorm<B>,
    /// The two `ConvEnhance` blocks, kept as a `Vec` so the checkpoint's
    /// `ce.0.*` / `ce.1.*` `ModuleList` indices line up directly.
    ce: Vec<ConvEnhance<B>>,
    attention: WindowAttention<B>,
    layernorm_after: PtLayerNorm<B>,
    intermediate: PtLinear<B>,
    output: PtLinear<B>,
    window_size: usize,
}

impl<B: Backend> SwinLayer<B> {
    /// Builds a block for a stage of dimension `dim` with `num_heads` heads.
    pub fn new(cfg: &SwinConfig, dim: usize, num_heads: usize, device: &B::Device) -> Self {
        let ln = || PtLayerNorm::init(dim, cfg.layer_norm_eps, device);
        let ffn_hidden = (cfg.mlp_ratio * dim as f64) as usize;
        Self {
            layernorm_before: ln(),
            ce: vec![ConvEnhance::new(dim, device), ConvEnhance::new(dim, device)],
            attention: WindowAttention::new(cfg, dim, num_heads, device),
            layernorm_after: ln(),
            intermediate: PtLinear::init(dim, ffn_hidden, true, device),
            output: PtLinear::init(ffn_hidden, dim, true, device),
            window_size: cfg.window_size,
        }
    }

    /// Runs the block. `hidden` is `[B, H*W, C]`; `(h, w)` is the spatial size.
    ///
    /// Padding to a multiple of the (possibly clamped) window size is applied and
    /// then removed, matching `maybe_pad` in the Python.
    pub fn forward(&self, hidden: Tensor<B, 3>, h: usize, w: usize) -> Tensor<B, 3> {
        let [b, _n, c] = hidden.dims();
        // Clamp window to the feature map (set_shift_and_window_size).
        let win = self.window_size.min(h).min(w);

        // First ConvEnhance, then the attention residual branch.
        let hidden = self.ce[0].forward(hidden, h, w);
        let shortcut = hidden.clone();

        let normed = self.layernorm_before.forward(hidden);
        let normed = normed.reshape([b, h, w, c]);

        // Pad H/W up to a multiple of the window size.
        let pad_b = (win - h % win) % win;
        let pad_r = (win - w % win) % win;
        let (hp, wp) = (h + pad_b, w + pad_r);
        let normed = if pad_b > 0 || pad_r > 0 {
            pad_hw(normed, hp, wp)
        } else {
            normed
        };

        // Windowed attention (no shift, no mask — see module docs).
        let windows = window_partition(normed, win); // [nw*B, win, win, C]
        let nwb = windows.dims()[0];
        let windows = windows.reshape([nwb, win * win, c]);
        let attn = self.attention.forward(windows, None);
        let num_windows_b = attn.dims()[0];
        let attn = attn.reshape([num_windows_b, win, win, c]);
        let attn = window_reverse(attn, win, hp, wp);

        // Remove padding.
        let attn = if pad_b > 0 || pad_r > 0 {
            attn.narrow(1, 0, h).narrow(2, 0, w)
        } else {
            attn
        };
        let attn = attn.reshape([b, h * w, c]);

        let hidden = shortcut + attn;

        // Second ConvEnhance, then the FFN residual branch.
        let hidden = self.ce[1].forward(hidden, h, w);
        let ff = self.layernorm_after.forward(hidden.clone());
        let ff = gelu(self.intermediate.forward(ff));
        let ff = self.output.forward(ff);
        hidden + ff
    }
}

/// Zero-pads a `[B, H, W, C]` tensor (bottom/right) to `(hp, wp)`.
fn pad_hw<B: Backend>(x: Tensor<B, 4>, hp: usize, wp: usize) -> Tensor<B, 4> {
    let [b, h, w, c] = x.dims();
    let device = x.device();
    let mut out: Tensor<B, 4> = Tensor::zeros([b, hp, wp, c], &device);
    out = out.slice_assign([0..b, 0..h, 0..w, 0..c], x);
    out
}
