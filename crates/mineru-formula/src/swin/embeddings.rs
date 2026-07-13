//! Swin patch embedding: the overlapping-conv **stem** + LayerNorm.
//!
//! Port of `StemLayer`, `UnimerSwinPatchEmbeddings`, and the patch-embedding part
//! of `UnimerSwinEmbeddings` from `unimer_swin/modeling_unimer_swin.py`.
//!
//! Unlike vanilla Swin (a single non-overlapping `patch_size × patch_size` conv),
//! UniMerNet uses a two-conv stem, each stride-2 (total stride 4 == `patch_size`):
//!
//! ```text
//! conv1: (3 -> embed_dim/2), 3x3, stride 2, pad 1
//! norm1: BatchNorm2d(embed_dim/2)
//! act:   GELU
//! conv2: (embed_dim/2 -> embed_dim), 3x3, stride 2, pad 1
//! ```
//!
//! The result is flattened to `[B, H*W, embed_dim]` and LayerNorm'd. Absolute /
//! 2-D position embeddings are **disabled** in `unimernet_hf_small_2503`
//! (`use_absolute_embeddings = use_2d_embeddings = false`), so none are added —
//! this is asserted by the checkpoint carrying no such tensors.

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use mineru_burn_common::nn::{FrozenBatchNorm2d, PtLayerNorm};

use crate::config::SwinConfig;

/// The stem's `norm1`, a `nn.Sequential(nn.BatchNorm2d(dim))`.
///
/// The extra wrapper exists only so the checkpoint's `norm1.0.*` `Sequential`
/// index lines up with a named Burn field (`bn`); the remap rewrites `norm1.0.` →
/// `norm1.bn.`. UniMerNet runs the Swin encoder in eval mode, so the BatchNorm is
/// a frozen per-channel affine over the running stats.
#[derive(Module, Debug)]
pub struct Norm1Seq<B: Backend> {
    bn: FrozenBatchNorm2d<B>,
}

/// The overlapping-conv stem that replaces Swin's single patch conv.
#[derive(Module, Debug)]
pub struct StemLayer<B: Backend> {
    conv1: Conv2d<B>,
    norm1: Norm1Seq<B>,
    conv2: Conv2d<B>,
}

impl<B: Backend> StemLayer<B> {
    /// Builds the stem for `in_chans -> out_chans` (out == `embed_dim`).
    pub fn new(in_chans: usize, out_chans: usize, cfg: &SwinConfig, device: &B::Device) -> Self {
        let mid = out_chans / 2;
        let conv = |ci: usize, co: usize| {
            Conv2dConfig::new([ci, co], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device)
        };
        Self {
            conv1: conv(in_chans, mid),
            norm1: Norm1Seq {
                bn: FrozenBatchNorm2d::init(mid, cfg.layer_norm_eps, device),
            },
            conv2: conv(mid, out_chans),
        }
    }

    /// `[B, C, H, W] -> [B, embed_dim, H/4, W/4]`.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(x);
        let x = self.norm1.bn.forward(x);
        let x = gelu(x);
        self.conv2.forward(x)
    }
}

/// Patch embedding: stem + flatten + LayerNorm.
#[derive(Module, Debug)]
pub struct PatchEmbeddings<B: Backend> {
    projection: StemLayer<B>,
    norm: PtLayerNorm<B>,
}

impl<B: Backend> PatchEmbeddings<B> {
    /// Builds the patch embedding for the given config.
    pub fn new(cfg: &SwinConfig, device: &B::Device) -> Self {
        Self {
            projection: StemLayer::new(cfg.num_channels, cfg.embed_dim, cfg, device),
            norm: PtLayerNorm::init(cfg.embed_dim, cfg.layer_norm_eps, device),
        }
    }

    /// Embeds pixel values.
    ///
    /// Input `[B, C, H, W]`. Returns `(embeddings [B, H'*W', embed_dim], (H', W'))`
    /// where `H' = H/4`, `W' = W/4`.
    pub fn forward(&self, pixel_values: Tensor<B, 4>) -> (Tensor<B, 3>, (usize, usize)) {
        let x = self.projection.forward(pixel_values); // [B, dim, H', W']
        let [b, dim, h, w] = x.dims();
        // flatten(2).transpose(1,2): [B, dim, H'*W'] -> [B, H'*W', dim]
        let x = x.reshape([b, dim, h * w]).swap_dims(1, 2);
        let x = self.norm.forward(x);
        (x, (h, w))
    }
}
