//! Differentiable Binarization head (PP-OCRv6 mode).
//!
//! A faithful Burn port of `DBHead` with `mode="ppocrv6"` from
//! `pytorchocr/modeling/heads/det_db_head.py`. Three upsampling stages
//! (conv-down → transpose-conv-up → transpose-conv-final) produce a single-channel
//! probability map, sigmoid-activated. Field names (`conv_down`, `conv_up`,
//! `conv_final`) match the reference so weights load 1:1.

use burn::module::Module;
use burn::nn::conv::{
    Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig,
};
use burn::nn::{BatchNorm, BatchNormConfig, PaddingConfig2d, Relu};
use burn::prelude::Backend;
use burn::tensor::Tensor;
use burn::tensor::activation::sigmoid;

/// Conv-BN-ReLU down-projection (`PPOCRV6DBConvBatchnormLayer`, non-transpose).
#[derive(Module, Debug)]
struct ConvDown<B: Backend> {
    convolution: Conv2d<B>,
    norm: BatchNorm<B>,
    act: Relu,
}

impl<B: Backend> ConvDown<B> {
    fn new(in_ch: usize, out_ch: usize, kernel: usize, device: &B::Device) -> Self {
        let pad = kernel / 2;
        Self {
            convolution: Conv2dConfig::new([in_ch, out_ch], [kernel, kernel])
                .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
                .with_bias(false)
                .init(device),
            norm: BatchNormConfig::new(out_ch).init(device),
            act: Relu::new(),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.convolution.forward(x);
        let x = self.norm.forward(x);
        self.act.forward(x)
    }
}

/// Transpose-conv-BN-ReLU up-projection (`PPOCRV6DBConvBatchnormLayer`, transpose).
#[derive(Module, Debug)]
struct ConvUp<B: Backend> {
    convolution: ConvTranspose2d<B>,
    norm: BatchNorm<B>,
    act: Relu,
}

impl<B: Backend> ConvUp<B> {
    fn new(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, device: &B::Device) -> Self {
        Self {
            convolution: ConvTranspose2dConfig::new([in_ch, out_ch], [kernel, kernel])
                .with_stride([stride, stride])
                .init(device),
            norm: BatchNormConfig::new(out_ch).init(device),
            act: Relu::new(),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.convolution.forward(x);
        let x = self.norm.forward(x);
        self.act.forward(x)
    }
}

/// DB head, PP-OCRv6 variant. Produces `{maps}` = sigmoid(shrink map).
#[derive(Module, Debug)]
pub struct DbHead<B: Backend> {
    conv_down: ConvDown<B>,
    conv_up: ConvUp<B>,
    conv_final: ConvTranspose2d<B>,
    #[module(skip)]
    fix_nan: bool,
}

impl<B: Backend> DbHead<B> {
    /// Builds the head. `in_channels` is the neck output; `kernel_list` is the
    /// `[down, up, final]` kernel sizes (`[3, 2, 2]` for PP-OCRv6).
    pub fn new(
        in_channels: usize,
        kernel_list: [usize; 3],
        fix_nan: bool,
        device: &B::Device,
    ) -> Self {
        let mid = in_channels / 4;
        Self {
            conv_down: ConvDown::new(in_channels, mid, kernel_list[0], device),
            conv_up: ConvUp::new(mid, mid, kernel_list[1], 2, device),
            conv_final: ConvTranspose2dConfig::new([mid, 1], [kernel_list[2], kernel_list[2]])
                .with_stride([2, 2])
                .init(device),
            fix_nan,
        }
    }

    /// Forward pass → `[N, 1, H, W]` probability map in `[0, 1]`.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv_down.forward(x);
        let x = self.conv_up.forward(x);
        let x = self.conv_final.forward(x);
        let x = sigmoid(x);
        if self.fix_nan {
            // nan_to_num: replace NaN with 0. Burn has no direct helper, so mask.
            let is_nan = x.clone().is_nan();
            x.mask_fill(is_nan, 0.0)
        } else {
            x
        }
    }
}
