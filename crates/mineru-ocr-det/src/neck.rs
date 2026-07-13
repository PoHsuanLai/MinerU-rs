//! RepLKFPN detection neck.
//!
//! A faithful Burn port of `RepLKFPN` from
//! `pytorchocr/modeling/necks/db_fpn.py` (PP-OCRv6 *small det*). Module nesting
//! mirrors the reference so weights load with no remapping: `neck.insert_conv.<i>.*`
//! (input projections with SE) and `neck.input_conv.<i>.*` (large-kernel depthwise
//! fusion). The neck fuses the four backbone stage maps into a single feature map for
//! the DB head.

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::nn::{PaddingConfig2d, Relu};
use burn::prelude::Backend;
use burn::tensor::Tensor;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};

/// Lightweight SE (`RepLKFPNSqueezeExcitationModule`), hard-sigmoid gating.
#[derive(Module, Debug)]
struct RepLkSqueezeExcite<B: Backend> {
    avg_pool: AdaptiveAvgPool2d,
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    act: Relu,
}

impl<B: Backend> RepLkSqueezeExcite<B> {
    fn new(in_ch: usize, reduction: usize, device: &B::Device) -> Self {
        let mid = in_ch / reduction;
        Self {
            avg_pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            conv1: Conv2dConfig::new([in_ch, mid], [1, 1]).init(device),
            conv2: Conv2dConfig::new([mid, in_ch], [1, 1]).init(device),
            act: Relu::new(),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = x.clone();
        let s = self.avg_pool.forward(x);
        let s = self.conv1.forward(s);
        let s = self.act.forward(s);
        let s = self.conv2.forward(s);
        // clamp(0.2*x + 0.5, 0, 1).
        let s = (s * 0.2 + 0.5).clamp(0.0, 1.0);
        residual * s
    }
}

/// Input projection with SE (`RepLKFPNResidualSqueezeExcitationLayer`).
#[derive(Module, Debug)]
struct InsertConv<B: Backend> {
    in_conv: Conv2d<B>,
    squeeze_excitation_block: RepLkSqueezeExcite<B>,
    shortcut: bool,
}

impl<B: Backend> InsertConv<B> {
    fn new(in_ch: usize, out_ch: usize, reduction: usize, shortcut: bool, device: &B::Device) -> Self {
        Self {
            in_conv: Conv2dConfig::new([in_ch, out_ch], [1, 1])
                .with_bias(false)
                .init(device),
            squeeze_excitation_block: RepLkSqueezeExcite::new(out_ch, reduction, device),
            shortcut,
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self.in_conv.forward(x);
        if self.shortcut {
            h.clone() + self.squeeze_excitation_block.forward(h)
        } else {
            self.squeeze_excitation_block.forward(h)
        }
    }
}

/// Large-kernel depthwise + pointwise fusion (`RepLKFPNDepthwiseSeparableConvLayer`).
#[derive(Module, Debug)]
struct InputConv<B: Backend> {
    depthwise_convolution: Conv2d<B>,
    squeeze_excitation_module: RepLkSqueezeExcite<B>,
    pointwise_convolution: Conv2d<B>,
}

impl<B: Backend> InputConv<B> {
    fn new(in_ch: usize, out_ch: usize, kernel: usize, reduction: usize, device: &B::Device) -> Self {
        Self {
            depthwise_convolution: Conv2dConfig::new([in_ch, out_ch], [kernel, kernel])
                .with_stride([1, 1])
                .with_groups(in_ch)
                .with_padding(PaddingConfig2d::Explicit(kernel / 2, kernel / 2, kernel / 2, kernel / 2))
                .with_bias(true)
                .init(device),
            squeeze_excitation_module: RepLkSqueezeExcite::new(out_ch / 4, reduction, device),
            pointwise_convolution: Conv2dConfig::new([out_ch, out_ch / 4], [1, 1])
                .with_bias(false)
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self.depthwise_convolution.forward(x);
        let h = self.pointwise_convolution.forward(h);
        h.clone() + self.squeeze_excitation_module.forward(h)
    }
}

/// PP-OCRv6 RepLKFPN neck.
#[derive(Module, Debug)]
pub struct RepLkFpn<B: Backend> {
    insert_conv: Vec<InsertConv<B>>,
    input_conv: Vec<InputConv<B>>,
    #[module(skip)]
    out_channels: usize,
}

impl<B: Backend> RepLkFpn<B> {
    /// Builds the neck for four backbone stages of channels `in_channels`, projecting
    /// each to `out_channels` and fusing with `dilated_kernel_size` depthwise convs.
    pub fn new(
        in_channels: &[usize],
        out_channels: usize,
        shortcut: bool,
        dilated_kernel_size: usize,
        reduction: usize,
        device: &B::Device,
    ) -> Self {
        let insert_conv = in_channels
            .iter()
            .map(|&c| InsertConv::new(c, out_channels, reduction, shortcut, device))
            .collect();
        let input_conv = in_channels
            .iter()
            .map(|_| InputConv::new(out_channels, out_channels, dilated_kernel_size, reduction, device))
            .collect();
        Self {
            insert_conv,
            input_conv,
            out_channels,
        }
    }

    /// The neck's concatenated output channels (4 × `out_channels / 4`).
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Fuses the four stage feature maps into one map (top-down + upsample-cat).
    pub fn forward(&self, feats: Vec<Tensor<B, 4>>) -> Tensor<B, 4> {
        // 1x1 projections with SE.
        let mut fused: Vec<Tensor<B, 4>> = self
            .insert_conv
            .iter()
            .zip(feats)
            .map(|(c, f)| c.forward(f))
            .collect();

        // Top-down: fused[i] += upsample(fused[i+1]) for i = 2,1,0.
        for idx in (0..3).rev() {
            let up = upsample2x(fused[idx + 1].clone());
            fused[idx] = fused[idx].clone() + up;
        }

        // Large-kernel fusion.
        let features: Vec<Tensor<B, 4>> = self
            .input_conv
            .iter()
            .zip(fused)
            .map(|(c, f)| c.forward(f))
            .collect();

        // Upsample by [1,2,4,8] and concat in reverse order.
        let scales = [1usize, 2, 4, 8];
        let processed: Vec<Tensor<B, 4>> = features
            .into_iter()
            .zip(scales.iter())
            .map(|(f, &s)| if s == 1 { f } else { upsample_n(f, s) })
            .collect();

        let reversed: Vec<Tensor<B, 4>> = processed.into_iter().rev().collect();
        Tensor::cat(reversed, 1)
    }
}

/// Nearest 2× upsample (`F.interpolate(x, scale_factor=2, mode="nearest")`).
fn upsample2x<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    upsample_n(x, 2)
}

/// Nearest `n`× upsample.
fn upsample_n<B: Backend>(x: Tensor<B, 4>, n: usize) -> Tensor<B, 4> {
    let [_, _, h, w] = x.dims();
    interpolate(
        x,
        [h * n, w * n],
        InterpolateOptions::new(InterpolateMode::Nearest),
    )
}
