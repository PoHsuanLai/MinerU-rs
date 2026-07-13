//! Common neural-network building blocks.
//!
//! Kept deliberately small: only blocks that are genuinely shared across the
//! vision models live here. The flagship is [`ConvBnRelu`], the conv → batch-norm
//! → ReLU triple that CNN backbones repeat dozens of times. Models add their own
//! specific layers rather than growing this module.

use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, PaddingConfig2d, Relu};
use burn::prelude::Backend;
use burn::tensor::Tensor;

/// Configuration for a [`ConvBnRelu`] block.
///
/// Wraps the handful of `Conv2d` knobs a backbone actually varies (channels,
/// kernel, stride, padding, groups) and always pairs the convolution with batch
/// norm, so the bias is disabled automatically (batch norm supplies the shift).
#[derive(Config, Debug)]
pub struct ConvBnReluConfig {
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Square kernel side length.
    #[config(default = 3)]
    pub kernel: usize,
    /// Convolution stride (applied to both spatial dims).
    #[config(default = 1)]
    pub stride: usize,
    /// Symmetric zero-padding (applied to both spatial dims).
    #[config(default = 1)]
    pub padding: usize,
    /// Grouped-convolution group count (1 = a normal convolution).
    #[config(default = 1)]
    pub groups: usize,
}

impl ConvBnReluConfig {
    /// Initialises the block's parameters on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvBnRelu<B> {
        let conv = Conv2dConfig::new(
            [self.in_channels, self.out_channels],
            [self.kernel, self.kernel],
        )
        .with_stride([self.stride, self.stride])
        .with_padding(PaddingConfig2d::Explicit(
            self.padding,
            self.padding,
            self.padding,
            self.padding,
        ))
        .with_groups(self.groups)
        // Batch norm's beta term subsumes the conv bias, so drop it.
        .with_bias(false)
        .init(device);

        let bn = BatchNormConfig::new(self.out_channels).init(device);

        ConvBnRelu {
            conv,
            bn,
            act: Relu::new(),
        }
    }
}

/// A `Conv2d` → `BatchNorm` → `ReLU` block.
///
/// The workhorse unit of CNN backbones (ResNet/MobileNet-style). Construct via
/// [`ConvBnReluConfig::init`].
#[derive(Module, Debug)]
pub struct ConvBnRelu<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
    act: Relu,
}

impl<B: Backend> ConvBnRelu<B> {
    /// Applies the block to an `[N, C_in, H, W]` tensor, producing
    /// `[N, C_out, H', W']`.
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.bn.forward(x);
        self.act.forward(x)
    }
}
