//! Common neural-network building blocks.
//!
//! Kept deliberately small: only blocks that are genuinely shared across the
//! vision models live here. Two families live side by side:
//!
//! - [`ConvBnRelu`] — the conv → batch-norm → ReLU triple that CNN backbones
//!   repeat dozens of times, built from Burn's own [`Conv2d`]/[`BatchNorm`]. Use
//!   it when the block is *trained in Burn* or its weights are remapped by hand.
//! - **PyTorch-layout primitives** ([`PtLinear`], [`PtLayerNorm`],
//!   [`FrozenBatchNorm2d`]) — drop-in replacements for `nn.Linear` / `nn.LayerNorm`
//!   / a frozen `nn.BatchNorm2d` that store parameters in the *checkpoint's* own
//!   layout and naming so a `.safetensors` / `.pth` state-dict loads
//!   byte-for-byte under the strict [`Coverage::Strict`](crate::weights::Coverage)
//!   check. See the note below for *why* Burn's built-ins can't do this.
//!
//! # Why not use Burn's built-in `Linear` / `LayerNorm` / `BatchNorm` for loading?
//!
//! The shared weight loader in [`crate::weights`] drives `burn-store`'s
//! `SafetensorsStore` / `PytorchStore` with the default `IdentityAdapter`. That
//! adapter does **not** transpose linear weights (`[out, in]` in PyTorch vs
//! `[in, out]` in Burn) and does **not** rename normalization params
//! (`weight`/`bias` in PyTorch vs `gamma`/`beta` in Burn). So a checkpoint tensor
//! either lands in the wrong field (silent garbage) or is left unconsumed (a
//! strict-coverage failure). The `Pt*` primitives store every parameter in the
//! checkpoint's layout/naming and do the transpose/affine math at forward time,
//! which is what makes strict "every key consumed" loading possible.
//!
//! Burn's [`Conv2d`](burn::nn::conv::Conv2d) and [`Embedding`](burn::nn::Embedding)
//! *do* already store weights in PyTorch layout (`[out, in/groups, kh, kw]` and
//! `[n, d]`) under the field name `weight`, so those are used directly.

use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, PaddingConfig2d, Relu};
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};

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

/// A `nn.Linear` with the checkpoint's `[out_features, in_features]` weight layout.
///
/// Forward computes `x @ weightᵀ (+ bias)`, matching PyTorch's `F.linear`. Storing
/// the weight untransposed is what lets a state-dict load byte-for-byte; see the
/// module docs.
#[derive(Module, Debug)]
pub struct PtLinear<B: Backend> {
    /// Weight of shape `[out_features, in_features]` (PyTorch layout).
    pub weight: Param<Tensor<B, 2>>,
    /// Optional bias of shape `[out_features]`.
    pub bias: Option<Param<Tensor<B, 1>>>,
}

impl<B: Backend> PtLinear<B> {
    /// Initialises a zeroed linear layer; weights are overwritten by loading.
    pub fn init(in_features: usize, out_features: usize, bias: bool, device: &B::Device) -> Self {
        let weight = Param::from_tensor(Tensor::zeros([out_features, in_features], device));
        let bias = bias.then(|| Param::from_tensor(Tensor::<B, 1>::zeros([out_features], device)));
        Self { weight, bias }
    }

    /// Applies the linear transform to a rank-`D` tensor over its last dim.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        // weightᵀ is [in, out]; burn's `linear` computes `input @ weight`.
        let weight_t = self.weight.val().transpose();
        burn::tensor::module::linear(input, weight_t, self.bias.as_ref().map(|b| b.val()))
    }
}

/// A `nn.LayerNorm` storing `weight`/`bias` (not Burn's `gamma`/`beta`).
///
/// Normalises over the last dimension with the configured epsilon, then applies
/// the affine `weight * x_hat + bias`.
#[derive(Module, Debug)]
pub struct PtLayerNorm<B: Backend> {
    /// Scale of shape `[normalized_shape]`.
    pub weight: Param<Tensor<B, 1>>,
    /// Shift of shape `[normalized_shape]`.
    pub bias: Param<Tensor<B, 1>>,
    /// Numerical-stability epsilon.
    epsilon: f64,
}

impl<B: Backend> PtLayerNorm<B> {
    /// Initialises a LayerNorm over `size` features.
    pub fn init(size: usize, epsilon: f64, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([size], device)),
            bias: Param::from_tensor(Tensor::zeros([size], device)),
            epsilon,
        }
    }

    /// Normalises `input` over its last dimension.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let last = D - 1;
        let mean = input.clone().mean_dim(last);
        let centered = input.sub(mean);
        // Biased variance (matches PyTorch LayerNorm, which divides by N).
        let var = centered.clone().powf_scalar(2.0).mean_dim(last);
        let normed = centered.div(var.add_scalar(self.epsilon).sqrt());

        let shape = weight_broadcast_shape::<D>(self.weight.dims()[0]);
        let weight = self.weight.val().reshape(shape);
        let bias = self.bias.val().reshape(shape);
        normed.mul(weight).add(bias)
    }
}

/// A frozen `BatchNorm2d`, evaluated as a per-channel affine (RT-DETR folds the
/// backbone BN into a frozen affine with epsilon `1e-5`).
///
/// Stores the four checkpoint buffers under their PyTorch names and computes
/// `y = (x - running_mean) / sqrt(running_var + eps) * weight + bias`, broadcast
/// over an `[N, C, H, W]` tensor.
#[derive(Module, Debug)]
pub struct FrozenBatchNorm2d<B: Backend> {
    /// Affine scale, shape `[C]`.
    pub weight: Param<Tensor<B, 1>>,
    /// Affine shift, shape `[C]`.
    pub bias: Param<Tensor<B, 1>>,
    /// Running mean, shape `[C]`.
    pub running_mean: Param<Tensor<B, 1>>,
    /// Running variance, shape `[C]`.
    pub running_var: Param<Tensor<B, 1>>,
    /// Numerical-stability epsilon.
    epsilon: f64,
}

impl<B: Backend> FrozenBatchNorm2d<B> {
    /// Initialises identity batch-norm buffers for `channels` channels.
    pub fn init(channels: usize, epsilon: f64, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([channels], device)),
            bias: Param::from_tensor(Tensor::zeros([channels], device)),
            running_mean: Param::from_tensor(Tensor::zeros([channels], device)),
            running_var: Param::from_tensor(Tensor::ones([channels], device)),
            epsilon,
        }
    }

    /// Applies the frozen affine to an `[N, C, H, W]` tensor.
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let c = self.weight.dims()[0];
        let shape = [1, c, 1, 1];
        let denom = self.running_var.val().add_scalar(self.epsilon).sqrt();
        let scale = self.weight.val().div(denom.clone()).reshape(shape);
        let shift = self
            .bias
            .val()
            .sub(self.running_mean.val().mul(self.weight.val()).div(denom))
            .reshape(shape);
        input.mul(scale).add(shift)
    }
}

/// Builds a `[1, 1, …, C]` broadcast shape for applying a per-feature vector to a
/// rank-`D` tensor over its last dim.
fn weight_broadcast_shape<const D: usize>(features: usize) -> [usize; D] {
    let mut shape = [1usize; D];
    shape[D - 1] = features;
    shape
}

/// Creates a float tensor from a flat `Vec<f32>` and an explicit shape.
///
/// A small helper for lifting computed constants (e.g. RoPE `inv_freq` tables)
/// into tensors without repeating the `TensorData` boilerplate.
pub fn tensor_from_vec<B: Backend, const D: usize>(
    values: Vec<f32>,
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D> {
    Tensor::<B, 1>::from_data(
        TensorData::new(values, [shape.iter().product::<usize>()]),
        device,
    )
    .reshape(shape)
}
