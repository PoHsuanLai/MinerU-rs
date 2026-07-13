//! PyTorch-layout neural-network primitives.
//!
//! # Why not use Burn's built-in `Linear` / `BatchNorm`?
//!
//! The shared weight loader in [`mineru_burn_common`] drives `burn-store`'s
//! `SafetensorsStore` with the default `IdentityAdapter`. That adapter does **not**
//! transpose linear weights (`[out, in]` in PyTorch vs `[in, out]` in Burn) and
//! does **not** rename normalization params (`weight`/`bias` in PyTorch vs
//! `gamma`/`beta` in Burn). To load the PP-DocLayoutV2 safetensors byte-for-byte
//! under strict "all keys consumed" coverage, this crate stores every parameter in
//! the checkpoint's own layout and naming, and does the transpose/affine math at
//! forward time. The primitives below implement exactly that.
//!
//! Burn's [`Conv2d`](burn::nn::conv::Conv2d) and [`Embedding`](burn::nn::Embedding)
//! *do* already store weights in PyTorch layout (`[out, in/groups, kh, kw]` and
//! `[n, d]`) under the field name `weight`, so those are used directly elsewhere.

use burn::module::{Module, Param};
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};

/// A `nn.Linear` with the checkpoint's `[out_features, in_features]` weight layout.
///
/// Forward computes `x @ weightᵀ (+ bias)`, matching PyTorch's `F.linear`.
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
        let bias =
            bias.then(|| Param::from_tensor(Tensor::<B, 1>::zeros([out_features], device)));
        Self { weight, bias }
    }

    /// Applies the linear transform to a rank-`D` tensor over its last dim.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        // weightᵀ is [in, out]; `linear` in burn multiplies `input @ weight`.
        let weight_t = self.weight.val().transpose();
        let out = burn::tensor::module::linear(
            input,
            weight_t,
            self.bias.as_ref().map(|b| b.val()),
        );
        out
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
        let scale = self
            .weight
            .val()
            .div(self.running_var.val().add_scalar(self.epsilon).sqrt())
            .reshape(shape);
        let shift = self
            .bias
            .val()
            .sub(self.running_mean.val().mul(self.weight.val()).div(
                self.running_var.val().add_scalar(self.epsilon).sqrt(),
            ))
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
/// A small helper used by the reading-order head to lift computed constants (e.g.
/// `inv_freq`) into tensors without repeating the `TensorData` boilerplate.
pub fn tensor_from_vec<B: Backend, const D: usize>(
    values: Vec<f32>,
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D> {
    Tensor::<B, 1>::from_data(TensorData::new(values, [shape.iter().product::<usize>()]), device)
        .reshape(shape)
}
