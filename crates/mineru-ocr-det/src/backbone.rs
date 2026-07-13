//! PP-LCNetV4 detection backbone.
//!
//! A faithful Burn port of `pytorchocr/modeling/backbones/rec_lcnetv4.py`'s
//! `PPLCNetV4` in its `det=True` configuration (PP-OCRv6 *small det*). The module
//! nesting mirrors the PyTorch attribute tree — `backbone.encoder.convolution.*`
//! and `backbone.encoder.blocks.<i>.blocks.<j>.*` — so the HF-flat safetensors keys
//! line up, with only two small renames (SE conv indices, and the reparameterised
//! depthwise conv) handled by the key remapper in [`crate::model`].
//!
//! The backbone returns the four stage feature maps that the RepLKFPN neck consumes.

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::nn::{Gelu, PaddingConfig2d, Relu};
use burn::prelude::Backend;
use burn::tensor::Tensor;
use mineru_burn_common::nn::FrozenBatchNorm2d;

/// Epsilon for the frozen batch-norm affine (PyTorch `BatchNorm2d` default).
const BN_EPS: f64 = 1e-5;

/// Detection stem channels for PP-OCRv6 *small* (`NET_CONFIG_DET["small"]`).
const STEM_CHANNELS: [usize; 3] = [3, 24, 48];

/// One `(kernel, in, out, stride, use_se)` block spec.
type BlockCfg = (usize, usize, usize, usize, bool);

fn det_block_configs() -> [Vec<BlockCfg>; 4] {
    [
        vec![(3, 48, 48, 1, true), (3, 48, 48, 1, false)],
        vec![
            (3, 48, 96, 2, false),
            (3, 96, 96, 1, true),
            (3, 96, 96, 1, false),
        ],
        vec![
            (3, 96, 192, 2, false),
            (3, 192, 192, 1, true),
            (3, 192, 192, 1, false),
            (3, 192, 192, 1, true),
            (3, 192, 192, 1, false),
        ],
        vec![
            (3, 192, 384, 2, false),
            (3, 384, 384, 1, true),
            (3, 384, 384, 1, false),
        ],
    ]
}

/// The four stage output channels — used by the neck for its projections.
pub const STAGE_OUT_CHANNELS: [usize; 4] = [48, 96, 192, 384];

/// Whether a [`ConvLayer`] applies a trailing ReLU.
#[derive(Debug, Clone, Copy)]
enum Act {
    Relu,
    None,
}

/// Conv → BN → (optional ReLU), matching `PPLCNetV4ConvLayer`.
///
/// Field names (`convolution`, `normalization`) match the reference so weights load
/// 1:1. The `relu` is parameterless, so its presence/absence does not affect keys.
#[derive(Module, Debug)]
struct ConvLayer<B: Backend> {
    convolution: Conv2d<B>,
    normalization: FrozenBatchNorm2d<B>,
    relu: Option<Relu>,
}

impl<B: Backend> ConvLayer<B> {
    fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        groups: usize,
        act: Act,
        device: &B::Device,
    ) -> Self {
        let pad = (kernel - 1) / 2;
        let convolution = Conv2dConfig::new([in_ch, out_ch], [kernel, kernel])
            .with_stride([stride, stride])
            .with_groups(groups)
            .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
            .with_bias(false)
            .init(device);
        Self {
            convolution,
            normalization: FrozenBatchNorm2d::init(out_ch, BN_EPS, device),
            relu: matches!(act, Act::Relu).then(Relu::new),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.convolution.forward(x);
        let x = self.normalization.forward(x);
        match &self.relu {
            Some(r) => r.forward(x),
            None => x,
        }
    }
}

/// Squeeze-and-excitation, matching `PPLCNetV4SqueezeExcitationModule`.
///
/// The reference keeps the reduce/expand convs at `convolutions.0` and
/// `convolutions.2` (indices `1`/`3` are parameterless activations). We store them as
/// named fields and let the model's key remapper rewrite `convolutions.0` → `reduce`
/// and `convolutions.2` → `expand`.
#[derive(Module, Debug)]
struct SqueezeExcite<B: Backend> {
    avg_pool: AdaptiveAvgPool2d,
    reduce: Conv2d<B>,
    expand: Conv2d<B>,
}

impl<B: Backend> SqueezeExcite<B> {
    fn new(channel: usize, reduction: usize, device: &B::Device) -> Self {
        let mid = channel / reduction;
        Self {
            avg_pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            reduce: Conv2dConfig::new([channel, mid], [1, 1]).init(device),
            expand: Conv2dConfig::new([mid, channel], [1, 1]).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = x.clone();
        let s = self.avg_pool.forward(x);
        let s = self.reduce.forward(s);
        let s = burn::tensor::activation::relu(s);
        let s = self.expand.forward(s);
        // nn.Hardsigmoid: clamp(x/6 + 0.5, 0, 1).
        let s = (s / 6.0 + 0.5).clamp(0.0, 1.0);
        residual * s
    }
}

/// Depthwise-separable block, matching `PPLCNetV4DepthwiseSeparableConvLayer`.
#[derive(Module, Debug)]
struct DepthwiseSeparable<B: Backend> {
    /// Reparameterised depthwise conv (bias, no BN) used when `stride==1 && in==out`.
    /// Remapped from `token_conv.*` when present as a plain conv in the checkpoint.
    token_conv_rep: Option<Conv2d<B>>,
    /// Conv-BN depthwise path used otherwise (`token_conv.convolution/normalization`).
    token_conv: Option<ConvLayer<B>>,
    token_squeeze_excitation: Option<SqueezeExcite<B>>,
    channel_conv1: ConvLayer<B>,
    channel_act: Gelu,
    channel_conv2: ConvLayer<B>,
    has_residual: bool,
}

impl<B: Backend> DepthwiseSeparable<B> {
    fn new(cfg: BlockCfg, device: &B::Device) -> Self {
        let (kernel, in_ch, out_ch, stride, use_se) = cfg;
        let has_residual = in_ch == out_ch && stride == 1;
        let use_rep_dw = stride == 1 && in_ch == out_ch;

        let (token_conv_rep, token_conv) = if use_rep_dw {
            let conv = Conv2dConfig::new([in_ch, out_ch], [kernel, kernel])
                .with_stride([1, 1])
                .with_groups(in_ch)
                .with_padding(PaddingConfig2d::Explicit(kernel / 2, kernel / 2, kernel / 2, kernel / 2))
                .with_bias(true)
                .init(device);
            (Some(conv), None)
        } else {
            (
                None,
                Some(ConvLayer::new(
                    in_ch, in_ch, kernel, stride, in_ch, Act::None, device,
                )),
            )
        };

        Self {
            token_conv_rep,
            token_conv,
            token_squeeze_excitation: use_se.then(|| SqueezeExcite::new(in_ch, 4, device)),
            channel_conv1: ConvLayer::new(in_ch, in_ch * 2, 1, 1, 1, Act::None, device),
            channel_act: Gelu::new(),
            channel_conv2: ConvLayer::new(in_ch * 2, out_ch, 1, 1, 1, Act::None, device),
            has_residual,
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut h = match (&self.token_conv_rep, &self.token_conv) {
            (Some(c), _) => c.forward(x),
            (_, Some(c)) => c.forward(x),
            _ => x,
        };
        if let Some(se) = &self.token_squeeze_excitation {
            h = se.forward(h);
        }
        let residual = h.clone();
        let h = self.channel_conv1.forward(h);
        let h = self.channel_act.forward(h);
        let h = self.channel_conv2.forward(h);
        if self.has_residual {
            residual + h
        } else {
            h
        }
    }
}

/// One stage (`PPLCNetV4Block`): a list of depthwise-separable blocks.
#[derive(Module, Debug)]
struct Stage<B: Backend> {
    blocks: Vec<DepthwiseSeparable<B>>,
}

impl<B: Backend> Stage<B> {
    fn new(cfgs: &[BlockCfg], device: &B::Device) -> Self {
        Self {
            blocks: cfgs.iter().map(|c| DepthwiseSeparable::new(*c, device)).collect(),
        }
    }

    fn forward(&self, mut x: Tensor<B, 4>) -> Tensor<B, 4> {
        for b in &self.blocks {
            x = b.forward(x);
        }
        x
    }
}

/// The branch stem (`PPLCNetV4LargeStem`).
#[derive(Module, Debug)]
struct Stem<B: Backend> {
    stem1: ConvLayer<B>,
    stem2a: ConvLayer<B>,
    stem2b: ConvLayer<B>,
    stem3: ConvLayer<B>,
    stem4: ConvLayer<B>,
    pool: MaxPool2d,
}

impl<B: Backend> Stem<B> {
    fn new(ch: [usize; 3], device: &B::Device) -> Self {
        Self {
            stem1: ConvLayer::new(ch[0], ch[1], 3, 2, 1, Act::Relu, device),
            stem2a: ConvLayer::new(ch[1], ch[1] / 2, 2, 1, 1, Act::Relu, device),
            stem2b: ConvLayer::new(ch[1] / 2, ch[1], 2, 1, 1, Act::Relu, device),
            stem3: ConvLayer::new(ch[1] * 2, ch[1], 3, 2, 1, Act::Relu, device),
            stem4: ConvLayer::new(ch[1], ch[2], 1, 1, 1, Act::Relu, device),
            pool: MaxPool2dConfig::new([2, 2]).with_strides([1, 1]).init(),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let emb = self.stem1.forward(x);
        let emb = pad_right_bottom(emb);
        let a = self.stem2a.forward(emb.clone());
        let a = pad_right_bottom(a);
        let a = self.stem2b.forward(a);
        // MaxPool2d kernel 2 stride 1 ceil_mode=True == pad right/bottom then floor-pool.
        let pooled = self.pool.forward(pad_right_bottom(emb));
        let emb = Tensor::cat(vec![pooled, a], 1);
        let emb = self.stem3.forward(emb);
        self.stem4.forward(emb)
    }
}

/// Right/bottom zero-pad by 1 pixel (`F.pad(x, (0,1,0,1))`).
fn pad_right_bottom<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [n, c, h, w] = x.dims();
    let device = x.device();
    let right = Tensor::zeros([n, c, h, 1], &device);
    let x = Tensor::cat(vec![x, right], 3);
    let bottom = Tensor::zeros([n, c, 1, w + 1], &device);
    Tensor::cat(vec![x, bottom], 2)
}

/// The `PPLCNetV4Encoder`: stem plus four stages.
#[derive(Module, Debug)]
struct Encoder<B: Backend> {
    convolution: Stem<B>,
    blocks: Vec<Stage<B>>,
}

impl<B: Backend> Encoder<B> {
    fn new(device: &B::Device) -> Self {
        let cfgs = det_block_configs();
        Self {
            convolution: Stem::new(STEM_CHANNELS, device),
            blocks: cfgs.iter().map(|s| Stage::new(s, device)).collect(),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        let mut h = self.convolution.forward(x);
        let mut out = Vec::with_capacity(4);
        for stage in &self.blocks {
            h = stage.forward(h);
            out.push(h.clone());
        }
        out
    }
}

/// PP-LCNetV4 detection backbone (`PPLCNetV4` with `det=True`).
#[derive(Module, Debug)]
pub struct PpLcNetV4Det<B: Backend> {
    encoder: Encoder<B>,
}

impl<B: Backend> PpLcNetV4Det<B> {
    /// Builds the *small det* backbone on `device`.
    pub fn new(device: &B::Device) -> Self {
        Self {
            encoder: Encoder::new(device),
        }
    }

    /// Runs the stem and four stages, returning the four stage feature maps.
    pub fn forward(&self, x: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        self.encoder.forward(x)
    }
}
