//! PP-LCNetV4 recognition backbone.
//!
//! A faithful Burn port of `pytorchocr/modeling/backbones/rec_lcnetv4.py`'s
//! `PPLCNetV4` in its recognition configuration (PP-OCRv6 *small rec*). The module
//! nesting mirrors the reference (`backbone.encoder.convolution.*`,
//! `backbone.encoder.blocks.<i>.blocks.<j>.*`) so HF-flat safetensors keys line up.
//!
//! The rec backbone differs from det in three ways:
//! - a wider stem (`[3, 48, 96]`) and rec-specific stage channels,
//! - asymmetric `(2, 1)` strides in the last two stages (downsample height only),
//! - a final height pooling (`avg_pool2d(kernel=[3, 2])`) that collapses the
//!   feature map to a sequence for the CTC head.
//!
//! The depthwise-separable / SE / stem blocks are structurally identical to det, so
//! this is a deliberate, small duplication of `mineru-ocr-det::backbone` (the crates
//! stay decoupled rather than sharing a fragile backbone abstraction).

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Gelu, PaddingConfig2d, Relu};
use burn::prelude::Backend;
use burn::tensor::Tensor;
use burn::tensor::module::avg_pool2d;

/// Recognition stem channels for PP-OCRv6 *small* (`NET_CONFIG_REC["small"]`).
const STEM_CHANNELS: [usize; 3] = [3, 48, 96];

/// A `(kernel, in, out, (stride_h, stride_w), use_se)` block spec.
type BlockCfg = (usize, usize, usize, (usize, usize), bool);

fn rec_block_configs() -> [Vec<BlockCfg>; 4] {
    [
        vec![(3, 96, 96, (1, 1), true)],
        vec![(3, 96, 96, (1, 1), false), (3, 96, 96, (1, 1), false)],
        vec![
            (3, 96, 192, (2, 1), false),
            (3, 192, 192, (1, 1), true),
            (3, 192, 192, (1, 1), false),
            (3, 192, 192, (1, 1), true),
            (3, 192, 192, (1, 1), false),
            (3, 192, 192, (1, 1), true),
            (3, 192, 192, (1, 1), false),
        ],
        vec![
            (3, 192, 384, (2, 1), false),
            (3, 384, 384, (1, 1), true),
            (3, 384, 384, (1, 1), false),
        ],
    ]
}

/// Final recognition feature channels (last stage output).
pub const REC_OUT_CHANNELS: usize = 384;

/// Whether a [`ConvLayer`] applies a trailing ReLU.
#[derive(Debug, Clone, Copy)]
enum Act {
    Relu,
    None,
}

/// Conv → BN → (optional ReLU), matching `PPLCNetV4ConvLayer`.
#[derive(Module, Debug)]
struct ConvLayer<B: Backend> {
    convolution: Conv2d<B>,
    normalization: BatchNorm<B>,
    relu: Option<Relu>,
}

impl<B: Backend> ConvLayer<B> {
    fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: (usize, usize),
        groups: usize,
        act: Act,
        device: &B::Device,
    ) -> Self {
        let pad = (kernel - 1) / 2;
        let convolution = Conv2dConfig::new([in_ch, out_ch], [kernel, kernel])
            .with_stride([stride.0, stride.1])
            .with_groups(groups)
            .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
            .with_bias(false)
            .init(device);
        Self {
            convolution,
            normalization: BatchNormConfig::new(out_ch).init(device),
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
        let s = (s / 6.0 + 0.5).clamp(0.0, 1.0);
        residual * s
    }
}

/// Depthwise-separable block (`PPLCNetV4DepthwiseSeparableConvLayer`).
#[derive(Module, Debug)]
struct DepthwiseSeparable<B: Backend> {
    token_conv_rep: Option<Conv2d<B>>,
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
        let is_unit_stride = stride == (1, 1);
        let has_residual = in_ch == out_ch && is_unit_stride;
        let use_rep_dw = is_unit_stride && in_ch == out_ch;

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
            channel_conv1: ConvLayer::new(in_ch, in_ch * 2, 1, (1, 1), 1, Act::None, device),
            channel_act: Gelu::new(),
            channel_conv2: ConvLayer::new(in_ch * 2, out_ch, 1, (1, 1), 1, Act::None, device),
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

/// One stage (`PPLCNetV4Block`).
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
            stem1: ConvLayer::new(ch[0], ch[1], 3, (2, 2), 1, Act::Relu, device),
            stem2a: ConvLayer::new(ch[1], ch[1] / 2, 2, (1, 1), 1, Act::Relu, device),
            stem2b: ConvLayer::new(ch[1] / 2, ch[1], 2, (1, 1), 1, Act::Relu, device),
            stem3: ConvLayer::new(ch[1] * 2, ch[1], 3, (2, 2), 1, Act::Relu, device),
            stem4: ConvLayer::new(ch[1], ch[2], 1, (1, 1), 1, Act::Relu, device),
            pool: MaxPool2dConfig::new([2, 2]).with_strides([1, 1]).init(),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let emb = self.stem1.forward(x);
        let emb = pad_right_bottom(emb);
        let a = self.stem2a.forward(emb.clone());
        let a = pad_right_bottom(a);
        let a = self.stem2b.forward(a);
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
        let cfgs = rec_block_configs();
        Self {
            convolution: Stem::new(STEM_CHANNELS, device),
            blocks: cfgs.iter().map(|s| Stage::new(s, device)).collect(),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut h = self.convolution.forward(x);
        for stage in &self.blocks {
            h = stage.forward(h);
        }
        h
    }
}

/// PP-LCNetV4 recognition backbone (`PPLCNetV4` with `det=False`).
#[derive(Module, Debug)]
pub struct PpLcNetV4Rec<B: Backend> {
    encoder: Encoder<B>,
}

impl<B: Backend> PpLcNetV4Rec<B> {
    /// Builds the *small rec* backbone on `device`.
    pub fn new(device: &B::Device) -> Self {
        Self {
            encoder: Encoder::new(device),
        }
    }

    /// Runs the backbone and the final height-pooling used for inference.
    ///
    /// Mirrors the eval-mode `PPLCNetV4.forward`: `avg_pool2d(x, [3, 2])`, which
    /// collapses the (now height-3) feature map into a `[N, C, 1, W']` sequence.
    /// Requires feature height ≥ 3; otherwise returns [`None`] so the caller can
    /// raise a shape error rather than panic.
    pub fn forward(&self, x: Tensor<B, 4>) -> Option<Tensor<B, 4>> {
        let feat = self.encoder.forward(x);
        let [_, _, h, _] = feat.dims();
        if h < 3 {
            return None;
        }
        // avg_pool2d kernel [3, 2], stride defaults to kernel size (Paddle/torch
        // F.avg_pool2d default). count_include_pad=true, ceil_mode=false, no pad.
        Some(avg_pool2d(feat, [3, 2], [3, 2], [0, 0], true, false))
    }
}
