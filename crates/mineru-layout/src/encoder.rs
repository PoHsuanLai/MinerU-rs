//! RT-DETR hybrid encoder: AIFI transformer on the top level + a CCFM FPN-PAN.
//!
//! Port of `RTDetrHybridEncoder` (+ the `encoder_input_proj` that lives on
//! `RTDetrModel`). Input: three backbone maps at strides `[8,16,32]`, channels
//! `[512,1024,2048]`. Output: three fused maps, all `d_model=256`, same strides.
//!
//! Encoder conv blocks use `conv` / `norm` naming (the RT-DETR convention).

use burn::module::Module;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::Backend;
use burn::tensor::activation::softmax;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn::tensor::{Tensor, TensorData};

use crate::config::DET;
use mineru_burn_common::nn::{FrozenBatchNorm2d, PtLayerNorm, PtLinear};

/// SiLU activation.
fn silu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    burn::tensor::activation::silu(x)
}

/// `Conv2d(k=1, bias=false)` → BatchNorm, the `encoder_input_proj` /
/// `decoder_input_proj` stem. Stored as a two-element `Vec` so the checkpoint's
/// `input_proj.N.0` (conv) / `input_proj.N.1` (bn) `Sequential` keys line up.
#[derive(Module, Debug)]
pub struct ProjConvBn<B: Backend> {
    conv: Conv2d<B>,
    bn: FrozenBatchNorm2d<B>,
}

impl<B: Backend> ProjConvBn<B> {
    /// Builds the `1×1 conv + BatchNorm` projection.
    pub fn init(in_c: usize, out_c: usize, device: &B::Device) -> Self {
        let conv = Conv2dConfig::new([in_c, out_c], [1, 1])
            .with_bias(false)
            .init(device);
        let bn = FrozenBatchNorm2d::init(out_c, DET.batch_norm_eps, device);
        Self { conv, bn }
    }

    /// Applies the projection.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.bn.forward(self.conv.forward(x))
    }
}

/// `RTDetrConvNormLayer`: conv → batch-norm → optional activation (silu here).
#[derive(Module, Debug)]
pub struct ConvNormLayer<B: Backend> {
    conv: Conv2d<B>,
    norm: FrozenBatchNorm2d<B>,
    #[module(skip)]
    act: bool,
}

impl<B: Backend> ConvNormLayer<B> {
    fn init(in_c: usize, out_c: usize, kernel: usize, stride: usize, act: bool, device: &B::Device) -> Self {
        let padding = (kernel - 1) / 2;
        let conv = Conv2dConfig::new([in_c, out_c], [kernel, kernel])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(padding, padding, padding, padding))
            .with_bias(false)
            .init(device);
        let norm = FrozenBatchNorm2d::init(out_c, DET.batch_norm_eps, device);
        Self { conv, norm, act }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.norm.forward(self.conv.forward(x));
        if self.act { silu(x) } else { x }
    }
}

/// `RTDetrRepVggBlock`: `silu(conv1(x) + conv2(x))` (3×3 and 1×1 branches).
#[derive(Module, Debug)]
pub struct RepVggBlock<B: Backend> {
    conv1: ConvNormLayer<B>,
    conv2: ConvNormLayer<B>,
}

impl<B: Backend> RepVggBlock<B> {
    fn init(channels: usize, device: &B::Device) -> Self {
        Self {
            conv1: ConvNormLayer::init(channels, channels, 3, 1, false, device),
            conv2: ConvNormLayer::init(channels, channels, 1, 1, false, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        silu(self.conv1.forward(x.clone()).add(self.conv2.forward(x)))
    }
}

/// `RTDetrCSPRepLayer`: two 1×1 branches, one through N RepVgg bottlenecks, summed.
///
/// `hidden_expansion = 1.0`, so `hidden_channels == out_channels` and the trailing
/// `conv3` is the identity (omitted).
#[derive(Module, Debug)]
pub struct CspRepLayer<B: Backend> {
    conv1: ConvNormLayer<B>,
    conv2: ConvNormLayer<B>,
    bottlenecks: Vec<RepVggBlock<B>>,
}

impl<B: Backend> CspRepLayer<B> {
    fn init(in_c: usize, out_c: usize, num_blocks: usize, device: &B::Device) -> Self {
        let conv1 = ConvNormLayer::init(in_c, out_c, 1, 1, true, device);
        let conv2 = ConvNormLayer::init(in_c, out_c, 1, 1, true, device);
        let bottlenecks = (0..num_blocks).map(|_| RepVggBlock::init(out_c, device)).collect();
        Self {
            conv1,
            conv2,
            bottlenecks,
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut y = self.conv1.forward(x.clone());
        for b in &self.bottlenecks {
            y = b.forward(y);
        }
        let z = self.conv2.forward(x);
        y.add(z)
    }
}

/// The AIFI transformer encoder layer (`RTDetrEncoderLayer`): post-norm
/// self-attention with position embeddings added to Q/K, then a GELU FFN.
#[derive(Module, Debug)]
pub struct EncoderLayer<B: Backend> {
    q_proj: PtLinear<B>,
    k_proj: PtLinear<B>,
    v_proj: PtLinear<B>,
    o_proj: PtLinear<B>,
    self_attn_layer_norm: PtLayerNorm<B>,
    fc1: PtLinear<B>,
    fc2: PtLinear<B>,
    final_layer_norm: PtLayerNorm<B>,
    #[module(skip)]
    num_heads: usize,
}

impl<B: Backend> EncoderLayer<B> {
    fn init(device: &B::Device) -> Self {
        let d = DET.d_model;
        let ff = DET.encoder_ffn_dim;
        Self {
            q_proj: PtLinear::init(d, d, true, device),
            k_proj: PtLinear::init(d, d, true, device),
            v_proj: PtLinear::init(d, d, true, device),
            o_proj: PtLinear::init(d, d, true, device),
            self_attn_layer_norm: PtLayerNorm::init(d, DET.layer_norm_eps, device),
            fc1: PtLinear::init(d, ff, true, device),
            fc2: PtLinear::init(ff, d, true, device),
            final_layer_norm: PtLayerNorm::init(d, DET.layer_norm_eps, device),
            num_heads: DET.encoder_attention_heads,
        }
    }

    /// `hidden`: `[B, L, d_model]`; `pos`: `[1, L, d_model]` sine embedding.
    fn forward(&self, hidden: Tensor<B, 3>, pos: Tensor<B, 3>) -> Tensor<B, 3> {
        let [bsz, seq, d] = hidden.dims();
        let head_dim = d / self.num_heads;
        let scale = (head_dim as f64).powf(-0.5);

        let q_in = hidden.clone().add(pos.clone());
        let k_in = hidden.clone().add(pos);
        let q = split_heads(self.q_proj.forward(q_in), bsz, seq, self.num_heads, head_dim);
        let k = split_heads(self.k_proj.forward(k_in), bsz, seq, self.num_heads, head_dim);
        let v = split_heads(self.v_proj.forward(hidden.clone()), bsz, seq, self.num_heads, head_dim);

        // [B, H, L, L]
        let scores = q.mul_scalar(scale).matmul(k.swap_dims(2, 3));
        let probs = softmax(scores, 3);
        let ctx = probs.matmul(v); // [B, H, L, hd]
        let ctx = ctx
            .swap_dims(1, 2)
            .reshape([bsz, seq, d]);
        let attn_out = self.o_proj.forward(ctx);

        let hidden = self.self_attn_layer_norm.forward(hidden.add(attn_out));

        let ff = self.fc2.forward(gelu(self.fc1.forward(hidden.clone())));
        self.final_layer_norm.forward(hidden.add(ff))
    }
}

/// GELU (erf-based, matching PyTorch's default `nn.GELU`).
fn gelu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    burn::tensor::activation::gelu(x)
}

/// Reshapes `[B, L, d]` into `[B, H, L, hd]` for multi-head attention.
fn split_heads<B: Backend>(x: Tensor<B, 3>, b: usize, l: usize, h: usize, hd: usize) -> Tensor<B, 4> {
    x.reshape([b, l, h, hd]).swap_dims(1, 2)
}

/// The AIFI module: one (or more) [`EncoderLayer`] over the highest-level map.
#[derive(Module, Debug)]
pub struct Aifi<B: Backend> {
    layers: Vec<EncoderLayer<B>>,
}

/// The full hybrid encoder.
#[derive(Module, Debug)]
pub struct HybridEncoder<B: Backend> {
    aifi: Aifi<B>,
    lateral_convs: Vec<ConvNormLayer<B>>,
    fpn_blocks: Vec<CspRepLayer<B>>,
    downsample_convs: Vec<ConvNormLayer<B>>,
    pan_blocks: Vec<CspRepLayer<B>>,
}

impl<B: Backend> HybridEncoder<B> {
    /// Initialises the hybrid encoder.
    pub fn init(device: &B::Device) -> Self {
        let d = DET.d_model;
        let aifi = Aifi {
            layers: (0..DET.encoder_layers).map(|_| EncoderLayer::init(device)).collect(),
        };
        // num_fpn_stages = num_pan_stages = 2.
        let lateral_convs = (0..2).map(|_| ConvNormLayer::init(d, d, 1, 1, true, device)).collect();
        let fpn_blocks = (0..2).map(|_| CspRepLayer::init(2 * d, d, 3, device)).collect();
        let downsample_convs = (0..2).map(|_| ConvNormLayer::init(d, d, 3, 2, true, device)).collect();
        let pan_blocks = (0..2).map(|_| CspRepLayer::init(2 * d, d, 3, device)).collect();
        Self {
            aifi,
            lateral_convs,
            fpn_blocks,
            downsample_convs,
            pan_blocks,
        }
    }

    /// Runs the hybrid encoder. `feats` are the three projected maps (256-ch) in
    /// low→high stride order. Returns the three PAN maps in the same order.
    pub fn forward(&self, feats: [Tensor<B, 4>; 3]) -> [Tensor<B, 4>; 3] {
        let [f0, f1, mut f2] = feats;

        // AIFI on the highest level (index 2) only.
        if let Some(layer) = self.aifi.layers.first() {
            let [b, c, h, w] = f2.dims();
            let flat = f2.reshape([b, c, h * w]).swap_dims(1, 2); // [B, HW, C]
            let pos = sine_position_embedding::<B>(h, w, c, &flat.device());
            let mut x = layer.forward(flat, pos);
            for layer in self.aifi.layers.iter().skip(1) {
                let pos = sine_position_embedding::<B>(h, w, c, &x.device());
                x = layer.forward(x, pos);
            }
            f2 = x.swap_dims(1, 2).reshape([b, c, h, w]);
        }

        // Top-down FPN (num_fpn_stages = 2). Written out for the fixed 3-level case
        // so no fallible `Vec` popping is needed.
        // Stage 0 fuses level 2 (top) with backbone level 1.
        let top2 = self.lateral_convs[0].forward(f2);
        let fused1 = Tensor::cat(vec![upsample2x(top2.clone()), f1], 1);
        let p1 = self.fpn_blocks[0].forward(fused1);
        // Stage 1 fuses the new level-1 map with backbone level 0.
        let top1 = self.lateral_convs[1].forward(p1);
        let fused0 = Tensor::cat(vec![upsample2x(top1.clone()), f0], 1);
        let p0 = self.fpn_blocks[1].forward(fused0);
        // FPN maps in low→high stride order: [p0(s8), top1(s16), top2(s32)].

        // Bottom-up PAN (num_pan_stages = 2).
        let down0 = self.downsample_convs[0].forward(p0.clone());
        let n1 = self.pan_blocks[0].forward(Tensor::cat(vec![down0, top1], 1));
        let down1 = self.downsample_convs[1].forward(n1.clone());
        let n2 = self.pan_blocks[1].forward(Tensor::cat(vec![down1, top2], 1));

        [p0, n1, n2]
    }
}

/// Nearest-neighbour 2× upsample (`F.interpolate(scale_factor=2, mode="nearest")`).
fn upsample2x<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [_, _, h, w] = x.dims();
    interpolate(
        x,
        [h * 2, w * 2],
        InterpolateOptions::new(InterpolateMode::Nearest),
    )
}

/// 2D sine-cosine positional embedding, `[1, H*W, embed_dim]`.
///
/// Reproduces `build_2d_sinusoidal_position_embedding` (temperature 10000): the
/// channel layout is `[sin_h | cos_h | sin_w | cos_w]`, `pos_dim = embed_dim/4`,
/// grid indexed `ij` (H outer). Computed on the host in `f64` then lifted to a
/// tensor, matching the float64 math of the reference.
fn sine_position_embedding<B: Backend>(h: usize, w: usize, embed_dim: usize, device: &B::Device) -> Tensor<B, 3> {
    let temperature = DET.positional_encoding_temperature;
    let pos_dim = embed_dim / 4;
    let omega: Vec<f64> = (0..pos_dim)
        .map(|i| 1.0 / temperature.powf(i as f64 / pos_dim as f64))
        .collect();

    let mut data = vec![0.0f32; h * w * embed_dim];
    for gy in 0..h {
        for gx in 0..w {
            let idx = gy * w + gx;
            let base = idx * embed_dim;
            let fx = gx as f64;
            let fy = gy as f64;
            for (k, &om) in omega.iter().enumerate() {
                let eh = fy * om;
                let ew = fx * om;
                // [sin_h | cos_h | sin_w | cos_w]
                data[base + k] = eh.sin() as f32;
                data[base + pos_dim + k] = eh.cos() as f32;
                data[base + 2 * pos_dim + k] = ew.sin() as f32;
                data[base + 3 * pos_dim + k] = ew.cos() as f32;
            }
        }
    }

    Tensor::<B, 1>::from_data(TensorData::new(data, [h * w * embed_dim]), device)
        .reshape([1, h * w, embed_dim])
}
