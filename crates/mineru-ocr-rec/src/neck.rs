//! LightSVTR recognition neck.
//!
//! A faithful Burn port of `EncoderWithLightSVTR` from
//! `pytorchocr/modeling/necks/rnn.py` (PP-OCRv6 CTC branch). It enhances the
//! backbone's `[N, C, H, W]` feature map with a lightweight local-conv +
//! global-attention block, then adds a skip connection — all before the feature is
//! flattened into a CTC sequence.
//!
//! In the PP-OCRv6 `MultiHead`, this neck and the final classifier are stored under
//! `head.encoder.*` and `head.head.*`, so this module is wired in beneath the head
//! (see [`crate::head`]).

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::prelude::Backend;
use burn::tensor::Tensor;
use burn::tensor::activation::{sigmoid, softmax};
use mineru_burn_common::nn::{FrozenBatchNorm2d, PtLayerNorm, PtLinear};

/// BatchNorm epsilon used by the checkpoint's `nn.BatchNorm2d`.
const BN_EPS: f64 = 1e-5;
/// LayerNorm epsilon used by LightSVTR (`nn.LayerNorm(eps=1e-6)`).
const LN_EPS: f64 = 1e-6;

/// Conv → BN → SiLU (`LightSVTRConvLayer`).
#[derive(Module, Debug)]
struct ConvLayer<B: Backend> {
    convolution: Conv2d<B>,
    normalization: FrozenBatchNorm2d<B>,
    #[module(skip)]
    use_silu: bool,
}

impl<B: Backend> ConvLayer<B> {
    fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: (usize, usize),
        groups: usize,
        use_silu: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            convolution: Conv2dConfig::new([in_ch, out_ch], [kernel.0, kernel.1])
                .with_stride([1, 1])
                .with_groups(groups)
                .with_padding(PaddingConfig2d::Explicit(kernel.0 / 2, kernel.1 / 2, kernel.0 / 2, kernel.1 / 2))
                .with_bias(false)
                .init(device),
            normalization: FrozenBatchNorm2d::init(out_ch, BN_EPS, device),
            use_silu,
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.convolution.forward(x);
        let x = self.normalization.forward(x);
        if self.use_silu {
            silu(x)
        } else {
            x
        }
    }
}

/// SiLU / swish activation: `x * sigmoid(x)`.
fn silu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone() * sigmoid(x)
}

/// Multi-head self-attention (`LightSVTRAttention`).
#[derive(Module, Debug)]
struct Attention<B: Backend> {
    // Fused q/k/v projection — one `[3*hidden, hidden]` weight, matching the
    // checkpoint's single `self_attn.qkv` tensor. Split into q/k/v after the
    // linear, in the forward pass.
    qkv: PtLinear<B>,
    projection: PtLinear<B>,
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    scale: f64,
}

impl<B: Backend> Attention<B> {
    fn new(hidden: usize, num_heads: usize, qkv_bias: bool, device: &B::Device) -> Self {
        let head_dim = hidden / num_heads;
        Self {
            qkv: PtLinear::init(hidden, 3 * hidden, qkv_bias, device),
            projection: PtLinear::init(hidden, hidden, true, device),
            num_heads,
            scale: (head_dim as f64).powf(-0.5),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, c] = x.dims();
        let head_dim = c / self.num_heads;

        // qkv: [b, n, 3c] -> [3, b, heads, n, head_dim].
        let qkv = self.qkv.forward(x);
        let qkv = qkv.reshape([b, n, 3, self.num_heads, head_dim]);
        // permute to [3, b, heads, n, head_dim].
        let qkv = qkv.permute([2, 0, 3, 1, 4]);
        let q = qkv.clone().narrow(0, 0, 1).squeeze_dim::<4>(0);
        let k = qkv.clone().narrow(0, 1, 1).squeeze_dim::<4>(0);
        let v = qkv.narrow(0, 2, 1).squeeze_dim::<4>(0);

        // attn = softmax(q @ k^T * scale).
        let attn = q.matmul(k.swap_dims(2, 3)) * self.scale;
        let attn = softmax(attn, 3);
        // out = attn @ v, written as `(vᵀ @ attnᵀ)ᵀ` (transpose over the last two
        // dims). `v` here is a permuted/narrowed view whose *batch* dims are not
        // contiguous; Burn 0.21's wgpu/cubecl batched matmul reads a batch-permuted
        // **right-hand** operand with the wrong strides (diverges from CPU by a large
        // margin), whereas keeping the permuted operand on the **left** with only a
        // last-two-dim transpose on the right matches CPU to float epsilon. No-op on
        // CPU. See the matching note in mineru-layout's `encoder.rs`.
        let out = v.swap_dims(2, 3).matmul(attn.swap_dims(2, 3)).swap_dims(2, 3);
        // -> [b, n, c].
        let out = out.swap_dims(1, 2).reshape([b, n, c]);
        self.projection.forward(out)
    }
}

/// Feed-forward MLP (`LightSVTRMLP`).
#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    fc1: PtLinear<B>,
    fc2: PtLinear<B>,
}

impl<B: Backend> Mlp<B> {
    fn new(hidden: usize, mlp_ratio: f64, device: &B::Device) -> Self {
        let inner = (hidden as f64 * mlp_ratio) as usize;
        Self {
            fc1: PtLinear::init(hidden, inner, true, device),
            fc2: PtLinear::init(inner, hidden, true, device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = silu(x);
        self.fc2.forward(x)
    }
}

/// One transformer block (`LightSVTRBlock`), pre-norm attention + pre-norm MLP.
#[derive(Module, Debug)]
struct SvtrBlock<B: Backend> {
    self_attn: Attention<B>,
    layer_norm1: PtLayerNorm<B>,
    mlp: Mlp<B>,
    layer_norm2: PtLayerNorm<B>,
}

impl<B: Backend> SvtrBlock<B> {
    fn new(hidden: usize, num_heads: usize, qkv_bias: bool, mlp_ratio: f64, device: &B::Device) -> Self {
        Self {
            self_attn: Attention::new(hidden, num_heads, qkv_bias, device),
            layer_norm1: PtLayerNorm::init(hidden, LN_EPS, device),
            mlp: Mlp::new(hidden, mlp_ratio, device),
            layer_norm2: PtLayerNorm::init(hidden, LN_EPS, device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let h = self.layer_norm1.forward(x);
        let x = residual + self.self_attn.forward(h);
        let residual = x.clone();
        let h = self.layer_norm2.forward(x);
        residual + self.mlp.forward(h)
    }
}

/// PP-OCRv6 LightSVTR neck (`EncoderWithLightSVTR`).
#[derive(Module, Debug)]
pub struct EncoderWithLightSvtr<B: Backend> {
    conv_block: Vec<ConvLayer<B>>,
    svtr_block: Vec<SvtrBlock<B>>,
    norm: PtLayerNorm<B>,
    #[module(skip)]
    out_channels: usize,
}

impl<B: Backend> EncoderWithLightSvtr<B> {
    /// Builds the neck. `in_channels` is the backbone output; `dims` the SVTR hidden
    /// width; `depth` the number of transformer blocks; `local_kernel` the local
    /// depthwise conv width (7 for PP-OCRv6).
    pub fn new(
        in_channels: usize,
        dims: usize,
        depth: usize,
        num_heads: usize,
        mlp_ratio: f64,
        local_kernel: usize,
        device: &B::Device,
    ) -> Self {
        let conv_block = vec![
            ConvLayer::new(in_channels, dims, (1, 1), 1, true, device),
            ConvLayer::new(in_channels, dims, (1, 1), 1, true, device),
            ConvLayer::new(dims, dims, (1, local_kernel), dims, true, device),
        ];
        let svtr_block = (0..depth)
            .map(|_| SvtrBlock::new(dims, num_heads, true, mlp_ratio, device))
            .collect();
        Self {
            conv_block,
            svtr_block,
            norm: PtLayerNorm::init(dims, LN_EPS, device),
            out_channels: dims,
        }
    }

    /// The neck's output channel count (`dims`).
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Enhances the `[N, C, H, W]` feature map, returning `[N, dims, H, W]`.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Skip branch (residual) and the local-conv-enhanced main branch.
        let residual = self.conv_block[0].forward(x.clone());
        let h = self.conv_block[1].forward(x);
        let h = h.clone() + self.conv_block[2].forward(h);

        let [b, c, height, width] = h.dims();
        // Flatten spatial dims -> [b, h*w, c] for the transformer.
        let mut seq = h.reshape([b, c, height * width]).swap_dims(1, 2);
        for block in &self.svtr_block {
            seq = block.forward(seq);
        }
        let seq = self.norm.forward(seq);
        // Back to [b, c, h, w] and add the skip.
        let out = seq
            .swap_dims(1, 2)
            .reshape([b, c, height, width]);
        out + residual
    }
}
