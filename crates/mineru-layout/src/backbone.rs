//! HGNetV2 backbone (arch "L"), the CNN feature extractor of PP-DocLayoutV2.
//!
//! Port of `RTDetrHGNetV2` / `HGNetV2Backbone` from HuggingFace `transformers`.
//! Produces three feature maps at strides `[8, 16, 32]` with channels
//! `[512, 1024, 2048]` (the `out_features = stage2/stage3/stage4` selection).
//!
//! # Key naming
//! Fields reproduce the checkpoint key chain
//! `model.backbone.model.{embedder,encoder.stages.…}` after the top-level prefix
//! remap. Conv+BN blocks name their parts `convolution` / `normalization` (the
//! HGNetV2 convention), distinct from the hybrid encoder's `conv` / `norm`.
//!
//! Because Burn's `Module` derive injects enum-variant names into parameter paths
//! (and the shared loader does not strip them), the light-vs-plain block layer is
//! modelled with a generic type parameter instead of an enum, and the four stages
//! are named fields (`stage0`..`stage3`) rather than a `Vec`. The [`crate::weights`]
//! remap rewrites the checkpoint's `stages.N.` to `stageN.` so keys line up.
//!
//! ## Generic block parameter
//! [`BasicLayer`] and [`Stage`] are generic over the block-conv type `L`, but they
//! bound `L` with the plain [`Module`] bound only — not the [`BlockConv`] trait.
//! Burn's `#[derive(Module)]` reads the *struct's* generic bounds to synthesise the
//! `AutodiffModule` impl, and a custom supertrait bound such as `L: BlockConv<B>`
//! forces the derive to demand `L::InnerModule: BlockConv<B::InnerBackend>`, a bound
//! the macro cannot express and which never gets satisfied. Keeping the struct bound
//! at `Module<B>` lets the derive succeed; the build/run behaviour is required only
//! where it is used, via `where L: BlockConv<B>` on the relevant `impl` blocks.
//!
//! # Fidelity notes
//! - `use_learnable_affine_block = false` for arch "L": there are no LAB params.
//! - Backbone BatchNorm is frozen (RT-DETR `RTDetrFrozenBatchNorm2d`, eps 1e-5).

use burn::module::Module;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::Backend;
use burn::tensor::Tensor;
use burn::tensor::activation::relu;

use crate::config::DET;
use crate::nn::FrozenBatchNorm2d;

/// Number of conv layers inside each basic block (`stage_numb_of_layers = 6`).
const LAYERS_PER_BLOCK: usize = 6;

/// A `HGNetV2ConvLayer`: convolution → frozen batch-norm → optional ReLU.
#[derive(Module, Debug)]
pub struct ConvLayer<B: Backend> {
    convolution: Conv2d<B>,
    normalization: FrozenBatchNorm2d<B>,
    #[module(skip)]
    act: bool,
}

impl<B: Backend> ConvLayer<B> {
    #[allow(clippy::too_many_arguments)]
    fn init(
        in_c: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        act: bool,
        device: &B::Device,
    ) -> Self {
        let convolution = Conv2dConfig::new([in_c, out_c], [kernel, kernel])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(padding, padding, padding, padding))
            .with_groups(groups)
            .with_bias(false)
            .init(device);
        let normalization = FrozenBatchNorm2d::init(out_c, DET.batch_norm_eps, device);
        Self {
            convolution,
            normalization,
            act,
        }
    }

    /// A `k×k` conv with symmetric `(k-1)/2` padding.
    fn plain(in_c: usize, out_c: usize, kernel: usize, stride: usize, act: bool, device: &B::Device) -> Self {
        Self::init(in_c, out_c, kernel, stride, (kernel - 1) / 2, 1, act, device)
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.convolution.forward(x);
        let x = self.normalization.forward(x);
        if self.act { relu(x) } else { x }
    }
}

/// A `HGNetV2ConvLayerLight`: pointwise 1×1 conv → depthwise `k×k` conv.
#[derive(Module, Debug)]
pub struct ConvLayerLight<B: Backend> {
    conv1: ConvLayer<B>,
    conv2: ConvLayer<B>,
}

impl<B: Backend> ConvLayerLight<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv2.forward(self.conv1.forward(x))
    }
}

/// A block layer, either plain or light, without an enum in the module tree.
///
/// Implemented for [`ConvLayer`] and [`ConvLayerLight`] so [`BasicLayer`] can be
/// generic over which one a stage uses. Modelling the choice as a generic type
/// parameter (rather than an enum) keeps the parameter paths identical to the
/// checkpoint: a plain stage's layers serialise as `layers.N.convolution.…` and a
/// light stage's as `layers.N.conv1.…` / `layers.N.conv2.…`, with no `Plain`/`Light`
/// variant segment for the [`crate::weights`] remap to reconcile.
///
/// This is deliberately declared as a supertrait of [`Module`] so that any
/// implementor is a module, yet [`BasicLayer`] / [`Stage`] never name this trait in
/// their own struct bounds — see the module-level "Generic block parameter" note for
/// why that matters to Burn's `#[derive(Module)]`.
pub trait BlockConv<B: Backend>: Module<B> {
    /// Builds one layer with the given input/mid channels and kernel size.
    fn build(in_c: usize, mid_c: usize, kernel: usize, device: &B::Device) -> Self;
    /// Runs the layer.
    fn run(&self, x: Tensor<B, 4>) -> Tensor<B, 4>;
}

impl<B: Backend> BlockConv<B> for ConvLayer<B> {
    fn build(in_c: usize, mid_c: usize, kernel: usize, device: &B::Device) -> Self {
        ConvLayer::plain(in_c, mid_c, kernel, 1, true, device)
    }
    fn run(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward(x)
    }
}

impl<B: Backend> BlockConv<B> for ConvLayerLight<B> {
    fn build(in_c: usize, mid_c: usize, kernel: usize, device: &B::Device) -> Self {
        // conv1: 1x1 no-act; conv2: depthwise k×k relu.
        let conv1 = ConvLayer::init(in_c, mid_c, 1, 1, 0, 1, false, device);
        let conv2 = ConvLayer::init(mid_c, mid_c, kernel, 1, (kernel - 1) / 2, mid_c, true, device);
        ConvLayerLight { conv1, conv2 }
    }
    fn run(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward(x)
    }
}

/// A `HGNetV2BasicLayer`: N conv layers, concat, squeeze/excite, optional residual.
///
/// Generic over the block-conv type `L`. The struct bounds `L` only with `Module<B>`
/// (so the derived `AutodiffModule` impl is satisfiable); construction and execution
/// live on the `impl … where L: BlockConv<B>` block below.
#[derive(Module, Debug)]
pub struct BasicLayer<B: Backend, L: Module<B>> {
    layers: Vec<L>,
    // Sequential(squeeze, excite) → checkpoint keys aggregation.0 / aggregation.1.
    // A Vec of two ConvLayers reproduces the numeric indices exactly.
    aggregation: Vec<ConvLayer<B>>,
    #[module(skip)]
    residual: bool,
    #[module(skip)]
    _marker: core::marker::PhantomData<B>,
}

impl<B: Backend, L: BlockConv<B>> BasicLayer<B, L> {
    #[allow(clippy::too_many_arguments)]
    fn init(
        in_c: usize,
        mid_c: usize,
        out_c: usize,
        kernel: usize,
        residual: bool,
        device: &B::Device,
    ) -> Self {
        let mut layers = Vec::with_capacity(LAYERS_PER_BLOCK);
        for i in 0..LAYERS_PER_BLOCK {
            let layer_in = if i == 0 { in_c } else { mid_c };
            layers.push(L::build(layer_in, mid_c, kernel, device));
        }
        let total = in_c + LAYERS_PER_BLOCK * mid_c;
        let aggregation = vec![
            ConvLayer::init(total, out_c / 2, 1, 1, 0, 1, true, device),
            ConvLayer::init(out_c / 2, out_c, 1, 1, 0, 1, true, device),
        ];
        Self {
            layers,
            aggregation,
            residual,
            _marker: core::marker::PhantomData,
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let identity = x.clone();
        let mut outs = Vec::with_capacity(self.layers.len() + 1);
        outs.push(x.clone());
        let mut cur = x;
        for layer in &self.layers {
            cur = layer.run(cur);
            outs.push(cur.clone());
        }
        let concat = Tensor::cat(outs, 1);
        let squeezed = self.aggregation[0].forward(concat);
        let aggregated = self.aggregation[1].forward(squeezed);
        if self.residual {
            aggregated.add(identity)
        } else {
            aggregated
        }
    }
}

/// A `HGNetV2Stage`: an optional depthwise downsample followed by N basic blocks.
///
/// Generic over the block-conv type `L`, bounded by `Module<B>` at the struct level
/// for the same derive reason as [`BasicLayer`]; the `BlockConv` bound is required
/// only on the `impl` block.
#[derive(Module, Debug)]
pub struct Stage<B: Backend, L: Module<B>> {
    downsample: Option<ConvLayer<B>>,
    blocks: Vec<BasicLayer<B, L>>,
}

impl<B: Backend, L: BlockConv<B>> Stage<B, L> {
    #[allow(clippy::too_many_arguments)]
    fn init(
        in_c: usize,
        mid_c: usize,
        out_c: usize,
        num_blocks: usize,
        kernel: usize,
        downsample: bool,
        device: &B::Device,
    ) -> Self {
        let downsample =
            downsample.then(|| ConvLayer::init(in_c, in_c, 3, 2, 1, in_c, false, device));
        let mut blocks = Vec::with_capacity(num_blocks);
        for b in 0..num_blocks {
            let (block_in, residual) = if b == 0 { (in_c, false) } else { (out_c, true) };
            blocks.push(BasicLayer::init(block_in, mid_c, out_c, kernel, residual, device));
        }
        Self { downsample, blocks }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = match &self.downsample {
            Some(d) => d.forward(x),
            None => x,
        };
        for block in &self.blocks {
            x = block.forward(x);
        }
        x
    }
}

/// The stem (`HGNetV2Embeddings`): 5 conv layers plus a maxpool branch.
#[derive(Module, Debug)]
pub struct Embedder<B: Backend> {
    stem1: ConvLayer<B>,
    stem2a: ConvLayer<B>,
    stem2b: ConvLayer<B>,
    stem3: ConvLayer<B>,
    stem4: ConvLayer<B>,
}

impl<B: Backend> Embedder<B> {
    fn init(device: &B::Device) -> Self {
        Self {
            stem1: ConvLayer::init(3, 32, 3, 2, 1, 1, true, device),
            stem2a: ConvLayer::init(32, 16, 2, 1, 0, 1, true, device),
            stem2b: ConvLayer::init(16, 32, 2, 1, 0, 1, true, device),
            stem3: ConvLayer::init(64, 32, 3, 2, 1, 1, true, device),
            stem4: ConvLayer::init(32, 48, 1, 1, 0, 1, true, device),
        }
    }

    fn forward(&self, pixel_values: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.stem1.forward(pixel_values);
        let x = pad_right_bottom(x);
        let a = self.stem2a.forward(x.clone());
        let a = pad_right_bottom(a);
        let a = self.stem2b.forward(a);
        let p = max_pool_2_ceil(x);
        let x = Tensor::cat(vec![p, a], 1);
        let x = self.stem3.forward(x);
        self.stem4.forward(x)
    }
}

/// The `HGNetV2Encoder`: the four stages, as named fields so no `Vec` index and
/// no enum leaks into the parameter paths. Stages 0/1 are plain, 2/3 are light.
#[derive(Module, Debug)]
pub struct HgEncoder<B: Backend> {
    stage0: Stage<B, ConvLayer<B>>,
    stage1: Stage<B, ConvLayer<B>>,
    stage2: Stage<B, ConvLayerLight<B>>,
    stage3: Stage<B, ConvLayerLight<B>>,
}

/// The full backbone: `embedder` + `encoder`.
#[derive(Module, Debug)]
pub struct Backbone<B: Backend> {
    embedder: Embedder<B>,
    encoder: HgEncoder<B>,
}

impl<B: Backend> Backbone<B> {
    /// Initialises the HGNetV2-L backbone.
    pub fn init(device: &B::Device) -> Self {
        // arch "L": in/mid/out channels, block counts, downsample, kernel per stage.
        let encoder = HgEncoder {
            stage0: Stage::init(48, 48, 128, 1, 3, false, device),
            stage1: Stage::init(128, 96, 512, 1, 3, true, device),
            stage2: Stage::init(512, 192, 1024, 3, 5, true, device),
            stage3: Stage::init(1024, 384, 2048, 1, 5, true, device),
        };
        Self {
            embedder: Embedder::init(device),
            encoder,
        }
    }

    /// Runs the backbone, returning the three feature maps in low→high stride
    /// order (`stage2`, `stage3`, `stage4` = encoder stages 1, 2, 3).
    pub fn forward(&self, pixel_values: Tensor<B, 4>) -> [Tensor<B, 4>; 3] {
        let stem = self.embedder.forward(pixel_values);
        let s0 = self.encoder.stage0.forward(stem);
        let s1 = self.encoder.stage1.forward(s0);
        let s2 = self.encoder.stage2.forward(s1.clone());
        let s3 = self.encoder.stage3.forward(s2.clone());
        [s1, s2, s3]
    }
}

/// Pads a tensor by one element on the right and bottom (`F.pad(x, (0,1,0,1))`).
fn pad_right_bottom<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [n, c, h, w] = x.dims();
    let device = x.device();
    let out = Tensor::zeros([n, c, h + 1, w + 1], &device);
    out.slice_assign([0..n, 0..c, 0..h, 0..w], x)
}

/// `MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)` over an NCHW tensor.
///
/// For a kernel-2, stride-1 pool every 2×2 window fits fully inside an `H×W` input,
/// so the output is `(H-1)×(W-1)` — `ceil_mode` adds no partial window because
/// `(H-2)/1` is already an integer. Output position `(i, j)` is the max over the
/// four input positions `(i..i+2, j..j+2)`. Implemented directly because Burn's
/// pooling exposes no `ceil_mode`; the `(H-1)` shrink matches the reference and is
/// what makes the pooled branch line up with the `stem2b` branch for concat.
///
/// Inputs must be at least 2×2 (they always are here: the stem runs on the padded
/// `stem1` output). Smaller inputs are returned unchanged to avoid a zero-size dim.
fn max_pool_2_ceil<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [_, _, h, w] = x.dims();
    if h < 2 || w < 2 {
        return x;
    }
    // The four 2×2-window offsets, each an (H-1)×(W-1) crop.
    let tl = x.clone().narrow(2, 0, h - 1).narrow(3, 0, w - 1);
    let tr = x.clone().narrow(2, 0, h - 1).narrow(3, 1, w - 1);
    let bl = x.clone().narrow(2, 1, h - 1).narrow(3, 0, w - 1);
    let br = x.narrow(2, 1, h - 1).narrow(3, 1, w - 1);
    tl.max_pair(tr).max_pair(bl).max_pair(br)
}
