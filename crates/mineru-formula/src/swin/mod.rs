//! Swin-Transformer vision encoder (`UnimerSwinModel`).
//!
//! Port of `unimer_swin/modeling_unimer_swin.py`. The encoder turns a
//! `[B, C, H, W]` image into a grid of visual tokens `[B, L, hidden_size]` that the
//! MBart decoder cross-attends over.
//!
//! Structure (four stages, `depths = [6, 6, 6, 6]`):
//!
//! ```text
//! PatchEmbeddings (stem, /4)  ->  [B, (H/4)(W/4), 96]
//!   Stage 0: 6x SwinLayer(dim=96),  PatchMerging -> dim 192, /2
//!   Stage 1: 6x SwinLayer(dim=192), PatchMerging -> dim 384, /2
//!   Stage 2: 6x SwinLayer(dim=384), PatchMerging -> dim 768, /2
//!   Stage 3: 6x SwinLayer(dim=768), (no merge)
//! -> last_hidden_state [B, L, 768]
//! ```
//!
//! There is no final LayerNorm on top (the docstring notes this is Donut-Swin,
//! which drops it), and `add_pooling_layer` output is unused by the decoder, so we
//! only return `last_hidden_state`.

pub mod attention;
pub mod embeddings;
pub mod layer;

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use mineru_burn_common::nn::{PtLayerNorm, PtLinear};

use crate::config::SwinConfig;
use embeddings::PatchEmbeddings;
use layer::SwinLayer;

/// Patch-merging downsample: `[B, H*W, C] -> [B, (H/2)(W/2), 2C]`.
///
/// Port of `UnimerSwinPatchMerging`. Concatenates the four strided sub-grids
/// (`[0::2,0::2], [1::2,0::2], [0::2,1::2], [1::2,1::2]`) into `4C`, LayerNorm's,
/// then a bias-free `Linear(4C -> 2C)`.
#[derive(Module, Debug)]
pub struct PatchMerging<B: Backend> {
    norm: PtLayerNorm<B>,
    reduction: PtLinear<B>,
}

impl<B: Backend> PatchMerging<B> {
    /// Builds a patch merging for input channel dim `dim`.
    pub fn new(cfg: &SwinConfig, dim: usize, device: &B::Device) -> Self {
        Self {
            norm: PtLayerNorm::init(4 * dim, cfg.layer_norm_eps, device),
            reduction: PtLinear::init(4 * dim, 2 * dim, false, device),
        }
    }

    /// Merges. `(h, w)` is the input spatial size (assumed even; the encoder feeds
    /// even sizes because 420/4 = 105 is odd — see the note below).
    ///
    /// The Python `maybe_pad`s odd H/W to even before striding. We reproduce that
    /// by slicing with a ceil-div count via strided [`sub_grid`].
    pub fn forward(&self, x: Tensor<B, 3>, h: usize, w: usize) -> Tensor<B, 3> {
        let [b, _n, c] = x.dims();
        let x = x.reshape([b, h, w, c]);
        // The four strided quadrants; ceil-div handles odd H/W like the pad path.
        let x0 = sub_grid(x.clone(), 0, 0);
        let x1 = sub_grid(x.clone(), 1, 0);
        let x2 = sub_grid(x.clone(), 0, 1);
        let x3 = sub_grid(x, 1, 1);
        let merged = Tensor::cat(vec![x0, x1, x2, x3], 3); // [B, H2, W2, 4C]
        let [_, h2, w2, c4] = merged.dims();
        let merged = merged.reshape([b, h2 * w2, c4]);
        let merged = self.norm.forward(merged);
        self.reduction.forward(merged)
    }
}

/// Extracts the strided sub-grid `x[:, ro::2, co::2, :]` from `[B, H, W, C]`.
fn sub_grid<B: Backend>(x: Tensor<B, 4>, ro: usize, co: usize) -> Tensor<B, 4> {
    let [b, h, w, c] = x.dims();
    // Number of kept rows/cols with a step of 2 starting at ro/co (ceil-div).
    let hh = h.saturating_sub(ro).div_ceil(2);
    let ww = w.saturating_sub(co).div_ceil(2);
    // Burn slices are contiguous; emulate strided gather via reshape when even,
    // otherwise fall back to a gather. For the common even case (H,W even) we can
    // reshape [B, hh, 2, ww, 2, C] and index the (ro, co) slot.
    if h % 2 == 0 && w % 2 == 0 {
        x.reshape([b, hh, 2, ww, 2, c])
            .narrow(2, ro, 1)
            .narrow(4, co, 1)
            .reshape([b, hh, ww, c])
    } else {
        // Odd dimension: pad to even (matching maybe_pad) then take the slot.
        let hp = h + (h % 2);
        let wp = w + (w % 2);
        let device = x.device();
        let mut padded: Tensor<B, 4> = Tensor::zeros([b, hp, wp, c], &device);
        padded = padded.slice_assign([0..b, 0..h, 0..w, 0..c], x);
        padded
            .reshape([b, hp / 2, 2, wp / 2, 2, c])
            .narrow(2, ro, 1)
            .narrow(4, co, 1)
            .reshape([b, hp / 2, wp / 2, c])
    }
}

/// One Swin stage: a run of [`SwinLayer`]s and an optional [`PatchMerging`].
#[derive(Module, Debug)]
pub struct SwinStage<B: Backend> {
    blocks: Vec<SwinLayer<B>>,
    downsample: Option<PatchMerging<B>>,
}

impl<B: Backend> SwinStage<B> {
    /// Builds a stage of `depth` blocks at dimension `dim`; `downsample` adds a
    /// patch merge at the end (all stages but the last).
    pub fn new(
        cfg: &SwinConfig,
        dim: usize,
        depth: usize,
        num_heads: usize,
        downsample: bool,
        device: &B::Device,
    ) -> Self {
        let blocks = (0..depth)
            .map(|_| SwinLayer::new(cfg, dim, num_heads, device))
            .collect();
        Self {
            blocks,
            downsample: downsample.then(|| PatchMerging::new(cfg, dim, device)),
        }
    }

    /// Runs the stage. Returns `(hidden, (h', w'))` — the post-merge spatial size.
    pub fn forward(&self, mut hidden: Tensor<B, 3>, h: usize, w: usize) -> (Tensor<B, 3>, (usize, usize)) {
        for block in &self.blocks {
            hidden = block.forward(hidden, h, w);
        }
        match &self.downsample {
            Some(merge) => {
                let out = merge.forward(hidden, h, w);
                (out, (h.div_ceil(2), w.div_ceil(2)))
            }
            None => (hidden, (h, w)),
        }
    }
}

/// The full Swin encoder.
#[derive(Module, Debug)]
pub struct SwinEncoder<B: Backend> {
    embeddings: PatchEmbeddings<B>,
    stages: Vec<SwinStage<B>>,
}

impl<B: Backend> SwinEncoder<B> {
    /// Builds the encoder from the config.
    pub fn new(cfg: &SwinConfig, device: &B::Device) -> Self {
        let embeddings = PatchEmbeddings::new(cfg, device);
        let stages = (0..SwinConfig::NUM_STAGES)
            .map(|i| {
                SwinStage::new(
                    cfg,
                    cfg.stage_dim(i),
                    cfg.depths[i],
                    cfg.num_heads[i],
                    i < SwinConfig::NUM_STAGES - 1,
                    device,
                )
            })
            .collect();
        Self { embeddings, stages }
    }

    /// Encodes pixel values `[B, C, H, W]` into visual tokens `[B, L, hidden_size]`.
    pub fn forward(&self, pixel_values: Tensor<B, 4>) -> Tensor<B, 3> {
        let (mut hidden, (mut h, mut w)) = self.embeddings.forward(pixel_values);
        for stage in &self.stages {
            let (out, (nh, nw)) = stage.forward(hidden, h, w);
            hidden = out;
            h = nh;
            w = nw;
        }
        hidden
    }

    /// Parity hook: forward that also returns the per-stage activations.
    ///
    /// Returns `(patch_embed, [stage_0, .., stage_3])` where `patch_embed` is the
    /// post-LayerNorm patch-embedding output and each stage entry is that stage's
    /// output (post-merge for stages 0..2). The final stage output is the encoder
    /// memory. Used only by the numerical-parity test; not part of the stable API.
    #[doc(hidden)]
    pub fn forward_stages(
        &self,
        pixel_values: Tensor<B, 4>,
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 3>>) {
        let (mut hidden, (mut h, mut w)) = self.embeddings.forward(pixel_values);
        let embed = hidden.clone();
        let mut stages = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            let (out, (nh, nw)) = stage.forward(hidden, h, w);
            hidden = out;
            h = nh;
            w = nw;
            stages.push(hidden.clone());
        }
        (embed, stages)
    }
}
