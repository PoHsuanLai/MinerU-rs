//! SLANet-plus forward pass (hand-ported Burn module) and weight loading.
//!
//! Architecture (from the PP-Structure SLANet-plus ONNX graph):
//!
//! ```text
//! image[1,3,488,488]
//!   └─ PP-LCNet backbone (depthwise-separable stages + 2 squeeze-excite blocks)
//!        └─ CSP-PAN neck (top-down upsample + bottom-up fuse) ─▶ [1,96,16,16]
//!             └─ flatten ─▶ [1,256,96] feature sequence
//!                  └─ SLAHead: autoregressive attention-GRU decoder
//!                       ├─ structure branch ─▶ structure_probs[1, L, 50]
//!                       └─ loc branch        ─▶ loc_preds[1, L, 8] (quad corners)
//! ```
//!
//! # Why hand-ported (not `burn-onnx` codegen)
//!
//! The backbone/neck import cleanly, but the SLAHead is exported as an ONNX
//! `Loop` (a dynamic-length autoregressive decoder). `burn-onnx` 0.21 fails to
//! import the graph well before the `Loop`: its type inference panics on the
//! pre-loop `ConstantOfShape` nodes —
//!
//! ```text
//! Type inference failed: Node 'constantofshape5' (ConstantOfShape):
//!   Type mismatch: expected Tensor, Shape, or ScalarTensor, got ScalarNative(I64)
//! ```
//!
//! (`onnx-simplifier`, which would fold those shape helpers away, segfaults on the
//! `Loop` subgraph). So — unlike the LCNet classifier and UNet, which do codegen —
//! the whole SLANet-plus network is hand-ported here: [`super::backbone`] for the
//! CNN and [`super::head`] for the unrolled decoder.
//!
//! # Weights
//!
//! Weights are the ONNX graph's constant tensors, exported to a flat
//! `.safetensors` (linears transposed to `[out, in]`, convs/BNs keyed by node
//! index) and loaded at runtime via [`mineru_burn_common::weights`]. [`SlaNet::load`]
//! is given the `.onnx` path (the pipeline's model path) and loads the sibling
//! `<stem>.safetensors`. Without that file the model stays unloaded and
//! [`SlaNet::forward`] reports [`Error::ModelUnavailable`], so callers degrade
//! gracefully rather than run on random weights.

use std::path::Path;

use burn::module::Module;
use burn::tensor::{Tensor, TensorData};
use mineru_burn_common::backend::{cpu_device, Cpu};
use mineru_burn_common::weights::{load_weights, Coverage, KeyRemap};

use crate::error::{Error, Result};

use super::backbone::Backbone;
use super::decode::RawPreds;
use super::head::{SlaHead, LOC_DIM, NUM_CLASSES};
use super::preprocess::TABLE_MAX_LEN;
use super::vocab::build_vocab;

/// Maximum autoregressive decode steps (the ONNX loop's `max_text_length` is 500).
const MAX_STEPS: usize = 500;

/// Raw SLANet outputs for one table: `structure_probs` `[L, C]` and `loc_preds`
/// reduced to `[L, 4]` axis-aligned boxes, on host `f32` buffers.
#[derive(Debug, Clone)]
pub struct SlaOutput {
    /// Flattened `[L, C]` structure class probabilities, row-major.
    pub structure_probs: Vec<f32>,
    /// Number of decode steps `L`.
    pub len: usize,
    /// Number of class channels `C`.
    pub num_classes: usize,
    /// Flattened `[L, 4]` box regressions, row-major.
    pub loc_preds: Vec<f32>,
}

impl SlaOutput {
    /// Borrows the buffers as a [`RawPreds`] view for the decoder.
    pub fn as_preds(&self) -> RawPreds<'_> {
        RawPreds {
            structure_probs: &self.structure_probs,
            len: self.len,
            num_classes: self.num_classes,
            loc_preds: &self.loc_preds,
        }
    }
}

/// The SLANet-plus Burn module (backbone + neck + head), carrying the parameters.
///
/// Wrapped by the public [`SlaNet`]; kept separate so the wrapper can hold the
/// `ready` flag and device without those becoming module parameters.
#[derive(Module, Debug)]
pub struct SlaNetInner<B: burn::prelude::Backend> {
    backbone: Backbone<B>,
    head: SlaHead<B>,
}

/// The hand-ported SLANet-plus model.
///
/// Construct it with [`SlaNet::load`]. When the sibling `.safetensors` weights are
/// absent, construction still succeeds but [`SlaNet::forward`] returns
/// [`Error::ModelUnavailable`], so the pipeline degrades gracefully.
#[derive(Debug)]
pub struct SlaNet {
    inner: SlaNetInner<Cpu>,
    device: burn::backend::ndarray::NdArrayDevice,
    ready: bool,
    num_classes: usize,
}

impl Default for SlaNet {
    fn default() -> Self {
        Self::new()
    }
}

impl SlaNet {
    /// Creates an unweighted model. `forward` reports the model unavailable.
    pub fn new() -> Self {
        let device = cpu_device();
        let inner = SlaNetInner {
            backbone: Backbone::new(&device),
            head: SlaHead::init(&device),
        };
        Self {
            inner,
            device,
            ready: false,
            // The decoder argmaxes over the model's real class-channel count.
            num_classes: NUM_CLASSES,
        }
    }

    /// Loads SLANet-plus weights.
    ///
    /// `path` is the SLANet-plus `.onnx` path (as the pipeline stores it); the
    /// weights are loaded from the sibling `<stem>.safetensors`. When that file is
    /// missing the model stays unloaded (so `forward` reports unavailable) rather
    /// than erroring, matching the crate's graceful-degradation policy for absent
    /// model files.
    ///
    /// # Errors
    ///
    /// [`Error::WeightLoad`] if the `.safetensors` exists but cannot be applied
    /// (bad file, or a key/shape mismatch under the strict coverage check).
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut model = Self::new();
        let weights = path.as_ref().with_extension("safetensors");
        if !weights.exists() {
            return Ok(model);
        }
        // The converted checkpoint keys the backbone convs/BNs as `conv.<N>.*` /
        // `bn.<M>.*`; the Burn field path nests them under the `backbone` module, so
        // prefix those (head keys already match `head.*`).
        let remap = KeyRemap::new()
            .rename(r"^(conv|bn)\.", "backbone.$1.")
            .map_err(|e| Error::WeightLoad(e.to_string()))?;
        load_weights::<Cpu, _>(&mut model.inner, &weights, &remap, Coverage::Strict)
            .map_err(|e| Error::WeightLoad(e.to_string()))?;
        model.ready = true;
        Ok(model)
    }

    /// Runs the CNN backbone + neck + SLAHead decoder over a preprocessed crop.
    ///
    /// Returns [`Error::ModelUnavailable`] when weights were not loaded. The box
    /// output is reduced from the model's four quad corners `[L, 8]` to the
    /// axis-aligned `[L, 4]` `[x_min, y_min, x_max, y_max]` the decoder contract
    /// expects (mirroring the Python matcher's `_normalize_cell_bboxes`).
    pub fn forward(&self, input: &super::preprocess::Preprocessed) -> Result<SlaOutput> {
        if !self.ready {
            return Err(Error::ModelUnavailable("slanet-plus"));
        }
        let side = TABLE_MAX_LEN as usize;
        if input.chw.len() != 3 * side * side {
            return Err(Error::OutputShape {
                expected: format!("[3, {side}, {side}]"),
                got: format!("[{}]", input.chw.len()),
            });
        }

        let data = TensorData::new(input.chw.clone(), [1, 3, side, side]);
        let x = Tensor::<Cpu, 4>::from_data(data, &self.device);

        let fea = self.inner.backbone.forward_sequence(x); // [1, T, 96]
        let end_idx = build_vocab().end_idx;
        let out = self.inner.head.forward(fea, MAX_STEPS, end_idx);

        let loc_preds = reduce_quads_to_boxes(&out.loc_preds, out.len);

        Ok(SlaOutput {
            structure_probs: out.structure_probs,
            len: out.len,
            num_classes: self.num_classes,
            loc_preds,
        })
    }
}

/// Parity hook (hidden): runs only the backbone + neck over a preprocessed crop
/// and returns the flattened `[T, 96]` feature sequence, for numeric comparison
/// against the ONNX reference. Not part of the public API.
#[doc(hidden)]
impl SlaNet {
    /// Returns the backbone/neck feature sequence `[T·96]` (row-major `[T, 96]`).
    pub fn debug_backbone_feature(&self, input: &super::preprocess::Preprocessed) -> Option<Vec<f32>> {
        let side = TABLE_MAX_LEN as usize;
        if input.chw.len() != 3 * side * side {
            return None;
        }
        let data = TensorData::new(input.chw.clone(), [1, 3, side, side]);
        let x = Tensor::<Cpu, 4>::from_data(data, &self.device);
        let fea = self.inner.backbone.forward_sequence(x); // [1, T, 96]
        fea.into_data().into_vec::<f32>().ok()
    }

    /// Returns per-step `(hidden[256], argmax, probs[50])` traces from the head,
    /// for step-by-step comparison against the ONNX `Loop`.
    pub fn debug_head_steps(
        &self,
        input: &super::preprocess::Preprocessed,
        steps: usize,
    ) -> Option<Vec<super::head::StepTrace>> {
        let side = TABLE_MAX_LEN as usize;
        if input.chw.len() != 3 * side * side {
            return None;
        }
        let data = TensorData::new(input.chw.clone(), [1, 3, side, side]);
        let x = Tensor::<Cpu, 4>::from_data(data, &self.device);
        let fea = self.inner.backbone.forward_sequence(x);
        Some(self.inner.head.debug_steps(fea, steps))
    }

    /// Runs the full decoder and returns the raw head outputs: flattened
    /// `[L, NUM_CLASSES]` structure probabilities, flattened `[L, LOC_DIM]` box
    /// *quad* corners (before the axis-aligned reduction [`SlaNet::forward`]
    /// applies), and the decoded step count `L`. For quad-level parity against the
    /// ONNX `[L, 8]` loc reference.
    pub fn debug_raw_head(
        &self,
        input: &super::preprocess::Preprocessed,
    ) -> Option<(Vec<f32>, Vec<f32>, usize)> {
        if !self.ready {
            return None;
        }
        let side = TABLE_MAX_LEN as usize;
        if input.chw.len() != 3 * side * side {
            return None;
        }
        let data = TensorData::new(input.chw.clone(), [1, 3, side, side]);
        let x = Tensor::<Cpu, 4>::from_data(data, &self.device);
        let fea = self.inner.backbone.forward_sequence(x);
        let end_idx = build_vocab().end_idx;
        let out = self.inner.head.forward(fea, MAX_STEPS, end_idx);
        Some((out.structure_probs, out.loc_preds, out.len))
    }
}

/// Reduces `[L, 8]` quadrilateral corner coordinates to `[L, 4]`
/// `[x_min, y_min, x_max, y_max]` axis-aligned boxes.
///
/// The SLANet-plus loc branch regresses four `(x, y)` corners per cell; the
/// downstream decoder and matcher work in axis-aligned boxes, so we collapse the
/// quad to its bounding rectangle exactly as the Python `_normalize_cell_bboxes`
/// does (min/max over the even/odd coordinate lanes).
fn reduce_quads_to_boxes(quads: &[f32], len: usize) -> Vec<f32> {
    let mut boxes = Vec::with_capacity(len * 4);
    for step in 0..len {
        let q = &quads[step * LOC_DIM..step * LOC_DIM + LOC_DIM];
        let xs = [q[0], q[2], q[4], q[6]];
        let ys = [q[1], q[3], q[5], q[7]];
        let x_min = xs.iter().copied().fold(f32::INFINITY, f32::min);
        let x_max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let y_min = ys.iter().copied().fold(f32::INFINITY, f32::min);
        let y_max = ys.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        boxes.extend_from_slice(&[x_min, y_min, x_max, y_max]);
    }
    boxes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unweighted_model_reports_unavailable() {
        let m = SlaNet::new();
        let pre = super::super::preprocess::preprocess(&image::RgbImage::new(64, 64));
        assert!(matches!(
            m.forward(&pre),
            Err(Error::ModelUnavailable("slanet-plus"))
        ));
    }

    #[test]
    fn quad_reduces_to_bounding_box() {
        // One quad: corners (0.1,0.2),(0.3,0.2),(0.3,0.4),(0.1,0.4) -> bbox.
        let quads = vec![0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.1, 0.4];
        let boxes = reduce_quads_to_boxes(&quads, 1);
        assert_eq!(boxes, vec![0.1, 0.2, 0.3, 0.4]);
    }
}
