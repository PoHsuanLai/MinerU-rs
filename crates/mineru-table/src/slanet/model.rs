//! SLANet-plus forward pass (hand-ported Burn module) and weight loading.
//!
//! Architecture recap (from the PP-Structure SLANet-plus ONNX graph):
//!
//! ```text
//! image[1,3,488,488]
//!   └─ PP-LCNet backbone (depthwise-separable conv stages) ─▶ feature map
//!        └─ SLAHead:
//!             ├─ structure branch: attention GRU over feature map,
//!             │    L autoregressive steps ─▶ structure_probs[1, L, C]
//!             └─ loc branch: per-step FC ─▶ loc_preds[1, L, 4]
//! ```
//!
//! The backbone is a plain CNN and would import cleanly; the SLAHead's
//! autoregressive attention decoder with a dynamic `L` is what makes `burn-onnx`
//! codegen unreliable, so the whole model is hand-ported here.
//!
//! The layer wiring below is a skeleton: it fixes the tensor contract the rest of
//! the crate depends on (`structure_probs[1,L,C]`, `loc_preds[1,L,4]`) and the
//! weight-loading entry point, but the exact per-layer graph must be finalized
//! against the real weight tensor names/shapes. Until then [`SlaNet::forward`]
//! reports [`Error::ModelUnavailable`], so callers degrade gracefully rather than
//! run with random weights.

use std::path::Path;

use crate::error::{Error, Result};

use super::decode::RawPreds;
use super::vocab::build_vocab;

/// Raw SLANet outputs for one table: `structure_probs` `[L, C]` and `loc_preds`
/// `[L, 4]`, already moved to host `f32` buffers.
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

/// The hand-ported SLANet-plus model.
///
/// Construct it with [`SlaNet::load`]. When built without the `onnx-import`
/// feature / weights, construction still succeeds but [`SlaNet::forward`] returns
/// [`Error::ModelUnavailable`].
#[derive(Debug, Default)]
pub struct SlaNet {
    /// Whether real weights are loaded and the forward pass is wired.
    ready: bool,
    /// Class-channel count, taken from the decode vocabulary.
    num_classes: usize,
}

impl SlaNet {
    /// Creates an unweighted model. `forward` will report the model unavailable.
    pub fn new() -> Self {
        Self {
            ready: false,
            num_classes: build_vocab().tokens.len(),
        }
    }

    /// Loads SLANet-plus weights from a safetensors/PyTorch file.
    ///
    /// The exact key remapping is finalized alongside the layer wiring; until the
    /// forward graph is complete this records that weights were supplied but keeps
    /// `forward` reporting unavailable, so no path silently runs on partial state.
    pub fn load<P: AsRef<Path>>(_weights: P) -> Result<Self> {
        // TODO: wire burn-store loading + SLAHead graph. See module docs.
        Ok(Self {
            ready: false,
            num_classes: build_vocab().tokens.len(),
        })
    }

    /// Runs the CNN backbone + SLAHead decoder.
    ///
    /// Returns [`Error::ModelUnavailable`] until the graph is finalized.
    pub fn forward(&self, _input: &super::preprocess::Preprocessed) -> Result<SlaOutput> {
        if !self.ready {
            return Err(Error::ModelUnavailable("slanet-plus"));
        }
        // Unreachable until `ready` is set by a completed loader.
        Ok(SlaOutput {
            structure_probs: Vec::new(),
            len: 0,
            num_classes: self.num_classes,
            loc_preds: Vec::new(),
        })
    }
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
}
