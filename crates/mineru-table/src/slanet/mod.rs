//! Wireless-table structure recognition (SLANet-plus), hand-ported to Burn.
//!
//! The Python model is a PP-LCNet CNN backbone + CSP-PAN neck feeding a
//! **SLAHead** attention decoder that autoregressively emits HTML structure
//! tokens and, for every `<td>` token, regresses a normalized cell box. Its ONNX
//! graph exports the decoder as a dynamic-length `Loop`, which `burn-onnx` 0.21
//! cannot import (its type inference fails on the pre-loop `ConstantOfShape`
//! nodes — see [`model`]), so the whole network is **hand-ported** rather than
//! codegen'd, and its weights load at runtime from a converted `.safetensors`.
//!
//! ## Pipeline
//!
//! 1. [`preprocess`](preprocess::preprocess) — resize/normalize/pad to 488².
//! 2. the forward pass — [`backbone`] (PP-LCNet + CSP-PAN) → [`head`] (attention
//!    GRU decoder) → structure probabilities + cell boxes; orchestrated by
//!    [`model::SlaNet`].
//! 3. [`decode`](decode::decode) — argmax token stream + per-`<td>` boxes.
//! 4. [`adapt_slanet_plus`](preprocess::adapt_slanet_plus) — box rescale.
//! 5. [`TableMatch`](crate::matching::TableMatch) — splice OCR text into HTML.
//!
//! [`model::SlaNet`] loads real weights and runs the full forward pass. When the
//! weight file is absent it returns [`Error::ModelUnavailable`] so the pipeline
//! degrades gracefully; all pure pre/post-processing runs unconditionally.

pub mod backbone;
pub mod decode;
pub mod head;
pub mod model;
pub mod preprocess;
pub mod vocab;

pub use decode::StructureResult;
pub use preprocess::{adapt_slanet_plus, preprocess, Preprocessed, TABLE_MAX_LEN};
pub use vocab::{build_vocab, Vocab};

use image::RgbImage;
use mineru_types::Html;

use crate::error::Result;
use crate::matching::TableMatch;
use crate::ocr::OcrSpan;

/// Recognizes a wireless table crop into HTML by matching OCR spans onto the
/// SLANet-predicted structure.
///
/// Requires a loaded [`model::SlaNet`]; when the crate is built without weights
/// this returns [`crate::error::Error::ModelUnavailable`]. The pure decode/match
/// steps are exercised directly in tests without a model.
pub fn recognize_wireless<B: burn::prelude::Backend>(
    model: &model::SlaNet<B>,
    img: &RgbImage,
    spans: &[OcrSpan],
) -> Result<Html> {
    let pre = preprocess(img);
    let raw = model.forward(&pre)?;
    let vocab = build_vocab();
    let mut structure = decode::decode(&raw.as_preds(), &vocab, pre.orig_w, pre.orig_h);
    adapt_slanet_plus(pre.orig_w, pre.orig_h, &mut structure.cell_bboxes);
    let html = TableMatch::default().run(&structure.tokens, &structure.cell_bboxes, spans);
    Ok(html)
}
