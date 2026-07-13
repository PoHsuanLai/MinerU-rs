//! Wireless-table structure recognition (SLANet-plus), hand-ported to Burn.
//!
//! The Python model is a PP-LCNet CNN backbone feeding a **SLAHead** attention
//! decoder that autoregressively emits HTML structure tokens and, for every
//! `<td>` token, regresses a normalized cell box. Its ONNX graph uses a dynamic
//! sequence-length decoder, which is exactly the case `burn-onnx` codegen is
//! fragile on — so per the crate plan this path is **hand-ported** rather than
//! codegen'd.
//!
//! ## Status
//!
//! The surrounding pipeline is complete and tested end to end on synthetic
//! decoder outputs:
//!
//! 1. [`preprocess`](preprocess::preprocess) — resize/normalize/pad to 488².
//! 2. the CNN + attention decoder forward pass — see [`model`].
//! 3. [`decode`](decode::decode) — argmax token stream + per-`<td>` boxes.
//! 4. [`adapt_slanet_plus`](preprocess::adapt_slanet_plus) — box rescale.
//! 5. [`TableMatch`](crate::matching::TableMatch) — splice OCR text into HTML.
//!
//! Only step 2's exact layer wiring depends on inspecting the real weight tensor
//! shapes; [`model::SlaNet`] carries the architecture skeleton and weight-loading
//! entry point, and returns [`Error::ModelUnavailable`] until wired against real
//! weights. Everything else runs today.

pub mod decode;
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
pub fn recognize_wireless(
    model: &model::SlaNet,
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
