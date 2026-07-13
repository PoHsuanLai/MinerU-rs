//! Table recognition for MinerU, ported to Burn with **no** onnxruntime.
//!
//! A detected table crop is first classified as *wired* (ruled) or *wireless*
//! (borderless) by a PP-LCNet classifier, then recognized by the matching engine:
//!
//! - **Wireless** → [`slanet`]: a hand-ported SLANet-plus CNN + attention decoder
//!   produces an HTML structure-token stream plus a regressed box per `<td>`;
//!   OCR spans are matched onto those cells and spliced into the HTML.
//! - **Wired** → [`unet`]: a UNet segments the ruling lines, classical recovery
//!   turns the mask into cell polygons, the logical grid is inferred, and OCR
//!   text is rendered into `<table>` HTML with `rowspan`/`colspan`.
//!
//! ## Model availability
//!
//! The neural networks are optional. The LCNet classifier and UNet segmenter are
//! generated from their `.onnx` files at build time behind the `onnx-import`
//! cargo feature; SLANet-plus is hand-ported and loaded from a weight file. When
//! a model's weights are absent the corresponding `recognize_*`/`classify` entry
//! point returns [`Error::ModelUnavailable`], while all pure pre/post-processing
//! (structure decode, OCR↔cell matching, grid recovery, HTML assembly) works
//! unconditionally and is fully unit-tested. This keeps the crate building — and
//! its logic testable — with no model files and no network access.
//!
//! ## Output
//!
//! Every recognizer returns raw table markup as [`mineru_types::Html`]; assembling
//! it into the document [`Block`](mineru_types::Block) tree happens elsewhere.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod cls;
pub mod error;
pub mod matching;
pub mod ocr;
pub mod slanet;
pub mod unet;

// Build-time generated Burn modules (present only under the `onnx-import`
// feature with the corresponding `.onnx` files). Guarded by cfgs the build
// script emits so the crate compiles with or without them.
#[cfg(any(lcnet_generated, unet_generated))]
mod model {
    #[cfg(lcnet_generated)]
    pub mod pp_lcnet_x1_0_table_cls {
        include!(concat!(env!("OUT_DIR"), "/model/pp_lcnet_x1_0_table_cls.rs"));
    }
    #[cfg(unet_generated)]
    pub mod unet {
        include!(concat!(env!("OUT_DIR"), "/model/unet.rs"));
    }
}

pub use cls::{classify, Classification, TableClass};
pub use error::{Error, Result};
pub use matching::{decode_logic_points, LogicPoint, TableMatch};
pub use ocr::OcrSpan;
pub use slanet::recognize_wireless;
pub use unet::recognize_wired;
