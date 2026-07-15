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
//! ## Models and weights
//!
//! The LCNet classifier and UNet segmenter are Burn networks committed to the
//! tree under [`generated`] (vendored `burn-onnx` output from the PDF-Extract-Kit
//! ONNX exports); SLANet-plus is hand-ported. All three are always compiled in.
//! Their weights are *not* embedded: the two generated models' `.bpk` files are
//! fetched once from a public GitHub release and cached on disk by [`weights`] on
//! first use (see that module for the cache location and the
//! `MINERU_TABLE_WEIGHTS_BASE`/`MINERU_MODELS_DIR` overrides). All pure
//! pre/post-processing (structure decode, OCR↔cell matching, grid recovery, HTML
//! assembly) works with no network access and is fully unit-tested; only the
//! neural forward paths need the fetched weights, and they surface a typed
//! [`Error::WeightFetch`]/[`Error::WeightLoad`]/[`Error::Cache`] on failure.
//!
//! ## Output
//!
//! Every recognizer returns raw table markup as [`mineru_types::Html`]; assembling
//! it into the document [`Block`](mineru_types::Block) tree happens elsewhere.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod cls;
pub mod error;
pub mod inline;
pub mod matching;
mod model_cache;
pub mod ocr;
pub mod orientation;
pub mod select;
pub mod slanet;
pub mod unet;
pub mod weights;

// Vendored, machine-generated Burn modules for the LCNet classifier and the UNet
// segmenter (formerly generated into `$OUT_DIR` at build time; now committed
// under `src/generated/` and compiled unconditionally). Do not hand-edit —
// regenerate with `burn-onnx` and replace the files wholesale.
mod generated;

pub use cls::{classify, Classification, TableClass};
pub use error::{Error, Result};
pub use inline::{assign_to_tables, mask_boxes, Assignment, PageFormula, TableFormula};
pub use matching::{decode_logic_points, LogicPoint, TableMatch};
pub use ocr::OcrSpan;
pub use orientation::{is_rotation_candidate, sample_boxes, select_rotation, OrientationScore, Rotation};
pub use select::{select, Choice, WIRELESS_TRUST_THRESHOLD};
pub use slanet::recognize_wireless;
pub use unet::recognize_wired;
