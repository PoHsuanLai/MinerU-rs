//! The local Burn-model **pipeline backend** for MinerU.
//!
//! This crate is the Rust analogue of Python's `pipeline_analyze.py` +
//! `pipeline_magic_model.py` + `para_split.py`: it composes the local Burn model
//! crates ([`mineru_layout`], [`mineru_ocr_det`], [`mineru_ocr_rec`],
//! [`mineru_formula`], [`mineru_table`]) and [`mineru_pdf`] into a
//! [`mineru_types::Document`], implementing the [`mineru_types::Backend`] trait.
//!
//! # Structure
//! - [`models`] — [`PipelineModels`]: best-effort loading + ownership of the
//!   Burn models under a models directory.
//! - [`analyze`] — [`PipelineBackend`]: opens the PDF, iterates pages **serially**
//!   (PDFium is not concurrency-safe), rasterizes at 200 DPI, runs layout +
//!   per-region recognition, and produces the [`Document`](mineru_types::Document).
//! - [`assemble`] — [`PageAssembler`](assemble::PageAssembler): the pure
//!   `Vec<LayoutDet>` → [`Block`](mineru_types::Block) tree converter (the
//!   `magic_model` analogue). Model-free and unit-tested.
//! - [`para`] — a light paragraph-merging pass (`para_split` analogue).
//! - [`error`] — the crate [`Error`] wrapping each model crate's error.
//!
//! The model crates emit **raw** outputs; the boundary between raw
//! [`LayoutDet`](mineru_layout::LayoutDet) and the typed
//! [`Block`](mineru_types::Block) tree lives entirely in [`assemble`], keeping
//! orchestration and conversion separately testable.
//!
//! # Best-effort degradation
//! Loading and recognition are best-effort: a missing weight file leaves that
//! stage unavailable and the pipeline skips it (still emitting layout structure)
//! rather than failing the whole run. See [`PipelineModels::load`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod analyze;
pub mod assemble;
pub mod error;
pub mod models;
pub mod para;

pub use analyze::PipelineBackend;
pub use assemble::{AssembledPage, PageAssembler, RecognizedLine, Region, RegionContent, RegionKind};
pub use error::{Error, Result};
pub use models::{ModelPaths, PipelineModels};
