//! Formula recognition for MinerU: **UniMerNet** ported to Rust + [Burn].
//!
//! UniMerNet is a Donut-style vision-encoder-decoder that reads a cropped formula
//! image and autoregressively emits LaTeX:
//!
//! - a **Swin-Transformer encoder** ([`swin`]) with UniMerNet's overlapping-conv
//!   stem patch embedding and depth-wise `ConvEnhance` blocks, producing a grid of
//!   visual tokens;
//! - an **MBart decoder** ([`mbart`]) with *squeeze attention*, causal self-
//!   attention, cross-attention over the encoder grid, and an LM head;
//! - a **greedy autoregressive decode loop** ([`generate`]) from the BOS token to
//!   EOS or a length cap.
//!
//! The public entry point is [`FormulaRecognizer::predict`], which takes an
//! [`image::RgbImage`] of a cropped formula and returns a [`mineru_types::Latex`].
//!
//! This crate is a faithful *architecture* translation of the Python reference at
//! `mineru/model/mfr/unimernet/`. See the crate README-in-code notes and each
//! module's docs for exactly what is COMPLETE, STUBBED, or UNCERTAIN — in
//! particular, [`generate`] currently runs the correct **non-cached** loop
//! (KV-cache is a noted optimization), and real weight loading is exercised only
//! behind `#[ignore]`d tests because the checkpoint is a multi-hundred-MB download.
//!
//! [Burn]: https://burn.dev
//!
//! # Backend
//! Backends come from [`mineru_burn_common::backend`] ([`Cpu`] by default). The
//! model graph is generic over `B: Backend`, so swapping to a GPU backend is a
//! type change at the call site.

#![warn(missing_docs)]

pub mod config;
pub mod error;
pub mod generate;
pub mod latex_cleanup;
pub mod mbart;
pub mod model;
pub mod preprocess;
pub mod swin;
pub mod tokenizer;
pub mod weights;

pub use config::{MBartConfig, SwinConfig, UniMerNetConfig};
pub use error::{Error, Result};
pub use model::{FormulaRecognizer, UniMerNet};

/// The default CPU backend, re-exported from the shared harness for convenience.
pub use mineru_burn_common::backend::Cpu;
