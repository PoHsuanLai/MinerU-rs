//! Vendored, machine-generated Burn modules.
//!
//! These two modules are the LCNet table classifier and the UNet line
//! segmenter, emitted verbatim by `burn-onnx` from the PDF-Extract-Kit ONNX
//! exports and committed to the tree (formerly generated at build time under the
//! `onnx-import` feature). They are compiled unconditionally now; their `.bpk`
//! weights are fetched at runtime by [`crate::weights`]. Do not hand-edit either
//! file — regenerate with `burn-onnx` and replace it wholesale (see each file's
//! header).

pub mod lcnet;
pub mod unet;
