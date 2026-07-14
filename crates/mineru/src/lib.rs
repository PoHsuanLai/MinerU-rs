//! MinerU — an umbrella crate for the Rust document parser.
//!
//! This crate is a thin facade: it re-exports the workspace's focused sub-crates
//! under one module tree, so a downstream user can depend on a single `mineru`
//! crate instead of wiring up a dozen. Each sub-crate remains independently
//! publishable, so `cargo add mineru-layout` still works for anyone who wants only
//! one model.
//!
//! ```no_run
//! use mineru::types::BBox;
//! # #[cfg(feature = "layout")]
//! use mineru::layout::LayoutModel;
//! ```
//!
//! # Features
//!
//! The heavy model and backend crates are optional so a caller only compiles what
//! they use — a VLM-only user never pulls the Burn deep-learning stack.
//!
//! - **`vlm`** — the OpenAI-compatible VLM backend ([`vlm_client`], [`backend::vlm`]).
//! - **`pipeline`** — the fully-local Burn pipeline backend ([`backend::pipeline`])
//!   and every model it composes (implies `ocr`, `layout`, `table`, `formula`).
//! - **`hybrid`** — the hybrid backend ([`backend::hybrid`]): pipeline layout drives
//!   per-region VLM extraction (implies `pipeline` + `vlm`).
//! - **`ocr`**, **`layout`**, **`table`**, **`formula`**, **`burn_common`** — pull a
//!   single model crate.
//! - **`cli`** (default) — builds the `mineru` binary; implies `pipeline` + `vlm`.
//!
//! The foundation crates ([`types`], [`config`], [`io`], [`pdf`], [`render`]) are
//! always available.

// ---- Foundation (always present) -------------------------------------------

/// Core domain types and the [`Backend`](mineru_types::Backend) trait
/// (re-export of `mineru-types`).
pub use mineru_types as types;
/// User configuration mirroring `mineru.json` (re-export of `mineru-config`).
pub use mineru_config as config;
/// Reader/writer abstractions + model download (re-export of `mineru-io`).
pub use mineru_io as io;
/// PDF rasterize / text-extract / repair (re-export of `mineru-pdf`).
pub use mineru_pdf as pdf;
/// Blocks → Markdown / content-list rendering (re-export of `mineru-render`).
pub use mineru_render as render;

// ---- Model crates (feature-gated) ------------------------------------------

/// Shared Burn harness: device init, weight loading, common NN blocks
/// (re-export of `mineru-burn-common`; feature `burn-common`).
#[cfg(feature = "burn-common")]
pub use mineru_burn_common as burn_common;

/// DBNet text-line detection (re-export of `mineru-ocr-det`; feature `ocr`).
#[cfg(feature = "ocr")]
pub use mineru_ocr_det as ocr_det;
/// SVTR + CTC text recognition (re-export of `mineru-ocr-rec`; feature `ocr`).
#[cfg(feature = "ocr")]
pub use mineru_ocr_rec as ocr_rec;

/// PP-DocLayoutV2 layout detection (re-export of `mineru-layout`; feature `layout`).
#[cfg(feature = "layout")]
pub use mineru_layout as layout;

/// SLANet/UNet/LCNet table recognition (re-export of `mineru-table`; feature `table`).
#[cfg(feature = "table")]
pub use mineru_table as table;

/// UniMerNet formula recognition (re-export of `mineru-formula`; feature `formula`).
#[cfg(feature = "formula")]
pub use mineru_formula as formula;

/// OpenAI-compatible VLM client (re-export of `mineru-vlm-client`; feature `vlm`).
#[cfg(feature = "vlm")]
pub use mineru_vlm_client as vlm_client;

/// The composed parsing backends, each implementing
/// [`Backend`](mineru_types::Backend).
pub mod backend {
    /// Fully-local Burn pipeline backend (re-export of `mineru-backend-pipeline`;
    /// feature `pipeline`).
    #[cfg(feature = "pipeline")]
    pub use mineru_backend_pipeline as pipeline;
    /// External VLM backend (re-export of `mineru-backend-vlm`; feature `vlm`).
    #[cfg(feature = "vlm")]
    pub use mineru_backend_vlm as vlm;
    /// Hybrid backend: pipeline layout drives per-region VLM extraction
    /// (re-export of `mineru-backend-hybrid`; feature `hybrid`).
    #[cfg(feature = "hybrid")]
    pub use mineru_backend_hybrid as hybrid;
}
