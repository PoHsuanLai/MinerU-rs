//! MinerU — an umbrella crate for the Rust document parser.
//!
//! This crate is two things at once:
//!
//! 1. A **thin re-export facade**: it re-exports the workspace's focused sub-crates
//!    under one module tree, so a downstream user can depend on a single `mineru`
//!    crate instead of wiring up a dozen. Each sub-crate remains independently
//!    publishable, so `cargo add mineru-layout` still works for anyone who wants
//!    only one model.
//! 2. A **builder front door**: [`Mineru::builder`] configures and constructs a
//!    ready-to-use parsing engine in a few lines, so a consumer never has to wire
//!    up [`PipelineModels`](backend::pipeline)/[`PipelineBackend`](backend::pipeline)/
//!    CPU-vs-GPU selection/weight downloading by hand. The engine implements the
//!    [`Backend`](types::Backend) trait, which stays the principled core.
//!
//! # The builder front door
//!
//! ```no_run
//! # #[cfg(feature = "pipeline")]
//! # async fn demo() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! use mineru::types::{Backend, DocInput, ParseOptions};
//!
//! let engine = mineru::Mineru::builder()
//!     .models_dir("/path/to/models") // optional; defaults to Config resolution
//!     .gpu(false)                     // optional; default auto (GPU if available)
//!     .auto_download(true)            // optional; default true — fetch missing weights
//!     .build()?;                      // -> Mineru (impls Backend)
//!
//! let opts = ParseOptions::default();
//! let bytes = std::fs::read("paper.pdf")?;
//! let doc = engine.analyze(DocInput::new(bytes), &opts).await?;
//! # let _ = doc;
//! # Ok(())
//! # }
//! ```
//!
//! # Re-export facade
//!
//! ```no_run
//! use mineru::types::BBox;
//! # #[cfg(feature = "layout")]
//! use mineru::layout::LayoutModel;
//! ```
//!
//! # Environment variables
//!
//! The `MINERU_*` variables are a first-class part of the interface: they steer
//! model/asset resolution and download for both the builder and the binary. Each
//! is honest and optional — unset means the documented default.
//!
//! | Variable | What it does | Default |
//! |---|---|---|
//! | `MINERU_MODELS_DIR` | Root directory holding (or caching) model weights. Overrides the config file. | `dirs::cache_dir()/mineru/models` (e.g. `~/Library/Caches/mineru/models`, `~/.cache/mineru/models`); `./mineru-models` if no cache dir resolves. |
//! | `MINERU_MODELS_BASE` | Base URL that missing pipeline weight files are auto-downloaded from. | Built-in release base URL (`DEFAULT_MODELS_BASE` in `mineru-config`). |
//! | `MINERU_PDFIUM_LIB_PATH` | Explicit path to the PDFium shared library; if the file is absent PDFium is downloaded there. | Unset — probes common system locations, then a cache dir under the models/cache root. |
//! | `MINERU_PDFIUM_DOWNLOAD_BASE` | Base URL PDFium binaries are downloaded from. | Built-in release base URL (`DEFAULT_DOWNLOAD_BASE` in `mineru-pdf`). |
//! | `MINERU_TABLE_WEIGHTS_BASE` | Base URL the table-model `.bpk` weights are fetched from. | Built-in release base URL (`DEFAULT_WEIGHTS_BASE` in `mineru-table`). |
//! | `MINERU_DEVICE_MODE` | Overrides the config's compute device (e.g. `cpu`, `cuda:1`, `mps`). | Config default (`cpu`). |
//! | `MINERU_MODEL_SOURCE` | Overrides where weights are fetched from (e.g. `huggingface`, `modelscope`, a local path). | Config default (`huggingface`). |
//! | `MINERU_TOOLS_CONFIG_JSON` | Path to a JSON config file to load. | Unset — falls back to `~/.mineru.json`, then built-in defaults. |
//! | `HF_HOME` | Hugging Face cache root, honored transitively by the `hf-hub`-based downloaders. | The `hf-hub` default (`~/.cache/huggingface`). |
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

// ---- Builder front door ----------------------------------------------------

mod engine;
pub use engine::{Error, Mineru, MineruBuilder};
/// Loads the pipeline models and boxes the backend (CPU/GPU selection). Shared
/// between the builder facade and the `mineru` binary so there is one copy.
#[cfg(feature = "pipeline")]
pub use engine::build_pipeline_backend;

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
