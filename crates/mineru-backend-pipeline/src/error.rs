//! Crate-local error type for the pipeline backend.
//!
//! Follows the workspace convention: every crate defines its own [`Error`] and a
//! [`Result`] alias, wrapping each composed model crate's error via `#[from]`. At
//! the [`Backend`](mineru_types::Backend) seam the error is type-erased into
//! [`BackendError`](mineru_types::BackendError) via the std blanket
//! `impl<E: Error + Send + Sync> From<E> for Box<dyn Error + Send + Sync>`: since
//! this [`Error`] derives [`std::error::Error`], `.map_err(Into::into)` at the
//! trait boundary boxes it with no bespoke conversion.

/// Errors originating in the pipeline backend.
///
/// Each variant wraps the underlying model crate's error so the source is
/// preserved for downcasting or printing, matching how the other MinerU crates
/// wrap their dependencies.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A PDF-input operation failed (open, page metadata, or rasterization).
    #[error("pdf input: {0}")]
    Pdf(#[from] mineru_pdf::Error),

    /// Layout detection failed.
    #[error("layout detection: {0}")]
    Layout(#[from] mineru_layout::Error),

    /// OCR text-line detection failed.
    #[error("ocr detection: {0}")]
    OcrDet(#[from] mineru_ocr_det::Error),

    /// OCR text recognition failed.
    #[error("ocr recognition: {0}")]
    OcrRec(#[from] mineru_ocr_rec::Error),

    /// Formula recognition failed.
    #[error("formula recognition: {0}")]
    Formula(#[from] mineru_formula::Error),

    /// Table recognition failed.
    #[error("table recognition: {0}")]
    Table(#[from] mineru_table::Error),

    /// A required model was not loaded into [`PipelineModels`](crate::PipelineModels).
    ///
    /// The pipeline is best-effort: a missing model file leaves the corresponding
    /// stage unloaded and the stage is skipped rather than failing the whole run.
    /// This variant is only surfaced when a caller explicitly demands a stage that
    /// is unavailable.
    #[error("model `{0}` is not loaded (its weight file was absent at construction)")]
    ModelUnavailable(&'static str),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
