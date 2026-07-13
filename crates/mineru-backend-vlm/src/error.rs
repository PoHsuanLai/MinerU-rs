//! Crate-local error type for the VLM backend.
//!
//! Follows the workspace convention: every crate defines its own [`Error`] and a
//! [`Result`] alias, wrapping each composed crate's error via `#[from]`. At the
//! [`Backend`](mineru_types::Backend) seam the error is type-erased into
//! [`BackendError`](mineru_types::BackendError) via the std blanket
//! `impl<E: Error + Send + Sync> From<E> for Box<dyn Error + Send + Sync>`: since
//! this [`Error`] derives [`std::error::Error`], `.map_err(Into::into)` at the
//! trait boundary boxes it with no bespoke conversion.

/// Errors originating in the VLM backend.
///
/// Each variant wraps the underlying crate's error so the source is preserved for
/// downcasting or printing, matching how the other MinerU backend crates wrap
/// their dependencies.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A PDF-input operation failed (open, page metadata, or rasterization).
    #[error("pdf input: {0}")]
    Pdf(#[from] mineru_pdf::Error),

    /// A call to the VLM server (or interpreting its output) failed.
    #[error("vlm client: {0}")]
    Vlm(#[from] mineru_vlm_client::Error),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
