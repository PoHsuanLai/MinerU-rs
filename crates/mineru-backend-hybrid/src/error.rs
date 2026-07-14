//! Crate-local error type for the hybrid backend.
//!
//! Follows the workspace convention: this crate defines its own [`Error`] and a
//! [`Result`] alias, wrapping each composed crate's error via `#[from]`. At the
//! [`Backend`](mineru_types::Backend) seam the error is type-erased into
//! [`BackendError`](mineru_types::BackendError) through the std blanket
//! `impl<E: Error + Send + Sync> From<E> for Box<dyn Error + Send + Sync>`: since
//! this [`Error`] derives [`std::error::Error`], `.map_err(Into::into)` at the
//! trait boundary boxes it with no bespoke conversion.
//!
//! No `anyhow` in the library; no `unwrap`/`expect`/`panic!` in `src/`.

/// Errors originating in the hybrid backend.
///
/// Each variant preserves the underlying crate's error (for downcasting or
/// printing) or names a hybrid-specific validation failure, mirroring how the
/// sibling MinerU backend crates wrap their dependencies.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A PDF-input operation failed (open, page metadata, or rasterization).
    #[error("pdf input: {0}")]
    Pdf(#[from] mineru_pdf::Error),

    /// A call to the VLM server (or interpreting its output) failed.
    #[error("vlm client: {0}")]
    Vlm(#[from] mineru_vlm_client::Error),

    /// The layout detector failed on a page.
    #[error("layout detect: {0}")]
    Layout(#[from] mineru_layout::Error),

    /// An unsupported or misspelled parse effort was requested.
    ///
    /// Mirrors the Python `_validate_parse_effort`, which raises
    /// `ValueError('effort must be "medium" or "high"')`.
    #[error("invalid parse effort {0:?}: must be \"medium\" or \"high\"")]
    InvalidEffort(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
