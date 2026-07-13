//! Crate-local error type for OCR text recognition.
//!
//! Follows the workspace convention: a small [`Error`] enum plus a [`Result`]
//! alias. Shared-harness failures (weight loading, preprocessing, CTC, backend)
//! are wrapped from [`mineru_burn_common::Error`] via `#[from]`; dictionary and
//! decode problems get their own variants.

/// Errors originating in the SVTR/CRNN + CTC text recognizer.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A failure in the shared Burn harness: weight loading, key coverage,
    /// preprocessing, or configuration.
    #[error(transparent)]
    Common(#[from] mineru_burn_common::Error),

    /// The character dictionary could not be read or was empty.
    #[error("character dictionary error: {0}")]
    Dict(String),

    /// A configuration value was outside its supported range.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// The logits tensor returned by the network had an unexpected shape.
    #[error("unexpected logits shape: {0}")]
    LogitsShape(String),

    /// An I/O error surfaced while touching a weight file, dictionary, or image.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
