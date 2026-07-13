//! Crate-local error type for OCR text detection.
//!
//! Follows the workspace convention: a small [`Error`] enum plus a [`Result`]
//! alias. Shared-harness failures (weight loading, preprocessing, backend) are
//! wrapped from [`mineru_burn_common::Error`] via `#[from]`; post-processing
//! geometry problems get their own variants.

/// Errors originating in the DBNet text detector.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A failure in the shared Burn harness: weight loading, key coverage,
    /// preprocessing, or configuration.
    #[error(transparent)]
    Common(#[from] mineru_burn_common::Error),

    /// A configuration value was outside its supported range.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// The probability map returned by the network had an unexpected shape.
    #[error("unexpected probability-map shape: {0}")]
    ProbMapShape(String),

    /// An I/O error surfaced while touching a weight file or image.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
