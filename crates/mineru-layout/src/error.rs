//! Crate-local error type.
//!
//! Follows the workspace convention: each crate owns a small [`Error`] enum and a
//! [`Result`] alias. The shared Burn-harness error from [`mineru_burn_common`] is
//! wrapped via `#[from]` so weight-loading and preprocessing failures surface
//! through this crate's error without losing their detail.

/// Errors originating in the layout-detection crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A failure from the shared Burn harness: weight loading, key remapping, or
    /// image preprocessing. Carries the underlying [`mineru_burn_common::Error`].
    #[error(transparent)]
    Common(#[from] mineru_burn_common::Error),

    /// The image could not be decoded or was otherwise unusable.
    #[error("image error: {0}")]
    Image(String),

    /// A tensor produced by the model did not have the shape the postprocessor
    /// required. This should never happen with a correctly loaded checkpoint, so
    /// it is surfaced rather than silently producing garbage detections.
    #[error("unexpected tensor shape: {0}")]
    Shape(String),

    /// A configuration value was outside its supported range.
    #[error("invalid configuration: {0}")]
    Config(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
