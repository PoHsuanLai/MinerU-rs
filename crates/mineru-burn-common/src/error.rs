//! Crate-local error type.
//!
//! Follows the workspace convention: each crate owns a small [`Error`] enum and a
//! [`Result`] alias; upstream crates wrap this one via `#[from]`.

/// Errors originating in the shared Burn harness.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A weight file could not be read or its records could not be applied to a
    /// module. Carries the underlying store error rendered as a string, because
    /// `burn-store`'s per-format error types are not `std::error::Error`.
    #[error("failed to load weights: {0}")]
    WeightLoad(String),

    /// Weight loading completed but some source tensors were never matched to a
    /// module field. This almost always means a key-remap rule is missing and the
    /// model is running with partially random weights, so it is treated as an error.
    #[error("weight load left {n} source key(s) unmapped: {keys:?}", n = keys.len())]
    UnmappedKeys {
        /// The source tensor keys that were never matched to a module field.
        keys: Vec<String>,
    },

    /// A tensor did not have the shape a helper required.
    #[error("shape mismatch: expected {expected}, got {got}")]
    Shape {
        /// Human-readable description of the expected shape.
        expected: String,
        /// Human-readable description of the shape that was actually seen.
        got: String,
    },

    /// A configuration value was outside its supported range.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// An I/O error surfaced while touching a weight file or image.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
