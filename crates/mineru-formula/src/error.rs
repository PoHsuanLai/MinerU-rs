//! Crate-local error type.
//!
//! Follows the workspace convention: each crate owns a small [`Error`] enum and a
//! [`Result`] alias. Errors from the shared Burn harness ([`mineru_burn_common`])
//! and from the tokenizer/image stacks are wrapped via `#[from]` / dedicated
//! variants so callers funnel every formula-recognition failure through one type.

/// Errors originating in the formula-recognition crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Weight loading (via the shared harness) failed — a missing file, a tensor
    /// that would not apply, or (under strict coverage) unmapped checkpoint keys.
    #[error("weight load failed: {0}")]
    Weights(#[from] mineru_burn_common::error::Error),

    /// The tokenizer file could not be loaded, or decoding token ids failed.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// Decoding or resizing the input image failed.
    #[error("image error: {0}")]
    Image(String),

    /// A configuration value was outside its supported range, or a config file
    /// could not be parsed.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// A model invariant was violated at runtime (an unexpected tensor shape, an
    /// empty generation, etc.). Indicates a bug in this crate or a corrupt input.
    #[error("model error: {0}")]
    Model(String),

    /// An I/O error surfaced while reading a model or tokenizer file.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
