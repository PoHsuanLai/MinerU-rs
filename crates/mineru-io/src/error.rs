//! Crate-local error type.
//!
//! Every crate in the workspace defines its own [`Error`] and a [`Result`] alias;
//! upstream crates wrap this one via `#[from]`.

/// Errors originating in the I/O layer.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An underlying filesystem operation failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A relative path attempted to escape its base directory (e.g. via a
    /// `..` component or an absolute path). The offending path is included.
    #[error("path escapes base directory: {0}")]
    PathEscape(String),

    /// A model download failed. The message carries the underlying cause.
    #[error("model download failed: {0}")]
    Download(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
