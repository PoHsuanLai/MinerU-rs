//! Crate-local error type.
//!
//! Every crate in the workspace defines its own [`Error`] and a [`Result`] alias;
//! upstream crates wrap this one via `#[from]`.

/// Errors originating in the configuration layer.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Reading the config file failed (missing files are handled as defaults,
    /// so this only surfaces genuine I/O problems).
    #[error("config I/O error")]
    Io(#[from] std::io::Error),

    /// The config file was present but not valid JSON, or didn't match the schema.
    #[error("config parse error")]
    Parse(#[from] serde_json::Error),

    /// A device string (from a file or `MINERU_DEVICE_MODE`) could not be parsed.
    #[error("invalid device specifier: {0:?}")]
    InvalidDevice(String),

    /// A model-source string (from a file or `MINERU_MODEL_SOURCE`) could not be parsed.
    #[error("invalid model source: {0:?}")]
    InvalidModelSource(String),

    /// Downloading a model weight file failed (network error, non-success HTTP
    /// status, or an empty body). See [`crate::download`].
    #[error("failed to download model weights: {0}")]
    Download(String),

    /// A filesystem operation on the models cache/target path failed, or no
    /// writable directory could be resolved. See [`crate::download`].
    #[error("model cache error: {0}")]
    Cache(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
