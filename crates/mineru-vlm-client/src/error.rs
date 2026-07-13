//! Crate-local error type.

/// Errors from calling the VLM server or interpreting its output.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The HTTP call to the VLM server failed.
    #[error("VLM request failed: {0}")]
    Request(String),

    /// The server's response could not be parsed into the expected block schema.
    #[error("failed to parse VLM response: {0}")]
    Parse(String),

    /// An image could not be encoded for transport.
    #[error("failed to encode image: {0}")]
    ImageEncode(String),

    /// The configured server URL was missing or invalid.
    #[error("invalid server configuration: {0}")]
    Config(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
