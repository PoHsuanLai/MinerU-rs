//! Crate-local error type.
//!
//! Follows the workspace convention: each crate owns a small [`Error`] enum and a
//! [`Result`] alias. No `unwrap`/`expect`/`panic` is used in library code; every
//! fallible path returns one of these variants.

/// Errors originating in the table-recognition crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A model's weights are not available for inference (e.g. the fetch has not
    /// been attempted, or the mask → polygon post-processing that consumes the
    /// forward output is not yet ported). The network itself is always compiled;
    /// pure post-processing still works. Fetch/load failures use the more
    /// specific [`Error::WeightFetch`]/[`Error::Cache`]/[`Error::WeightLoad`].
    #[error("model `{0}` is not available")]
    ModelUnavailable(&'static str),

    /// Downloading a model's `.bpk` weights from the release failed (network
    /// error, non-200 HTTP status, or a SHA-256 mismatch on the fetched bytes).
    #[error("failed to fetch model weights: {0}")]
    WeightFetch(String),

    /// Resolving or writing the on-disk weight cache failed (no writable cache
    /// directory could be determined, or a filesystem I/O error occurred).
    #[error("weight cache error: {0}")]
    Cache(String),

    /// An input image had an unusable size for a model's fixed input geometry.
    #[error("image too small for {model}: got {width}x{height}, need at least {min_width}x{min_height}")]
    ImageTooSmall {
        /// Name of the model that rejected the image.
        model: &'static str,
        /// Actual image width.
        width: u32,
        /// Actual image height.
        height: u32,
        /// Minimum acceptable width.
        min_width: u32,
        /// Minimum acceptable height.
        min_height: u32,
    },

    /// A model produced tensor output whose shape did not match expectations.
    #[error("unexpected model output shape: expected {expected}, got {got}")]
    OutputShape {
        /// Human-readable description of the expected shape.
        expected: String,
        /// Human-readable description of the observed shape.
        got: String,
    },

    /// A structure-token stream could not be decoded (e.g. empty or malformed).
    #[error("failed to decode table structure: {0}")]
    Decode(String),

    /// Loading model weights from a file failed (missing/corrupt file, or a
    /// key/shape mismatch against the module under the strict coverage check).
    #[error("failed to load model weights: {0}")]
    WeightLoad(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
