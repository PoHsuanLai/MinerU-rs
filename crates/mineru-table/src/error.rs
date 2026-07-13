//! Crate-local error type.
//!
//! Follows the workspace convention: each crate owns a small [`Error`] enum and a
//! [`Result`] alias. No `unwrap`/`expect`/`panic` is used in library code; every
//! fallible path returns one of these variants.

/// Errors originating in the table-recognition crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A model was requested but the crate was built without the `onnx-import`
    /// feature (or without the corresponding `.onnx` file), so no weights are
    /// compiled in. Inference is unavailable; pure post-processing still works.
    #[error("model `{0}` is not available: build with the `onnx-import` feature and the ONNX files present")]
    ModelUnavailable(&'static str),

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
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
