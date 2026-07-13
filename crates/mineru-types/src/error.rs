//! Crate-local error type.
//!
//! Every crate in the workspace defines its own [`Error`] and a [`Result`] alias;
//! upstream crates wrap this one via `#[from]`.

/// Errors originating in the core type layer.
///
/// Kept deliberately small: this crate is mostly data definitions, so the only
/// fallible operations are constructing values from out-of-range inputs.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A bounding box was built with `x1 < x0` or `y1 < y0`.
    #[error("degenerate bbox: ({x0}, {y0}) .. ({x1}, {y1})")]
    DegenerateBBox { x0: f32, y0: f32, x1: f32, y1: f32 },

    /// A title level exceeded the supported heading depth.
    #[error("title level {0} out of range (max {max})", max = crate::document::MAX_TITLE_LEVEL)]
    TitleLevelOutOfRange(u8),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
