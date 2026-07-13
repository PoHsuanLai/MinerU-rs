//! Crate-local error type.
//!
//! Every crate in the workspace defines its own [`Error`] and a [`Result`] alias.
//! Rendering the document tree to Markdown or `content_list` is *infallible* —
//! the public renderers return plain `String`/`Vec` — so this module exists only
//! to keep the per-crate convention and to give downstream code a stable place to
//! wrap a failure should a fallible rendering path ever be added (e.g. writing
//! output to disk).

/// Errors originating in the render layer.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Serializing a [`ContentItem`](crate::ContentItem) to JSON failed.
    ///
    /// Not produced by the in-memory renderers; reserved for callers that hand a
    /// [`ContentItem`](crate::ContentItem) to `serde_json` and want to funnel the
    /// failure through this crate's `Result`.
    #[error("failed to serialize content_list to JSON: {0}")]
    Json(#[from] serde_json::Error),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
