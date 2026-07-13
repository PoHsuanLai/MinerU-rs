//! Crate-local error type for PDF input.
//!
//! Follows the workspace convention: every crate defines its own [`Error`] and a
//! [`Result`] alias; upstream crates wrap this one via `#[from]`. Pdfium's own
//! errors are carried as strings because [`pdfium_render::prelude::PdfiumError`]
//! is not `Send + Sync + 'static` in a form we want to leak into our public API.

/// Errors originating in the PDF input layer.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The Pdfium native library could not be bound (missing/incompatible
    /// `libpdfium` dynamic library, or an unsupported platform).
    #[error("failed to bind Pdfium library: {0}")]
    Bind(String),

    /// The PDF bytes could not be parsed into a document.
    #[error("failed to open PDF: {0}")]
    Open(String),

    /// A page index was requested that the document does not contain.
    #[error("page index {index} out of range (document has {count} pages)")]
    PageIndexOutOfRange {
        /// The out-of-range index that was requested.
        index: usize,
        /// The number of pages the document actually has.
        count: usize,
    },

    /// A page failed to rasterize.
    //
    // Field is named `message` (not `source`) deliberately: `thiserror` treats a
    // field literally named `source` as a nested `std::error::Error`, which a
    // `String` is not. We carry Pdfium's error as a plain message string.
    #[error("failed to render page {page}: {message}")]
    Render {
        /// Zero-based index of the page that failed.
        page: usize,
        /// The underlying Pdfium error message.
        message: String,
    },
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
