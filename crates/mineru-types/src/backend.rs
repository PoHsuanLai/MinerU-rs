//! The [`Backend`] trait: the single seam every parsing engine implements.
//!
//! `vlm`, `pipeline`, and `hybrid` backends each `impl Backend`, and the CLI holds
//! a `Box<dyn Backend>`. Kept small and object-safe on purpose.

use async_trait::async_trait;

use crate::content::Lang;
use crate::document::Document;

/// Input to a backend: the raw document bytes plus how they should be parsed.
///
/// Rasterization and native-text extraction live in `mineru-pdf`; a backend that
/// needs page images calls into it. Passing bytes (not pre-rendered pages) keeps
/// this trait independent of the pdf crate.
#[derive(Debug, Clone)]
pub struct DocInput {
    /// The source document bytes (PDF; images are pre-wrapped into a PDF upstream).
    pub bytes: Vec<u8>,
}

impl DocInput {
    /// Wraps document bytes.
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }
}

/// Options controlling a parse run. Small and `Default`-able; call sites tweak
/// individual fields rather than going through a builder.
#[derive(Debug, Clone)]
pub struct ParseOptions {
    /// Language hint for OCR; `None` lets the backend auto-detect.
    pub lang: Option<Lang>,
    /// Whether to recognize formulas.
    pub formula: bool,
    /// Whether to recognize tables.
    pub table: bool,
    /// Inclusive start / exclusive-ish end page range; `None` means all pages.
    pub page_range: Option<(usize, Option<usize>)>,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self {
            lang: None,
            formula: true,
            table: true,
            page_range: None,
        }
    }
}

/// A boxed, type-erased error carried across the [`Backend`] seam.
///
/// Each backend crate keeps its own rich `Error` enum internally and converts to
/// this at the trait boundary, so `Box<dyn Backend>` stays object-safe while
/// callers can still downcast or print the underlying error.
pub type BackendError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// A document-parsing engine.
///
/// Object-safe so backends can be selected at runtime as `Box<dyn Backend>`.
/// Uses [`async_trait`] because native async-fn-in-trait is not yet object-safe.
/// The error is type-erased to [`BackendError`] so differing backend error types
/// can share one `dyn` seam.
#[async_trait]
pub trait Backend: Send + Sync {
    /// Parses a document into the canonical [`Document`] tree.
    async fn analyze(
        &self,
        input: DocInput,
        opts: &ParseOptions,
    ) -> std::result::Result<Document, BackendError>;
}
