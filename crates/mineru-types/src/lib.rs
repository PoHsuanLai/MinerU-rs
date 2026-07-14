//! Core domain types for the MinerU document parser.
//!
//! This crate defines the canonical parsed-document tree ([`Document`] and friends),
//! the shared geometry primitive ([`BBox`]), and the [`Backend`] trait that every
//! parsing engine implements. It is the foundation every other crate depends on and
//! pulls in no heavy dependencies.
//!
//! The document model is deliberately *not* a transliteration of Python's
//! dynamically-typed `middle_json` dict: block and span kinds are enums with
//! variant-specific payloads, so illegal states are unrepresentable and renderers
//! are exhaustively checked by the compiler.

pub mod backend;
pub mod content;
pub mod document;
pub mod error;
pub mod geom;
pub mod image_sink;

pub use backend::{Backend, BackendError, DocInput, ParseOptions};
pub use content::{Html, ImageRef, Lang, Latex, Score};
pub use image_sink::ImageWriter;
pub use document::{
    Block, Captioned, CodeBody, Document, ImageBody, Page, PageSize, Span, TableBody, TextBlock,
    TextLine, TextRole, TitleLevel, MAX_TITLE_LEVEL,
};
pub use error::{Error, Result};
pub use geom::BBox;
