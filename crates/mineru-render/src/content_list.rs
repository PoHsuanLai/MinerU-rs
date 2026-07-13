//! The `content_list.json` renderer — the wire-compatibility boundary.
//!
//! [`ContentItem`] is a *separate* serialization type, kept apart from the
//! in-memory [`Block`] enum on purpose: its field names, tag, and `snake_case`
//! shape must match Python's `content_list.json` exactly, and pinning that shape
//! to the domain model would let an internal refactor silently break the wire
//! format. The mapping from [`Block`] to [`ContentItem`] lives in
//! [`render_content_list`].

use mineru_types::{Block, Captioned, CodeBody, Document, ImageBody, TableBody, TextRole};
use serde::Serialize;

use crate::path::join_image;
use crate::text::{collect_texts, merge_lines};

/// One entry in a `content_list.json` document.
///
/// Tagged externally by a `type` field with `snake_case` variant names to match
/// the Python output. Empty collections and absent optionals are omitted so the
/// JSON matches Python's conditional-key construction.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentItem {
    /// Flowing text, optionally a heading (`text_level` present for titles).
    Text {
        /// The merged text of the block.
        text: String,
        /// Heading depth; absent for body text and for the doc title (level 0),
        /// matching Python which only sets `text_level` for `level != 0`.
        #[serde(skip_serializing_if = "Option::is_none")]
        text_level: Option<u8>,
    },
    /// A standalone equation, carried as LaTeX.
    Equation {
        /// The LaTeX source.
        text: String,
        /// Always `"latex"`; names the format of `text`.
        text_format: String,
    },
    /// A figure with its captions and footnotes.
    Image {
        /// The resolved image path.
        img_path: String,
        /// Caption lines, in order.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        image_caption: Vec<String>,
        /// Footnote lines, in order.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        image_footnote: Vec<String>,
    },
    /// A chart with its captions and footnotes.
    Chart {
        /// The resolved image path.
        img_path: String,
        /// Caption lines, in order.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        chart_caption: Vec<String>,
        /// Footnote lines, in order.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        chart_footnote: Vec<String>,
    },
    /// A table: its HTML body, an optional cropped raster, captions and footnotes.
    Table {
        /// The resolved path of the table's cropped raster, if any.
        #[serde(skip_serializing_if = "String::is_empty")]
        img_path: String,
        /// The table's HTML markup.
        table_body: String,
        /// Caption lines, in order.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        table_caption: Vec<String>,
        /// Footnote lines, in order.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        table_footnote: Vec<String>,
    },
    /// A code block: its text body, captions and footnotes.
    Code {
        /// The code text (lines joined with newlines).
        code_body: String,
        /// The language tag, if known.
        #[serde(skip_serializing_if = "Option::is_none")]
        code_language: Option<String>,
        /// Caption lines, in order.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        code_caption: Vec<String>,
        /// Footnote lines, in order.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        code_footnote: Vec<String>,
    },
}

/// Renders a document to a `content_list.json`-compatible vector.
///
/// `image_dir` resolves image references (same joining as the Markdown
/// renderer). Blocks with discard-y text roles produce no entry.
pub fn render_content_list(doc: &Document, image_dir: &str) -> Vec<ContentItem> {
    doc.blocks()
        .filter_map(|block| content_item(block, image_dir))
        .collect()
}

/// Maps one block to at most one [`ContentItem`].
///
/// Exhaustive over [`Block`] for the same reason as the Markdown renderer: a new
/// variant must be handled here before it compiles.
fn content_item(block: &Block, image_dir: &str) -> Option<ContentItem> {
    match block {
        Block::Text { role, lines, .. } => text_item(*role, merge_lines(lines)),
        Block::InterlineEquation { latex, .. } => Some(ContentItem::Equation {
            text: latex.to_string(),
            text_format: "latex".to_owned(),
        }),
        Block::Image(c) => Some(image_item(c, image_dir)),
        Block::Chart(c) => Some(chart_item(c, image_dir)),
        Block::Table(c) => Some(table_item(c, image_dir)),
        Block::Code(c) => Some(code_item(c)),
    }
}

/// Builds a text/heading item, dropping discard-y roles.
fn text_item(role: TextRole, text: String) -> Option<ContentItem> {
    match role {
        TextRole::Title(level) => Some(ContentItem::Text {
            text,
            // Python omits `text_level` for the doc title (level 0).
            text_level: (level.0 != 0).then_some(level.0),
        }),
        TextRole::Header
        | TextRole::Footer
        | TextRole::PageNumber
        | TextRole::PageFootnote
        | TextRole::AsideText => None,
        TextRole::Body
        | TextRole::List
        | TextRole::Index
        | TextRole::Abstract
        | TextRole::RefText => Some(ContentItem::Text {
            text,
            text_level: None,
        }),
    }
}

/// Builds an image item.
fn image_item(c: &Captioned<ImageBody>, image_dir: &str) -> ContentItem {
    ContentItem::Image {
        img_path: join_image(image_dir, c.body.image.as_str()),
        image_caption: collect_texts(&c.captions),
        image_footnote: collect_texts(&c.footnotes),
    }
}

/// Builds a chart item.
fn chart_item(c: &Captioned<ImageBody>, image_dir: &str) -> ContentItem {
    ContentItem::Chart {
        img_path: join_image(image_dir, c.body.image.as_str()),
        chart_caption: collect_texts(&c.captions),
        chart_footnote: collect_texts(&c.footnotes),
    }
}

/// Builds a table item; `img_path` is empty when the table has no cropped raster.
fn table_item(c: &Captioned<TableBody>, image_dir: &str) -> ContentItem {
    let img_path = c
        .body
        .image
        .as_ref()
        .map(|r| join_image(image_dir, r.as_str()))
        .unwrap_or_default();
    ContentItem::Table {
        img_path,
        table_body: c.body.html.to_string(),
        table_caption: collect_texts(&c.captions),
        table_footnote: collect_texts(&c.footnotes),
    }
}

/// Builds a code item; code lines join with newlines to preserve layout.
fn code_item(c: &Captioned<CodeBody>) -> ContentItem {
    let code_body = c
        .body
        .lines
        .iter()
        .map(crate::text::flatten_line)
        .collect::<Vec<_>>()
        .join("\n");
    ContentItem::Code {
        code_body,
        code_language: c.body.language.as_ref().map(|l| l.to_string()),
        code_caption: collect_texts(&c.captions),
        code_footnote: collect_texts(&c.footnotes),
    }
}
