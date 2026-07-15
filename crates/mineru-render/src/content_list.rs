//! The `content_list.json` renderer — the wire-compatibility boundary.
//!
//! [`ContentItem`]/[`ContentBody`] are *separate* serialization types, kept apart
//! from the in-memory [`Block`] enum on purpose: their field names, tag, and
//! `snake_case` shape must match Python's `content_list.json` exactly, and pinning
//! that shape to the domain model would let an internal refactor silently break
//! the wire format. The mapping from [`Block`] to [`ContentItem`] lives in
//! [`render_content_list`].

use mineru_types::{
    BBox, Block, Captioned, CodeBody, Document, ImageBody, PageSize, TableBody, TextRole,
};
use serde::Serialize;

use crate::path::join_image;
use crate::text::{collect_texts, merge_lines};

/// The edge length every `bbox` is normalized against, matching Python's 0–1000
/// coordinate mapping (`_build_bbox`).
const BBOX_SCALE: f32 = 1000.0;

/// One entry in a `content_list.json` document: its type-specific payload plus
/// the locator fields (`bbox`, `page_idx`) every entry carries.
///
/// The payload is `#[serde(flatten)]`ed, so this serializes to a single flat
/// object (`{"type": "text", "text": …, "bbox": …, "page_idx": 0}`) — matching
/// Python, which appends the same two keys to every item regardless of type.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ContentItem {
    /// The type-tagged, type-specific content.
    #[serde(flatten)]
    pub content: ContentBody,
    /// The block's box, normalized to `0..=1000` against the page size. Absent
    /// when the page has no usable size (matching Python's `_build_bbox`, which
    /// returns `None` and omits the key).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox: Option<[i32; 4]>,
    /// Zero-based index of the page this entry came from.
    pub page_idx: usize,
}

/// The type-specific body of a [`ContentItem`].
///
/// Tagged internally by a `type` field with `snake_case` variant names to match
/// the Python output. Empty collections and absent optionals are omitted so the
/// JSON matches Python's conditional-key construction.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBody {
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
    /// A list: a run of adjacent items rendered as one entry. `sub_type` names
    /// what the run was built from (currently only `ref_text`), matching Python's
    /// `{'type': 'list', 'sub_type': 'ref_text', 'list_items': [...]}`.
    List {
        /// What kind of items this list holds.
        sub_type: ListSubType,
        /// The item texts, in reading order.
        list_items: Vec<String>,
    },
    /// A page header, emitted under its own `type` rather than as `text`.
    Header {
        /// The merged text of the block.
        text: String,
    },
    /// A page footer.
    Footer {
        /// The merged text of the block.
        text: String,
    },
    /// A page number.
    PageNumber {
        /// The merged text of the block.
        text: String,
    },
    /// Marginal / aside text near the page edge.
    AsideText {
        /// The merged text of the block.
        text: String,
    },
    /// A page footnote (distinct from a footnote nested onto a visual body).
    PageFootnote {
        /// The merged text of the block.
        text: String,
    },
}

/// The `sub_type` of a [`ContentBody::List`].
///
/// A dedicated enum rather than a bare `String` so the serialized tag cannot
/// drift from the Python vocabulary by typo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ListSubType {
    /// A run of reference-list items.
    RefText,
}

/// Renders a document to a `content_list.json`-compatible vector.
///
/// `image_dir` resolves image references (same joining as the Markdown
/// renderer).
///
/// Both the main flow and the page's *discarded* blocks are emitted: Python's
/// `union_make` builds this list from `para_blocks + discarded_blocks`
/// (`pipeline_middle_json_mkcontent.py:983`), so headers, footers, page numbers,
/// aside text and page footnotes each appear under their own `type`. "Discarded"
/// means "out of the reading flow" (they are excluded from the Markdown), not
/// "absent from the content list".
pub fn render_content_list(doc: &Document, image_dir: &str) -> Vec<ContentItem> {
    // Iterate pages rather than `doc.blocks()`: each entry needs its page's index
    // and size, which flattening over blocks would discard.
    doc.pages
        .iter()
        .flat_map(|page| {
            let items = group_ref_text(page.blocks.iter().chain(page.discarded.iter()));
            items.into_iter().filter_map(move |item| {
                Some(ContentItem {
                    content: item.body(image_dir)?,
                    bbox: item.bbox().and_then(|b| scale_bbox(b, page.size)),
                    page_idx: page.index,
                })
            })
        })
        .collect()
}

/// One reading-order entry: either a single block or a run of adjacent
/// reference-list blocks that collapse into one `list` entry.
enum Entry<'a> {
    Single(&'a Block),
    /// A run of two or more adjacent `RefText` blocks.
    RefRun(Vec<&'a Block>),
}

impl<'a> Entry<'a> {
    /// The entry's box: for a run, the first item's, matching Python's
    /// `flush_ref_group`, which takes `ref_group[0]['bbox']`.
    fn bbox(&self) -> Option<BBox> {
        match self {
            Entry::Single(b) => Some(b.bbox()),
            Entry::RefRun(blocks) => blocks.first().map(|b| b.bbox()),
        }
    }

    fn body(&self, image_dir: &str) -> Option<ContentBody> {
        match self {
            Entry::Single(b) => content_body(b, image_dir),
            Entry::RefRun(blocks) => Some(ContentBody::List {
                sub_type: ListSubType::RefText,
                list_items: blocks
                    .iter()
                    .filter_map(|b| match b {
                        Block::Text { lines, .. } => Some(merge_lines(lines)),
                        _ => None,
                    })
                    .filter(|s| !s.trim().is_empty())
                    .collect(),
            }),
        }
    }
}

/// Collapses each run of adjacent `RefText` blocks into one [`Entry::RefRun`].
///
/// Ports Python's `merge_adjacent_ref_text_blocks_for_content`
/// (`pipeline_middle_json_mkcontent.py:448`): consecutive `ref_text` blocks group
/// into a single `list` entry, and — matching `flush_ref_group`'s `len == 1`
/// branch — a lone `ref_text` stays a plain `text` entry rather than becoming a
/// one-item list.
fn group_ref_text<'a>(blocks: impl Iterator<Item = &'a Block>) -> Vec<Entry<'a>> {
    let mut out: Vec<Entry<'a>> = Vec::new();
    let mut run: Vec<&'a Block> = Vec::new();

    // A lone ref_text stays `text`; two or more become one `list`.
    let flush = |run: &mut Vec<&'a Block>, out: &mut Vec<Entry<'a>>| match run.len() {
        0 => {}
        1 => out.extend(run.drain(..).map(Entry::Single)),
        _ => out.push(Entry::RefRun(std::mem::take(run))),
    };

    for block in blocks {
        if matches!(
            block,
            Block::Text {
                role: TextRole::RefText,
                ..
            }
        ) {
            run.push(block);
            continue;
        }
        flush(&mut run, &mut out);
        out.push(Entry::Single(block));
    }
    flush(&mut run, &mut out);
    out
}

/// Normalizes a block's box to Python's `0..=1000` coordinate range.
///
/// Mirrors `_build_bbox`: divide by the page dimension, scale by 1000, and
/// **truncate** (Python's `int()`), not round. Returns `None` for a degenerate
/// page size, where the mapping is undefined and Python omits the key.
fn scale_bbox(bbox: BBox, size: PageSize) -> Option<[i32; 4]> {
    if size.width <= 0.0 || size.height <= 0.0 {
        return None;
    }
    let sx = |v: f32| (v * BBOX_SCALE / size.width) as i32;
    let sy = |v: f32| (v * BBOX_SCALE / size.height) as i32;
    Some([sx(bbox.x0), sy(bbox.y0), sx(bbox.x1), sy(bbox.y1)])
}

/// Maps one block to at most one [`ContentBody`].
///
/// Exhaustive over [`Block`] for the same reason as the Markdown renderer: a new
/// variant must be handled here before it compiles.
fn content_body(block: &Block, image_dir: &str) -> Option<ContentBody> {
    match block {
        Block::Text { role, lines, .. } => text_item(*role, merge_lines(lines)),
        Block::InterlineEquation { latex, .. } => Some(ContentBody::Equation {
            text: latex.to_string(),
            text_format: "latex".to_owned(),
        }),
        Block::Image(c) => Some(image_item(c, image_dir)),
        Block::Chart(c) => Some(chart_item(c, image_dir)),
        Block::Table(c) => Some(table_item(c, image_dir)),
        Block::Code(c) => Some(code_item(c)),
    }
}

/// Builds the item for a text block, routing each role to its Python `type`.
///
/// Mirrors `make_blocks_to_content_list` (`pipeline_middle_json_mkcontent.py:612`):
/// body-ish roles become `text`, while header/footer/page-number/aside-text/
/// page-footnote each carry their own type tag (line 622). A `RefText` reaching
/// here is a *lone* one — runs are collapsed into a `list` by [`group_ref_text`]
/// before this point — and Python renders that as plain `text`.
fn text_item(role: TextRole, text: String) -> Option<ContentBody> {
    match role {
        TextRole::Title(level) => Some(ContentBody::Text {
            text,
            // Python omits `text_level` for the doc title (level 0).
            text_level: (level.0 != 0).then_some(level.0),
        }),
        TextRole::Header => Some(ContentBody::Header { text }),
        TextRole::Footer => Some(ContentBody::Footer { text }),
        TextRole::PageNumber => Some(ContentBody::PageNumber { text }),
        TextRole::AsideText => Some(ContentBody::AsideText { text }),
        TextRole::PageFootnote => Some(ContentBody::PageFootnote { text }),
        TextRole::Body
        | TextRole::List
        | TextRole::Index
        | TextRole::Abstract
        | TextRole::RefText => Some(ContentBody::Text {
            text,
            text_level: None,
        }),
    }
}

/// Builds an image item.
fn image_item(c: &Captioned<ImageBody>, image_dir: &str) -> ContentBody {
    ContentBody::Image {
        img_path: join_image(image_dir, c.body.image.as_str()),
        image_caption: collect_texts(&c.captions),
        image_footnote: collect_texts(&c.footnotes),
    }
}

/// Builds a chart item.
fn chart_item(c: &Captioned<ImageBody>, image_dir: &str) -> ContentBody {
    ContentBody::Chart {
        img_path: join_image(image_dir, c.body.image.as_str()),
        chart_caption: collect_texts(&c.captions),
        chart_footnote: collect_texts(&c.footnotes),
    }
}

/// Builds a table item; `img_path` is empty when the table has no cropped raster.
fn table_item(c: &Captioned<TableBody>, image_dir: &str) -> ContentBody {
    let img_path = c
        .body
        .image
        .as_ref()
        .map(|r| join_image(image_dir, r.as_str()))
        .unwrap_or_default();
    ContentBody::Table {
        img_path,
        table_body: c.body.html.to_string(),
        table_caption: collect_texts(&c.captions),
        table_footnote: collect_texts(&c.footnotes),
    }
}

/// Builds a code item; code lines join with newlines to preserve layout.
fn code_item(c: &Captioned<CodeBody>) -> ContentBody {
    let code_body = c
        .body
        .lines
        .iter()
        .map(crate::text::flatten_line)
        .collect::<Vec<_>>()
        .join("\n");
    ContentBody::Code {
        code_body,
        code_language: c.body.language.as_ref().map(|l| l.to_string()),
        code_caption: collect_texts(&c.captions),
        code_footnote: collect_texts(&c.footnotes),
    }
}
