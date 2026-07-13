//! The parsed-document tree — the Rust-native replacement for Python's
//! dynamically-typed `middle_json` dict.
//!
//! The design makes illegal states unrepresentable: block and span kinds are
//! enums whose variants carry only their own data (no stringly `type` field with
//! a pile of optional keys), and the ~40 Python block-type strings collapse into
//! a handful of variants — most of them folding *into structure* via
//! [`Captioned`] and [`TextRole`].

use serde::{Deserialize, Serialize};

use crate::content::{Html, ImageRef, Lang, Latex, Score};
use crate::error::{Error, Result};
use crate::geom::BBox;

/// Maximum supported heading depth (doc title = 0).
pub const MAX_TITLE_LEVEL: u8 = 6;

/// A parsed document: an ordered list of pages. (Python's `middle_json`.)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Document {
    pub pages: Vec<Page>,
}

impl Document {
    /// Iterates over every content block on every page, in reading order.
    pub fn blocks(&self) -> impl Iterator<Item = &Block> {
        self.pages.iter().flat_map(|p| p.blocks.iter())
    }
}

impl IntoIterator for Document {
    type Item = Page;
    type IntoIter = std::vec::IntoIter<Page>;
    fn into_iter(self) -> Self::IntoIter {
        self.pages.into_iter()
    }
}

impl<'a> IntoIterator for &'a Document {
    type Item = &'a Page;
    type IntoIter = std::slice::Iter<'a, Page>;
    fn into_iter(self) -> Self::IntoIter {
        self.pages.iter()
    }
}

/// One page: its content blocks (reading order) and its discarded blocks
/// (headers/footers/page numbers stripped for semantic coherence).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Page {
    pub index: usize,
    pub size: PageSize,
    pub blocks: Vec<Block>,
    #[serde(default)]
    pub discarded: Vec<Block>,
}

/// Page dimensions in points `(width, height)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(from = "[f32; 2]", into = "[f32; 2]")]
pub struct PageSize {
    pub width: f32,
    pub height: f32,
}

impl From<[f32; 2]> for PageSize {
    fn from([width, height]: [f32; 2]) -> Self {
        Self { width, height }
    }
}

impl From<PageSize> for [f32; 2] {
    fn from(s: PageSize) -> Self {
        [s.width, s.height]
    }
}

/// A block of parsed content.
///
/// The ~40 Python block-type strings collapse here: caption/body/footnote roles
/// become [`Captioned`] fields, and the flat text roles (title, header, list, …)
/// fold into a single [`Text`](Block::Text) variant tagged with [`TextRole`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Block {
    /// Flowing text. `role` distinguishes title/list/header/etc. — they share the
    /// identical `{bbox, lines}` shape, so one variant + a role tag beats ~11
    /// near-duplicate variants.
    Text {
        bbox: BBox,
        role: TextRole,
        lines: Vec<TextLine>,
    },
    /// A standalone (interline) equation.
    InterlineEquation { bbox: BBox, latex: Latex },
    /// A figure with its captions/footnotes.
    Image(Captioned<ImageBody>),
    /// A table with its captions/footnotes.
    Table(Captioned<TableBody>),
    /// A chart with its captions/footnotes.
    Chart(Captioned<ImageBody>),
    /// A code block with its captions/footnotes.
    Code(Captioned<CodeBody>),
}

impl Block {
    /// The block's bounding box, wherever it lives across the variants.
    pub fn bbox(&self) -> BBox {
        match self {
            Block::Text { bbox, .. } | Block::InterlineEquation { bbox, .. } => *bbox,
            Block::Image(c) | Block::Chart(c) => c.body.bbox,
            Block::Table(c) => c.body.bbox,
            Block::Code(c) => c.body.bbox,
        }
    }
}

/// The role of a flowing-text block. Folds ~11 Python text block-type strings
/// (all sharing the `{bbox, lines}` shape) into one enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextRole {
    Body,
    Title(TitleLevel),
    List,
    Index,
    Abstract,
    RefText,
    Header,
    Footer,
    PageNumber,
    AsideText,
    PageFootnote,
}

/// A heading level; `0` is the document title, `1..=MAX_TITLE_LEVEL` are sections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TitleLevel(pub u8);

impl TitleLevel {
    /// Builds a level, rejecting depths beyond [`MAX_TITLE_LEVEL`].
    pub fn new(level: u8) -> Result<Self> {
        if level <= MAX_TITLE_LEVEL {
            Ok(Self(level))
        } else {
            Err(Error::TitleLevelOutOfRange(level))
        }
    }
}

/// A visual block: a typed `body` plus its optional captions and footnotes.
///
/// Written once, reused for image/table/chart/code — this is the single biggest
/// dedup versus the Python model, which repeats body/caption/footnote handling
/// across roughly a dozen block-type strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Captioned<B> {
    pub body: B,
    #[serde(default)]
    pub captions: Vec<TextBlock>,
    #[serde(default)]
    pub footnotes: Vec<TextBlock>,
}

impl<B> Captioned<B> {
    /// Wraps a body with no captions or footnotes.
    pub fn bare(body: B) -> Self {
        Self {
            body,
            captions: Vec::new(),
            footnotes: Vec::new(),
        }
    }
}

/// The plain-text content used by captions and footnotes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBlock {
    pub bbox: BBox,
    pub lines: Vec<TextLine>,
}

/// A figure/chart body: the extracted raster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageBody {
    pub bbox: BBox,
    pub image: ImageRef,
}

/// A table body: its HTML, and optionally a cropped raster of the table region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableBody {
    pub bbox: BBox,
    pub html: Html,
    #[serde(default)]
    pub image: Option<ImageRef>,
}

/// A code body: its text lines and optional language tag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeBody {
    pub bbox: BBox,
    pub lines: Vec<TextLine>,
    #[serde(default)]
    pub language: Option<Lang>,
}

/// A line of text: a bounding box plus the inline spans it contains.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextLine {
    pub bbox: BBox,
    pub spans: Vec<Span>,
}

/// Inline content within a line. The Python `content`/`text`/`latex`/`html`
/// optional-key soup becomes variant payloads — a table span *always* carries
/// HTML, a text span *cannot* carry an image path.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Span {
    Text {
        bbox: BBox,
        text: String,
        score: Score,
    },
    InlineEquation {
        bbox: BBox,
        latex: Latex,
        score: Score,
    },
    Image {
        bbox: BBox,
        image: ImageRef,
    },
}

impl Span {
    /// The span's bounding box.
    pub fn bbox(&self) -> BBox {
        match self {
            Span::Text { bbox, .. }
            | Span::InlineEquation { bbox, .. }
            | Span::Image { bbox, .. } => *bbox,
        }
    }
}
