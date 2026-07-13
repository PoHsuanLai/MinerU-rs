//! The raw block schema the VLM emits, before conversion to the document tree.
//!
//! The model returns, per page, a flat list of blocks each carrying a normalized
//! bounding box (`0.0..=1.0`), a label string, and optional content. This mirrors
//! the intermediate dicts the Python `MagicModel` consumes; converting it into the
//! typed [`Document`](mineru_types::Document) tree lives in [`crate::assemble`].

use serde::Deserialize;

/// A single block as returned by the VLM for one page.
#[derive(Debug, Clone, Deserialize)]
pub struct VlmBlock {
    /// Normalized `[x0, y0, x1, y1]` in `0.0..=1.0`, relative to page size.
    pub bbox: [f32; 4],
    /// The block label (e.g. `"text"`, `"title"`, `"image"`, `"table"`, `"equation"`).
    #[serde(rename = "type")]
    pub label: String,
    /// The block's textual content: plain text, LaTeX, or table HTML depending on label.
    #[serde(default)]
    pub content: Option<String>,
    /// Rotation angle in degrees, if the model reported one.
    #[serde(default)]
    pub angle: i32,
    /// Optional finer classification for image/chart blocks (e.g. a seal).
    #[serde(default)]
    pub sub_type: Option<String>,
}

/// The VLM's output for a single page: its pixel size and the blocks it found.
#[derive(Debug, Clone)]
pub struct VlmPage {
    /// Page width in pixels (used to denormalize block boxes).
    pub width: f32,
    /// Page height in pixels.
    pub height: f32,
    /// The blocks the model emitted, in reading order.
    pub blocks: Vec<VlmBlock>,
}
