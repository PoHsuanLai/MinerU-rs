//! OCR result inputs shared by both table paths.
//!
//! Table recognition takes an already-computed OCR result (detection boxes plus
//! recognized text) and matches those text spans onto the structural cells the
//! table model predicts. This module defines the small value type that carries
//! one OCR detection, mirroring the `[box, text, score]` triples the Python code
//! consumes.

use mineru_types::BBox;

/// A single OCR detection: a text span, its axis-aligned bounding box in the
/// table image's pixel coordinates, and a confidence score.
///
/// The Python pipeline passes quadrilateral detection boxes; downstream matching
/// only ever uses their axis-aligned extent, so we store a [`BBox`] directly.
#[derive(Debug, Clone, PartialEq)]
pub struct OcrSpan {
    /// Axis-aligned bounding box of the detected text, in image pixels.
    pub bbox: BBox,
    /// The recognized text (already HTML-escaped by the caller, as in Python).
    pub text: String,
    /// Recognition confidence in `0.0..=1.0`.
    pub score: f32,
}

impl OcrSpan {
    /// Constructs an [`OcrSpan`].
    pub fn new(bbox: BBox, text: impl Into<String>, score: f32) -> Self {
        Self {
            bbox,
            text: text.into(),
            score,
        }
    }
}
