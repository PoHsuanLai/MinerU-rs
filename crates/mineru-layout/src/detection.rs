//! The raw detection output type.
//!
//! [`LayoutDet`] is intentionally *not* a [`mineru_types::Block`]: this crate only
//! produces flat, scored, ordered boxes. Assembling the `Block` tree (grouping,
//! captioning, span extraction) happens later in the pipeline backend.

use mineru_types::BBox;

use crate::label::LayoutLabel;

/// A single detected layout region.
///
/// `bbox` is in the coordinate space of the *original* input image (pixels, top-
/// left origin), already scaled back from the 800×800 model input. `order` is the
/// reading-order rank assigned by the pointer network (0 = read first).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayoutDet {
    /// The region's bounding box, in original-image pixel coordinates.
    pub bbox: BBox,
    /// The predicted layout class.
    pub label: LayoutLabel,
    /// Confidence score in `0.0..=1.0` (sigmoid of the class logit).
    pub score: f32,
    /// Reading-order rank (0-based); lower is read earlier.
    pub order: usize,
}

impl LayoutDet {
    /// Constructs a detection.
    pub fn new(bbox: BBox, label: LayoutLabel, score: f32, order: usize) -> Self {
        Self {
            bbox,
            label,
            score,
            order,
        }
    }
}
