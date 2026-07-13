//! Geometry primitives shared across the whole workspace.
//!
//! [`BBox`] is the single bounding-box type; every model crate, block, line, and
//! span reuses it rather than passing bare `[f32; 4]` arrays around.

use serde::{Deserialize, Serialize};

/// An axis-aligned bounding box in page coordinates (top-left origin, PDF points).
///
/// Serializes to and from a `[x0, y0, x1, y1]` array so the on-disk JSON matches
/// the Python `middle_json` format, while in-memory access stays named.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(from = "[f32; 4]", into = "[f32; 4]")]
pub struct BBox {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl BBox {
    /// Constructs a box, normalizing so that `x0 <= x1` and `y0 <= y1`.
    pub fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Self {
            x0: x0.min(x1),
            y0: y0.min(y1),
            x1: x0.max(x1),
            y1: y0.max(y1),
        }
    }

    /// Width, clamped to non-negative.
    pub fn width(&self) -> f32 {
        (self.x1 - self.x0).max(0.0)
    }

    /// Height, clamped to non-negative.
    pub fn height(&self) -> f32 {
        (self.y1 - self.y0).max(0.0)
    }

    /// Area of the box.
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// Center point `(x, y)`.
    pub fn center(&self) -> (f32, f32) {
        ((self.x0 + self.x1) * 0.5, (self.y0 + self.y1) * 0.5)
    }

    /// Intersection rectangle, or `None` when the boxes are disjoint.
    ///
    /// Returning `Option` keeps a disjoint pair from silently reporting a
    /// zero-area overlap that callers might misread as "touching".
    pub fn intersection(&self, other: &BBox) -> Option<BBox> {
        let x0 = self.x0.max(other.x0);
        let y0 = self.y0.max(other.y0);
        let x1 = self.x1.min(other.x1);
        let y1 = self.y1.min(other.y1);
        (x0 < x1 && y0 < y1).then_some(BBox { x0, y0, x1, y1 })
    }

    /// Intersection-over-union with another box, in `0.0..=1.0`.
    pub fn iou(&self, other: &BBox) -> f32 {
        let inter = self.intersection(other).map_or(0.0, |b| b.area());
        let union = self.area() + other.area() - inter;
        if union > 0.0 {
            inter / union
        } else {
            0.0
        }
    }

    /// Fraction of `self`'s area covered by `other` (containment, not symmetric).
    pub fn overlap_ratio(&self, other: &BBox) -> f32 {
        let area = self.area();
        if area > 0.0 {
            self.intersection(other).map_or(0.0, |b| b.area()) / area
        } else {
            0.0
        }
    }
}

impl From<[f32; 4]> for BBox {
    fn from([x0, y0, x1, y1]: [f32; 4]) -> Self {
        Self::new(x0, y0, x1, y1)
    }
}

impl From<BBox> for [f32; 4] {
    fn from(b: BBox) -> Self {
        [b.x0, b.y0, b.x1, b.y1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_swapped_corners() {
        let b = BBox::new(10.0, 20.0, 0.0, 5.0);
        assert_eq!((b.x0, b.y0, b.x1, b.y1), (0.0, 5.0, 10.0, 20.0));
    }

    #[test]
    fn disjoint_intersection_is_none() {
        let a = BBox::new(0.0, 0.0, 1.0, 1.0);
        let b = BBox::new(2.0, 2.0, 3.0, 3.0);
        assert!(a.intersection(&b).is_none());
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn iou_of_identical_is_one() {
        let a = BBox::new(0.0, 0.0, 2.0, 2.0);
        assert_eq!(a.iou(&a), 1.0);
    }

    #[test]
    fn serde_roundtrips_as_array() {
        let b = BBox::new(1.0, 2.0, 3.0, 4.0);
        let json = serde_json::to_string(&b).unwrap();
        assert_eq!(json, "[1.0,2.0,3.0,4.0]");
        assert_eq!(serde_json::from_str::<BBox>(&json).unwrap(), b);
    }
}
