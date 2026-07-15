//! Assigning page-level inline objects (formulas) to the table that contains them.
//!
//! Port of `_extract_table_inline_objects` (`batch_analyze.py`). A formula printed
//! inside a table is not found by the table path at all: the page layout model
//! detects it as an ordinary page formula, and formula recognition has already
//! turned it into LaTeX by the time tables run. Without this step that LaTeX is
//! emitted as a standalone block *next to* the table, while the table's own cell
//! is either empty or filled with whatever OCR made of the formula's pixels.
//!
//! The reference's trick — mirrored here — is to treat a recognized formula as if
//! it were an OCR span: [`assign_to_tables`] works out which table each formula
//! falls in and where it sits in that table's crop, the caller masks those pixels
//! out so OCR never reads them, and the LaTeX is handed to the structure matcher
//! as one more span among the real ones. The matcher needs no formula concept;
//! a cell whose best-matching span happens to be LaTeX just contains LaTeX.
//!
//! This module is pure geometry and owns no model: the caller supplies formulas
//! that something else recognized, exactly as [`crate::orientation`] takes OCR it
//! does not run itself.

use image::RgbImage;
use mineru_types::BBox;

/// Paints each box white on a copy of the crop.
///
/// Run over the assigned formulas' [`crop_bbox`](TableFormula::crop_bbox) before
/// the crop goes to OCR detection. The formula's LaTeX is spliced in separately as
/// a span, so leaving its pixels in place would get the same content read twice:
/// once correctly as LaTeX, and once as whatever OCR makes of a formula's glyphs —
/// two spans competing to match the same cell.
///
/// Boxes are clamped to the image, so one running past an edge paints what overlaps
/// rather than being dropped.
pub fn mask_boxes(crop: &RgbImage, boxes: impl IntoIterator<Item = BBox>) -> RgbImage {
    const WHITE: image::Rgb<u8> = image::Rgb([255, 255, 255]);

    let mut out = crop.clone();
    let (w, h) = (out.width(), out.height());
    for bbox in boxes {
        let x0 = bbox.x0.max(0.0) as u32;
        let y0 = bbox.y0.max(0.0) as u32;
        let x1 = (bbox.x1.max(0.0) as u32).min(w);
        let y1 = (bbox.y1.max(0.0) as u32).min(h);
        for y in y0..y1 {
            for x in x0..x1 {
                out.put_pixel(x, y, WHITE);
            }
        }
    }
    out
}

/// A page-level formula that has already been recognized into LaTeX.
///
/// `bbox` is in the same space as the table boxes passed alongside it — page
/// pixels, in practice — since the two are compared directly.
#[derive(Debug, Clone, PartialEq)]
pub struct PageFormula {
    /// Where the formula sits on the page.
    pub bbox: BBox,
    /// The recognized LaTeX, without delimiters.
    pub latex: String,
}

/// A formula placed inside one table's crop.
#[derive(Debug, Clone, PartialEq)]
pub struct TableFormula {
    /// Index into the `formulas` slice given to [`assign_to_tables`].
    pub formula: usize,
    /// The formula's box in **crop-local** pixels: page coordinates minus the
    /// table's top-left corner, clipped to the crop. This is the space both the
    /// OCR spans and the structure model's cell boxes live in.
    pub crop_bbox: BBox,
}

/// Which formulas belong to which table.
///
/// Returns one entry per table in `tables`, in the same order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Assignment {
    /// Per-table formulas, parallel to the input `tables`.
    pub per_table: Vec<Vec<TableFormula>>,
    /// Indices of formulas claimed by some table. The caller must drop these from
    /// the page's own blocks: the reference removes a claimed formula from the
    /// page layout list so it is not emitted twice — once inside the table's HTML
    /// and once as a standalone block beside it.
    pub claimed: Vec<usize>,
}

/// Assigns each formula to the table whose box contains the formula's **center
/// point**, resolving a formula that lands in more than one table by greatest
/// overlap area.
///
/// Center containment rather than overlap is the reference's rule, and it is the
/// forgiving choice on purpose: a display formula often bleeds a few pixels past
/// the ruling it sits in, and an overlap test would either claim it for a
/// neighbouring table or need a threshold to tune. A center is unambiguous.
///
/// `tables` whose entry is `None` opt out of receiving formulas — a rotated table
/// is the case that matters (see below) — but still occupy a slot in `per_table`
/// so the result stays index-aligned with the caller's tables.
///
/// # Rotated tables
///
/// A table crop that gets deskewed is rotated *after* this runs, which would move
/// every cell out from under the crop-local boxes computed here. The reference
/// skips inline objects for such tables (`_table_supports_inline_objects` gates on
/// `rotate_label == "0"`) and so must any caller: pass `None` for a table that
/// will be rotated. Its formulas then stay unclaimed and are emitted as ordinary
/// page blocks, which is a visible but correct degradation.
pub fn assign_to_tables(tables: &[Option<BBox>], formulas: &[PageFormula]) -> Assignment {
    let mut per_table: Vec<Vec<TableFormula>> = vec![Vec::new(); tables.len()];
    let mut claimed = Vec::new();

    for (index, formula) in formulas.iter().enumerate() {
        let (cx, cy) = formula.bbox.center();

        // Among the tables containing this center, the one it overlaps most. A
        // center can only be inside two tables if their boxes overlap, which the
        // layout model does occasionally emit; picking by area beats picking by
        // whichever came first.
        let winner = tables
            .iter()
            .enumerate()
            .filter_map(|(slot, table)| table.map(|bbox| (slot, bbox)))
            .filter(|(_, bbox)| {
                cx >= bbox.x0 && cx <= bbox.x1 && cy >= bbox.y0 && cy <= bbox.y1
            })
            .max_by(|(_, a), (_, b)| {
                let overlap = |t: &BBox| {
                    formula
                        .bbox
                        .intersection(t)
                        .map(|i| i.area())
                        .unwrap_or(0.0)
                };
                overlap(a)
                    .partial_cmp(&overlap(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        let Some((slot, table)) = winner else { continue };
        // Clip to the crop: a formula overhanging the table edge would otherwise
        // produce a span box reaching outside the image the matcher works in.
        let Some(clipped) = formula.bbox.intersection(&table) else {
            continue;
        };
        let crop_bbox = BBox::new(
            clipped.x0 - table.x0,
            clipped.y0 - table.y0,
            clipped.x1 - table.x0,
            clipped.y1 - table.y0,
        );
        if let Some(slot) = per_table.get_mut(slot) {
            slot.push(TableFormula {
                formula: index,
                crop_bbox,
            });
        }
        claimed.push(index);
    }

    Assignment { per_table, claimed }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn formula(x0: f32, y0: f32, x1: f32, y1: f32) -> PageFormula {
        PageFormula {
            bbox: BBox::new(x0, y0, x1, y1),
            latex: "x^2".into(),
        }
    }

    #[test]
    fn assigns_formula_to_the_table_containing_its_center() {
        let tables = [Some(BBox::new(0.0, 0.0, 100.0, 100.0))];
        let a = assign_to_tables(&tables, &[formula(10.0, 10.0, 30.0, 30.0)]);

        assert_eq!(a.claimed, vec![0]);
        assert_eq!(a.per_table[0].len(), 1);
        // Crop-local: the table's origin is (0,0) here, so it is unchanged.
        assert_eq!(a.per_table[0][0].crop_bbox, BBox::new(10.0, 10.0, 30.0, 30.0));
    }

    #[test]
    fn crop_box_is_relative_to_the_table_origin() {
        let tables = [Some(BBox::new(50.0, 40.0, 200.0, 140.0))];
        let a = assign_to_tables(&tables, &[formula(60.0, 50.0, 80.0, 70.0)]);

        assert_eq!(a.per_table[0][0].crop_bbox, BBox::new(10.0, 10.0, 30.0, 30.0));
    }

    #[test]
    fn formula_outside_every_table_is_left_unclaimed() {
        let tables = [Some(BBox::new(0.0, 0.0, 100.0, 100.0))];
        let a = assign_to_tables(&tables, &[formula(200.0, 200.0, 220.0, 220.0)]);

        assert!(a.claimed.is_empty());
        assert!(a.per_table[0].is_empty());
    }

    /// The center rule, at its whole point: a formula mostly outside the table is
    /// still claimed, and one mostly inside is not, purely on where the center is.
    #[test]
    fn center_decides_not_overlap() {
        let tables = [Some(BBox::new(0.0, 0.0, 100.0, 100.0))];
        // Center (95, 50) is inside; most of the box is not.
        let a = assign_to_tables(&tables, &[formula(90.0, 40.0, 100.0, 60.0)]);
        assert_eq!(a.claimed, vec![0]);

        // Center (105, 50) is outside, though it overlaps the table substantially.
        let b = assign_to_tables(&tables, &[formula(90.0, 40.0, 120.0, 60.0)]);
        assert!(b.claimed.is_empty());
    }

    #[test]
    fn overlapping_tables_are_broken_by_overlap_area() {
        // Both contain the formula's center (25, 25); the second overlaps it more.
        let tables = [
            Some(BBox::new(0.0, 0.0, 30.0, 30.0)),
            Some(BBox::new(0.0, 0.0, 100.0, 100.0)),
        ];
        let a = assign_to_tables(&tables, &[formula(20.0, 20.0, 40.0, 40.0)]);

        assert!(a.per_table[0].is_empty());
        assert_eq!(a.per_table[1].len(), 1);
        assert_eq!(a.claimed, vec![0]);
    }

    #[test]
    fn opted_out_table_claims_nothing() {
        let tables = [None];
        let a = assign_to_tables(&tables, &[formula(10.0, 10.0, 30.0, 30.0)]);

        assert!(a.claimed.is_empty());
        assert_eq!(a.per_table.len(), 1, "slot is kept so indices stay aligned");
        assert!(a.per_table[0].is_empty());
    }

    #[test]
    fn overhanging_formula_is_clipped_to_the_crop() {
        let tables = [Some(BBox::new(0.0, 0.0, 100.0, 100.0))];
        // Center (95, 50) is inside, so it is claimed; the box runs past x1=100.
        let a = assign_to_tables(&tables, &[formula(90.0, 40.0, 100.0, 60.0)]);

        let got = a.per_table[0][0].crop_bbox;
        assert!(got.x1 <= 100.0, "crop box must not exceed the crop: {got:?}");
    }

    #[test]
    fn mask_paints_only_the_given_box_white() {
        let crop = RgbImage::from_pixel(10, 10, image::Rgb([0, 0, 0]));
        let out = mask_boxes(&crop, [BBox::new(2.0, 2.0, 5.0, 5.0)]);

        assert_eq!(*out.get_pixel(2, 2), image::Rgb([255, 255, 255]), "inside");
        assert_eq!(*out.get_pixel(4, 4), image::Rgb([255, 255, 255]), "inside");
        // x1/y1 are exclusive, matching the half-open convention crops use.
        assert_eq!(*out.get_pixel(5, 5), image::Rgb([0, 0, 0]), "past the edge");
        assert_eq!(*out.get_pixel(0, 0), image::Rgb([0, 0, 0]), "outside");
    }

    /// A box overhanging the crop must paint its overlap, not panic on the
    /// out-of-range coordinate and not silently skip the whole box.
    #[test]
    fn mask_clamps_a_box_running_past_the_edge() {
        let crop = RgbImage::from_pixel(10, 10, image::Rgb([0, 0, 0]));
        let out = mask_boxes(&crop, [BBox::new(8.0, 8.0, 50.0, 50.0)]);

        assert_eq!(*out.get_pixel(9, 9), image::Rgb([255, 255, 255]));
        assert_eq!(*out.get_pixel(7, 7), image::Rgb([0, 0, 0]));
    }

    #[test]
    fn mask_with_no_boxes_is_the_original() {
        let crop = RgbImage::from_pixel(4, 4, image::Rgb([7, 8, 9]));
        assert_eq!(mask_boxes(&crop, []), crop);
    }

    #[test]
    fn every_table_gets_a_slot_even_with_no_formulas() {
        let tables = [Some(BBox::new(0.0, 0.0, 10.0, 10.0)), None];
        let a = assign_to_tables(&tables, &[]);

        assert_eq!(a.per_table.len(), 2);
        assert!(a.claimed.is_empty());
    }
}
