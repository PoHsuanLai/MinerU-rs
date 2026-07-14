//! DBPostProcess: probability map → text-line quadrilaterals.
//!
//! A faithful Rust port of `DBPostProcess.boxes_from_bitmap` (quad mode) from
//! `pytorchocr/postprocess/db_postprocess.py`. The neural net emits a per-pixel
//! probability map; this turns it into oriented boxes with no further network:
//!
//! 1. **Binarize** the map at `thresh`.
//! 2. **Find contours** of the foreground regions (`imageproc::contours`, the
//!    Suzuki–Abe algorithm — the same one OpenCV's `findContours` uses).
//! 3. For each contour, take its **minimum-area rectangle** (`get_mini_boxes`),
//!    drop ones smaller than `min_size`.
//! 4. Score the box by the **mean probability** inside it (`box_score_fast`);
//!    drop boxes below `box_thresh`.
//! 5. **Unclip** (dilate) the box outward by a polygon offset proportional to
//!    `unclip_ratio` (`clipper2` round-join inflate, matching `pyclipper`).
//! 6. Re-fit the min-area rect, rescale from map coords to source-image coords.
//!
//! Boxes are returned as four `(x, y)` corner points in source-image pixels.

use clipper2::{EndType, JoinType, Paths};
use image::GrayImage;
use imageproc::contours::find_contours;
use imageproc::point::Point;
use mineru_burn_common::geometry::min_area_rectangle;

/// A detected text-line quadrilateral: four corner points in source-image pixels,
/// plus the mean-probability score that passed `box_thresh`.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadBox {
    /// The four corners, ordered as produced by `get_mini_boxes`.
    pub points: [(f32, f32); 4],
    /// Mean probability inside the box (the DB "box score").
    pub score: f32,
}

/// Tunable DBPostProcess parameters (defaults match PP-OCRv6 text detection).
#[derive(Debug, Clone, Copy)]
pub struct DbPostConfig {
    /// Binarization threshold applied to the probability map.
    pub thresh: f32,
    /// Minimum mean-probability score for a box to be kept.
    pub box_thresh: f32,
    /// Maximum number of contours to consider.
    pub max_candidates: usize,
    /// Polygon dilation ratio for `unclip`.
    pub unclip_ratio: f32,
    /// Minimum side length (in map pixels) for a box to survive.
    pub min_size: f32,
}

impl Default for DbPostConfig {
    fn default() -> Self {
        // Matches pytorchocr_utility.init_args() DB defaults / TextSystem usage.
        Self {
            thresh: 0.3,
            box_thresh: 0.6,
            max_candidates: 1000,
            unclip_ratio: 1.5,
            min_size: 3.0,
        }
    }
}

/// A probability map borrowed for post-processing: `data[y * width + x]` in `[0, 1]`.
pub struct ProbMap<'a> {
    /// Row-major probability values.
    pub data: &'a [f32],
    /// Map width in pixels.
    pub width: usize,
    /// Map height in pixels.
    pub height: usize,
}

impl ProbMap<'_> {
    fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }
}

/// Extracts quad boxes from a probability map.
///
/// `dest_width`/`dest_height` are the *source* image dimensions the boxes are
/// rescaled to (the map is typically 1/4 resolution). Mirrors
/// `boxes_from_bitmap(pred, bitmap, dest_width, dest_height)`.
pub fn boxes_from_bitmap(
    pred: &ProbMap,
    cfg: &DbPostConfig,
    dest_width: f32,
    dest_height: f32,
) -> Vec<QuadBox> {
    let (w, h) = (pred.width, pred.height);
    if w == 0 || h == 0 {
        return Vec::new();
    }

    // Step 1: binarize -> a GrayImage with foreground = 255.
    let mut bitmap = GrayImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            if pred.at(x, y) > cfg.thresh {
                bitmap.put_pixel(x as u32, y as u32, image::Luma([255]));
            }
        }
    }

    // Step 2: contours (Suzuki-Abe, as OpenCV's findContours).
    let contours: Vec<imageproc::contours::Contour<i32>> = find_contours(&bitmap);

    let mut boxes = Vec::new();
    for contour in contours.iter().take(cfg.max_candidates) {
        if contour.points.len() < 4 {
            // Fewer than 4 points cannot form a min-area rect meaningfully.
            continue;
        }

        // Step 3: get_mini_boxes -> ordered 4 corners + shorter side length.
        let contour_pts: Vec<Point<f64>> = contour
            .points
            .iter()
            .map(|p| Point::new(p.x as f64, p.y as f64))
            .collect();
        let (mini, sside) = match get_mini_boxes(&contour_pts) {
            Some(v) => v,
            None => continue,
        };
        if sside < cfg.min_size {
            continue;
        }

        // Step 4: box score (fast: mean prob within the axis-aligned bbox mask).
        let score = box_score_fast(pred, &mini);
        if cfg.box_thresh > score {
            continue;
        }

        // Step 5: unclip (dilate) then re-fit min-area rect.
        let expanded = match unclip(&mini, cfg.unclip_ratio) {
            Some(pts) if !pts.is_empty() => pts,
            _ => continue,
        };
        let expanded_pts: Vec<Point<f64>> = expanded
            .iter()
            .map(|&(x, y)| Point::new(x as f64, y as f64))
            .collect();
        let (mut box2, sside2) = match get_mini_boxes(&expanded_pts) {
            Some(v) => v,
            None => continue,
        };
        if sside2 < cfg.min_size + 2.0 {
            continue;
        }

        // Step 6: rescale from map coords to source coords, clamp, round.
        for p in box2.iter_mut() {
            p.0 = (p.0 / w as f32 * dest_width).round().clamp(0.0, dest_width);
            p.1 = (p.1 / h as f32 * dest_height).round().clamp(0.0, dest_height);
        }

        boxes.push(QuadBox {
            points: box2,
            score,
        });
    }

    boxes
}

/// `get_mini_boxes`: minimum-area rectangle, corners ordered, plus the shorter side.
///
/// Returns the four points ordered as the reference does — sorted by `x`, then the
/// two-of-each-side reordering by `y` — so downstream crop geometry matches
/// `cv2.minAreaRect` + `cv2.boxPoints`.
fn get_mini_boxes(points: &[Point<f64>]) -> Option<([(f32, f32); 4], f32)> {
    let pts_f: Vec<(f64, f64)> = points.iter().map(|p| (p.x, p.y)).collect();
    let (rect, sside) = min_area_rectangle(&pts_f)?;
    let mut pts: Vec<(f32, f32)> = rect.iter().map(|&(x, y)| (x as f32, y as f32)).collect();

    // Reference: sort the 4 corners by x, then assign corners by y comparisons.
    // `total_cmp` gives a strict total order (no NaN inconsistency for `sort_by`).
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (idx1, idx4) = if pts[1].1 > pts[0].1 { (0, 1) } else { (1, 0) };
    let (idx2, idx3) = if pts[3].1 > pts[2].1 { (2, 3) } else { (3, 2) };
    let ordered = [pts[idx1], pts[idx2], pts[idx3], pts[idx4]];
    Some((ordered, sside as f32))
}

/// Euclidean distance between two points.
fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

/// `box_score_fast`: mean probability inside the polygon's axis-aligned bounding box.
///
/// The reference builds a polygon mask, but for a convex quad the bbox mean is what
/// PP-OCR actually uses in "fast" mode via `cv2.fillPoly`+`cv2.mean`. We fill the
/// quad into a mask and average only the covered pixels, matching that behaviour.
fn box_score_fast(pred: &ProbMap, box_pts: &[(f32, f32); 4]) -> f32 {
    let (w, h) = (pred.width as i32, pred.height as i32);
    let xs: Vec<f32> = box_pts.iter().map(|p| p.0).collect();
    let ys: Vec<f32> = box_pts.iter().map(|p| p.1).collect();
    let xmin = (xs.iter().cloned().fold(f32::MAX, f32::min).floor() as i32).clamp(0, w - 1);
    let xmax = (xs.iter().cloned().fold(f32::MIN, f32::max).ceil() as i32).clamp(0, w - 1);
    let ymin = (ys.iter().cloned().fold(f32::MAX, f32::min).floor() as i32).clamp(0, h - 1);
    let ymax = (ys.iter().cloned().fold(f32::MIN, f32::max).ceil() as i32).clamp(0, h - 1);
    if xmax < xmin || ymax < ymin {
        return 0.0;
    }

    // Polygon points relative to the bbox origin.
    let poly: Vec<(f32, f32)> = box_pts
        .iter()
        .map(|&(x, y)| (x - xmin as f32, y - ymin as f32))
        .collect();

    let mut sum = 0.0f64;
    let mut count = 0u64;
    for y in ymin..=ymax {
        for x in xmin..=xmax {
            let px = (x - xmin) as f32 + 0.5;
            let py = (y - ymin) as f32 + 0.5;
            if point_in_poly(px, py, &poly) {
                sum += pred.at(x as usize, y as usize) as f64;
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    }
}

/// Even-odd point-in-polygon test.
fn point_in_poly(x: f32, y: f32, poly: &[(f32, f32)]) -> bool {
    let mut inside = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// `unclip`: dilate the box polygon outward by `area * ratio / perimeter`.
///
/// Uses `clipper2`'s round-join `inflate`, the direct analogue of pyclipper's
/// `PyclipperOffset` with `JT_ROUND` / `ET_CLOSEDPOLYGON`. Returns the offset
/// polygon's vertices, or `None` if the offset produced no/split geometry (the
/// reference discards boxes whose unclip yields more than one path).
fn unclip(box_pts: &[(f32, f32); 4], unclip_ratio: f32) -> Option<Vec<(f32, f32)>> {
    let area = polygon_area(box_pts);
    let perim = polygon_perimeter(box_pts);
    if perim <= 0.0 {
        return None;
    }
    let distance = (area * unclip_ratio / perim) as f64;

    let subject: Vec<(f64, f64)> = box_pts.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
    let paths: Paths = subject.into();
    let result = clipper2::inflate(paths, distance, JoinType::Round, EndType::Polygon, 2.0);

    // The reference rejects `len(box) > 1` — more than one output path means the
    // offset split the region, so the detection is ambiguous.
    if result.len() != 1 {
        return None;
    }
    let path = result.iter().next()?;
    let pts: Vec<(f32, f32)> = path.iter().map(|p| (p.x() as f32, p.y() as f32)).collect();
    if pts.is_empty() {
        None
    } else {
        Some(pts)
    }
}

/// Shoelace polygon area (absolute), matching `shapely.Polygon.area`.
fn polygon_area(pts: &[(f32, f32); 4]) -> f32 {
    let mut area = 0.0f32;
    let n = pts.len();
    for i in 0..n {
        let (x0, y0) = pts[i];
        let (x1, y1) = pts[(i + 1) % n];
        area += x0 * y1 - x1 * y0;
    }
    (area * 0.5).abs()
}

/// Polygon perimeter, matching `shapely.Polygon.length`.
fn polygon_perimeter(pts: &[(f32, f32); 4]) -> f32 {
    let n = pts.len();
    (0..n).map(|i| dist(pts[i], pts[(i + 1) % n])).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a probability map with a filled rectangle of `1.0` inside a border of 0.
    fn rect_map(w: usize, h: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> Vec<f32> {
        let mut data = vec![0.0f32; w * h];
        for y in y0..y1 {
            for x in x0..x1 {
                data[y * w + x] = 1.0;
            }
        }
        data
    }

    #[test]
    fn extracts_single_box_from_rectangle_blob() {
        // 40x40 map, a 10..30 x 10..30 solid rectangle.
        let (w, h) = (40usize, 40usize);
        let data = rect_map(w, h, 10, 10, 30, 30);
        let map = ProbMap {
            data: &data,
            width: w,
            height: h,
        };
        let cfg = DbPostConfig::default();
        let boxes = boxes_from_bitmap(&map, &cfg, w as f32, h as f32);
        assert_eq!(boxes.len(), 1, "one blob -> one box");
        let b = &boxes[0];
        assert!(b.score > 0.9, "solid region scores near 1.0, got {}", b.score);

        // The unclipped box must be larger than the original 20x20 region.
        let xs: Vec<f32> = b.points.iter().map(|p| p.0).collect();
        let ys: Vec<f32> = b.points.iter().map(|p| p.1).collect();
        let bw = xs.iter().cloned().fold(f32::MIN, f32::max)
            - xs.iter().cloned().fold(f32::MAX, f32::min);
        let bh = ys.iter().cloned().fold(f32::MIN, f32::max)
            - ys.iter().cloned().fold(f32::MAX, f32::min);
        assert!(bw >= 20.0 && bh >= 20.0, "unclip should not shrink: {bw}x{bh}");
    }

    #[test]
    fn empty_map_yields_no_boxes() {
        let (w, h) = (16usize, 16usize);
        let data = vec![0.0f32; w * h];
        let map = ProbMap {
            data: &data,
            width: w,
            height: h,
        };
        let boxes = boxes_from_bitmap(&map, &DbPostConfig::default(), w as f32, h as f32);
        assert!(boxes.is_empty());
    }

    #[test]
    fn larger_unclip_ratio_grows_the_box() {
        let (w, h) = (60usize, 60usize);
        let data = rect_map(w, h, 20, 20, 40, 40);
        let map = ProbMap {
            data: &data,
            width: w,
            height: h,
        };

        let span = |cfg: &DbPostConfig| -> f32 {
            let boxes = boxes_from_bitmap(&map, cfg, w as f32, h as f32);
            let b = &boxes[0];
            let xs: Vec<f32> = b.points.iter().map(|p| p.0).collect();
            xs.iter().cloned().fold(f32::MIN, f32::max)
                - xs.iter().cloned().fold(f32::MAX, f32::min)
        };

        let small = DbPostConfig {
            unclip_ratio: 1.5,
            ..Default::default()
        };
        let large = DbPostConfig {
            unclip_ratio: 3.0,
            ..Default::default()
        };
        assert!(
            span(&large) > span(&small),
            "ratio 3.0 must dilate more than 1.5",
        );
    }

    #[test]
    fn box_thresh_filters_low_scoring_regions() {
        // A faint rectangle at 0.4 probability: above `thresh` (0.3) so it binarizes,
        // but below `box_thresh` (0.6) so it must be dropped.
        let (w, h) = (40usize, 40usize);
        let mut data = vec![0.0f32; w * h];
        for y in 10..30 {
            for x in 10..30 {
                data[y * w + x] = 0.4;
            }
        }
        let map = ProbMap {
            data: &data,
            width: w,
            height: h,
        };
        let boxes = boxes_from_bitmap(&map, &DbPostConfig::default(), w as f32, h as f32);
        assert!(boxes.is_empty(), "0.4 < box_thresh 0.6 -> dropped");
    }

    #[test]
    fn polygon_area_and_perimeter_of_unit_square() {
        let sq = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
        assert!((polygon_area(&sq) - 4.0).abs() < 1e-4);
        assert!((polygon_perimeter(&sq) - 8.0).abs() < 1e-4);
    }
}
