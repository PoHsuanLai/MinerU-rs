//! UNet segmentation mask → cell-quadrilateral extraction.
//!
//! This is the classical-CV middle stage of the wired-table pipeline, ported from
//! `table_structure_unet.py::TSRUnet.postprocess` (and its `utils_table_line_rec`
//! helpers). It sits between the neural forward pass (which emits a per-pixel
//! 3-class ruling-line mask: `0` background / `1` horizontal / `2` vertical) and
//! the grid recovery in [`super::recover`].
//!
//! ## Algorithm (faithful to the Python order)
//!
//! 1. **Split** the mask into a horizontal-line image (`class == 1`) and a
//!    vertical-line image (`class == 2`).
//! 2. **Morphological close** each along its own axis to bridge gaps in the ruling
//!    lines: an anisotropic `(k, 1)` kernel for horizontals, `(1, k)` for
//!    verticals, with `k = round(sqrt(side) * 1.2)` exactly as the Python computes
//!    it. `imageproc::morphology` only offers isotropic `Norm`-disk kernels, so the
//!    one op it cannot express — a 1-D line structuring element — is hand-rolled as
//!    a separable dilate-then-erode along a single axis (that *is* a morphological
//!    close). Everything else reuses shared/`imageproc` code.
//! 3. **Extract line boxes** (`get_table_line`): 8-connected components of each
//!    line image, keep those long enough along their axis, and reduce each to an
//!    axis-aligned line segment via `min_area_rect` (the hoisted
//!    [`mineru_burn_common::geometry::min_area_rectangle`], i.e. `cv2.minAreaRect`).
//! 4. **Draw** all kept line segments onto a blank canvas.
//! 5. **Cell regions** (`cal_region_boxes`): 8-connected components of the
//!    *inverted* line canvas (the enclosed cell interiors), each reduced to a
//!    min-area quad, with the same small/large-area filters as the Python.
//!
//! ## Deliberately-omitted heuristics (honesty)
//!
//! The Python `postprocess` additionally runs `adjust_lines` / `final_adjust_lines`
//! / `line_to_line` (extend line endpoints so nearly-touching rulings connect) and
//! an optional deskew (`cal_rotate_angle` + `warpAffine` + `unrotate_polygons`).
//! Those are refinements on top of the core extraction; this port implements the
//! core faithfully and leaves the endpoint-extension and deskew heuristics out, so
//! the cell count can be *lower* than Python's on tables whose rulings have gaps or
//! are visibly rotated. See [`extract_cell_polygons`].

use image::{GrayImage, Luma};
use mineru_burn_common::geometry::min_area_rectangle;

use super::recover::Poly;

/// A component's axis-aligned bounding box in pixel space: `(min_x, min_y, max_x,
/// max_y)` (inclusive extents), plus the pixel coordinates that belong to it.
struct Component {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    /// `(x, y)` pixels in this component.
    coords: Vec<(u32, u32)>,
}

/// Labels the 8-connected foreground components of `mask` (foreground = any
/// non-zero pixel), returning one [`Component`] per label with its bbox + pixels.
///
/// This is the `cv2.connectedComponentsWithStats(connectivity=8)` step. We use a
/// direct union-find scan rather than `imageproc::region_labelling::connected_components`
/// so we get the per-label pixel coordinates (needed for `minAreaRect`) in one pass;
/// the labelling itself is the standard two-pass 8-connectivity algorithm.
fn connected_components(mask: &GrayImage) -> Vec<Component> {
    let (w, h) = mask.dimensions();
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let w = w as usize;
    let h = h as usize;

    // Union-find over pixel indices; only foreground pixels are ever unioned.
    let mut parent: Vec<u32> = (0..(w * h) as u32).collect();
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }
    fn union(parent: &mut [u32], a: u32, b: u32) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra as usize] = rb;
        }
    }

    let fg = |x: usize, y: usize| mask.get_pixel(x as u32, y as u32).0[0] != 0;

    for y in 0..h {
        for x in 0..w {
            if !fg(x, y) {
                continue;
            }
            let idx = (y * w + x) as u32;
            // Look at the four already-visited 8-neighbours (W, NW, N, NE).
            if x > 0 && fg(x - 1, y) {
                union(&mut parent, idx, (y * w + x - 1) as u32);
            }
            if y > 0 && fg(x, y - 1) {
                union(&mut parent, idx, ((y - 1) * w + x) as u32);
            }
            if y > 0 && x > 0 && fg(x - 1, y - 1) {
                union(&mut parent, idx, ((y - 1) * w + x - 1) as u32);
            }
            if y > 0 && x + 1 < w && fg(x + 1, y - 1) {
                union(&mut parent, idx, ((y - 1) * w + x + 1) as u32);
            }
        }
    }

    // Bucket foreground pixels by their root label.
    use std::collections::HashMap;
    let mut buckets: HashMap<u32, Component> = HashMap::new();
    for y in 0..h {
        for x in 0..w {
            if !fg(x, y) {
                continue;
            }
            let root = find(&mut parent, (y * w + x) as u32);
            let (xu, yu) = (x as u32, y as u32);
            let c = buckets.entry(root).or_insert_with(|| Component {
                min_x: xu,
                min_y: yu,
                max_x: xu,
                max_y: yu,
                coords: Vec::new(),
            });
            c.min_x = c.min_x.min(xu);
            c.min_y = c.min_y.min(yu);
            c.max_x = c.max_x.max(xu);
            c.max_y = c.max_y.max(yu);
            c.coords.push((xu, yu));
        }
    }
    buckets.into_values().collect()
}

/// Separable 1-D morphological close along one axis with a length-`k` line kernel.
///
/// `cv2.morphologyEx(MORPH_CLOSE, getStructuringElement(MORPH_RECT, (k,1)))` for a
/// horizontal kernel (or `(1,k)` vertical) is a dilate-then-erode with a 1-D line
/// structuring element — the one structuring element `imageproc`'s `Norm`-disk
/// morphology cannot express. It is a max over the window (dilate) followed by a
/// min over the window (erode); on a binary image that closes gaps up to `k`.
///
/// `horizontal = true` runs the window along x (rows); `false` along y (columns).
/// A kernel of length `< 2` is a no-op (returns a clone).
fn morph_close_1d(mask: &GrayImage, k: u32, horizontal: bool) -> GrayImage {
    if k < 2 {
        return mask.clone();
    }
    let dilated = morph_pass_1d(mask, k, horizontal, true);
    morph_pass_1d(&dilated, k, horizontal, false)
}

/// One separable morphological pass: `dilate = true` takes the window max, `false`
/// the window min. The window is centred (OpenCV's default anchor) and length `k`.
fn morph_pass_1d(src: &GrayImage, k: u32, horizontal: bool, dilate: bool) -> GrayImage {
    let (w, h) = src.dimensions();
    let mut out = GrayImage::new(w, h);
    let half = (k / 2) as i32;
    // OpenCV anchors an even kernel just left/above centre; k/2 each side is the
    // standard symmetric choice and the difference is sub-pixel for gap closing.
    for y in 0..h {
        for x in 0..w {
            let mut acc: u8 = if dilate { 0 } else { 255 };
            for d in -half..=half {
                // Clamp out-of-bounds samples to the border pixel (OpenCV's
                // `BORDER_REPLICATE`), so a close doesn't erode ruling lines away at
                // the image edges (a zero-pad border would shrink every line by
                // half the kernel and detach the outer cells).
                let (sx, sy) = if horizontal {
                    ((x as i32 + d).clamp(0, w as i32 - 1), y as i32)
                } else {
                    (x as i32, (y as i32 + d).clamp(0, h as i32 - 1))
                };
                let v = src.get_pixel(sx as u32, sy as u32).0[0];
                if dilate {
                    acc = acc.max(v);
                } else {
                    acc = acc.min(v);
                }
            }
            out.put_pixel(x, y, Luma([acc]));
        }
    }
    out
}

/// An axis-aligned line segment `[xmin, ymin, xmax, ymax]` (the `min_area_rect`
/// reduction the Python uses for ruling lines).
type LineBox = [f32; 4];

/// `min_area_rect` for a ruling-line component: min-area quad → axis-aligned segment.
///
/// Ports `utils_table_line_rec.min_area_rect`: take the component's min-area
/// rectangle, then collapse it to a segment along its long axis (the mid-line
/// between the two long edges), returned as `[xmin, ymin, xmax, ymax]`.
fn line_min_area_rect(coords: &[(u32, u32)]) -> Option<LineBox> {
    let pts: Vec<(f64, f64)> = coords.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
    let (rect, _) = min_area_rectangle(&pts)?;
    // rect corners are [min_u/min_v, max_u/min_v, max_u/max_v, min_u/max_v] in the
    // rotated frame; sort into image order via the same ordering the Python uses.
    let ordered = order_points(rect);
    let [ (x1, y1), (x2, y2), (x3, y3), (x4, y4) ] = ordered;
    // Long axis: compare the width (tl->tr) against the height (tl->bl).
    let w = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
    let h = ((x4 - x1).powi(2) + (y4 - y1).powi(2)).sqrt();
    // Corner order is [tl, tr, br, bl] = (x1..x4). This mirrors the Python
    // `min_area_rect`: when the box is taller than wide it is a vertical ruling, so
    // the segment runs top→bottom through the midpoints of the two horizontal
    // (tl-tr, bl-br) edges; otherwise it runs left→right through the midpoints of
    // the two vertical (tl-bl, tr-br) edges.
    let (xmin, ymin, xmax, ymax) = if w < h {
        // Vertical ruling: (tl+tr)/2 -> (br+bl)/2.
        (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            (x3 + x4) / 2.0,
            (y3 + y4) / 2.0,
        )
    } else {
        // Horizontal ruling: (tl+bl)/2 -> (tr+br)/2.
        (
            (x1 + x4) / 2.0,
            (y1 + y4) / 2.0,
            (x2 + x3) / 2.0,
            (y2 + y3) / 2.0,
        )
    };
    Some([xmin as f32, ymin as f32, xmax as f32, ymax as f32])
}

/// Orders four rect corners into `[top-left, top-right, bottom-right, bottom-left]`,
/// mirroring `utils_table_line_rec._order_points`.
fn order_points(pts: [(f64, f64); 4]) -> [(f64, f64); 4] {
    let mut p = pts;
    // Sort by x.
    p.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Left two, right two; within left, smaller y is top-left.
    let (mut l0, mut l1) = (p[0], p[1]);
    if l0.1 > l1.1 {
        std::mem::swap(&mut l0, &mut l1);
    }
    let (tl, bl) = (l0, l1);
    // Of the right two, the farther from tl is bottom-right, nearer is top-right.
    let (r0, r1) = (p[2], p[3]);
    let d0 = (r0.0 - tl.0).powi(2) + (r0.1 - tl.1).powi(2);
    let d1 = (r1.0 - tl.0).powi(2) + (r1.1 - tl.1).powi(2);
    let (br, tr) = if d0 >= d1 { (r0, r1) } else { (r1, r0) };
    [tl, tr, br, bl]
}

/// Extracts axis-aligned line segments from a line image (`get_table_line`).
///
/// `axis_horizontal = true` keeps components wider than `line_w` (horizontal
/// rulings); `false` keeps components taller than `line_w` (vertical rulings).
fn get_table_line(line_img: &GrayImage, axis_horizontal: bool, line_w: u32) -> Vec<LineBox> {
    let comps = connected_components(line_img);
    let mut boxes = Vec::new();
    for c in comps {
        let width = c.max_x - c.min_x;
        let height = c.max_y - c.min_y;
        let keep = if axis_horizontal {
            width > line_w
        } else {
            height > line_w
        };
        if !keep {
            continue;
        }
        if let Some(b) = line_min_area_rect(&c.coords) {
            boxes.push(b);
        }
    }
    boxes
}

/// Draws a line segment onto `img` (foreground = 255) with the given half-width,
/// using a simple Bresenham walk thickened perpendicular by `line_w` pixels.
fn draw_line(img: &mut GrayImage, b: &LineBox, line_w: i32) {
    let (w, h) = img.dimensions();
    let (x0, y0, x1, y1) = (
        b[0].round() as i32,
        b[1].round() as i32,
        b[2].round() as i32,
        b[3].round() as i32,
    );
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let (mut x, mut y) = (x0, y0);
    let half = (line_w / 2).max(0);
    loop {
        for oy in -half..=half {
            for ox in -half..=half {
                let (px, py) = (x + ox, y + oy);
                if px >= 0 && py >= 0 && (px as u32) < w && (py as u32) < h {
                    img.put_pixel(px as u32, py as u32, Luma([255]));
                }
            }
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

/// Inverts a line canvas (background where lines are absent) so connected
/// components of the result are the enclosed cell interiors.
fn invert(img: &GrayImage) -> GrayImage {
    let (w, h) = img.dimensions();
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = img.get_pixel(x, y).0[0];
            out.put_pixel(x, y, Luma([if v == 0 { 255 } else { 0 }]));
        }
    }
    out
}

/// A cell quad ordered top-left, top-right, bottom-right, bottom-left, with its
/// axis-aligned width/height for filtering (`min_area_rect_box_from_components`).
struct CellQuad {
    corners: [(f32, f32); 4],
    w: f32,
    h: f32,
}

/// Reduces a cell-interior component to a min-area quad (`cv2.minAreaRect` +
/// `boxPoints`), ordered and measured like the Python `min_area_rect_box`.
fn cell_min_area_rect(coords: &[(u32, u32)]) -> Option<CellQuad> {
    let pts: Vec<(f64, f64)> = coords.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
    let (rect, _) = min_area_rectangle(&pts)?;
    let ordered = order_points(rect);
    let [ (x1, y1), (x2, y2), (x3, y3), (x4, y4) ] = ordered;
    // width = mean of the two horizontal edges, height = mean of the two vertical.
    let w = (((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
        + ((x3 - x4).powi(2) + (y3 - y4).powi(2)).sqrt())
        / 2.0;
    let h = (((x2 - x3).powi(2) + (y2 - y3).powi(2)).sqrt()
        + ((x1 - x4).powi(2) + (y1 - y4).powi(2)).sqrt())
        / 2.0;
    Some(CellQuad {
        corners: [
            (x1 as f32, y1 as f32),
            (x2 as f32, y2 as f32),
            (x3 as f32, y3 as f32),
            (x4 as f32, y4 as f32),
        ],
        w: w as f32,
        h: h as f32,
    })
}

/// `cal_region_boxes`: cell quads from the inverted line canvas.
///
/// Applies the same filters as `min_area_rect_box_from_components`:
/// - drop components whose bbox area exceeds `3/4 * W * H` (the whole-table blob),
/// - keep quads whose `w * h < 0.5 * W * H`,
/// - with `filtersmall`, drop quads with `w < 15` or `h < 15`.
fn cal_region_boxes(line_img: &GrayImage) -> Vec<Poly> {
    let (w, h) = line_img.dimensions();
    let (wf, hf) = (w as f32, h as f32);
    let inverted = invert(line_img);
    let comps = connected_components(&inverted);

    let mut polys = Vec::new();
    for c in comps {
        let bbox_w = (c.max_x - c.min_x + 1) as f32;
        let bbox_h = (c.max_y - c.min_y + 1) as f32;
        let bbox_area = bbox_w * bbox_h;
        if bbox_area > wf * hf * 3.0 / 4.0 {
            continue; // whole-table region
        }
        let quad = match cell_min_area_rect(&c.coords) {
            Some(q) => q,
            None => continue,
        };
        if quad.w * quad.h >= 0.5 * wf * hf {
            continue;
        }
        if quad.w < 15.0 || quad.h < 15.0 {
            continue; // filtersmall
        }
        polys.push(to_recover_poly(&quad.corners));
    }
    polys
}

/// Converts a `[tl, tr, br, bl]` corner quad into the recover-module [`Poly`]
/// convention `[top-left, bottom-left, bottom-right, top-right]`.
///
/// The Python `__call__` performs the same swap after `postprocess`: it swaps
/// corner 1 and 3 so the CCW order the recovery expects is
/// `tl, bl, br, tr`.
fn to_recover_poly(c: &[(f32, f32); 4]) -> Poly {
    // c = [tl, tr, br, bl]  ->  [tl, bl, br, tr]
    [
        [c[0].0, c[0].1], // top-left
        [c[3].0, c[3].1], // bottom-left
        [c[2].0, c[2].1], // bottom-right
        [c[1].0, c[1].1], // top-right
    ]
}

/// Rounds a kernel side the way the Python does: `int(sqrt(side) * 1.2)`.
fn kernel_len(side: u32) -> u32 {
    ((side as f64).sqrt() * 1.2) as u32
}

/// Turns a 3-class ruling-line mask into cell-quadrilaterals.
///
/// `classes` is row-major `mask_h * mask_w` with values `0` background, `1`
/// horizontal line, `2` vertical line (as produced by the UNet forward pass). The
/// returned polygons are in mask-pixel coordinates in the [`Poly`] convention the
/// recovery stage consumes.
///
/// Implements the core of `TSRUnet.postprocess`; see the module docs for the
/// endpoint-extension and deskew heuristics that are intentionally not ported.
pub fn extract_cell_polygons(classes: &[i64], mask_w: usize, mask_h: usize) -> Vec<Poly> {
    if mask_w == 0 || mask_h == 0 || classes.len() != mask_w * mask_h {
        return Vec::new();
    }
    let (w, h) = (mask_w as u32, mask_h as u32);

    // 1. Split into horizontal (class 1) and vertical (class 2) line images.
    let mut hpred = GrayImage::new(w, h);
    let mut vpred = GrayImage::new(w, h);
    for y in 0..mask_h {
        for x in 0..mask_w {
            match classes[y * mask_w + x] {
                1 => hpred.put_pixel(x as u32, y as u32, Luma([255])),
                2 => vpred.put_pixel(x as u32, y as u32, Luma([255])),
                _ => {}
            }
        }
    }

    // 2. Anisotropic morphological close per axis (k = int(sqrt(side) * 1.2)).
    let hors_k = kernel_len(w);
    let vert_k = kernel_len(h);
    let vpred = morph_close_1d(&vpred, vert_k, false); // vertical kernel (1, k)
    let hpred = morph_close_1d(&hpred, hors_k, true); // horizontal kernel (k, 1)

    // 3. Line boxes (get_table_line). Python lineW: col=30 (vertical), row=50 (horizontal).
    const COL_LINE_W: u32 = 30;
    const ROW_LINE_W: u32 = 50;
    let colboxes = get_table_line(&vpred, false, COL_LINE_W);
    let rowboxes = get_table_line(&hpred, true, ROW_LINE_W);

    // 4. Draw every line onto a blank canvas (lineW = 2 in Python).
    let mut line_img = GrayImage::new(w, h);
    for b in rowboxes.iter().chain(colboxes.iter()) {
        draw_line(&mut line_img, b, 2);
    }

    // 5. Cell regions from the inverted canvas (cal_region_boxes).
    cal_region_boxes(&line_img)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a class mask with a `rows x cols` grid of ruling lines over `side`.
    /// Horizontal rulings are class 1, vertical are class 2.
    fn grid_mask(side: usize, rows: usize, cols: usize) -> Vec<i64> {
        let mut m = vec![0i64; side * side];
        // Horizontal lines at evenly spaced y.
        for r in 0..=rows {
            let y = (r * (side - 1)) / rows;
            for x in 0..side {
                m[y * side + x] = 1;
            }
        }
        // Vertical lines at evenly spaced x.
        for c in 0..=cols {
            let x = (c * (side - 1)) / cols;
            for y in 0..side {
                m[y * side + x] = 2;
            }
        }
        m
    }

    #[test]
    fn empty_mask_yields_no_cells() {
        let polys = extract_cell_polygons(&vec![0i64; 64 * 64], 64, 64);
        assert!(polys.is_empty());
    }

    #[test]
    fn bad_shape_yields_no_cells() {
        assert!(extract_cell_polygons(&[0, 1, 2], 4, 4).is_empty());
    }

    #[test]
    fn grid_produces_expected_cell_count() {
        // 300x300 grid, 3 rows x 3 cols of cells = 9 interior cells.
        let side = 300;
        let m = grid_mask(side, 3, 3);
        let polys = extract_cell_polygons(&m, side, side);
        // Expect ~9 cells; allow slack for border components dropped by filters.
        assert!(
            (6..=12).contains(&polys.len()),
            "expected ~9 cells, got {}",
            polys.len()
        );
        // Every cell quad must be within the mask bounds.
        for p in &polys {
            for [x, y] in p {
                assert!(*x >= -1.0 && *x <= side as f32 + 1.0, "x {x}");
                assert!(*y >= -1.0 && *y <= side as f32 + 1.0, "y {y}");
            }
        }
    }

    #[test]
    fn morph_close_1d_bridges_horizontal_gap() {
        // A horizontal line with a 3-px gap; a k=7 close should bridge it.
        let mut img = GrayImage::new(20, 3);
        for x in 0..8 {
            img.put_pixel(x, 1, Luma([255]));
        }
        for x in 11..20 {
            img.put_pixel(x, 1, Luma([255]));
        }
        let closed = morph_close_1d(&img, 7, true);
        // The gap pixels (8,9,10) at y=1 should now be filled.
        assert_eq!(closed.get_pixel(9, 1).0[0], 255, "gap should be bridged");
    }

    #[test]
    fn kernel_len_matches_python_formula() {
        // int(sqrt(1024) * 1.2) = int(32 * 1.2) = int(38.4) = 38.
        assert_eq!(kernel_len(1024), 38);
    }

    #[test]
    fn extract_then_recover_renders_grid_html() {
        use crate::unet::{recover, COL_THRESHOLD, ROW_THRESHOLD};
        use std::collections::HashMap;

        let side = 300;
        let polys = extract_cell_polygons(&grid_mask(side, 3, 3), side, side);
        assert!(!polys.is_empty(), "grid must yield cells");

        let logic = recover(&polys, ROW_THRESHOLD, COL_THRESHOLD);
        assert_eq!(logic.len(), polys.len());
        // A 3x3 grid should recover 3 distinct row starts and 3 col starts.
        let max_row = logic.iter().map(|l| l.row_end).max().unwrap_or(0);
        let max_col = logic.iter().map(|l| l.col_end).max().unwrap_or(0);
        assert!(max_row >= 2, "expected >=3 rows, got max_row {max_row}");
        assert!(max_col >= 2, "expected >=3 cols, got max_col {max_col}");

        let text: HashMap<usize, Vec<String>> = HashMap::new();
        let html = crate::unet::plot_html_table(&logic, &text);
        assert!(html.contains("<table>"), "html: {html}");
        assert!(html.contains("</table>"));
    }
}
