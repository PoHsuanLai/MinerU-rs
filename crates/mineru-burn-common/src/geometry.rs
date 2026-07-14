//! Shared 2-D geometry primitives for the vision post-processors.
//!
//! Why this lives in the shared harness: several model crates need OpenCV's
//! `cv2.minAreaRect` + `cv2.boxPoints` semantics (text-line detection, wired-table
//! line extraction, …) to turn a point cloud into an oriented bounding quad. The
//! obvious candidate — `imageproc::geometry::min_area_rect` — **panics** on the
//! degenerate / collinear convex hulls that axis-aligned pixel blobs routinely
//! produce (Rust's total-order-checked sort trips on the ties). Rather than each
//! crate carrying its own copy, the robust rotating-calipers implementation lives
//! here once and is reused everywhere.

/// Minimum-area bounding rectangle via convex hull + rotating calipers.
///
/// A self-contained port of `cv2.minAreaRect` semantics that tolerates the
/// degenerate / collinear hulls (points, segments, axis-aligned boxes) which make
/// `imageproc`'s version panic. Returns the four rectangle corners (in the rotated
/// frame's `min_u/min_v … min_u/max_v` order) and the length of the shorter side,
/// or [`None`] if `points` is empty.
///
/// Points are `(x, y)` in `f64`. The returned corners are `(x, y)` as well.
pub fn min_area_rectangle(points: &[(f64, f64)]) -> Option<([(f64, f64); 4], f64)> {
    let hull = convex_hull(points);
    if hull.is_empty() {
        return None;
    }
    if hull.len() < 3 {
        // A point or a segment: return a degenerate rect with shorter side 0.
        let p = hull[0];
        let q = *hull.last().unwrap_or(&p);
        return Some(([p, q, q, p], 0.0));
    }

    let mut best_area = f64::MAX;
    let mut best_rect = [(0.0, 0.0); 4];
    let mut best_sside = 0.0;

    let n = hull.len();
    for i in 0..n {
        let a = hull[i];
        let b = hull[(i + 1) % n];
        // Edge direction (unit vector); skip zero-length edges.
        let (ex, ey) = (b.0 - a.0, b.1 - a.1);
        let len = (ex * ex + ey * ey).sqrt();
        if len < 1e-9 {
            continue;
        }
        let (ux, uy) = (ex / len, ey / len);
        // Perpendicular unit vector.
        let (px, py) = (-uy, ux);

        // Project every hull point onto (u, p) to get the rotated bbox extents.
        let (mut min_u, mut max_u, mut min_v, mut max_v) =
            (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for &(x, y) in &hull {
            let proj_u = x * ux + y * uy;
            let proj_v = x * px + y * py;
            min_u = min_u.min(proj_u);
            max_u = max_u.max(proj_u);
            min_v = min_v.min(proj_v);
            max_v = max_v.max(proj_v);
        }
        let w = max_u - min_u;
        let h = max_v - min_v;
        let area = w * h;
        if area < best_area {
            best_area = area;
            best_sside = w.min(h);
            // Reconstruct the four corners in (u, v) space, map back to (x, y).
            let to_xy = |cu: f64, cv: f64| (cu * ux + cv * px, cu * uy + cv * py);
            best_rect = [
                to_xy(min_u, min_v),
                to_xy(max_u, min_v),
                to_xy(max_u, max_v),
                to_xy(min_u, max_v),
            ];
        }
    }
    Some((best_rect, best_sside))
}

/// Andrew's monotone-chain convex hull, counter-clockwise, no repeated endpoint.
///
/// Uses `total_cmp` for a strict total order so the sort never trips on ties/NaN,
/// which is exactly the case `imageproc`'s hull mishandles.
pub fn convex_hull(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut pts: Vec<(f64, f64)> = points.to_vec();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    pts.dedup();
    let n = pts.len();
    if n < 3 {
        return pts;
    }

    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };

    let mut lower: Vec<(f64, f64)> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<(f64, f64)> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hull_of_axis_aligned_square_keeps_four_corners() {
        // A filled 3x3 grid of points: hull is the 4 corners (interior dropped).
        let mut pts = Vec::new();
        for y in 0..3 {
            for x in 0..3 {
                pts.push((x as f64, y as f64));
            }
        }
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4, "square hull has 4 corners, got {hull:?}");
    }

    #[test]
    fn min_area_rect_of_axis_aligned_box_matches_extent() {
        // Points on a 10x4 axis-aligned rectangle border.
        let mut pts = Vec::new();
        for x in 0..=10 {
            pts.push((x as f64, 0.0));
            pts.push((x as f64, 4.0));
        }
        let (rect, sside) = min_area_rectangle(&pts).expect("non-empty");
        // Shorter side is the height (4).
        assert!((sside - 4.0).abs() < 1e-6, "sside {sside}");
        // All corners lie on the axis-aligned extent.
        for &(x, y) in &rect {
            assert!((-1e-6..=10.0 + 1e-6).contains(&x), "x {x}");
            assert!((-1e-6..=4.0 + 1e-6).contains(&y), "y {y}");
        }
    }

    #[test]
    fn empty_is_none() {
        assert!(min_area_rectangle(&[]).is_none());
    }

    #[test]
    fn single_point_is_degenerate() {
        let (rect, sside) = min_area_rectangle(&[(2.0, 3.0)]).expect("one point");
        assert_eq!(sside, 0.0);
        assert!(rect.iter().all(|&p| p == (2.0, 3.0)));
    }
}
