//! Bilinear sampling that matches OpenCV's convention.
//!
//! Every model in this crate is fed by a `cv2.resize` in the reference, and every
//! one of them defaults to `INTER_LINEAR`. Sampling nearest instead is not the
//! harmless approximation it looks like: on a ruled table the sampled pixel either
//! lands on a rule or misses it entirely, so thin ruling lines become aliasing
//! noise rather than lines. That was enough to invert the wired/wireless classifier
//! on a real table (see [`crate::cls`]), which is why this convention is shared
//! rather than re-approximated per model.
//!
//! Two details make it OpenCV's and not merely "some bilinear":
//! - **Half-pixel centres**: source coordinate is `(dst + 0.5) / scale - 0.5`, so
//!   pixel centres map to pixel centres.
//! - **`BORDER_REPLICATE`**: taps clamp at the edges instead of wrapping or
//!   sampling a border constant.

/// Resolves a continuous source coordinate to its two bilinear taps and the
/// interpolation weight, clamping at the edges (OpenCV's `BORDER_REPLICATE`).
///
/// Returns `(low, high, weight)` where the sample is
/// `pixel[low] * (1 - weight) + pixel[high] * weight`.
pub fn bilinear_taps(src: f32, extent: u32) -> (u32, u32, f32) {
    let max = extent.saturating_sub(1);
    let clamped = src.clamp(0.0, max as f32);
    let low = clamped.floor();
    let weight = clamped - low;
    let low = low as u32;
    (low, (low + 1).min(max), weight)
}

/// Maps a destination pixel index to its continuous source coordinate, using
/// OpenCV's half-pixel centre convention.
pub fn src_coord(dst: u32, scale: f32) -> f32 {
    (dst as f32 + 0.5) / scale - 0.5
}

/// Bilinearly samples one channel of an RGB image at `(sx, sy)`.
///
/// `sx`/`sy` are continuous source coordinates (see [`src_coord`]).
pub fn sample_channel(img: &image::RgbImage, sx: f32, sy: f32, channel: usize) -> f32 {
    let (x0, x1, wx) = bilinear_taps(sx, img.width());
    let (y0, y1, wy) = bilinear_taps(sy, img.height());
    let at = |x: u32, y: u32| -> f32 {
        img.get_pixel(x, y)
            .0
            .get(channel)
            .map(|v| f32::from(*v))
            .unwrap_or(0.0)
    };
    let top = at(x0, y0) * (1.0 - wx) + at(x1, y0) * wx;
    let bottom = at(x0, y1) * (1.0 - wx) + at(x1, y1) * wx;
    top * (1.0 - wy) + bottom * wy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taps_interpolate_between_neighbours() {
        let (low, high, w) = bilinear_taps(1.5, 10);
        assert_eq!((low, high), (1, 2));
        assert!((w - 0.5).abs() < 1e-6, "midpoint weights both taps equally");
    }

    #[test]
    fn taps_clamp_at_the_low_edge() {
        let (low, high, w) = bilinear_taps(-3.0, 10);
        assert_eq!((low, high), (0, 1));
        assert_eq!(w, 0.0, "clamped coordinate takes the edge pixel exactly");
    }

    #[test]
    fn taps_clamp_at_the_high_edge() {
        let (low, high, _) = bilinear_taps(99.0, 10);
        assert_eq!((low, high), (9, 9), "both taps pin to the last pixel");
    }

    #[test]
    fn taps_on_a_single_pixel_extent_stay_put() {
        let (low, high, _) = bilinear_taps(0.7, 1);
        assert_eq!((low, high), (0, 0));
    }

    /// The blend must actually average: a nearest sampler returns one endpoint.
    #[test]
    fn sample_blends_two_neighbours() {
        let mut img = image::RgbImage::new(2, 1);
        img.put_pixel(0, 0, image::Rgb([0, 0, 0]));
        img.put_pixel(1, 0, image::Rgb([100, 100, 100]));

        let v = sample_channel(&img, 0.5, 0.0, 0);
        assert!(
            (v - 50.0).abs() < 1e-3,
            "midway between 0 and 100 must be 50, got {v} — nearest would give 0 or 100"
        );
    }

    #[test]
    fn src_coord_uses_half_pixel_centres() {
        // Upscaling 2x: destination pixel 0's centre sits at source -0.25.
        assert!((src_coord(0, 2.0) - (-0.25)).abs() < 1e-6);
        assert!((src_coord(1, 2.0) - 0.25).abs() < 1e-6);
    }
}
