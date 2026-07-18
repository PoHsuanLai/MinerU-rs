//! SLANet-plus image preprocessing.
//!
//! Port of `TablePreprocess`: resize so the longest side is 488, normalize with
//! ImageNet mean/std over `1/255`-scaled pixels (HWC order), pad to 488×488, then
//! emit CHW. Produces the `[3, 488, 488]` planar buffer the CNN backbone consumes
//! plus the original crop dimensions the decoder needs to rescale boxes.

use image::RgbImage;

/// SLANet-plus fixed square input side.
pub const TABLE_MAX_LEN: u32 = 488;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const SCALE: f32 = 1.0 / 255.0;

/// Preprocessed SLANet input.
pub struct Preprocessed {
    /// Planar CHW `f32` buffer of length `3 * 488 * 488`.
    pub chw: Vec<f32>,
    /// Original crop width (pixels), used to rescale regressed boxes.
    pub orig_w: f32,
    /// Original crop height (pixels).
    pub orig_h: f32,
}

/// Runs the SLANet-plus preprocessing on an RGB crop.
///
/// The resize is **bilinear**, matching the reference's bare
/// `cv2.resize(img, (resize_w, resize_h))` (`table_structure_utils.py:354`),
/// whose default interpolation is `INTER_LINEAR`. This previously sampled
/// nearest, on the reasoning that SLANet is robust to the interpolation choice —
/// which was an assumption, never a measurement, and the same assumption proved
/// wrong for the sibling classifier, where nearest sampling was enough to invert
/// its verdict on a real table (see [`crate::cls`]).
pub fn preprocess(img: &RgbImage) -> Preprocessed {
    let (w, h) = (img.width(), img.height());
    let orig_w = w as f32;
    let orig_h = h as f32;

    let ratio = TABLE_MAX_LEN as f32 / (w.max(h) as f32);
    let resize_w = ((w as f32) * ratio) as u32;
    let resize_h = ((h as f32) * ratio) as u32;
    let resize_w = resize_w.clamp(1, TABLE_MAX_LEN);
    let resize_h = resize_h.clamp(1, TABLE_MAX_LEN);

    let side = TABLE_MAX_LEN as usize;
    // Zero-padded CHW buffer (padding region stays 0, matching PaddingTableImage
    // which pads the *normalized* image with zeros).
    let mut chw = vec![0.0f32; 3 * side * side];

    for ry in 0..resize_h {
        let sy = crate::resample::src_coord(ry, ratio);
        for rx in 0..resize_w {
            let sx = crate::resample::src_coord(rx, ratio);
            for c in 0..3 {
                let sampled = crate::resample::sample_channel(img, sx, sy, c);
                let v = (sampled * SCALE - MEAN[c]) / STD[c];
                let dst = c * side * side + (ry as usize) * side + (rx as usize);
                if let Some(slot) = chw.get_mut(dst) {
                    *slot = v;
                }
            }
        }
    }

    Preprocessed {
        chw,
        orig_w,
        orig_h,
    }
}

/// Applies the SLANet-plus box rescale (`adapt_slanet_plus`).
///
/// The decoder's boxes are in the padded-488 frame; this maps them back onto the
/// original crop scale so they line up with OCR boxes.
pub fn adapt_slanet_plus(orig_w: f32, orig_h: f32, cell_bboxes: &mut [mineru_types::BBox]) {
    let resized = TABLE_MAX_LEN as f32;
    let ratio = (resized / orig_h).min(resized / orig_w);
    let w_ratio = resized / (orig_w * ratio);
    let h_ratio = resized / (orig_h * ratio);
    for b in cell_bboxes.iter_mut() {
        *b = mineru_types::BBox::new(b.x0 * w_ratio, b.y0 * h_ratio, b.x1 * w_ratio, b.y1 * h_ratio);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins bilinear resampling, matching the reference's `cv2.resize` default.
    ///
    /// The fixture **upscales** a small gradient. That is deliberate: when
    /// downscaling, a destination pixel's two taps are adjacent source pixels that
    /// usually sit on the same side of any edge, so bilinear and nearest agree
    /// almost everywhere and the test cannot see the difference. Upscaling puts
    /// destination pixels *between* source pixels, where bilinear must blend and
    /// nearest can only ever echo an original value.
    #[test]
    fn resamples_bilinearly_not_nearest() {
        // A horizontal ramp: each source column is a distinct value, so any blend
        // lands on a value no source pixel has.
        let mut img = RgbImage::new(8, 8);
        for y in 0..8u32 {
            for x in 0..8u32 {
                let v = (x * 30) as u8;
                img.put_pixel(x, y, image::Rgb([v, v, v]));
            }
        }

        let out = preprocess(&img);
        let side = TABLE_MAX_LEN as usize;
        // Undo the normalization to recover 0..255 samples from the first row.
        let sample = |x: usize| -> f32 {
            let v = out.chw.get(x).copied().unwrap_or_default();
            (v * STD[0] + MEAN[0]) / SCALE
        };

        // Values a nearest sampler could produce: exactly the source column values.
        let originals: Vec<f32> = (0..8).map(|x| (x * 30) as f32).collect();
        let is_original = |v: f32| originals.iter().any(|o| (o - v).abs() < 0.5);

        let ratio = TABLE_MAX_LEN as f32 / 8.0;
        let resize_w = ((8.0 * ratio) as usize).min(side);
        let blended = (0..resize_w).filter(|x| !is_original(sample(*x))).count();
        assert!(
            blended > 0,
            "every sample is an original source value — resize is nearest, not bilinear"
        );
    }

    #[test]
    fn output_is_correct_length_and_padded() {
        let img = RgbImage::new(200, 100);
        let out = preprocess(&img);
        let side = TABLE_MAX_LEN as usize;
        assert_eq!(out.chw.len(), 3 * side * side);
        assert_eq!(out.orig_w, 200.0);
        assert_eq!(out.orig_h, 100.0);
        // A black pixel normalizes to (0 - mean)/std; padded region is 0.
        // Bottom-right corner is padding (image only fills 488 x 244).
        let corner = out.chw[(side - 1) * side + (side - 1)];
        assert_eq!(corner, 0.0);
    }

    #[test]
    fn adapt_rescales_boxes() {
        // Square crop: ratio=1, so w_ratio=h_ratio=1 -> boxes unchanged.
        let mut boxes = vec![mineru_types::BBox::new(10.0, 20.0, 30.0, 40.0)];
        adapt_slanet_plus(488.0, 488.0, &mut boxes);
        assert_eq!(boxes[0], mineru_types::BBox::new(10.0, 20.0, 30.0, 40.0));
    }
}
