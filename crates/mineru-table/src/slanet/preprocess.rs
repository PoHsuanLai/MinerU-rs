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
/// The resize uses nearest-neighbour sampling for determinism; SLANet is robust
/// to the interpolation choice and this keeps the pure-Rust path dependency-free.
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
        // Nearest source row.
        let sy = ((ry as f32 + 0.5) / ratio - 0.5)
            .round()
            .clamp(0.0, (h - 1) as f32) as u32;
        for rx in 0..resize_w {
            let sx = ((rx as f32 + 0.5) / ratio - 0.5)
                .round()
                .clamp(0.0, (w - 1) as f32) as u32;
            let px = img.get_pixel(sx, sy);
            for c in 0..3 {
                let v = (px[c] as f32 * SCALE - MEAN[c]) / STD[c];
                let dst = c * side * side + (ry as usize) * side + (rx as usize);
                chw[dst] = v;
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
