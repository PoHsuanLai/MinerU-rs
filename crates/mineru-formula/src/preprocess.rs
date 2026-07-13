//! Image preprocessing, a port of `UnimerSwinImageProcessor`.
//!
//! Python reference: `unimer_swin/image_processing_unimer_swin.py`. The pipeline
//! for one cropped formula image is:
//!
//! 1. **crop margins** — grayscale, min-max normalise, threshold at `< 200`, take
//!    the bounding box of the dark (text) pixels and crop to it;
//! 2. **resize** preserving aspect ratio so the result fits inside the target
//!    `[height, width]` (default `[192, 672]`);
//! 3. **pad** (centered) with black to exactly the target size;
//! 4. **grayscale + normalise** with UniMerNet's constants
//!    `(gray - 0.7931*255) / (0.1738*255)` into a single-channel `f32` tensor.
//!
//! The encoder wants 3 channels, so the entry point repeats this one channel three
//! times (mirroring `generate()`'s `pixel_values.repeat(1, 3, 1, 1)` in
//! `modeling_unimernet.py`). This module returns the single normalised channel as
//! a flat `Vec<f32>` in row-major `[H, W]` order plus its dimensions; turning that
//! into a Burn tensor lives in [`crate::model`] so this stays backend-free and
//! trivially unit-testable.
//!
//! # Fidelity
//! - Margin cropping matches the NumPy path (`crop_margin_numpy`).
//! - Resize uses a triangle (bilinear) filter; OpenCV's default `INTER_LINEAR` is
//!   also bilinear, so this is a faithful match up to boundary-sampling details.
//! - Normalisation constants are exact.

use image::imageops::FilterType;
use image::{DynamicImage, GrayImage, RgbImage};

use crate::error::{Error, Result};

/// UniMerNet grayscale-normalisation mean, in `[0, 1]` space (`0.7931`).
pub const NORM_MEAN: f32 = 0.7931;
/// UniMerNet grayscale-normalisation std, in `[0, 1]` space (`0.1738`).
pub const NORM_STD: f32 = 0.1738;
/// Threshold (on the 0-255 min-max-normalised gray image) below which a pixel
/// counts as "text" for margin cropping.
const CROP_THRESHOLD: u8 = 200;

/// A preprocessed single-channel image: `data` is row-major `f32` of length
/// `height * width`, already margin-cropped, resized, padded, and normalised.
#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    /// Normalised pixel values, row-major `[height, width]`.
    pub data: Vec<f32>,
    /// Output height (target height).
    pub height: usize,
    /// Output width (target width).
    pub width: usize,
}

/// Target output size `[height, width]` for the processor.
///
/// The Python `UnimerSwinImageProcessor` default; `modeling_unimernet.py`
/// constructs it with no arguments, so this is what runs.
pub const DEFAULT_TARGET: [usize; 2] = [192, 672];

/// Runs the full preprocessing pipeline on an RGB image.
///
/// Returns the single normalised channel. `target` is `[height, width]`.
///
/// # Errors
/// Returns [`Error::Image`] if the (post-crop) image is empty.
pub fn preprocess(image: &RgbImage, target: [usize; 2]) -> Result<PreprocessedImage> {
    let cropped = crop_margin(image);
    let (cw, ch) = (cropped.width(), cropped.height());
    if cw == 0 || ch == 0 {
        return Err(Error::Image("image is empty after margin crop".into()));
    }

    let [target_h, target_w] = target;
    // Scale to preserve aspect ratio, fitting inside the target box.
    let scale = f64::min(target_h as f64 / ch as f64, target_w as f64 / cw as f64);
    let new_w = ((cw as f64 * scale) as u32).max(1);
    let new_h = ((ch as f64 * scale) as u32).max(1);

    let resized = image::imageops::resize(&cropped, new_w, new_h, FilterType::Triangle);

    // Center-pad with black to the target size.
    let padded = pad_center(&resized, target_w as u32, target_h as u32);

    // Grayscale + normalise.
    let gray = DynamicImage::ImageRgb8(padded).into_luma8();
    let data = normalize(&gray);

    Ok(PreprocessedImage {
        data,
        height: target_h,
        width: target_w,
    })
}

/// Crops the margins of an RGB image, matching `crop_margin_numpy`.
///
/// Converts to gray, min-max normalises to `[0, 255]`, thresholds at `< 200` to
/// find text pixels, and crops to their bounding box. If the image is flat
/// (`max == min`) the original is returned unchanged.
pub fn crop_margin(image: &RgbImage) -> RgbImage {
    let gray = DynamicImage::ImageRgb8(image.clone()).into_luma8();
    let (w, h) = (gray.width(), gray.height());

    let (mut min_v, mut max_v) = (u8::MAX, u8::MIN);
    for &p in gray.as_raw() {
        min_v = min_v.min(p);
        max_v = max_v.max(p);
    }
    if max_v == min_v {
        return image.clone();
    }

    let range = (max_v - min_v) as f32;
    // Bounding box of dark ("text") pixels.
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0u32, 0u32);
    let mut found = false;
    for y in 0..h {
        for x in 0..w {
            let g = gray.get_pixel(x, y).0[0];
            let normalized = ((g - min_v) as f32 / range * 255.0) as u8;
            if normalized < CROP_THRESHOLD {
                found = true;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    if !found {
        return image.clone();
    }

    let cw = x1 - x0 + 1;
    let ch = y1 - y0 + 1;
    image::imageops::crop_imm(image, x0, y0, cw, ch).to_image()
}

/// Center-pads an RGB image with black to `(target_w, target_h)`.
///
/// If the image already meets or exceeds a target dimension, no padding is added
/// on that axis (the source is copied from the top-left, matching the fact that
/// the resize step guarantees `new <= target`).
fn pad_center(img: &RgbImage, target_w: u32, target_h: u32) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    let pad_w = target_w.saturating_sub(w);
    let pad_h = target_h.saturating_sub(h);
    let left = pad_w / 2;
    let top = pad_h / 2;

    let mut out = RgbImage::new(target_w, target_h); // zero-initialised == black
    for y in 0..h.min(target_h) {
        for x in 0..w.min(target_w) {
            out.put_pixel(x + left, y + top, *img.get_pixel(x, y));
        }
    }
    out
}

/// Applies UniMerNet grayscale normalisation, producing row-major `f32`.
fn normalize(gray: &GrayImage) -> Vec<f32> {
    let mean = NORM_MEAN * 255.0;
    let std = NORM_STD * 255.0;
    gray.as_raw()
        .iter()
        .map(|&p| (p as f32 - mean) / std)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn output_shape_matches_target() {
        // A 100x40 mid-gray image with a dark rectangle in the middle.
        let mut img = RgbImage::from_pixel(100, 40, Rgb([200, 200, 200]));
        for y in 10..30 {
            for x in 20..80 {
                img.put_pixel(x, y, Rgb([10, 10, 10]));
            }
        }
        let out = preprocess(&img, [192, 672]).expect("preprocess");
        assert_eq!(out.height, 192);
        assert_eq!(out.width, 672);
        assert_eq!(out.data.len(), 192 * 672);
    }

    #[test]
    fn normalization_constants_are_applied() {
        // A pure-white 8x8 image: after crop (flat -> unchanged) it pads/normalises.
        // White (255) -> (255 - 0.7931*255) / (0.1738*255).
        let img = RgbImage::from_pixel(8, 8, Rgb([255, 255, 255]));
        let out = preprocess(&img, [8, 8]).expect("preprocess");
        let expected = (255.0 - NORM_MEAN * 255.0) / (NORM_STD * 255.0);
        // The center pixels are white; check at least one equals the expected value.
        assert!(out.data.iter().any(|&v| (v - expected).abs() < 1e-3));
    }

    #[test]
    fn crop_margin_reduces_flat_border() {
        // Dark 20x20 block centered in a 60x60 white image.
        let mut img = RgbImage::from_pixel(60, 60, Rgb([255, 255, 255]));
        for y in 20..40 {
            for x in 20..40 {
                img.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
        let cropped = crop_margin(&img);
        // The dark block is 20x20; the crop bounding box should be exactly that.
        assert_eq!(cropped.width(), 20);
        assert_eq!(cropped.height(), 20);
    }

    #[test]
    fn flat_image_crop_is_identity() {
        let img = RgbImage::from_pixel(16, 16, Rgb([128, 128, 128]));
        let cropped = crop_margin(&img);
        assert_eq!(cropped.dimensions(), (16, 16));
    }
}
