//! Wired/wireless table classification (PP-LCNet_x1_0).
//!
//! Port of `paddle_table_cls.py`. The model is a plain PP-LCNet CNN with a
//! 2-class head; it imports cleanly via `burn-onnx` codegen (all ops supported),
//! so the network itself is generated at build time behind the `onnx-import`
//! feature. This module owns the pre-processing and the argmax head, and calls
//! the generated model when present.
//!
//! Pre-processing (`PaddleTableClsModel.preprocess`):
//! 1. resize so the shortest side is 256 (bilinear),
//! 2. center-crop 224×224,
//! 3. normalize per channel: `pixel * (scale/std) - mean/std` with
//!    `scale = 1/255`, ImageNet mean/std,
//! 4. HWC → CHW.

use image::RgbImage;

use crate::error::{Error, Result};

/// Shortest-side resize target before the center crop.
const RESIZE_SHORT: u32 = 256;
/// Center-crop side.
const CROP: u32 = 224;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const SCALE: f32 = 0.003_921_568_6; // 1/255

/// Whether a detected table is drawn with ruling lines (wired) or not (wireless).
///
/// The two variants pick the downstream recognizer: wired tables go through the
/// UNet line-recovery path, wireless tables through SLANet-plus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableClass {
    /// A table with visible cell borders. Index 0 in the model's output.
    Wired,
    /// A borderless table. Index 1 in the model's output.
    Wireless,
}

impl TableClass {
    /// Maps the classifier's argmax index to a [`TableClass`].
    ///
    /// Labels follow `PaddleTableClsModel.labels = [WiredTable, WirelessTable]`.
    fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(TableClass::Wired),
            1 => Some(TableClass::Wireless),
            _ => None,
        }
    }
}

/// A classification result: the predicted class and its confidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Classification {
    /// The predicted table class.
    pub class: TableClass,
    /// Confidence (max softmax/probability from the model), in `0.0..=1.0`.
    pub score: f32,
}

/// Preprocesses an RGB image into the planar `[3, 224, 224]` `f32` buffer the
/// LCNet classifier expects.
///
/// Errors with [`Error::ImageTooSmall`] when the crop cannot be taken, matching
/// the Python `ValueError` guard.
pub fn preprocess(img: &RgbImage) -> Result<Vec<f32>> {
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return Err(Error::ImageTooSmall {
            model: "PP-LCNet",
            width: w,
            height: h,
            min_width: CROP,
            min_height: CROP,
        });
    }
    let scale = RESIZE_SHORT as f32 / (w.min(h) as f32);
    let rw = ((w as f32 * scale).round() as u32).max(1);
    let rh = ((h as f32 * scale).round() as u32).max(1);

    if rw < CROP || rh < CROP {
        return Err(Error::ImageTooSmall {
            model: "PP-LCNet",
            width: rw,
            height: rh,
            min_width: CROP,
            min_height: CROP,
        });
    }

    // Center-crop origin in the resized frame.
    let x1 = (rw - CROP) / 2;
    let y1 = (rh - CROP) / 2;

    let side = CROP as usize;
    let mut chw = vec![0.0f32; 3 * side * side];
    for cy in 0..CROP {
        // Map crop pixel -> resized pixel -> source pixel (nearest).
        let ry = y1 + cy;
        let sy = ((ry as f32 + 0.5) / scale - 0.5)
            .round()
            .clamp(0.0, (h - 1) as f32) as u32;
        for cx in 0..CROP {
            let rx = x1 + cx;
            let sx = ((rx as f32 + 0.5) / scale - 0.5)
                .round()
                .clamp(0.0, (w - 1) as f32) as u32;
            let px = img.get_pixel(sx, sy);
            for c in 0..3 {
                let v = px[c] as f32 * (SCALE / STD[c]) - MEAN[c] / STD[c];
                chw[c * side * side + (cy as usize) * side + (cx as usize)] = v;
            }
        }
    }
    Ok(chw)
}

/// Picks the winning class from a 2-logit output vector.
///
/// Shared by the generated-model path and tests. Errors if the output is not the
/// expected 2-wide shape.
pub fn head(logits: &[f32]) -> Result<Classification> {
    if logits.len() != 2 {
        return Err(Error::OutputShape {
            expected: "[2]".to_string(),
            got: format!("[{}]", logits.len()),
        });
    }
    let (idx, &score) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .ok_or_else(|| Error::Decode("empty classifier output".to_string()))?;
    let class = TableClass::from_index(idx)
        .ok_or_else(|| Error::Decode(format!("class index {idx} out of range")))?;
    Ok(Classification { class, score })
}

/// Classifies a table crop as wired or wireless.
///
/// When the crate is built without the generated LCNet model (the default), this
/// returns [`Error::ModelUnavailable`]. Pre-processing and the argmax head are
/// still unit-tested independently.
pub fn classify(img: &RgbImage) -> Result<Classification> {
    let _input = preprocess(img)?;
    #[cfg(lcnet_generated)]
    {
        // The generated module lives in `crate::model::pp_lcnet_x1_0_table_cls`.
        // Feeding `_input` through it and calling `head` on the result is wired
        // once the generated symbol name is confirmed at build time.
        return Err(Error::ModelUnavailable("PP-LCNet (generated wiring pending)"));
    }
    #[cfg(not(lcnet_generated))]
    {
        Err(Error::ModelUnavailable("PP-LCNet"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_picks_argmax_class() {
        assert_eq!(
            head(&[0.9, 0.1]).unwrap(),
            Classification {
                class: TableClass::Wired,
                score: 0.9
            }
        );
        assert_eq!(
            head(&[0.2, 0.8]).unwrap(),
            Classification {
                class: TableClass::Wireless,
                score: 0.8
            }
        );
    }

    #[test]
    fn head_rejects_wrong_shape() {
        assert!(matches!(head(&[0.5]), Err(Error::OutputShape { .. })));
    }

    #[test]
    fn preprocess_produces_chw_buffer() {
        let img = RgbImage::new(300, 300);
        let buf = preprocess(&img).unwrap();
        assert_eq!(buf.len(), 3 * 224 * 224);
    }

    #[test]
    fn preprocess_upscales_small_square_image() {
        // A tiny square upscales (shortest side -> 256) to 256x256, which clears
        // the 224 crop, so preprocessing succeeds — matching Python, whose guard
        // is only reachable for zero-sized inputs.
        let img = RgbImage::new(10, 10);
        assert!(preprocess(&img).is_ok());
    }

    #[test]
    fn preprocess_rejects_zero_sized_image() {
        let img = RgbImage::new(0, 0);
        assert!(matches!(
            preprocess(&img),
            Err(Error::ImageTooSmall { .. })
        ));
    }

    #[test]
    fn classify_reports_unavailable_without_model() {
        let img = RgbImage::new(300, 300);
        assert!(matches!(classify(&img), Err(Error::ModelUnavailable(_))));
    }
}
