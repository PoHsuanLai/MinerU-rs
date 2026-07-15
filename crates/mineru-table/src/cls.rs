//! Wired/wireless table classification (PP-LCNet_x1_0).
//!
//! Port of `paddle_table_cls.py`. The model is a plain PP-LCNet CNN with a
//! 2-class head; it imports cleanly via `burn-onnx` codegen (all ops supported),
//! so the network is committed as vendored generated source (see
//! [`crate::generated::lcnet`]) and always compiled in. This module owns the
//! pre-processing and the argmax head, and runs the generated model with weights
//! fetched and cached at runtime by [`crate::weights`].
//!
//! Pre-processing (`PaddleTableClsModel.preprocess`):
//! 1. resize so the shortest side is 256 (bilinear),
//! 2. center-crop 224×224,
//! 3. normalize per channel: `pixel * (scale/std) - mean/std` with
//!    `scale = 1/255`, ImageNet mean/std,
//! 4. HWC → CHW.

use image::RgbImage;

use crate::error::{Error, Result};
use crate::resample::bilinear_taps;
use crate::generated::lcnet as generated;

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
        // Map crop pixel -> resized pixel -> continuous source coordinate, using
        // OpenCV's half-pixel convention. The resize is BILINEAR to match Python's
        // `cv2.resize(..., interpolation=1)` (`paddle_table_cls.py:34`): nearest
        // sampling here inverts the wired/wireless argmax on borderline crops,
        // because the aliased ruling lines it produces read as a borderless table.
        let ry = y1 + cy;
        let (sy0, sy1, wy) = bilinear_taps((ry as f32 + 0.5) / scale - 0.5, h);
        for cx in 0..CROP {
            let rx = x1 + cx;
            let (sx0, sx1, wx) = bilinear_taps((rx as f32 + 0.5) / scale - 0.5, w);

            let p00 = img.get_pixel(sx0, sy0);
            let p01 = img.get_pixel(sx1, sy0);
            let p10 = img.get_pixel(sx0, sy1);
            let p11 = img.get_pixel(sx1, sy1);

            for c in 0..3 {
                let top = p00[c] as f32 * (1.0 - wx) + p01[c] as f32 * wx;
                let bot = p10[c] as f32 * (1.0 - wx) + p11[c] as f32 * wx;
                let s = top * (1.0 - wy) + bot * wy;
                let v = s * (SCALE / STD[c]) - MEAN[c] / STD[c];
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
/// Runs the real generated LCNet forward on the Burn backend `B`. The model's
/// `.bpk` weights are fetched from the release and cached on first use (see
/// [`crate::weights`]); a fetch or load failure surfaces as a typed [`Error`].
/// Pre-processing and the argmax head are also unit-tested independently.
pub fn classify<B: burn::prelude::Backend>(img: &RgbImage) -> Result<Classification> {
    let input = preprocess(img)?;
    let out = debug_forward::<B>(input)?;
    head(&out)
}

/// Loads the generated LCNet model with runtime-fetched weights, panic-free.
///
/// Unlike the vendored `Model::from_bytes` (which `.expect()`s), this drives the
/// [`burn_store::BurnpackStore`] directly and maps every failure to a typed
/// [`Error`]. Used by the process-lifetime cache in [`debug_forward`].
fn load_lcnet<B: burn::prelude::Backend>(device: &B::Device) -> Result<generated::Model<B>> {
    use burn_store::{BurnpackStore, ModuleSnapshot};

    let path = crate::weights::weight_path(crate::weights::TableWeight::Lcnet)?;
    let mut model = generated::Model::<B>::new(device);
    let mut store = BurnpackStore::from_file(&path);
    let result = model
        .load_from(&mut store)
        .map_err(|e| Error::WeightLoad(format!("LCNet weights from {}: {e}", path.display())))?;
    if !result.errors.is_empty() || !result.missing.is_empty() {
        return Err(Error::WeightLoad(format!(
            "LCNet weights from {}: {} apply error(s), {} missing tensor(s)",
            path.display(),
            result.errors.len(),
            result.missing.len()
        )));
    }
    Ok(model)
}

/// Runs the generated LCNet forward over an already-preprocessed CHW buffer and
/// returns the raw 2-wide output vector, on the Burn backend `B`.
///
/// The output is post-*softmax* probabilities, not raw logits: the ONNX graph
/// (and therefore the generated forward) ends in a `Softmax`. This is a
/// `#[doc(hidden)]` parity hook — the `lcnet_real` numeric gate feeds the exact
/// same preprocessed input the Python ONNX dumper used and diffs this vector
/// against the committed reference dump. Not part of the public API.
///
/// The whole CNN + its weights are expensive to build, so the loaded model is
/// cached for the process lifetime (keyed by backend type — see
/// [`crate::model_cache`]); repeated calls only pay for the forward pass. The
/// weights come from the runtime-fetched cached `.bpk` ([`crate::weights`]); a
/// fetch/load failure is returned as a typed [`Error`].
#[doc(hidden)]
pub fn debug_forward<B: burn::prelude::Backend>(input: Vec<f32>) -> Result<Vec<f32>> {
    use burn::tensor::{Tensor, TensorData};

    // Cache the (weight-load-failable) model for the process lifetime, per backend
    // type. The first successful load wins; a failed load is not cached, so a later
    // call can retry (e.g. after the network recovers).
    let device = B::Device::default();
    let model: std::sync::Arc<generated::Model<B>> =
        crate::model_cache::get_or_try_init(|| load_lcnet::<B>(&device))?;

    // CHW `Vec<f32>` -> `[1, 3, 224, 224]` NCHW tensor.
    let side = CROP as usize;
    let data = TensorData::new(input, [1, 3, side, side]);
    let x = Tensor::<B, 4>::from_data(data, &device);
    model
        .forward(x)
        .into_data()
        .into_vec::<f32>()
        .map_err(|e| Error::Decode(format!("classifier output decode: {e:?}")))
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
    fn bilinear_taps_interpolate_between_neighbours() {
        // Exactly on a pixel centre: no blend, weight 0.
        assert_eq!(bilinear_taps(3.0, 10), (3, 4, 0.0));
        // Halfway: both taps weighted equally.
        assert_eq!(bilinear_taps(3.5, 10), (3, 4, 0.5));
        // Clamps at both edges (BORDER_REPLICATE), never sampling out of bounds.
        assert_eq!(bilinear_taps(-2.0, 10), (0, 1, 0.0));
        assert_eq!(bilinear_taps(99.0, 10), (9, 9, 0.0));
    }

    /// The resize must be BILINEAR, matching `cv2.resize(..., interpolation=1)`.
    ///
    /// Nearest sampling inverts the wired/wireless argmax on real borderline
    /// crops (verified: feeding nearest into the reference ONNX classifier flips
    /// p8 from wired=0.93 to wired=0.36). Pixel-exact assertions here would be a
    /// false-green trap, so this asserts the property that separates the two
    /// filters: on a horizontal gradient, downsampling by a non-integer factor
    /// must produce values that are NOT all exact source-pixel values — a blend
    /// only bilinear can create.
    #[test]
    fn preprocess_resamples_bilinearly_not_nearest() {
        // 512² downscales by exactly 2, so every sampled x lands on a .5 midpoint
        // between two adjacent source columns. Adjacent columns must therefore
        // differ for the blend to be observable: step by 8 per column, so a
        // midpoint averages to an odd multiple of 4 that no source pixel takes.
        let (w, h) = (512u32, 512u32);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = (x * 8 % 256) as u8;
                img.put_pixel(x, y, image::Rgb([v, v, v]));
            }
        }

        let buf = preprocess(&img).expect("gradient image preprocesses");

        // Invert the normalization back to raw 0..255 for channel 0.
        let side = CROP as usize;
        let raw: Vec<f32> = buf[..side * side]
            .iter()
            .map(|v| (v + MEAN[0] / STD[0]) / (SCALE / STD[0]))
            .collect();

        // Bilinear blending must produce at least one value strictly between two
        // adjacent source levels (i.e. not a multiple of 8). Nearest sampling can
        // only ever emit exact source levels, so it produces none and fails here.
        let blended = raw
            .iter()
            .filter(|v| {
                let r = v.rem_euclid(8.0);
                r > 0.5 && r < 7.5
            })
            .count();
        assert!(
            blended > 0,
            "no interpolated samples found — resize is nearest, not bilinear"
        );
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

    // `classify` now always runs the real forward, which triggers a runtime
    // weight fetch — a network operation not exercised by the unit tests. Its
    // success path is covered by the `#[ignore]`d `real_models`/`lcnet_real`
    // integration gates; the pre-processing and argmax head are unit-tested above.
}
