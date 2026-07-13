//! UNet line-segmentation model wrapper.
//!
//! The network is a conv/convtranspose/concat/sigmoid encoder-decoder that
//! `burn-onnx` codegen can import directly (behind the `onnx-import` feature);
//! this wrapper feeds the preprocessed 1024×1024 image through it and turns the
//! 3-class ruling-line mask into cell quadrilaterals.
//!
//! ## Status
//!
//! - **Neural forward pass** — wired under `onnx-import`: the generated network is
//!   loaded (embedded `.bpk` weights) and run, producing the per-pixel 3-class
//!   argmax mask (`0` background / `1` horizontal / `2` vertical line). See
//!   [`UnetModel::segment_mask`].
//! - **Mask → polygon extraction** (OpenCV morphology, 8-connectivity labeling,
//!   min-area-rect) is **not yet ported**, so [`UnetModel::segment_cells`] still
//!   reports the model unavailable rather than fabricate polygons. The downstream
//!   grid recovery + HTML assembly it feeds is fully implemented and tested (see
//!   [`super::recover_and_render`]).

use image::RgbImage;

use crate::error::{Error, Result};

use super::recover::Poly;

/// UNet fixed input side (Python `inp_height = inp_width = 1024`).
pub const INPUT_SIDE: u32 = 1024;

/// A segmentation mask: per-pixel class indices (`0` background, `1` horizontal
/// line, `2` vertical line) laid out row-major over `height * width`.
#[cfg(unet_generated)]
#[derive(Debug, Clone)]
pub struct SegMask {
    /// Mask height in pixels.
    pub height: usize,
    /// Mask width in pixels.
    pub width: usize,
    /// Per-pixel class indices, `height * width` entries, row-major.
    pub classes: Vec<i64>,
}

/// The wired-table line-segmentation model.
#[derive(Debug, Default)]
pub struct UnetModel {
    ready: bool,
}

impl UnetModel {
    /// Creates an unweighted model; `segment_cells` reports it unavailable.
    pub fn new() -> Self {
        Self { ready: false }
    }

    /// Creates a model with the embedded weights loaded.
    ///
    /// Under `onnx-import` the generated network is loadable and its forward pass
    /// runs (see [`UnetModel::segment_mask`]); without the feature this is
    /// identical to [`UnetModel::new`].
    #[cfg(unet_generated)]
    pub fn loaded() -> Self {
        Self { ready: true }
    }

    /// Runs segmentation and returns recovered cell quadrilaterals.
    ///
    /// The neural forward pass is wired (see [`UnetModel::segment_mask`]), but the
    /// mask → polygon extraction stage (morphology + connected components +
    /// min-area rect) is not yet ported, so this returns
    /// [`Error::ModelUnavailable`] rather than fabricate cells.
    pub fn segment_cells(&self, _img: &RgbImage) -> Result<Vec<Poly>> {
        if !self.ready {
            return Err(Error::ModelUnavailable("unet"));
        }
        // The forward pass produces a mask, but turning it into cell polygons
        // (OpenCV morphology / connected components / min-area rect) is unported.
        Err(Error::ModelUnavailable(
            "unet (mask→cell extraction not yet ported)",
        ))
    }
}

#[cfg(unet_generated)]
impl UnetModel {
    /// Preprocesses an RGB image into the planar `[3, 1024, 1024]` `f32` buffer
    /// the UNet expects: resize to `INPUT_SIDE` square, scale to `[0, 1]`, HWC →
    /// CHW. (Mirrors the Python `cv2.resize` + `/255` pipeline.)
    fn preprocess(img: &RgbImage) -> Vec<f32> {
        let side = INPUT_SIDE;
        let resized =
            image::imageops::resize(img, side, side, image::imageops::FilterType::Triangle);
        let sz = side as usize;
        let mut chw = vec![0.0f32; 3 * sz * sz];
        for y in 0..sz {
            for x in 0..sz {
                let px = resized.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    chw[c * sz * sz + y * sz + x] = px[c] as f32 / 255.0;
                }
            }
        }
        chw
    }

    /// Runs the generated UNet forward pass and returns the per-pixel 3-class
    /// argmax segmentation mask.
    ///
    /// This exercises the real neural network end-to-end (weight load is cached
    /// for the process lifetime). It is the wired half of the wired-table path;
    /// the mask → polygon post-processing that would consume this is still
    /// unported (see [`UnetModel::segment_cells`]).
    pub fn segment_mask(&self, img: &RgbImage) -> Result<SegMask> {
        use burn::tensor::{Tensor, TensorData};

        type B = burn::backend::NdArray<f32>;

        use crate::model::unet::Model;
        use std::sync::OnceLock;
        static MODEL: OnceLock<Model<B>> = OnceLock::new();

        let device = burn::backend::ndarray::NdArrayDevice::default();
        let model = MODEL.get_or_init(|| {
            let bytes = burn::tensor::Bytes::from_bytes_vec(
                include_bytes!(concat!(env!("OUT_DIR"), "/model/unet.bpk")).to_vec(),
            );
            Model::from_bytes(bytes, &device)
        });

        let input = Self::preprocess(img);
        let sz = INPUT_SIDE as usize;
        let data = TensorData::new(input, [1, 3, sz, sz]);
        let x = Tensor::<B, 4>::from_data(data, &device);

        // The generated top-level `forward` already argmaxes to an int class mask
        // of shape `[N, 1, H, W]`.
        let mask = model.forward(x);
        let dims = mask.dims();
        let (h, w) = (dims[2], dims[3]);
        let classes: Vec<i64> = mask
            .into_data()
            .into_vec::<i64>()
            .map_err(|e| Error::Decode(format!("unet mask decode: {e:?}")))?;
        Ok(SegMask {
            height: h,
            width: w,
            classes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unweighted_reports_unavailable() {
        let m = UnetModel::new();
        assert!(matches!(
            m.segment_cells(&RgbImage::new(32, 32)),
            Err(Error::ModelUnavailable("unet"))
        ));
    }
}
