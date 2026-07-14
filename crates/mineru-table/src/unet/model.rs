//! UNet line-segmentation model wrapper.
//!
//! The network is a conv/convtranspose/concat/sigmoid encoder-decoder that
//! `burn-onnx` codegen imported directly; the generated source is committed as
//! vendored code (see [`crate::generated::unet`]) and always compiled in. This
//! wrapper feeds the preprocessed 1024×1024 image through it and turns the
//! 3-class ruling-line mask into cell quadrilaterals. The model's `.bpk` weights
//! are fetched from the release and cached at runtime by [`crate::weights`].
//!
//! ## Status
//!
//! - **Neural forward pass** — the generated network is loaded (runtime-fetched
//!   `.bpk` weights) and run, producing the per-pixel 3-class argmax mask
//!   (`0` background / `1` horizontal / `2` vertical line). See
//!   [`UnetModel::segment_mask`].
//! - **Mask → polygon extraction** (morphology, 8-connectivity labeling,
//!   min-area-rect) is ported in [`super::extract`], so [`UnetModel::segment_cells`]
//!   runs the full wired-table segmentation → recovery → HTML path (see
//!   [`super::recover_and_render`]).

use image::RgbImage;

use crate::error::{Error, Result};
use crate::generated::unet as generated;

use super::recover::Poly;

/// UNet fixed input side (Python `inp_height = inp_width = 1024`).
pub const INPUT_SIDE: u32 = 1024;

/// The Burn CPU backend the vendored generated UNet compiles against.
///
/// The crate depends on `burn` with the `ndarray` feature, so the NdArray backend
/// is the one available here.
type Backend = burn::backend::NdArray<f32>;

/// A segmentation mask: per-pixel class indices (`0` background, `1` horizontal
/// line, `2` vertical line) laid out row-major over `height * width`.
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
    /// Whether this handle is expected to run inference. A `new()` handle reports
    /// [`Error::ModelUnavailable`] rather than trigger a weight fetch, matching
    /// the previous behaviour; `loaded()` handles run the real forward.
    ready: bool,
}

impl UnetModel {
    /// Creates a handle that reports the model unavailable; use [`Self::loaded`]
    /// to run inference.
    pub fn new() -> Self {
        Self { ready: false }
    }

    /// Creates a handle that runs the real UNet forward.
    ///
    /// The generated network is loaded with runtime-fetched weights on first use
    /// (cached for the process lifetime by [`Self::segment_mask`]).
    pub fn loaded() -> Self {
        Self { ready: true }
    }

    /// Runs segmentation and returns recovered cell quadrilaterals.
    ///
    /// Runs the neural forward pass ([`Self::segment_mask`]) and then the
    /// classical mask → polygon extraction
    /// ([`super::extract::extract_cell_polygons`]). A handle from [`Self::new`]
    /// (or a weight fetch/load failure) returns [`Error::ModelUnavailable`] /
    /// the relevant typed error rather than fabricate cells.
    pub fn segment_cells(&self, img: &RgbImage) -> Result<Vec<Poly>> {
        if !self.ready {
            return Err(Error::ModelUnavailable("unet"));
        }
        let mask = self.segment_mask(img)?;
        Ok(super::extract::extract_cell_polygons(
            &mask.classes,
            mask.width,
            mask.height,
        ))
    }

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
    /// for the process lifetime). It is the wired half of the wired-table path.
    pub fn segment_mask(&self, img: &RgbImage) -> Result<SegMask> {
        Self::run_mask(Self::preprocess(img))
    }

    /// Preprocesses `img` into the planar `[3, 1024, 1024]` `f32` buffer the UNet
    /// consumes.
    ///
    /// A `#[doc(hidden)]` parity hook: the `unet_real` numeric gate dumps this
    /// exact buffer so the Python ONNX dumper can feed the *identical* input to
    /// `onnxruntime` (the `image` crate's Triangle resize is not trivially
    /// reproducible in numpy, so both sides must consume the same bytes). Not
    /// part of the public API.
    #[doc(hidden)]
    pub fn debug_preprocess(img: &RgbImage) -> Vec<f32> {
        Self::preprocess(img)
    }

    /// Runs the generated UNet forward over an already-preprocessed CHW buffer.
    ///
    /// A `#[doc(hidden)]` parity hook mirroring [`Self::debug_preprocess`]: the
    /// `unet_real` gate runs this on the same buffer it dumped and diffs the
    /// resulting class mask against the committed ONNX reference. Not part of the
    /// public API.
    #[doc(hidden)]
    pub fn debug_segment_from_input(&self, input: Vec<f32>) -> Result<SegMask> {
        Self::run_mask(input)
    }

    /// Loads the generated UNet model with runtime-fetched weights, panic-free.
    ///
    /// Unlike the vendored `Model::from_bytes` (which `.expect()`s), this drives
    /// the [`burn_store::BurnpackStore`] directly and maps every failure to a
    /// typed [`Error`].
    fn load_unet(
        device: &burn::backend::ndarray::NdArrayDevice,
    ) -> Result<generated::Model<Backend>> {
        use burn_store::{BurnpackStore, ModuleSnapshot};

        let path = crate::weights::weight_path(crate::weights::TableWeight::Unet)?;
        let mut model = generated::Model::<Backend>::new(device);
        let mut store = BurnpackStore::from_file(&path);
        let result = model
            .load_from(&mut store)
            .map_err(|e| Error::WeightLoad(format!("UNet weights from {}: {e}", path.display())))?;
        if !result.errors.is_empty() || !result.missing.is_empty() {
            return Err(Error::WeightLoad(format!(
                "UNet weights from {}: {} apply error(s), {} missing tensor(s)",
                path.display(),
                result.errors.len(),
                result.missing.len()
            )));
        }
        Ok(model)
    }

    /// Runs the generated UNet forward over a preprocessed CHW buffer, returning
    /// the per-pixel 3-class argmax mask. The weights are cached for the process
    /// lifetime; repeated calls only pay for the forward pass.
    fn run_mask(input: Vec<f32>) -> Result<SegMask> {
        use burn::tensor::{Tensor, TensorData};
        use std::sync::OnceLock;

        // Cache the (weight-load-failable) model for the process lifetime. A
        // failed load is not cached, so a later call can retry.
        static MODEL: OnceLock<generated::Model<Backend>> = OnceLock::new();

        let device = burn::backend::ndarray::NdArrayDevice::default();
        let model = match MODEL.get() {
            Some(m) => m,
            None => {
                let m = Self::load_unet(&device)?;
                let _ = MODEL.set(m);
                MODEL.get().ok_or_else(|| {
                    Error::WeightLoad("UNet model cache unexpectedly empty".to_string())
                })?
            }
        };

        let sz = INPUT_SIDE as usize;
        let data = TensorData::new(input, [1, 3, sz, sz]);
        let x = Tensor::<Backend, 4>::from_data(data, &device);

        // The generated top-level `forward` already argmaxes to an int class mask
        // of shape `[1, N, H, W]`.
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
