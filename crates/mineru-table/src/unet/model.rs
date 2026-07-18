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

use std::marker::PhantomData;

use burn::prelude::Backend;
use image::RgbImage;

use crate::error::{Error, Result};
use crate::generated::unet as generated;

use super::recover::Poly;

/// UNet fixed input side (Python `inp_height = inp_width = 1024`).
pub const INPUT_SIDE: u32 = 1024;

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
///
/// Generic over the Burn backend `B`; in practice the pipeline instantiates it on
/// [`Cpu`](mineru_burn_common::backend::Cpu). The handle itself carries no
/// parameters (the loaded network is cached for the process lifetime, keyed by `B`
/// — see [`crate::model_cache`]); the `B` type only selects which cached instance
/// the forward pass uses.
#[derive(Debug)]
pub struct UnetModel<B: Backend> {
    /// Whether this handle is expected to run inference. A `new()` handle reports
    /// [`Error::ModelUnavailable`] rather than trigger a weight fetch, matching
    /// the previous behaviour; `loaded()` handles run the real forward.
    ready: bool,
    _backend: PhantomData<B>,
}

// `#[derive(Default)]` would require `B: Default`; `PhantomData<B>` is `Default`
// for any `B`, so a hand-written impl over any backend is both correct and less
// constrained.
impl<B: Backend> Default for UnetModel<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> UnetModel<B> {
    /// Creates a handle that reports the model unavailable; use [`Self::loaded`]
    /// to run inference.
    pub fn new() -> Self {
        Self {
            ready: false,
            _backend: PhantomData,
        }
    }

    /// Creates a handle that runs the real UNet forward.
    ///
    /// The generated network is loaded with runtime-fetched weights on first use
    /// (cached for the process lifetime by [`Self::segment_mask`]).
    pub fn loaded() -> Self {
        Self {
            ready: true,
            _backend: PhantomData,
        }
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

    /// Preprocesses an RGB image into the planar `[3, H, W]` `f32` buffer the UNet
    /// expects, returning it with its dimensions.
    ///
    /// Two details are load-bearing and were both wrong before, each on its own
    /// enough to wreck the mask:
    ///
    /// - **Aspect ratio is preserved.** `resize_img(..., keep_ratio=True)` scales
    ///   the image to *fit* `INPUT_SIDE` (`utils.py:205-223` → `imrescale`), so a
    ///   708x1116 crop becomes 650x1024, not a square. Squashing it stretches the
    ///   ruling lines off their axes, which is exactly what the class-1/class-2
    ///   split keys on. The ONNX declares dynamic `height`/`width` and the
    ///   generated model takes a plain rank-4 tensor, so non-square input is fine.
    /// - **Mean/std normalization, on 0-255 values.** The reference subtracts
    ///   `[123.675, 116.28, 103.53]` and divides by `[58.395, 57.12, 57.375]`
    ///   (`table_structure_unet.py:29-30, :70-74`) — ImageNet statistics *not*
    ///   rescaled to `[0, 1]`. Feeding `/255` values leaves the input in a range
    ///   the network never saw.
    fn preprocess(img: &RgbImage) -> (Vec<f32>, u32, u32) {
        // Fit inside INPUT_SIDE on the long edge, as `imrescale` does.
        let (w, h) = (img.width().max(1), img.height().max(1));
        let scale = f64::from(INPUT_SIDE) / f64::from(w.max(h));
        let rw = ((f64::from(w) * scale).round() as u32).max(1);
        let rh = ((f64::from(h) * scale).round() as u32).max(1);

        // `imrescale` picks area when shrinking and bicubic when growing; Triangle
        // is the closest the `image` crate offers to both without a hand-rolled
        // resampler, and the mask is a coarse per-pixel class map rather than a
        // value the next stage reads numerically.
        let resized = image::imageops::resize(img, rw, rh, image::imageops::FilterType::Triangle);

        const MEAN: [f32; 3] = [123.675, 116.28, 103.53];
        const STD: [f32; 3] = [58.395, 57.12, 57.375];

        let (rw_u, rh_u) = (rw as usize, rh as usize);
        let plane = rw_u * rh_u;
        let mut chw = vec![0.0f32; 3 * plane];
        for y in 0..rh_u {
            for x in 0..rw_u {
                let px = resized.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    let v = (f32::from(px[c]) - MEAN[c]) / STD[c];
                    chw[c * plane + y * rw_u + x] = v;
                }
            }
        }
        (chw, rw, rh)
    }

    /// Runs the generated UNet forward pass and returns the per-pixel 3-class
    /// argmax segmentation mask.
    ///
    /// This exercises the real neural network end-to-end (weight load is cached
    /// for the process lifetime). It is the wired half of the wired-table path.
    pub fn segment_mask(&self, img: &RgbImage) -> Result<SegMask> {
        let (input, w, h) = Self::preprocess(img);
        Self::run_mask(input, w, h)
    }

    /// Preprocesses `img` into the planar `[3, H, W]` buffer the UNet consumes,
    /// with its dimensions.
    ///
    /// A `#[doc(hidden)]` parity hook: the `unet_real` numeric gate dumps this
    /// exact buffer so the Python ONNX dumper can feed the *identical* input to
    /// `onnxruntime` (the `image` crate's Triangle resize is not trivially
    /// reproducible in numpy, so both sides must consume the same bytes). The
    /// dimensions come back because the buffer is no longer square. Not part of
    /// the public API.
    #[doc(hidden)]
    pub fn debug_preprocess(img: &RgbImage) -> (Vec<f32>, u32, u32) {
        Self::preprocess(img)
    }

    /// Runs the generated UNet forward over an already-preprocessed CHW buffer.
    ///
    /// A `#[doc(hidden)]` parity hook mirroring [`Self::debug_preprocess`]: the
    /// `unet_real` gate runs this on the same buffer it dumped and diffs the
    /// resulting class mask against the committed ONNX reference. Not part of the
    /// public API.
    #[doc(hidden)]
    pub fn debug_segment_from_input(&self, input: Vec<f32>, width: u32, height: u32) -> Result<SegMask> {
        Self::run_mask(input, width, height)
    }

    /// Loads the generated UNet model with runtime-fetched weights, panic-free.
    ///
    /// Unlike the vendored `Model::from_bytes` (which `.expect()`s), this drives
    /// the [`burn_store::BurnpackStore`] directly and maps every failure to a
    /// typed [`Error`].
    fn load_unet(device: &B::Device) -> Result<generated::Model<B>> {
        use burn_store::{BurnpackStore, ModuleSnapshot};

        let path = crate::weights::weight_path(crate::weights::TableWeight::Unet)?;
        let mut model = generated::Model::<B>::new(device);
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
    fn run_mask(input: Vec<f32>, width: u32, height: u32) -> Result<SegMask> {
        use burn::tensor::{Tensor, TensorData};

        // Cache the (weight-load-failable) model for the process lifetime, per
        // backend type (see [`crate::model_cache`]). A failed load is not cached,
        // so a later call can retry.
        let device = B::Device::default();
        let model: std::sync::Arc<generated::Model<B>> =
            crate::model_cache::get_or_try_init(|| Self::load_unet(&device))?;

        let data = TensorData::new(input, [1, 3, height as usize, width as usize]);
        let x = Tensor::<B, 4>::from_data(data, &device);

        // The generated top-level `forward` already argmaxes to an int class mask
        // of shape `[1, N, H, W]`.
        let mask = model.forward(x);
        let dims = mask.dims();
        let (h, w) = (dims[2], dims[3]);
        // `int_to_vec_i64` coerces the backend's storage dtype; a direct
        // `into_vec::<i64>()` is a dtype mismatch on every backend here, since flex
        // and wgpu both store ints as `i32`.
        let classes: Vec<i64> = mineru_burn_common::host::int_to_vec_i64(mask);
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

    type B = mineru_burn_common::backend::Cpu;

    /// The long edge lands on `INPUT_SIDE` and the aspect ratio survives —
    /// `imrescale`'s contract. Squashing to a square stretches ruling lines off
    /// their axes, which is what the horizontal/vertical class split reads.
    #[test]
    fn preprocess_fits_the_long_edge_and_keeps_the_aspect_ratio() {
        let img = RgbImage::from_pixel(1116, 708, image::Rgb([0, 0, 0]));
        let (buf, w, h) = UnetModel::<B>::preprocess(&img);

        assert_eq!(w, INPUT_SIDE, "long edge scales to INPUT_SIDE");
        assert_eq!(h, 650, "short edge keeps the ratio: round(708 * 1024/1116)");
        assert_eq!(buf.len(), 3 * w as usize * h as usize, "buffer is [3,H,W]");
    }

    #[test]
    fn preprocess_fits_a_tall_image_on_its_height() {
        let img = RgbImage::from_pixel(708, 1116, image::Rgb([0, 0, 0]));
        let (_, w, h) = UnetModel::<B>::preprocess(&img);

        assert_eq!(h, INPUT_SIDE, "the long edge is the height here");
        assert_eq!(w, 650);
    }

    /// Pins the exact normalization: ImageNet mean/std over **0-255** values, not
    /// over `[0, 1]`. A plain `/255` leaves the input in a range the network never
    /// saw, and the mask degrades without any error surfacing.
    #[test]
    fn preprocess_applies_mean_std_normalization_over_0_255() {
        // Already 1024 on the long edge, so resampling cannot perturb the values.
        let img = RgbImage::from_pixel(INPUT_SIDE, INPUT_SIDE, image::Rgb([0, 0, 0]));
        let (buf, w, h) = UnetModel::<B>::preprocess(&img);
        let plane = (w * h) as usize;

        // Black maps to (0 - mean) / std, per channel.
        let want = [
            (0.0 - 123.675) / 58.395,
            (0.0 - 116.28) / 57.12,
            (0.0 - 103.53) / 57.375,
        ];
        for (c, want) in want.iter().enumerate() {
            let got = buf.get(c * plane).copied().unwrap_or_default();
            assert!(
                (got - want).abs() < 1e-4,
                "channel {c}: got {got}, want {want} — normalization must use the \
                 reference's 0-255 statistics"
            );
        }
        // A `/255` pipeline would put every value in [0,1]; these must not be.
        assert!(
            buf.iter().any(|v| *v < -1.0),
            "normalized black must fall well below 0, not sit in [0,1]"
        );
    }

    #[test]
    fn preprocess_normalizes_white_above_zero() {
        let img = RgbImage::from_pixel(INPUT_SIDE, INPUT_SIDE, image::Rgb([255, 255, 255]));
        let (buf, _, _) = UnetModel::<B>::preprocess(&img);

        let got = buf.first().copied().unwrap_or_default();
        let want = (255.0 - 123.675) / 58.395;
        assert!((got - want).abs() < 1e-4, "white: got {got}, want {want}");
    }

    #[test]
    fn unweighted_reports_unavailable() {
        let m = UnetModel::<mineru_burn_common::backend::Cpu>::new();
        assert!(matches!(
            m.segment_cells(&RgbImage::new(32, 32)),
            Err(Error::ModelUnavailable("unet"))
        ));
    }
}
