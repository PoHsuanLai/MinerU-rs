//! Image → tensor preprocessing shared by every vision model.
//!
//! Vision models all need the same pipeline — resize, scale to `[0, 1]`, optional
//! per-channel mean/std normalisation, pack into an `NCHW` float tensor — but with
//! different constants. [`Preprocess`] captures those constants as data so the
//! pipeline itself lives in one place ([`Preprocess::apply`]).

use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use image::{RgbImage, imageops::FilterType};

use crate::error::{Error, Result};

/// Target spatial size for the resized image, in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Size {
    /// Target width (number of columns).
    pub width: u32,
    /// Target height (number of rows).
    pub height: u32,
}

impl Size {
    /// Convenience constructor.
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// A square size, `side` × `side`.
    pub fn square(side: u32) -> Self {
        Self::new(side, side)
    }
}

/// How pixel values are mapped into the tensor.
#[derive(Debug, Clone, PartialEq)]
pub enum Normalize {
    /// Only divide by 255, giving values in `[0, 1]`. Used by models that fold
    /// their own normalisation into the first layer.
    Rescale,
    /// Divide by 255, then apply `(x - mean) / std` per channel (RGB order). This
    /// is the ImageNet-style normalisation most detection/classification backbones
    /// expect. `mean` and `std` are in `[0, 1]` units (i.e. already /255).
    MeanStd {
        /// Per-channel means, RGB order, in `[0, 1]` units.
        mean: [f32; 3],
        /// Per-channel standard deviations, RGB order, in `[0, 1]` units.
        std: [f32; 3],
    },
}

impl Normalize {
    /// The common ImageNet mean/std (in `[0, 1]` units).
    pub fn imagenet() -> Self {
        Normalize::MeanStd {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }
}

/// A reusable image-preprocessing configuration.
///
/// Build one per model (the constants differ) and call [`Preprocess::apply`] on
/// each input image to get a `[1, 3, H, W]` tensor ready for the model.
#[derive(Debug, Clone, PartialEq)]
pub struct Preprocess {
    /// Size the image is resized to before tensorisation.
    pub size: Size,
    /// Pixel-value mapping.
    pub normalize: Normalize,
    /// Resampling filter used for the resize.
    pub filter: FilterType,
}

impl Preprocess {
    /// Creates a preprocessing config with the given target size and normalisation,
    /// using triangle (bilinear) resampling — the standard choice for these models.
    pub fn new(size: Size, normalize: Normalize) -> Self {
        Self {
            size,
            normalize,
            filter: FilterType::Triangle,
        }
    }

    /// Overrides the resampling filter.
    pub fn with_filter(mut self, filter: FilterType) -> Self {
        self.filter = filter;
        self
    }

    /// Resizes `image`, normalises it, and returns a `[1, 3, H, W]` float tensor on
    /// `device`. Channel order is RGB, matching `image::RgbImage`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] if the target size has a zero dimension.
    pub fn apply<B: Backend>(&self, image: &RgbImage, device: &B::Device) -> Result<Tensor<B, 4>> {
        if self.size.width == 0 || self.size.height == 0 {
            return Err(Error::Config(format!(
                "preprocess target size has a zero dimension: {:?}",
                self.size
            )));
        }

        let resized =
            image::imageops::resize(image, self.size.width, self.size.height, self.filter);
        let (w, h) = (self.size.width as usize, self.size.height as usize);

        // Build the CHW plane directly so the data is already in tensor layout.
        let (mean, std) = match &self.normalize {
            Normalize::Rescale => ([0.0_f32; 3], [1.0_f32; 3]),
            Normalize::MeanStd { mean, std } => (*mean, *std),
        };

        let mut data = vec![0.0_f32; 3 * h * w];
        for (c, (&m, &s)) in mean.iter().zip(std.iter()).enumerate() {
            let plane = &mut data[c * h * w..(c + 1) * h * w];
            for (i, px) in resized.pixels().enumerate() {
                let v = px.0[c] as f32 / 255.0;
                plane[i] = (v - m) / s;
            }
        }

        let tensor = Tensor::<B, 1>::from_data(TensorData::new(data, [3 * h * w]), device);
        Ok(tensor.reshape([1, 3, h, w]))
    }
}
