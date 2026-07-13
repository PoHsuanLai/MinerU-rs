//! Input preprocessing: resize the page image to 800×800 and scale to `[0, 1]`.
//!
//! Port of `_preprocess_single_image`: bicubic resize to `[800, 800]`, then a
//! plain `× 1/255` rescale — **no** mean/std normalisation. Delegates the actual
//! pixel work to [`mineru_burn_common::preprocess::Preprocess`] so the resize path
//! is shared with the other vision models.
//!
//! # Fidelity note
//! The reference uses `torchvision`'s `BICUBIC` with `antialias=False`. This crate
//! uses the `image` crate's `CatmullRom` filter (a bicubic kernel). The two are
//! close but not bit-identical at the boundary — flagged as a parity risk for the
//! detection coordinates.

use image::RgbImage;
use image::imageops::FilterType;
use mineru_burn_common::backend::cpu_device;
use mineru_burn_common::preprocess::{Normalize, Preprocess, Size};
use burn::prelude::Backend;
use burn::tensor::Tensor;

use crate::config::INPUT_SIZE;
use crate::error::Result;

/// Returns the shared [`Preprocess`] config for PP-DocLayoutV2: 800×800 bicubic
/// resize (`CatmullRom`) with a bare `/255` rescale.
pub fn preprocess_config() -> Preprocess {
    Preprocess::new(Size::square(INPUT_SIZE), Normalize::Rescale).with_filter(FilterType::CatmullRom)
}

/// Preprocesses `image` into a `[1, 3, 800, 800]` tensor on the CPU device.
///
/// Also returns the original `(width, height)` so the postprocessor can scale the
/// normalised boxes back to pixel coordinates.
pub fn preprocess_cpu(
    image: &RgbImage,
) -> Result<(Tensor<mineru_burn_common::backend::Cpu, 4>, (u32, u32))> {
    let device = cpu_device();
    let size = (image.width(), image.height());
    let tensor = preprocess_config().apply::<mineru_burn_common::backend::Cpu>(image, &device)?;
    Ok((tensor, size))
}

/// Backend-generic variant of [`preprocess_cpu`] for callers holding a device.
pub fn preprocess<B: Backend>(
    image: &RgbImage,
    device: &B::Device,
) -> Result<(Tensor<B, 4>, (u32, u32))> {
    let size = (image.width(), image.height());
    let tensor = preprocess_config().apply::<B>(image, device)?;
    Ok((tensor, size))
}
