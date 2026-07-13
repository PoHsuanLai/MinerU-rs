//! UNet line-segmentation model wrapper.
//!
//! The network is a conv/convtranspose/concat/sigmoid encoder-decoder that
//! `burn-onnx` codegen can import directly (behind the `onnx-import` feature);
//! this wrapper feeds the preprocessed 1024×1024 image through it and would turn
//! the 3-class ruling-line mask into cell quadrilaterals.
//!
//! The mask → polygon extraction (morphology, connected components, min-area
//! rect) is not yet ported, so [`UnetModel::segment_cells`] currently reports the
//! model unavailable. The downstream grid recovery + HTML assembly it feeds is
//! fully implemented and tested (see [`super::recover_and_render`]).

use image::RgbImage;

use crate::error::{Error, Result};

use super::recover::Poly;

/// UNet fixed input side (Python `inp_height = inp_width = 1024`).
pub const INPUT_SIDE: u32 = 1024;

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

    /// Runs segmentation and returns recovered cell quadrilaterals.
    ///
    /// Returns [`Error::ModelUnavailable`] until the generated network + mask
    /// post-processing are wired.
    pub fn segment_cells(&self, _img: &RgbImage) -> Result<Vec<Poly>> {
        if !self.ready {
            return Err(Error::ModelUnavailable("unet"));
        }
        Ok(Vec::new())
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
