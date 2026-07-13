//! Document layout detection for MinerU: a Rust/Burn port of **PP-DocLayoutV2**.
//!
//! PP-DocLayoutV2 is an RT-DETR-L object detector (HGNetV2 backbone → hybrid
//! encoder → DETR-style deformable decoder, NMS-free — topk over sigmoid logits)
//! plus a small reading-order pointer network. This crate loads the released
//! `opendatalab` safetensors checkpoint and runs inference, producing flat scored
//! [`LayoutDet`]s in reading order. Assembly into the `mineru_types::Block` tree
//! happens later in the pipeline backend, not here.
//!
//! # Structure
//! - [`backbone`] — HGNetV2-L feature extractor.
//! - [`encoder`] — the RT-DETR hybrid encoder (AIFI transformer + CCFM FPN-PAN).
//! - [`decoder`] — query selection + deformable-attention decoder + box/class heads.
//! - [`reading_order`] — the LayoutLMv3-style reading-order pointer network.
//! - [`model`] — assembles the graph and runs the full forward pass.
//! - [`preprocess`] / [`postprocess`] — the 800×800 `/255` input pipeline and the
//!   cxcywh→xyxy + topk + reading-order-sort output pipeline.
//! - [`weights`] — the safetensors-key → Burn-field remap.
//! - [`nn`] — PyTorch-layout primitives used so weights load byte-for-byte.
//!
//! # Weights
//! Repo `opendatalab/PDF-Extract-Kit-1.0`, path
//! `models/Layout/PP-DocLayoutV2/model.safetensors`. Load with
//! [`LayoutModel::from_safetensors`].

pub mod backbone;
pub mod config;
pub mod decoder;
pub mod detection;
pub mod encoder;
pub mod error;
pub mod label;
pub mod model;
pub mod nn;
pub mod postprocess;
pub mod preprocess;
pub mod reading_order;
pub mod weights;

use std::path::Path;

use image::RgbImage;
use mineru_burn_common::backend::{Cpu, cpu_device};
use mineru_burn_common::model::Model;
use mineru_burn_common::weights::load_weights;
use burn::prelude::Backend;

pub use detection::LayoutDet;
pub use error::{Error, Result};
pub use label::{LayoutLabel, CLASS_ORDER, CLASS_THRESHOLDS, NUM_CLASSES};
pub use model::PpDocLayoutV2;
pub use postprocess::DEFAULT_CONF;

/// A ready-to-run layout detector: the loaded network plus its config.
///
/// Generic over the Burn backend `B`; [`LayoutModel::from_safetensors`] builds the
/// default CPU model.
pub struct LayoutModel<B: Backend> {
    model: PpDocLayoutV2<B>,
    device: B::Device,
    conf: f32,
}

impl<B: Backend> LayoutModel<B> {
    /// Wraps an already-initialised network for the given device.
    pub fn new(model: PpDocLayoutV2<B>, device: B::Device) -> Self {
        Self {
            model,
            device,
            conf: DEFAULT_CONF,
        }
    }

    /// Sets the final confidence threshold (default [`DEFAULT_CONF`]).
    pub fn with_conf(mut self, conf: f32) -> Self {
        self.conf = conf;
        self
    }

    /// Loads weights from a `.safetensors` file into an initialised model on
    /// `device`, applying the [`weights::key_remap`] and strict coverage.
    ///
    /// # Errors
    /// Propagates weight-load, key-remap, or coverage failures.
    pub fn load(path: impl AsRef<Path>, device: B::Device) -> Result<Self> {
        let mut model = PpDocLayoutV2::<B>::init(&device);
        let remap = weights::key_remap()?;
        load_weights::<B, _>(&mut model, path, &remap, weights::COVERAGE)?;
        Ok(Self::new(model, device))
    }

    /// Runs detection on one image, returning reading-order-sorted detections.
    ///
    /// # Errors
    /// Propagates preprocessing and postprocessing failures.
    pub fn detect(&self, image: &RgbImage) -> Result<Vec<LayoutDet>> {
        let (pixel_values, (img_w, img_h)) = preprocess::preprocess::<B>(image, &self.device)?;
        let outputs = self.model.forward(pixel_values);

        // cxcywh -> xyxy (normalised), topk over sigmoid class scores.
        let boxes_xyxy = postprocess::boxes_to_xyxy::<B>(&outputs.pred_boxes)?;
        let (logits, num_q, num_cls) = postprocess::logits_flat::<B>(&outputs.logits)?;
        let topk = postprocess::topk_over_classes(&logits, num_q, num_cls);

        // reading-order ranks from the pairwise order logits.
        let (order_flat, seq) = postprocess::order_logits_flat::<B>(&outputs.order_logits)?;
        let ranks = postprocess::order_seqs(&order_flat, seq);

        postprocess::assemble(
            &boxes_xyxy,
            &topk,
            &ranks,
            img_w as f32,
            img_h as f32,
            self.conf,
        )
    }
}

impl LayoutModel<Cpu> {
    /// Loads the model on the default CPU backend from a `.safetensors` file.
    pub fn from_safetensors(path: impl AsRef<Path>) -> Result<Self> {
        Self::load(path, cpu_device())
    }
}

impl<B: Backend> Model for LayoutModel<B> {
    type Input = RgbImage;
    type Output = Vec<LayoutDet>;

    fn predict(&self, input: Self::Input) -> mineru_burn_common::Result<Self::Output> {
        // The harness trait uses the common error; wrap this crate's error into it.
        self.detect(&input).map_err(|e| match e {
            Error::Common(c) => c,
            other => mineru_burn_common::Error::Config(other.to_string()),
        })
    }
}
