//! End-to-end DBNet text detector.
//!
//! Ties the backbone → neck → head → post-process pipeline together and exposes
//! [`TextDetector::detect`], mirroring `TextDetector.__call__` in
//! `pytorchocr/tools/infer/predict_det.py`:
//!
//! 1. **Resize** the image so its shorter/longer side hits `limit_side_len` and
//!    both sides are multiples of 32 (`DetResizeForTest.resize_image_type0`).
//! 2. **Normalize** with ImageNet mean/std (via [`mineru_burn_common::Preprocess`]).
//! 3. Run backbone + neck + head to get a `[1, 1, H', W']` probability map.
//! 4. **Post-process** ([`crate::postprocess`]) into quads, rescaled to the source
//!    image size.
//!
//! Output is `Vec<mineru_types::BBox>` axis-aligned boxes plus the raw quads via
//! [`TextDetector::detect_quads`] for callers that need the oriented geometry.

use std::path::Path;

use burn::module::Module;
use burn::prelude::Backend;
use burn::tensor::Tensor;
use image::RgbImage;
use mineru_burn_common::preprocess::{Normalize, Preprocess, Size};
use mineru_burn_common::weights::load_weights;
use mineru_types::BBox;

use crate::backbone::{PpLcNetV4Det, STAGE_OUT_CHANNELS};
use crate::error::{Error, Result};
use crate::head::DbHead;
use crate::neck::RepLkFpn;
use crate::postprocess::{boxes_from_bitmap, DbPostConfig, ProbMap, QuadBox};
use crate::weights::{key_remap, COVERAGE};

/// Configuration for the detector's resize/threshold behaviour.
#[derive(Debug, Clone, Copy)]
pub struct DetConfig {
    /// Target side length for the shorter/longer side (see `limit_type`).
    pub limit_side_len: u32,
    /// `true` = limit the *max* side (default for standard text OCR), `false` =
    /// limit the *min* side (seal OCR).
    pub limit_max_side: bool,
    /// Cap on either side after resizing.
    pub max_side_limit: u32,
    /// Post-process parameters.
    pub post: DbPostConfig,
}

impl Default for DetConfig {
    fn default() -> Self {
        // Matches PytorchPaddleOCR standard-text defaults: max side 960.
        Self {
            limit_side_len: 960,
            limit_max_side: true,
            max_side_limit: 4000,
            post: DbPostConfig::default(),
        }
    }
}

/// Backbone + neck, mirroring the checkpoint's `model.{backbone,neck}` subtree.
///
/// The DB head's checkpoint keys are top-level (`head.*`), so it is *not* nested
/// here; see [`Net`]. Keeping this split lets the whole checkpoint load into one
/// module tree under strict coverage with no prefix rewriting.
#[derive(Module, Debug)]
struct NetInner<B: Backend> {
    backbone: PpLcNetV4Det<B>,
    neck: RepLkFpn<B>,
}

/// The full detection network, laid out to match the checkpoint's module tree:
/// backbone + neck under `model.*`, and the DB head under `head.*`.
#[derive(Module, Debug)]
struct Net<B: Backend> {
    model: NetInner<B>,
    head: DbHead<B>,
}

impl<B: Backend> Net<B> {
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let feats = self.model.backbone.forward(input);
        let fused = self.model.neck.forward(feats);
        self.head.forward(fused)
    }
}

/// DBNet detector: PP-LCNetV4 backbone → RepLKFPN neck → DB head → post-process.
pub struct TextDetector<B: Backend> {
    net: Net<B>,
    config: DetConfig,
    device: B::Device,
    preprocess_template: Preprocess,
}

impl<B: Backend> TextDetector<B> {
    /// Builds the PP-OCRv6 *small det* network (unloaded) on `device`.
    ///
    /// Call [`TextDetector::load_weights`] before [`TextDetector::detect`].
    pub fn new(config: DetConfig, device: B::Device) -> Self {
        let backbone = PpLcNetV4Det::new(&device);
        // PP-OCRv6 RepLKFPN: out_channels 96, dilated kernel 7, reduction 4.
        let neck = RepLkFpn::new(&STAGE_OUT_CHANNELS, 96, true, 7, 4, &device);
        // DB head consumes the concatenated neck output (4 × 96/4 = 96 channels).
        let head = DbHead::new(neck.out_channels(), [3, 2, 2], true, &device);
        // A template Preprocess; each image overrides the Size before applying.
        let preprocess_template = Preprocess::new(Size::square(32), Normalize::imagenet());
        Self {
            net: Net {
                model: NetInner { backbone, neck },
                head,
            },
            config,
            device,
            preprocess_template,
        }
    }

    /// Loads PP-OCRv6 safetensors (or `.pth`) weights into the whole network.
    ///
    /// The whole checkpoint is applied to one module tree ([`Net`], whose layout
    /// mirrors the checkpoint's `model.{backbone,neck}` / `head` prefixes), with the
    /// [`key_remap`] bridging the two small structural differences (SE conv indices,
    /// reparameterised depthwise conv). Loads with [`Coverage::Strict`] so any
    /// unmatched source key is surfaced as an [`Error::UnmappedKeys`].
    ///
    /// [`Error::UnmappedKeys`]: mineru_burn_common::Error::UnmappedKeys
    pub fn load_weights(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let remap = key_remap()?;
        load_weights::<B, _>(&mut self.net, path.as_ref(), &remap, COVERAGE)?;
        Ok(())
    }

    /// Runs detection on `image`, returning axis-aligned bounding boxes.
    ///
    /// This is the convenience API; [`TextDetector::detect_quads`] preserves the
    /// oriented quadrilaterals (needed for rotated-crop recognition).
    pub fn detect(&self, image: &RgbImage) -> Result<Vec<BBox>> {
        Ok(self
            .detect_quads(image)?
            .into_iter()
            .map(|q| quad_to_bbox(&q))
            .collect())
    }

    /// Runs detection on `image`, returning oriented quad boxes in source pixels.
    pub fn detect_quads(&self, image: &RgbImage) -> Result<Vec<QuadBox>> {
        let (src_w, src_h) = (image.width(), image.height());
        if src_w == 0 || src_h == 0 {
            return Err(Error::Config("input image has a zero dimension".into()));
        }

        let target = self.resize_target(src_w, src_h);
        let pre = Preprocess {
            size: target,
            ..self.preprocess_template.clone()
        };
        let input = pre.apply::<B>(image, &self.device)?;

        // Forward pass -> probability map [1, 1, Hm, Wm].
        let prob = self.net.forward(input);

        let dims = prob.dims();
        if dims[0] != 1 || dims[1] != 1 {
            return Err(Error::ProbMapShape(format!("expected [1,1,H,W], got {dims:?}")));
        }
        let (mh, mw) = (dims[2], dims[3]);
        let data = prob
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| Error::ProbMapShape(format!("prob map not f32: {e:?}")))?;

        let map = ProbMap {
            data: &data,
            width: mw,
            height: mh,
        };
        Ok(boxes_from_bitmap(
            &map,
            &self.config.post,
            src_w as f32,
            src_h as f32,
        ))
    }

    /// Computes the network input size from the source size, matching
    /// `DetResizeForTest.resize_image_type0`: scale so the limited side hits
    /// `limit_side_len`, cap at `max_side_limit`, then round each side to a
    /// multiple of 32 (min 32).
    fn resize_target(&self, w: u32, h: u32) -> Size {
        let (w, h) = (w as f32, h as f32);
        let limit = self.config.limit_side_len as f32;

        let ratio = if self.config.limit_max_side {
            if w.max(h) > limit {
                limit / w.max(h)
            } else {
                1.0
            }
        } else if w.min(h) < limit {
            limit / w.min(h)
        } else {
            1.0
        };

        let mut rh = h * ratio;
        let mut rw = w * ratio;
        let cap = self.config.max_side_limit as f32;
        if rh.max(rw) > cap {
            let r2 = cap / rh.max(rw);
            rh *= r2;
            rw *= r2;
        }

        let round32 = |v: f32| -> u32 {
            let r = ((v / 32.0).round() * 32.0) as i64;
            r.max(32) as u32
        };
        Size::new(round32(rw), round32(rh))
    }
}

/// Converts a quad to its axis-aligned bounding box.
fn quad_to_bbox(q: &QuadBox) -> BBox {
    let xs = q.points.iter().map(|p| p.0);
    let ys = q.points.iter().map(|p| p.1);
    let x0 = xs.clone().fold(f32::MAX, f32::min);
    let x1 = xs.fold(f32::MIN, f32::max);
    let y0 = ys.clone().fold(f32::MAX, f32::min);
    let y1 = ys.fold(f32::MIN, f32::max);
    BBox::new(x0, y0, x1, y1)
}
