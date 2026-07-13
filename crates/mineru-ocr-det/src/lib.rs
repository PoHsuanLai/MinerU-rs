//! DBNet text-line detection for MinerU, ported to Burn.
//!
//! This crate is a faithful Burn translation of MinerU's vendored `pytorchocr`
//! text detector — PP-OCRv6 *small det*, which is a Differentiable Binarization
//! (DB) network with a **PP-LCNetV4 backbone**, a **RepLKFPN neck**, and a **DB
//! head**. The neural pipeline emits a single-channel probability map; the
//! [`postprocess`] module then turns that map into oriented text-line
//! quadrilaterals with pure geometry (contours + min-area rects + polygon
//! dilation) — no second network.
//!
//! # Module layout (mirrors `pytorchocr`)
//!
//! - [`backbone`] — `PPLCNetV4(det=True)` (`modeling/backbones/rec_lcnetv4.py`).
//! - [`neck`] — `RepLKFPN` (`modeling/necks/db_fpn.py`).
//! - [`head`] — `DBHead(mode="ppocrv6")` (`modeling/heads/det_db_head.py`).
//! - [`postprocess`] — `DBPostProcess` (`postprocess/db_postprocess.py`).
//! - [`model`] — [`model::TextDetector`], the `predict_det.py` orchestration.
//!
//! Weights load as HF-flat safetensors (or `.pth`) through
//! [`mineru_burn_common::weights`], with a strict *"every source key consumed"*
//! check guarding against silent layer-name mismatches.
//!
//! # Example
//!
//! ```no_run
//! use image::RgbImage;
//! use mineru_burn_common::backend::{Cpu, cpu_device};
//! use mineru_ocr_det::model::{DetConfig, TextDetector};
//!
//! let device = cpu_device();
//! let mut det = TextDetector::<Cpu>::new(DetConfig::default(), device);
//! det.load_weights("ch_PP-OCRv6_small_det_infer.safetensors")?;
//! let image = RgbImage::new(640, 480);
//! let boxes = det.detect(&image)?;
//! # Ok::<(), mineru_ocr_det::Error>(())
//! ```

pub mod backbone;
pub mod error;
pub mod head;
pub mod model;
pub mod neck;
pub mod postprocess;
pub mod weights;

pub use error::{Error, Result};
pub use model::{DetConfig, TextDetector};
pub use postprocess::{DbPostConfig, QuadBox};
