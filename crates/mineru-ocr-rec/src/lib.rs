//! SVTR/CRNN + CTC text recognition for MinerU, ported to Burn.
//!
//! This crate is a faithful Burn translation of MinerU's vendored `pytorchocr`
//! text recognizer — PP-OCRv6 *small rec*, an `SVTR_LCNet` model with a
//! **PP-LCNetV4 backbone**, a **LightSVTR neck** feeding a **CTC head**, decoded
//! greedily against a character dictionary.
//!
//! # Module layout (mirrors `pytorchocr`)
//!
//! - [`backbone`] — `PPLCNetV4(det=False)` (`modeling/backbones/rec_lcnetv4.py`).
//! - [`neck`] — `EncoderWithLightSVTR` (`modeling/necks/rnn.py`).
//! - [`head`] — the CTC branch of `MultiHead` (`modeling/heads/rec_multi_head.py`).
//! - [`dict`] — the character dictionary + CTC label mapping
//!   (`postprocess/rec_postprocess.py`, `CTCLabelDecode`).
//! - [`model`] — [`model::TextRecognizer`], the `predict_rec.py` orchestration.
//!
//! CTC greedy decoding is delegated to [`mineru_burn_common::ctc`]; weights load as
//! HF-flat safetensors (or `.pth`) through [`mineru_burn_common::weights`].
//!
//! # Example
//!
//! ```no_run
//! use image::RgbImage;
//! use mineru_burn_common::backend::{Cpu, cpu_device};
//! use mineru_ocr_rec::{CharDict, RecConfig, TextRecognizer};
//!
//! let device = cpu_device();
//! let dict = CharDict::from_file("ppocrv6_dict.txt", true)?;
//! let mut rec = TextRecognizer::<Cpu>::new(dict, RecConfig::default(), device);
//! rec.load_weights("ch_PP-OCRv6_small_rec_infer.safetensors")?;
//! let crop = RgbImage::new(120, 48);
//! let (text, score) = rec.recognize(&crop)?;
//! # Ok::<(), mineru_ocr_rec::Error>(())
//! ```

pub mod backbone;
pub mod dict;
pub mod error;
pub mod head;
pub mod model;
pub mod neck;

pub use dict::CharDict;
pub use error::{Error, Result};
#[doc(hidden)]
pub use model::{RecStageDumps, StageDump};
pub use model::{RecConfig, TextRecognizer};
