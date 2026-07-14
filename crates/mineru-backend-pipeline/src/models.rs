//! Model loading and ownership.
//!
//! [`PipelineModels`] holds the loaded Burn models and is the single place that
//! knows where weight files live under a models directory. Loading is
//! *best-effort*: each model is loaded independently and a missing or unloadable
//! weight file leaves that stage `None`, so the pipeline still runs (skipping the
//! unavailable stages) rather than failing wholesale. The orchestration in
//! [`analyze`](crate::analyze) checks each `Option` before use.

use std::path::{Path, PathBuf};

use mineru_burn_common::backend::{cpu_device, Cpu};
use mineru_burn_common::weights::Coverage;
use mineru_formula::FormulaRecognizer;
use mineru_layout::LayoutModel;
use mineru_ocr_det::{DetConfig, TextDetector};
use mineru_ocr_rec::{CharDict, RecConfig, TextRecognizer};
use mineru_table::slanet::model::SlaNet;
use mineru_table::unet::model::UnetModel;

/// Filesystem layout of the models directory.
///
/// Each field is the path (relative to the models root) of one model's weights,
/// mirroring the on-disk layout of the released `opendatalab` checkpoints. Kept as
/// data so tests and callers can point at alternative locations without touching
/// the loader.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    /// PP-DocLayoutV2 layout detector safetensors.
    pub layout: PathBuf,
    /// PP-OCRv6 text-line detector safetensors.
    pub ocr_det: PathBuf,
    /// PP-OCRv6 text recognizer safetensors.
    pub ocr_rec: PathBuf,
    /// Character dictionary for the recognizer.
    pub ocr_rec_dict: PathBuf,
    /// UniMerNet formula-recognition checkpoint *directory*
    /// (`model.safetensors` + `tokenizer.json`).
    pub formula_dir: PathBuf,
    /// SLANet-plus wireless-table weights.
    pub table_wireless: PathBuf,
    /// UNet wired-table weights (currently loaded via [`UnetModel::new`]).
    pub table_wired: PathBuf,
}

impl ModelPaths {
    /// Derives the default paths under `models_dir`, matching the on-disk layout
    /// of the PDF-Extract-Kit-1.0 release.
    ///
    /// `models_dir` is the `models/` directory of the release (e.g.
    /// `/path/to/PDF-Extract-Kit-1.0/models`). All model
    /// weights are subpaths of it *except* the OCR character dictionary, which
    /// ships with the application rather than the model download — see
    /// [`ocr_rec_dict`](Self::ocr_rec_dict).
    pub fn under(models_dir: impl AsRef<Path>) -> Self {
        let root = models_dir.as_ref();
        Self {
            layout: root.join("Layout/PP-DocLayoutV2/model.safetensors"),
            // The PP-OCRv6 torch checkpoints live flat under `OCR/paddleocr_torch/`.
            ocr_det: root.join("OCR/paddleocr_torch/ch_PP-OCRv6_small_det_infer.safetensors"),
            ocr_rec: root.join("OCR/paddleocr_torch/ch_PP-OCRv6_small_rec_infer.safetensors"),
            // The v6 charset is NOT in the model release; it ships with the app
            // (mirrors the Python repo's
            // `model/utils/pytorchocr/utils/resources/dict/ppocrv6_dict.txt`).
            // Placed alongside the recognizer here as the default; override via
            // config when the app bundles it elsewhere.
            ocr_rec_dict: root.join("OCR/paddleocr_torch/ppocrv6_dict.txt"),
            formula_dir: root.join("MFR/unimernet_hf_small_2503"),
            // Table structure/segmentation ship as ONNX (loaded via burn-import
            // codegen in mineru-table), under `TabRec/`.
            table_wireless: root.join("TabRec/SlanetPlus/slanet-plus.onnx"),
            table_wired: root.join("TabRec/UnetStructure/unet.onnx"),
        }
    }
}

/// The loaded model stages, each optional so the pipeline degrades gracefully when
/// a weight file is absent.
///
/// Construct with [`PipelineModels::load`] (best-effort, from a models directory)
/// or assemble field-by-field for tests. All models run on the CPU
/// ([`Cpu`]) backend.
#[derive(Default)]
pub struct PipelineModels {
    /// Layout detector; drives every downstream stage.
    pub layout: Option<LayoutModel<Cpu>>,
    /// OCR text-line detector.
    pub ocr_det: Option<TextDetector<Cpu>>,
    /// OCR text recognizer.
    pub ocr_rec: Option<TextRecognizer<Cpu>>,
    /// Formula recognizer.
    pub formula: Option<FormulaRecognizer<Cpu>>,
    /// Wireless-table structure recognizer.
    pub table_wireless: Option<SlaNet>,
    /// Wired-table line-segmentation recognizer.
    pub table_wired: Option<UnetModel>,
}

impl PipelineModels {
    /// Loads every stage best-effort from the default paths under `models_dir`.
    ///
    /// A stage whose weight file is missing or fails to load is left `None` and a
    /// warning is traced; the returned `PipelineModels` is always valid. This never
    /// errors, so a partially-provisioned models directory still yields a usable
    /// (if reduced) pipeline.
    pub fn load(models_dir: impl AsRef<Path>) -> Self {
        Self::load_from(&ModelPaths::under(models_dir))
    }

    /// Loads every stage best-effort from explicit [`ModelPaths`].
    pub fn load_from(paths: &ModelPaths) -> Self {
        let device = cpu_device();

        let layout = load_stage("layout", || {
            LayoutModel::<Cpu>::from_safetensors(&paths.layout).map_err(Into::into)
        });

        let ocr_det = load_stage("ocr-det", || {
            let mut det = TextDetector::<Cpu>::new(DetConfig::default(), device);
            det.load_weights(&paths.ocr_det)?;
            Ok(det)
        });

        let ocr_rec = load_stage("ocr-rec", || {
            // The v6 charset ships with the app, not the weight release. Use an
            // external dict file if one is present at the configured path
            // (e.g. a different language's dict), else fall back to the PP-OCRv6
            // dict embedded in mineru-ocr-rec so recognition works out of the box.
            let dict = if paths.ocr_rec_dict.exists() {
                CharDict::from_file(&paths.ocr_rec_dict, true)?
            } else {
                CharDict::ppocrv6(true)?
            };
            let mut rec = TextRecognizer::<Cpu>::new(dict, RecConfig::default(), device);
            rec.load_weights(&paths.ocr_rec)?;
            Ok(rec)
        });

        let formula = load_stage("formula", || {
            FormulaRecognizer::<Cpu>::from_pretrained(&paths.formula_dir, Coverage::Lenient)
                .map_err(Into::into)
        });

        let table_wireless = load_stage("table-wireless", || {
            SlaNet::load(&paths.table_wireless).map_err(Into::into)
        });

        // UNet has no on-disk loader yet (see mineru-table); construct the
        // (currently model-unavailable) handle so the wiring is in place.
        let table_wired = Some(UnetModel::new());

        Self {
            layout,
            ocr_det,
            ocr_rec,
            formula,
            table_wireless,
            table_wired,
        }
    }
}

/// Runs one stage loader, converting any failure into `None` with a warning.
///
/// Centralizes the best-effort policy so each stage's loader stays a single
/// expression and no `?`/panic leaks out of construction.
fn load_stage<T>(name: &'static str, load: impl FnOnce() -> crate::Result<T>) -> Option<T> {
    match load() {
        Ok(model) => Some(model),
        Err(e) => {
            tracing::warn!(stage = name, error = %e, "model stage unavailable; skipping");
            None
        }
    }
}
