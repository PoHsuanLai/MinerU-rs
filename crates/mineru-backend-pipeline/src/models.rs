//! Model loading and ownership.
//!
//! [`PipelineModels`] holds the loaded Burn models and is the single place that
//! knows where weight files live under a models directory. Loading is
//! *best-effort*: each model is loaded independently and a missing or unloadable
//! weight file leaves that stage `None`, so the pipeline still runs (skipping the
//! unavailable stages) rather than failing wholesale. The orchestration in
//! [`analyze`](crate::analyze) checks each `Option` before use.

use std::path::{Path, PathBuf};

use burn::prelude::Backend;
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
    ///
    /// **The relative paths below are the counterpart to
    /// `mineru_config::REQUIRED_MODEL_FILES`**, which drives the auto-download of
    /// missing weights. That list lives in `mineru-config` (foundational, must not
    /// depend on this crate) and carries a reciprocal note.
    ///
    /// The two are *deliberately not identical*: this struct names every path the
    /// loader may read, while the download list names only what upstream actually
    /// hosts. Three paths here are intentionally absent there — `ocr_rec_dict`
    /// (embedded in `mineru-ocr-rec`), the SlaNet `.safetensors` sibling
    /// (generated locally from the `.onnx`), and the UNet weights (`UnetModel::new`
    /// takes none; `mineru-table` fetches its own `.bpk`). Adding a path here that
    /// upstream does host means adding it there too.
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
/// Construct with [`PipelineModels::load`] (best-effort CPU load from a models
/// directory), [`PipelineModels::load_on`] (best-effort on an explicit backend/
/// device — e.g. the wgpu GPU), or assemble field-by-field for tests.
///
/// The neural stages (layout, OCR det/rec, formula) are generic over the Burn
/// backend `B` (default [`Cpu`]) so the whole pipeline can run on the GPU. The
/// **table** stages ([`SlaNet`], [`UnetModel`]) are *themselves* backend-generic
/// (there is no hardcoded CPU inside `mineru-table`), but this pipeline wires them
/// on [`Cpu`] specifically: tables are a tiny fraction of wall-clock, so there is
/// no reason to GPU-port them, and pinning them here keeps a single
/// `PipelineModels<B>` from having to mix two backends. A GPU pipeline is therefore
/// a *hybrid*: layout/OCR/formula on the GPU, tables on CPU — a deliberate wiring
/// choice, not a limitation of the table types.
pub struct PipelineModels<B: Backend = Cpu> {
    /// Layout detector; drives every downstream stage.
    pub layout: Option<LayoutModel<B>>,
    /// OCR text-line detector.
    pub ocr_det: Option<TextDetector<B>>,
    /// OCR text recognizer.
    pub ocr_rec: Option<TextRecognizer<B>>,
    /// Formula recognizer.
    pub formula: Option<FormulaRecognizer<B>>,
    /// Wireless-table structure recognizer (wired on [`Cpu`]; see the struct docs).
    pub table_wireless: Option<SlaNet<Cpu>>,
    /// Wired-table line-segmentation recognizer (wired on [`Cpu`]; see the struct docs).
    pub table_wired: Option<UnetModel<Cpu>>,
}

// `#[derive(Default)]` would require `B: Default`; the fields are all `Option`/
// `None`, so a hand-written impl over any backend is both correct and less
// constrained.
impl<B: Backend> Default for PipelineModels<B> {
    fn default() -> Self {
        Self {
            layout: None,
            ocr_det: None,
            ocr_rec: None,
            formula: None,
            table_wireless: None,
            table_wired: None,
        }
    }
}

impl<B: Backend> PipelineModels<B> {
    /// Loads every stage best-effort from the default paths under `models_dir`,
    /// with the neural stages on backend `B`/`device` and the tables on CPU.
    ///
    /// A stage whose weight file is missing or fails to load is left `None` and a
    /// warning is traced; the returned `PipelineModels` is always valid.
    pub fn load_on(models_dir: impl AsRef<Path>, device: B::Device) -> Self {
        Self::load_from_on(&ModelPaths::under(models_dir), device)
    }

    /// Loads every stage best-effort from explicit [`ModelPaths`], with the neural
    /// stages on backend `B`/`device` and the tables on CPU.
    pub fn load_from_on(paths: &ModelPaths, device: B::Device) -> Self {
        let layout = load_stage("layout", || {
            LayoutModel::<B>::load(&paths.layout, device.clone()).map_err(Into::into)
        });

        let ocr_det = load_stage("ocr-det", || {
            let mut det = TextDetector::<B>::new(DetConfig::default(), device.clone());
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
            let mut rec = TextRecognizer::<B>::new(dict, RecConfig::default(), device.clone());
            rec.load_weights(&paths.ocr_rec)?;
            Ok(rec)
        });

        let formula = load_stage("formula", || {
            FormulaRecognizer::<B>::from_pretrained_on(
                &paths.formula_dir,
                Coverage::Lenient,
                device.clone(),
            )
            .map_err(Into::into)
        });

        // Table stages: the mineru-table types are backend-generic, but this
        // pipeline wires them on Cpu (see the struct docs). The `Option<SlaNet<Cpu>>`
        // field type pins the backend here.
        let table_wireless = load_stage("table-wireless", || {
            SlaNet::load(&paths.table_wireless).map_err(Into::into)
        });

        // `loaded()`, not `new()`: `new()` is an inert handle that reports
        // `ModelUnavailable` instead of running, so the wired engine never
        // executed and every ruled table fell back to the wireless one. The UNet's
        // weights are fetched and cached on first use, so this stays lazy —
        // nothing is loaded unless a table actually reaches the wired engine.
        let table_wired = Some(UnetModel::loaded());

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

impl PipelineModels<Cpu> {
    /// Loads every stage best-effort on the CPU backend from the default paths
    /// under `models_dir`. Convenience wrapper over [`PipelineModels::load_on`].
    pub fn load(models_dir: impl AsRef<Path>) -> Self {
        Self::load_on(models_dir, cpu_device())
    }

    /// Loads every stage best-effort on the CPU backend from explicit
    /// [`ModelPaths`]. Convenience wrapper over [`PipelineModels::load_from_on`].
    pub fn load_from(paths: &ModelPaths) -> Self {
        Self::load_from_on(paths, cpu_device())
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
