//! Backend selection: builds the requested `Box<dyn Backend>`.
//!
//! The one-shot flow ([`crate::run`]) holds its engine as a `Box<dyn Backend>`, so
//! construction lives here in one place. The pipeline backend loads model weights
//! best-effort from the config's `models_dir`; the VLM backend wires an HTTP client
//! to an external server.

use anyhow::{bail, Context};
use mineru_backend_pipeline::{PipelineBackend, PipelineModels};
use mineru_backend_vlm::VlmBackend;
use mineru_config::Config;
use mineru_types::Backend;
use mineru_vlm_client::VlmClientConfig;

use crate::cli::BackendKind;

/// Overrides for the VLM client, sourced from CLI flags.
///
/// Any field left `None` falls back to the config's `vlm_server_url` (for the URL)
/// and then the client's built-in default.
#[derive(Debug, Default, Clone)]
pub struct VlmOverrides {
    /// Base URL of the OpenAI-compatible VLM server.
    pub url: Option<String>,
    /// Served model name.
    pub model: Option<String>,
}

/// Builds the selected backend as a `Box<dyn Backend>`.
///
/// The pipeline path requires a models directory: it errors clearly if
/// `config.models_dir` is unset (there is no baked-in machine default — set
/// `MINERU_MODELS_DIR` or `models_dir` in the config file). Once a directory is
/// given, missing individual weights degrade to skipped stages per
/// [`PipelineModels::load`]. The VLM path only wires a client and cannot fail
/// here — a bad URL surfaces on the first request.
pub fn build_backend(
    kind: BackendKind,
    config: &Config,
    vlm: &VlmOverrides,
) -> anyhow::Result<Box<dyn Backend>> {
    match kind {
        BackendKind::Pipeline => {
            if config.models_dir.as_os_str().is_empty() {
                bail!(
                    "no model directory configured for the pipeline backend: set \
                     MINERU_MODELS_DIR (or `models_dir` in the config file) to the \
                     directory containing the model weights"
                );
            }
            let models_dir = config.models_dir.canonicalize().with_context(|| {
                format!(
                    "model directory {} does not exist (set MINERU_MODELS_DIR to a valid path)",
                    config.models_dir.display()
                )
            })?;
            build_pipeline_backend(&models_dir)
        }
        BackendKind::Vlm => {
            let mut client = VlmClientConfig::default();
            if let Some(url) = vlm.url.clone().or_else(|| config.vlm_server_url.clone()) {
                client.base_url = url;
            }
            if let Some(model) = vlm.model.clone() {
                client.model = model;
            }
            Ok(Box::new(VlmBackend::new(client)))
        }
    }
}

/// Loads the pipeline models and boxes the backend, selecting the wgpu GPU when
/// the `gpu` feature is compiled in *and* `MINERU_GPU` is set to a truthy value
/// (`1`/`true`/`yes`), otherwise the CPU backend.
///
/// The neural stages (layout/OCR/formula) run on the selected backend; the table
/// stages always run on CPU (their generated ONNX / SLANet types are CPU-pinned),
/// so a GPU run is a hybrid. Selection is a runtime env var rather than a CLI flag
/// so the same binary serves both without a plumbing change to the arg parser.
fn build_pipeline_backend(models_dir: &std::path::Path) -> anyhow::Result<Box<dyn Backend>> {
    #[cfg(feature = "gpu")]
    {
        if gpu_requested() {
            use mineru_burn_common::backend::{gpu_device, Gpu};
            tracing::info!("pipeline backend: wgpu GPU (neural stages) + CPU tables");
            let models = PipelineModels::<Gpu>::load_on(models_dir, gpu_device());
            return Ok(Box::new(PipelineBackend::new(models)));
        }
    }
    tracing::info!("pipeline backend: CPU");
    let models = PipelineModels::load(models_dir);
    Ok(Box::new(PipelineBackend::new(models)))
}

/// Whether `MINERU_GPU` requests the GPU backend (truthy: `1`, `true`, `yes`).
#[cfg(feature = "gpu")]
fn gpu_requested() -> bool {
    std::env::var("MINERU_GPU")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}
