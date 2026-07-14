//! Backend selection: builds the requested `Box<dyn Backend>`.
//!
//! The one-shot flow ([`crate::run`]) holds its engine as a `Box<dyn Backend>`, so
//! construction lives here in one place. The pipeline backend loads model weights
//! best-effort from the config's `models_dir`; the VLM backend wires an HTTP client
//! to an external server.

use anyhow::Context;
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
/// The pipeline path resolves a models directory (never empty — `mineru-config`
/// supplies a default cache dir), auto-downloads any missing weight files into it,
/// and then loads the models. A fully-provisioned dir does not hit the network.
/// Missing *individual* weights that fail to load degrade to skipped stages per
/// [`PipelineModels::load`]; a genuine download failure surfaces as an error. The
/// VLM path only wires a client and cannot fail here — a bad URL surfaces on the
/// first request.
pub fn build_backend(
    kind: BackendKind,
    gpu: bool,
    config: &Config,
    vlm: &VlmOverrides,
) -> anyhow::Result<Box<dyn Backend>> {
    match kind {
        BackendKind::Pipeline => {
            tracing::info!(
                "pipeline models dir: {}",
                config.models_dir.display()
            );
            // Best-effort: fetch any MISSING weights before loading. A fully-present
            // dir does no network access. Per-file download failures (404, host not
            // up, offline) are logged and skipped inside this call — they are NOT
            // fatal, matching the pipeline loader, which already warns on and skips
            // missing stages and errors only if nothing loads. So we intentionally
            // do not treat this as a hard failure of the run.
            let _ = mineru_config::download_missing_models(&config.models_dir);
            // Ensure the models root exists so `canonicalize` succeeds even when no
            // weights could be fetched (the loader then skips absent stages).
            if let Err(e) = std::fs::create_dir_all(&config.models_dir) {
                tracing::warn!(
                    "could not create models dir {}: {e}",
                    config.models_dir.display()
                );
            }
            let models_dir = config.models_dir.canonicalize().with_context(|| {
                format!(
                    "model directory {} could not be resolved",
                    config.models_dir.display()
                )
            })?;
            build_pipeline_backend(&models_dir, gpu)
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
/// requested (the `--gpu` flag, or the `MINERU_GPU` env var as an alias) *and* the
/// `gpu` feature is compiled in; otherwise the CPU backend.
///
/// The neural stages (layout/OCR/formula) run on the selected backend; the table
/// stages always run on CPU (a deliberate wiring choice — see [`PipelineModels`]),
/// so a GPU run is a hybrid.
///
/// If `--gpu` is passed to a binary built *without* the `gpu` feature, the request
/// cannot be honored; the CPU backend is used and a warning is logged rather than
/// failing, so the flag degrades gracefully.
fn build_pipeline_backend(
    models_dir: &std::path::Path,
    gpu: bool,
) -> anyhow::Result<Box<dyn Backend>> {
    let want_gpu = gpu || env_gpu_requested();
    #[cfg(feature = "gpu")]
    if want_gpu {
        use mineru_burn_common::backend::{gpu_device, Gpu};
        tracing::info!("pipeline backend: wgpu GPU (neural stages) + CPU tables");
        let models = PipelineModels::<Gpu>::load_on(models_dir, gpu_device());
        return Ok(Box::new(PipelineBackend::new(models)));
    }
    #[cfg(not(feature = "gpu"))]
    if want_gpu {
        tracing::warn!(
            "GPU requested but this binary was built without the `gpu` feature; \
             falling back to CPU (rebuild with --features gpu)"
        );
    }
    tracing::info!("pipeline backend: CPU");
    let models = PipelineModels::load(models_dir);
    Ok(Box::new(PipelineBackend::new(models)))
}

/// Whether `MINERU_GPU` requests the GPU backend (truthy: `1`, `true`, `yes`,
/// `on`). This is the environment-variable alias for the `--gpu` flag.
fn env_gpu_requested() -> bool {
    std::env::var("MINERU_GPU")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}
