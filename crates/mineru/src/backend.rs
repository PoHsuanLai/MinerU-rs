//! Backend selection: builds the requested `Box<dyn Backend>`.
//!
//! The one-shot flow ([`crate::run`]) holds its engine as a `Box<dyn Backend>`, so
//! the binary's backend *selection* lives here. The pipeline's actual CPU/GPU
//! construction is delegated to the library facade
//! ([`mineru::build_pipeline_backend`]) so the binary and downstream library
//! consumers share one implementation. The VLM backend wires an HTTP client to an
//! external server.

use anyhow::Context;
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
/// and then loads the models (via [`mineru::build_pipeline_backend`]). A
/// fully-provisioned dir does not hit the network. Missing *individual* weights
/// that fail to load degrade to skipped stages; a genuine download failure
/// surfaces as an error. The VLM path only wires a client and cannot fail here — a
/// bad URL surfaces on the first request.
pub fn build_backend(
    kind: BackendKind,
    try_gpu: bool,
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
            // Shared with the library facade: CPU/GPU selection lives in one place
            // (`mineru::build_pipeline_backend`), so binary and library agree.
            Ok(mineru::build_pipeline_backend(&models_dir, try_gpu))
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

