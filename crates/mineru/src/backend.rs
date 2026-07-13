//! Backend selection: builds the requested `Box<dyn Backend>`.
//!
//! Both the one-shot flow ([`crate::run`]) and the server ([`crate::serve`]) hold
//! their engine as a `Box<dyn Backend>`, so construction lives here in one place.
//! The pipeline backend loads model weights best-effort from the config's
//! `models_dir`; the VLM backend wires an HTTP client to an external server.

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
/// The pipeline path is infallible (missing weights degrade to skipped stages, per
/// [`PipelineModels::load`]). The VLM path only wires a client and cannot fail
/// here — a bad URL surfaces on the first request.
pub fn build_backend(
    kind: BackendKind,
    config: &Config,
    vlm: &VlmOverrides,
) -> Box<dyn Backend> {
    match kind {
        BackendKind::Pipeline => {
            let models = PipelineModels::load(&config.models_dir);
            Box::new(PipelineBackend::new(models))
        }
        BackendKind::Vlm => {
            let mut client = VlmClientConfig::default();
            if let Some(url) = vlm.url.clone().or_else(|| config.vlm_server_url.clone()) {
                client.base_url = url;
            }
            if let Some(model) = vlm.model.clone() {
                client.model = model;
            }
            Box::new(VlmBackend::new(client))
        }
    }
}
