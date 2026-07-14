//! Backend selection: translates the CLI choice into a `Box<dyn Backend>`.
//!
//! The one-shot flow ([`crate::run`]) holds its engine as a `Box<dyn Backend>`.
//! All construction — models-dir resolution, auto-download, CPU/GPU selection, and
//! VLM client wiring — is delegated to the library facade ([`mineru::Mineru`]), so
//! the binary and downstream library consumers share exactly one implementation.
//! This module only maps CLI arguments onto builder calls.

use mineru_config::Config;
use mineru_types::Backend;

use mineru::HybridEffort;

use crate::cli::{BackendKind, EffortArg};

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

/// Builds the selected backend as a `Box<dyn Backend>` via the library facade.
///
/// Every backend is constructed through [`mineru::Mineru::builder`]: the pipeline
/// path resolves the models directory, auto-downloads any missing weights
/// (best-effort — a fully-provisioned dir does no network access), and loads on the
/// GPU or CPU per `try_gpu`; the VLM path wires an HTTP client; hybrid does both
/// (its local models always load on the CPU). A build error (e.g. the models
/// directory cannot be resolved) is surfaced with context.
pub fn build_backend(
    kind: BackendKind,
    try_gpu: bool,
    config: &Config,
    vlm: &VlmOverrides,
    effort: Option<EffortArg>,
) -> anyhow::Result<Box<dyn Backend>> {
    let mut builder = mineru::Mineru::builder()
        .config(config.clone())
        .gpu(try_gpu);

    builder = match kind {
        BackendKind::Pipeline => builder,
        BackendKind::Vlm => builder.vlm(vlm.url.clone(), vlm.model.clone()),
        BackendKind::Hybrid => builder.hybrid(
            vlm.url.clone(),
            vlm.model.clone(),
            effort.map(hybrid_effort),
        ),
    };

    let engine = builder
        .build()
        .map_err(|e| anyhow::anyhow!("building the {kind:?} backend failed: {e}"))?;
    Ok(engine.into_backend())
}

/// Maps the CLI effort onto the facade's.
fn hybrid_effort(effort: EffortArg) -> HybridEffort {
    match effort {
        EffortArg::Medium => HybridEffort::Medium,
        EffortArg::High => HybridEffort::High,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two parallel two-variant enums are exactly the shape that silently
    /// inverts: swapping the arms still compiles and every backend still builds,
    /// but every run would take the other code path.
    #[test]
    fn effort_maps_without_inverting() {
        assert_eq!(hybrid_effort(EffortArg::Medium), HybridEffort::Medium);
        assert_eq!(hybrid_effort(EffortArg::High), HybridEffort::High);
    }
}
