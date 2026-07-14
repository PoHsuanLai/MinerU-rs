//! The library facade: [`Mineru`] and its [`MineruBuilder`].
//!
//! This is the "front door" for embedding MinerU in another Rust program. Where a
//! caller would otherwise wire up a [`Config`](mineru_config::Config), pick a
//! backend crate, resolve a models directory, fetch missing weights, and select
//! CPU vs GPU by hand, they instead write:
//!
//! ```no_run
//! # #[cfg(feature = "pipeline")]
//! # async fn demo() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! use mineru::types::{Backend, DocInput, ParseOptions};
//!
//! let engine = mineru::Mineru::builder()
//!     .models_dir("/path/to/models") // optional; defaults to Config resolution
//!     .gpu(false)                     // optional; default auto (GPU if available)
//!     .auto_download(true)            // optional; default true
//!     .build()?;                      // -> Mineru (impls Backend)
//!
//! let opts = ParseOptions::default();
//! let bytes = std::fs::read("paper.pdf")?;
//! let doc = engine.analyze(DocInput::new(bytes), &opts).await?;
//! # let _ = doc;
//! # Ok(())
//! # }
//! ```
//!
//! [`Mineru`] itself implements [`Backend`](mineru_types::Backend), so it drops
//! straight into any code already written against that trait — the trait stays the
//! principled core; this is only a convenience constructor over it.
//!
//! The construction logic here (auto-download + CPU/GPU selection) is the single
//! shared implementation: the `mineru` binary's backend selection calls into
//! [`build_pipeline_backend`] too, so there is exactly one copy.

use std::path::PathBuf;

use async_trait::async_trait;
use mineru_config::Config;
use mineru_types::{Backend, BackendError, DocInput, Document, ParseOptions};

/// Errors returned by the library facade ([`MineruBuilder::build`]).
///
/// The facade never uses `anyhow` (that stays binary-only per the workspace
/// convention); a consumer gets a typed, matchable error instead.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The requested backend needs a feature that was not compiled in.
    ///
    /// For example, calling [`MineruBuilder::build`] on a build with neither the
    /// `pipeline` nor the `vlm` feature, or selecting the VLM path on a build
    /// without `vlm`.
    #[error(
        "no usable backend: this build lacks the required feature ({0}); \
         rebuild the `mineru` crate with that feature enabled"
    )]
    MissingFeature(&'static str),

    /// The models directory could not be created or resolved on disk.
    #[error("model directory {path} could not be resolved: {source}")]
    ModelsDir {
        /// The directory that failed to resolve.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },
}

/// A configured document-parsing engine — the library front door.
///
/// Build one with [`Mineru::builder`]. It wraps a boxed
/// [`Backend`](mineru_types::Backend) and implements that trait by delegation, so
/// `engine.analyze(..)` works directly and a `&Mineru` is usable anywhere a
/// `&dyn Backend` is.
pub struct Mineru {
    backend: Box<dyn Backend>,
}

impl Mineru {
    /// Starts building an engine. See [`MineruBuilder`] for the options.
    #[must_use]
    pub fn builder() -> MineruBuilder {
        MineruBuilder::default()
    }

    /// Consumes the engine, yielding the boxed backend it wraps.
    ///
    /// Useful when a caller wants to store the erased `Box<dyn Backend>` directly
    /// (for example alongside other backends selected at runtime).
    #[must_use]
    pub fn into_backend(self) -> Box<dyn Backend> {
        self.backend
    }
}

#[async_trait]
impl Backend for Mineru {
    async fn analyze(
        &self,
        input: DocInput,
        opts: &ParseOptions,
    ) -> std::result::Result<Document, BackendError> {
        self.backend.analyze(input, opts).await
    }
}

/// Which backend the builder should construct.
///
/// The pipeline (fully-local Burn models) is the default. VLM points the engine at
/// an external OpenAI-compatible server.
#[derive(Debug, Clone, Default)]
enum Kind {
    /// Fully-local Burn pipeline backend.
    #[default]
    Pipeline,
    /// External OpenAI-compatible VLM server.
    #[cfg_attr(not(feature = "vlm"), allow(dead_code))]
    Vlm {
        /// Override for the server base URL; falls back to the config, then the
        /// client default.
        url: Option<String>,
        /// Override for the served model name.
        model: Option<String>,
    },
}

/// Builder for [`Mineru`]. Obtain one from [`Mineru::builder`].
///
/// Every setter is optional; the defaults resolve a config from the environment
/// (see the crate-level docs' environment-variable table), use the GPU when one is
/// available (falling back to CPU), and auto-download any missing pipeline weights.
#[derive(Default)]
pub struct MineruBuilder {
    config: Option<Config>,
    models_dir: Option<PathBuf>,
    /// `None` = the default auto behavior (try GPU, fall back to CPU); `Some(true)`
    /// = same as auto but explicit; `Some(false)` = force CPU (no probe).
    gpu: Option<bool>,
    auto_download: Option<bool>,
    kind: Kind,
}

impl MineruBuilder {
    /// Supplies a fully-built [`Config`] as the base (escape hatch).
    ///
    /// Later setters like [`models_dir`](Self::models_dir) still override the
    /// corresponding field. When omitted, the config is resolved with
    /// [`Config::load`] at [`build`](Self::build) time (env + file resolution).
    #[must_use]
    pub fn config(mut self, config: Config) -> Self {
        self.config = Some(config);
        self
    }

    /// Sets the models directory, overriding any config/environment value.
    ///
    /// When unset, the directory comes from the resolved [`Config`]
    /// (`MINERU_MODELS_DIR`, the config file, or the per-user cache default).
    #[must_use]
    pub fn models_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.models_dir = Some(dir.into());
        self
    }

    /// Controls GPU use. Defaults to the auto behavior when never called.
    ///
    /// - unset (default): try the GPU, fall back to CPU if no usable wgpu adapter
    ///   is present (or the build lacks the `gpu` feature). This never fails — a
    ///   probe confirms the device works before committing to it.
    /// - `gpu(true)`: same auto behavior, stated explicitly.
    /// - `gpu(false)`: force the CPU (skip the probe) — the exact, reproducible
    ///   path.
    ///
    /// The table stages always run on CPU, so a GPU run is a hybrid.
    #[must_use]
    pub fn gpu(mut self, gpu: bool) -> Self {
        self.gpu = Some(gpu);
        self
    }

    /// Fetches any missing pipeline weights before loading. Default `true`.
    ///
    /// A fully-provisioned models directory never hits the network. Set `false`
    /// to load only what is already on disk (missing stages are skipped by the
    /// loader). Ignored by the VLM backend.
    #[must_use]
    pub fn auto_download(mut self, auto_download: bool) -> Self {
        self.auto_download = Some(auto_download);
        self
    }

    /// Selects the external VLM backend instead of the local pipeline.
    ///
    /// `url` overrides the server base URL (falling back to the config's
    /// `vlm_server_url`, then the client default); `model` overrides the served
    /// model name. Requires the `vlm` feature. When this is not called, the
    /// builder constructs the pipeline backend (the local default).
    #[must_use]
    pub fn vlm(mut self, url: Option<String>, model: Option<String>) -> Self {
        self.kind = Kind::Vlm { url, model };
        self
    }

    /// Builds the engine.
    ///
    /// For the pipeline: resolves the models directory, optionally auto-downloads
    /// missing weights, then loads the models on the selected device. For VLM:
    /// wires an HTTP client (no network access until the first request).
    ///
    /// # Errors
    /// Returns [`Error::MissingFeature`] if the selected backend's feature is not
    /// compiled in, or [`Error::ModelsDir`] if the pipeline models directory
    /// cannot be created or canonicalized.
    pub fn build(self) -> Result<Mineru, Error> {
        // Resolve the base config: caller-supplied, else env/file resolution. A
        // resolution failure is not fatal — fall back to defaults so `build` can
        // still proceed (e.g. with an explicit `models_dir`).
        let mut config = self.config.unwrap_or_else(|| {
            Config::load().unwrap_or_else(|e| {
                tracing::warn!("could not load config ({e}); using defaults");
                Config::default()
            })
        });
        if let Some(dir) = self.models_dir {
            config.models_dir = dir;
        }
        let auto_download = self.auto_download.unwrap_or(true);
        // Unset => auto (try GPU, fall back to CPU). `gpu(false)` forces CPU.
        let try_gpu = self.gpu.unwrap_or(true);

        let backend = match self.kind {
            Kind::Pipeline => build_pipeline(&config, try_gpu, auto_download)?,
            Kind::Vlm { url, model } => build_vlm(&config, url, model)?,
        };
        Ok(Mineru { backend })
    }
}

/// Builds the pipeline backend: resolve dir, (optionally) fetch weights, load.
///
/// Shared by [`MineruBuilder::build`] and the `mineru` binary. Requires the
/// `pipeline` feature.
#[cfg(feature = "pipeline")]
fn build_pipeline(
    config: &Config,
    try_gpu: bool,
    auto_download: bool,
) -> Result<Box<dyn Backend>, Error> {
    tracing::info!("pipeline models dir: {}", config.models_dir.display());
    if auto_download {
        // Best-effort: fetch any MISSING weights before loading. A fully-present
        // dir does no network access. Per-file download failures (404, host not
        // up, offline) are logged and skipped inside this call — they are NOT
        // fatal, matching the pipeline loader, which warns on and skips missing
        // stages and errors only if nothing loads.
        let _ = mineru_config::download_missing_models(&config.models_dir);
    }
    // Ensure the models root exists so `canonicalize` succeeds even when no
    // weights could be fetched (the loader then skips absent stages).
    if let Err(e) = std::fs::create_dir_all(&config.models_dir) {
        tracing::warn!("could not create models dir {}: {e}", config.models_dir.display());
    }
    let models_dir = config
        .models_dir
        .canonicalize()
        .map_err(|source| Error::ModelsDir { path: config.models_dir.clone(), source })?;
    Ok(build_pipeline_backend(&models_dir, try_gpu))
}

#[cfg(not(feature = "pipeline"))]
fn build_pipeline(
    _config: &Config,
    _gpu: bool,
    _auto_download: bool,
) -> Result<Box<dyn Backend>, Error> {
    Err(Error::MissingFeature("pipeline"))
}

/// Loads the pipeline models and boxes the backend, choosing the GPU or CPU device.
///
/// `try_gpu` requests the auto path: when `true` (the default), the GPU is used
/// *if* the build has the `gpu` feature **and** a usable wgpu adapter is present
/// (confirmed by [`gpu_available`](mineru_burn_common::backend::gpu_available), a
/// probe that runs a trivial op end-to-end). If the probe fails — no adapter,
/// headless host, no `gpu` feature — the CPU backend is used and a single line is
/// logged. When `try_gpu` is `false` (the binary's `--cpu` flag), the CPU is used
/// directly with no probe, giving the exact, reproducible path.
///
/// This makes the GPU the default without ever crashing on a GPU-less machine: the
/// probe prevents committing to a device that would panic mid-pipeline (the wgpu
/// dtype/round-trip failure class can `panic!` rather than error, so it cannot be
/// recovered from once the run starts — it must be avoided up front).
///
/// The neural stages (layout/OCR/formula) run on the chosen backend; the table
/// stages always run on CPU (a deliberate wiring choice — see [`PipelineModels`]),
/// so a GPU run is a hybrid.
///
/// [`PipelineModels`]: mineru_backend_pipeline::PipelineModels
#[cfg(feature = "pipeline")]
pub fn build_pipeline_backend(models_dir: &std::path::Path, try_gpu: bool) -> Box<dyn Backend> {
    use mineru_backend_pipeline::{PipelineBackend, PipelineModels};

    #[cfg(feature = "gpu")]
    if try_gpu {
        use mineru_burn_common::backend::{gpu_available, gpu_device, Gpu};
        if gpu_available() {
            tracing::info!("pipeline backend: wgpu GPU (neural stages) + CPU tables");
            let models = PipelineModels::<Gpu>::load_on(models_dir, gpu_device());
            return Box::new(PipelineBackend::new(models));
        }
        tracing::info!("GPU unavailable (no usable wgpu adapter); running on CPU");
    }
    #[cfg(not(feature = "gpu"))]
    if try_gpu {
        tracing::info!(
            "this build lacks the `gpu` feature; running on CPU \
             (rebuild with --features gpu for GPU acceleration)"
        );
    }
    tracing::info!("pipeline backend: CPU");
    let models = PipelineModels::load(models_dir);
    Box::new(PipelineBackend::new(models))
}

/// Builds the VLM backend from the config plus optional overrides. Requires the
/// `vlm` feature.
#[cfg(feature = "vlm")]
fn build_vlm(
    config: &Config,
    url: Option<String>,
    model: Option<String>,
) -> Result<Box<dyn Backend>, Error> {
    use mineru_backend_vlm::VlmBackend;
    use mineru_vlm_client::VlmClientConfig;

    let mut client = VlmClientConfig::default();
    if let Some(url) = url.or_else(|| config.vlm_server_url.clone()) {
        client.base_url = url;
    }
    if let Some(model) = model {
        client.model = model;
    }
    Ok(Box::new(VlmBackend::new(client)))
}

#[cfg(not(feature = "vlm"))]
fn build_vlm(
    _config: &Config,
    _url: Option<String>,
    _model: Option<String>,
) -> Result<Box<dyn Backend>, Error> {
    Err(Error::MissingFeature("vlm"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The builder must construct an engine usable as `&dyn Backend` without
    /// hitting the network: `auto_download(false)` against a temp dir loads only
    /// what is on disk (nothing), so the loader skips every stage but still yields
    /// a backend.
    #[cfg(feature = "pipeline")]
    #[test]
    fn builder_builds_pipeline_offline() {
        let tmp = std::env::temp_dir().join(format!("mineru-facade-test-{}", std::process::id()));
        let engine = Mineru::builder()
            .models_dir(&tmp)
            .auto_download(false)
            .build()
            .expect("build should succeed against an empty models dir");
        // Usable as `&dyn Backend`.
        let _erased: &dyn Backend = &engine;
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Selecting VLM on a build without the `vlm` feature is a clear error, not a
    /// panic.
    #[cfg(not(feature = "vlm"))]
    #[test]
    fn vlm_without_feature_errors() {
        // `Mineru` wraps a non-`Debug` boxed backend, so match rather than
        // `unwrap_err` (which would require `Ok: Debug`).
        match Mineru::builder().vlm(None, None).build() {
            Err(Error::MissingFeature("vlm")) => {}
            other => panic!("expected MissingFeature(\"vlm\"), got {:?}", other.err()),
        }
    }

    /// With no backend feature at all, `build` reports the missing feature rather
    /// than panicking.
    #[cfg(not(feature = "pipeline"))]
    #[test]
    fn pipeline_without_feature_errors() {
        match Mineru::builder().build() {
            Err(Error::MissingFeature("pipeline")) => {}
            other => panic!("expected MissingFeature(\"pipeline\"), got {:?}", other.err()),
        }
    }
}
