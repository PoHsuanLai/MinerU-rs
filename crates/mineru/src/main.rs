//! The MinerU command-line entry point.
//!
//! Wires the CLI ([`cli`]) to a selectable [`Backend`](mineru_types::Backend)
//! ([`backend`]) and the one-shot parse flow ([`run`]). This is the only crate in
//! the workspace permitted to use `anyhow` (at the top level) — the libraries keep
//! their own typed errors.

mod backend;
mod cli;
mod run;

use anyhow::Context;
use clap::Parser;
use mineru_config::Config;

use crate::backend::{build_backend, VlmOverrides};
use crate::cli::Cli;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    let config = load_config(cli.config.as_deref())?;

    let args = cli.parse;
    let input = args
        .input()
        .context("no input given; pass a PDF path (e.g. `mineru paper.pdf`)")?
        .to_path_buf();
    let opts = args.parse_options()?;
    let mode = args.make_mode();
    let vlm = VlmOverrides {
        url: args.vlm_url.clone(),
        model: args.vlm_model.clone(),
    };
    // Default is auto (try GPU, fall back to CPU); `--cpu` forces CPU.
    let backend = build_backend(args.backend, !args.cpu, &config, &vlm, args.effort)?;
    // The written paths are reported by `run_parse` via tracing; no separate
    // stdout print, so all output goes through the subscriber.
    run::run_parse(
        backend.as_ref(),
        &input,
        &args.output,
        &opts,
        mode,
        args.debug_output,
    )
    .await?;
    Ok(())
}

/// Loads configuration from an explicit path or the default resolution chain.
///
/// With `--config`, reads that file (a missing file falls back to defaults). Env
/// overrides are applied in both cases.
fn load_config(explicit: Option<&std::path::Path>) -> anyhow::Result<Config> {
    let config = match explicit {
        Some(path) => {
            let mut config = Config::from_file_or_default(path)
                .with_context(|| format!("loading config {}", path.display()))?;
            config
                .apply_env_overrides(|key| std::env::var(key).ok())
                .context("applying config env overrides")?;
            config
        }
        None => Config::load().context("loading config")?,
    };
    Ok(config)
}

/// Dependencies whose `info` output is noise for a CLI user, pinned to `warn` in
/// the default filter.
///
/// `cubecl_wgpu` is the motivating case: on every GPU run it logs the adapter
/// twice, once as a multi-line dump of every wgpu feature flag the device
/// supports. That is debugging detail for a GPU-backend author, not for someone
/// parsing a PDF — and `mineru::engine` already reports the chosen device in one
/// line. `wgpu_core`/`wgpu_hal`/`naga` sit under it and are equally chatty.
///
/// These are quieted, not silenced: their `warn` and `error` still print, and any
/// `RUST_LOG` overrides the whole default (see [`init_tracing`]).
const NOISY_DEPENDENCIES: &[&str] = &["cubecl_wgpu", "wgpu_core", "wgpu_hal", "naga"];

/// Initializes tracing to stderr; `--verbose` raises the level to debug.
///
/// `RUST_LOG` takes precedence when set — it replaces this default entirely, so
/// `RUST_LOG=cubecl_wgpu=info` brings the GPU chatter back on a plain run.
/// Otherwise the default is `info` for everything except [`NOISY_DEPENDENCIES`],
/// which are held at `warn`.
///
/// `--verbose` turns the dependency quieting **off** as well as raising the level:
/// asking for debug output and then hiding a dependency's output would defeat the
/// point of the flag.
fn init_tracing(verbose: bool) {
    use tracing_subscriber::EnvFilter;

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| default_filter(verbose));
    // A second init (e.g. in tests) is harmless; ignore the error.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

/// Builds the default filter used when `RUST_LOG` is unset.
///
/// Split out of [`init_tracing`] so it is testable without touching the
/// process-global environment or installing a subscriber.
fn default_filter(verbose: bool) -> tracing_subscriber::EnvFilter {
    use tracing_subscriber::EnvFilter;

    if verbose {
        return EnvFilter::new("debug");
    }
    // Later directives win, so the per-crate `warn`s override the global level.
    let mut filter = EnvFilter::new("info");
    for dep in NOISY_DEPENDENCIES {
        // Each directive is a compile-time-known literal; a parse failure is
        // impossible, but skip rather than panic in the unreachable case.
        if let Ok(directive) = format!("{dep}=warn").parse() {
            filter = filter.add_directive(directive);
        }
    }
    filter
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default filter must quiet the noisy dependencies to `warn` while
    /// leaving our own crates at `info` — the whole point of the directive list.
    #[test]
    fn default_filter_quiets_noisy_deps_but_not_us() {
        let rendered = default_filter(false).to_string();
        for dep in NOISY_DEPENDENCIES {
            assert!(
                rendered.contains(&format!("{dep}=warn")),
                "{dep} should be pinned to warn; filter was {rendered}"
            );
        }
        assert!(
            rendered.contains("info"),
            "the global level should stay info; filter was {rendered}"
        );
    }

    /// `--verbose` must not keep quieting dependencies: asking for debug output
    /// and then filtering a dependency out would defeat the flag.
    #[test]
    fn verbose_unquiets_everything() {
        let rendered = default_filter(true).to_string();
        assert_eq!(rendered, "debug");
        for dep in NOISY_DEPENDENCIES {
            assert!(
                !rendered.contains(dep),
                "{dep} must not be pinned under --verbose; filter was {rendered}"
            );
        }
    }
}
