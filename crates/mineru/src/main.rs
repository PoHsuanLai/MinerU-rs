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
    let backend = build_backend(args.backend, !args.cpu, &config, &vlm)?;
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

/// Initializes tracing to stderr; `--verbose` raises the level to debug.
///
/// Respects `RUST_LOG` when set; otherwise defaults to `info` (or `debug` under
/// `--verbose`).
fn init_tracing(verbose: bool) {
    use tracing_subscriber::EnvFilter;

    let default = if verbose { "debug" } else { "info" };
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(default));
    // A second init (e.g. in tests) is harmless; ignore the error.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}
