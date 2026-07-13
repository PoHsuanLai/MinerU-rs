//! The MinerU command-line entry point.
//!
//! Wires the CLI ([`cli`]) to a selectable [`Backend`](mineru_types::Backend)
//! ([`backend`]) and either the one-shot parse flow ([`run`]) or the minimal HTTP
//! server ([`serve`]). This is the only crate in the workspace permitted to use
//! `anyhow` (at the top level) — the libraries keep their own typed errors.

mod backend;
mod cli;
mod run;
mod serve;

use anyhow::Context;
use clap::Parser;
use mineru_config::Config;

use crate::backend::{build_backend, VlmOverrides};
use crate::cli::{Cli, Command};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    let config = load_config(cli.config.as_deref())?;

    match cli.command {
        Some(Command::Serve(args)) => {
            let vlm = VlmOverrides {
                url: args.vlm_url,
                model: args.vlm_model,
            };
            let backend = build_backend(args.backend, &config, &vlm);
            serve::serve(&args.bind, backend).await
        }
        None => {
            let args = cli.parse;
            let input = args
                .input
                .as_deref()
                .context("no input given; pass -p/--path <PDF> (or use `mineru serve`)")?;
            let opts = args.parse_options()?;
            let mode = args.mode.into();
            let vlm = VlmOverrides {
                url: args.vlm_url.clone(),
                model: args.vlm_model.clone(),
            };
            let backend = build_backend(args.backend, &config, &vlm);
            let (md, json) =
                run::run_parse(backend.as_ref(), input, &args.output, &opts, mode).await?;
            println!("wrote {}", md.display());
            println!("wrote {}", json.display());
            Ok(())
        }
    }
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
