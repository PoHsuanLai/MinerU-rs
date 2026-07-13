//! Command-line interface definition (clap derive).
//!
//! Defines the argument surface and translates it into the domain types the core
//! flow consumes: a [`Backend`](mineru_types::Backend) selection, a
//! [`ParseOptions`](mineru_types::ParseOptions), and an output
//! [`MakeMode`](mineru_render::MakeMode). Keeping the parsing here means
//! [`crate::run`] and [`crate::serve`] work in domain terms, not raw flags.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use mineru_render::MakeMode;
use mineru_types::{Lang, ParseOptions};

/// Parse a PDF with MinerU and write Markdown + a content list.
#[derive(Debug, Parser)]
#[command(name = "mineru", version, about, long_about = None)]
pub struct Cli {
    /// Path to a JSON config file. Falls back to `MINERU_TOOLS_CONFIG_JSON` /
    /// `~/.mineru.json` and then built-in defaults when omitted.
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    /// Enable verbose (debug-level) logging.
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Optional subcommand; with none, the tool runs a one-shot parse (see the
    /// top-level flags).
    #[command(subcommand)]
    pub command: Option<Command>,

    /// The one-shot parse arguments, used when no subcommand is given.
    #[command(flatten)]
    pub parse: ParseArgs,
}

/// Subcommands beyond the default one-shot parse.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run a minimal HTTP server exposing the parser over one POST route.
    Serve(ServeArgs),
}

/// Arguments for the default one-shot parse.
#[derive(Debug, clap::Args)]
pub struct ParseArgs {
    /// Input PDF path.
    #[arg(short = 'p', long = "path")]
    pub input: Option<PathBuf>,

    /// Output directory for `<stem>.md` and `<stem>_content_list.json`.
    #[arg(short = 'o', long = "output", default_value = "output")]
    pub output: PathBuf,

    /// Which parsing backend to use.
    #[arg(short = 'b', long = "backend", value_enum, default_value_t = BackendKind::Pipeline)]
    pub backend: BackendKind,

    /// Language hint for OCR (e.g. `ch`, `en`). Omit to auto-detect.
    #[arg(long)]
    pub lang: Option<String>,

    /// Disable formula recognition.
    #[arg(long)]
    pub no_formula: bool,

    /// Disable table recognition.
    #[arg(long)]
    pub no_table: bool,

    /// Page range `START` or `START:END` (0-based, END exclusive). Omit for all
    /// pages.
    #[arg(long)]
    pub pages: Option<String>,

    /// Markdown output flavor.
    #[arg(long, value_enum, default_value_t = MarkdownMode::Mm)]
    pub mode: MarkdownMode,

    /// Override the VLM server base URL (vlm backend only). Falls back to the
    /// config's `vlm_server_url`, then the client default.
    #[arg(long)]
    pub vlm_url: Option<String>,

    /// Override the VLM served model name (vlm backend only).
    #[arg(long)]
    pub vlm_model: Option<String>,
}

/// Arguments for the `serve` subcommand.
#[derive(Debug, clap::Args)]
pub struct ServeArgs {
    /// Address to bind, e.g. `127.0.0.1:8000`.
    #[arg(long, default_value = "127.0.0.1:8000")]
    pub bind: String,

    /// Which parsing backend the server uses for every request.
    #[arg(short = 'b', long = "backend", value_enum, default_value_t = BackendKind::Pipeline)]
    pub backend: BackendKind,

    /// Override the VLM server base URL (vlm backend only).
    #[arg(long)]
    pub vlm_url: Option<String>,

    /// Override the VLM served model name (vlm backend only).
    #[arg(long)]
    pub vlm_model: Option<String>,
}

/// The selectable parsing backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BackendKind {
    /// Fully local Burn-model pipeline (needs model weights on disk).
    Pipeline,
    /// External OpenAI-compatible VLM server (needs a running server).
    Vlm,
}

/// The Markdown output flavor selectable on the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum MarkdownMode {
    /// Multimodal Markdown: images and charts are embedded.
    Mm,
    /// Natural-language Markdown: images and charts are dropped.
    Nlp,
}

impl From<MarkdownMode> for MakeMode {
    fn from(mode: MarkdownMode) -> Self {
        match mode {
            MarkdownMode::Mm => MakeMode::MmMarkdown,
            MarkdownMode::Nlp => MakeMode::NlpMarkdown,
        }
    }
}

impl ParseArgs {
    /// Builds [`ParseOptions`] from the flags, parsing the page range.
    ///
    /// # Errors
    /// Returns an error if `--pages` is not a valid `START` or `START:END`.
    pub fn parse_options(&self) -> anyhow::Result<ParseOptions> {
        Ok(ParseOptions {
            lang: self.lang.clone().map(Lang),
            formula: !self.no_formula,
            table: !self.no_table,
            page_range: parse_page_range(self.pages.as_deref())?,
        })
    }
}

/// Parses a `START` or `START:END` page range into the `ParseOptions` shape.
///
/// `None` (no `--pages`) means all pages. `START` alone means from `START` to the
/// end. `START:END` is a 0-based, END-exclusive range.
///
/// # Errors
/// Returns an error on non-numeric fields or a malformed range.
fn parse_page_range(spec: Option<&str>) -> anyhow::Result<Option<(usize, Option<usize>)>> {
    let Some(spec) = spec else {
        return Ok(None);
    };
    let spec = spec.trim();
    match spec.split_once(':') {
        Some((start, end)) => {
            let start: usize = start.trim().parse()?;
            let end = end.trim();
            let end = if end.is_empty() {
                None
            } else {
                Some(end.parse()?)
            };
            Ok(Some((start, end)))
        }
        None => {
            let start: usize = spec.parse()?;
            Ok(Some((start, None)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_range_none_is_all() {
        assert_eq!(parse_page_range(None).unwrap(), None);
    }

    #[test]
    fn page_range_start_only_is_open_ended() {
        assert_eq!(parse_page_range(Some("2")).unwrap(), Some((2, None)));
    }

    #[test]
    fn page_range_start_end() {
        assert_eq!(parse_page_range(Some("1:5")).unwrap(), Some((1, Some(5))));
    }

    #[test]
    fn page_range_open_end() {
        assert_eq!(parse_page_range(Some("3:")).unwrap(), Some((3, None)));
    }

    #[test]
    fn page_range_rejects_garbage() {
        assert!(parse_page_range(Some("x")).is_err());
    }

    #[test]
    fn parse_options_reflects_flags() {
        let args = ParseArgs {
            input: None,
            output: PathBuf::from("out"),
            backend: BackendKind::Pipeline,
            lang: Some("ch".to_owned()),
            no_formula: true,
            no_table: false,
            pages: None,
            mode: MarkdownMode::Mm,
            vlm_url: None,
            vlm_model: None,
        };
        let opts = args.parse_options().unwrap();
        assert_eq!(opts.lang, Some(Lang("ch".to_owned())));
        assert!(!opts.formula);
        assert!(opts.table);
    }

    #[test]
    fn markdown_mode_maps() {
        assert_eq!(MakeMode::from(MarkdownMode::Mm), MakeMode::MmMarkdown);
        assert_eq!(MakeMode::from(MarkdownMode::Nlp), MakeMode::NlpMarkdown);
    }

    #[test]
    fn cli_parses_without_panicking() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
