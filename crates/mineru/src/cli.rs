//! Command-line interface definition (clap derive).
//!
//! Defines the argument surface and translates it into the domain types the core
//! flow consumes: a [`Backend`](mineru_types::Backend) selection, a
//! [`ParseOptions`](mineru_types::ParseOptions), and an output
//! [`MakeMode`](mineru_render::MakeMode). Keeping the parsing here means
//! [`crate::run`] works in domain terms, not raw flags.

use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use mineru_render::MakeMode;
use mineru_types::{Lang, ParseOptions};

/// Parse a PDF with MinerU and write Markdown + a content list.
#[derive(Debug, Parser)]
#[command(name = "mineru", version, about, long_about = None)]
pub struct Cli {
    /// Path to a JSON config file. Falls back to `MINERU_TOOLS_CONFIG_JSON` /
    /// `~/.mineru.json` and then built-in defaults when omitted.
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Enable verbose (debug-level) logging.
    #[arg(short, long)]
    pub verbose: bool,

    /// The parse arguments.
    #[command(flatten)]
    pub parse: ParseArgs,
}

/// Arguments for the one-shot parse.
#[derive(Debug, clap::Args)]
pub struct ParseArgs {
    /// Input PDF path.
    ///
    /// Given positionally (`mineru paper.pdf`). The legacy `-p/--path` flag is
    /// kept as a hidden alias so existing scripts keep working.
    #[arg(value_name = "PDF")]
    pub input: Option<PathBuf>,

    /// Legacy alias for the positional input path (hidden; use the positional
    /// argument instead).
    #[arg(short = 'p', long = "path", value_name = "PDF", hide = true, conflicts_with = "input")]
    pub input_flag: Option<PathBuf>,

    /// Output directory for `<stem>.md` and `<stem>_content_list.json`.
    #[arg(short = 'o', long = "output", default_value = "output")]
    pub output: PathBuf,

    /// Which parsing backend to use.
    #[arg(short = 'b', long = "backend", value_enum, default_value_t = BackendKind::Pipeline)]
    pub backend: BackendKind,

    /// Run the neural stages on the GPU (wgpu/Metal) instead of the CPU.
    ///
    /// Faster on formula/layout-heavy documents; the table stages still run on
    /// CPU (a hybrid). Requires the binary to be built with the `gpu` feature.
    /// Falls back to the CPU when unset. The `MINERU_GPU` environment variable is
    /// honored as an alias when this flag is not given.
    #[arg(long)]
    pub gpu: bool,

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

    /// Drop images and charts from the Markdown output.
    ///
    /// By default images/charts are embedded (multimodal Markdown). With this
    /// flag they are omitted, producing natural-language Markdown — matching the
    /// shape of `--no-formula` / `--no-table`.
    #[arg(long)]
    pub no_images: bool,

    /// Override the VLM server base URL (vlm backend only). Falls back to the
    /// config's `vlm_server_url`, then the client default.
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

impl ParseArgs {
    /// The resolved input path: the positional argument, or the legacy
    /// `-p/--path` alias when that was used instead. `None` when neither is given.
    pub fn input(&self) -> Option<&std::path::Path> {
        self.input
            .as_deref()
            .or(self.input_flag.as_deref())
    }

    /// The Markdown flavor: multimodal by default, natural-language (images
    /// dropped) when `--no-images` is set.
    pub fn make_mode(&self) -> MakeMode {
        if self.no_images {
            MakeMode::NlpMarkdown
        } else {
            MakeMode::MmMarkdown
        }
    }

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

    /// Builds a `ParseArgs` with defaults, overriding via the closure — so each
    /// test states only the fields it cares about and survives new fields.
    fn args_with(f: impl FnOnce(&mut ParseArgs)) -> ParseArgs {
        let mut args = ParseArgs {
            input: None,
            input_flag: None,
            output: PathBuf::from("out"),
            backend: BackendKind::Pipeline,
            gpu: false,
            lang: None,
            no_formula: false,
            no_table: false,
            pages: None,
            no_images: false,
            vlm_url: None,
            vlm_model: None,
        };
        f(&mut args);
        args
    }

    #[test]
    fn parse_options_reflects_flags() {
        let args = args_with(|a| {
            a.lang = Some("ch".to_owned());
            a.no_formula = true;
        });
        let opts = args.parse_options().unwrap();
        assert_eq!(opts.lang, Some(Lang("ch".to_owned())));
        assert!(!opts.formula);
        assert!(opts.table);
    }

    #[test]
    fn no_images_selects_nlp_markdown() {
        assert_eq!(args_with(|_| {}).make_mode(), MakeMode::MmMarkdown);
        assert_eq!(
            args_with(|a| a.no_images = true).make_mode(),
            MakeMode::NlpMarkdown
        );
    }

    #[test]
    fn input_prefers_positional_then_flag_alias() {
        assert_eq!(args_with(|_| {}).input(), None);
        assert_eq!(
            args_with(|a| a.input_flag = Some(PathBuf::from("via-flag.pdf"))).input(),
            Some(std::path::Path::new("via-flag.pdf"))
        );
        assert_eq!(
            args_with(|a| a.input = Some(PathBuf::from("positional.pdf"))).input(),
            Some(std::path::Path::new("positional.pdf"))
        );
    }

    #[test]
    fn cli_parses_without_panicking() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
