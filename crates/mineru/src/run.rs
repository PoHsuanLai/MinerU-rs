//! The one-shot parse flow: PDF in, Markdown + content-list JSON out.

use std::path::Path;

use anyhow::Context;
use mineru_render::{render_content_list, render_markdown, MakeMode};
use mineru_types::{Backend, DocInput, ParseOptions};

/// Subdirectory (relative to the output dir) that image references point into.
///
/// Image extraction/writing is not yet wired end-to-end; references in the
/// rendered output resolve against this directory by convention, matching the
/// Python output layout.
const IMAGE_DIR: &str = "images";

/// Runs the full parse and writes outputs, returning the two written paths.
///
/// Reads `input`, parses it with `backend`, and writes `<stem>.md` and
/// `<stem>_content_list.json` under `output_dir`.
///
/// # Errors
/// Propagates I/O errors and any backend/analysis error.
pub async fn run_parse(
    backend: &dyn Backend,
    input: &Path,
    output_dir: &Path,
    opts: &ParseOptions,
    mode: MakeMode,
) -> anyhow::Result<(std::path::PathBuf, std::path::PathBuf)> {
    let bytes = std::fs::read(input)
        .with_context(|| format!("reading input PDF {}", input.display()))?;

    tracing::info!(input = %input.display(), bytes = bytes.len(), "parsing document");
    let doc = backend
        .analyze(DocInput::new(bytes), opts)
        .await
        .map_err(|e| anyhow::anyhow!("backend analyze failed: {e}"))?;
    tracing::info!(pages = doc.pages.len(), "parsed document");

    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("creating output dir {}", output_dir.display()))?;

    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    let md = render_markdown(&doc, mode, IMAGE_DIR);
    let md_path = output_dir.join(format!("{stem}.md"));
    std::fs::write(&md_path, md)
        .with_context(|| format!("writing markdown {}", md_path.display()))?;

    let content_list = render_content_list(&doc, IMAGE_DIR);
    let json = serde_json::to_string_pretty(&content_list)
        .context("serializing content list")?;
    let json_path = output_dir.join(format!("{stem}_content_list.json"));
    std::fs::write(&json_path, json)
        .with_context(|| format!("writing content list {}", json_path.display()))?;

    tracing::info!(markdown = %md_path.display(), content_list = %json_path.display(), "wrote outputs");
    Ok((md_path, json_path))
}
