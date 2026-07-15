//! The one-shot parse flow: PDF in, Markdown + content-list JSON out.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use mineru_io::LocalFsImageWriter;
use mineru_render::{render_content_list, render_markdown, MakeMode};
use mineru_types::{Backend, DocInput, ParseOptions};

/// Subdirectory (relative to the output dir) that image references point into.
///
/// Backends crop image/chart/table regions into this directory (via the
/// [`ImageWriter`](mineru_types::ImageWriter) sink injected below), and rendered
/// references resolve against it — matching the Python output layout.
const IMAGE_DIR: &str = "images";

/// Runs the full parse and writes the outputs.
///
/// Reads `input`, parses it with `backend`, and writes `<stem>.md` and
/// `<stem>_content_list.json` under `output_dir` — plus `<stem>_document.json`
/// (the full parsed tree) when `debug_output` is set. Written paths are reported
/// via `tracing`.
///
/// # Errors
/// Propagates I/O errors and any backend/analysis error.
pub async fn run_parse(
    backend: &dyn Backend,
    input: &Path,
    output_dir: &Path,
    opts: &ParseOptions,
    mode: MakeMode,
    debug_output: bool,
) -> anyhow::Result<()> {
    let bytes = std::fs::read(input)
        .with_context(|| format!("reading input PDF {}", input.display()))?;

    // Inject a disk image sink so backends crop image/chart/table regions into
    // `output_dir/images/`. Only in image-keeping modes: under `--no-images`
    // (`NlpMarkdown`) we leave the sink `None`, so no crops are written and the
    // backends stay mode-agnostic (they simply see no sink).
    let opts = if mode.keeps_images() {
        let images_dir = output_dir.join(IMAGE_DIR);
        std::fs::create_dir_all(&images_dir)
            .with_context(|| format!("creating images dir {}", images_dir.display()))?;
        let mut opts = opts.clone();
        opts.image_sink = Some(Arc::new(LocalFsImageWriter::new(images_dir)));
        std::borrow::Cow::Owned(opts)
    } else {
        std::borrow::Cow::Borrowed(opts)
    };

    tracing::info!(input = %input.display(), bytes = bytes.len(), "parsing document");
    let doc = backend
        .analyze(DocInput::new(bytes), &opts)
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

    // The full document tree, opt-in via `--debug-output`: it is large, and most
    // runs only want the Markdown and content list.
    if debug_output {
        let doc_json = serde_json::to_string_pretty(&doc).context("serializing document tree")?;
        let doc_path = output_dir.join(format!("{stem}_document.json"));
        std::fs::write(&doc_path, doc_json)
            .with_context(|| format!("writing document tree {}", doc_path.display()))?;
        tracing::info!(document = %doc_path.display(), "wrote document tree");
    }

    Ok(())
}
