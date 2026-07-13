//! The Markdown renderer.
//!
//! Walks the [`Document`] block-by-block and maps each [`Block`] to at most one
//! Markdown fragment; the fragments are joined with blank lines. The core
//! [`render_block`] match is *exhaustive over `Block`*, on purpose: adding a new
//! block variant to `mineru-types` must fail to compile here until it is
//! rendered, rather than silently vanishing from the output.

use mineru_types::{Block, Captioned, CodeBody, Document, ImageBody, TextRole};

use crate::mode::MakeMode;
use crate::path::join_image;
use crate::text::{collect_texts, merge_lines};

/// Renders a whole document to a Markdown string.
///
/// `image_dir` is the directory image references are resolved against (joined
/// with each [`ImageRef`](mineru_types::ImageRef)). Blocks that produce no
/// Markdown in the chosen `mode` (discarded roles, or images under
/// [`MakeMode::NlpMarkdown`]) are skipped, and the rest are joined with blank
/// lines.
pub fn render_markdown(doc: &Document, mode: MakeMode, image_dir: &str) -> String {
    doc.blocks()
        .filter_map(|block| render_block(block, mode, image_dir))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Renders a single block, or `None` when it contributes nothing to the flow.
///
/// The `match` is deliberately exhaustive over [`Block`]: a newly added variant
/// will break compilation here until it is given a rendering, which is the whole
/// point of modeling blocks as an enum.
fn render_block(block: &Block, mode: MakeMode, image_dir: &str) -> Option<String> {
    match block {
        Block::Text { role, lines, .. } => render_text(*role, merge_lines(lines)),
        Block::InterlineEquation { latex, .. } => {
            Some(format!("$$\n{}\n$$", latex.as_str()))
        }
        Block::Image(captioned) | Block::Chart(captioned) => {
            render_image(captioned, mode, image_dir)
        }
        Block::Table(captioned) => Some(render_table(captioned)),
        Block::Code(captioned) => Some(render_code(captioned)),
    }
}

/// Renders a flowing-text block per its role.
///
/// Titles become ATX headings; discard-y roles (headers, footers, page numbers,
/// footnotes, aside text) are kept out of the main flow entirely; every other
/// role emits its merged line text.
fn render_text(role: TextRole, text: String) -> Option<String> {
    match role {
        TextRole::Title(level) => {
            let depth = usize::from(level.0).max(1);
            Some(format!("{} {}", "#".repeat(depth), text))
        }
        TextRole::Header
        | TextRole::Footer
        | TextRole::PageNumber
        | TextRole::PageFootnote
        | TextRole::AsideText => None,
        TextRole::Body
        | TextRole::List
        | TextRole::Index
        | TextRole::Abstract
        | TextRole::RefText => Some(text),
    }
}

/// Renders an image/chart body plus its captions.
///
/// Returns `None` under [`MakeMode::NlpMarkdown`] (text-only). Otherwise emits an
/// image tag with the resolved path, then each caption on its own line.
fn render_image(
    captioned: &Captioned<ImageBody>,
    mode: MakeMode,
    image_dir: &str,
) -> Option<String> {
    if !mode.keeps_images() {
        return None;
    }
    let path = join_image(image_dir, captioned.body.image.as_str());
    let mut out = format!("![]({path})");
    for caption in collect_texts(&captioned.captions) {
        out.push('\n');
        out.push_str(&caption);
    }
    Some(out)
}

/// Renders a table: its captions above, then the raw HTML markup (tables stay as
/// HTML in Markdown output).
fn render_table(captioned: &Captioned<mineru_types::TableBody>) -> String {
    let mut out = String::new();
    for caption in collect_texts(&captioned.captions) {
        out.push_str(&caption);
        out.push('\n');
    }
    out.push_str(captioned.body.html.as_str());
    out
}

/// Renders a code block as a fenced block, tagging the fence with the language
/// when one is known.
fn render_code(captioned: &Captioned<CodeBody>) -> String {
    let lang = captioned
        .body
        .language
        .as_ref()
        .map(|l| l.as_str())
        .unwrap_or("");
    let body = merge_code_lines(&captioned.body);
    format!("```{lang}\n{body}\n```")
}

/// Joins a code body's lines with newlines (code preserves line breaks, unlike
/// flowing paragraphs).
fn merge_code_lines(body: &CodeBody) -> String {
    body.lines
        .iter()
        .map(crate::text::flatten_line)
        .collect::<Vec<_>>()
        .join("\n")
}
