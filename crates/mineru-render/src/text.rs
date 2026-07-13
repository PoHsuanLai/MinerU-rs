//! Shared text-flattening helpers.
//!
//! Both the Markdown and the `content_list` renderers need to collapse a
//! [`TextLine`]'s inline [`Span`]s into a single string and to collect the text
//! of a set of caption/footnote [`TextBlock`]s. Those two operations live here
//! once so neither renderer re-implements them.

use mineru_types::{Span, TextBlock, TextLine};

/// Flattens one line's spans into a single string.
///
/// - [`Span::Text`] contributes its text verbatim.
/// - [`Span::InlineEquation`] contributes `$latex$` so inline math survives into
///   Markdown.
/// - [`Span::Image`] is skipped: an inline raster has no textual form, and both
///   renderers handle block-level images separately.
pub(crate) fn flatten_line(line: &TextLine) -> String {
    let mut out = String::new();
    for span in &line.spans {
        match span {
            Span::Text { text, .. } => out.push_str(text),
            Span::InlineEquation { latex, .. } => {
                out.push('$');
                out.push_str(latex.as_str());
                out.push('$');
            }
            Span::Image { .. } => {}
        }
    }
    out
}

/// Merges a block's lines into one string, joining lines with a single space.
///
/// This mirrors Python's `merge_para_with_text`, which concatenates a
/// paragraph's lines into flowing text.
pub(crate) fn merge_lines(lines: &[TextLine]) -> String {
    lines
        .iter()
        .map(flatten_line)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Collects the merged text of each caption/footnote block, dropping any that
/// flatten to empty.
pub(crate) fn collect_texts(blocks: &[TextBlock]) -> Vec<String> {
    blocks
        .iter()
        .map(|b| merge_lines(&b.lines))
        .filter(|s| !s.trim().is_empty())
        .collect()
}
