//! Converts the VLM's raw per-page blocks into the typed [`Document`] tree.
//!
//! This is the Rust analogue of the Python `vlm_magic_model.py` "magic model": it
//! denormalizes boxes, maps label strings to typed blocks, splits inline `\(...\)`
//! formulas into text + inline-equation spans, and regroups visual bodies with
//! their captions/footnotes into [`Captioned`] blocks. Body/caption/footnote
//! grouping is spatial (a caption is attached to the nearest visual body).

use mineru_types::document::{Block, Page, PageSize, TextLine, TextRole, TitleLevel};
use mineru_types::{
    BBox, Captioned, CodeBody, Document, Html, ImageBody, Latex, Score, Span, TableBody, TextBlock,
};

use crate::raw::{VlmBlock, VlmPage};

/// Assembles a full document from per-page VLM output.
pub fn assemble_document(pages: Vec<VlmPage>) -> Document {
    let pages = pages
        .into_iter()
        .enumerate()
        .map(|(index, page)| assemble_page(index, page))
        .collect();
    Document { pages }
}

/// The role a raw block plays once mapped from its label string.
enum Mapped {
    /// A flowing-text block with a resolved role.
    Text(TextRole, BBox, Vec<Span>),
    /// A standalone equation.
    Equation(BBox, Latex),
    /// A visual body (image/chart/table/code) plus what kind it is.
    Visual(VisualKind, BBox, VisualBody),
    /// A caption fragment to be attached to a nearby visual body.
    Caption(BBox, Vec<Span>),
    /// A footnote fragment to be attached to a nearby visual body.
    Footnote(BBox, Vec<Span>),
    /// A block whose label was unrecognized; dropped.
    Ignored,
}

#[derive(Clone, Copy, PartialEq)]
enum VisualKind {
    Image,
    Chart,
    Table,
    Code,
}

enum VisualBody {
    Image(ImageBody),
    Table(TableBody),
    Code(CodeBody),
}

/// A visual body accumulating the captions/footnotes attached to it before it
/// becomes a [`Captioned`] block.
struct PendingVisual {
    kind: VisualKind,
    bbox: BBox,
    body: VisualBody,
    captions: Vec<TextBlock>,
    footnotes: Vec<TextBlock>,
}

fn assemble_page(index: usize, page: VlmPage) -> Page {
    let (w, h) = (page.width, page.height);
    let mapped: Vec<Mapped> = page
        .blocks
        .into_iter()
        .map(|b| map_block(b, w, h))
        .collect();

    // First pass: collect visual bodies (to attach captions to) and flat blocks.
    let mut blocks: Vec<Block> = Vec::new();
    let mut discarded: Vec<Block> = Vec::new();
    let mut visuals: Vec<PendingVisual> = Vec::new();
    let mut pending_captions: Vec<(BBox, TextBlock)> = Vec::new();
    let mut pending_footnotes: Vec<(BBox, TextBlock)> = Vec::new();

    for m in mapped {
        match m {
            Mapped::Text(role, bbox, spans) => {
                let block = Block::Text {
                    bbox,
                    role,
                    lines: vec![TextLine { bbox, spans }],
                };
                if is_discarded(role) {
                    discarded.push(block);
                } else {
                    blocks.push(block);
                }
            }
            Mapped::Equation(bbox, latex) => {
                blocks.push(Block::InterlineEquation { bbox, latex });
            }
            Mapped::Visual(kind, bbox, body) => {
                visuals.push(PendingVisual {
                    kind,
                    bbox,
                    body,
                    captions: Vec::new(),
                    footnotes: Vec::new(),
                });
            }
            Mapped::Caption(bbox, spans) => {
                pending_captions.push((bbox, TextBlock { bbox, lines: vec![TextLine { bbox, spans }] }));
            }
            Mapped::Footnote(bbox, spans) => {
                pending_footnotes.push((bbox, TextBlock { bbox, lines: vec![TextLine { bbox, spans }] }));
            }
            Mapped::Ignored => {}
        }
    }

    // Attach each caption/footnote to the nearest visual body it overlaps or sits under.
    for (cbbox, tb) in pending_captions {
        match nearest_visual(&visuals, cbbox) {
            Some(idx) => visuals[idx].captions.push(tb),
            // Orphan caption becomes body text (matches Python's unmatched fallback).
            None => blocks.push(caption_as_text(tb)),
        }
    }
    for (fbbox, tb) in pending_footnotes {
        match nearest_visual(&visuals, fbbox) {
            Some(idx) => visuals[idx].footnotes.push(tb),
            None => blocks.push(caption_as_text(tb)),
        }
    }

    // Emit the regrouped visual blocks.
    for v in visuals {
        let (captions, footnotes) = (v.captions, v.footnotes);
        let block = match v.body {
            VisualBody::Image(b) if v.kind == VisualKind::Chart => {
                Block::Chart(Captioned { body: b, captions, footnotes })
            }
            VisualBody::Image(b) => Block::Image(Captioned { body: b, captions, footnotes }),
            VisualBody::Table(b) => Block::Table(Captioned { body: b, captions, footnotes }),
            VisualBody::Code(b) => Block::Code(Captioned { body: b, captions, footnotes }),
        };
        blocks.push(block);
    }

    Page {
        index,
        size: PageSize {
            width: w,
            height: h,
        },
        blocks,
        discarded,
    }
}

/// Maps one raw block to its typed role, denormalizing its box to pixels.
fn map_block(b: VlmBlock, w: f32, h: f32) -> Mapped {
    let bbox = denormalize(b.bbox, w, h);
    let content = b.content.unwrap_or_default();

    match b.label.as_str() {
        "text" | "phonetic" => Mapped::Text(TextRole::Body, bbox, text_spans(&content, bbox)),
        "title" => Mapped::Text(
            TextRole::Title(TitleLevel(1)),
            bbox,
            text_spans(&normalize_title(&content), bbox),
        ),
        "list" => Mapped::Text(TextRole::List, bbox, text_spans(&content, bbox)),
        "ref_text" => Mapped::Text(TextRole::RefText, bbox, text_spans(&content, bbox)),
        "header" => Mapped::Text(TextRole::Header, bbox, text_spans(&content, bbox)),
        "footer" => Mapped::Text(TextRole::Footer, bbox, text_spans(&content, bbox)),
        "page_number" => Mapped::Text(TextRole::PageNumber, bbox, text_spans(&content, bbox)),
        "aside_text" => Mapped::Text(TextRole::AsideText, bbox, text_spans(&content, bbox)),
        "page_footnote" => Mapped::Text(TextRole::PageFootnote, bbox, text_spans(&content, bbox)),

        "image_caption" | "table_caption" | "code_caption" => {
            Mapped::Caption(bbox, text_spans(&content, bbox))
        }
        "image_footnote" | "table_footnote" => Mapped::Footnote(bbox, text_spans(&content, bbox)),

        "image" | "image_block" => Mapped::Visual(
            VisualKind::Image,
            bbox,
            VisualBody::Image(ImageBody {
                bbox,
                // The extracted raster path is filled in by the caller after cropping.
                image: mineru_types::ImageRef(String::new()),
            }),
        ),
        "chart" => Mapped::Visual(
            VisualKind::Chart,
            bbox,
            VisualBody::Image(ImageBody {
                bbox,
                image: mineru_types::ImageRef(String::new()),
            }),
        ),
        "table" => Mapped::Visual(
            VisualKind::Table,
            bbox,
            VisualBody::Table(TableBody {
                bbox,
                html: Html(content),
                image: None,
            }),
        ),
        "code" | "algorithm" => Mapped::Visual(
            VisualKind::Code,
            bbox,
            VisualBody::Code(CodeBody {
                bbox,
                lines: vec![TextLine {
                    bbox,
                    spans: text_spans(&content, bbox),
                }],
                language: None,
            }),
        ),
        "equation" => Mapped::Equation(bbox, Latex(content.trim().to_owned())),

        _ => Mapped::Ignored,
    }
}

/// Denormalizes a `0..1` box to pixels and orders its corners.
fn denormalize([x0, y0, x1, y1]: [f32; 4], w: f32, h: f32) -> BBox {
    BBox::new(x0 * w, y0 * h, x1 * w, y1 * h)
}

/// Builds spans from text, splitting inline `\(...\)` fragments into inline equations.
fn text_spans(content: &str, bbox: BBox) -> Vec<Span> {
    let opens = content.matches("\\(").count();
    let closes = content.matches("\\)").count();
    if opens == 0 || opens != closes {
        return vec![Span::Text {
            bbox,
            text: content.to_owned(),
            score: Score(1.0),
        }];
    }

    let mut spans = Vec::new();
    let mut rest = content;
    while let Some(open) = rest.find("\\(") {
        let (before, after_open) = rest.split_at(open);
        if !before.trim().is_empty() {
            spans.push(Span::Text {
                bbox,
                text: before.to_owned(),
                score: Score(1.0),
            });
        }
        let after_open = &after_open[2..]; // skip "\("
        match after_open.find("\\)") {
            Some(close) => {
                let formula = &after_open[..close];
                spans.push(Span::InlineEquation {
                    bbox,
                    latex: Latex(formula.trim().to_owned()),
                    score: Score(1.0),
                });
                rest = &after_open[close + 2..]; // skip "\)"
            }
            None => {
                // Unbalanced despite the count check; keep remainder as text.
                spans.push(Span::Text {
                    bbox,
                    text: after_open.to_owned(),
                    score: Score(1.0),
                });
                rest = "";
                break;
            }
        }
    }
    if !rest.trim().is_empty() {
        spans.push(Span::Text {
            bbox,
            text: rest.to_owned(),
            score: Score(1.0),
        });
    }
    if spans.is_empty() {
        spans.push(Span::Text {
            bbox,
            text: String::new(),
            score: Score(1.0),
        });
    }
    spans
}

/// Titles collapse internal newlines to single spaces (matches the Python).
fn normalize_title(content: &str) -> String {
    let mut out = String::with_capacity(content.len());
    let mut prev_ws_after_nl = false;
    for c in content.chars() {
        if c == '\n' {
            out.push(' ');
            prev_ws_after_nl = true;
        } else if prev_ws_after_nl && c.is_whitespace() {
            // collapse the run of whitespace following a newline
        } else {
            prev_ws_after_nl = false;
            out.push(c);
        }
    }
    out.trim().to_owned()
}

/// Roles that are stripped from the main flow into `discarded`.
fn is_discarded(role: TextRole) -> bool {
    matches!(
        role,
        TextRole::Header
            | TextRole::Footer
            | TextRole::PageNumber
            | TextRole::AsideText
            | TextRole::PageFootnote
    )
}

/// Finds the index of the visual body a caption/footnote box best belongs to:
/// the one with greatest overlap, else the nearest vertically-adjacent one.
fn nearest_visual(visuals: &[PendingVisual], bbox: BBox) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, v) in visuals.iter().enumerate() {
        let overlap = bbox.overlap_ratio(&v.bbox);
        let (cx, cy) = bbox.center();
        let (vx, vy) = v.bbox.center();
        let dist = ((cx - vx).powi(2) + (cy - vy).powi(2)).sqrt();
        // Prefer overlap; fall back to proximity (negative distance as score).
        let score = if overlap > 0.0 { overlap } else { -dist / 1e6 };
        if best.map(|(_, s)| score > s).unwrap_or(true) {
            best = Some((i, score));
        }
    }
    best.map(|(i, _)| i)
}

/// Turns an orphaned caption into a plain body text block.
fn caption_as_text(tb: TextBlock) -> Block {
    Block::Text {
        bbox: tb.bbox,
        role: TextRole::Body,
        lines: tb.lines,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(label: &str, content: &str, bbox: [f32; 4]) -> VlmBlock {
        VlmBlock {
            bbox,
            label: label.to_owned(),
            content: Some(content.to_owned()),
            angle: 0,
            sub_type: None,
        }
    }

    #[test]
    fn denormalizes_and_maps_title() {
        let page = VlmPage {
            width: 200.0,
            height: 100.0,
            blocks: vec![block("title", "Hello", [0.0, 0.0, 0.5, 0.1])],
        };
        let doc = assemble_document(vec![page]);
        let p = &doc.pages[0];
        assert_eq!(p.blocks.len(), 1);
        match &p.blocks[0] {
            Block::Text { role: TextRole::Title(l), bbox, .. } => {
                assert_eq!(l.0, 1);
                assert_eq!((bbox.x1, bbox.y1), (100.0, 10.0));
            }
            _ => panic!("expected title"),
        }
    }

    #[test]
    fn splits_inline_formula() {
        let spans = text_spans("energy \\(E=mc^2\\) equation", BBox::new(0.0, 0.0, 1.0, 1.0));
        assert_eq!(spans.len(), 3);
        assert!(matches!(spans[1], Span::InlineEquation { .. }));
    }

    #[test]
    fn header_is_discarded() {
        let page = VlmPage {
            width: 100.0,
            height: 100.0,
            blocks: vec![block("header", "running head", [0.0, 0.0, 1.0, 0.05])],
        };
        let doc = assemble_document(vec![page]);
        assert!(doc.pages[0].blocks.is_empty());
        assert_eq!(doc.pages[0].discarded.len(), 1);
    }

    #[test]
    fn caption_attaches_to_image() {
        let page = VlmPage {
            width: 100.0,
            height: 100.0,
            blocks: vec![
                block("image", "", [0.1, 0.1, 0.9, 0.6]),
                block("image_caption", "Figure 1", [0.1, 0.61, 0.9, 0.7]),
            ],
        };
        let doc = assemble_document(vec![page]);
        let imgs: Vec<_> = doc.pages[0]
            .blocks
            .iter()
            .filter(|b| matches!(b, Block::Image(_)))
            .collect();
        assert_eq!(imgs.len(), 1);
        if let Block::Image(c) = imgs[0] {
            assert_eq!(c.captions.len(), 1, "caption should attach to the image");
        }
    }

    #[test]
    fn table_carries_html() {
        let page = VlmPage {
            width: 100.0,
            height: 100.0,
            blocks: vec![block("table", "<table><tr><td>1</td></tr></table>", [0.0, 0.0, 1.0, 1.0])],
        };
        let doc = assemble_document(vec![page]);
        match &doc.pages[0].blocks[0] {
            Block::Table(c) => assert!(c.body.html.as_str().contains("<table>")),
            _ => panic!("expected table"),
        }
    }
}
