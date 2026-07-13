//! Renders the parsed [`Document`](mineru_types::Document) tree to output formats.
//!
//! Two renderers share one document walk:
//!
//! - [`render_markdown`] produces Markdown, in a [`MakeMode`] that controls
//!   whether images/charts are embedded (`mm`) or dropped (`nlp`).
//! - [`render_content_list`] produces a `Vec<`[`ContentItem`]`>` that serializes
//!   to Python's `content_list.json` shape.
//!
//! Both map [`Block`](mineru_types::Block)s through an *exhaustive* `match`, so a
//! new block variant in `mineru-types` will fail to compile here until it is
//! given a rendering — the type system, not review, guards output coverage.
//!
//! Rendering is infallible: the public entry points return plain `String` /
//! `Vec` rather than [`Result`]. The [`error`] module exists for the per-crate
//! convention and for callers that serialize a [`ContentItem`] to JSON.

pub mod content_list;
pub mod error;
pub mod markdown;
pub mod mode;

mod path;
mod text;

pub use content_list::{render_content_list, ContentItem};
pub use error::{Error, Result};
pub use markdown::render_markdown;
pub use mode::MakeMode;

#[cfg(test)]
mod tests {
    use super::*;
    use mineru_types::{
        BBox, Block, Captioned, CodeBody, Document, ImageBody, ImageRef, Lang, Latex, Page,
        PageSize, Score, Span, TableBody, TextBlock, TextLine, TextRole, TitleLevel,
    };

    fn bbox() -> BBox {
        BBox::new(0.0, 0.0, 10.0, 10.0)
    }

    /// A single-span text line.
    fn text_line(s: &str) -> TextLine {
        TextLine {
            bbox: bbox(),
            spans: vec![Span::Text {
                bbox: bbox(),
                text: s.to_owned(),
                score: Score(1.0),
            }],
        }
    }

    fn text_block(role: TextRole, s: &str) -> Block {
        Block::Text {
            bbox: bbox(),
            role,
            lines: vec![text_line(s)],
        }
    }

    /// Wraps blocks into a one-page document.
    fn doc(blocks: Vec<Block>) -> Document {
        Document {
            pages: vec![Page {
                index: 0,
                size: PageSize {
                    width: 100.0,
                    height: 100.0,
                },
                blocks,
                discarded: Vec::new(),
            }],
        }
    }

    // NOTE ON EXHAUSTIVENESS: `render_block` in `markdown.rs` and `content_item`
    // in `content_list.rs` each `match` on `Block` with no wildcard arm. Adding a
    // variant to `mineru_types::Block` will break *compilation* of this crate
    // until it is rendered — that compile-time guarantee is the intended safety
    // net, so these tests assert behavior rather than trying (and failing) to
    // prove a negative at runtime.

    #[test]
    fn title_becomes_heading() {
        let d = doc(vec![text_block(TextRole::Title(TitleLevel(2)), "Intro")]);
        assert_eq!(render_markdown(&d, MakeMode::MmMarkdown, "img"), "## Intro");
    }

    #[test]
    fn doc_title_level_zero_uses_single_hash() {
        let d = doc(vec![text_block(TextRole::Title(TitleLevel(0)), "Doc")]);
        // level 0 clamps to depth 1 in markdown.
        assert_eq!(render_markdown(&d, MakeMode::MmMarkdown, "img"), "# Doc");
    }

    #[test]
    fn body_is_plain_text() {
        let d = doc(vec![text_block(TextRole::Body, "Hello world")]);
        assert_eq!(
            render_markdown(&d, MakeMode::MmMarkdown, "img"),
            "Hello world"
        );
    }

    #[test]
    fn discard_roles_are_dropped_from_markdown() {
        let d = doc(vec![
            text_block(TextRole::Header, "page header"),
            text_block(TextRole::Body, "keep me"),
            text_block(TextRole::PageNumber, "1"),
        ]);
        assert_eq!(render_markdown(&d, MakeMode::MmMarkdown, "img"), "keep me");
    }

    #[test]
    fn interline_equation_is_fenced() {
        let d = doc(vec![Block::InterlineEquation {
            bbox: bbox(),
            latex: Latex("E=mc^2".to_owned()),
        }]);
        assert_eq!(
            render_markdown(&d, MakeMode::MmMarkdown, "img"),
            "$$\nE=mc^2\n$$"
        );
    }

    #[test]
    fn inline_equation_span_survives_into_line() {
        let line = TextLine {
            bbox: bbox(),
            spans: vec![
                Span::Text {
                    bbox: bbox(),
                    text: "a ".to_owned(),
                    score: Score(1.0),
                },
                Span::InlineEquation {
                    bbox: bbox(),
                    latex: Latex("x^2".to_owned()),
                    score: Score(1.0),
                },
            ],
        };
        let d = doc(vec![Block::Text {
            bbox: bbox(),
            role: TextRole::Body,
            lines: vec![line],
        }]);
        assert_eq!(render_markdown(&d, MakeMode::MmMarkdown, "img"), "a $x^2$");
    }

    fn image_block() -> Block {
        Block::Image(Captioned {
            body: ImageBody {
                bbox: bbox(),
                image: ImageRef("fig1.png".to_owned()),
            },
            captions: vec![TextBlock {
                bbox: bbox(),
                lines: vec![text_line("Figure 1")],
            }],
            footnotes: Vec::new(),
        })
    }

    #[test]
    fn image_renders_with_path_and_caption_in_mm() {
        let d = doc(vec![image_block()]);
        assert_eq!(
            render_markdown(&d, MakeMode::MmMarkdown, "images"),
            "![](images/fig1.png)\nFigure 1"
        );
    }

    #[test]
    fn image_dropped_in_nlp() {
        let d = doc(vec![image_block(), text_block(TextRole::Body, "text")]);
        assert_eq!(render_markdown(&d, MakeMode::NlpMarkdown, "img"), "text");
    }

    #[test]
    fn table_renders_html_with_caption_above() {
        let d = doc(vec![Block::Table(Captioned {
            body: TableBody {
                bbox: bbox(),
                html: "<table></table>".into(),
                image: None,
            },
            captions: vec![TextBlock {
                bbox: bbox(),
                lines: vec![text_line("Table 1")],
            }],
            footnotes: Vec::new(),
        })]);
        assert_eq!(
            render_markdown(&d, MakeMode::MmMarkdown, "img"),
            "Table 1\n<table></table>"
        );
    }

    #[test]
    fn code_renders_fenced_with_language() {
        let d = doc(vec![Block::Code(Captioned::bare(CodeBody {
            bbox: bbox(),
            lines: vec![text_line("print(1)"), text_line("print(2)")],
            language: Some(Lang("python".to_owned())),
        }))]);
        assert_eq!(
            render_markdown(&d, MakeMode::MmMarkdown, "img"),
            "```python\nprint(1)\nprint(2)\n```"
        );
    }

    #[test]
    fn blocks_join_with_blank_line() {
        let d = doc(vec![
            text_block(TextRole::Title(TitleLevel(1)), "Title"),
            text_block(TextRole::Body, "Para"),
        ]);
        assert_eq!(
            render_markdown(&d, MakeMode::MmMarkdown, "img"),
            "# Title\n\nPara"
        );
    }

    #[test]
    fn content_list_title_sets_text_level() {
        let d = doc(vec![text_block(TextRole::Title(TitleLevel(3)), "Sec")]);
        let items = render_content_list(&d, "img");
        assert_eq!(
            items,
            vec![ContentItem::Text {
                text: "Sec".to_owned(),
                text_level: Some(3),
            }]
        );
    }

    #[test]
    fn content_list_doc_title_omits_level() {
        let d = doc(vec![text_block(TextRole::Title(TitleLevel(0)), "Doc")]);
        let json = serde_json::to_value(render_content_list(&d, "img")).unwrap();
        // No `text_level` key for a level-0 (doc) title.
        assert_eq!(json, serde_json::json!([{ "type": "text", "text": "Doc" }]));
    }

    #[test]
    fn content_list_image_shape_matches_python() {
        let d = doc(vec![image_block()]);
        let json = serde_json::to_value(render_content_list(&d, "images")).unwrap();
        assert_eq!(
            json,
            serde_json::json!([{
                "type": "image",
                "img_path": "images/fig1.png",
                "image_caption": ["Figure 1"],
            }])
        );
    }

    #[test]
    fn content_list_equation_carries_latex_format() {
        let d = doc(vec![Block::InterlineEquation {
            bbox: bbox(),
            latex: Latex("E=mc^2".to_owned()),
        }]);
        let json = serde_json::to_value(render_content_list(&d, "img")).unwrap();
        assert_eq!(
            json,
            serde_json::json!([{
                "type": "equation",
                "text": "E=mc^2",
                "text_format": "latex",
            }])
        );
    }

    #[test]
    fn content_list_table_with_raster() {
        let d = doc(vec![Block::Table(Captioned::bare(TableBody {
            bbox: bbox(),
            html: "<table></table>".into(),
            image: Some(ImageRef("t.png".to_owned())),
        }))]);
        let json = serde_json::to_value(render_content_list(&d, "img")).unwrap();
        assert_eq!(
            json,
            serde_json::json!([{
                "type": "table",
                "img_path": "img/t.png",
                "table_body": "<table></table>",
            }])
        );
    }
}
