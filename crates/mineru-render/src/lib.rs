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

pub use content_list::{render_content_list, ContentBody, ContentItem};
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
        // Heading depth = level + 1: a section title (level 1) is `##`, matching
        // Python's `paragraph_title` → level 2 → `##`.
        let d = doc(vec![text_block(TextRole::Title(TitleLevel(1)), "Intro")]);
        assert_eq!(render_markdown(&d, MakeMode::MmMarkdown, "img"), "## Intro");
        // A deeper subsection (level 2) is `###`.
        let d = doc(vec![text_block(TextRole::Title(TitleLevel(2)), "Sub")]);
        assert_eq!(render_markdown(&d, MakeMode::MmMarkdown, "img"), "### Sub");
    }

    #[test]
    fn doc_title_level_zero_uses_single_hash() {
        let d = doc(vec![text_block(TextRole::Title(TitleLevel(0)), "Doc")]);
        // The document title (level 0) is a single `#` (Python `doc_title` → level 1).
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

    /// A one-page document whose `discarded` list holds `discarded`.
    fn doc_with_discarded(blocks: Vec<Block>, discarded: Vec<Block>) -> Document {
        Document {
            pages: vec![Page {
                index: 0,
                size: PageSize {
                    width: 100.0,
                    height: 100.0,
                },
                blocks,
                discarded,
            }],
        }
    }

    /// The `type` tag each item serializes under.
    fn types(items: &[ContentItem]) -> Vec<String> {
        items
            .iter()
            .map(|i| match serde_json::to_value(i) {
                Ok(serde_json::Value::Object(m)) => m
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("?")
                    .to_owned(),
                _ => "?".to_owned(),
            })
            .collect()
    }

    /// Python emits header/footer/page-number/aside-text/page-footnote under
    /// their own `type`, sourced from `discarded_blocks` — not as `text`, and
    /// not dropped (`pipeline_middle_json_mkcontent.py:622`).
    #[test]
    fn discarded_blocks_are_emitted_under_their_own_types() {
        let d = doc_with_discarded(
            vec![text_block(TextRole::Body, "keep me")],
            vec![
                text_block(TextRole::Header, "hdr"),
                text_block(TextRole::Footer, "ftr"),
                text_block(TextRole::PageNumber, "7"),
                text_block(TextRole::AsideText, "aside"),
                text_block(TextRole::PageFootnote, "note"),
            ],
        );
        let items = render_content_list(&d, "img");
        assert_eq!(
            types(&items),
            ["text", "header", "footer", "page_number", "aside_text", "page_footnote"]
        );
        // The text must survive, not just the tag.
        assert_eq!(
            items.get(3).map(|i| i.content.clone()),
            Some(ContentBody::PageNumber {
                text: "7".to_owned()
            })
        );
    }

    /// Adjacent `ref_text` blocks collapse into one `list` entry carrying each
    /// item's text (Python `merge_adjacent_ref_text_blocks_for_content`).
    #[test]
    fn adjacent_ref_text_blocks_become_one_list() {
        let d = doc(vec![
            text_block(TextRole::Body, "before"),
            text_block(TextRole::RefText, "[1] First."),
            text_block(TextRole::RefText, "[2] Second."),
            text_block(TextRole::Body, "after"),
        ]);
        let items = render_content_list(&d, "img");
        assert_eq!(types(&items), ["text", "list", "text"]);
        assert_eq!(
            items.get(1).map(|i| i.content.clone()),
            Some(ContentBody::List {
                sub_type: content_list::ListSubType::RefText,
                list_items: vec!["[1] First.".to_owned(), "[2] Second.".to_owned()],
            })
        );
    }

    /// Python's `flush_ref_group` keeps a *lone* ref_text as a plain block, so it
    /// renders as `text` — only runs of 2+ become a `list`.
    #[test]
    fn lone_ref_text_stays_text() {
        let d = doc(vec![
            text_block(TextRole::Body, "before"),
            text_block(TextRole::RefText, "[1] Only."),
            text_block(TextRole::Body, "after"),
        ]);
        assert_eq!(types(&render_content_list(&d, "img")), ["text", "text", "text"]);
    }

    /// Two ref_text runs separated by other content stay two distinct lists.
    #[test]
    fn separated_ref_runs_do_not_merge() {
        let d = doc(vec![
            text_block(TextRole::RefText, "[1] a"),
            text_block(TextRole::RefText, "[2] b"),
            text_block(TextRole::Title(TitleLevel(1)), "Appendix"),
            text_block(TextRole::RefText, "[3] c"),
            text_block(TextRole::RefText, "[4] d"),
        ]);
        assert_eq!(types(&render_content_list(&d, "img")), ["list", "text", "list"]);
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
            text_block(TextRole::Title(TitleLevel(0)), "Title"),
            text_block(TextRole::Body, "Para"),
        ]);
        assert_eq!(
            render_markdown(&d, MakeMode::MmMarkdown, "img"),
            "# Title\n\nPara"
        );
    }

    /// The regression this guards: `render_content_list` used to walk
    /// `doc.blocks()`, which flattens pages away — every entry silently lost its
    /// page number. A single-page fixture cannot catch that (index 0 is the
    /// default), so this asserts across pages with a non-zero index and a
    /// differently-sized page.
    #[test]
    fn content_list_carries_per_page_index_and_scales_bbox_per_page() {
        let d = Document {
            pages: vec![
                Page {
                    index: 0,
                    size: PageSize {
                        width: 100.0,
                        height: 100.0,
                    },
                    blocks: vec![text_block(TextRole::Body, "first")],
                    discarded: Vec::new(),
                },
                Page {
                    index: 4,
                    // Half-width: the same (0,0,10,10) box must scale to twice
                    // the x extent, proving each page's own size is used.
                    size: PageSize {
                        width: 50.0,
                        height: 100.0,
                    },
                    blocks: vec![text_block(TextRole::Body, "second")],
                    discarded: Vec::new(),
                },
            ],
        };
        let items = render_content_list(&d, "img");
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].page_idx, 0);
        assert_eq!(items[0].bbox, Some([0, 0, 100, 100]));
        assert_eq!(items[1].page_idx, 4, "page index must not be flattened away");
        assert_eq!(items[1].bbox, Some([0, 0, 200, 100]));
    }

    /// A degenerate page size makes the 0..1000 mapping undefined; Python's
    /// `_build_bbox` returns `None` and the key is omitted rather than emitting
    /// an infinity/NaN.
    #[test]
    fn content_list_omits_bbox_for_degenerate_page_size() {
        let d = Document {
            pages: vec![Page {
                index: 0,
                size: PageSize {
                    width: 0.0,
                    height: 0.0,
                },
                blocks: vec![text_block(TextRole::Body, "x")],
                discarded: Vec::new(),
            }],
        };
        let json = serde_json::to_value(render_content_list(&d, "img")).unwrap();
        assert_eq!(
            json,
            serde_json::json!([{ "type": "text", "text": "x", "page_idx": 0 }]),
            "no bbox key when the page size is unusable"
        );
    }

    #[test]
    fn content_list_title_sets_text_level() {
        let d = doc(vec![text_block(TextRole::Title(TitleLevel(3)), "Sec")]);
        let items = render_content_list(&d, "img");
        assert_eq!(
            items,
            vec![ContentItem {
                content: ContentBody::Text {
                    text: "Sec".to_owned(),
                    text_level: Some(3),
                },
                // `bbox()` is (0,0,10,10) on a 100x100 page -> tenths of the 0..1000 scale.
                bbox: Some([0, 0, 100, 100]),
                page_idx: 0,
            }]
        );
    }

    #[test]
    fn content_list_doc_title_omits_level() {
        let d = doc(vec![text_block(TextRole::Title(TitleLevel(0)), "Doc")]);
        let json = serde_json::to_value(render_content_list(&d, "img")).unwrap();
        // No `text_level` key for a level-0 (doc) title.
        assert_eq!(
            json,
            serde_json::json!([{
                "type": "text",
                "text": "Doc",
                "bbox": [0, 0, 100, 100],
                "page_idx": 0,
            }])
        );
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
                "bbox": [0, 0, 100, 100],
                "page_idx": 0,
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
                "bbox": [0, 0, 100, 100],
                "page_idx": 0,
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
                "bbox": [0, 0, 100, 100],
                "page_idx": 0,
            }])
        );
    }
}
