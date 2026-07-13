//! Phase 0 end-to-end gate: a hand-built `Document` renders to the expected
//! Markdown and content-list JSON shape. Proves `mineru-types` + `mineru-render`
//! compose into a working output path independent of any model.

use mineru_render::{render_content_list, render_markdown, MakeMode};
use mineru_types::{
    document::{Block, Page, PageSize, TextLine, TextRole, TitleLevel},
    BBox, Captioned, Document, ImageBody, ImageRef, Score, Span, TableBody, TextBlock,
};

fn text_line(text: &str) -> TextLine {
    TextLine {
        bbox: BBox::new(0.0, 0.0, 100.0, 10.0),
        spans: vec![Span::Text {
            bbox: BBox::new(0.0, 0.0, 100.0, 10.0),
            text: text.to_owned(),
            score: Score(1.0),
        }],
    }
}

fn sample_document() -> Document {
    let title = Block::Text {
        bbox: BBox::new(0.0, 0.0, 100.0, 10.0),
        role: TextRole::Title(TitleLevel(1)),
        lines: vec![text_line("Hello MinerU")],
    };
    let body = Block::Text {
        bbox: BBox::new(0.0, 12.0, 100.0, 22.0),
        role: TextRole::Body,
        lines: vec![text_line("A paragraph of body text.")],
    };
    let figure = Block::Image(Captioned {
        body: ImageBody {
            bbox: BBox::new(0.0, 24.0, 100.0, 80.0),
            image: ImageRef("fig1.png".to_owned()),
        },
        captions: vec![TextBlock {
            bbox: BBox::new(0.0, 82.0, 100.0, 90.0),
            lines: vec![text_line("Figure 1.")],
        }],
        footnotes: vec![],
    });
    let table = Block::Table(Captioned::bare(TableBody {
        bbox: BBox::new(0.0, 92.0, 100.0, 140.0),
        html: mineru_types::Html("<table><tr><td>x</td></tr></table>".to_owned()),
        image: None,
    }));

    Document {
        pages: vec![Page {
            index: 0,
            size: PageSize {
                width: 200.0,
                height: 300.0,
            },
            blocks: vec![title, body, figure, table],
            discarded: vec![],
        }],
    }
}

#[test]
fn renders_markdown() {
    let doc = sample_document();
    let md = render_markdown(&doc, MakeMode::MmMarkdown, "images");

    assert!(md.contains("# Hello MinerU"), "title heading missing:\n{md}");
    assert!(md.contains("A paragraph of body text."), "body missing:\n{md}");
    assert!(
        md.contains("![](images/fig1.png)"),
        "image markdown missing:\n{md}"
    );
    assert!(md.contains("Figure 1."), "caption missing:\n{md}");
    assert!(md.contains("<table>"), "table html missing:\n{md}");
}

#[test]
fn nlp_markdown_drops_images() {
    let doc = sample_document();
    let md = render_markdown(&doc, MakeMode::NlpMarkdown, "images");
    assert!(
        !md.contains("fig1.png"),
        "nlp markdown should drop images:\n{md}"
    );
    assert!(md.contains("# Hello MinerU"));
}

#[test]
fn content_list_has_items_for_each_block() {
    let doc = sample_document();
    let items = render_content_list(&doc, "images");
    // title, body, image, table => 4 items
    assert_eq!(items.len(), 4, "expected one content item per block");

    let json = serde_json::to_string(&items).expect("serialize content list");
    assert!(json.contains("\"type\":\"text\""));
    assert!(json.contains("\"type\":\"image\""));
    assert!(json.contains("\"type\":\"table\""));
}
