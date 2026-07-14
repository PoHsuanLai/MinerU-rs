//! Guards that a serialized `Document` deserializes back into an equal tree.
//!
//! `mineru --debug-output` writes this shape to `<stem>_document.json`, so it is
//! a consumed data surface: a field that serializes but cannot be read back
//! would break anyone building on the dump.

use mineru_types::*;

#[test]
fn document_json_round_trips() {
    let doc = Document {
        pages: vec![Page {
            index: 3,
            size: PageSize {
                width: 612.0,
                height: 792.0,
            },
            blocks: vec![
                Block::Text {
                    bbox: BBox::new(1.0, 2.0, 3.0, 4.0),
                    role: TextRole::Title(TitleLevel(2)),
                    lines: vec![TextLine {
                        bbox: BBox::new(1.0, 2.0, 3.0, 4.0),
                        spans: vec![Span::Text {
                            bbox: BBox::new(1.0, 2.0, 3.0, 4.0),
                            text: "Hello".to_owned(),
                            score: Score(0.9),
                        }],
                    }],
                },
                Block::InterlineEquation {
                    bbox: BBox::new(5.0, 6.0, 7.0, 8.0),
                    latex: Latex("E=mc^2".to_owned()),
                },
            ],
            // Discarded blocks are the dump's reason to exist (they appear in no
            // other output), so they must survive the round trip too.
            discarded: vec![Block::Text {
                bbox: BBox::new(0.0, 0.0, 10.0, 5.0),
                role: TextRole::Header,
                lines: vec![],
            }],
        }],
    };

    let json = serde_json::to_string(&doc).expect("serialize");
    let back: Document = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(back.pages.len(), 1);
    let page = &back.pages[0];
    assert_eq!(page.index, 3);
    assert_eq!(page.size, PageSize { width: 612.0, height: 792.0 });
    assert_eq!(page.blocks.len(), 2);
    assert_eq!(page.discarded.len(), 1, "discarded blocks must survive");

    match &page.blocks[0] {
        Block::Text { role, lines, .. } => {
            assert_eq!(*role, TextRole::Title(TitleLevel(2)));
            assert_eq!(lines.len(), 1);
        }
        other => panic!("expected text block, got {other:?}"),
    }
    match &page.blocks[1] {
        Block::InterlineEquation { latex, .. } => assert_eq!(latex.as_str(), "E=mc^2"),
        other => panic!("expected equation, got {other:?}"),
    }
    match &page.discarded[0] {
        Block::Text { role, .. } => assert_eq!(*role, TextRole::Header),
        other => panic!("expected header, got {other:?}"),
    }
}
