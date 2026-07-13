//! A light `para_split.py` analogue.
//!
//! The Python `para_split` merges text lines/blocks into paragraphs across layout
//! regions. This v1 pass is intentionally minimal: it merges *adjacent* text
//! blocks of the identical [`TextRole`] whose boxes are vertically close, folding
//! their lines into the earlier block. It is a pure `Vec<Block> -> Vec<Block>`
//! transform run per page after assembly; richer cross-column paragraph logic is
//! left for a later phase.

use mineru_types::{BBox, Block, TextRole};

/// Vertical gap (page points) under which two same-role text blocks are merged.
const MERGE_GAP: f32 = 12.0;

/// Merges adjacent same-role text blocks that are vertically close.
///
/// Non-text blocks and role changes act as hard boundaries, so a title never
/// absorbs body text and an image never merges with anything. Reading order is
/// preserved.
pub fn merge_paragraphs(blocks: Vec<Block>) -> Vec<Block> {
    let mut out: Vec<Block> = Vec::with_capacity(blocks.len());
    for block in blocks {
        if let Some(prev) = out.last_mut() {
            if should_merge(prev, &block) {
                merge_into(prev, block);
                continue;
            }
        }
        out.push(block);
    }
    out
}

/// Whether `next` should fold into `prev`: both body-flow text, same role, and
/// vertically adjacent.
fn should_merge(prev: &Block, next: &Block) -> bool {
    match (prev, next) {
        (
            Block::Text { role: r0, bbox: b0, .. },
            Block::Text { role: r1, bbox: b1, .. },
        ) => is_mergeable_role(*r0) && r0 == r1 && vertical_gap(b0, b1) <= MERGE_GAP,
        _ => false,
    }
}

/// Only flowing prose merges; titles/headers/etc. stay atomic.
fn is_mergeable_role(role: TextRole) -> bool {
    matches!(
        role,
        TextRole::Body | TextRole::List | TextRole::RefText | TextRole::Abstract
    )
}

/// Signed vertical gap from the bottom of `a` to the top of `b` (negative when
/// they overlap). Absolute value is compared so blocks in either vertical order
/// merge when close.
fn vertical_gap(a: &BBox, b: &BBox) -> f32 {
    (b.y0 - a.y1).abs().min((a.y0 - b.y1).abs())
}

/// Folds `next`'s lines into `prev` and grows `prev`'s bbox to cover both.
fn merge_into(prev: &mut Block, next: Block) {
    if let (
        Block::Text { bbox: pb, lines: pl, .. },
        Block::Text { bbox: nb, lines: nl, .. },
    ) = (prev, next)
    {
        *pb = union(*pb, nb);
        pl.extend(nl);
    }
}

/// Smallest box covering both inputs.
fn union(a: BBox, b: BBox) -> BBox {
    BBox::new(
        a.x0.min(b.x0),
        a.y0.min(b.y0),
        a.x1.max(b.x1),
        a.y1.max(b.y1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mineru_types::{Score, Span, TextLine};

    fn text(bbox: BBox, role: TextRole, s: &str) -> Block {
        Block::Text {
            bbox,
            role,
            lines: vec![TextLine {
                bbox,
                spans: vec![Span::Text {
                    bbox,
                    text: s.into(),
                    score: Score(1.0),
                }],
            }],
        }
    }

    #[test]
    fn merges_adjacent_body_blocks() {
        let a = text(BBox::new(0.0, 0.0, 100.0, 20.0), TextRole::Body, "one");
        let b = text(BBox::new(0.0, 25.0, 100.0, 45.0), TextRole::Body, "two");
        let merged = merge_paragraphs(vec![a, b]);
        assert_eq!(merged.len(), 1);
        match &merged[0] {
            Block::Text { lines, bbox, .. } => {
                assert_eq!(lines.len(), 2);
                assert_eq!(bbox.y1, 45.0);
            }
            other => panic!("expected merged text, got {other:?}"),
        }
    }

    #[test]
    fn does_not_merge_across_roles() {
        let title = text(BBox::new(0.0, 0.0, 100.0, 20.0), TextRole::Title(mineru_types::TitleLevel(1)), "T");
        let body = text(BBox::new(0.0, 25.0, 100.0, 45.0), TextRole::Body, "b");
        assert_eq!(merge_paragraphs(vec![title, body]).len(), 2);
    }

    #[test]
    fn does_not_merge_when_far_apart() {
        let a = text(BBox::new(0.0, 0.0, 100.0, 20.0), TextRole::Body, "one");
        let b = text(BBox::new(0.0, 200.0, 100.0, 220.0), TextRole::Body, "two");
        assert_eq!(merge_paragraphs(vec![a, b]).len(), 2);
    }
}
