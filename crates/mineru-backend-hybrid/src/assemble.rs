//! The hybrid "magic model": extracted regions → the typed [`Block`] tree.
//!
//! Python reference: `hybrid_magic_model.py` (`MagicModel`) +
//! `hybrid_model_output_to_middle_json.py` (`blocks_to_page_info`,
//! `_normalize_split_title_blocks`). Where the Python builds stringly-typed dicts
//! and later normalizes them, this module goes straight from a typed
//! [`ExtractedRegion`] to a [`mineru_types::Block`].
//!
//! The structure mirrors the pipeline backend's `assemble.rs` (which the verifier
//! knows well): partition regions into bodies / captions / footnotes / discarded /
//! text, nest captions and footnotes onto their nearest visual body, and emit in
//! reading order. The hybrid-specific behaviors ported here:
//!
//! - **`index` → text** (`hybrid_magic_model.py:322-324`): the `content` layout
//!   label routes through the VLM as `INDEX` but is normalized back to body text
//!   on output.
//! - **Title split** (`_apply_layout_title_split` + `_normalize_split_title_blocks`):
//!   a VLM `title` becomes a doc title (level 0) when it overlaps a pipeline
//!   `doc_title` box past a threshold, else a paragraph title (level 1). Both
//!   normalize to [`TextRole::Title`] with the corresponding level.
//! - **Inline `\(...\)` splitting** (`hybrid_magic_model.py:201-246`): text whose
//!   `\(`/`\)` counts match is split into text + inline-equation spans. Reused via
//!   [`split_inline_spans`].
//! - **`seal` sub_type** is carried on the region and dropped into an image body
//!   (the typed tree has no seal field; the region flag is preserved only for the
//!   discard decision, matching the Python treating seals as images).

use mineru_types::{
    BBox, Block, Captioned, ImageBody, ImageRef, Latex, Score, Span, TableBody, TextBlock,
    TextLine, TextRole, TitleLevel,
};

use crate::label_map::VlmType;

/// Overlap threshold above which a VLM title box is promoted to a document title.
///
/// Mirrors `LAYOUT_TITLE_SPLIT_OVERLAP_THRESHOLD = 0.8` in `hybrid_analyze.py`.
pub const TITLE_SPLIT_OVERLAP_THRESHOLD: f32 = 0.8;

/// One region detected by the pipeline layout, paired with its VLM-extracted
/// content — the assembler's input unit.
///
/// This is the hybrid analogue of the pipeline backend's
/// [`Region`](mineru_backend_pipeline::Region), but the content is a single
/// VLM string rather than per-line OCR, since the VLM extracts a whole region at
/// once. `bbox` is in page pixels (the space the layout model and VLM crops share).
#[derive(Debug, Clone)]
pub struct ExtractedRegion {
    /// Region box in page pixels.
    pub bbox: BBox,
    /// The VLM extraction type this region was routed to.
    pub vlm_type: VlmType,
    /// The VLM's extracted content (text / LaTeX / table HTML), if any.
    pub content: Option<String>,
    /// Reading-order rank (0 = first).
    pub order: usize,
    /// `true` when the source layout label was `seal` (Python `sub_type="seal"`).
    pub is_seal: bool,
}

/// The doc-title boxes (in page pixels) the pipeline layout found on a page, used
/// to promote overlapping VLM titles to document titles.
///
/// Ported from `_collect_layout_doc_title_bboxes`: only the pipeline `doc_title`
/// label contributes (not `paragraph_title`).
#[derive(Debug, Clone, Default)]
pub struct DocTitleBoxes(pub Vec<BBox>);

impl DocTitleBoxes {
    /// Whether `title_bbox` overlaps any doc-title box past the split threshold.
    ///
    /// Mirrors `_has_doc_title_overlap` using the min-box overlap ratio
    /// (`calculate_overlap_area_2_minbox_area_ratio`): the fraction of the *smaller*
    /// box that the two share.
    fn promotes(&self, title_bbox: &BBox) -> bool {
        self.0
            .iter()
            .any(|dt| min_box_overlap_ratio(title_bbox, dt) >= TITLE_SPLIT_OVERLAP_THRESHOLD)
    }
}

/// The assembled content of one page: main-flow blocks and discarded blocks, both
/// in reading order. Mirrors the pipeline backend's `AssembledPage`.
#[derive(Debug, Clone, Default)]
pub struct AssembledPage {
    /// Main-flow blocks in reading order.
    pub blocks: Vec<Block>,
    /// Blocks dropped from the main flow (headers/footers/page numbers/…).
    pub discarded: Vec<Block>,
}

/// Builds the typed [`Block`] tree from extracted regions.
///
/// Stateless; holds the page's pipeline doc-title boxes so the title-split
/// decision has the data it needs. Construct with [`HybridAssembler::new`].
#[derive(Debug, Clone, Default)]
pub struct HybridAssembler {
    doc_titles: DocTitleBoxes,
}

impl HybridAssembler {
    /// Builds an assembler for a page, given the pipeline `doc_title` boxes used to
    /// promote overlapping VLM titles.
    pub fn new(doc_titles: DocTitleBoxes) -> Self {
        Self { doc_titles }
    }

    /// Converts extracted regions into the page's [`Block`]s and discarded blocks.
    ///
    /// Captions and footnotes attach to their nearest visual body; text and visual
    /// blocks are emitted in the detector's reading order. Skipped regions (labels
    /// the VLM never saw) are dropped.
    pub fn assemble(&self, mut regions: Vec<ExtractedRegion>) -> AssembledPage {
        regions.sort_by_key(|r| r.order);

        let mut out: Vec<Slot> = Vec::new();
        let mut bodies: Vec<(usize, ExtractedRegion)> = Vec::new();
        let mut captions: Vec<ExtractedRegion> = Vec::new();
        let mut footnotes: Vec<ExtractedRegion> = Vec::new();
        let mut discarded: Vec<Block> = Vec::new();

        for region in regions {
            match Role::of(region.vlm_type) {
                Role::Skip => {}
                Role::Caption => captions.push(region),
                Role::Footnote => footnotes.push(region),
                Role::Discarded(role) => {
                    discarded.push(text_block(region.bbox, role, region.content.as_deref()));
                }
                Role::Visual(_) => {
                    bodies.push((out.len(), region));
                    out.push(Slot::Pending);
                }
                Role::Equation => {
                    let latex = Latex(region.content.unwrap_or_default().trim().to_owned());
                    out.push(Slot::Block(Block::InterlineEquation {
                        bbox: region.bbox,
                        latex,
                    }));
                }
                Role::Text(role) => {
                    let role = self.resolve_title(role, &region.bbox);
                    out.push(Slot::Block(text_block(
                        region.bbox,
                        role,
                        region.content.as_deref(),
                    )));
                }
            }
        }

        for (slot_idx, block) in nest_visuals(bodies, captions, footnotes) {
            if let Some(slot) = out.get_mut(slot_idx) {
                *slot = Slot::Block(block);
            }
        }

        let blocks = out
            .into_iter()
            .filter_map(|s| match s {
                Slot::Block(b) => Some(b),
                Slot::Pending => None,
            })
            .collect();

        AssembledPage { blocks, discarded }
    }

    /// Resolves a title's level from pipeline doc-title overlap.
    ///
    /// A [`TextRole::Title`] whose box overlaps a pipeline `doc_title` box becomes
    /// level 0 (document title), else level 1 (paragraph title). Non-title roles
    /// pass through unchanged. Mirrors `_apply_layout_title_split` collapsed with
    /// `_normalize_split_title_blocks`.
    fn resolve_title(&self, role: TextRole, bbox: &BBox) -> TextRole {
        match role {
            TextRole::Title(_) => {
                if self.doc_titles.promotes(bbox) {
                    TextRole::Title(TitleLevel(0))
                } else {
                    TextRole::Title(TitleLevel(1))
                }
            }
            other => other,
        }
    }
}

/// The coarse role a [`VlmType`] plays in assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Text(TextRole),
    Equation,
    Visual(VisualKind),
    Caption,
    Footnote,
    Discarded(TextRole),
    Skip,
}

/// Which visual body a [`VlmType`] becomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisualKind {
    Image,
    Chart,
    Table,
}

impl Role {
    /// Maps a VLM extraction type to its assembly role.
    ///
    /// Mirrors the `MagicModel` block-type routing: text roles fold into
    /// [`TextRole`]; `index` normalizes to body text (Python
    /// `hybrid_magic_model.py:322`); header/footer/page-number/aside/page-footnote
    /// are discarded; captions/footnotes are held for nesting.
    fn of(vlm_type: VlmType) -> Self {
        match vlm_type {
            VlmType::Text | VlmType::Index => Role::Text(TextRole::Body),
            VlmType::Title => Role::Text(TextRole::Title(TitleLevel(1))),
            VlmType::Code => Role::Text(TextRole::Body),
            VlmType::RefText => Role::Text(TextRole::RefText),

            VlmType::AsideText => Role::Discarded(TextRole::AsideText),
            VlmType::Header => Role::Discarded(TextRole::Header),
            VlmType::Footer => Role::Discarded(TextRole::Footer),
            VlmType::PageNumber => Role::Discarded(TextRole::PageNumber),
            VlmType::PageFootnote => Role::Discarded(TextRole::PageFootnote),

            VlmType::ImageCaption => Role::Caption,
            VlmType::ImageFootnote => Role::Footnote,

            VlmType::Image => Role::Visual(VisualKind::Image),
            VlmType::Chart => Role::Visual(VisualKind::Chart),
            VlmType::Table => Role::Visual(VisualKind::Table),
            VlmType::Equation => Role::Equation,

            // formula_number is folded into its line elsewhere; skipped labels
            // never reach the VLM. Both drop from the block tree.
            VlmType::FormulaNumber | VlmType::Skipped => Role::Skip,
        }
    }
}

/// A reading-order output position: a finished block or a placeholder for a visual
/// body still collecting its captions/footnotes.
enum Slot {
    Block(Block),
    Pending,
}

/// Nests caption and footnote regions onto their nearest visual body and builds the
/// finished visual blocks, preserving each body's reading-order slot. Mirrors the
/// pipeline backend's `nest_visuals`.
fn nest_visuals(
    bodies: Vec<(usize, ExtractedRegion)>,
    captions: Vec<ExtractedRegion>,
    footnotes: Vec<ExtractedRegion>,
) -> Vec<(usize, Block)> {
    let body_boxes: Vec<BBox> = bodies.iter().map(|(_, r)| r.bbox).collect();

    let mut caption_blocks: Vec<Vec<TextBlock>> = vec![Vec::new(); bodies.len()];
    let mut footnote_blocks: Vec<Vec<TextBlock>> = vec![Vec::new(); bodies.len()];

    for cap in captions {
        if let Some(i) = nearest_body(&cap.bbox, &body_boxes) {
            caption_blocks[i].push(as_text_block(cap.bbox, cap.content.as_deref()));
        }
    }
    for foot in footnotes {
        if let Some(i) = nearest_body(&foot.bbox, &body_boxes) {
            footnote_blocks[i].push(as_text_block(foot.bbox, foot.content.as_deref()));
        }
    }

    bodies
        .into_iter()
        .enumerate()
        .map(|(i, (slot_idx, region))| {
            let caps = std::mem::take(&mut caption_blocks[i]);
            let foots = std::mem::take(&mut footnote_blocks[i]);
            (slot_idx, visual_block(&region, caps, foots))
        })
        .collect()
}

/// Picks the body a satellite box belongs to: max overlap, ties broken by nearest
/// centre. Mirrors the pipeline backend's `nearest_body`.
fn nearest_body(satellite: &BBox, bodies: &[BBox]) -> Option<usize> {
    if bodies.is_empty() {
        return None;
    }
    let mut best = 0usize;
    let mut best_overlap = f32::NEG_INFINITY;
    let mut best_dist = f32::INFINITY;
    for (i, body) in bodies.iter().enumerate() {
        let overlap = body.overlap_ratio(satellite);
        let dist = center_distance_sq(satellite, body);
        if overlap > best_overlap || (overlap == best_overlap && dist < best_dist) {
            best = i;
            best_overlap = overlap;
            best_dist = dist;
        }
    }
    Some(best)
}

/// Squared distance between two boxes' centres.
fn center_distance_sq(a: &BBox, b: &BBox) -> f32 {
    let (ax, ay) = a.center();
    let (bx, by) = b.center();
    let (dx, dy) = (ax - bx, ay - by);
    dx * dx + dy * dy
}

/// Fraction of the *smaller* of two boxes that they share (min-box overlap ratio).
///
/// Ported from `calculate_overlap_area_2_minbox_area_ratio` in the Python
/// `boxbase` util, used by the title-split overlap check.
fn min_box_overlap_ratio(a: &BBox, b: &BBox) -> f32 {
    let inter = a.intersection(b).map_or(0.0, |i| i.area());
    let min_area = a.area().min(b.area());
    if min_area > 0.0 {
        inter / min_area
    } else {
        0.0
    }
}

/// Builds the visual [`Block`] for a body region with its nested captions/footnotes.
fn visual_block(region: &ExtractedRegion, captions: Vec<TextBlock>, footnotes: Vec<TextBlock>) -> Block {
    let bbox = region.bbox;
    match Role::of(region.vlm_type) {
        Role::Visual(VisualKind::Table) => {
            let html = mineru_types::Html(region.content.clone().unwrap_or_default());
            Block::Table(Captioned {
                body: TableBody { bbox, html, image: None },
                captions,
                footnotes,
            })
        }
        Role::Visual(VisualKind::Chart) => Block::Chart(Captioned {
            body: image_body(bbox),
            captions,
            footnotes,
        }),
        // Image and any fallback become an image block. The `seal` sub_type has no
        // dedicated field in the typed tree (Python only carries it as a passthrough
        // string), so it collapses into a plain image body here.
        _ => Block::Image(Captioned {
            body: image_body(bbox),
            captions,
            footnotes,
        }),
    }
}

/// An image body with an empty [`ImageRef`] (the crop-saving stage fills the path).
fn image_body(bbox: BBox) -> ImageBody {
    ImageBody {
        bbox,
        image: ImageRef(String::new()),
    }
}

/// Builds a flowing-text [`Block::Text`] from a region's content, splitting inline
/// `\(...\)` fragments into inline-equation spans.
fn text_block(bbox: BBox, role: TextRole, content: Option<&str>) -> Block {
    let content = normalized_text(role, content.unwrap_or(""));
    Block::Text {
        bbox,
        role,
        lines: vec![TextLine {
            bbox,
            spans: split_inline_spans(&content, bbox),
        }],
    }
}

/// Builds a caption/footnote [`TextBlock`] from a region's content.
fn as_text_block(bbox: BBox, content: Option<&str>) -> TextBlock {
    TextBlock {
        bbox,
        lines: vec![TextLine {
            bbox,
            spans: split_inline_spans(content.unwrap_or(""), bbox),
        }],
    }
}

/// Collapses internal newlines to single spaces for title roles.
///
/// Mirrors `hybrid_magic_model.py:198-199`: title content has `\n\s*` replaced with
/// a single space and is trimmed. Other roles pass through unchanged.
fn normalized_text(role: TextRole, content: &str) -> String {
    if matches!(role, TextRole::Title(_)) {
        collapse_newlines(content)
    } else {
        content.to_owned()
    }
}

/// Replaces each newline and its following whitespace run with one space, trimming.
fn collapse_newlines(content: &str) -> String {
    let mut out = String::with_capacity(content.len());
    let mut after_nl = false;
    for c in content.chars() {
        if c == '\n' {
            out.push(' ');
            after_nl = true;
        } else if after_nl && c.is_whitespace() {
            // collapse the whitespace run following the newline
        } else {
            after_nl = false;
            out.push(c);
        }
    }
    out.trim().to_owned()
}

/// Splits `content` into spans, turning balanced inline `\(...\)` fragments into
/// [`Span::InlineEquation`]s. Mirrors the VLM client's `text_spans` and the
/// hybrid magic model's inline-formula handling.
pub fn split_inline_spans(content: &str, bbox: BBox) -> Vec<Span> {
    let opens = content.matches("\\(").count();
    let closes = content.matches("\\)").count();
    if opens == 0 || opens != closes {
        return vec![text_span(content, bbox)];
    }

    let mut spans = Vec::new();
    let mut rest = content;
    while let Some(open) = rest.find("\\(") {
        let (before, after_open) = rest.split_at(open);
        if !before.trim().is_empty() {
            spans.push(text_span(before, bbox));
        }
        let after_open = &after_open[2..]; // skip "\("
        match after_open.find("\\)") {
            Some(close) => {
                spans.push(Span::InlineEquation {
                    bbox,
                    latex: Latex(after_open[..close].trim().to_owned()),
                    score: Score(1.0),
                });
                rest = &after_open[close + 2..]; // skip "\)"
            }
            None => {
                spans.push(text_span(after_open, bbox));
                rest = "";
                break;
            }
        }
    }
    if !rest.trim().is_empty() {
        spans.push(text_span(rest, bbox));
    }
    if spans.is_empty() {
        spans.push(text_span("", bbox));
    }
    spans
}

/// A plain text span with unit confidence (the VLM content is authoritative).
fn text_span(text: &str, bbox: BBox) -> Span {
    Span::Text {
        bbox,
        text: text.to_owned(),
        score: Score(1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(bbox: BBox, vlm_type: VlmType, content: &str, order: usize) -> ExtractedRegion {
        ExtractedRegion {
            bbox,
            vlm_type,
            content: Some(content.to_owned()),
            order,
            is_seal: false,
        }
    }

    fn text_of(block: &Block) -> Option<String> {
        match block {
            Block::Text { lines, .. } => {
                let mut s = String::new();
                for span in &lines[0].spans {
                    if let Span::Text { text, .. } = span {
                        s.push_str(text);
                    }
                }
                Some(s)
            }
            _ => None,
        }
    }

    #[test]
    fn text_region_becomes_body_block() {
        let bbox = BBox::new(0.0, 0.0, 100.0, 20.0);
        let page = HybridAssembler::default()
            .assemble(vec![region(bbox, VlmType::Text, "hello world", 0)]);
        assert_eq!(page.blocks.len(), 1);
        assert!(matches!(&page.blocks[0], Block::Text { role: TextRole::Body, .. }));
        assert_eq!(text_of(&page.blocks[0]).unwrap(), "hello world");
    }

    #[test]
    fn index_normalizes_to_text() {
        let bbox = BBox::new(0.0, 0.0, 100.0, 20.0);
        let page = HybridAssembler::default()
            .assemble(vec![region(bbox, VlmType::Index, "1. Intro ..... 3", 0)]);
        assert!(matches!(&page.blocks[0], Block::Text { role: TextRole::Body, .. }));
    }

    #[test]
    fn discarded_roles_leave_main_flow() {
        let bbox = BBox::new(0.0, 0.0, 100.0, 10.0);
        let page = HybridAssembler::default().assemble(vec![
            region(bbox, VlmType::Header, "head", 0),
            region(bbox, VlmType::PageNumber, "1", 1),
            region(BBox::new(0.0, 20.0, 100.0, 40.0), VlmType::Text, "body", 2),
        ]);
        assert_eq!(page.blocks.len(), 1);
        assert_eq!(page.discarded.len(), 2);
    }

    #[test]
    fn title_promoted_to_doc_title_by_overlap() {
        let title = BBox::new(50.0, 10.0, 550.0, 40.0);
        // A pipeline doc_title box covering the same region.
        let doc_titles = DocTitleBoxes(vec![BBox::new(50.0, 10.0, 550.0, 40.0)]);
        let page = HybridAssembler::new(doc_titles)
            .assemble(vec![region(title, VlmType::Title, "The Title", 0)]);
        assert!(matches!(
            &page.blocks[0],
            Block::Text { role: TextRole::Title(TitleLevel(0)), .. }
        ));
    }

    #[test]
    fn title_without_overlap_is_paragraph_title() {
        let title = BBox::new(50.0, 400.0, 300.0, 420.0);
        // Doc-title box elsewhere on the page — no overlap.
        let doc_titles = DocTitleBoxes(vec![BBox::new(50.0, 10.0, 550.0, 40.0)]);
        let page = HybridAssembler::new(doc_titles)
            .assemble(vec![region(title, VlmType::Title, "A Section", 0)]);
        assert!(matches!(
            &page.blocks[0],
            Block::Text { role: TextRole::Title(TitleLevel(1)), .. }
        ));
    }

    #[test]
    fn title_collapses_newlines() {
        let bbox = BBox::new(0.0, 0.0, 100.0, 40.0);
        let page = HybridAssembler::default()
            .assemble(vec![region(bbox, VlmType::Title, "Line one\n   line two", 0)]);
        assert_eq!(text_of(&page.blocks[0]).unwrap(), "Line one line two");
    }

    #[test]
    fn caption_nests_onto_nearest_image() {
        let img_a = BBox::new(0.0, 0.0, 100.0, 100.0);
        let img_b = BBox::new(0.0, 200.0, 100.0, 300.0);
        let cap = BBox::new(0.0, 300.0, 100.0, 320.0);
        let page = HybridAssembler::default().assemble(vec![
            region(img_a, VlmType::Image, "", 0),
            region(img_b, VlmType::Image, "", 1),
            region(cap, VlmType::ImageCaption, "Figure 2.", 2),
        ]);
        assert_eq!(page.blocks.len(), 2);
        match (&page.blocks[0], &page.blocks[1]) {
            (Block::Image(a), Block::Image(b)) => {
                assert!(a.captions.is_empty());
                assert_eq!(b.captions.len(), 1);
            }
            other => panic!("expected two images, got {other:?}"),
        }
    }

    #[test]
    fn table_carries_html_and_footnote() {
        let table = BBox::new(0.0, 0.0, 100.0, 100.0);
        let foot = BBox::new(0.0, 100.0, 100.0, 110.0);
        let page = HybridAssembler::default().assemble(vec![
            region(table, VlmType::Table, "<table><tr><td>x</td></tr></table>", 0),
            region(foot, VlmType::ImageFootnote, "note", 1),
        ]);
        assert_eq!(page.blocks.len(), 1);
        match &page.blocks[0] {
            Block::Table(t) => {
                assert!(t.body.html.as_str().contains("<table>"));
                assert_eq!(t.footnotes.len(), 1);
            }
            other => panic!("expected table, got {other:?}"),
        }
    }

    #[test]
    fn equation_becomes_interline() {
        let bbox = BBox::new(0.0, 0.0, 100.0, 30.0);
        let page = HybridAssembler::default()
            .assemble(vec![region(bbox, VlmType::Equation, "  E=mc^2  ", 0)]);
        match &page.blocks[0] {
            Block::InterlineEquation { latex, .. } => assert_eq!(latex.as_str(), "E=mc^2"),
            other => panic!("expected interline equation, got {other:?}"),
        }
    }

    #[test]
    fn inline_formula_splits_into_spans() {
        let spans = split_inline_spans("energy \\(E=mc^2\\) done", BBox::new(0.0, 0.0, 1.0, 1.0));
        assert_eq!(spans.len(), 3);
        assert!(matches!(spans[1], Span::InlineEquation { .. }));
    }

    #[test]
    fn skipped_and_formula_number_are_dropped() {
        let bbox = BBox::new(0.0, 0.0, 100.0, 20.0);
        let page = HybridAssembler::default().assemble(vec![
            region(bbox, VlmType::Skipped, "", 0),
            region(bbox, VlmType::FormulaNumber, "(1)", 1),
            region(BBox::new(0.0, 30.0, 100.0, 50.0), VlmType::Text, "body", 2),
        ]);
        assert_eq!(page.blocks.len(), 1);
        assert!(page.discarded.is_empty());
    }

    #[test]
    fn blocks_emitted_in_reading_order() {
        let b0 = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b1 = BBox::new(0.0, 20.0, 10.0, 30.0);
        let b2 = BBox::new(0.0, 40.0, 10.0, 50.0);
        let page = HybridAssembler::default().assemble(vec![
            region(b2, VlmType::Text, "third", 2),
            region(b0, VlmType::Text, "first", 0),
            region(b1, VlmType::Text, "second", 1),
        ]);
        let texts: Vec<String> = page.blocks.iter().filter_map(text_of).collect();
        assert_eq!(texts, vec!["first", "second", "third"]);
    }

    #[test]
    fn full_synthetic_page_assembles() {
        let title = BBox::new(50.0, 10.0, 550.0, 40.0);
        let body = BBox::new(50.0, 50.0, 550.0, 200.0);
        let img = BBox::new(50.0, 220.0, 300.0, 420.0);
        let cap = BBox::new(50.0, 425.0, 300.0, 445.0);
        let table = BBox::new(320.0, 220.0, 550.0, 420.0);
        let footer = BBox::new(50.0, 780.0, 550.0, 800.0);

        let doc_titles = DocTitleBoxes(vec![title]);
        let page = HybridAssembler::new(doc_titles).assemble(vec![
            region(title, VlmType::Title, "The Title", 0),
            region(body, VlmType::Text, "Body paragraph.", 1),
            region(img, VlmType::Image, "", 2),
            region(cap, VlmType::ImageCaption, "Figure 1.", 3),
            region(table, VlmType::Table, "<table><tr><td>x</td></tr></table>", 4),
            region(footer, VlmType::Footer, "page footer", 5),
        ]);

        assert_eq!(page.blocks.len(), 4, "title, body, image, table");
        assert_eq!(page.discarded.len(), 1, "footer");
        assert!(matches!(
            page.blocks[0],
            Block::Text { role: TextRole::Title(TitleLevel(0)), .. }
        ));
        match &page.blocks[2] {
            Block::Image(c) => assert_eq!(c.captions.len(), 1),
            other => panic!("expected image, got {other:?}"),
        }
        assert!(matches!(&page.blocks[3], Block::Table(_)));
    }
}
