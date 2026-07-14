//! The [`PageAssembler`]: raw `Vec<LayoutDet>` (+ recognized content) → the typed
//! [`Block`] tree. This is the `pipeline_magic_model.py` analogue.
//!
//! The model crates emit *raw* outputs ([`LayoutDet`], recognized strings, HTML,
//! LaTeX). This module is where those become [`mineru_types::Block`]s. It is kept
//! deliberately model-free and side-effect-free — a pile of small typed converter
//! functions rather than a god-class — so the label mapping and caption nesting
//! are unit-testable on synthetic detections with no weights.
//!
//! # Pipeline
//! 1. [`RegionKind::classify`] maps every [`LayoutLabel`] to a coarse role.
//! 2. Regions are split into *bodies* (visual: image/table/chart/formula),
//!    *captions*, *footnotes*, *discarded* (headers/footers/page-numbers), and
//!    plain *text*.
//! 3. Captions and footnotes are nested onto their nearest body via
//!    [`nest_visuals`] (spatial overlap / centre distance).
//! 4. Everything is emitted in the detector's reading order.

use mineru_layout::{LayoutDet, LayoutLabel};
use mineru_types::{
    BBox, Block, Captioned, Html, ImageBody, ImageRef, Latex, Score, Span, TableBody, TextBlock,
    TextLine, TextRole, TitleLevel,
};

/// The recognized content attached to one detected region.
///
/// The orchestrator ([`crate::analyze`]) runs the relevant model per region and
/// fills the matching field; the assembler reads whichever field the region's kind
/// calls for. Bundling content this way keeps the assembler a pure function of
/// `(det, content)` with no model dependency.
#[derive(Debug, Clone, Default)]
pub struct RegionContent {
    /// OCR text lines recognized in a text-like region (one entry per line box).
    pub text_lines: Vec<RecognizedLine>,
    /// LaTeX for a display-formula region.
    pub latex: Option<Latex>,
    /// HTML for a table region.
    pub table_html: Option<Html>,
    /// An extracted raster reference for an image/chart region.
    pub image: Option<ImageRef>,
}

/// One recognized OCR line: its box (page points) plus text and confidence.
#[derive(Debug, Clone)]
pub struct RecognizedLine {
    /// Line bounding box in page points.
    pub bbox: BBox,
    /// Recognized text.
    pub text: String,
    /// Recognition confidence in `0.0..=1.0`.
    pub score: f32,
}

/// A detection paired with its recognized content, the assembler's input unit.
#[derive(Debug, Clone)]
pub struct Region {
    /// The raw layout detection (bbox in page points, label, reading order).
    pub det: LayoutDet,
    /// The recognized content for this region.
    pub content: RegionContent,
}

/// The coarse role a [`LayoutLabel`] plays in assembly.
///
/// Collapses the 25 detector classes into the handful of shapes the [`Block`] tree
/// distinguishes. This is the single source of truth for "what does this label
/// become", so the mapping is testable in isolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind {
    /// A flowing-text block with the given role.
    Text(TextRole),
    /// A standalone display equation.
    Equation,
    /// An image body.
    Image,
    /// A chart body.
    Chart,
    /// A table body.
    Table,
    /// A caption to be nested onto the nearest visual body.
    Caption,
    /// A footnote to be nested onto the nearest visual body.
    Footnote,
    /// A region dropped from the main flow (header/footer/page-number/seal/…).
    Discarded(TextRole),
    /// A region the pipeline does not emit (e.g. inline formula / formula number,
    /// which are folded into their parent line elsewhere).
    Ignored,
}

impl RegionKind {
    /// Maps a detector label to its assembly role.
    ///
    /// Mirrors the Python `magic_model` block-type routing, adapted to the typed
    /// [`Block`]/[`TextRole`] model:
    /// - body text (`Content`/`Text`/`Algorithm`/`VerticalText`) → [`TextRole::Body`]
    /// - `DocTitle` → [`TextRole::Title(0)`], `ParagraphTitle` → `Title(1)`
    /// - `Abstract` → [`TextRole::Abstract`], reference items → [`TextRole::RefText`]
    /// - `AsideText` → [`TextRole::AsideText`], `Footnote` → [`TextRole::PageFootnote`]
    /// - `Image`/`Chart`/`Table`/`DisplayFormula` → their visual bodies
    /// - `FigureTitle` → [`RegionKind::Caption`], `VisionFootnote` → [`RegionKind::Footnote`]
    /// - `Header`/`Footer`/`Number`/`Seal`/images-in-margins → [`RegionKind::Discarded`]
    pub fn classify(label: LayoutLabel) -> Self {
        use LayoutLabel as L;
        match label {
            L::Content | L::Text | L::Algorithm | L::VerticalText => {
                RegionKind::Text(TextRole::Body)
            }
            L::DocTitle => RegionKind::Text(TextRole::Title(TitleLevel(0))),
            L::ParagraphTitle => RegionKind::Text(TextRole::Title(TitleLevel(1))),
            L::Abstract => RegionKind::Text(TextRole::Abstract),
            L::Reference | L::ReferenceContent => RegionKind::Text(TextRole::RefText),
            L::AsideText => RegionKind::Text(TextRole::AsideText),
            L::Footnote => RegionKind::Text(TextRole::PageFootnote),

            L::Image | L::HeaderImage | L::FooterImage => RegionKind::Image,
            L::Chart => RegionKind::Chart,
            L::Table => RegionKind::Table,
            L::DisplayFormula => RegionKind::Equation,

            L::FigureTitle => RegionKind::Caption,
            L::VisionFootnote => RegionKind::Footnote,

            L::Header => RegionKind::Discarded(TextRole::Header),
            L::Footer => RegionKind::Discarded(TextRole::Footer),
            L::Number => RegionKind::Discarded(TextRole::PageNumber),
            L::Seal => RegionKind::Discarded(TextRole::Body),

            // Inline formula / formula number are folded into their parent line
            // during span extraction, not emitted as standalone blocks.
            L::InlineFormula | L::FormulaNumber => RegionKind::Ignored,
        }
    }

    /// Whether this kind is a visual body that captions/footnotes nest onto.
    fn is_visual_body(self) -> bool {
        matches!(self, RegionKind::Image | RegionKind::Chart | RegionKind::Table)
    }
}

/// Builds the typed [`Block`] tree from raw regions.
///
/// Stateless: [`PageAssembler::assemble`] is a pure function of its input regions.
/// Held as a unit struct so callers read as `PageAssembler::default().assemble(..)`
/// and so future tuning knobs (overlap thresholds) have a home without changing
/// call sites.
#[derive(Debug, Clone, Copy, Default)]
pub struct PageAssembler;

/// The assembled content of one page: main-flow blocks and discarded blocks, both
/// in reading order.
#[derive(Debug, Clone, Default)]
pub struct AssembledPage {
    /// Main-flow blocks in reading order.
    pub blocks: Vec<Block>,
    /// Blocks dropped from the main flow (headers/footers/page numbers).
    pub discarded: Vec<Block>,
}

impl PageAssembler {
    /// Converts regions into the page's [`Block`]s and discarded blocks.
    ///
    /// Captions and footnotes are attached to their nearest visual body; text and
    /// visual blocks are emitted in the detector's reading order.
    pub fn assemble(&self, mut regions: Vec<Region>) -> AssembledPage {
        // Reading order is the detector's `order`; sort once so output is stable.
        regions.sort_by_key(|r| r.det.order);

        // Partition by role. Captions/footnotes are held back for nesting.
        let mut bodies: Vec<(usize, Region)> = Vec::new();
        let mut captions: Vec<Region> = Vec::new();
        let mut footnotes: Vec<Region> = Vec::new();
        let mut out: Vec<Slot> = Vec::new();
        let mut discarded: Vec<Block> = Vec::new();

        for region in regions {
            match RegionKind::classify(region.det.label) {
                RegionKind::Caption => captions.push(region),
                RegionKind::Footnote => footnotes.push(region),
                RegionKind::Ignored => {}
                RegionKind::Discarded(role) => {
                    discarded.push(text_block(region.det.bbox, role, &region.content));
                }
                kind if kind.is_visual_body() => {
                    bodies.push((out.len(), region));
                    out.push(Slot::Pending);
                }
                RegionKind::Text(role) => {
                    out.push(Slot::Block(text_block(
                        region.det.bbox,
                        role,
                        &region.content,
                    )));
                }
                RegionKind::Equation => {
                    let latex = region.content.latex.clone().unwrap_or_else(|| Latex(String::new()));
                    out.push(Slot::Block(Block::InterlineEquation {
                        bbox: region.det.bbox,
                        latex,
                    }));
                }
                // `is_visual_body`/`Text`/`Equation` above are exhaustive for the
                // remaining kinds; nothing else reaches here.
                _ => {}
            }
        }

        let visuals = nest_visuals(bodies, captions, footnotes);
        for (slot_idx, block) in visuals {
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
}

/// A reading-order output position: either a finished block or a placeholder for a
/// visual body whose captions/footnotes are still being nested.
enum Slot {
    Block(Block),
    Pending,
}

/// Nests caption and footnote regions onto their nearest visual body and builds the
/// finished visual blocks, preserving each body's original reading-order slot.
///
/// A caption/footnote attaches to the body it most overlaps; with no overlap it
/// falls to the body whose centre is nearest. Bodies with no candidate keep empty
/// caption/footnote lists.
fn nest_visuals(
    bodies: Vec<(usize, Region)>,
    captions: Vec<Region>,
    footnotes: Vec<Region>,
) -> Vec<(usize, Block)> {
    let body_boxes: Vec<BBox> = bodies.iter().map(|(_, r)| r.det.bbox).collect();

    let mut caption_blocks: Vec<Vec<TextBlock>> = vec![Vec::new(); bodies.len()];
    let mut footnote_blocks: Vec<Vec<TextBlock>> = vec![Vec::new(); bodies.len()];

    for cap in captions {
        if let Some(i) = nearest_body(&cap.det.bbox, &body_boxes) {
            caption_blocks[i].push(as_text_block(cap.det.bbox, &cap.content));
        }
    }
    for foot in footnotes {
        if let Some(i) = nearest_body(&foot.det.bbox, &body_boxes) {
            footnote_blocks[i].push(as_text_block(foot.det.bbox, &foot.content));
        }
    }

    bodies
        .into_iter()
        .enumerate()
        .map(|(i, (slot_idx, region))| {
            let caps = std::mem::take(&mut caption_blocks[i]);
            let foots = std::mem::take(&mut footnote_blocks[i]);
            let block = visual_block(&region, caps, foots);
            (slot_idx, block)
        })
        .collect()
}

/// Picks the body index a satellite box belongs to: max overlap ratio, breaking
/// ties (and the no-overlap case) by nearest centre. `None` when there are no
/// bodies.
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
        let better = overlap > best_overlap || (overlap == best_overlap && dist < best_dist);
        if better {
            best = i;
            best_overlap = overlap;
            best_dist = dist;
        }
    }
    Some(best)
}

/// Squared Euclidean distance between two boxes' centres.
fn center_distance_sq(a: &BBox, b: &BBox) -> f32 {
    let (ax, ay) = a.center();
    let (bx, by) = b.center();
    let (dx, dy) = (ax - bx, ay - by);
    dx * dx + dy * dy
}

/// Builds the visual [`Block`] for a body region, wrapping it with its nested
/// captions and footnotes.
fn visual_block(region: &Region, captions: Vec<TextBlock>, footnotes: Vec<TextBlock>) -> Block {
    let bbox = region.det.bbox;
    match RegionKind::classify(region.det.label) {
        RegionKind::Table => {
            let html = region.content.table_html.clone().unwrap_or_else(|| Html(String::new()));
            Block::Table(Captioned {
                body: TableBody { bbox, html, image: region.content.image.clone() },
                captions,
                footnotes,
            })
        }
        RegionKind::Chart => Block::Chart(Captioned {
            body: image_body(bbox, &region.content),
            captions,
            footnotes,
        }),
        // Image (and any body-classified fallback) becomes an image block.
        _ => Block::Image(Captioned {
            body: image_body(bbox, &region.content),
            captions,
            footnotes,
        }),
    }
}

/// The image body for a region, defaulting to an empty [`ImageRef`] when the
/// orchestrator did not save a crop.
fn image_body(bbox: BBox, content: &RegionContent) -> ImageBody {
    ImageBody {
        bbox,
        image: content.image.clone().unwrap_or_else(|| ImageRef(String::new())),
    }
}

/// Builds a flowing-text [`Block::Text`] from a region's recognized lines.
fn text_block(bbox: BBox, role: TextRole, content: &RegionContent) -> Block {
    Block::Text {
        bbox,
        role,
        lines: text_lines(content),
    }
}

/// Builds a [`TextBlock`] (caption/footnote payload) from recognized lines.
fn as_text_block(bbox: BBox, content: &RegionContent) -> TextBlock {
    TextBlock {
        bbox,
        lines: text_lines(content),
    }
}

/// Converts recognized OCR lines into typed [`TextLine`]s, one text span each.
fn text_lines(content: &RegionContent) -> Vec<TextLine> {
    content
        .text_lines
        .iter()
        .map(|line| TextLine {
            bbox: line.bbox,
            spans: vec![Span::Text {
                bbox: line.bbox,
                text: line.text.clone(),
                score: Score(line.score),
            }],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(bbox: BBox, label: LayoutLabel, order: usize) -> LayoutDet {
        LayoutDet::new(bbox, label, 0.9, order)
    }

    fn region(bbox: BBox, label: LayoutLabel, order: usize, content: RegionContent) -> Region {
        Region {
            det: det(bbox, label, order),
            content,
        }
    }

    fn lines(text: &str, bbox: BBox) -> RegionContent {
        RegionContent {
            text_lines: vec![RecognizedLine {
                bbox,
                text: text.to_string(),
                score: 0.99,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn label_maps_to_expected_roles() {
        use LayoutLabel as L;
        assert_eq!(
            RegionKind::classify(L::Text),
            RegionKind::Text(TextRole::Body)
        );
        assert_eq!(
            RegionKind::classify(L::Content),
            RegionKind::Text(TextRole::Body)
        );
        assert_eq!(
            RegionKind::classify(L::DocTitle),
            RegionKind::Text(TextRole::Title(TitleLevel(0)))
        );
        assert_eq!(
            RegionKind::classify(L::ParagraphTitle),
            RegionKind::Text(TextRole::Title(TitleLevel(1)))
        );
        assert_eq!(RegionKind::classify(L::Table), RegionKind::Table);
        assert_eq!(RegionKind::classify(L::Image), RegionKind::Image);
        assert_eq!(RegionKind::classify(L::Chart), RegionKind::Chart);
        assert_eq!(RegionKind::classify(L::DisplayFormula), RegionKind::Equation);
        assert_eq!(RegionKind::classify(L::FigureTitle), RegionKind::Caption);
        assert_eq!(RegionKind::classify(L::VisionFootnote), RegionKind::Footnote);
        assert_eq!(
            RegionKind::classify(L::Header),
            RegionKind::Discarded(TextRole::Header)
        );
        assert_eq!(
            RegionKind::classify(L::Number),
            RegionKind::Discarded(TextRole::PageNumber)
        );
        assert_eq!(RegionKind::classify(L::InlineFormula), RegionKind::Ignored);
    }

    #[test]
    fn text_region_becomes_text_block_with_lines() {
        let bbox = BBox::new(0.0, 0.0, 100.0, 20.0);
        let page = PageAssembler.assemble(vec![region(
            bbox,
            LayoutLabel::Text,
            0,
            lines("hello world", bbox),
        )]);
        assert_eq!(page.blocks.len(), 1);
        match &page.blocks[0] {
            Block::Text { role, lines, .. } => {
                assert_eq!(*role, TextRole::Body);
                assert_eq!(lines.len(), 1);
                match &lines[0].spans[0] {
                    Span::Text { text, .. } => assert_eq!(text, "hello world"),
                    other => panic!("expected text span, got {other:?}"),
                }
            }
            other => panic!("expected text block, got {other:?}"),
        }
    }

    #[test]
    fn header_and_number_go_to_discarded() {
        let bbox = BBox::new(0.0, 0.0, 100.0, 10.0);
        let page = PageAssembler.assemble(vec![
            region(bbox, LayoutLabel::Header, 0, RegionContent::default()),
            region(bbox, LayoutLabel::Number, 1, RegionContent::default()),
            region(
                BBox::new(0.0, 20.0, 100.0, 40.0),
                LayoutLabel::Text,
                2,
                RegionContent::default(),
            ),
        ]);
        assert_eq!(page.blocks.len(), 1);
        assert_eq!(page.discarded.len(), 2);
    }

    #[test]
    fn caption_nests_onto_nearest_image() {
        // Two images; the caption overlaps the second one.
        let img_a = BBox::new(0.0, 0.0, 100.0, 100.0);
        let img_b = BBox::new(0.0, 200.0, 100.0, 300.0);
        let cap = BBox::new(0.0, 300.0, 100.0, 320.0); // just below img_b
        let page = PageAssembler.assemble(vec![
            region(img_a, LayoutLabel::Image, 0, RegionContent::default()),
            region(img_b, LayoutLabel::Image, 1, RegionContent::default()),
            region(cap, LayoutLabel::FigureTitle, 2, lines("Figure 2.", cap)),
        ]);
        assert_eq!(page.blocks.len(), 2);
        // First image (reading order 0) has no caption; second image has it.
        match (&page.blocks[0], &page.blocks[1]) {
            (Block::Image(a), Block::Image(b)) => {
                assert!(a.captions.is_empty());
                assert_eq!(b.captions.len(), 1);
            }
            other => panic!("expected two image blocks, got {other:?}"),
        }
    }

    #[test]
    fn vision_footnote_nests_as_footnote_on_table() {
        let table = BBox::new(0.0, 0.0, 100.0, 100.0);
        let foot = BBox::new(0.0, 100.0, 100.0, 110.0);
        let page = PageAssembler.assemble(vec![
            region(
                table,
                LayoutLabel::Table,
                0,
                RegionContent {
                    table_html: Some(Html("<table></table>".into())),
                    ..Default::default()
                },
            ),
            region(foot, LayoutLabel::VisionFootnote, 1, lines("note", foot)),
        ]);
        assert_eq!(page.blocks.len(), 1);
        match &page.blocks[0] {
            Block::Table(t) => {
                assert_eq!(t.body.html, Html("<table></table>".into()));
                assert_eq!(t.footnotes.len(), 1);
                assert!(t.captions.is_empty());
            }
            other => panic!("expected table block, got {other:?}"),
        }
    }

    #[test]
    fn display_formula_becomes_interline_equation() {
        let bbox = BBox::new(0.0, 0.0, 100.0, 30.0);
        let page = PageAssembler.assemble(vec![region(
            bbox,
            LayoutLabel::DisplayFormula,
            0,
            RegionContent {
                latex: Some(Latex("E=mc^2".into())),
                ..Default::default()
            },
        )]);
        match &page.blocks[0] {
            Block::InterlineEquation { latex, .. } => assert_eq!(latex, &Latex("E=mc^2".into())),
            other => panic!("expected interline equation, got {other:?}"),
        }
    }

    #[test]
    fn blocks_are_emitted_in_reading_order() {
        // Supply out of order; assembler sorts by `order`.
        let b0 = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b1 = BBox::new(0.0, 20.0, 10.0, 30.0);
        let b2 = BBox::new(0.0, 40.0, 10.0, 50.0);
        let page = PageAssembler.assemble(vec![
            region(b2, LayoutLabel::Text, 2, lines("third", b2)),
            region(b0, LayoutLabel::DocTitle, 0, lines("first", b0)),
            region(b1, LayoutLabel::Text, 1, lines("second", b1)),
        ]);
        let texts: Vec<&str> = page
            .blocks
            .iter()
            .filter_map(|b| match b {
                Block::Text { lines, .. } => match &lines[0].spans[0] {
                    Span::Text { text, .. } => Some(text.as_str()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["first", "second", "third"]);
    }

    #[test]
    fn full_synthetic_page_assembles() {
        // A hand-built Vec<LayoutDet>-driven page: title, body, image + caption,
        // table, a discarded footer.
        let title = BBox::new(50.0, 10.0, 550.0, 40.0);
        let body = BBox::new(50.0, 50.0, 550.0, 200.0);
        let img = BBox::new(50.0, 220.0, 300.0, 420.0);
        let cap = BBox::new(50.0, 425.0, 300.0, 445.0);
        let table = BBox::new(320.0, 220.0, 550.0, 420.0);
        let footer = BBox::new(50.0, 780.0, 550.0, 800.0);

        let regions = vec![
            region(title, LayoutLabel::DocTitle, 0, lines("The Title", title)),
            region(body, LayoutLabel::Text, 1, lines("Body paragraph.", body)),
            region(img, LayoutLabel::Image, 2, RegionContent {
                image: Some(ImageRef("p0_img0.png".into())),
                ..Default::default()
            }),
            region(cap, LayoutLabel::FigureTitle, 3, lines("Figure 1.", cap)),
            region(table, LayoutLabel::Table, 4, RegionContent {
                table_html: Some(Html("<table><tr><td>x</td></tr></table>".into())),
                ..Default::default()
            }),
            region(footer, LayoutLabel::Footer, 5, lines("page footer", footer)),
        ];

        let page = PageAssembler.assemble(regions);
        assert_eq!(page.blocks.len(), 4, "title, body, image, table");
        assert_eq!(page.discarded.len(), 1, "footer");

        // Title is first and level 0.
        assert!(matches!(
            page.blocks[0],
            Block::Text { role: TextRole::Title(TitleLevel(0)), .. }
        ));
        // Image carries the caption and the saved ref.
        match &page.blocks[2] {
            Block::Image(c) => {
                assert_eq!(c.captions.len(), 1);
                assert_eq!(c.body.image, ImageRef("p0_img0.png".into()));
            }
            other => panic!("expected image block, got {other:?}"),
        }
        // Table carries its HTML.
        assert!(matches!(&page.blocks[3], Block::Table(_)));
    }
}
