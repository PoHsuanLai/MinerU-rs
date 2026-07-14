//! Native (embedded) text extraction from digital PDFs.
//!
//! This is the Rust port of MinerU's `pdftext`-based text path
//! (`mineru/utils/pdf_text_tool.py`, `pdftext/pdf/{chars,pages}.py`, and
//! `mineru/utils/span_pre_proc.py`). It reads the PDFium text layer for a page,
//! groups glyphs into spans and lines the way `pdftext` does, and back-fills
//! layout-detected span boxes from those native glyphs so digital PDFs read their
//! embedded text instead of being OCR'd line by line.
//!
//! # Why port the algorithm rather than call PDFium's line API
//!
//! PDFium exposes pre-grouped [`segments`](pdfium_render::prelude::PdfPageText::segments),
//! but MinerU's downstream layout logic depends on the *exact* span/line grouping
//! `pdftext` produces (font-change and superscript-driven span breaks, positional
//! line breaks, the char-in-span geometry test). Reproducing that grouping keeps
//! parity with the Python reference. We read raw per-char data from PDFium and run
//! the same heuristics.
//!
//! # Coordinate transform (the load-bearing conversion)
//!
//! PDFium reports character boxes as [`PdfRect`](pdfium_render::prelude::PdfRect)
//! `{ left, bottom, right, top }` in **PDF points with a bottom-left origin**. The
//! rest of MinerU (and [`mineru_types::BBox`]) uses **top-left origin** page
//! points. Mirroring `pdftext`'s `get_chars`, with page bbox origin `(x0, y0)` and
//! `page_height = ceil(|y_end - y_start|)`, a glyph box transforms as
//!
//! ```text
//! bbox.x0 = left  - page_x0
//! bbox.x1 = right - page_x0
//! bbox.y0 = page_height - (top    - page_y0)   // top edge, flipped
//! bbox.y1 = page_height - (bottom - page_y0)   // bottom edge, flipped
//! ```
//!
//! (then normalized so `x0 <= x1`, `y0 <= y1`, which [`BBox::new`] does).
//!
//! # Scope / faithfully-stubbed sub-heuristics
//!
//! Ported: `get_page_chars` (+ the near-identical / offset-duplicate glyph dedup),
//! `get_spans` / `get_lines` grouping, and `fill_char_in_spans` (native back-fill
//! with the `calculate_char_in_span` geometry and the empty-span → OCR routing).
//!
//! Deliberately **out of scope** for this phase (the Python code handles these; we
//! do not, and callers get the honest, simpler behavior instead):
//!
//! - **Rotated / vertical text.** `pdftext` rotates the whole page by
//!   `page_rotation` and MinerU has a separate vertical-span fill path. We only
//!   handle the un-rotated (rotation `0`) standard-orientation case; pages with a
//!   non-zero `/Rotate` or vertical spans fall back to OCR. See
//!   [`PageText::supports_native_fill`].
//! - **`quote_loosebox`.** `pdftext` uses the tight box for apostrophes when
//!   `quote_loosebox=False`; MinerU passes the default `True`, so we always use the
//!   loose box (with a tight-box fallback) and never special-case U+0027.
//! - **Superscript/subscript `<sup>`/`<sub>` wrapping** and **PUA post-OCR
//!   fallback** from `span_pre_proc.py` are not reproduced; span text is the plain
//!   concatenation of its chars with `pdftext`'s inter-char space insertion.

use mineru_types::BBox;
use pdfium_render::prelude::{PdfPage, PdfPoints, PdfRect};

use crate::error::{Error, Result};

/// Tolerance (PDF points) within which two glyph boxes are "the same position".
///
/// Mirrors `NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE` in `pdf_text_tool.py`.
const NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE: f32 = 1.0;
/// Upper bound (points) for an offset-duplicate (shadow) glyph translation.
const OFFSET_DUPLICATE_CHAR_BBOX_TOLERANCE: f32 = 2.5;
/// Max allowed difference between a shadow glyph's start- and end-edge offset;
/// a real shadow is a rigid translation of the same box.
const OFFSET_DUPLICATE_TRANSLATION_TOLERANCE: f32 = 0.1;
/// Minimum overlap (of the smaller box's area) for a shadow-duplicate match.
const OFFSET_DUPLICATE_MIN_BBOX_OVERLAP_RATIO: f32 = 0.45;

/// `superscript_height_threshold` default from MinerU's `get_page` call site.
const SUPERSCRIPT_HEIGHT_THRESHOLD: f32 = 0.7;
/// `line_distance_threshold` default from MinerU's `get_page` call site.
const LINE_DISTANCE_THRESHOLD: f32 = 0.1;

/// PDFium char code for a soft-hyphen line break (`prev_code == 2`).
const CODE_HYPHEN: u32 = 2;
/// PDFium char code for a newline (`prev_code == 10`).
const CODE_NEWLINE: u32 = 10;

/// `Span_Height_Ratio`: a char's mid-line may differ from the span's by at most
/// this fraction of the span height to count as inside the span.
const SPAN_HEIGHT_RATIO: f32 = 0.33;

/// Trailing punctuation that is allowed to attach to a span by its left edge
/// rather than its centre (`LINE_STOP_FLAG` in `span_pre_proc.py`).
const LINE_STOP_FLAG: &[char] = &[
    '.', '!', '?', '。', '！', '？', ')', '）', '"', '”', ':', '：', ';', '；', ']', '】', '}',
    '>', '》', '、', ',', '，', '-', '—', '–',
];
/// Leading punctuation that is allowed to attach to a span by its right edge
/// (`LINE_START_FLAG` in `span_pre_proc.py`).
const LINE_START_FLAG: &[char] = &[
    '(', '（', '"', '“', '【', '{', '《', '<', '「', '『', '[',
];

/// Font descriptor for one glyph: everything `get_spans` breaks a span on.
///
/// Two chars belong to the same span run only when their [`Font`]s and rotations
/// match, so `PartialEq` here is the span-break font test.
#[derive(Debug, Clone, PartialEq)]
pub struct Font {
    /// PostScript font name reported by PDFium (may be empty).
    pub name: String,
    /// Raw font descriptor flag bits.
    pub flags: u32,
    /// Font size in points (`FPDFText_GetFontSize`), rounded to a stable key.
    pub size_millipoints: i32,
    /// Font weight (`FPDFText_GetFontWeight`), or `0` when unknown.
    pub weight: i32,
}

/// A single native glyph: its character, page-space box, font, and provenance.
#[derive(Debug, Clone)]
pub struct TextChar {
    /// The Unicode character.
    pub ch: char,
    /// Glyph box in top-left-origin page points (see the module coordinate note).
    pub bbox: BBox,
    /// Text rotation in radians (`FPDFText_GetCharAngle`); `0` for upright text.
    pub rotation: f32,
    /// The font applied to this glyph.
    pub font: Font,
    /// Original PDFium char index on the page (reading order key).
    pub char_idx: usize,
}

impl TextChar {
    /// Whether this glyph is upright (rotation ≈ 0), the only orientation the
    /// native-fill path handles this phase.
    fn is_upright(&self) -> bool {
        self.rotation.abs() < 1e-3
    }
}

/// A run of same-font, same-rotation glyphs (`pdftext`'s `Span`).
#[derive(Debug, Clone)]
pub struct TextSpan {
    /// Span box (union of its chars) in top-left page points.
    pub bbox: BBox,
    /// Concatenated span text (one `char` per glyph, no inter-char spacing).
    pub text: String,
    /// Font shared by every char in the span.
    pub font: Font,
    /// Text rotation in radians.
    pub rotation: f32,
    /// The glyphs making up this span, in reading order.
    pub chars: Vec<TextChar>,
}

/// A visual line: consecutive spans `pdftext`'s `get_lines` kept together.
#[derive(Debug, Clone)]
pub struct TextLine {
    /// Line box (union of its spans) in top-left page points.
    pub bbox: BBox,
    /// Line rotation in radians.
    pub rotation: f32,
    /// The spans making up this line, in reading order.
    pub spans: Vec<TextSpan>,
}

impl TextLine {
    /// The line's text: its spans' texts concatenated.
    pub fn text(&self) -> String {
        self.spans.iter().map(|s| s.text.as_str()).collect()
    }
}

/// The native text layer of one page: its glyphs and their line grouping.
///
/// `chars` are every surviving glyph (post-dedup) in PDFium reading order; `lines`
/// is the `get_lines_from_chars` grouping over them. `fill_regions` fills a region
/// from its overlapping `lines` (each a single-line span for the char-in-span
/// geometry); `lines` is also exposed for callers that want ready-made line text.
#[derive(Debug, Clone, Default)]
pub struct PageText {
    /// Every glyph on the page, deduplicated, in reading order.
    pub chars: Vec<TextChar>,
    /// Glyphs grouped into spans and lines.
    pub lines: Vec<TextLine>,
    /// Page rotation flag from PDFium (`0` for upright pages).
    pub rotation: i32,
}

impl PageText {
    /// Whether this page can be served from its native text layer.
    ///
    /// False for empty pages, rotated pages, and pages whose glyphs are all
    /// rotated/vertical — those must fall back to OCR (see the scope note).
    pub fn supports_native_fill(&self) -> bool {
        self.rotation == 0 && self.chars.iter().any(TextChar::is_upright)
    }

    /// Back-fills each region box with the native glyphs inside it, returning the
    /// filled text per region (`None` for a region no native text could fill).
    ///
    /// Port of `fill_char_in_spans` + `chars_to_content` (`span_pre_proc.py`).
    ///
    /// # Why the region is filled line-by-line
    ///
    /// In the Python pipeline `fill_char_in_spans` receives *single-line* span
    /// boxes (one per detected text line), so [`calculate_char_in_span`]'s
    /// vertical [`SPAN_HEIGHT_RATIO`] test — which keeps only chars near the box
    /// mid-line — is exactly right. This Rust caller instead passes whole
    /// multi-line layout regions. Running the same mid-line test against a
    /// multi-line box would drop every char far from the region centre (its first
    /// and last text lines), so we must **not** treat a region as one span.
    ///
    /// Instead each region is intersected with the page's already-correct
    /// [`get_lines_from_chars`] grouping ([`Self::lines`]): every overlapping
    /// [`TextLine`] acts as the single-line span the geometry test expects. Chars
    /// are collected per line (no vertical drop), lines are ordered top-to-bottom
    /// and their chars left-to-right, each line is joined with `pdftext`'s
    /// inter-char space insertion ([`join_with_spacing`], `median_width`
    /// recomputed per line), and wrapped lines join with a single space — matching
    /// how the pipeline later stitches a paragraph's line spans back together. A
    /// region that ends up empty, or whose text is too sparse for its width (the
    /// `len*height < width*0.5` guard), returns `None` so the caller OCRs it.
    ///
    /// `region_boxes` are in the same top-left page-point space as [`TextChar`].
    /// The returned vector is parallel to `region_boxes`.
    pub fn fill_regions(&self, region_boxes: &[BBox]) -> Vec<Option<FilledRegion>> {
        region_boxes
            .iter()
            .map(|region| self.fill_region(region))
            .collect()
    }

    /// Fills a single region by collecting each overlapping page line's chars and
    /// assembling them in reading order (see [`Self::fill_regions`]).
    fn fill_region(&self, region: &BBox) -> Option<FilledRegion> {
        // Each overlapping page line is a single-line "span" for the geometry
        // test. Ordered top-to-bottom by the line's mid-y so wrapped lines read in
        // visual order (mirrors `fill_char_in_spans`' top-to-bottom span sort).
        let mut region_lines: Vec<RegionLine> = Vec::new();
        for line in &self.lines {
            // A line belongs to this region when its vertical centre falls inside
            // the box. Using the centre (not mere overlap) keeps a neighbouring
            // line that only clips the region's top/bottom edge from being slurped
            // in, while still admitting a line that extends past the box sideways.
            let (_, cy) = line.bbox.center();
            if !(region.y0 <= cy && cy <= region.y1) {
                continue;
            }
            let chars = collect_line_chars(line, region);
            if chars.is_empty() {
                continue;
            }
            region_lines.push(RegionLine { center_y: cy, chars });
        }
        finish_region(region_lines, region)
    }
}

/// One page line's glyphs that fell inside a region, tagged with the line's
/// vertical centre so lines can be ordered top-to-bottom within the region.
struct RegionLine<'a> {
    /// Mid-y of the source [`TextLine`] box (region top-to-bottom sort key).
    center_y: f32,
    /// The line's glyphs inside the region, in reading order.
    chars: Vec<&'a TextChar>,
}

/// Collects the upright glyphs of one page line that fall inside `region`.
///
/// Runs the same char-in-span geometry as Python's `fill_char_in_spans`, but
/// against the *line's* box (a true single-line span) rather than the multi-line
/// region — so [`SPAN_HEIGHT_RATIO`] no longer drops a region's leading/trailing
/// lines. The horizontal reject and leading/trailing-punctuation relaxation match
/// the Python inner loop.
fn collect_line_chars<'a>(line: &'a TextLine, region: &BBox) -> Vec<&'a TextChar> {
    let mut chars: Vec<&TextChar> = Vec::new();
    for span in &line.spans {
        for ch in &span.chars {
            if !ch.is_upright() {
                continue;
            }
            let (cx, _cy) = ch.bbox.center();
            // A char belongs to the region only if it is horizontally inside it —
            // this is what confines a page-wide line to the region's column.
            let is_flag = LINE_STOP_FLAG.contains(&ch.ch) || LINE_START_FLAG.contains(&ch.ch);
            if !(is_flag || (region.x0 < cx && cx < region.x1)) {
                continue;
            }
            // Vertical membership is tested against the line box (single line), so
            // no char is dropped for being far from a multi-line region's centre.
            if calculate_char_in_span(&ch.bbox, &line.bbox, ch.ch) {
                chars.push(ch);
            }
        }
    }
    chars
}

/// A region successfully filled from native text.
#[derive(Debug, Clone)]
pub struct FilledRegion {
    /// The joined region text (control line-breaks removed, spaces inserted).
    pub text: String,
    /// The glyphs that filled it, ordered by original char index.
    pub chars: Vec<TextChar>,
}

/// Assembles a region's per-line glyphs into its final text, or `None` when the
/// region should fall through to OCR.
///
/// Lines are ordered top-to-bottom; each line's chars are ordered left-to-right by
/// their original char index (`chars_to_content`), joined with `pdftext`'s
/// inter-char spacing ([`join_with_spacing`], `median_width` recomputed per line),
/// and the resulting line texts joined with a single space — the way a paragraph's
/// wrapped lines are later stitched back together.
fn finish_region(mut lines: Vec<RegionLine<'_>>, region: &BBox) -> Option<FilledRegion> {
    if lines.is_empty() {
        return None;
    }
    // Top-to-bottom reading order (top-left origin: smaller y is higher).
    lines.sort_by(|a, b| {
        a.center_y
            .partial_cmp(&b.center_y)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut line_texts: Vec<String> = Vec::with_capacity(lines.len());
    // Every glyph that filled the region, in reading order, for the caller.
    let mut kept: Vec<TextChar> = Vec::new();
    for line in &mut lines {
        // `chars_to_content`: order by original char index (usually already sorted).
        line.chars.sort_by_key(|c| c.char_idx);
        // Drop control line-breaks so they don't drive spurious spacing.
        let visible: Vec<&TextChar> = line
            .chars
            .iter()
            .copied()
            .filter(|c| c.ch != '\r' && c.ch != '\n')
            .collect();
        if visible.is_empty() {
            continue;
        }
        let line_text = join_with_spacing(&visible);
        if !line_text.trim().is_empty() {
            line_texts.push(line_text.trim().to_string());
            kept.extend(visible.into_iter().cloned());
        }
    }
    if line_texts.is_empty() {
        return None;
    }

    // Wrapped lines rejoin with a single space (the pipeline's paragraph line join).
    let text = line_texts.join(" ");
    let trimmed = text.trim().to_string();

    // Empty-span guard: `len(content)*height < width*0.5` → treat as unfilled.
    let width = region.width();
    let height = region.height();
    if (trimmed.chars().count() as f32) * height < width * 0.5 {
        return None;
    }

    Some(FilledRegion {
        text: trimmed,
        chars: kept,
    })
}

/// Joins one line's glyphs into text, inserting a space where the horizontal gap to
/// the next glyph exceeds `0.25 * median_char_width` (port of the spacing loop in
/// `chars_to_content`). `median_char_width` is computed over *this line's* chars, so
/// the threshold tracks the line's font size. Ligatures and the `\u{0002}`
/// soft-hyphen are normalized.
fn join_with_spacing(chars: &[&TextChar]) -> String {
    let median_w = median(chars.iter().map(|c| c.bbox.width()));
    let mut out = String::new();
    for (idx, ch) in chars.iter().enumerate() {
        push_normalized(&mut out, ch.ch);
        if let Some(next) = chars.get(idx + 1) {
            let gap = next.bbox.x0 - ch.bbox.x1;
            if gap > median_w * 0.25 && ch.ch != ' ' && next.ch != ' ' {
                out.push(' ');
            }
        }
    }
    out
}

/// Applies `pdftext`'s ligature/unicode substitutions for a single char.
fn push_normalized(out: &mut String, ch: char) {
    match ch {
        // `__replace_unicode`:  (soft hyphen marker) → '-'.
        '\u{0002}' => out.push('-'),
        // `__replace_ligatures`.
        'ﬁ' => out.push_str("fi"),
        'ﬂ' => out.push_str("fl"),
        'ﬀ' => out.push_str("ff"),
        'ﬃ' => out.push_str("ffi"),
        'ﬄ' => out.push_str("ffl"),
        'ﬅ' => out.push_str("ft"),
        'ﬆ' => out.push_str("st"),
        other => out.push(other),
    }
}

/// The char-in-span geometry test (`calculate_char_in_span` in `span_pre_proc.py`).
///
/// A glyph is in the span when its centre lies inside the box **and** within
/// [`SPAN_HEIGHT_RATIO`] of the span mid-line. Leading/trailing punctuation gets a
/// relaxed edge-based test so end-of-line marks still attach.
pub fn calculate_char_in_span(char_bbox: &BBox, span_bbox: &BBox, ch: char) -> bool {
    let (char_cx, char_cy) = char_bbox.center();
    let (_span_cx, span_cy) = span_bbox.center();
    let span_height = span_bbox.height();
    let within_axis = (char_cy - span_cy).abs() < span_height * SPAN_HEIGHT_RATIO;

    if span_bbox.x0 < char_cx
        && char_cx < span_bbox.x1
        && span_bbox.y0 < char_cy
        && char_cy < span_bbox.y1
        && within_axis
    {
        return true;
    }

    if LINE_STOP_FLAG.contains(&ch) {
        return (span_bbox.x1 - span_height) < char_bbox.x0
            && char_bbox.x0 < span_bbox.x1
            && char_cx > span_bbox.x0
            && span_bbox.y0 < char_cy
            && char_cy < span_bbox.y1
            && within_axis;
    }
    if LINE_START_FLAG.contains(&ch) {
        return span_bbox.x0 < char_bbox.x1
            && char_bbox.x1 < (span_bbox.x0 + span_height)
            && char_cx < span_bbox.x1
            && span_bbox.y0 < char_cy
            && char_cy < span_bbox.y1
            && within_axis;
    }
    false
}

/// Reads and groups the native text layer of `page`.
///
/// Runs `get_page_chars` (extract + dedup) then `get_lines_from_chars` (span/line
/// grouping) — the two functions MinerU's `get_page` composes.
pub(crate) fn extract_page_text(page: &PdfPage<'_>, page_index: usize) -> Result<PageText> {
    let rotation = page.rotation().map(rotation_degrees).unwrap_or(0);
    let page_height = page.height().value.abs().ceil();

    let chars = get_page_chars(page, page_index, page_height)?;
    let lines = get_lines_from_chars(&chars);
    Ok(PageText {
        chars,
        lines,
        rotation,
    })
}

/// Degrees for a PDFium page rotation enum, or `0` for the upright case.
fn rotation_degrees(rot: pdfium_render::prelude::PdfPageRenderRotation) -> i32 {
    use pdfium_render::prelude::PdfPageRenderRotation as R;
    match rot {
        R::None => 0,
        R::Degrees90 => 90,
        R::Degrees180 => 180,
        R::Degrees270 => 270,
    }
}

/// `get_page_chars`: pull every glyph from the PDFium text page, convert to
/// top-left page points, and drop near-identical / shadow-offset duplicate glyphs.
fn get_page_chars(page: &PdfPage<'_>, page_index: usize, page_height: f32) -> Result<Vec<TextChar>> {
    let text = page.text().map_err(|e| Error::Text {
        page: page_index,
        message: e.to_string(),
    })?;

    let mut raw = Vec::new();
    for (char_idx, pdf_char) in text.chars().iter().enumerate() {
        // Skip glyphs with no Unicode mapping (PDFium returns 0 / invalid).
        let Some(ch) = pdf_char.unicode_char() else {
            continue;
        };
        let rotation = pdf_char.angle_radians().unwrap_or(0.0);
        // Prefer the loose box (matches `quote_loosebox=True`); fall back to tight.
        let rect = match pdf_char.loose_bounds() {
            Ok(r) => r,
            Err(_) => match pdf_char.tight_bounds() {
                Ok(r) => r,
                Err(_) => continue,
            },
        };
        let bbox = rect_to_bbox(&rect, page_height);
        let font = Font {
            name: pdf_char.font_name(),
            flags: 0,
            size_millipoints: (pdf_char.scaled_font_size().value * 1000.0).round() as i32,
            weight: pdf_char
                .font_weight()
                .and_then(font_weight_value)
                .unwrap_or(0),
        };
        raw.push(TextChar {
            ch,
            bbox,
            rotation,
            font,
            char_idx,
        });
    }

    Ok(deduplicate_near_identical_chars(raw))
}

/// Numeric weight for a PDFium font-weight enum (used only as a span-break key).
fn font_weight_value(w: pdfium_render::prelude::PdfFontWeight) -> Option<i32> {
    use pdfium_render::prelude::PdfFontWeight as W;
    Some(match w {
        W::Weight100 => 100,
        W::Weight200 => 200,
        W::Weight300 => 300,
        W::Weight400Normal => 400,
        W::Weight500 => 500,
        W::Weight600 => 600,
        W::Weight700Bold => 700,
        W::Weight800 => 800,
        W::Weight900 => 900,
        W::Custom(v) => v as i32,
    })
}

/// Converts a PDFium [`PdfRect`] (bottom-left origin) to a top-left-origin
/// [`BBox`]. See the module-level coordinate note.
fn rect_to_bbox(rect: &PdfRect, page_height: f32) -> BBox {
    let left = value(rect.left());
    let right = value(rect.right());
    let top = value(rect.top());
    let bottom = value(rect.bottom());
    // Y flip about page_height; BBox::new normalizes ordering.
    BBox::new(left, page_height - top, right, page_height - bottom)
}

/// Extracts the scalar from a [`PdfPoints`].
fn value(p: PdfPoints) -> f32 {
    p.value
}

/// `_deduplicate_near_identical_chars`: drop shadow-offset duplicates (adjacent,
/// same glyph, rigid diagonal translation) and near-identical overlapping glyphs
/// at the same position/font.
fn deduplicate_near_identical_chars(chars: Vec<TextChar>) -> Vec<TextChar> {
    // Signature → list of kept boxes at that (visible-char, font, rotation).
    let mut seen: std::collections::HashMap<CharSignature, Vec<BBox>> =
        std::collections::HashMap::new();
    let mut out: Vec<TextChar> = Vec::with_capacity(chars.len());

    for ch in chars {
        // Whitespace passes through untouched (Python keeps it verbatim).
        if ch.ch.is_whitespace() {
            out.push(ch);
            continue;
        }
        // Adjacent shadow-offset duplicate: compare against the last kept char.
        if let Some(prev) = out.last() {
            if is_adjacent_offset_duplicate(prev, &ch) {
                continue;
            }
        }
        let sig = char_signature(&ch);
        let entry = seen.entry(sig).or_default();
        if entry
            .iter()
            .any(|seen_box| is_near_identical_bbox(&ch.bbox, seen_box))
        {
            continue;
        }
        entry.push(ch.bbox);
        out.push(ch);
    }
    out
}

/// The dedup signature: visible char, font, and rounded rotation (bbox excluded so
/// near-identical positions are compared separately).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CharSignature {
    ch: char,
    font_name: String,
    flags: u32,
    size_millipoints: i32,
    weight: i32,
    rotation_milli: i32,
}

/// Builds a [`CharSignature`] for a glyph.
fn char_signature(ch: &TextChar) -> CharSignature {
    CharSignature {
        ch: ch.ch,
        font_name: ch.font.name.clone(),
        flags: ch.font.flags,
        size_millipoints: ch.font.size_millipoints,
        weight: ch.font.weight,
        rotation_milli: (ch.rotation * 1000.0).round() as i32,
    }
}

/// `_is_near_identical_bbox`: every corner within [`NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE`].
fn is_near_identical_bbox(a: &BBox, b: &BBox) -> bool {
    (a.x0 - b.x0).abs() <= NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE
        && (a.y0 - b.y0).abs() <= NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE
        && (a.x1 - b.x1).abs() <= NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE
        && (a.y1 - b.y1).abs() <= NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE
}

/// `_calculate_bbox_overlap_in_smaller_area`: intersection / smaller-box area.
fn bbox_overlap_in_smaller_area(a: &BBox, b: &BBox) -> f32 {
    let inter = a.intersection(b).map_or(0.0, |r| r.area());
    let smaller = a.area().min(b.area());
    if smaller == 0.0 {
        0.0
    } else {
        inter / smaller
    }
}

/// `_is_adjacent_offset_duplicate_char`: `current` is a diagonal-shadow copy of the
/// immediately-preceding `previous` glyph (same signature, rigid small translation,
/// high overlap).
fn is_adjacent_offset_duplicate(previous: &TextChar, current: &TextChar) -> bool {
    if char_signature(previous) != char_signature(current) {
        return false;
    }
    let x_start = current.bbox.x0 - previous.bbox.x0;
    let y_start = current.bbox.y0 - previous.bbox.y0;
    let x_end = current.bbox.x1 - previous.bbox.x1;
    let y_end = current.bbox.y1 - previous.bbox.y1;

    // Must be a rigid translation (start and end edges shift equally).
    if (x_start - x_end).abs() > OFFSET_DUPLICATE_TRANSLATION_TOLERANCE
        || (y_start - y_end).abs() > OFFSET_DUPLICATE_TRANSLATION_TOLERANCE
    {
        return false;
    }
    // Translation magnitude must be in the (near-identical, shadow] band on both axes.
    let in_band = |v: f32| {
        NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE < v.abs() && v.abs() <= OFFSET_DUPLICATE_CHAR_BBOX_TOLERANCE
    };
    if !(in_band(x_start) && in_band(y_start)) {
        return false;
    }
    bbox_overlap_in_smaller_area(&previous.bbox, &current.bbox) >= OFFSET_DUPLICATE_MIN_BBOX_OVERLAP_RATIO
}

/// `get_lines_from_chars`: build spans then group them into lines.
pub(crate) fn get_lines_from_chars(chars: &[TextChar]) -> Vec<TextLine> {
    let spans = get_spans(chars);
    get_lines(spans)
}

/// `get_spans`: split the char stream into runs at font/rotation changes,
/// hyphen/newline breaks, and superscript starts (port of `pdftext`'s `get_spans`).
fn get_spans(chars: &[TextChar]) -> Vec<TextSpan> {
    if chars.is_empty() {
        return Vec::new();
    }
    let mut spans = Vec::new();
    let mut start = 0usize;
    // Running span bbox, accumulated as we extend the current run.
    let mut acc = chars[0].bbox;

    for j in 1..chars.len() {
        let prev = &chars[j - 1];
        let cur = &chars[j];
        let start_char = &chars[start];
        let height = acc.y1 - acc.y0;

        let prev_code = prev.ch as u32;
        let font_or_rotation_change =
            cur.font != start_char.font || cur.rotation != start_char.rotation;
        let hyphen_or_newline = prev_code == CODE_HYPHEN || prev_code == CODE_NEWLINE;
        // Superscript start: top above the run, bottom under the height threshold,
        // and to the right of the run's right edge.
        let is_superscript = cur.bbox.y0 < acc.y0 - height * LINE_DISTANCE_THRESHOLD
            && cur.bbox.y1 < height * SUPERSCRIPT_HEIGHT_THRESHOLD + acc.y0
            && cur.bbox.x0 > acc.x1;

        if font_or_rotation_change || hyphen_or_newline || is_superscript {
            spans.push(build_span(&chars[start..j], acc));
            start = j;
            acc = cur.bbox;
        } else {
            acc = union(&acc, &cur.bbox);
        }
    }
    spans.push(build_span(&chars[start..], acc));
    spans
}

/// Materializes a [`TextSpan`] from a slice of chars and their accumulated box.
fn build_span(chars: &[TextChar], bbox: BBox) -> TextSpan {
    let text: String = chars.iter().map(|c| c.ch).collect();
    let (font, rotation) = chars
        .first()
        .map(|c| (c.font.clone(), c.rotation))
        .unwrap_or_else(|| (default_font(), 0.0));
    TextSpan {
        bbox,
        text,
        font,
        rotation,
        chars: chars.to_vec(),
    }
}

/// An empty placeholder font (only used for the degenerate empty-slice case).
fn default_font() -> Font {
    Font {
        name: String::new(),
        flags: 0,
        size_millipoints: 0,
        weight: 0,
    }
}

/// `get_lines`: fold spans into lines, breaking on trailing linebreaks,
/// near-perpendicular rotation changes, or a downward positional jump.
fn get_lines(spans: Vec<TextSpan>) -> Vec<TextLine> {
    let mut lines: Vec<TextLine> = Vec::new();

    for span in spans {
        let break_line = match lines.last() {
            None => true,
            Some(line) => {
                let last_text = line
                    .spans
                    .last()
                    .map(|s| s.text.as_str())
                    .unwrap_or_default();
                let ends_break = last_text.ends_with('\n') || last_text.ends_with('\u{0002}');
                let rotation_break = {
                    let mut diff = (span.rotation - line.rotation).abs() % (2.0 * std::f32::consts::PI);
                    diff = diff.min(2.0 * std::f32::consts::PI - diff);
                    span.rotation != line.rotation
                        && (std::f32::consts::PI / 4.0..=3.0 * std::f32::consts::PI / 4.0).contains(&diff)
                };
                // Positional break: the span starts below the line's bottom.
                let positional_break = span.bbox.y0 > line.bbox.y1;
                ends_break || rotation_break || positional_break
            }
        };

        if break_line {
            lines.push(TextLine {
                bbox: span.bbox,
                rotation: span.rotation,
                spans: vec![span],
            });
        } else if let Some(line) = lines.last_mut() {
            line.bbox = union(&line.bbox, &span.bbox);
            line.spans.push(span);
        }
    }
    lines
}

/// Union of two boxes (`Bbox.merge`).
fn union(a: &BBox, b: &BBox) -> BBox {
    BBox::new(
        a.x0.min(b.x0),
        a.y0.min(b.y0),
        a.x1.max(b.x1),
        a.y1.max(b.y1),
    )
}

/// Median of a value iterator (empty → `0.0`), matching `statistics.median`
/// (mean of the two middle values for an even count).
fn median<I: Iterator<Item = f32>>(iter: I) -> f32 {
    let mut vals: Vec<f32> = iter.collect();
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();
    if n % 2 == 1 {
        vals[n / 2]
    } else {
        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn font(name: &str, size: i32) -> Font {
        Font {
            name: name.to_string(),
            flags: 0,
            size_millipoints: size,
            weight: 400,
        }
    }

    fn ch(c: char, x0: f32, y0: f32, x1: f32, y1: f32, f: Font, idx: usize) -> TextChar {
        TextChar {
            ch: c,
            bbox: BBox::new(x0, y0, x1, y1),
            rotation: 0.0,
            font: f,
            char_idx: idx,
        }
    }

    /// A row of same-font glyphs at the same y clusters into one line/span.
    #[test]
    fn chars_in_a_row_form_one_line() {
        let f = font("Arial", 12000);
        let chars: Vec<TextChar> = "Hello"
            .chars()
            .enumerate()
            .map(|(i, c)| {
                let x = 10.0 + i as f32 * 8.0;
                ch(c, x, 100.0, x + 7.0, 112.0, f.clone(), i)
            })
            .collect();
        let lines = get_lines_from_chars(&chars);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text(), "Hello");
    }

    /// A vertical drop between rows splits them into two lines.
    ///
    /// PDFium injects a newline glyph (`\n`, code 10) at each visual line break;
    /// that ends the current span (`prev_code == 10`), and the positional test in
    /// `get_lines` then starts a fresh line for the row below. This mirrors real
    /// text-page data (`get_spans` never breaks a run on vertical position alone).
    #[test]
    fn positional_drop_splits_lines() {
        let f = font("Arial", 12000);
        let mut chars = Vec::new();
        for (i, c) in "AB".chars().enumerate() {
            let x = 10.0 + i as f32 * 8.0;
            chars.push(ch(c, x, 100.0, x + 7.0, 112.0, f.clone(), i));
        }
        // PDFium's end-of-line newline: ends the first span.
        chars.push(ch('\n', 26.0, 100.0, 26.0, 112.0, f.clone(), 2));
        // Next row is well below the first row's bottom (y0 120 > y1 112).
        for (i, c) in "CD".chars().enumerate() {
            let x = 10.0 + i as f32 * 8.0;
            chars.push(ch(c, x, 120.0, x + 7.0, 132.0, f.clone(), i + 3));
        }
        let lines = get_lines_from_chars(&chars);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text(), "AB\n");
        assert_eq!(lines[1].text(), "CD");
    }

    /// A font change starts a new span within the same line.
    #[test]
    fn font_change_splits_span_not_line() {
        let a = font("Arial", 12000);
        let b = font("Times", 12000);
        let chars = vec![
            ch('X', 10.0, 100.0, 17.0, 112.0, a.clone(), 0),
            ch('Y', 18.0, 100.0, 25.0, 112.0, b.clone(), 1),
        ];
        let lines = get_lines_from_chars(&chars);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].spans.len(), 2, "font change → two spans");
        assert_eq!(lines[0].text(), "XY");
    }

    /// Near-identical duplicate glyphs (fake-bold double render) are dropped.
    #[test]
    fn near_identical_duplicates_dropped() {
        let f = font("Arial", 12000);
        let chars = vec![
            ch('A', 10.0, 100.0, 17.0, 112.0, f.clone(), 0),
            // Same glyph, same font, box within 1pt tolerance → duplicate.
            ch('A', 10.3, 100.2, 17.2, 112.1, f.clone(), 1),
        ];
        let out = deduplicate_near_identical_chars(chars);
        assert_eq!(out.len(), 1);
    }

    /// A diagonal shadow copy (rigid ~2pt translation) of the previous glyph is dropped.
    #[test]
    fn adjacent_offset_shadow_dropped() {
        let f = font("Arial", 12000);
        let chars = vec![
            ch('B', 10.0, 100.0, 17.0, 112.0, f.clone(), 0),
            // Rigid +1.8 / +1.8 translation, high overlap → shadow duplicate.
            ch('B', 11.8, 101.8, 18.8, 113.8, f.clone(), 1),
        ];
        let out = deduplicate_near_identical_chars(chars);
        assert_eq!(out.len(), 1);
    }

    /// Distinct adjacent glyphs at different positions are kept.
    #[test]
    fn distinct_chars_kept() {
        let f = font("Arial", 12000);
        let chars = vec![
            ch('A', 10.0, 100.0, 17.0, 112.0, f.clone(), 0),
            ch('B', 18.0, 100.0, 25.0, 112.0, f.clone(), 1),
        ];
        let out = deduplicate_near_identical_chars(chars);
        assert_eq!(out.len(), 2);
    }

    /// Builds a [`PageText`] from chars, grouping lines the way the real page does.
    fn page_from_chars(chars: Vec<TextChar>) -> PageText {
        let lines = get_lines_from_chars(&chars);
        PageText {
            chars,
            lines,
            rotation: 0,
        }
    }

    /// One text row plus a PDFium end-of-line newline glyph, at vertical `y0`.
    ///
    /// The trailing `\n` (code 10) ends the row's span, so a following row starts a
    /// fresh line — the shape real PDFium text pages have (see
    /// `positional_drop_splits_lines`).
    fn text_row(word: &str, y0: f32, x_start: f32, start_idx: usize, f: &Font) -> Vec<TextChar> {
        let mut out = Vec::new();
        let mut x = x_start;
        for (i, c) in word.chars().enumerate() {
            out.push(ch(c, x, y0, x + 7.0, y0 + 12.0, f.clone(), start_idx + i));
            x += 8.0;
        }
        out.push(ch('\n', x, y0, x, y0 + 12.0, f.clone(), start_idx + word.chars().count()));
        out
    }

    /// A span fill collects exactly the chars whose centre lies in the region box.
    #[test]
    fn fill_region_picks_inside_chars() {
        let f = font("Arial", 12000);
        // Two rows: row 1 inside region, row 2 outside (below).
        let mut chars = text_row("IN", 100.0, 12.0, 0, &f);
        chars.extend(text_row("NO", 300.0, 12.0, 10, &f));
        let page = page_from_chars(chars);
        // Region covering only row 1.
        let region = BBox::new(8.0, 96.0, 40.0, 116.0);
        let filled = page.fill_regions(&[region]);
        assert_eq!(filled.len(), 1);
        let r = filled[0].as_ref().expect("region filled from native text");
        assert_eq!(r.text, "IN");
    }

    /// A region with no native chars falls through (None → OCR).
    #[test]
    fn empty_region_returns_none() {
        let page = PageText {
            chars: Vec::new(),
            lines: Vec::new(),
            rotation: 0,
        };
        let region = BBox::new(0.0, 0.0, 100.0, 20.0);
        let filled = page.fill_regions(&[region]);
        assert!(filled[0].is_none());
    }

    /// Wide gaps between glyphs insert a space (word separation within a line).
    #[test]
    fn wide_gap_inserts_space() {
        let f = font("Arial", 12000);
        // "AB" then a big gap then "CD" on one line.
        let chars = vec![
            ch('A', 10.0, 100.0, 16.0, 112.0, f.clone(), 0),
            ch('B', 16.5, 100.0, 22.5, 112.0, f.clone(), 1),
            // Gap of ~10pt (>> 0.25 * ~6pt median width).
            ch('C', 40.0, 100.0, 46.0, 112.0, f.clone(), 2),
            ch('D', 46.5, 100.0, 52.5, 112.0, f.clone(), 3),
        ];
        let page = page_from_chars(chars);
        let region = BBox::new(5.0, 96.0, 60.0, 116.0);
        let filled = page.fill_regions(&[region]);
        let r = filled[0].as_ref().expect("filled");
        assert_eq!(r.text, "AB CD");
    }

    /// Regression (Defect #2): a two-line region must include BOTH lines' text, in
    /// top-to-bottom order — the OLD whole-region fill dropped chars far from the
    /// region mid-line (its first/last lines).
    #[test]
    fn two_line_region_keeps_both_lines_in_order() {
        let f = font("Arial", 12000);
        // Two rows, ~13pt apart, both inside a tall region.
        let mut chars = text_row("First", 100.0, 12.0, 0, &f);
        chars.extend(text_row("Second", 113.0, 12.0, 10, &f));
        let page = page_from_chars(chars);
        // Region tall enough to span BOTH rows; its mid-line sits between them.
        let region = BBox::new(8.0, 96.0, 80.0, 130.0);
        let r = page.fill_regions(&[region])[0]
            .as_ref()
            .expect("region filled")
            .clone();
        assert_eq!(r.text, "First Second");
    }

    /// Regression (Defect #2, vertical drop): chars whose centre is far from the
    /// multi-line region's centre are NOT dropped. A three-line region keeps its
    /// leading line — the exact symptom of the Abstract starting mid-sentence.
    #[test]
    fn far_from_center_chars_not_dropped() {
        let f = font("Arial", 12000);
        let mut chars = text_row("Top", 100.0, 12.0, 0, &f);
        chars.extend(text_row("Mid", 120.0, 12.0, 10, &f));
        chars.extend(text_row("Bot", 140.0, 12.0, 20, &f));
        let page = page_from_chars(chars);
        // Tall region: "Top" (y~106) is ~26pt above the region mid-line (~129),
        // which the old SPAN_HEIGHT_RATIO test against the whole region dropped.
        let region = BBox::new(8.0, 96.0, 60.0, 158.0);
        let r = page.fill_regions(&[region])[0]
            .as_ref()
            .expect("region filled")
            .clone();
        assert_eq!(r.text, "Top Mid Bot");
        assert!(r.text.starts_with("Top"), "leading line must not be dropped");
    }

    /// Regression (Defect #3): a line wrap yields a SPACE, not a word-join, even
    /// though the next line's x0 is left of the previous line's x1. The OLD path
    /// joined the whole region by char_idx as one line → "reproduceright".
    #[test]
    fn line_wrap_joins_with_space_not_wordjoin() {
        let f = font("Arial", 12000);
        // Line 1 ends at the right; line 2 restarts at the left (x0 < line1.x1).
        let mut chars = text_row("reproduce", 100.0, 40.0, 0, &f);
        chars.extend(text_row("right", 113.0, 12.0, 20, &f));
        let page = page_from_chars(chars);
        let region = BBox::new(8.0, 96.0, 130.0, 130.0);
        let r = page.fill_regions(&[region])[0]
            .as_ref()
            .expect("region filled")
            .clone();
        assert_eq!(r.text, "reproduce right");
        assert!(!r.text.contains("reproduceright"), "no cross-line word-join");
    }

    /// Regression (Defect #3): the intra-line space threshold uses the LINE's
    /// median char width, so a real word gap in a small-font line yields a space
    /// even when another line in the same region has a much larger font.
    #[test]
    fn per_line_median_width_spaces_small_font_line() {
        let big = font("Arial", 24000);
        let small = font("Arial", 10000);
        // Big-font heading line (wide chars) at the top.
        let mut chars = Vec::new();
        let mut x = 12.0;
        for (i, c) in "TITLE".chars().enumerate() {
            chars.push(ch(c, x, 100.0, x + 18.0, 124.0, big.clone(), i));
            x += 20.0;
        }
        chars.push(ch('\n', x, 100.0, x, 124.0, big.clone(), 5));
        // Small-font line below: "to" gap "do" — gap ~4pt, small median width ~5pt.
        let mut chars2 = vec![
            ch('t', 12.0, 130.0, 17.0, 140.0, small.clone(), 6),
            ch('o', 17.5, 130.0, 22.5, 140.0, small.clone(), 7),
            // Word gap ~4pt (> 0.25 * ~5pt small median, but < 0.25 * big median).
            ch('d', 27.0, 130.0, 32.0, 140.0, small.clone(), 8),
            ch('o', 32.5, 130.0, 37.5, 140.0, small.clone(), 9),
        ];
        chars.append(&mut chars2);
        let page = page_from_chars(chars);
        let region = BBox::new(8.0, 96.0, 120.0, 146.0);
        let r = page.fill_regions(&[region])[0]
            .as_ref()
            .expect("region filled")
            .clone();
        assert_eq!(r.text, "TITLE to do");
    }

    /// Median matches Python's `statistics.median` (even-count average).
    #[test]
    fn median_even_count_averages_middle() {
        assert_eq!(median([1.0, 2.0, 3.0, 4.0].into_iter()), 2.5);
        assert_eq!(median([5.0, 1.0, 3.0].into_iter()), 3.0);
        assert_eq!(median(std::iter::empty()), 0.0);
    }
}
