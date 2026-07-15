//! Choosing between the wired and wireless recognitions of the same table.
//!
//! Port of the selection in `UnetTableModel.predict` (`unet_table/main.py:274`).
//! The classifier alone is not trusted to pick the engine: it is a 224x224
//! whole-crop judgement, and a table whose rules are faint or partial reads as
//! borderless to it while the wired engine still recovers the grid perfectly.
//! So the reference runs *both* whenever the call is close, and decides on what
//! the two engines actually produced.
//!
//! The rule is deliberately biased: wired wins by default, and wireless has to
//! earn the switch. The wired engine finds real ruling lines, so when it works it
//! is more faithful than a structure model's guess; but when the rules are broken
//! it collapses whole rows into one cell, and *that* is what these thresholds
//! detect — not "which is better" in the abstract, but "did wired obviously fail".
//!
//! Pure arithmetic over the two HTML strings and the OCR spans, owning no model:
//! the caller runs the engines, as it does for [`crate::orientation`].

use crate::OcrSpan;

/// The confidence at or above which a `Wireless` classification is taken at face
/// value. Below it, the wired engine runs too and [`select`] decides.
///
/// The reference gates on this in `batch_analyze.py:666-670`; a `Wired`
/// classification always runs both regardless of score.
pub const WIRELESS_TRUST_THRESHOLD: f32 = 0.9;

/// One engine's output, reduced to what the choice depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Counts {
    /// Physical cells: `<td`/`<th` occurrences, so a merged cell counts once.
    cells: usize,
    /// Cells with no text.
    blank: usize,
}

impl Counts {
    fn non_blank(&self) -> usize {
        self.cells.saturating_sub(self.blank)
    }
}

/// Counts `<td`/`<th` tags and how many of them are empty.
///
/// Deliberately a scan rather than a parse, mirroring `count_table_cells_physical`
/// (`main.py:255-264`) plus the reference's BeautifulSoup blank count: these
/// numbers only feed the comparison below, and the engines' own HTML is what is
/// returned, so a full parse would buy nothing but a dependency.
fn count(html: &str) -> Counts {
    let lower = html.to_lowercase();
    let mut counts = Counts::default();
    let mut rest = lower.as_str();

    // The *earliest* of the two tags: searching for one and falling back to the
    // other skips a `<th` that sits before a later `<td`.
    while let Some(open) = [rest.find("<td"), rest.find("<th")].into_iter().flatten().min() {
        counts.cells += 1;
        // Text is what sits between this tag's `>` and the next `<`.
        let after = &rest[open..];
        let Some(gt) = after.find('>') else { break };
        let body = &after[gt + 1..];
        let text = body.split('<').next().unwrap_or("");
        if text.trim().is_empty() {
            counts.blank += 1;
        }
        rest = &after[gt + 1..];
    }
    counts
}

/// How many of the OCR spans' texts appear in the HTML.
///
/// A substring test, as in the reference (`main.py:319-323`): a span whose text
/// never made it into the markup is a span the engine dropped, and the count of
/// those is the most direct evidence of an engine losing content.
fn text_hits(html: &str, spans: &[OcrSpan]) -> usize {
    spans
        .iter()
        .filter(|s| !s.text.trim().is_empty() && html.contains(s.text.as_str()))
        .count()
}

/// Which engine's HTML to keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    /// Keep the wired engine's output.
    Wired,
    /// Keep the wireless engine's output.
    Wireless,
}

/// Picks between the two recognitions of one table.
///
/// Wired is the default; each of the four conditions below is a way for wireless
/// to win, and they mirror `main.py:337-357` one-for-one:
///
/// 1. **Wireless finds a row or column's worth more content.** Estimating the
///    grid as square from the non-blank count, wireless must beat wired by about
///    two rows *or* two columns — enough that wired plainly dropped structure,
///    not just a stray cell.
/// 2. Cell counts are close but wired's is much smaller (`<= 75%`).
/// 3. Cell counts are equal and the table is tiny (`<= 4`), where the wired
///    engine has too little to go on.
/// 4. Wired placed far less OCR text (`<= 60%`) on a table with real text in it.
pub fn select(wired_html: &str, wireless_html: &str, spans: &[OcrSpan]) -> Choice {
    let wired = count(wired_html);
    let wireless = count(wireless_html);

    // Reference computes `wireless_len - wired_len` on unsigned counts; a wired
    // table with more cells makes it negative, which the `0 <= gap` test below
    // rejects. Signed here to keep that behaviour rather than wrap.
    let gap = wireless.cells as i64 - wired.cells as i64;

    let switch = wireless.non_blank() > wired.non_blank() && {
        // A square table of `n` non-blank cells has about `sqrt(n)` per side, so
        // `scale * 2` is two columns and `scale * (scale + 2)` is two extra rows.
        let scale = (wired.non_blank() as f64).sqrt().round() as usize;
        let plus_two_cols = wired.non_blank() + scale * 2;
        let plus_two_rows = scale * (scale + 2);
        wireless.non_blank() + 3 >= plus_two_cols.max(plus_two_rows)
    };

    let wired_text = text_hits(wired_html, spans);
    let wireless_text = text_hits(wireless_html, spans);

    let sparse = (0..=5).contains(&gap) && wired.cells <= (wireless.cells as f64 * 0.75).round() as usize;
    let tiny = gap == 0 && wired.cells <= 4;
    let text_poor = wired_text as f64 <= wireless_text as f64 * 0.6 && wireless_text >= 10;

    if switch || sparse || tiny || text_poor {
        Choice::Wireless
    } else {
        Choice::Wired
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mineru_types::BBox;

    fn cell(text: &str) -> String {
        format!("<td>{text}</td>")
    }

    /// Builds a table of `n` cells, the first `filled` of them carrying text.
    fn table(n: usize, filled: usize) -> String {
        let cells: String = (0..n)
            .map(|i| if i < filled { cell(&format!("c{i}")) } else { cell("") })
            .collect();
        format!("<table><tr>{cells}</tr></table>")
    }

    fn span(text: &str) -> OcrSpan {
        OcrSpan::new(BBox::new(0.0, 0.0, 1.0, 1.0), text, 1.0)
    }

    #[test]
    fn counts_physical_cells_and_blanks() {
        let c = count("<table><tr><td>a</td><td></td><th>h</th><td>   </td></tr></table>");
        assert_eq!(c.cells, 4);
        assert_eq!(c.blank, 2, "empty and whitespace-only cells are blank");
        assert_eq!(c.non_blank(), 2);
    }

    #[test]
    fn counts_a_merged_cell_once() {
        let c = count(r#"<table><tr><td colspan="3">wide</td></tr></table>"#);
        assert_eq!(c.cells, 1, "colspan is one physical cell");
        assert_eq!(c.blank, 0);
    }

    /// Wired is the default and must survive a tie: the whole rule is built to
    /// prefer real ruling lines unless wired obviously failed.
    #[test]
    fn keeps_wired_when_the_two_agree() {
        let html = table(20, 20);
        assert_eq!(select(&html, &html, &[]), Choice::Wired);
    }

    #[test]
    fn switches_when_wireless_finds_two_more_rows_of_content() {
        // 5x5-ish wired (25 non-blank) vs a wireless that finds ~2 more rows.
        let wired = table(25, 25);
        let wireless = table(35, 35);
        assert_eq!(select(&wired, &wireless, &[]), Choice::Wireless);
    }

    /// The switch rule must not fire on a small edge: one extra cell is noise,
    /// not a dropped row.
    #[test]
    fn does_not_switch_for_a_single_extra_cell() {
        let wired = table(25, 25);
        let wireless = table(26, 26);
        assert_eq!(select(&wired, &wireless, &[]), Choice::Wired);
    }

    /// Pins the `max` in the switch rule: wireless must clear *both* the
    /// two-column and the two-row bar, not the easier of them.
    ///
    /// At 3 non-blank wired cells the two bars diverge (7 vs 8), and wireless's 4
    /// clears the lower but not the higher — so `min` would switch here and `max`
    /// must not. Both tables are padded to 20 cells so that `sparse` and `tiny`,
    /// which key off total cell counts, stay quiet and `switch` alone decides.
    #[test]
    fn switch_requires_clearing_the_harder_of_the_two_bars() {
        let wired = table(20, 3);
        let wireless = table(20, 4);
        assert_eq!(select(&wired, &wireless, &[]), Choice::Wired);
    }

    #[test]
    fn switches_when_wired_is_much_sparser_at_a_similar_size() {
        // gap of 3 (in 0..=5) and wired's 9 <= 75% of 12.
        let wired = table(9, 1);
        let wireless = table(12, 1);
        assert_eq!(select(&wired, &wireless, &[]), Choice::Wireless);
    }

    #[test]
    fn switches_on_a_tiny_table_with_equal_cell_counts() {
        let wired = table(4, 0);
        let wireless = table(4, 0);
        assert_eq!(select(&wired, &wireless, &[]), Choice::Wireless);
    }

    /// Isolates the text rule: same cell count and same non-blank count on both
    /// sides, so `switch`/`sparse`/`tiny` cannot fire and only *which* text landed
    /// in the cells can decide. Without this the fixture proves nothing — a
    /// wired table that is merely smaller trips `switch` on its own, and the test
    /// passes with the text rule deleted.
    #[test]
    fn switches_when_wired_placed_far_less_of_the_ocr_text() {
        let spans: Vec<OcrSpan> = (0..12).map(|i| span(&format!("t{i}"))).collect();
        // 25 filled cells each; the first `hits` carry span text, the rest filler.
        let with_hits = |hits: usize| -> String {
            let cells: String = (0..25)
                .map(|i| {
                    if i < hits {
                        cell(&format!("t{i}"))
                    } else {
                        cell("filler")
                    }
                })
                .collect();
            format!("<table><tr>{cells}</tr></table>")
        };
        let wireless = with_hits(12);
        let wired = with_hits(2);

        assert_eq!(select(&wired, &wireless, &spans), Choice::Wireless);
    }

    /// The text rule needs a table with real text: below `>= 10` placed spans the
    /// ratio is noise, so wired keeps a table that the rule would otherwise flip.
    ///
    /// The fixture has to be big enough that the *other* three rules stay quiet —
    /// on a tiny table `switch` and `tiny` both fire on their own and the text
    /// rule is never the reason for the answer.
    #[test]
    fn text_rule_ignores_tables_with_little_text() {
        // 9 spans: under the reference's `wireless_text_count >= 10` guard.
        let spans: Vec<OcrSpan> = (0..9).map(|i| span(&format!("t{i}"))).collect();
        let filled = |n: usize| -> String {
            let cells: String = (0..25)
                .map(|i| if i < n { cell(&format!("t{i}")) } else { cell("x") })
                .collect();
            format!("<table><tr>{cells}</tr></table>")
        };
        // Equal cells, none blank — only the text rule can fire, and it must not
        // because wireless placed just 9 spans.
        let wireless = filled(9);
        let wired = filled(2);

        assert_eq!(select(&wired, &wireless, &spans), Choice::Wired);
    }

    #[test]
    fn empty_wired_output_loses_to_a_real_wireless_one() {
        assert_eq!(select("", &table(12, 12), &[]), Choice::Wireless);
    }

    /// A wired table with *more* cells makes the reference's unsigned gap
    /// negative; the sparse rule must not fire on it.
    #[test]
    fn wired_with_more_cells_is_kept() {
        let wired = table(30, 30);
        let wireless = table(20, 20);
        assert_eq!(select(&wired, &wireless, &[]), Choice::Wired);
    }
}
