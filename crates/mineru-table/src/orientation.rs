//! Table orientation detection (0°/90°/270°).
//!
//! Port of `MineruTableOrientationClsModel` (`mineru_table_ori_cls.py`). Tables
//! are sometimes typeset sideways on the page — a wide table rotated to fit a
//! portrait column. The crop is then upright-but-sideways: OCR still finds text
//! lines, still recognizes them, and returns confident-looking nonsense
//! (`Forest age (years)` read column-wise becomes `20 20 20 20 20`), so nothing
//! downstream can tell the table apart from a badly-printed one.
//!
//! Unlike [`crate::cls`], this classifier has **no model of its own**. It decides
//! by asking OCR to read the crop at each candidate angle and keeping the angle
//! OCR is most confident about. That keeps the policy here — pure, deterministic,
//! testable — while the caller supplies the OCR, which lives outside this crate.
//!
//! The two-stage shape mirrors the reference: [`is_rotation_candidate`] is a
//! cheap gate over detection-box shapes (most tables are upright, and scoring all
//! three angles costs two extra OCR passes), and [`select_rotation`] is the
//! decision over the scores the caller collects.

use mineru_types::BBox;

/// Aspect ratio (w/h) below which a detection box counts as "tall and narrow",
/// i.e. plausibly a sideways-read text line.
const ROTATED_TEXT_ASPECT_RATIO_THRESHOLD: f32 = 0.8;
/// Fraction of boxes that must look rotated before a crop is a candidate.
const ROTATED_TEXT_RATIO_THRESHOLD: f32 = 0.28;
/// Minimum number of rotated-looking boxes; guards against tiny tables where a
/// couple of stray tall boxes would trip the ratio.
const ROTATED_TEXT_MIN_BOXES: usize = 3;
/// Cap on boxes sampled per angle when scoring, bounding the OCR cost.
pub const ORIENTATION_SCORE_MAX_SAMPLE_BOXES: usize = 18;
/// Minimum recognized spans for an angle's score to count at all.
const ORIENTATION_SCORE_MIN_VALID_RESULTS: usize = 5;
/// Score at 0° above which the crop is taken as upright without comparing.
const ORIENTATION_ZERO_SCORE_PRIORITY_THRESHOLD: f32 = 0.9;
/// A rotated angle must beat 0° by more than this to be chosen.
const ORIENTATION_SCORE_TIE_THRESHOLD: f32 = 0.08;

/// A candidate orientation for a table crop.
///
/// Only these three are considered: a table rotated 180° reads as upside-down
/// text, which OCR scores poorly at every angle, and the reference does not
/// handle it either.
///
/// The variants name the **correction to apply**, not the page's rotation, and
/// deliberately avoid "clockwise"/"counter-clockwise": the reference's labels
/// inconveniently invert (its `"90"` applies `ROTATE_90_COUNTERCLOCKWISE`, its
/// `"270"` applies `ROTATE_90_CLOCKWISE`), and the two conventions are easy to
/// swap silently — a swap yields a 180°-wrong crop that OCR reads as plausible
/// garbage rather than an error. [`crate::orientation`]'s tests pin the mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rotation {
    /// Upright; the crop is used as-is.
    None,
    /// Turn the crop a quarter-turn so its left edge becomes the top edge.
    ///
    /// Corrects a table whose text reads bottom-to-top. Equivalent to OpenCV's
    /// `ROTATE_90_CLOCKWISE` and the reference's `"270"` label.
    LeftEdgeToTop,
    /// Turn the crop a quarter-turn so its right edge becomes the top edge.
    ///
    /// Corrects a table whose text reads top-to-bottom. Equivalent to OpenCV's
    /// `ROTATE_90_COUNTERCLOCKWISE` and the reference's `"90"` label.
    RightEdgeToTop,
}

impl Rotation {
    /// Every candidate, in the reference's scoring order (`("0", "90", "270")`).
    pub const ALL: [Rotation; 3] = [
        Rotation::None,
        Rotation::RightEdgeToTop,
        Rotation::LeftEdgeToTop,
    ];
}

/// How well OCR read a crop at one candidate angle.
///
/// Ordering is by confidence first, then span count, then character count —
/// mirroring the reference's `max` over the `(score, valid_count, char_count)`
/// tuple, which breaks confidence ties toward the angle that simply read *more*.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrientationScore {
    /// Mean recognition confidence over the recognized spans, or `0.0` when
    /// fewer than [`ORIENTATION_SCORE_MIN_VALID_RESULTS`] spans were read.
    pub confidence: f32,
    /// How many sampled spans produced non-empty text.
    pub valid_count: usize,
    /// Total characters across those spans.
    pub char_count: usize,
}

impl OrientationScore {
    /// The score of an angle nothing was read at.
    pub const ZERO: Self = Self {
        confidence: 0.0,
        valid_count: 0,
        char_count: 0,
    };

    /// Scores one angle from the text/confidence pairs OCR returned for it.
    ///
    /// Empty and whitespace-only texts are ignored. An angle with too few
    /// readable spans scores `0.0` confidence (while keeping its counts): a
    /// sideways crop often yields a handful of high-confidence digits, and
    /// without this floor those would outscore a correct angle that read a whole
    /// table of words.
    pub fn from_reads<'a>(reads: impl IntoIterator<Item = (&'a str, f32)>) -> Self {
        let mut confidences = Vec::new();
        let mut char_count = 0;
        for (text, score) in reads {
            if text.trim().is_empty() {
                continue;
            }
            confidences.push(score);
            char_count += text.chars().count();
        }
        let valid_count = confidences.len();
        if valid_count < ORIENTATION_SCORE_MIN_VALID_RESULTS {
            return Self {
                confidence: 0.0,
                valid_count,
                char_count,
            };
        }
        let mean = confidences.iter().sum::<f32>() / valid_count as f32;
        Self {
            confidence: mean,
            valid_count,
            char_count,
        }
    }

    /// Orders two scores the way the reference's tuple `max` does.
    ///
    /// `f32` is only `PartialOrd`, and a NaN confidence (possible only from a
    /// malformed recognizer score) must not silently order as "greater"; it
    /// compares as `Equal` here and so loses to an existing best.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.confidence
            .partial_cmp(&other.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(self.valid_count.cmp(&other.valid_count))
            .then(self.char_count.cmp(&other.char_count))
    }
}

/// Whether `det_boxes` look like a sideways table worth scoring at every angle.
///
/// Upright text lines are wider than they are tall; a crop where a meaningful
/// share of boxes are tall and narrow is a rotation candidate. This is only a
/// gate — it over-admits deliberately (a column of short cells looks tall too),
/// and [`select_rotation`] makes the real decision.
pub fn is_rotation_candidate(det_boxes: &[BBox]) -> bool {
    if det_boxes.is_empty() {
        return false;
    }
    let vertical = det_boxes
        .iter()
        .filter(|b| {
            let (w, h) = (b.x1 - b.x0, b.y1 - b.y0);
            // A degenerate (zero-height) box has no meaningful ratio; the
            // reference treats it as 1.0, i.e. not rotated.
            let ratio = if h > 0.0 { w / h } else { 1.0 };
            ratio < ROTATED_TEXT_ASPECT_RATIO_THRESHOLD
        })
        .count();
    vertical >= ROTATED_TEXT_MIN_BOXES
        && vertical as f32 >= det_boxes.len() as f32 * ROTATED_TEXT_RATIO_THRESHOLD
}

/// Evenly samples up to [`ORIENTATION_SCORE_MAX_SAMPLE_BOXES`] boxes.
///
/// Scoring reads a subset rather than the whole table: enough to judge an angle,
/// bounded so a dense table does not cost three full OCR passes. Sampling is
/// spread across the table (not the first N) so a mixed-content table is judged
/// on more than its header.
pub fn sample_boxes(det_boxes: &[BBox]) -> Vec<BBox> {
    if det_boxes.len() <= ORIENTATION_SCORE_MAX_SAMPLE_BOXES {
        return det_boxes.to_vec();
    }
    // `linspace(0, n-1, k)` then dedup, matching the reference's rounding.
    let n = det_boxes.len();
    let k = ORIENTATION_SCORE_MAX_SAMPLE_BOXES;
    let mut picked: Vec<usize> = (0..k)
        .map(|i| {
            let t = i as f32 / (k - 1) as f32;
            (t * (n - 1) as f32).round() as usize
        })
        .collect();
    picked.dedup();
    picked.into_iter().filter_map(|i| det_boxes.get(i).copied()).collect()
}

/// Picks the orientation to apply, given each candidate's score.
///
/// Biased toward leaving the crop alone: a confident 0° short-circuits, and a
/// rotated angle must beat 0° by more than [`ORIENTATION_SCORE_TIE_THRESHOLD`].
/// Most tables are upright, and wrongly rotating one destroys a table that would
/// otherwise have read correctly, so ties resolve to no-op.
pub fn select_rotation(scores: &[(Rotation, OrientationScore)]) -> Rotation {
    let score_of = |r: Rotation| {
        scores
            .iter()
            .find(|(candidate, _)| *candidate == r)
            .map(|(_, s)| *s)
            .unwrap_or(OrientationScore::ZERO)
    };

    let zero = score_of(Rotation::None);
    if zero.confidence >= ORIENTATION_ZERO_SCORE_PRIORITY_THRESHOLD {
        return Rotation::None;
    }

    let Some(best) = Rotation::ALL
        .iter()
        .copied()
        .max_by(|a, b| score_of(*a).cmp(&score_of(*b)))
    else {
        return Rotation::None;
    };

    if best != Rotation::None && score_of(best).confidence - zero.confidence < ORIENTATION_SCORE_TIE_THRESHOLD
    {
        return Rotation::None;
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wide(n: usize) -> Vec<BBox> {
        (0..n)
            .map(|i| BBox::new(0.0, i as f32 * 10.0, 100.0, i as f32 * 10.0 + 8.0))
            .collect()
    }

    fn tall(n: usize) -> Vec<BBox> {
        (0..n)
            .map(|i| BBox::new(i as f32 * 10.0, 0.0, i as f32 * 10.0 + 8.0, 100.0))
            .collect()
    }

    #[test]
    fn upright_text_is_not_a_candidate() {
        assert!(!is_rotation_candidate(&wide(10)));
    }

    #[test]
    fn sideways_text_is_a_candidate() {
        assert!(is_rotation_candidate(&tall(10)));
    }

    #[test]
    fn no_boxes_is_not_a_candidate() {
        assert!(!is_rotation_candidate(&[]));
    }

    #[test]
    fn a_couple_of_tall_boxes_do_not_trip_the_gate() {
        // 2 tall out of 20 clears neither the ratio nor the minimum count.
        let mut boxes = wide(18);
        boxes.extend(tall(2));
        assert!(!is_rotation_candidate(&boxes));
    }

    #[test]
    fn confident_zero_short_circuits() {
        // Even with a better 90°, a confident 0° wins outright.
        let scores = [
            (Rotation::None, OrientationScore { confidence: 0.95, valid_count: 10, char_count: 50 }),
            (Rotation::RightEdgeToTop, OrientationScore { confidence: 0.99, valid_count: 10, char_count: 50 }),
        ];
        assert_eq!(select_rotation(&scores), Rotation::None);
    }

    #[test]
    fn clearly_better_rotation_wins() {
        // The page-4 case: noise at 0°, clean text at 90°.
        let scores = [
            (Rotation::None, OrientationScore { confidence: 0.66, valid_count: 20, char_count: 40 }),
            (Rotation::RightEdgeToTop, OrientationScore { confidence: 0.99, valid_count: 20, char_count: 90 }),
            (Rotation::LeftEdgeToTop, OrientationScore { confidence: 0.70, valid_count: 18, char_count: 45 }),
        ];
        assert_eq!(select_rotation(&scores), Rotation::RightEdgeToTop);
    }

    #[test]
    fn marginal_win_keeps_zero() {
        // 0.85 vs 0.80 is inside the tie threshold: don't rotate on a hunch.
        let scores = [
            (Rotation::None, OrientationScore { confidence: 0.80, valid_count: 10, char_count: 50 }),
            (Rotation::RightEdgeToTop, OrientationScore { confidence: 0.85, valid_count: 10, char_count: 50 }),
        ];
        assert_eq!(select_rotation(&scores), Rotation::None);
    }

    #[test]
    fn no_scores_leaves_crop_alone() {
        assert_eq!(select_rotation(&[]), Rotation::None);
    }

    #[test]
    fn too_few_reads_score_zero_confidence() {
        // 4 spans is under the minimum, so confidence floors to 0 even though
        // each read was perfect — the guard against a sideways crop winning on a
        // few stray digits.
        let s = OrientationScore::from_reads([("1", 1.0), ("2", 1.0), ("3", 1.0), ("4", 1.0)]);
        assert_eq!(s.confidence, 0.0);
        assert_eq!(s.valid_count, 4);
    }

    #[test]
    fn blank_reads_are_ignored() {
        let s = OrientationScore::from_reads([
            ("a", 1.0), ("  ", 0.9), ("b", 1.0), ("", 0.9), ("c", 1.0), ("d", 1.0), ("e", 1.0),
        ]);
        assert_eq!(s.valid_count, 5);
        assert_eq!(s.char_count, 5);
        assert_eq!(s.confidence, 1.0);
    }

    #[test]
    fn score_ties_break_on_count_then_chars() {
        let fewer = OrientationScore { confidence: 0.9, valid_count: 5, char_count: 10 };
        let more = OrientationScore { confidence: 0.9, valid_count: 9, char_count: 10 };
        let longer = OrientationScore { confidence: 0.9, valid_count: 9, char_count: 20 };
        assert_eq!(fewer.cmp(&more), std::cmp::Ordering::Less);
        assert_eq!(more.cmp(&longer), std::cmp::Ordering::Less);
    }

    #[test]
    fn sampling_is_capped_and_spread() {
        let boxes = wide(100);
        let s = sample_boxes(&boxes);
        assert_eq!(s.len(), ORIENTATION_SCORE_MAX_SAMPLE_BOXES);
        // Spans the whole table, not just the top.
        assert_eq!(s[0], boxes[0]);
        assert_eq!(s[s.len() - 1], boxes[99]);
    }

    #[test]
    fn sampling_passes_small_inputs_through() {
        let boxes = wide(5);
        assert_eq!(sample_boxes(&boxes), boxes);
    }
}
