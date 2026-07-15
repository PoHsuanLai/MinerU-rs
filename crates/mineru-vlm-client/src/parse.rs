//! Parser for the VLM's layout-detection response.
//!
//! The model returns layout as a flat string of special-token-delimited blocks
//! rather than JSON:
//!
//! ```text
//! <|box_start|>x1 y1 x2 y2<|box_end|><|ref_start|>type<|ref_end|>[<|rotate_up|>]<tail>
//! ```
//!
//! This is a direct port of the reference `mineru_vl_utils` regex
//! (`parse_layout_output`): one pattern matched repeatedly over the whole string,
//! with capture groups for the four box coordinates, the ref type, an optional
//! rotate token, and the trailing text up to the next block. Coordinates are
//! integers in `0..=1000`, validated/corner-ordered and rescaled to normalized
//! `0.0..=1.0`. Matching Python exactly here matters: the hand-split predecessor
//! could silently drop a block when anything sat between the box and its ref.

use once_cell::sync::Lazy;
use regex::Regex;

use crate::raw::VlmBlock;

/// The layout-block pattern: `<|box_start|>(d) (d) (d) (d)<|box_end|>`
/// `<|ref_start|>(type)<|ref_end|>(rotate?)`.
///
/// This mirrors the reference `_layout_re` for the parts we consume — the four
/// box ints, the ref type, and the optional rotate token. The reference has a
/// trailing `(.*?)(?=<|box_start|>|$)` group that captures inter-block text for a
/// `merge_prev` flag we don't use; the Rust `regex` crate has no look-around, and
/// dropping that group is equivalent for our purposes: [`Regex::captures_iter`]
/// resumes after each match and finds the next block regardless of the prose
/// between them. Matching each block independently is what fixes the predecessor's
/// bug (it dropped a block when anything sat between the box and its ref).
static LAYOUT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concat!(
        r"<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
        r"<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>",
        r"(?:(<\|rotate_(?:up|right|down|left)\|>))?",
    ))
    // The pattern is a compile-time constant and known valid; on the impossible
    // event of a build-time regex-syntax error, fall back to a never-matching
    // pattern rather than panicking in library code.
    .unwrap_or_else(|_| Regex::new("$.^").expect("trivial never-match regex is valid"))
});

/// Parses a full layout response into blocks, skipping malformed entries.
///
/// Mirrors the reference `parse_layout_output`: for each regex match, converts the
/// box, lower-cases the ref type, remaps `unknown` → `image`, drops
/// `inline_formula` (folded elsewhere), and reads the angle from the optional
/// rotate token. Blocks with an out-of-range or degenerate box are skipped.
pub fn parse_layout(response: &str) -> Vec<VlmBlock> {
    let mut blocks = Vec::new();

    for caps in LAYOUT_RE.captures_iter(response) {
        // Groups 1..=4 are the box ints; the pattern only matches on `\d+`, so a
        // parse failure here is impossible, but stay panic-free regardless.
        let coords: Option<[i32; 4]> = (|| {
            Some([
                caps.get(1)?.as_str().parse().ok()?,
                caps.get(2)?.as_str().parse().ok()?,
                caps.get(3)?.as_str().parse().ok()?,
                caps.get(4)?.as_str().parse().ok()?,
            ])
        })();
        let Some(coords) = coords else { continue };
        let Some(bbox) = convert_bbox(coords) else {
            continue; // out of range or degenerate — skip (matches Python).
        };

        let Some(ref_raw) = caps.get(5) else { continue };
        let mut label = ref_raw.as_str().to_ascii_lowercase();
        // Python: `unknown` -> `image`; `inline_formula` blocks are dropped here
        // (they are folded into surrounding text downstream, not laid out).
        if label == "unknown" {
            label = "image".to_owned();
        }
        if label == "inline_formula" {
            continue;
        }

        let angle = caps
            .get(6)
            .map(|m| angle_from_token(m.as_str()))
            .unwrap_or(0);

        blocks.push(VlmBlock {
            bbox,
            label,
            content: None,
            angle,
            sub_type: None,
            image_ref: None,
        });
    }

    blocks
}

/// Ports `_convert_bbox`: reject any coord outside `0..=1000`, corner-order the
/// box, reject degenerate (zero-area) boxes, and rescale to normalized `0.0..=1.0`.
fn convert_bbox([a, b, c, d]: [i32; 4]) -> Option<[f32; 4]> {
    if [a, b, c, d].iter().any(|&n| !(0..=1000).contains(&n)) {
        return None;
    }
    let (x0, x1) = (a.min(c), a.max(c));
    let (y0, y1) = (b.min(d), b.max(d));
    if x0 == x1 || y0 == y1 {
        return None;
    }
    Some([
        x0 as f32 / 1000.0,
        y0 as f32 / 1000.0,
        x1 as f32 / 1000.0,
        y1 as f32 / 1000.0,
    ])
}

/// Maps a `<|rotate_*|>` token to its angle (`up`=0, `right`=90, `down`=180,
/// `left`=270), matching `ANGLE_MAPPING`.
fn angle_from_token(token: &str) -> i32 {
    match token {
        "<|rotate_right|>" => 90,
        "<|rotate_down|>" => 180,
        "<|rotate_left|>" => 270,
        _ => 0, // includes `<|rotate_up|>`
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_block() {
        let resp = "<|box_start|>12 34 560 780<|box_end|><|ref_start|>text<|ref_end|>";
        let blocks = parse_layout(resp);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].label, "text");
        assert_eq!(blocks[0].bbox, [0.012, 0.034, 0.560, 0.780]);
        assert_eq!(blocks[0].angle, 0);
    }

    #[test]
    fn parses_multiple_blocks_and_rotation() {
        let resp = "<|box_start|>0 0 500 100<|box_end|><|ref_start|>title<|ref_end|>\
                    <|box_start|>0 200 500 900<|box_end|><|ref_start|>table<|ref_end|><|rotate_right|>";
        let blocks = parse_layout(resp);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].label, "title");
        assert_eq!(blocks[1].label, "table");
        assert_eq!(blocks[1].angle, 90);
    }

    #[test]
    fn rejects_out_of_range_and_degenerate() {
        // out of range
        assert!(parse_layout("<|box_start|>0 0 1200 100<|box_end|><|ref_start|>text<|ref_end|>").is_empty());
        // degenerate (x0 == x1)
        assert!(parse_layout("<|box_start|>50 0 50 100<|box_end|><|ref_start|>text<|ref_end|>").is_empty());
    }

    #[test]
    fn orders_swapped_corners() {
        let blocks = parse_layout("<|box_start|>500 800 100 200<|box_end|><|ref_start|>text<|ref_end|>");
        assert_eq!(blocks[0].bbox, [0.1, 0.2, 0.5, 0.8]);
    }

    #[test]
    fn skips_inter_block_text_and_multiline_tails() {
        // Real responses interleave prose/newlines between blocks; the DOTALL tail
        // must consume it so the next block still matches (the hand-split parser's
        // failure mode was dropping blocks after any inter-block text).
        let resp = "<|box_start|>0 0 500 100<|box_end|><|ref_start|>title<|ref_end|>\n\
                    some trailing description across\nmultiple lines\n\
                    <|box_start|>0 200 500 900<|box_end|><|ref_start|>text<|ref_end|>";
        let blocks = parse_layout(resp);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].label, "title");
        assert_eq!(blocks[1].label, "text");
    }

    #[test]
    fn remaps_unknown_to_image_and_drops_inline_formula() {
        let resp = "<|box_start|>0 0 500 100<|box_end|><|ref_start|>unknown<|ref_end|>\
                    <|box_start|>0 200 500 300<|box_end|><|ref_start|>inline_formula<|ref_end|>\
                    <|box_start|>0 400 500 500<|box_end|><|ref_start|>Text<|ref_end|>";
        let blocks = parse_layout(resp);
        assert_eq!(blocks.len(), 2, "inline_formula dropped, unknown kept");
        assert_eq!(blocks[0].label, "image", "unknown remapped to image");
        assert_eq!(blocks[1].label, "text", "ref type lower-cased");
    }
}
