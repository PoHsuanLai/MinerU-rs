//! Parser for the VLM's layout-detection response.
//!
//! The model returns layout as a flat string of special-token-delimited blocks
//! rather than JSON:
//!
//! ```text
//! <|box_start|>x1 y1 x2 y2<|box_end|><|ref_start|>type<|ref_end|>[<|rotate_up|>]
//! ```
//!
//! Coordinates are integers in `0..=1000`; this module validates and rescales them
//! to normalized `0.0..=1.0` and yields [`VlmBlock`]s ready for assembly.

use crate::raw::VlmBlock;

const BOX_START: &str = "<|box_start|>";
const BOX_END: &str = "<|box_end|>";
const REF_START: &str = "<|ref_start|>";
const REF_END: &str = "<|ref_end|>";

/// Parses a full layout response into blocks, skipping malformed entries.
///
/// The `angle` of each block is read from a trailing `<|rotate_*|>` marker when
/// present (`up`=0, `right`=90, `down`=180, `left`=270), defaulting to 0.
pub fn parse_layout(response: &str) -> Vec<VlmBlock> {
    let mut blocks = Vec::new();
    let mut rest = response;

    while let Some(start) = rest.find(BOX_START) {
        rest = &rest[start + BOX_START.len()..];
        let Some(box_end) = rest.find(BOX_END) else {
            break;
        };
        let coords_str = &rest[..box_end];
        rest = &rest[box_end + BOX_END.len()..];

        // The ref (type) should immediately follow the box.
        let Some((label, after_ref)) = parse_ref(rest) else {
            continue;
        };
        rest = after_ref;

        let Some(bbox) = parse_bbox(coords_str) else {
            continue;
        };
        let angle = parse_leading_angle(rest);

        blocks.push(VlmBlock {
            bbox,
            label,
            content: None,
            angle,
            sub_type: None,
        });
    }

    blocks
}

/// Parses `<|ref_start|>type<|ref_end|>` at the start of `s` (after optional
/// whitespace), returning the type and the remainder.
fn parse_ref(s: &str) -> Option<(String, &str)> {
    let s = s.trim_start();
    let after_start = s.strip_prefix(REF_START)?;
    let end = after_start.find(REF_END)?;
    let label = after_start[..end].trim().to_owned();
    Some((label, &after_start[end + REF_END.len()..]))
}

/// Parses four whitespace-separated integers in `0..=1000` into a normalized,
/// corner-ordered `[x0, y0, x1, y1]`. Returns `None` for out-of-range or
/// degenerate boxes (matching the Python `_convert_bbox`).
fn parse_bbox(s: &str) -> Option<[f32; 4]> {
    let nums: Vec<i32> = s.split_whitespace().filter_map(|t| t.parse().ok()).collect();
    let [a, b, c, d] = nums[..] else {
        return None;
    };
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

/// Reads a leading `<|rotate_*|>` marker into an angle, defaulting to 0.
fn parse_leading_angle(s: &str) -> i32 {
    let s = s.trim_start();
    for (marker, angle) in [
        ("<|rotate_up|>", 0),
        ("<|rotate_right|>", 90),
        ("<|rotate_down|>", 180),
        ("<|rotate_left|>", 270),
    ] {
        if s.starts_with(marker) {
            return angle;
        }
    }
    0
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
}
