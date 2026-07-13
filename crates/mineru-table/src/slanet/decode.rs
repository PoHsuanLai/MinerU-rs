//! Structure-token decoding for SLANet-plus.
//!
//! Port of `TableLabelDecode.decode` (the single-batch inference path). Given the
//! decoder's per-step class probabilities and regressed boxes, it produces the
//! structure token list (wrapped in the `<html><body><table>` frame, matching
//! `TableStructurer.process`) and one cell box per `<td>` token.

use mineru_types::BBox;

use super::vocab::Vocab;

/// Decoded structure output: the HTML-skeleton token stream and the per-`<td>`
/// cell boxes, aligned by emission order.
#[derive(Debug, Clone)]
pub struct StructureResult {
    /// Structure tokens, wrapped in `<html><body><table> ... </table></body></html>`.
    pub tokens: Vec<String>,
    /// One box per `<td>`-like token, in the order those tokens appear.
    pub cell_bboxes: Vec<BBox>,
    /// Mean class probability across the decoded (non-ignored) steps.
    pub score: f32,
}

/// Per-step decoder outputs for a single table.
///
/// * `structure_probs` is `[L, C]` (length × class channels).
/// * `loc_preds` is `[L, 4]` normalized box coordinates.
pub struct RawPreds<'a> {
    /// Flattened `[L, C]` class probabilities, row-major.
    pub structure_probs: &'a [f32],
    /// Number of decode steps `L`.
    pub len: usize,
    /// Number of class channels `C`.
    pub num_classes: usize,
    /// Flattened `[L, 4]` box regressions, row-major.
    pub loc_preds: &'a [f32],
}

/// Decodes the raw model outputs into structure tokens and cell boxes.
///
/// `orig_w`/`orig_h` are the *original* crop dimensions; the regressed boxes are
/// normalized and get scaled by these (Python `_bbox_decode` uses `shape[:2]`,
/// i.e. the source height/width). The caller applies the SLANet-plus ratio
/// correction afterwards (see [`super::adapt_slanet_plus`]).
pub fn decode(preds: &RawPreds, vocab: &Vocab, orig_w: f32, orig_h: f32) -> StructureResult {
    let mut tokens: Vec<String> = Vec::new();
    let mut cell_bboxes: Vec<BBox> = Vec::new();
    let mut score_sum = 0.0f32;
    let mut score_n = 0usize;

    for step in 0..preds.len {
        let base = step * preds.num_classes;
        let row = &preds.structure_probs[base..base + preds.num_classes];
        let (char_idx, max_prob) = argmax(row);

        // Break at eos (but never on the very first step).
        if step > 0 && char_idx == vocab.end_idx {
            break;
        }
        if vocab.is_ignored(char_idx) {
            continue;
        }

        if vocab.is_td(char_idx) {
            let lb = step * 4;
            let bbox = decode_bbox(&preds.loc_preds[lb..lb + 4], orig_w, orig_h);
            cell_bboxes.push(bbox);
        }

        if let Some(tok) = vocab.tokens.get(char_idx) {
            tokens.push(tok.clone());
        }
        score_sum += max_prob;
        score_n += 1;
    }

    // Wrap with the HTML frame, matching TableStructurer.process.
    let mut wrapped = vec![
        "<html>".to_string(),
        "<body>".to_string(),
        "<table>".to_string(),
    ];
    wrapped.extend(tokens);
    wrapped.extend([
        "</table>".to_string(),
        "</body>".to_string(),
        "</html>".to_string(),
    ]);

    let score = if score_n > 0 {
        score_sum / score_n as f32
    } else {
        0.0
    };
    StructureResult {
        tokens: wrapped,
        cell_bboxes,
        score,
    }
}

/// Argmax over a probability row, returning `(index, value)`.
fn argmax(row: &[f32]) -> (usize, f32) {
    let mut best_idx = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best {
            best = v;
            best_idx = i;
        }
    }
    (best_idx, best)
}

/// Scales a normalized `[x0, y0, x1, y1]` box by the crop dimensions.
///
/// Mirrors `_bbox_decode`: even indices (x) × width, odd indices (y) × height.
fn decode_bbox(loc: &[f32], w: f32, h: f32) -> BBox {
    BBox::new(loc[0] * w, loc[1] * h, loc[2] * w, loc[3] * h)
}

#[cfg(test)]
mod tests {
    use super::super::vocab::build_vocab;
    use super::*;

    /// Builds a one-hot `[L, C]` prob buffer from a list of token indices.
    fn one_hot(indices: &[usize], num_classes: usize) -> Vec<f32> {
        let mut buf = vec![0.0f32; indices.len() * num_classes];
        for (step, &idx) in indices.iter().enumerate() {
            buf[step * num_classes + idx] = 1.0;
        }
        buf
    }

    #[test]
    fn decodes_tokens_and_breaks_on_eos() {
        let vocab = build_vocab();
        let c = vocab.tokens.len();
        let tr = vocab.tokens.iter().position(|t| t == "<tr>").unwrap();
        let td = vocab.tokens.iter().position(|t| t == "<td></td>").unwrap();
        let etr = vocab.tokens.iter().position(|t| t == "</tr>").unwrap();

        // Sequence: <tr> <td></td> </tr> eos (extra tokens after eos ignored).
        let seq = [tr, td, etr, vocab.end_idx, tr];
        let probs = one_hot(&seq, c);
        let locs = vec![0.5f32; seq.len() * 4];

        let preds = RawPreds {
            structure_probs: &probs,
            len: seq.len(),
            num_classes: c,
            loc_preds: &locs,
        };
        let out = decode(&preds, &vocab, 100.0, 200.0);

        assert_eq!(
            out.tokens,
            vec![
                "<html>", "<body>", "<table>", "<tr>", "<td></td>", "</tr>", "</table>",
                "</body>", "</html>"
            ]
        );
        // One <td></td> => one cell box, scaled: 0.5*100=50, 0.5*200=100.
        assert_eq!(out.cell_bboxes.len(), 1);
        let b = out.cell_bboxes[0];
        assert_eq!((b.x0, b.y0, b.x1, b.y1), (50.0, 100.0, 50.0, 100.0));
    }

    #[test]
    fn ignores_leading_sentinel() {
        let vocab = build_vocab();
        let c = vocab.tokens.len();
        let td = vocab.tokens.iter().position(|t| t == "<td></td>").unwrap();
        // Step 0 is beg sentinel (ignored, not a break since it's not eos).
        let seq = [vocab.beg_idx, td];
        let probs = one_hot(&seq, c);
        let locs = vec![0.25f32; seq.len() * 4];
        let preds = RawPreds {
            structure_probs: &probs,
            len: seq.len(),
            num_classes: c,
            loc_preds: &locs,
        };
        let out = decode(&preds, &vocab, 40.0, 40.0);
        assert!(out.tokens.contains(&"<td></td>".to_string()));
        assert_eq!(out.cell_bboxes.len(), 1);
    }
}
