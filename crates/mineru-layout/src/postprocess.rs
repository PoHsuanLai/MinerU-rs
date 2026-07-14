//! Detection postprocessing: cxcywh→xyxy, topk over sigmoid logits, per-query
//! reading-order ranking, and final filtering to [`LayoutDet`]s.
//!
//! Port of `_post_process_object_detection` + `_get_order_seqs` + `_parse_prediction`
//! from the reference. The numeric core is expressed on plain slices so it is unit-
//! testable without a Burn backend or weights.

use burn::prelude::Backend;
use burn::tensor::Tensor;
use mineru_types::BBox;

use crate::detection::LayoutDet;
use crate::error::{Error, Result};
use crate::label::LayoutLabel;

/// Final confidence threshold applied after the topk (the reference's `self.conf`).
pub const DEFAULT_CONF: f32 = 0.45;

/// Converts one center-form box `[cx, cy, w, h]` to corner form `[x0, y0, x1, y1]`.
///
/// Pure helper mirroring `torch.cat([c - 0.5*d, c + 0.5*d])`.
pub fn cxcywh_to_xyxy(b: [f32; 4]) -> [f32; 4] {
    let [cx, cy, w, h] = b;
    [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]
}

/// A flattened-topk selection over per-query sigmoid class scores.
///
/// `logits` is row-major `[num_queries, num_classes]`. Returns up to
/// `num_queries` `(query_index, class_id, score)` triples, highest score first —
/// matching `torch.topk(sigmoid(logits).flatten(1), num_top_queries)` with
/// `label = index % num_classes`, `query = index // num_classes`.
pub fn topk_over_classes(
    logits: &[f32],
    num_queries: usize,
    num_classes: usize,
) -> Vec<(usize, usize, f32)> {
    let mut scored: Vec<(usize, usize, f32)> = Vec::with_capacity(num_queries * num_classes);
    for q in 0..num_queries {
        for c in 0..num_classes {
            let s = sigmoid_scalar(logits[q * num_classes + c]);
            scored.push((q, c, s));
        }
    }
    // topk by score, descending; ties broken by original flat index (stable) to
    // mirror torch.topk's deterministic ordering closely enough for filtering.
    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(num_queries);
    scored
}

/// Sigmoid for a single scalar.
fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Computes per-query reading-order ranks from the pairwise order logits.
///
/// Port of `_get_order_seqs`: `order_votes = triu(sigmoid(O),1).sum(1) +
/// tril(1 - sigmoid(O)ᵀ, -1).sum(1)`, then `order_seq = argsort(order_votes)`
/// inverted into ranks. `order_logits` is row-major `[seq, seq]`. Returns the
/// rank of each query (`ranks[q]` = reading position of query `q`).
pub fn order_seqs(order_logits: &[f32], seq: usize) -> Vec<usize> {
    let s = |i: usize, j: usize| sigmoid_scalar(order_logits[i * seq + j]);

    // votes[j] = sum_{i<j} sigmoid(O[i][j]) + sum_{i>j} (1 - sigmoid(O[j][i]))
    // Derivation: triu(S,1).sum(dim=1) sums over rows i for each column j>i;
    // tril(1 - Sᵀ, -1).sum(dim=1) sums over rows i for each column j<i using Sᵀ.
    let mut votes = vec![0f64; seq];
    for (j, vote) in votes.iter_mut().enumerate() {
        let mut acc = 0f64;
        for i in 0..j {
            acc += s(i, j) as f64;
        }
        for i in (j + 1)..seq {
            acc += 1.0 - s(j, i) as f64;
        }
        *vote = acc;
    }

    // pointers = argsort(votes) ascending; rank[pointers[k]] = k.
    let mut pointers: Vec<usize> = (0..seq).collect();
    pointers.sort_by(|&a, &b| votes[a].partial_cmp(&votes[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0usize; seq];
    for (rank, &p) in pointers.iter().enumerate() {
        ranks[p] = rank;
    }
    ranks
}

/// Assembles the final, sorted detections for one image.
///
/// - `boxes_xyxy`: per-query corner boxes in the model's normalised `[0,1]` space,
///   row-major `[num_queries, 4]` (already cxcywh→xyxy).
/// - `scores`/`labels`/`query_of`: the topk selection (score, class id, source
///   query index) as produced by [`topk_over_classes`].
/// - `ranks`: per-query reading-order rank from [`order_seqs`].
/// - `(img_w, img_h)`: the original image size to scale boxes back to.
/// - `conf`: final confidence threshold.
///
/// Keeps entries with `score >= conf`, scales boxes to pixels, orders them by
/// reading-order rank, and assigns 0-based `order`.
#[allow(clippy::too_many_arguments)]
pub fn assemble(
    boxes_xyxy: &[f32],
    topk: &[(usize, usize, f32)],
    ranks: &[usize],
    img_w: f32,
    img_h: f32,
    conf: f32,
) -> Result<Vec<LayoutDet>> {
    // Collect kept entries with their reading-order rank.
    let mut kept: Vec<(usize, LayoutLabel, f32, [f32; 4])> = Vec::new();
    for &(q, c, score) in topk {
        if score < conf {
            continue;
        }
        let label = LayoutLabel::from_id(c)?;
        let base = q * 4;
        let bx = boxes_xyxy
            .get(base..base + 4)
            .ok_or_else(|| Error::Shape(format!("box index {q} out of range")))?;
        let scaled = [
            bx[0] * img_w,
            bx[1] * img_h,
            bx[2] * img_w,
            bx[3] * img_h,
        ];
        let rank = ranks.get(q).copied().unwrap_or(usize::MAX);
        kept.push((rank, label, score, scaled));
    }

    // Sort by reading-order rank (mirrors `torch.sort(order_seq)`).
    kept.sort_by_key(|k| k.0);

    let mut out = Vec::with_capacity(kept.len());
    for (order, (_, label, score, bx)) in kept.into_iter().enumerate() {
        let bbox = BBox::new(bx[0], bx[1], bx[2], bx[3]);
        out.push(LayoutDet::new(bbox, label, round4(score), order));
    }
    Ok(out)
}

/// Rounds a score to 4 decimals, matching the reference's `round(score, 4)`.
fn round4(x: f32) -> f32 {
    (x * 10_000.0).round() / 10_000.0
}

/// Extracts a `[num_queries, 4]` xyxy box slice from the model's `pred_boxes`
/// tensor (cxcywh, `[0,1]`), converting each box. Batch dim must be 1.
pub fn boxes_to_xyxy<B: Backend>(pred_boxes: &Tensor<B, 3>) -> Result<Vec<f32>> {
    let dims = pred_boxes.dims();
    if dims[0] != 1 {
        return Err(Error::Shape(format!("expected batch size 1, got {}", dims[0])));
    }
    let data = mineru_burn_common::float_to_vec_f32(pred_boxes.clone());
    let num_q = dims[1];
    let mut out = vec![0f32; num_q * 4];
    for q in 0..num_q {
        let b = [
            data[q * 4],
            data[q * 4 + 1],
            data[q * 4 + 2],
            data[q * 4 + 3],
        ];
        let xyxy = cxcywh_to_xyxy(b);
        out[q * 4..q * 4 + 4].copy_from_slice(&xyxy);
    }
    Ok(out)
}

/// Flattens a `[1, seq, seq]` order-logits tensor into a row-major `Vec` + `seq`.
pub fn order_logits_flat<B: Backend>(order_logits: &Tensor<B, 3>) -> Result<(Vec<f32>, usize)> {
    let dims = order_logits.dims();
    let seq = dims[1];
    let data = mineru_burn_common::float_to_vec_f32(order_logits.clone());
    Ok((data, seq))
}

/// Flattens a `[1, num_queries, num_classes]` logits tensor into a row-major `Vec`.
pub fn logits_flat<B: Backend>(logits: &Tensor<B, 3>) -> Result<(Vec<f32>, usize, usize)> {
    let dims = logits.dims();
    let data = mineru_burn_common::float_to_vec_f32(logits.clone());
    Ok((data, dims[1], dims[2]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cxcywh_to_xyxy_centers_correctly() {
        let out = cxcywh_to_xyxy([10.0, 20.0, 4.0, 6.0]);
        assert_eq!(out, [8.0, 17.0, 12.0, 23.0]);
    }

    #[test]
    fn topk_selects_highest_scores() {
        // 2 queries, 3 classes. Highest logit per cell wins.
        // q0: [0.0, 5.0, -1.0]; q1: [3.0, 0.0, 0.0]
        let logits = [0.0, 5.0, -1.0, 3.0, 0.0, 0.0];
        let top = topk_over_classes(&logits, 2, 3);
        assert_eq!(top.len(), 2);
        // Highest is q0,class1 (sigmoid(5)); then q1,class0 (sigmoid(3)).
        assert_eq!((top[0].0, top[0].1), (0, 1));
        assert_eq!((top[1].0, top[1].1), (1, 0));
        assert!(top[0].2 > top[1].2);
    }

    #[test]
    fn order_seqs_identity_when_lower_triangular_dominates() {
        // Construct O so that earlier queries clearly precede later ones:
        // large positive O[i][j] for i<j means "i before j".
        let seq = 3;
        let big = 20.0;
        // O[i][j] = big for i<j, else -big.
        let mut o = vec![0f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                o[i * seq + j] = if i < j { big } else { -big };
            }
        }
        let ranks = order_seqs(&o, seq);
        // Query 0 should read first (rank 0), then 1, then 2.
        assert_eq!(ranks, vec![0, 1, 2]);
    }

    #[test]
    fn order_seqs_reversed_when_upper_triangular_dominates() {
        let seq = 3;
        let big = 20.0;
        // O[i][j] = big for i>j (later precedes earlier) -> reversed reading order.
        let mut o = vec![0f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                o[i * seq + j] = if i > j { big } else { -big };
            }
        }
        let ranks = order_seqs(&o, seq);
        assert_eq!(ranks, vec![2, 1, 0]);
    }

    #[test]
    fn assemble_filters_and_orders() {
        // Two queries: q0 kept with rank 1, q1 kept with rank 0 -> q1 first.
        let boxes = [
            0.0, 0.0, 0.5, 0.5, // q0
            0.5, 0.5, 1.0, 1.0, // q1
        ];
        let topk = vec![
            (0usize, LayoutLabel::Text.id(), 0.9f32),
            (1usize, LayoutLabel::Table.id(), 0.8f32),
        ];
        let ranks = vec![1usize, 0usize];
        let dets = assemble(&boxes, &topk, &ranks, 100.0, 200.0, 0.45).expect("assemble");
        assert_eq!(dets.len(), 2);
        // q1 (rank 0) comes first.
        assert_eq!(dets[0].label, LayoutLabel::Table);
        assert_eq!(dets[0].order, 0);
        assert_eq!(dets[1].label, LayoutLabel::Text);
        assert_eq!(dets[1].order, 1);
        // Box scaled by (100, 200).
        assert_eq!(dets[0].bbox, BBox::new(50.0, 100.0, 100.0, 200.0));
    }

    #[test]
    fn assemble_drops_below_conf() {
        let boxes = [0.0, 0.0, 0.1, 0.1];
        let topk = vec![(0usize, LayoutLabel::Text.id(), 0.2f32)];
        let ranks = vec![0usize];
        let dets = assemble(&boxes, &topk, &ranks, 100.0, 100.0, 0.45).expect("assemble");
        assert!(dets.is_empty());
    }
}
