//! CTC greedy decoding, shared by OCR recognition.
//!
//! Given per-timestep class logits, greedy (best-path) CTC decoding takes the
//! argmax class at each step, then collapses runs of the same class and drops the
//! blank symbol. The core is a pure function so it can be unit-tested without a
//! backend and reused by any model that produces a `[T, C]` logit grid.

use burn::prelude::Backend;
use burn::tensor::Tensor;

/// Greedily decodes a `[T, C]` logit grid into a class-index sequence.
///
/// `logits` is a row-major slice of `time_steps` rows, each `num_classes` wide;
/// `logits[t * num_classes + c]` is the score of class `c` at step `t`. The
/// decoder takes the argmax over each row, collapses consecutive duplicate
/// classes, and removes `blank_idx`.
///
/// Returns an empty vector if `time_steps` or `num_classes` is zero, or if the
/// slice is shorter than `time_steps * num_classes` (the trailing partial row is
/// ignored). This function never panics.
///
/// # Examples
///
/// ```
/// use mineru_burn_common::ctc::ctc_greedy_decode_slice;
///
/// // 4 timesteps, 3 classes, blank = 0.
/// // argmax per step: [1, 1, 0(blank), 2] -> collapse+drop-blank -> [1, 2]
/// let logits = [
///     0.1, 0.8, 0.1, // t0 -> 1
///     0.2, 0.7, 0.1, // t1 -> 1 (collapsed)
///     0.9, 0.05, 0.05, // t2 -> 0 (blank, dropped)
///     0.1, 0.2, 0.7, // t3 -> 2
/// ];
/// assert_eq!(ctc_greedy_decode_slice(&logits, 4, 3, 0), vec![1, 2]);
/// ```
pub fn ctc_greedy_decode_slice(
    logits: &[f32],
    time_steps: usize,
    num_classes: usize,
    blank_idx: usize,
) -> Vec<usize> {
    if time_steps == 0 || num_classes == 0 {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut prev: Option<usize> = None;

    for t in 0..time_steps {
        let start = t * num_classes;
        let Some(row) = logits.get(start..start + num_classes) else {
            break;
        };

        // argmax over the row; ties resolve to the lowest index.
        let mut best = 0usize;
        let mut best_val = row[0];
        for (c, &v) in row.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best = c;
            }
        }

        // Collapse repeats, then drop blanks.
        if prev != Some(best) {
            if best != blank_idx {
                out.push(best);
            }
            prev = Some(best);
        }
    }

    out
}

/// Greedily decodes a Burn `[T, C]` (or `[1, T, C]`) logit tensor.
///
/// A convenience wrapper over [`ctc_greedy_decode_slice`] that reads the tensor's
/// data back to the host first. A leading batch dimension of size 1 is accepted
/// and squeezed. Returns an empty vector if the data cannot be read as `f32`.
pub fn ctc_greedy_decode<B: Backend>(logits: Tensor<B, 2>, blank_idx: usize) -> Vec<usize> {
    let [time_steps, num_classes] = logits.dims();
    let data = logits.to_data();
    match data.into_vec::<f32>() {
        Ok(v) => ctc_greedy_decode_slice(&v, time_steps, num_classes, blank_idx),
        Err(_) => Vec::new(),
    }
}
