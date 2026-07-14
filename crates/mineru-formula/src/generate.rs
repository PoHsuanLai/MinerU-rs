//! Greedy autoregressive decode loop.
//!
//! Mirrors what `VisionEncoderDecoderModel.generate()` does for this model with
//! `do_sample=False` (greedy): start from the decoder-start token (BOS), and at
//! each step run the decoder over the whole prefix, take the argmax of the last
//! position's logits, append it, and stop at EOS or the length cap.
//!
//! # What is faithful vs simplified
//! - **Greedy** selection matches the Python default (`do_sample=False`, no beam).
//! - This loop is **non-cached**: each step re-runs the decoder over the growing
//!   prefix, which is `O(T²)` in decoder work. HF uses a KV cache for `O(T)`.
//!   Correctness is identical; only speed differs. Adding a cache is the single
//!   biggest performance TODO (see the crate notes).
//! - `forced_eos_token_id` at `max_length` is honored by simply stopping; we do
//!   not overwrite the final token, which only matters at the exact cap.
//!
//! The loop is generic over a [`DecodeStep`] so it can be unit-tested against a
//! mock that returns canned logits without any weights (see the tests).

use crate::config::MBartConfig;

/// A single decode step: given the token ids produced so far, return the logits
/// for the **next** token (a `vocab`-length row).
///
/// The real implementation runs the Swin encoder once and the MBart decoder over
/// the prefix; the encoder output is captured by the closure/impl. Abstracting it
/// lets [`greedy_decode`] be tested with a pure-logic mock.
pub trait DecodeStep {
    /// Returns the greedy next-token id for the given prefix `ids`.
    ///
    /// The argmax is taken by the implementor — on the tensor backend where the
    /// logits live — so only the single chosen id crosses to the host, not the
    /// whole vocab-length row. This matters on GPU backends (e.g. wgpu), where a
    /// per-token `vocab`-wide device→host copy would stall the decode loop. Ties
    /// must break toward the **lower** index to match `torch.argmax` (see
    /// [`argmax`]).
    fn step(&mut self, ids: &[u32]) -> u32;
}

/// Result of a greedy decode: the generated token ids **excluding** the initial
/// start token, in order, up to (and excluding) EOS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decoded {
    /// Generated token ids (no BOS, no trailing EOS).
    pub tokens: Vec<u32>,
    /// Whether generation stopped because EOS was produced (vs hitting the cap).
    pub hit_eos: bool,
}

/// Runs greedy decoding.
///
/// - `start_token`: the decoder-start / BOS id to seed the prefix.
/// - `eos_token`: stop once this id is produced.
/// - `max_new_tokens`: hard cap on generated tokens.
///
/// Returns the generated ids (BOS dropped, EOS not included).
pub fn greedy_decode<S: DecodeStep>(
    step: &mut S,
    start_token: u32,
    eos_token: u32,
    max_new_tokens: usize,
) -> Decoded {
    let mut ids = vec![start_token];
    let mut out = Vec::new();
    let mut hit_eos = false;

    for _ in 0..max_new_tokens {
        let next = step.step(&ids);
        if next == eos_token {
            hit_eos = true;
            break;
        }
        out.push(next);
        ids.push(next);
    }

    Decoded {
        tokens: out,
        hit_eos,
    }
}

/// Convenience wrapper reading `start`/`eos`/cap straight from the decoder config.
pub fn greedy_decode_with_config<S: DecodeStep>(
    step: &mut S,
    cfg: &MBartConfig,
    max_new_tokens: usize,
) -> Decoded {
    greedy_decode(
        step,
        cfg.bos_token_id as u32,
        cfg.eos_token_id as u32,
        max_new_tokens,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host argmax (ties toward the lower index, matching `torch.argmax` and the
    /// on-device `Tensor::argmax` the real [`DecodeStep`] impls use). Test-only: the
    /// production path argmaxes on the tensor backend, not the host.
    fn argmax(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best as u32
    }

    /// A mock that plays back a fixed logits sequence, one row per step. The vocab
    /// is tiny; each row is the argmax target one-hot-ish.
    struct MockSteps {
        rows: Vec<Vec<f32>>,
        idx: usize,
    }

    impl DecodeStep for MockSteps {
        fn step(&mut self, _ids: &[u32]) -> u32 {
            // Mirror the real impl: argmax the row (the implementor's job now),
            // so the tests still exercise the lower-index tie-break via `argmax`.
            let row = &self.rows[self.idx.min(self.rows.len() - 1)];
            let next = argmax(row);
            self.idx += 1;
            next
        }
    }

    fn one_hot(vocab: usize, i: usize) -> Vec<f32> {
        let mut v = vec![0.0; vocab];
        v[i] = 1.0;
        v
    }

    #[test]
    fn stops_at_eos_and_drops_it() {
        // vocab=5, eos=2. Emit 3, 4, then eos.
        let mut m = MockSteps {
            rows: vec![one_hot(5, 3), one_hot(5, 4), one_hot(5, 2)],
            idx: 0,
        };
        let d = greedy_decode(&mut m, 0, 2, 100);
        assert_eq!(d.tokens, vec![3, 4]);
        assert!(d.hit_eos);
    }

    #[test]
    fn respects_max_new_tokens() {
        // Never emits eos; always token 1. Cap at 4.
        let mut m = MockSteps {
            rows: vec![one_hot(3, 1)],
            idx: 0,
        };
        let d = greedy_decode(&mut m, 0, 2, 4);
        assert_eq!(d.tokens, vec![1, 1, 1, 1]);
        assert!(!d.hit_eos);
    }

    #[test]
    fn argmax_picks_highest_lowest_index_on_tie() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        // tie between index 0 and 2 -> lower index wins
        assert_eq!(argmax(&[0.5, 0.2, 0.5]), 0);
    }

    #[test]
    fn empty_generation_when_first_token_is_eos() {
        let mut m = MockSteps {
            rows: vec![one_hot(4, 2)],
            idx: 0,
        };
        let d = greedy_decode(&mut m, 0, 2, 10);
        assert!(d.tokens.is_empty());
        assert!(d.hit_eos);
    }
}
