//! Greedy autoregressive decode loop.
//!
//! Mirrors what `VisionEncoderDecoderModel.generate()` does for this model with
//! `do_sample=False` (greedy): start from the decoder-start token (BOS), and at
//! each step run the decoder over the whole prefix, take the argmax of the last
//! position's logits, append it, and stop at EOS or the length cap.
//!
//! # What is faithful vs simplified
//! - **Greedy** selection matches the Python default (`do_sample=False`, no beam).
//! - This loop is **KV-cached** (`O(T)` decoder work): each step feeds only the one
//!   new token, and the [`DecodeStep`] impl reuses cached self-/cross-attention K/V
//!   from prior steps (see [`crate::mbart::DecoderCache`]). HF's reference does the
//!   same. The generated token sequence is byte-identical to the earlier non-cached
//!   loop — the cache changes only *how* each step's logits are computed, not their
//!   value. Only the last position's logits are ever needed for greedy argmax, which
//!   is exactly what the cached step returns.
//! - `forced_eos_token_id` at `max_length` is honored by simply stopping; we do
//!   not overwrite the final token, which only matters at the exact cap.
//!
//! The loop is generic over a [`DecodeStep`] so it can be unit-tested against a
//! mock that returns canned tokens without any weights (see the tests).

use crate::config::MBartConfig;

/// A single incremental decode step: given the **one** token emitted at the previous
/// step and its position, return the greedy next-token id.
///
/// The real implementation runs the Swin encoder once (before the loop) and, per
/// step, runs the MBart decoder over just the new token, reusing cached
/// self-/cross-attention K/V from prior steps ([`crate::mbart::DecoderCache`]). The
/// impl owns that mutable cache, which is why [`DecodeStep::step`] takes `&mut self`.
/// Abstracting it lets [`greedy_decode`] be tested with a pure-logic mock.
///
/// # Contract
/// [`greedy_decode`] calls [`DecodeStep::step`] with `position = 0` for the
/// decoder-start (BOS) token, then `1, 2, …`, each time passing the token the
/// previous call returned. Implementations must be driven in this strict order:
/// each step appends exactly one token's K/V to its cache.
pub trait DecodeStep {
    /// Advances the decode by one token.
    ///
    /// - `token`: the id to feed this step (BOS on the first call, else the id the
    ///   previous call returned).
    /// - `position`: the 0-based position of `token` in the sequence.
    ///
    /// Returns the greedy next-token id. The argmax is taken by the implementor — on
    /// the tensor backend where the logits live — so only the single chosen id
    /// crosses to the host, not the whole vocab-length row. This matters on GPU
    /// backends (e.g. wgpu), where a per-token `vocab`-wide device→host copy would
    /// stall the decode loop. Ties must break toward the **lower** index to match
    /// `torch.argmax` (see [`argmax`]).
    fn step(&mut self, token: u32, position: usize) -> u32;
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
    let mut out = Vec::new();
    let mut hit_eos = false;

    // Feed the BOS token at position 0, then each emitted token at the next position.
    // The [`DecodeStep`] impl appends exactly one token's K/V to its cache per call,
    // so the driving order here is part of the contract.
    let mut token = start_token;
    for position in 0..max_new_tokens {
        let next = step.step(token, position);
        if next == eos_token {
            hit_eos = true;
            break;
        }
        out.push(next);
        token = next;
    }

    Decoded {
        tokens: out,
        hit_eos,
    }
}

/// The batched sibling of [`DecodeStep`]: advances **N independent lanes** by one
/// token each, in lockstep.
///
/// # Why lane independence holds
/// Every tensor in the decoder carries the batch as its leading dim and no operation
/// mixes across it. Self-attention shapes q/k/v to `[B, heads, len, dim]` and the
/// score matmul contracts only the last two dims, so row `i` of the batch attends
/// solely to row `i` of the cached K/V ([`crate::mbart::attention`]). The additive
/// attention mask is reshaped to `[1, 1, tgt, src]` and *broadcast* over the batch —
/// it never carries information between rows. Cross-attention K/V are per-row slices
/// of the encoder grid. Therefore decoding N crops as one batch is arithmetically the
/// same as decoding each alone, and the token ids must be byte-identical. That is the
/// invariant the real-weights batch parity gate proves.
///
/// # Contract
/// [`greedy_decode_batch`] calls [`BatchDecodeStep::step_batch`] with a slice of
/// exactly `batch_len` tokens (all lanes' BOS on the first call, then each lane's
/// previously returned token) and a single shared `position`. Every lane advances one
/// position per call, including lanes that already finished — see
/// [`greedy_decode_batch`] for why. Implementations must return exactly `batch_len`
/// ids, index-aligned to the input lanes.
pub trait BatchDecodeStep {
    /// Advances every lane by one token.
    ///
    /// - `tokens`: one id per lane (BOS on the first call, else the ids the previous
    ///   call returned).
    /// - `position`: the 0-based position of these tokens; shared by all lanes, which
    ///   is what keeps the per-lane KV caches aligned.
    ///
    /// Returns the greedy next-token id per lane, in lane order. As with
    /// [`DecodeStep::step`], the argmax is taken on the tensor backend so only the
    /// chosen ids cross to the host. Ties must break toward the **lower** index.
    fn step_batch(&mut self, tokens: &[u32], position: usize) -> Vec<u32>;
}

/// Runs greedy decoding over `batch_len` independent lanes in lockstep.
///
/// Returns one [`Decoded`] per lane, **index-aligned to the input lanes**.
///
/// # Ragged EOS
/// Lanes finish at different steps. Rather than compacting the batch (which would
/// mean slicing every layer's KV cache — more cost than it saves, and it would breach
/// the cache's encapsulation), a finished lane is simply kept in the batch and fed
/// `eos_token` forever. Its emitted ids are discarded via the host-side done mask.
/// This keeps `position` and the cached K/V aligned across all lanes at zero
/// bookkeeping cost. The loop breaks as soon as every lane is done, so the batch runs
/// for as long as its longest lane — which is why callers group similar-length work.
pub fn greedy_decode_batch<S: BatchDecodeStep>(
    step: &mut S,
    start_token: u32,
    eos_token: u32,
    max_new_tokens: usize,
    batch_len: usize,
) -> Vec<Decoded> {
    let mut out: Vec<Decoded> = (0..batch_len)
        .map(|_| Decoded {
            tokens: Vec::new(),
            hit_eos: false,
        })
        .collect();
    if batch_len == 0 {
        return out;
    }

    let mut tokens = vec![start_token; batch_len];
    let mut done = vec![false; batch_len];

    for position in 0..max_new_tokens {
        let next = step.step_batch(&tokens, position);
        // A short return would silently misalign lanes; treat it as "all lanes
        // finished" rather than indexing past the end.
        if next.len() != batch_len {
            break;
        }

        for (lane, &id) in next.iter().enumerate() {
            match done.get(lane) {
                Some(true) | None => continue,
                Some(false) => {}
            }
            if id == eos_token {
                if let Some(slot) = out.get_mut(lane) {
                    slot.hit_eos = true;
                }
                if let Some(flag) = done.get_mut(lane) {
                    *flag = true;
                }
            } else if let Some(slot) = out.get_mut(lane) {
                slot.tokens.push(id);
            }
        }

        if done.iter().all(|&d| d) {
            break;
        }

        // Finished lanes are re-fed EOS so their positions stay in lockstep with the
        // live ones; their outputs are already excluded by the done mask.
        for (lane, slot) in tokens.iter_mut().enumerate() {
            *slot = match done.get(lane) {
                Some(true) | None => eos_token,
                Some(false) => next.get(lane).copied().unwrap_or(eos_token),
            };
        }
    }

    out
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
        fn step(&mut self, _token: u32, _position: usize) -> u32 {
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

    /// A batched mock: `scripts[lane]` is the id that lane emits at each step. A lane
    /// whose script is exhausted repeats its last id forever (so callers can script a
    /// lane that never stops). Records what it was fed to prove finished lanes keep
    /// being driven with EOS.
    struct MockBatch {
        scripts: Vec<Vec<u32>>,
        fed: Vec<Vec<u32>>,
        positions: Vec<usize>,
    }

    impl MockBatch {
        fn new(scripts: Vec<Vec<u32>>) -> Self {
            Self {
                scripts,
                fed: Vec::new(),
                positions: Vec::new(),
            }
        }
    }

    impl BatchDecodeStep for MockBatch {
        fn step_batch(&mut self, tokens: &[u32], position: usize) -> Vec<u32> {
            self.fed.push(tokens.to_vec());
            self.positions.push(position);
            self.scripts
                .iter()
                .map(|s| match s.get(position) {
                    Some(&id) => id,
                    None => s.last().copied().unwrap_or(0),
                })
                .collect()
        }
    }

    #[test]
    fn batch_lanes_finishing_at_different_steps() {
        // eos=2. Lane 0 stops after 1 token, lane 1 after 3, lane 2 after 2.
        let mut m = MockBatch::new(vec![
            vec![5, 2],
            vec![6, 7, 8, 2],
            vec![9, 4, 2],
        ]);
        let d = greedy_decode_batch(&mut m, 0, 2, 10, 3);

        assert_eq!(d[0].tokens, vec![5]);
        assert_eq!(d[1].tokens, vec![6, 7, 8]);
        assert_eq!(d[2].tokens, vec![9, 4]);
        assert!(d.iter().all(|x| x.hit_eos));
        // Ran exactly as long as the longest lane (4 steps), then stopped.
        assert_eq!(m.positions, vec![0, 1, 2, 3]);
        // Lane 0 finished at step 1, so from step 2 on it must be fed EOS to stay
        // positionally aligned with the still-live lanes.
        assert_eq!(m.fed[2][0], 2);
        assert_eq!(m.fed[3][0], 2);
        // Lane 1 was still live at step 3 and must be fed its own last real token.
        assert_eq!(m.fed[3][1], 8);
    }

    #[test]
    fn batch_lane_hitting_eos_on_step_zero() {
        // Lane 1 emits EOS immediately; the other lanes must be unaffected.
        let mut m = MockBatch::new(vec![vec![5, 6, 2], vec![2], vec![7, 2]]);
        let d = greedy_decode_batch(&mut m, 0, 2, 10, 3);

        assert_eq!(d[0].tokens, vec![5, 6]);
        assert!(d[1].tokens.is_empty(), "lane 1 stopped before emitting");
        assert!(d[1].hit_eos);
        assert_eq!(d[2].tokens, vec![7]);
    }

    #[test]
    fn batch_all_lanes_hit_the_cap() {
        // No lane ever emits eos=2; every lane must stop at the cap with hit_eos false.
        let mut m = MockBatch::new(vec![vec![1], vec![3], vec![4]]);
        let d = greedy_decode_batch(&mut m, 0, 2, 4, 3);

        assert_eq!(d[0].tokens, vec![1, 1, 1, 1]);
        assert_eq!(d[1].tokens, vec![3, 3, 3, 3]);
        assert_eq!(d[2].tokens, vec![4, 4, 4, 4]);
        assert!(d.iter().all(|x| !x.hit_eos));
        assert_eq!(m.positions.len(), 4, "cap must bound the step count");
    }

    #[test]
    fn batch_of_one_equals_scalar_greedy_decode() {
        let script = vec![7u32, 8, 9, 2];

        let mut batched_mock = MockBatch::new(vec![script.clone()]);
        let batched = greedy_decode_batch(&mut batched_mock, 0, 2, 20, 1);

        // Drive the scalar loop off the identical script.
        struct Scripted {
            script: Vec<u32>,
        }
        impl DecodeStep for Scripted {
            fn step(&mut self, _token: u32, position: usize) -> u32 {
                self.script.get(position).copied().unwrap_or(0)
            }
        }
        let mut scalar_mock = Scripted { script };
        let scalar = greedy_decode(&mut scalar_mock, 0, 2, 20);

        assert_eq!(batched.len(), 1);
        assert_eq!(batched[0], scalar);
    }

    #[test]
    fn batch_of_zero_lanes_returns_empty() {
        let mut m = MockBatch::new(vec![]);
        let d = greedy_decode_batch(&mut m, 0, 2, 10, 0);
        assert!(d.is_empty());
        assert!(m.positions.is_empty(), "must not drive the model at all");
    }
}
