//! SLANet-plus structure-attention head (`SLAHead`).
//!
//! Faithful Burn port of the autoregressive attention-GRU decoder that the
//! SLANet-plus ONNX graph bakes into an ONNX `Loop`. `burn-onnx` codegen cannot
//! import that `Loop` (its type inference rejects the graph's `ConstantOfShape`
//! nodes — see [`super::model`]), so the head is hand-ported here and the loop is
//! unrolled in Rust.
//!
//! For each of `max_steps` decode steps, over the `[1, T, C_feat]` encoder feature
//! sequence (`T = H·W`, `C_feat = 96`):
//!
//! 1. **Attention** — project the feature (`linear0`) and the hidden state
//!    (`linear1`), add (broadcast over `T`), `tanh`, score with `linear2`,
//!    `softmax` over `T`, and take the weighted sum → a `[1, C_feat]` context.
//! 2. **GRU** — feed `concat(context, prev_onehot)` (`[1, 146]`) through a GRU
//!    cell (gate order reset/update/candidate, `h' = (1-z)·n + z·h`) → new hidden.
//! 3. **Heads** — `linear4(linear3(h))` gives the `[1, 50]` structure logits
//!    (softmaxed for the probability output), `sigmoid(linear6(linear5(h)))` gives
//!    the `[1, 8]` quadrilateral-corner box. Both branches are plain two-layer
//!    linear stacks with **no** intermediate activation (the ONNX `Loop` body has
//!    no ReLU between the pairs). The next step's one-hot input is the `argmax` of
//!    the current structure logits (greedy teacher-forcing at inference).
//!
//! The step-0 hidden state is zero and the step-0 one-hot input is class 0, both
//! matching the ONNX loop's initial carried values.
//!
//! # Numeric fidelity
//!
//! Every per-step operation is a node-for-node port of the ONNX `Loop` body,
//! cross-checked against `onnxruntime`:
//!
//! - the backbone/neck feature feed matches to `~5e-6`;
//! - the attention context matches to `~1e-6`;
//! - the GRU hidden state matches to `~5e-7` at every step.
//!
//! A dependency trace of the ONNX loop body confirms the per-step structure logits
//! and next hidden state are functions of only three carried values — the previous
//! hidden state, the previous token's one-hot, and the (fixed) feature sequence —
//! plus the weights: there is no hidden counter, position code, or scatter-buffer
//! feedback.
//!
//! The head branches are the subtle part: the ONNX `Loop` computes
//! `linear4(linear3(h))` and `sigmoid(linear6(linear5(h)))` with **no ReLU**
//! between the linear pairs. An earlier port inserted a ReLU there; because ReLU
//! only changes the output where `linear3(h)` has negative entries, the argmax
//! happened to agree for the first few tokens and then flipped (structure token 6
//! vs 48 at step 4 of the reference grid), giving the classic "matches N steps
//! then diverges" symptom even though the carried hidden state was bit-accurate.
//! Removing the ReLU makes the per-step argmax and probabilities match the ONNX
//! `Loop` across the whole decode (see `tests/slanet_real.rs`).

use burn::module::{Module, Param};
use burn::prelude::Backend;
use burn::tensor::{activation, Tensor, TensorData};

use mineru_burn_common::nn::PtLinear;

/// Encoder feature channel count feeding the head (`hardswish_72` has 96 channels).
pub const FEATURE_CHANNELS: usize = 96;
/// GRU hidden size.
pub const HIDDEN: usize = 256;
/// Structure class-channel count (the model's `out_channels`; 48 real tokens plus
/// two unused padding channels — decode looks up only the real indices).
pub const NUM_CLASSES: usize = 50;
/// Box-regression width: four `(x, y)` corners = eight coordinates.
pub const LOC_DIM: usize = 8;
/// Concatenated GRU input width: context (`96`) + previous one-hot (`50`).
const GRU_IN: usize = FEATURE_CHANNELS + NUM_CLASSES;

/// A Paddle-layout GRU cell.
///
/// Stores the input/hidden gate projections stacked as `[3·HIDDEN, in]` (the
/// order reset/update/candidate) plus their biases, exactly as the checkpoint
/// ships them, and applies the `nn.GRUCell` recurrence at forward time.
#[derive(Module, Debug)]
pub struct GruCell<B: Backend> {
    /// Input→gates weight, shape `[3·HIDDEN, in]`.
    pub w_ih: Param<Tensor<B, 2>>,
    /// Hidden→gates weight, shape `[3·HIDDEN, HIDDEN]`.
    pub w_hh: Param<Tensor<B, 2>>,
    /// Input→gates bias, shape `[3·HIDDEN]`.
    pub b_ih: Param<Tensor<B, 1>>,
    /// Hidden→gates bias, shape `[3·HIDDEN]`.
    pub b_hh: Param<Tensor<B, 1>>,
}

impl<B: Backend> GruCell<B> {
    /// Initialises a zeroed GRU cell; parameters are overwritten by loading.
    pub fn init(input_size: usize, hidden: usize, device: &B::Device) -> Self {
        Self {
            w_ih: Param::from_tensor(Tensor::zeros([3 * hidden, input_size], device)),
            w_hh: Param::from_tensor(Tensor::zeros([3 * hidden, hidden], device)),
            b_ih: Param::from_tensor(Tensor::zeros([3 * hidden], device)),
            b_hh: Param::from_tensor(Tensor::zeros([3 * hidden], device)),
        }
    }

    /// Advances the cell one step: `x` is `[1, in]`, `h` is `[1, HIDDEN]`.
    ///
    /// Computes `gi = x·w_ihᵀ + b_ih` and `gh = h·w_hhᵀ + b_hh`, splits each into
    /// the reset/update/candidate thirds, and applies
    /// `r = σ(gi_r + gh_r)`, `z = σ(gi_z + gh_z)`,
    /// `n = tanh(gi_n + r·gh_n)`, `h' = (1 − z)·n + z·h`.
    pub fn forward(&self, x: Tensor<B, 2>, h: Tensor<B, 2>) -> Tensor<B, 2> {
        let hidden = self.w_hh.dims()[1];
        let gi = x.matmul(self.w_ih.val().transpose()).add(self.b_ih.val().unsqueeze());
        let gh = h.clone().matmul(self.w_hh.val().transpose()).add(self.b_hh.val().unsqueeze());

        let gi_r = gi.clone().narrow(1, 0, hidden);
        let gi_z = gi.clone().narrow(1, hidden, hidden);
        let gi_n = gi.narrow(1, 2 * hidden, hidden);
        let gh_r = gh.clone().narrow(1, 0, hidden);
        let gh_z = gh.clone().narrow(1, hidden, hidden);
        let gh_n = gh.narrow(1, 2 * hidden, hidden);

        let r = activation::sigmoid(gi_r.add(gh_r));
        let z = activation::sigmoid(gi_z.add(gh_z));
        let n = activation::tanh(gi_n.add(r.mul(gh_n)));
        // h' = (1 - z) * n + z * h
        let one_minus_z = z.clone().neg().add_scalar(1.0);
        one_minus_z.mul(n).add(z.mul(h))
    }
}

/// The SLANet-plus attention-GRU structure head.
#[derive(Module, Debug)]
pub struct SlaHead<B: Backend> {
    /// Feature projection for attention (`[HIDDEN, FEATURE_CHANNELS]`, no bias).
    pub linear0: PtLinear<B>,
    /// Hidden-state projection for attention (`[HIDDEN, HIDDEN]`).
    pub linear1: PtLinear<B>,
    /// Attention score projection (`[1, HIDDEN]`, no bias).
    pub linear2: PtLinear<B>,
    /// Structure head hidden layer (`[HIDDEN, HIDDEN]`).
    pub linear3: PtLinear<B>,
    /// Structure head output (`[NUM_CLASSES, HIDDEN]`).
    pub linear4: PtLinear<B>,
    /// Box head hidden layer (`[HIDDEN, HIDDEN]`).
    pub linear5: PtLinear<B>,
    /// Box head output (`[LOC_DIM, HIDDEN]`).
    pub linear6: PtLinear<B>,
    /// The recurrent cell.
    pub gru: GruCell<B>,
}

/// Raw head outputs: flattened `[L, NUM_CLASSES]` structure probabilities and
/// `[L, LOC_DIM]` box corners, with the step count `L`.
pub struct HeadOutput {
    /// Row-major `[L·NUM_CLASSES]` softmaxed structure probabilities.
    pub structure_probs: Vec<f32>,
    /// Row-major `[L·LOC_DIM]` sigmoid box-corner regressions.
    pub loc_preds: Vec<f32>,
    /// Number of decoded steps `L`.
    pub len: usize,
}

impl<B: Backend> SlaHead<B> {
    /// Initialises a zeroed head; parameters are overwritten by loading.
    pub fn init(device: &B::Device) -> Self {
        Self {
            linear0: PtLinear::init(FEATURE_CHANNELS, HIDDEN, false, device),
            linear1: PtLinear::init(HIDDEN, HIDDEN, true, device),
            linear2: PtLinear::init(HIDDEN, 1, false, device),
            linear3: PtLinear::init(HIDDEN, HIDDEN, true, device),
            linear4: PtLinear::init(HIDDEN, NUM_CLASSES, true, device),
            linear5: PtLinear::init(HIDDEN, HIDDEN, true, device),
            linear6: PtLinear::init(HIDDEN, LOC_DIM, true, device),
            gru: GruCell::init(GRU_IN, HIDDEN, device),
        }
    }

    /// Runs the unrolled decoder over the feature sequence `fea` (`[1, T, 96]`).
    ///
    /// Stops early once the argmaxed structure token is the end sentinel (never on
    /// the first step), otherwise after `max_steps` steps. Returns host `f32`
    /// buffers in the [`HeadOutput`] contract the decoder consumes.
    pub fn forward(&self, fea: Tensor<B, 3>, max_steps: usize, end_idx: usize) -> HeadOutput {
        let device = fea.device();

        // Attention feature projection is step-invariant: [1, T, HIDDEN].
        let fea_proj = self.linear0.forward(fea.clone());

        let mut h = Tensor::<B, 2>::zeros([1, HIDDEN], &device);
        // Previous one-hot input, initialised to the `sos` class (index 0), the
        // ONNX loop's initial carried char index. Each subsequent step feeds the
        // one-hot of the previous step's structure argmax (greedy teacher forcing).
        let mut prev_onehot = onehot::<B>(0, NUM_CLASSES, &device);

        let mut structure_probs = Vec::with_capacity(max_steps * NUM_CLASSES);
        let mut loc_preds = Vec::with_capacity(max_steps * LOC_DIM);
        let mut len = 0usize;

        for step in 0..max_steps {
            // Attention: tanh(fea_proj + linear1(h)) -> score -> softmax over T.
            let h_proj = self.linear1.forward(h.clone()); // [1, HIDDEN]
            let e = activation::tanh(fea_proj.clone().add(h_proj.unsqueeze_dim(1))); // [1, T, HIDDEN]
            let score = self.linear2.forward(e); // [1, T, 1]
            let alpha = activation::softmax(score, 1); // over T
            // Context = sum_t alpha_t * fea_t -> [1, 96].
            let context = alpha.mul(fea.clone()).sum_dim(1).squeeze_dim::<2>(1);

            // GRU step over concat(context, prev_onehot) -> [1, HIDDEN].
            let gru_in = Tensor::cat(vec![context, prev_onehot.clone()], 1); // [1, 146]
            h = self.gru.forward(gru_in, h.clone());

            // Structure logits and box. The two head branches are plain
            // two-layer linear stacks with NO intermediate activation — the ONNX
            // `Loop` body computes `linear4(linear3(h))` and
            // `sigmoid(linear6(linear5(h)))` directly, with no ReLU between the
            // pairs (verified node-for-node against the graph).
            let s_hidden = self.linear3.forward(h.clone());
            let s_logits = self.linear4.forward(s_hidden); // [1, NUM_CLASSES]
            let s_probs = activation::softmax(s_logits.clone(), 1);

            let l_hidden = self.linear5.forward(h.clone());
            let l_box = activation::sigmoid(self.linear6.forward(l_hidden)); // [1, LOC_DIM]

            let s_vec = to_vec::<B>(s_probs);
            let l_vec = to_vec::<B>(l_box);
            let char_idx = argmax(&s_vec);

            structure_probs.extend_from_slice(&s_vec);
            loc_preds.extend_from_slice(&l_vec);
            len += 1;

            // Early stop on the end sentinel (but never on the first step).
            if step > 0 && char_idx == end_idx {
                break;
            }
            prev_onehot = onehot::<B>(char_idx, NUM_CLASSES, &device);
        }

        HeadOutput {
            structure_probs,
            loc_preds,
            len,
        }
    }
}

/// One per-step decode trace entry: `(hidden[HIDDEN], structure argmax,
/// structure probs[NUM_CLASSES])`, used only by the hidden parity hooks.
#[doc(hidden)]
pub type StepTrace = (Vec<f32>, usize, Vec<f32>);

/// Parity hook (hidden): per-step decode trace for numeric comparison against
/// the ONNX `Loop`. Not part of the public API.
#[doc(hidden)]
impl<B: Backend> SlaHead<B> {
    /// Runs `steps` decode steps and returns, per step, the post-GRU hidden state
    /// `[HIDDEN]`, the structure argmax index, and the full `[NUM_CLASSES]` probs.
    pub fn debug_steps(&self, fea: Tensor<B, 3>, steps: usize) -> Vec<StepTrace> {
        let device = fea.device();
        let fea_proj = self.linear0.forward(fea.clone());
        let mut h = Tensor::<B, 2>::zeros([1, HIDDEN], &device);
        let mut prev_onehot = onehot::<B>(0, NUM_CLASSES, &device);
        let mut out = Vec::with_capacity(steps);
        for _ in 0..steps {
            let h_proj = self.linear1.forward(h.clone());
            let e = activation::tanh(fea_proj.clone().add(h_proj.unsqueeze_dim(1)));
            let score = self.linear2.forward(e);
            let alpha = activation::softmax(score, 1);
            let context = alpha.mul(fea.clone()).sum_dim(1).squeeze_dim::<2>(1);
            let gru_in = Tensor::cat(vec![context, prev_onehot.clone()], 1);
            h = self.gru.forward(gru_in, h.clone());
            let s_hidden = self.linear3.forward(h.clone());
            let s_logits = self.linear4.forward(s_hidden);
            let s_probs = activation::softmax(s_logits, 1);
            let s_vec = to_vec::<B>(s_probs);
            let idx = argmax(&s_vec);
            let h_vec = to_vec::<B>(h.clone());
            out.push((h_vec, idx, s_vec));
            prev_onehot = onehot::<B>(idx, NUM_CLASSES, &device);
        }
        out
    }
}

/// Builds a `[1, num_classes]` one-hot row tensor with `1.0` at `idx`.
fn onehot<B: Backend>(idx: usize, num_classes: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut data = vec![0.0f32; num_classes];
    if idx < num_classes {
        data[idx] = 1.0;
    }
    Tensor::<B, 2>::from_data(TensorData::new(data, [1, num_classes]), device)
}

/// Moves a `[1, N]` tensor to a host `Vec<f32>`.
fn to_vec<B: Backend>(t: Tensor<B, 2>) -> Vec<f32> {
    t.into_data().into_vec::<f32>().unwrap_or_default()
}

/// Index of the maximum element (first on ties).
fn argmax(row: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best {
            best = v;
            best_idx = i;
        }
    }
    best_idx
}
