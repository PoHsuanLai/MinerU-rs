//! RT-DETR decoder: query selection from encoder memory + 6 deformable-attention
//! decoder layers with iterative box refinement.
//!
//! Port of the `RTDetrModel` query-selection tail and `RTDetrDecoder`. Produces
//! the final `logits` `[B, Q, num_labels]` and `pred_boxes` `[B, Q, 4]` (cxcywh in
//! `[0,1]`), plus the intermediate hidden state used by the reading-order head.

use burn::module::Module;
use burn::prelude::Backend;
use burn::tensor::activation::{relu, sigmoid, softmax};
use burn::tensor::{Int, Tensor, TensorData};

use crate::config::DET;
use mineru_burn_common::nn::{PtLayerNorm, PtLinear};

/// An `RTDetrMLPPredictionHead`: `num_layers` linears with ReLU between them.
#[derive(Module, Debug)]
pub struct MlpHead<B: Backend> {
    layers: Vec<PtLinear<B>>,
}

impl<B: Backend> MlpHead<B> {
    fn init(in_dim: usize, hidden: usize, out_dim: usize, num_layers: usize, device: &B::Device) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let din = if i == 0 { in_dim } else { hidden };
            let dout = if i == num_layers - 1 { out_dim } else { hidden };
            layers.push(PtLinear::init(din, dout, true, device));
        }
        Self { layers }
    }

    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let n = self.layers.len();
        let mut x = x;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i < n - 1 {
                x = relu(x);
            }
        }
        x
    }
}

/// Multiscale deformable attention (`RTDetrMultiscaleDeformableAttention`).
#[derive(Module, Debug)]
pub struct DeformableAttention<B: Backend> {
    sampling_offsets: PtLinear<B>,
    attention_weights: PtLinear<B>,
    value_proj: PtLinear<B>,
    output_proj: PtLinear<B>,
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    num_levels: usize,
    #[module(skip)]
    num_points: usize,
}

impl<B: Backend> DeformableAttention<B> {
    fn init(device: &B::Device) -> Self {
        let d = DET.d_model;
        let h = DET.decoder_attention_heads;
        let l = DET.num_feature_levels;
        let p = DET.decoder_n_points;
        Self {
            sampling_offsets: PtLinear::init(d, h * l * p * 2, true, device),
            attention_weights: PtLinear::init(d, h * l * p, true, device),
            value_proj: PtLinear::init(d, d, true, device),
            output_proj: PtLinear::init(d, d, true, device),
            num_heads: h,
            num_levels: l,
            num_points: p,
        }
    }

    /// `hidden`: `[B, Q, d]` (query + pos already added by the layer).
    /// `value_states`: `[B, L, d]` flattened encoder memory.
    /// `reference_points`: `[B, Q, num_levels, 4]` (cxcywh in `[0,1]`).
    /// `spatial_shapes`: `[(H_l, W_l); num_levels]`, `level_start`: prefix offsets.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        value_states: Tensor<B, 3>,
        reference_points: Tensor<B, 4>,
        spatial_shapes: &[(usize, usize)],
        level_start: &[usize],
    ) -> Tensor<B, 3> {
        let [bsz, num_q, d] = hidden.dims();
        let h = self.num_heads;
        let l = self.num_levels;
        let p = self.num_points;
        let head_dim = d / h;
        let device = hidden.device();

        let value = self.value_proj.forward(value_states); // [B, L, d]
        let total_len = value.dims()[1];
        // [B, L, H, head_dim]
        let value = value.reshape([bsz, total_len, h, head_dim]);

        // [B, Q, H, L, P, 2]
        let offsets = self
            .sampling_offsets
            .forward(hidden.clone())
            .reshape([bsz, num_q, h, l, p, 2]);
        // attention weights softmaxed over L*P.
        let weights = self
            .attention_weights
            .forward(hidden)
            .reshape([bsz, num_q, h, l * p]);
        let weights = softmax(weights, 3).reshape([bsz, num_q, h, l, p]);

        // reference_points: [B, Q, L, 4] -> center [.., :2], wh [.., 2:].
        let ref_center = reference_points.clone().narrow(3, 0, 2); // [B,Q,L,2]
        let ref_wh = reference_points.narrow(3, 2, 2); // [B,Q,L,2]
        // Broadcast to [B, Q, H, L, P, 2].
        let ref_center = ref_center
            .reshape([bsz, num_q, 1, l, 1, 2])
            .expand([bsz, num_q, h, l, p, 2]);
        let ref_wh = ref_wh
            .reshape([bsz, num_q, 1, l, 1, 2])
            .expand([bsz, num_q, h, l, p, 2]);
        // sampling_locations = ref_center + offsets / num_points * ref_wh * 0.5
        let sampling_locations = ref_center.add(
            offsets
                .div_scalar(p as f64)
                .mul(ref_wh)
                .mul_scalar(0.5),
        );

        // Per-level deformable sampling via grid_sample, then weighted sum.
        let mut sampled_per_level: Vec<Tensor<B, 4>> = Vec::with_capacity(l);
        for (lvl, &(hh, ww)) in spatial_shapes.iter().enumerate() {
            let start = level_start[lvl];
            let count = hh * ww;
            // value_l: [B, count, H, head_dim] -> [B*H, head_dim, hh, ww]
            let value_l = value
                .clone()
                .narrow(1, start, count)
                .reshape([bsz, hh, ww, h, head_dim])
                .permute([0, 3, 4, 1, 2]) // [B, H, head_dim, hh, ww]
                .reshape([bsz * h, head_dim, hh, ww]);

            // grid for this level: [B, Q, H, P, 2] -> [B*H, Q, P, 2], in [-1, 1].
            let grid_l = sampling_locations
                .clone()
                .narrow(3, lvl, 1) // [B,Q,H,1,P,2]
                .reshape([bsz, num_q, h, p, 2])
                .permute([0, 2, 1, 3, 4]) // [B, H, Q, P, 2]
                .reshape([bsz * h, num_q, p, 2]);
            let grid_l = grid_l.mul_scalar(2.0).sub_scalar(1.0);

            // sampled: [B*H, head_dim, Q, P]
            let sampled = value_l.grid_sample_2d(
                grid_l,
                burn::tensor::ops::GridSampleOptions::new(
                    burn::tensor::ops::InterpolateMode::Bilinear,
                ),
            );
            sampled_per_level.push(sampled);
        }

        // Stack levels along a new dim 3: each `sampled` is rank-4
        // [B*H, head_dim, Q, P]; `stack` inserts the L axis to give
        // [B*H, head_dim, Q, L, P]. (Do NOT pre-`reshape` to insert the axis —
        // `stack` already `unsqueeze`s each tensor, so pre-inserting would make
        // them rank-6 and fail the rank-5 stack check.)
        let stacked = Tensor::stack::<5>(sampled_per_level, 3)
            .reshape([bsz * h, head_dim, num_q, l * p]);

        // weights: [B, Q, H, L, P] -> [B*H, 1, Q, L*P]
        let w = weights
            .permute([0, 2, 1, 3, 4]) // [B, H, Q, L, P]
            .reshape([bsz * h, 1, num_q, l * p]);

        // weighted sum over (L*P): [B*H, head_dim, Q]
        let out = stacked.mul(w).sum_dim(3).reshape([bsz * h, head_dim, num_q]);
        // -> [B, Q, d]
        let out = out
            .reshape([bsz, h, head_dim, num_q])
            .permute([0, 3, 1, 2])
            .reshape([bsz, num_q, d]);

        let _ = &device;
        self.output_proj.forward(out)
    }
}

/// One decoder layer (`RTDetrDecoderLayer`): self-attn, cross deform-attn, FFN,
/// all post-norm.
#[derive(Module, Debug)]
pub struct DecoderLayer<B: Backend> {
    q_proj: PtLinear<B>,
    k_proj: PtLinear<B>,
    v_proj: PtLinear<B>,
    o_proj: PtLinear<B>,
    self_attn_layer_norm: PtLayerNorm<B>,
    encoder_attn: DeformableAttention<B>,
    encoder_attn_layer_norm: PtLayerNorm<B>,
    fc1: PtLinear<B>,
    fc2: PtLinear<B>,
    final_layer_norm: PtLayerNorm<B>,
    #[module(skip)]
    num_heads: usize,
}

impl<B: Backend> DecoderLayer<B> {
    fn init(device: &B::Device) -> Self {
        let d = DET.d_model;
        let ff = DET.decoder_ffn_dim;
        Self {
            q_proj: PtLinear::init(d, d, true, device),
            k_proj: PtLinear::init(d, d, true, device),
            v_proj: PtLinear::init(d, d, true, device),
            o_proj: PtLinear::init(d, d, true, device),
            self_attn_layer_norm: PtLayerNorm::init(d, DET.layer_norm_eps, device),
            encoder_attn: DeformableAttention::init(device),
            encoder_attn_layer_norm: PtLayerNorm::init(d, DET.layer_norm_eps, device),
            fc1: PtLinear::init(d, ff, true, device),
            fc2: PtLinear::init(ff, d, true, device),
            final_layer_norm: PtLayerNorm::init(d, DET.layer_norm_eps, device),
            num_heads: DET.decoder_attention_heads,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        query_pos: Tensor<B, 3>,
        value_states: Tensor<B, 3>,
        reference_points: Tensor<B, 4>,
        spatial_shapes: &[(usize, usize)],
        level_start: &[usize],
    ) -> Tensor<B, 3> {
        let [bsz, num_q, d] = hidden.dims();
        let head_dim = d / self.num_heads;
        let scale = (head_dim as f64).powf(-0.5);

        // Self-attention with pos added to Q,K.
        let q_in = hidden.clone().add(query_pos.clone());
        let k_in = hidden.clone().add(query_pos.clone());
        let q = self
            .q_proj
            .forward(q_in)
            .reshape([bsz, num_q, self.num_heads, head_dim])
            .swap_dims(1, 2);
        let k = self
            .k_proj
            .forward(k_in)
            .reshape([bsz, num_q, self.num_heads, head_dim])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(hidden.clone())
            .reshape([bsz, num_q, self.num_heads, head_dim])
            .swap_dims(1, 2);
        let scores = q.mul_scalar(scale).matmul(k.swap_dims(2, 3));
        let probs = softmax(scores, 3);
        let ctx = probs
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([bsz, num_q, d]);
        let attn_out = self.o_proj.forward(ctx);
        let hidden = self.self_attn_layer_norm.forward(hidden.add(attn_out));

        // Cross deformable attention (pos added inside via hidden+query_pos).
        let cross_in = hidden.clone().add(query_pos);
        let cross = self.encoder_attn.forward(
            cross_in,
            value_states,
            reference_points,
            spatial_shapes,
            level_start,
        );
        let hidden = self.encoder_attn_layer_norm.forward(hidden.add(cross));

        // FFN.
        let ff = self.fc2.forward(relu(self.fc1.forward(hidden.clone())));
        self.final_layer_norm.forward(hidden.add(ff))
    }
}

/// The query-selection head + stack of decoder layers + per-layer box/class heads.
#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    // Query selection (lives on RTDetrModel in PyTorch; grouped here for locality).
    enc_output: EncOutput<B>,
    enc_score_head: PtLinear<B>,
    enc_bbox_head: MlpHead<B>,
    decoder_input_proj: Vec<crate::encoder::ProjConvBn<B>>,
    // Decoder proper.
    query_pos_head: MlpHead<B>,
    layers: Vec<DecoderLayer<B>>,
    class_embed: Vec<PtLinear<B>>,
    bbox_embed: Vec<MlpHead<B>>,
}

/// `enc_output` = `Sequential(Linear, LayerNorm)`.
#[derive(Module, Debug)]
pub struct EncOutput<B: Backend> {
    // Vec of length 2 reproduces the `enc_output.0` / `enc_output.1` keys, but the
    // two members have different types, so they are named fields plus a remap.
    linear: PtLinear<B>,
    norm: PtLayerNorm<B>,
}

impl<B: Backend> EncOutput<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.norm.forward(self.linear.forward(x))
    }
}

/// The output of the decoder needed downstream.
pub struct DecoderOutput<B: Backend> {
    /// Final-layer class logits `[B, Q, num_labels]`.
    pub logits: Tensor<B, 3>,
    /// Final-layer boxes `[B, Q, 4]`, cxcywh in `[0, 1]`.
    pub pred_boxes: Tensor<B, 3>,
    /// Final-layer decoder hidden state `[B, Q, d_model]` (fed to reading order).
    pub last_hidden: Tensor<B, 3>,
}

impl<B: Backend> Decoder<B> {
    /// Initialises the decoder and query-selection heads.
    pub fn init(device: &B::Device) -> Self {
        let d = DET.d_model;
        let enc_output = EncOutput {
            linear: PtLinear::init(d, d, true, device),
            norm: PtLayerNorm::init(d, DET.layer_norm_eps, device),
        };
        let decoder_input_proj = (0..DET.num_feature_levels)
            .map(|_| crate::encoder::ProjConvBn::init(d, d, device))
            .collect();
        Self {
            enc_output,
            enc_score_head: PtLinear::init(d, DET.num_labels, true, device),
            enc_bbox_head: MlpHead::init(d, d, 4, 3, device),
            decoder_input_proj,
            query_pos_head: MlpHead::init(4, 2 * d, d, 2, device),
            layers: (0..DET.decoder_layers).map(|_| DecoderLayer::init(device)).collect(),
            class_embed: (0..DET.decoder_layers)
                .map(|_| PtLinear::init(d, DET.num_labels, true, device))
                .collect(),
            bbox_embed: (0..DET.decoder_layers)
                .map(|_| MlpHead::init(d, d, 4, 3, device))
                .collect(),
        }
    }

    /// Runs query selection and the decoder over the three encoder maps.
    pub fn forward(&self, encoder_maps: [Tensor<B, 4>; 3]) -> DecoderOutput<B> {
        let device = encoder_maps[0].device();
        let d = DET.d_model;

        // Project + flatten each level.
        let mut spatial_shapes: Vec<(usize, usize)> = Vec::with_capacity(3);
        let mut level_start: Vec<usize> = Vec::with_capacity(3);
        let mut flats: Vec<Tensor<B, 3>> = Vec::with_capacity(3);
        let mut offset = 0usize;
        let bsz = encoder_maps[0].dims()[0];
        for (lvl, map) in encoder_maps.into_iter().enumerate() {
            let projected = self.decoder_input_proj[lvl].forward(map);
            let [b, c, hh, ww] = projected.dims();
            spatial_shapes.push((hh, ww));
            level_start.push(offset);
            offset += hh * ww;
            flats.push(projected.reshape([b, c, hh * ww]).swap_dims(1, 2));
        }
        let source_flatten = Tensor::cat(flats, 1); // [B, L, d]
        let total_len = offset;

        // Anchors + valid mask.
        let (anchors, valid_mask) = generate_anchors::<B>(&spatial_shapes, &device); // [1,L,4], [1,L,1]
        let valid_mask_b = valid_mask.clone().expand([bsz, total_len, 1]);

        // memory = valid_mask * source; output_memory = enc_output(memory).
        let memory = source_flatten.clone().mul(valid_mask_b.clone());
        let output_memory = self.enc_output.forward(memory);

        let enc_class = self.enc_score_head.forward(output_memory.clone()); // [B,L,25]
        let enc_coord = self
            .enc_bbox_head
            .forward(output_memory.clone())
            .add(anchors.expand([bsz, total_len, 4])); // [B,L,4] logit space

        // topk over max class score.
        let num_q = DET.num_queries;
        let class_max = enc_class.max_dim(2).reshape([bsz, total_len]); // [B,L]
        let (_, topk_idx) = class_max.topk_with_indices(num_q, 1); // [B, num_q]

        let ref_unact = gather_rows(enc_coord, topk_idx.clone(), 4); // [B,num_q,4]
        let target = gather_rows(output_memory, topk_idx, d); // [B,num_q,d]

        // Decoder loop with box refinement. Seeded so the final `logits`/`boxes`
        // are always defined even though the loop always runs (>=1 layer).
        let mut reference_points = sigmoid(ref_unact); // [B,num_q,4] in [0,1]
        let mut hidden = target;
        // Seeded; both are overwritten on the first (always-executed) iteration.
        let mut last_logits = Tensor::<B, 3>::zeros([bsz, num_q, DET.num_labels], &device);
        let mut last_boxes = reference_points.clone();

        for i in 0..self.layers.len() {
            let query_pos = self.query_pos_head.forward(reference_points.clone()); // [B,num_q,d]
            let ref_input = reference_points
                .clone()
                .reshape([bsz, num_q, 1, 4])
                .expand([bsz, num_q, DET.num_feature_levels, 4]);

            hidden = self.layers[i].forward(
                hidden,
                query_pos,
                source_flatten.clone(),
                ref_input,
                &spatial_shapes,
                &level_start,
            );

            let delta = self.bbox_embed[i].forward(hidden.clone()); // [B,num_q,4]
            let new_ref = sigmoid(delta.add(inverse_sigmoid(reference_points.clone())));
            reference_points = new_ref.clone();
            last_logits = self.class_embed[i].forward(hidden.clone());
            last_boxes = new_ref;
        }

        DecoderOutput {
            logits: last_logits,
            pred_boxes: last_boxes,
            last_hidden: hidden,
        }
    }
}

/// Generates RT-DETR anchors and the validity mask.
///
/// For each level `l` with shape `(H, W)`: a `(x+0.5)/W, (y+0.5)/H` grid with per-
/// level `wh = 0.05 * 2^l`; the mask keeps anchors strictly inside `(eps, 1-eps)`
/// with `eps=0.01`; then `anchors = logit(anchors)` with invalid entries set to a
/// large value. Returns `(anchors [1, L, 4], valid_mask [1, L, 1])`.
fn generate_anchors<B: Backend>(
    spatial_shapes: &[(usize, usize)],
    device: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let grid_size = 0.05f64;
    let eps = 0.01f64;
    let mut anchors: Vec<f32> = Vec::new();
    let mut mask: Vec<f32> = Vec::new();

    for (lvl, &(h, w)) in spatial_shapes.iter().enumerate() {
        let wh = grid_size * 2f64.powi(lvl as i32);
        for gy in 0..h {
            for gx in 0..w {
                let cx = (gx as f64 + 0.5) / w as f64;
                let cy = (gy as f64 + 0.5) / h as f64;
                let coords = [cx, cy, wh, wh];
                let valid = coords.iter().all(|&c| c > eps && c < 1.0 - eps);
                mask.push(if valid { 1.0 } else { 0.0 });
                // logit(c); invalid entries overwritten with a large value.
                for &c in &coords {
                    let logit = (c / (1.0 - c)).ln();
                    anchors.push(if valid { logit as f32 } else { f32::MAX });
                }
            }
        }
    }

    let total: usize = spatial_shapes.iter().map(|&(h, w)| h * w).sum();
    let anchors = Tensor::<B, 1>::from_data(TensorData::new(anchors, [total * 4]), device)
        .reshape([1, total, 4]);
    let mask = Tensor::<B, 1>::from_data(TensorData::new(mask, [total]), device).reshape([1, total, 1]);
    (anchors, mask)
}

/// Gathers rows `[B, num_q, feat]` from `[B, L, feat]` by `indices [B, num_q]`.
fn gather_rows<B: Backend>(x: Tensor<B, 3>, indices: Tensor<B, 2, Int>, feat: usize) -> Tensor<B, 3> {
    let [bsz, num_q] = indices.dims();
    let idx = indices
        .reshape([bsz, num_q, 1])
        .expand([bsz, num_q, feat]);
    x.gather(1, idx)
}

/// Stable inverse sigmoid: `log(x / (1 - x))` with `x` clamped to `[eps, 1]`.
fn inverse_sigmoid<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let eps = 1e-5;
    let x = x.clamp(0.0, 1.0);
    let x1 = x.clone().clamp(eps, 1.0);
    let x2 = (x.mul_scalar(-1.0).add_scalar(1.0)).clamp(eps, 1.0);
    x1.div(x2).log()
}
