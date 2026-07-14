//! The reading-order pointer network (`PPDocLayoutV2ReadingOrder`).
//!
//! A LayoutLMv3-style transformer over the detected boxes plus a GlobalPointer
//! head that emits pairwise ordering logits. Port of the `reading_order` subtree.
//!
//! The forward pass takes the padded, filtered boxes (in `[0, 1000]` int space),
//! their remapped reading-order category ids, and the per-query keep mask, and
//! returns `order_logits [B, seq, seq]` for the query slots.
//!
//! # Fidelity
//! This is the least-trodden numerical path of the port; the CogView-stabilised
//! softmax, the RoPE-style relative-position bias, and the token/spatial embedding
//! construction are reproduced from the reference but are flagged as the parts
//! most worth adversarial verification.

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::Backend;
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::{Int, Tensor, TensorData};

use crate::config::{READING_ORDER as RO, ReadingOrderConfig};
use mineru_burn_common::nn::{PtLayerNorm, PtLinear};

/// The LayoutLMv3-derived text/spatial embeddings.
#[derive(Module, Debug)]
pub struct TextEmbeddings<B: Backend> {
    word_embeddings: burn::nn::Embedding<B>,
    token_type_embeddings: burn::nn::Embedding<B>,
    position_embeddings: burn::nn::Embedding<B>,
    x_position_embeddings: burn::nn::Embedding<B>,
    y_position_embeddings: burn::nn::Embedding<B>,
    h_position_embeddings: burn::nn::Embedding<B>,
    w_position_embeddings: burn::nn::Embedding<B>,
    spatial_proj: PtLinear<B>,
    norm: PtLayerNorm<B>,
}

impl<B: Backend> TextEmbeddings<B> {
    fn init(device: &B::Device) -> Self {
        use burn::nn::EmbeddingConfig;
        let hs = RO.hidden_size;
        Self {
            word_embeddings: EmbeddingConfig::new(RO.vocab_size, hs).init(device),
            token_type_embeddings: EmbeddingConfig::new(RO.type_vocab_size, hs).init(device),
            position_embeddings: EmbeddingConfig::new(RO.max_position_embeddings, hs).init(device),
            x_position_embeddings: EmbeddingConfig::new(RO.max_2d_position_embeddings, RO.coordinate_size)
                .init(device),
            y_position_embeddings: EmbeddingConfig::new(RO.max_2d_position_embeddings, RO.coordinate_size)
                .init(device),
            h_position_embeddings: EmbeddingConfig::new(RO.max_2d_position_embeddings, RO.shape_size)
                .init(device),
            w_position_embeddings: EmbeddingConfig::new(RO.max_2d_position_embeddings, RO.shape_size)
                .init(device),
            spatial_proj: PtLinear::init(
                4 * RO.coordinate_size + 2 * RO.shape_size,
                hs,
                true,
                device,
            ),
            norm: PtLayerNorm::init(hs, RO.layer_norm_eps, device),
        }
    }

    /// Computes the spatial-position embedding from integer boxes `[B, S, 4]`
    /// (xyxy in `[0, 1000]`), reproducing `calculate_spatial_position_embeddings`.
    fn spatial_position(&self, bbox: Tensor<B, 3, Int>) -> Tensor<B, 3> {
        let x0 = bbox.clone().narrow(2, 0, 1).squeeze_dim::<2>(2);
        let y0 = bbox.clone().narrow(2, 1, 1).squeeze_dim::<2>(2);
        let x1 = bbox.clone().narrow(2, 2, 1).squeeze_dim::<2>(2);
        let y1 = bbox.clone().narrow(2, 3, 1).squeeze_dim::<2>(2);

        let left = self.x_position_embeddings.forward(x0.clone());
        let upper = self.y_position_embeddings.forward(y0.clone());
        let right = self.x_position_embeddings.forward(x1.clone());
        let lower = self.y_position_embeddings.forward(y1.clone());

        // h = y1 - y0, w = x1 - x0 (clamped to the table range).
        let h = (y1 - y0).clamp(0, (RO.max_2d_position_embeddings - 1) as i64);
        let w = (x1 - x0).clamp(0, (RO.max_2d_position_embeddings - 1) as i64);
        let h_emb = self.h_position_embeddings.forward(h);
        let w_emb = self.w_position_embeddings.forward(w);

        Tensor::cat(vec![left, upper, right, lower, h_emb, w_emb], 2)
    }

    /// Full embedding: word + token-type + position + spatial, then LayerNorm is
    /// applied by the caller after adding the label projection.
    fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        bbox: Tensor<B, 3, Int>,
        position_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let [bsz, seq] = input_ids.dims();
        let device = input_ids.device();
        let token_type_ids = Tensor::<B, 2, Int>::zeros([bsz, seq], &device);

        let words = self.word_embeddings.forward(input_ids);
        let token_type = self.token_type_embeddings.forward(token_type_ids);
        let positions = self.position_embeddings.forward(position_ids);
        let spatial = self.spatial_proj.forward(self.spatial_position(bbox));
        words.add(token_type).add(positions).add(spatial)
    }
}

/// The RoPE-style relative-position bias (`PPDocLayoutV2PositionRelationEmbedding`).
#[derive(Module, Debug)]
pub struct PositionRelationEmbedding<B: Backend> {
    pos_proj: Conv2d<B>,
    #[module(skip)]
    inv_freq: Vec<f64>,
}

impl<B: Backend> PositionRelationEmbedding<B> {
    fn init(cfg: &ReadingOrderConfig, device: &B::Device) -> Self {
        let dim = cfg.relation_bias_embed_dim;
        let half = dim / 2;
        let inv_freq: Vec<f64> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / cfg.relation_bias_theta.powf(i as f64 / half as f64))
            .collect();
        let pos_proj = Conv2dConfig::new([dim * 4, cfg.num_attention_heads], [1, 1]).init(device);
        Self { pos_proj, inv_freq }
    }

    /// Returns the relative-position bias `[B, heads, S, S]` for center-form boxes
    /// `[B, S, 4]` (cx, cy, w, h).
    fn forward(&self, boxes: Tensor<B, 3>) -> Tensor<B, 4> {
        let [bsz, seq, _] = boxes.dims();
        let device = boxes.device();
        let eps = 1e-5f64;

        // box_relative_encoding: source unsqueezed at -2, target at -3.
        let src = boxes.clone().reshape([bsz, seq, 1, 4]);
        let tgt = boxes.reshape([bsz, 1, seq, 4]);
        let src_c = src.clone().narrow(3, 0, 2);
        let src_d = src.narrow(3, 2, 2);
        let tgt_c = tgt.clone().narrow(3, 0, 2);
        let tgt_d = tgt.narrow(3, 2, 2);

        let diff = (src_c.expand([bsz, seq, seq, 2]) - tgt_c.expand([bsz, seq, seq, 2])).abs();
        let src_d = src_d.expand([bsz, seq, seq, 2]);
        let tgt_d = tgt_d.expand([bsz, seq, seq, 2]);
        let rel_coord = diff.div(src_d.clone().add_scalar(eps)).add_scalar(1.0).log();
        let rel_dim = src_d.add_scalar(eps).div(tgt_d.add_scalar(eps)).log();
        let rel = Tensor::cat(vec![rel_coord, rel_dim], 3); // [B, S, S, 4]

        // get_position_embedding: (rel*scale) ⊗ inv_freq -> [sin, cos] flatten.
        let pos = self.position_embedding(rel, RO.relation_bias_scale, &device); // [B,S,S,4*embed_dim]
        // permute to [B, C, S, S] and project.
        let pos = pos.permute([0, 3, 1, 2]);
        self.pos_proj.forward(pos)
    }

    /// Builds the sinusoidal embedding of the relative encoding tensor `[B,S,S,4]`.
    fn position_embedding(&self, x: Tensor<B, 4>, scale: f64, device: &B::Device) -> Tensor<B, 4> {
        // For each of the last-dim-4 relative components, multiply by scale and by
        // each inv_freq, then take sin and cos, and flatten the (component, freq)
        // axes. Output last dim = 4 * (2 * inv_freq.len()) = 4 * embed_dim.
        let [bsz, s0, s1, comps] = x.dims();
        let data = mineru_burn_common::float_to_vec_f32(x.mul_scalar(scale));
        let nfreq = self.inv_freq.len();
        let out_dim = comps * 2 * nfreq;
        let mut out = vec![0f32; bsz * s0 * s1 * out_dim];
        for i in 0..(bsz * s0 * s1) {
            for c in 0..comps {
                let val = data[i * comps + c] as f64;
                let base = i * out_dim + c * 2 * nfreq;
                for (f, &freq) in self.inv_freq.iter().enumerate() {
                    let a = val * freq;
                    out[base + f] = a.sin() as f32;
                    out[base + nfreq + f] = a.cos() as f32;
                }
            }
        }
        Tensor::<B, 1>::from_data(TensorData::new(out, [bsz * s0 * s1 * out_dim]), device)
            .reshape([bsz, s0, s1, out_dim])
    }
}

/// One reading-order transformer layer (LayoutLMv3-style with `query`/`key`/`value`
/// and `output.dense`/`output.norm` naming).
#[derive(Module, Debug)]
pub struct RoLayer<B: Backend> {
    // attention.self.{query,key,value}
    attn_query: PtLinear<B>,
    attn_key: PtLinear<B>,
    attn_value: PtLinear<B>,
    // attention.output.{dense,norm}
    attn_out_dense: PtLinear<B>,
    attn_out_norm: PtLayerNorm<B>,
    // intermediate.dense
    intermediate_dense: PtLinear<B>,
    // output.{dense,norm}
    out_dense: PtLinear<B>,
    out_norm: PtLayerNorm<B>,
    #[module(skip)]
    num_heads: usize,
}

impl<B: Backend> RoLayer<B> {
    fn init(device: &B::Device) -> Self {
        let hs = RO.hidden_size;
        Self {
            attn_query: PtLinear::init(hs, hs, true, device),
            attn_key: PtLinear::init(hs, hs, true, device),
            attn_value: PtLinear::init(hs, hs, true, device),
            attn_out_dense: PtLinear::init(hs, hs, true, device),
            attn_out_norm: PtLayerNorm::init(hs, RO.layer_norm_eps, device),
            intermediate_dense: PtLinear::init(hs, RO.intermediate_size, true, device),
            out_dense: PtLinear::init(RO.intermediate_size, hs, true, device),
            out_norm: PtLayerNorm::init(hs, RO.layer_norm_eps, device),
            num_heads: RO.num_attention_heads,
        }
    }

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        attention_mask: Tensor<B, 4>,
        rel_2d_pos: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [bsz, seq, hs] = hidden.dims();
        let head_dim = hs / self.num_heads;

        let q = self
            .attn_query
            .forward(hidden.clone())
            .reshape([bsz, seq, self.num_heads, head_dim])
            .swap_dims(1, 2);
        let k = self
            .attn_key
            .forward(hidden.clone())
            .reshape([bsz, seq, self.num_heads, head_dim])
            .swap_dims(1, 2);
        let v = self
            .attn_value
            .forward(hidden.clone())
            .reshape([bsz, seq, self.num_heads, head_dim])
            .swap_dims(1, 2);

        // scores = (q / sqrt(head_dim)) @ kᵀ + rel_2d_pos + attention_mask
        let scale = (head_dim as f64).sqrt();
        let mut scores = q.div_scalar(scale).matmul(k.swap_dims(2, 3));
        scores = scores.add(rel_2d_pos).add(attention_mask);
        let probs = cogview_attention(scores, RO.cogview_alpha);
        // ctx = probs @ v, written as `(vᵀ @ probsᵀ)ᵀ`: `v` is a batch-permuted
        // (`swap_dims(1, 2)`) view and Burn 0.21's wgpu matmul reads a batch-permuted
        // right-hand operand with wrong strides. Keeping it on the left with a
        // last-two-dim transpose on the right avoids the bug (a no-op on CPU).
        // See the matching note in `encoder.rs`.
        let ctx = v
            .swap_dims(2, 3)
            .matmul(probs.swap_dims(2, 3))
            .swap_dims(2, 3)
            .swap_dims(1, 2)
            .reshape([bsz, seq, hs]);

        // self-output: dense -> +residual -> norm
        let attn = self.attn_out_norm.forward(self.attn_out_dense.forward(ctx).add(hidden));
        // intermediate (gelu) + output
        let inter = gelu(self.intermediate_dense.forward(attn.clone()));
        self.out_norm.forward(self.out_dense.forward(inter).add(attn))
    }
}

/// CogView-stabilised softmax over the last dim.
fn cogview_attention<B: Backend>(scores: Tensor<B, 4>, alpha: f64) -> Tensor<B, 4> {
    let scaled = scores.div_scalar(alpha);
    let max = scaled.clone().max_dim(3);
    softmax(scaled.sub(max).mul_scalar(alpha), 3)
}

/// The GlobalPointer head (`PPDocLayoutV2GlobalPointer`).
#[derive(Module, Debug)]
pub struct GlobalPointer<B: Backend> {
    dense: PtLinear<B>,
    #[module(skip)]
    head_size: usize,
}

impl<B: Backend> GlobalPointer<B> {
    fn init(device: &B::Device) -> Self {
        Self {
            dense: PtLinear::init(RO.hidden_size, RO.global_pointer_head_size * 2, true, device),
            head_size: RO.global_pointer_head_size,
        }
    }

    /// `inputs`: `[B, S, hidden]` → ordering logits `[B, S, S]`.
    fn forward(&self, inputs: Tensor<B, 3>) -> Tensor<B, 3> {
        let [bsz, seq, _] = inputs.dims();
        let device = inputs.device();
        let proj = self.dense.forward(inputs).reshape([bsz, seq, 2, self.head_size]);
        let queries = proj.clone().narrow(2, 0, 1).squeeze_dim::<3>(2); // [B,S,hd]
        let keys = proj.narrow(2, 1, 1).squeeze_dim::<3>(2); // [B,S,hd]
        let logits = queries
            .matmul(keys.swap_dims(1, 2))
            .div_scalar((self.head_size as f64).sqrt()); // [B,S,S]

        // masked_fill lower triangle (tril, diagonal 0) with -1e4.
        let ones = Tensor::<B, 2>::ones([seq, seq], &device);
        let tri = ones.tril(0).reshape([1, seq, seq]).expand([bsz, seq, seq]);
        let mask = tri.greater_elem(0.5);
        logits.mask_fill(mask, -1e4)
    }
}

/// The full reading-order network.
#[derive(Module, Debug)]
pub struct ReadingOrder<B: Backend> {
    embeddings: TextEmbeddings<B>,
    label_embeddings: burn::nn::Embedding<B>,
    label_features_projection: PtLinear<B>,
    encoder: RoEncoder<B>,
    relative_head: GlobalPointer<B>,
}

/// The reading-order transformer encoder: layers + the relative-bias module.
#[derive(Module, Debug)]
pub struct RoEncoder<B: Backend> {
    layer: Vec<RoLayer<B>>,
    rel_bias_module: PositionRelationEmbedding<B>,
}

impl<B: Backend> ReadingOrder<B> {
    /// Initialises the reading-order network.
    pub fn init(device: &B::Device) -> Self {
        use burn::nn::EmbeddingConfig;
        Self {
            embeddings: TextEmbeddings::init(device),
            label_embeddings: EmbeddingConfig::new(RO.num_classes, RO.hidden_size).init(device),
            label_features_projection: PtLinear::init(RO.hidden_size, RO.hidden_size, true, device),
            encoder: RoEncoder {
                layer: (0..RO.num_hidden_layers).map(|_| RoLayer::init(device)).collect(),
                rel_bias_module: PositionRelationEmbedding::init(&RO, device),
            },
            relative_head: GlobalPointer::init(device),
        }
    }

    /// Runs the reading-order head.
    ///
    /// - `boxes`: `[B, num_q, 4]` padded boxes in `[0, 1000]` (xyxy), float.
    /// - `labels`: `[B, num_q]` reading-order category ids.
    /// - `keep_mask`: `[B, num_q]` bool-as-float, 1 where a query was kept.
    ///
    /// Returns `order_logits [B, num_q, num_q]` (the query slots of the padded
    /// sequence), matching the reference's `[:, :, :num_queries]` slice.
    pub fn forward(
        &self,
        boxes: Tensor<B, 3>,
        labels: Tensor<B, 2, Int>,
        keep_mask: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let [bsz, num_q, _] = boxes.dims();
        let device = boxes.device();
        let seq = num_q; // reference uses seq_len = mask.shape[1] = num_queries

        // num_pred per batch = sum(keep_mask).
        let num_pred = keep_mask.clone().sum_dim(1).reshape([bsz]); // [B]
        let num_pred_vals = mineru_burn_common::float_to_vec_f32(num_pred);

        // Build padded input_ids [B, seq+2], boxes, labels, position_ids on host.
        let ext = seq + 2;
        let mut input_ids = vec![RO.pad_token_id; bsz * ext];
        for b in 0..bsz {
            let np = num_pred_vals.get(b).copied().unwrap_or(0.0) as usize;
            input_ids[b * ext] = RO.start_token_id;
            for j in 1..=np.min(seq) {
                input_ids[b * ext + j] = RO.pred_token_id;
            }
            let end_pos = (np + 1).min(ext - 1);
            input_ids[b * ext + end_pos] = RO.end_token_id;
        }
        let input_ids_t =
            Tensor::<B, 1, Int>::from_data(TensorData::new(int_i64(&input_ids), [bsz * ext]), &device)
                .reshape([bsz, ext]);

        // pad boxes: prepend/append a zero box -> [B, seq+2, 4].
        let zero_box = Tensor::<B, 3>::zeros([bsz, 1, 4], &device);
        let pad_boxes = Tensor::cat(vec![zero_box.clone(), boxes.clone(), zero_box], 1);
        let pad_boxes_int = pad_boxes.clone().int();

        // embeddings (word+type+position+spatial).
        let position_ids = create_position_ids(&input_ids, bsz, ext, RO.pad_token_id, &device);
        let bbox_embedding = self
            .embeddings
            .forward(input_ids_t.clone(), pad_boxes_int, position_ids.clone());

        // label projection (padded like boxes).
        let label_zero = Tensor::<B, 3>::zeros([bsz, 1, RO.hidden_size], &device);
        let label_proj_core = self
            .label_features_projection
            .forward(self.label_embeddings.forward(labels));
        let label_proj = Tensor::cat(vec![label_zero.clone(), label_proj_core, label_zero], 1);

        let final_embeddings = self.embeddings.norm.forward(bbox_embedding.add(label_proj));

        // attention mask: keep positions < num_pred + 2.
        let attention_mask = bidirectional_mask(&num_pred_vals, bsz, ext, &device);

        // 2D relative bias from padded boxes (center form).
        let rel_2d = self.encoder.rel_bias_module.forward(center_form(pad_boxes));

        // encoder layers.
        let mut hidden = final_embeddings;
        for layer in &self.encoder.layer {
            hidden = layer.forward(hidden, attention_mask.clone(), rel_2d.clone());
        }

        // token = hidden[:, 1 : 1+seq, :]
        let token = hidden.narrow(1, 1, seq);
        self.relative_head.forward(token)
    }
}

/// Converts xyxy boxes `[B, S, 4]` to center form `[cx, cy, w, h]` with clamped wh.
fn center_form<B: Backend>(bbox: Tensor<B, 3>) -> Tensor<B, 3> {
    let x0 = bbox.clone().narrow(2, 0, 1);
    let y0 = bbox.clone().narrow(2, 1, 1);
    let x1 = bbox.clone().narrow(2, 2, 1);
    let y1 = bbox.narrow(2, 3, 1);
    let w = (x1.clone() - x0.clone()).clamp(1e-3, f32::MAX);
    let h = (y1.clone() - y0.clone()).clamp(1e-3, f32::MAX);
    let cx = (x0 + x1).mul_scalar(0.5);
    let cy = (y0 + y1).mul_scalar(0.5);
    Tensor::cat(vec![cx, cy, w, h], 2)
}

/// Builds the additive bidirectional attention mask `[B, 1, ext, ext]`.
///
/// Positions `< num_pred + 2` are attended (0), the rest are masked (large
/// negative), matching `_create_bidirectional_mask` over `pred_col_idx < num_pred+2`.
fn bidirectional_mask<B: Backend>(
    num_pred: &[f32],
    bsz: usize,
    ext: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let neg = -1.0e30_f32.max(f32::MIN);
    let mut data = vec![0f32; bsz * ext];
    for b in 0..bsz {
        let limit = num_pred.get(b).copied().unwrap_or(0.0) as usize + 2;
        for j in 0..ext {
            data[b * ext + j] = if j < limit { 0.0 } else { neg };
        }
    }
    // [B, 1, 1, ext] broadcast over the query dim inside attention.
    Tensor::<B, 1>::from_data(TensorData::new(data, [bsz * ext]), device).reshape([bsz, 1, 1, ext])
}

/// Position ids: `padding_idx + 1 + cumulative non-pad count`, matching
/// `create_position_ids_from_input_ids`.
fn create_position_ids<B: Backend>(
    input_ids: &[i64],
    bsz: usize,
    ext: usize,
    pad_token_id: i64,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let mut ids = vec![0i64; bsz * ext];
    for b in 0..bsz {
        let mut running = 0i64;
        for j in 0..ext {
            let tok = input_ids[b * ext + j];
            if tok != pad_token_id {
                running += 1;
                ids[b * ext + j] = running + pad_token_id;
            } else {
                ids[b * ext + j] = pad_token_id;
            }
        }
    }
    Tensor::<B, 1, Int>::from_data(TensorData::new(ids, [bsz * ext]), device).reshape([bsz, ext])
}

/// Copies a `&[i64]` for `TensorData` (which needs an owned `Vec`).
fn int_i64(v: &[i64]) -> Vec<i64> {
    v.to_vec()
}
