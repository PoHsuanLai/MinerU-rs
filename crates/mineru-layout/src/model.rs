//! The assembled PP-DocLayoutV2 model and its forward pass.
//!
//! Field layout mirrors the checkpoint's top-level module tree so weights load
//! under strict coverage (see [`crate::weights`] for the key remap). The forward
//! reproduces `PPDocLayoutV2ForObjectDetection.forward`: detector → per-class
//! threshold + sort → reading-order head, returning the tensors the postprocessor
//! consumes.

use burn::module::Module;
use burn::prelude::Backend;
use burn::tensor::activation::sigmoid;
use burn::tensor::{Int, Tensor, TensorData};

use crate::backbone::Backbone;
use crate::config::DET;
use crate::decoder::{Decoder, DecoderOutput};
use crate::encoder::{HybridEncoder, ProjConvBn};
use crate::label::CLASS_ORDER;
use crate::reading_order::ReadingOrder;

/// The full model graph.
///
/// `enc_output`, `enc_score_head`, `enc_bbox_head`, and `decoder_input_proj` live
/// inside [`Decoder`] for code locality, but their checkpoint keys are top-level
/// (`model.enc_output`, …); the remap in [`crate::weights`] bridges that.
#[derive(Module, Debug)]
pub struct PpDocLayoutV2<B: Backend> {
    backbone: Backbone<B>,
    encoder_input_proj: Vec<ProjConvBn<B>>,
    encoder: HybridEncoder<B>,
    decoder: Decoder<B>,
    reading_order: ReadingOrder<B>,
}

/// Raw model outputs consumed by the postprocessor.
pub struct RawOutputs<B: Backend> {
    /// Final class logits `[B, Q, num_labels]` (query order matches `pred_boxes`).
    pub logits: Tensor<B, 3>,
    /// Boxes `[B, Q, 4]`, cxcywh in `[0, 1]`.
    pub pred_boxes: Tensor<B, 3>,
    /// Reading-order pairwise logits `[B, Q, Q]`.
    pub order_logits: Tensor<B, 3>,
}

/// Per-stage activations captured for numerical parity testing.
///
/// Mirrors the tensors the Python reference dumper (`py_ref_layout.py`) writes: the
/// three backbone feature maps, the three `encoder_input_proj` outputs, the three
/// hybrid-encoder fused maps, and the final decoder/reading-order tensors. Exposed
/// only for the `#[ignore]`d parity test — not part of the inference API.
#[doc(hidden)]
pub struct ForwardStages<B: Backend> {
    /// HGNetV2 stage2/3/4 maps, channels `[512, 1024, 2048]` (low→high stride).
    pub backbone: [Tensor<B, 4>; 3],
    /// `encoder_input_proj` outputs (1×1 conv + BN → 256-ch), per level.
    pub proj: [Tensor<B, 4>; 3],
    /// Hybrid-encoder fused maps (AIFI + CCFM), all 256-ch, per level.
    pub encoder: [Tensor<B, 4>; 3],
    /// Final-layer decoder class logits `[B, Q, num_labels]`.
    pub logits: Tensor<B, 3>,
    /// Final-layer decoder boxes `[B, Q, 4]`, cxcywh in `[0, 1]`.
    pub pred_boxes: Tensor<B, 3>,
    /// Reading-order pairwise logits `[B, Q, Q]`.
    pub order_logits: Tensor<B, 3>,
}

impl<B: Backend> PpDocLayoutV2<B> {
    /// Initialises the whole graph with zeroed parameters (overwritten by loading).
    pub fn init(device: &B::Device) -> Self {
        let d = DET.d_model;
        let encoder_input_proj = DET
            .encoder_in_channels
            .iter()
            .map(|&c| ProjConvBn::init(c, d, device))
            .collect();
        Self {
            backbone: Backbone::init(device),
            encoder_input_proj,
            encoder: HybridEncoder::init(device),
            decoder: Decoder::init(device),
            reading_order: ReadingOrder::init(device),
        }
    }

    /// Runs the full forward pass on a preprocessed `[B, 3, 800, 800]` tensor.
    pub fn forward(&self, pixel_values: Tensor<B, 4>) -> RawOutputs<B> {
        let [f0, f1, f2] = self.backbone.forward(pixel_values);
        // encoder_input_proj: 1x1 conv+bn per level -> 256-ch.
        let proj = [
            self.encoder_input_proj[0].forward(f0),
            self.encoder_input_proj[1].forward(f1),
            self.encoder_input_proj[2].forward(f2),
        ];
        let encoder_maps = self.encoder.forward(proj);
        let DecoderOutput {
            logits,
            pred_boxes,
            last_hidden: _,
        } = self.decoder.forward(encoder_maps);

        // Threshold-sort the queries and run the reading-order head. Returns the
        // logits/boxes reordered onto the SAME query axis as `order_logits`, which
        // is what the reference `PPDocLayoutV2ForObjectDetection.forward` returns
        // and what the postprocessor's shared topk/gather over the three tensors
        // requires (see `sort_and_read_order`).
        let (logits, pred_boxes, order_logits) = self.sort_and_read_order(logits, pred_boxes);

        RawOutputs {
            logits,
            pred_boxes,
            order_logits,
        }
    }

    /// Runs the forward pass, capturing every intermediate stage for parity testing.
    ///
    /// Same computation as [`Self::forward`] but returns the backbone maps, the
    /// `encoder_input_proj` outputs, the hybrid-encoder maps, and the final tensors
    /// so a test can diff each stage against the Python reference. `#[doc(hidden)]`
    /// and not part of the inference API.
    #[doc(hidden)]
    pub fn forward_stages(&self, pixel_values: Tensor<B, 4>) -> ForwardStages<B> {
        let [f0, f1, f2] = self.backbone.forward(pixel_values);
        let backbone = [f0.clone(), f1.clone(), f2.clone()];

        let proj = [
            self.encoder_input_proj[0].forward(f0),
            self.encoder_input_proj[1].forward(f1),
            self.encoder_input_proj[2].forward(f2),
        ];
        let encoder_maps = self.encoder.forward(proj.clone());

        let DecoderOutput {
            logits,
            pred_boxes,
            last_hidden: _,
        } = self.decoder.forward(encoder_maps.clone());

        let (logits, pred_boxes, order_logits) = self.sort_and_read_order(logits, pred_boxes);

        ForwardStages {
            backbone,
            proj,
            encoder: encoder_maps,
            logits,
            pred_boxes,
            order_logits,
        }
    }

    /// Reproduces the threshold sort the reference does, reorders the decoder
    /// `logits`/`pred_boxes` onto that query axis, and runs the reading-order head.
    ///
    /// The reference (`PPDocLayoutV2ForObjectDetection.forward`): from the raw
    /// cxcywh boxes it takes the argmax class + its sigmoid prob, applies per-class
    /// thresholds to build a keep mask, argsorts the mask descending so kept queries
    /// float to the front, and reorders `logits`, `pred_boxes`, the xyxy boxes, and
    /// the class ids by that SAME permutation. The xyxy boxes are zeroed on dropped
    /// slots, labels remapped through `CLASS_ORDER`, and fed to the head; the order
    /// logits are the `[:, :, :num_queries]` slice.
    ///
    /// All three returned tensors (`logits`, `pred_boxes`, `order_logits`) share one
    /// query axis, which the postprocessor relies on: it runs a single topk over the
    /// flattened logit scores and gathers boxes and order sequences by that index.
    ///
    /// The permutation is a *stable* descending partition (kept queries first in
    /// original order, then dropped in original order). PyTorch's `argsort(
    /// descending=True)` is non-stable and its exact tie-break is implementation
    /// defined and not reproducible across backends; since the reorder is purely a
    /// consistent relabelling of the query axis (the postprocessor is invariant to
    /// it), the stable partition is the reproducible canonical form. The reference
    /// dumper emits its `logits`/`pred_boxes` under the same stable sort so the
    /// parity comparison is well defined.
    fn sort_and_read_order(
        &self,
        logits: Tensor<B, 3>,
        pred_boxes: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let device = logits.device();
        let [bsz, num_q, num_cls] = logits.dims();

        // xyxy×1000 clamped (reading-order head input).
        let centers = pred_boxes.clone().narrow(2, 0, 2);
        let sizes = pred_boxes.clone().narrow(2, 2, 2);
        let x0y0 = centers.clone().sub(sizes.clone().mul_scalar(0.5));
        let x1y1 = centers.add(sizes.mul_scalar(0.5));
        let boxes_xyxy = Tensor::cat(vec![x0y0, x1y1], 2)
            .mul_scalar(1000.0)
            .clamp(0.0, 1000.0); // [B, Q, 4]

        // argmax class + sigmoid prob, threshold per class.
        let (max_logit, class_ids) = logits.clone().max_dim_with_indices(2); // [B,Q,1]
        let max_logit = max_logit.reshape([bsz, num_q]);
        let class_ids = class_ids.reshape([bsz, num_q]);
        let probs = sigmoid(max_logit); // [B,Q]

        let thresholds = gather_thresholds::<B>(&class_ids, &device); // [B,Q]
        let keep = probs.greater_equal(thresholds); // bool [B,Q]
        let keep_f = keep.clone().float();

        // Stable descending keep-first permutation, computed on CPU because Burn's
        // `argsort` is unstable (it would scramble the dropped-query block, which
        // matters once we reorder `logits`/`pred_boxes` whose dropped slots are not
        // zeroed).
        let order = stable_keep_first_order::<B>(&keep, &device); // [B,Q]

        // Reorder the returned decoder outputs onto the sorted query axis.
        let sorted_logits = gather_rows_f::<B>(logits, order.clone(), num_cls);
        let sorted_pred_boxes = gather_rows_f::<B>(pred_boxes, order.clone(), 4);

        // Reading-order head inputs, on the same axis.
        let sorted_boxes = gather_rows_f::<B>(boxes_xyxy, order.clone(), 4);
        let sorted_class = gather_scalar::<B>(class_ids, order.clone());
        let sorted_keep = gather_scalar_f::<B>(keep_f, order);

        // zero dropped boxes, remap labels through CLASS_ORDER.
        let keep3 = sorted_keep.clone().reshape([bsz, num_q, 1]).expand([bsz, num_q, 4]);
        let pad_boxes = sorted_boxes.mul(keep3);
        let sorted_keep_i = sorted_keep.clone().int();
        let masked_class = sorted_class.mul(sorted_keep_i); // zero where dropped
        let ro_labels = remap_class_order::<B>(&masked_class, &device);

        // The head's keep mask is the *unsorted* mask in the reference; it only
        // uses its row-sum (per-batch count), which is order-invariant, so the
        // sorted mask is equivalent for counting.
        let order_logits = self.reading_order.forward(pad_boxes, ro_labels, sorted_keep);
        (sorted_logits, sorted_pred_boxes, order_logits)
    }
}

/// Builds the stable descending keep-first permutation `[B, Q]`: for each batch
/// row, kept queries (`keep == true`) first in ascending original-index order,
/// then dropped queries in ascending original-index order.
///
/// Matches PyTorch `argsort(keep.int(), dim=1, descending=True, stable=True)`.
/// Computed on the host because Burn's tensor `argsort` is unstable.
fn stable_keep_first_order<B: Backend>(keep: &Tensor<B, 2, burn::tensor::Bool>, device: &B::Device) -> Tensor<B, 2, Int> {
    let [bsz, num_q] = keep.dims();
    let flags = keep
        .clone()
        .int()
        .into_data()
        .into_vec::<i64>()
        .unwrap_or_else(|_| vec![0; bsz * num_q]);
    let mut order: Vec<i64> = Vec::with_capacity(bsz * num_q);
    for b in 0..bsz {
        let row = &flags[b * num_q..(b + 1) * num_q];
        order.extend((0..num_q).filter(|&q| row[q] != 0).map(|q| q as i64));
        order.extend((0..num_q).filter(|&q| row[q] == 0).map(|q| q as i64));
    }
    Tensor::<B, 1, Int>::from_data(TensorData::new(order, [bsz * num_q]), device).reshape([bsz, num_q])
}

/// Gathers per-query thresholds `[B, Q]` from `CLASS_THRESHOLDS` via class ids.
fn gather_thresholds<B: Backend>(class_ids: &Tensor<B, 2, Int>, device: &B::Device) -> Tensor<B, 2> {
    let ids = class_ids.clone().into_data().into_vec::<i64>().unwrap_or_default();
    let dims = class_ids.dims();
    let data: Vec<f32> = ids
        .iter()
        .map(|&c| {
            crate::label::CLASS_THRESHOLDS
                .get(c as usize)
                .copied()
                .unwrap_or(1.0)
        })
        .collect();
    Tensor::<B, 1>::from_data(TensorData::new(data, [dims[0] * dims[1]]), device).reshape(dims)
}

/// Remaps detection class ids to reading-order category ids via `CLASS_ORDER`.
fn remap_class_order<B: Backend>(class_ids: &Tensor<B, 2, Int>, device: &B::Device) -> Tensor<B, 2, Int> {
    let ids = class_ids.clone().into_data().into_vec::<i64>().unwrap_or_default();
    let dims = class_ids.dims();
    let data: Vec<i64> = ids
        .iter()
        .map(|&c| CLASS_ORDER.get(c as usize).copied().unwrap_or(0))
        .collect();
    Tensor::<B, 1, Int>::from_data(TensorData::new(data, [dims[0] * dims[1]]), device).reshape(dims)
}

/// Gathers rows `[B, Q, feat]` from `[B, Q, feat]` by an index `[B, Q]`.
fn gather_rows_f<B: Backend>(x: Tensor<B, 3>, idx: Tensor<B, 2, Int>, feat: usize) -> Tensor<B, 3> {
    let [bsz, num_q] = idx.dims();
    let idx = idx.reshape([bsz, num_q, 1]).expand([bsz, num_q, feat]);
    x.gather(1, idx)
}

/// Gathers a scalar-per-query int tensor `[B, Q]` by index `[B, Q]`.
fn gather_scalar<B: Backend>(x: Tensor<B, 2, Int>, idx: Tensor<B, 2, Int>) -> Tensor<B, 2, Int> {
    x.gather(1, idx)
}

/// Gathers a scalar-per-query float tensor `[B, Q]` by index `[B, Q]`.
fn gather_scalar_f<B: Backend>(x: Tensor<B, 2>, idx: Tensor<B, 2, Int>) -> Tensor<B, 2> {
    x.gather(1, idx)
}
