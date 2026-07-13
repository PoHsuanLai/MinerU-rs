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

        // Reading-order pre-processing (mirrors the reference forward).
        let order_logits = self.run_reading_order(&logits, &pred_boxes);

        RawOutputs {
            logits,
            pred_boxes,
            order_logits,
        }
    }

    /// Reproduces the box filtering/sorting the reference does before calling the
    /// reading-order head, then runs it.
    ///
    /// The reference: from the raw cxcywh boxes it computes xyxy×1000 clamped to
    /// `[0,1000]`, takes the argmax class + its sigmoid prob, applies per-class
    /// thresholds to build a keep mask, argsorts the mask descending (a stable
    /// partition that floats kept queries to the front), reorders boxes/labels,
    /// zeroes the dropped slots, remaps labels through `CLASS_ORDER`, and feeds
    /// the head. The returned order logits are the `[:, :, :num_queries]` slice.
    fn run_reading_order(&self, logits: &Tensor<B, 3>, pred_boxes: &Tensor<B, 3>) -> Tensor<B, 3> {
        let device = logits.device();
        let [bsz, num_q, num_cls] = logits.dims();

        // xyxy×1000 clamped.
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

        // argsort(keep descending) — kept slots first, stable within.
        let order = keep.clone().int().argsort(1).flip([1]); // descending via reverse of ascending
        // NOTE: argsort is ascending; flipping gives keep=1 first but reverses the
        // stable order within each group. The reference uses descending stable
        // sort; see fidelity note in weights/postprocess docs.

        let sorted_boxes = gather_rows_f::<B>(boxes_xyxy, order.clone(), 4);
        let sorted_class = gather_scalar::<B>(class_ids, order.clone());
        let sorted_keep = gather_scalar_f::<B>(keep_f, order);

        // zero dropped boxes, remap labels through CLASS_ORDER.
        let keep3 = sorted_keep.clone().reshape([bsz, num_q, 1]).expand([bsz, num_q, 4]);
        let pad_boxes = sorted_boxes.mul(keep3);
        let sorted_keep_i = sorted_keep.clone().int();
        let masked_class = sorted_class.mul(sorted_keep_i); // zero where dropped
        let ro_labels = remap_class_order::<B>(&masked_class, &device);

        let _ = num_cls;
        // The head's keep mask is the *unsorted* mask in the reference; it only
        // uses its row-sum (per-batch count), which is order-invariant, so the
        // sorted mask is equivalent for counting.
        self.reading_order.forward(pad_boxes, ro_labels, sorted_keep)
    }
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
