//! MultiHead CTC classifier (PP-OCRv6 LightSVTR branch).
//!
//! A faithful Burn port of the inference path through `MultiHead` in
//! `pytorchocr/modeling/heads/rec_multi_head.py`, restricted to the CTC branch that
//! PP-OCRv6 uses. The reference wraps the [`EncoderWithLightSvtr`](crate::neck)
//! neck and a final `nn.Linear` classifier under `head.encoder.*` and
//! `head.head.*`; this module reproduces that nesting so weights load 1:1.
//!
//! Inference forward (`use_light_svtr_head` path):
//! ```text
//! ctc = encoder(x)                       # [N, dims, 1, W]
//! ctc = ctc.squeeze(2).permute(0, 2, 1)  # [N, W, dims]
//! logits = head(ctc)                     # [N, W, num_classes]  (raw logits)
//! ```
//! Raw logits are returned; CTC greedy decode + softmax-max confidence happen in
//! [`crate::model`] via [`mineru_burn_common::ctc`].

use burn::module::Module;
use burn::prelude::Backend;
use burn::tensor::Tensor;
use mineru_burn_common::nn::PtLinear;

use crate::neck::EncoderWithLightSvtr;

/// The PP-OCRv6 CTC head: LightSVTR neck (`encoder`) + linear classifier (`head`).
#[derive(Module, Debug)]
pub struct CtcMultiHead<B: Backend> {
    encoder: EncoderWithLightSvtr<B>,
    head: PtLinear<B>,
    #[module(skip)]
    num_classes: usize,
}

impl<B: Backend> CtcMultiHead<B> {
    /// Builds the head. `in_channels` is the backbone output; `dims`/`depth`/
    /// `local_kernel` configure the LightSVTR neck; `num_classes` is the CTC output
    /// size (blank + dictionary + optional space).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        dims: usize,
        depth: usize,
        num_heads: usize,
        mlp_ratio: f64,
        local_kernel: usize,
        num_classes: usize,
        device: &B::Device,
    ) -> Self {
        let encoder =
            EncoderWithLightSvtr::new(in_channels, dims, depth, num_heads, mlp_ratio, local_kernel, device);
        let head = PtLinear::init(encoder.out_channels(), num_classes, true, device);
        Self {
            encoder,
            head,
            num_classes,
        }
    }

    /// The CTC output size (number of classes, including blank).
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Forward pass → raw logits `[N, T, num_classes]`.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let ctc = self.encoder.forward(x);
        // squeeze(dim=2): [N, C, 1, W] -> [N, C, W]. The backbone's height pooling
        // guarantees H == 1 here.
        let [n, c, _h, w] = ctc.dims();
        let ctc = ctc.reshape([n, c, w]);
        // permute(0, 2, 1): [N, C, W] -> [N, W, C].
        let ctc = ctc.swap_dims(1, 2);
        self.head.forward(ctc)
    }
}
