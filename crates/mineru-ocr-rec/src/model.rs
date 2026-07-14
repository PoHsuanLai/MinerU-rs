//! End-to-end SVTR/CRNN + CTC text recognizer.
//!
//! Ties backbone → CTC head → CTC decode together, mirroring
//! `TextRecognizer.__call__` / `resize_norm_img` in
//! `pytorchocr/tools/infer/predict_rec.py`:
//!
//! 1. **Resize** the crop to height 48, width `ceil(48 * aspect)` (capped), keeping
//!    aspect ratio and zero-padding the right side to the target width.
//! 2. **Normalize** with `x / 127.5 - 1` (i.e. mean 0.5, std 0.5) in **BGR** channel
//!    order — the channel order the PaddleOCR/cv2 pipeline trained on.
//! 3. Run backbone + CTC head → raw logits `[1, T, num_classes]`.
//! 4. **CTC greedy decode** ([`mineru_burn_common::ctc`]) → class indices; map to a
//!    string via the [`CharDict`], and average the per-step max-softmax probability
//!    for the confidence score.
//!
//! [`TextRecognizer::recognize`] returns `(String, f32)` per crop.

use std::path::Path;

use burn::module::Module;
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use image::{imageops::FilterType, RgbImage};
use mineru_burn_common::ctc::ctc_greedy_decode_slice;
use mineru_burn_common::weights::{Coverage, KeyRemap, load_weights};

use crate::backbone::{PpLcNetV4Rec, REC_OUT_CHANNELS};
use crate::dict::CharDict;
use crate::error::{Error, Result};
use crate::head::CtcMultiHead;

/// The loadable network: PP-LCNetV4 backbone + LightSVTR/CTC head.
///
/// Grouped as one [`Module`] so the whole PP-OCRv6 checkpoint loads in a single
/// strict pass — every source key (both `model.backbone.*` and `head.*`) must land
/// in a real field. The field names (`backbone`, `head`) are chosen so the remap
/// only bridges structural prefix differences (see [`TextRecognizer::build_remap`]).
#[derive(Module, Debug)]
pub struct RecNet<B: Backend> {
    backbone: PpLcNetV4Rec<B>,
    head: CtcMultiHead<B>,
}

/// Fixed recognition input geometry and normalization.
#[derive(Debug, Clone, Copy)]
pub struct RecConfig {
    /// Target crop height (48 for PP-OCRv6).
    pub image_height: u32,
    /// Base target width before aspect scaling (`3, 48, 320` → 320).
    pub image_width: u32,
    /// Maximum padded width.
    pub limited_max_width: u32,
    /// Minimum padded width.
    pub limited_min_width: u32,
}

impl Default for RecConfig {
    fn default() -> Self {
        // Matches predict_rec defaults: rec_image_shape 3,48,320; width clamps.
        Self {
            image_height: 48,
            image_width: 320,
            limited_max_width: 2560,
            limited_min_width: 16,
        }
    }
}

/// A single activation dump: flat row-major `f32` data plus its rank-`N` shape.
/// Used only by the parity hook [`TextRecognizer::forward_stages`].
#[doc(hidden)]
pub type StageDump = (Vec<f32>, Vec<usize>);

/// The per-stage activations returned by [`TextRecognizer::forward_stages`]:
/// `(backbone_stages, backbone_pooled, neck_out, logits)`.
#[doc(hidden)]
pub type RecStageDumps = (Vec<StageDump>, StageDump, StageDump, StageDump);

/// LightSVTR CTC recognizer: PP-LCNetV4 backbone → CTC head → CTC decode.
pub struct TextRecognizer<B: Backend> {
    net: RecNet<B>,
    dict: CharDict,
    config: RecConfig,
    device: B::Device,
}

impl<B: Backend> TextRecognizer<B> {
    /// Builds the PP-OCRv6 *small rec* network (unloaded) on `device`.
    ///
    /// `dict` must already include the blank/space handling (build it via
    /// [`CharDict::from_file`]); its class count sizes the CTC head.
    pub fn new(dict: CharDict, config: RecConfig, device: B::Device) -> Self {
        let backbone = PpLcNetV4Rec::new(&device);
        // PP-OCRv6 small rec head: lightsvtr neck dims=120, depth=2, heads=8,
        // mlp_ratio=2.0, local_kernel=7.
        let head = CtcMultiHead::new(
            REC_OUT_CHANNELS,
            120,
            2,
            8,
            2.0,
            7,
            dict.num_classes(),
            &device,
        );
        Self {
            net: RecNet { backbone, head },
            dict,
            config,
            device,
        }
    }

    /// Loads PP-OCRv6 safetensors (or `.pth`) weights into the network.
    ///
    /// Loads the whole checkpoint in a single [`Coverage::Strict`] pass into the
    /// combined [`RecNet`]: every source key (`model.backbone.*` and `head.*`) must
    /// land in a real field or the load fails with the unmapped list. The remap only
    /// bridges structural prefix differences (see [`Self::build_remap`]).
    pub fn load_weights(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let remap = self.build_remap()?;
        load_weights::<B, _>(&mut self.net, path, &remap, Coverage::Strict)?;
        Ok(())
    }

    /// Builds the checkpoint-key → field-path remap for [`RecNet`].
    ///
    /// The module field names already match the checkpoint leaf names, so the remap
    /// only bridges *structural* prefix differences:
    /// - the doubly-nested backbone prefix `model.backbone.` → `backbone.` (the
    ///   `RecNet` field). Head keys are already `head.*` = the `head` field name;
    /// - the SE module's `convolutions.0`/`.2` `ModuleList` indices → the named
    ///   `reduce`/`expand` fields;
    /// - the depthwise `token_conv` that is a *raw* `nn.Conv2d` (rep case: unit
    ///   stride, `in == out`) stores `token_conv.{weight,bias}` directly, which this
    ///   crate holds in a separate typed field `token_conv_rep` (Burn cannot store
    ///   two different module types under one field name). This routes those two
    ///   leaves to `token_conv_rep`; the strided `token_conv` (a ConvLayer with
    ///   `token_conv.convolution.*`/`.normalization.*`) is left untouched.
    fn build_remap(&self) -> Result<KeyRemap> {
        let map_err = |e: mineru_burn_common::Error| Error::Common(e);
        KeyRemap::new()
            .rename(r"^model\.backbone\.", "backbone.")
            .map_err(map_err)?
            .rename(r"\.token_conv\.(weight|bias)$", ".token_conv_rep.$1")
            .map_err(map_err)?
            .rename(r"\.convolutions\.0\.", ".reduce.")
            .map_err(map_err)?
            .rename(r"\.convolutions\.2\.", ".expand.")
            .map_err(map_err)
    }

    /// Recognizes the text in a single crop, returning `(text, mean_confidence)`.
    pub fn recognize(&self, crop: &RgbImage) -> Result<(String, f32)> {
        let input = self.preprocess(crop)?;
        let logits = self.forward(input)?;

        let dims = logits.dims();
        if dims[0] != 1 {
            return Err(Error::LogitsShape(format!("expected batch 1, got {dims:?}")));
        }
        let (t, c) = (dims[1], dims[2]);
        // Host read of the full logits: both the CTC greedy decode (repeat-collapse
        // over a `Vec`) and the confidence softmax are host-side reductions, so the
        // copy cannot be eliminated. Use the dtype-agnostic helper so it is correct on
        // backends whose float storage isn't `f32` (e.g. `wgpu`).
        let data = mineru_burn_common::float_to_vec_f32(logits);

        // Greedy decode (collapse repeats, drop blank) on the raw logits.
        let indices = ctc_greedy_decode_slice(&data, t, c, 0);
        let text = self.dict.decode(&indices);
        let score = mean_max_softmax(&data, t, c);
        Ok((text, score))
    }

    /// Test/parity hook: runs the network on an already-preprocessed `[1, 3, H, W]`
    /// tensor and returns every stage's activations as flat row-major `f32` data
    /// paired with its shape.
    ///
    /// Returns `(backbone_stages, backbone_pooled, neck_out, logits)`:
    /// - `backbone_stages`: the four `PPLCNetV4Block` outputs in order;
    /// - `backbone_pooled`: the `avg_pool2d([3, 2])` height-pooled feature the head
    ///   consumes (`[1, C, 1, W]`);
    /// - `neck_out`: the LightSVTR neck output before squeeze/permute (`[1, dims, 1, W]`);
    /// - `logits`: the raw CTC logits `[1, T, num_classes]`.
    ///
    /// Bypasses resize/normalise so a caller can feed the exact tensor the Python
    /// reference used and diff each stage to localise any divergence. Not part of the
    /// public recognition API — it exists for the numerical-parity test.
    #[doc(hidden)]
    pub fn forward_stages(&self, input: Tensor<B, 4>) -> Result<RecStageDumps> {
        // Dtype-agnostic host read so the parity hook works on any backend's float
        // storage; library code never panics, so no `.expect`.
        let dump = |t: &Tensor<B, 4>| -> StageDump {
            let d = t.dims().to_vec();
            let v = mineru_burn_common::float_to_vec_f32(t.clone());
            (v, d)
        };
        let (stages, pooled) = self
            .net
            .backbone
            .forward_stages(input)
            .ok_or_else(|| Error::LogitsShape("backbone feature height < 3".into()))?;
        let stage_dumps: Vec<_> = stages.iter().map(dump).collect();
        let pooled_dump = dump(&pooled);
        let (neck, logits) = self.net.head.forward_stages(pooled);
        let neck_dump = dump(&neck);
        let ld = logits.dims().to_vec();
        let logits_v = mineru_burn_common::float_to_vec_f32(logits);
        Ok((stage_dumps, pooled_dump, neck_dump, (logits_v, ld)))
    }

    /// Test/parity hook: preprocesses `crop` exactly as [`TextRecognizer::recognize`]
    /// does, returning the `[1, 3, H, W]` input tensor. Lets a parity test drive
    /// [`TextRecognizer::forward_stages`] with the same tensor the Python reference
    /// consumed. Not part of the public API.
    #[doc(hidden)]
    pub fn preprocess_at(&self, crop: &RgbImage) -> Result<Tensor<B, 4>> {
        self.preprocess(crop)
    }

    /// Runs backbone + head, returning raw logits `[1, T, num_classes]`.
    fn forward(&self, input: Tensor<B, 4>) -> Result<Tensor<B, 3>> {
        let feat = self
            .net
            .backbone
            .forward(input)
            .ok_or_else(|| Error::LogitsShape("backbone feature height < 3".into()))?;
        Ok(self.net.head.forward(feat))
    }

    /// Resizes/normalizes a crop into a `[1, 3, H, W]` BGR tensor with right padding.
    ///
    /// Mirrors `resize_norm_img`: aspect-preserving resize to height `image_height`,
    /// width `ceil(h_ratio)` capped at the padded target, then `x/127.5 - 1`.
    fn preprocess(&self, crop: &RgbImage) -> Result<Tensor<B, 4>> {
        let (src_w, src_h) = (crop.width(), crop.height());
        if src_w == 0 || src_h == 0 {
            return Err(Error::Config("crop has a zero dimension".into()));
        }
        let img_h = self.config.image_height;

        // Target padded width: base image_width, but at least min and at most max,
        // and grown with aspect ratio like the reference `max_wh_ratio` path. For a
        // single crop we use its own aspect ratio.
        let aspect = src_w as f32 / src_h as f32;
        let mut target_w = (img_h as f32 * aspect).ceil() as u32;
        target_w = target_w
            .max(self.config.limited_min_width)
            .min(self.config.limited_max_width);
        // The reference pads into a canvas of width max(image_width, h*aspect).
        let canvas_w = target_w.max(self.config.image_width).min(self.config.limited_max_width);

        // Aspect-preserving resized width (<= canvas_w).
        let resized_w = ((img_h as f32 * aspect).ceil() as u32)
            .max(1)
            .min(canvas_w);

        let resized = image::imageops::resize(crop, resized_w, img_h, FilterType::Triangle);

        let (cw, ch) = (canvas_w as usize, img_h as usize);
        // CHW, BGR order, normalized x/127.5 - 1; right-padded with zeros.
        let mut data = vec![0.0f32; 3 * ch * cw];
        // RGB channel c -> BGR plane index: R(0)->2, G(1)->1, B(2)->0.
        let bgr_plane = [2usize, 1, 0];
        for (rgb_c, &plane_c) in bgr_plane.iter().enumerate() {
            let plane = &mut data[plane_c * ch * cw..(plane_c + 1) * ch * cw];
            for y in 0..ch {
                for x in 0..(resized_w as usize) {
                    let px = resized.get_pixel(x as u32, y as u32);
                    let v = px.0[rgb_c] as f32;
                    plane[y * cw + x] = v / 127.5 - 1.0;
                }
            }
        }

        let tensor = Tensor::<B, 1>::from_data(TensorData::new(data, [3 * ch * cw]), &self.device);
        Ok(tensor.reshape([1, 3, ch, cw]))
    }

    /// Access the loaded character dictionary.
    pub fn dict(&self) -> &CharDict {
        &self.dict
    }
}

/// Mean over timesteps of the max softmax probability — the CTC confidence score.
///
/// Reproduces `CTCLabelDecode._decode_raw_logits` averaged over the kept steps: for
/// each step compute `exp(max_logit - logsumexp(logits))`, then average across all
/// timesteps. (The reference averages only over non-blank kept steps; averaging over
/// all steps is a close, stable proxy and avoids a second decode pass.)
fn mean_max_softmax(logits: &[f32], t: usize, c: usize) -> f32 {
    if t == 0 || c == 0 {
        return 1.0;
    }
    let mut acc = 0.0f64;
    let mut steps = 0u64;
    for step in 0..t {
        let start = step * c;
        let Some(row) = logits.get(start..start + c) else {
            break;
        };
        let max = row.iter().cloned().fold(f32::MIN, f32::max);
        let sum_exp: f64 = row.iter().map(|&v| ((v - max) as f64).exp()).sum();
        // exp(max - logsumexp) = 1 / sum(exp(v - max)).
        let prob = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
        acc += prob;
        steps += 1;
    }
    if steps == 0 {
        1.0
    } else {
        (acc / steps as f64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_max_softmax_of_confident_steps_is_high() {
        // 2 steps, 3 classes; each step strongly favours one class.
        let logits = [
            10.0, 0.0, 0.0, // step 0 -> ~1.0
            0.0, 10.0, 0.0, // step 1 -> ~1.0
        ];
        let s = mean_max_softmax(&logits, 2, 3);
        assert!(s > 0.99, "confident steps should score ~1.0, got {s}");
    }

    #[test]
    fn mean_max_softmax_of_uniform_is_low() {
        // Uniform logits over 4 classes -> max softmax = 0.25.
        let logits = [0.0, 0.0, 0.0, 0.0];
        let s = mean_max_softmax(&logits, 1, 4);
        assert!((s - 0.25).abs() < 1e-4, "uniform -> 0.25, got {s}");
    }
}
