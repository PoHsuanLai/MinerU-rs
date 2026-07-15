//! Shared Burn harness for MinerU's model crates.
//!
//! This crate's one responsibility is to give every Burn model in MinerU
//! (`ocr-det`, `ocr-rec`, `layout`, `formula`, `table`) the same foundation, so
//! the model crates only carry their own architecture and nothing else. It owns:
//!
//! - **Backends** ([`backend`]): a default CPU alias ([`backend::Cpu`]) and an
//!   optional GPU alias behind the `gpu` feature.
//! - **Weight loading** ([`weights`]): `.pth` and `.safetensors` loading via
//!   `burn-store`, regex key remapping ([`weights::KeyRemap`]), and a strict
//!   *"every source key was consumed"* check ([`weights::Coverage`]) — the single
//!   biggest correctness risk when porting PyTorch checkpoints.
//! - **Preprocessing** ([`preprocess`]): one parameterised `image::RgbImage` →
//!   `[1, 3, H, W]` tensor pipeline reused by every vision model.
//! - **CTC decoding** ([`ctc`]): greedy best-path decode for OCR recognition.
//! - **Geometry** ([`geometry`]): a robust `cv2.minAreaRect` port + convex hull,
//!   shared by the detection and table post-processors.
//! - **A uniform [`model::Model`] trait** and common [`nn`] blocks.
//!
//! All backends default to CPU (`flex`), so the crate and its tests build and
//! run with no GPU toolchain.

pub mod backend;
pub mod ctc;
pub mod error;
pub mod geometry;
pub mod host;
pub mod model;
pub mod nn;
pub mod preprocess;
pub mod weights;

pub use backend::{Cpu, cpu_device};
pub use ctc::{ctc_greedy_decode, ctc_greedy_decode_slice};
pub use geometry::{convex_hull, min_area_rectangle};
pub use host::{float_to_vec_f32, int_to_vec_i64};
pub use error::{Error, Result};
pub use model::Model;
pub use nn::{ConvBnRelu, ConvBnReluConfig, FrozenBatchNorm2d, PtLayerNorm, PtLinear, tensor_from_vec};
pub use preprocess::{Normalize, Preprocess, Size};
pub use weights::{Coverage, KeyRemap, LoadWeights, assert_all_keys_consumed, load_weights};

#[cfg(feature = "gpu")]
pub use backend::{Gpu, gpu_device};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;
    use burn::tensor::{Tensor, TensorData};
    use image::{Rgb, RgbImage};

    #[test]
    fn ctc_collapses_repeats_and_drops_blanks() {
        // 6 timesteps, 4 classes, blank = 0.
        // per-step argmax: [1, 1, 2, 0(blank), 2, 3]
        // collapse repeats: [1, 2, 0, 2, 3]  (the blank breaks the 2-run)
        // drop blank:       [1, 2, 2, 3]
        let logits = [
            0.1, 0.9, 0.0, 0.0, // t0 -> 1
            0.1, 0.8, 0.1, 0.0, // t1 -> 1 (collapsed)
            0.1, 0.2, 0.7, 0.0, // t2 -> 2
            0.9, 0.0, 0.05, 0.05, // t3 -> 0 (blank)
            0.0, 0.1, 0.8, 0.1, // t4 -> 2 (not collapsed: blank broke the run)
            0.0, 0.1, 0.2, 0.7, // t5 -> 3
        ];
        let decoded = ctc_greedy_decode_slice(&logits, 6, 4, 0);
        assert_eq!(decoded, vec![1, 2, 2, 3]);
    }

    #[test]
    fn ctc_all_blank_is_empty() {
        let logits = [0.9, 0.1, 0.9, 0.1];
        assert_eq!(ctc_greedy_decode_slice(&logits, 2, 2, 0), Vec::<usize>::new());
    }

    #[test]
    fn ctc_tensor_wrapper_matches_slice() {
        let data = TensorData::new(
            vec![0.1_f32, 0.9, 0.0, 0.2, 0.7, 0.1, 0.9, 0.0, 0.1],
            [3, 3],
        );
        let device = cpu_device();
        let logits = Tensor::<Cpu, 2>::from_data(data, &device);
        // steps -> [1, 1, 0]; collapse+drop-blank -> [1]
        assert_eq!(ctc_greedy_decode::<Cpu>(logits, 0), vec![1]);
    }

    #[test]
    fn preprocess_produces_nchw_shape() {
        let img = RgbImage::from_pixel(10, 7, Rgb([128, 64, 32]));
        let pre = Preprocess::new(Size::new(32, 24), Normalize::imagenet());
        let device = cpu_device();
        let tensor = pre
            .apply::<Cpu>(&img, &device)
            .expect("preprocess should succeed");
        // [1, 3, H, W] with H = height = 24, W = width = 32.
        assert_eq!(tensor.dims(), [1, 3, 24, 32]);
    }

    #[test]
    fn preprocess_rescale_maps_into_unit_range() {
        // A pure-white pixel under Rescale becomes 1.0 in every channel.
        let img = RgbImage::from_pixel(4, 4, Rgb([255, 255, 255]));
        let pre = Preprocess::new(Size::square(4), Normalize::Rescale);
        let device = cpu_device();
        let tensor = pre.apply::<Cpu>(&img, &device).expect("preprocess");
        let values = tensor.to_data().into_vec::<f32>().expect("f32 data");
        assert!(values.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn preprocess_rejects_zero_size() {
        let img = RgbImage::from_pixel(4, 4, Rgb([0, 0, 0]));
        let pre = Preprocess::new(Size::new(0, 8), Normalize::Rescale);
        let device = cpu_device();
        assert!(matches!(
            pre.apply::<Cpu>(&img, &device),
            Err(Error::Config(_))
        ));
    }

    #[test]
    fn keyremap_renames_prefix() {
        let remap = KeyRemap::new()
            .rename(r"^backbone\.(.*)$", "encoder.$1")
            .expect("valid regex");
        assert_eq!(
            remap.apply_str("backbone.conv.weight").as_deref(),
            Some("encoder.conv.weight"),
        );
    }

    #[test]
    fn keyremap_leaves_unmatched_keys_alone() {
        let remap = KeyRemap::new()
            .rename(r"^backbone\.(.*)$", "encoder.$1")
            .expect("valid regex");
        assert_eq!(remap.apply_str("head.fc.weight"), None);
    }

    #[test]
    fn keyremap_rejects_bad_regex() {
        let result = KeyRemap::new().rename(r"([unclosed", "x");
        assert!(matches!(result, Err(Error::Config(_))));
    }

    #[test]
    fn coverage_strict_flags_unmapped_keys() {
        let unused = vec!["extra.tensor".to_string()];
        let err = assert_all_keys_consumed(&unused, Coverage::Strict);
        assert!(matches!(err, Err(Error::UnmappedKeys { keys }) if keys == unused));
    }

    #[test]
    fn coverage_lenient_tolerates_unmapped_keys() {
        let unused = vec!["extra.tensor".to_string()];
        assert!(assert_all_keys_consumed(&unused, Coverage::Lenient).is_ok());
    }

    #[test]
    fn coverage_ok_when_all_consumed() {
        assert!(assert_all_keys_consumed(&[], Coverage::Strict).is_ok());
    }

    #[test]
    fn pt_linear_applies_pytorch_layout_weight() {
        use crate::nn::PtLinear;
        let device = cpu_device();
        // A `[out=2, in=3]` weight in PyTorch layout. Row 0 sums the inputs,
        // row 1 negates the first. Loading stores it untransposed; forward must
        // still compute `x @ Wᵀ` so a state-dict loads byte-for-byte.
        let mut lin = PtLinear::<Cpu>::init(3, 2, true, &device);
        lin.weight = burn::module::Param::from_tensor(Tensor::<Cpu, 2>::from_data(
            TensorData::new(vec![1.0_f32, 1.0, 1.0, -1.0, 0.0, 0.0], [2, 3]),
            &device,
        ));
        lin.bias = Some(burn::module::Param::from_tensor(Tensor::<Cpu, 1>::from_data(
            TensorData::new(vec![10.0_f32, 0.0], [2]),
            &device,
        )));
        let x = Tensor::<Cpu, 2>::from_data(TensorData::new(vec![2.0_f32, 3.0, 4.0], [1, 3]), &device);
        let y = lin.forward(x).into_data().into_vec::<f32>().expect("f32");
        // row0: 2+3+4 + 10 = 19 ; row1: -2 + 0 = -2
        assert!((y[0] - 19.0).abs() < 1e-5, "got {}", y[0]);
        assert!((y[1] + 2.0).abs() < 1e-5, "got {}", y[1]);
    }

    #[test]
    fn pt_layer_norm_zero_mean_unit_var() {
        use crate::nn::PtLayerNorm;
        let device = cpu_device();
        // Identity affine (weight=1, bias=0): output should be the standardised input.
        let ln = PtLayerNorm::<Cpu>::init(4, 1e-5, &device);
        let x =
            Tensor::<Cpu, 2>::from_data(TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [1, 4]), &device);
        let y = ln.forward(x).into_data().into_vec::<f32>().expect("f32");
        let mean: f32 = y.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean should be ~0, got {mean}");
        // Symmetric input → outer values equal-and-opposite.
        assert!((y[0] + y[3]).abs() < 1e-5);
    }

    #[test]
    fn frozen_batch_norm_is_identity_for_default_buffers() {
        use crate::nn::FrozenBatchNorm2d;
        let device = cpu_device();
        // weight=1, bias=0, mean=0, var=1 → the affine is (x-0)/sqrt(1+eps) ≈ x.
        let bn = FrozenBatchNorm2d::<Cpu>::init(2, 1e-5, &device);
        let x = Tensor::<Cpu, 4>::from_data(
            TensorData::new(vec![1.0_f32, -2.0, 3.0, -4.0], [1, 2, 1, 2]),
            &device,
        );
        let y = bn.forward(x).into_data().into_vec::<f32>().expect("f32");
        for (got, want) in y.iter().zip([1.0_f32, -2.0, 3.0, -4.0]) {
            assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
        }
    }
}
