// Generated from ONNX "/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0/models/TabRec/UnetStructure/unet.onnx" by burn-onnx
//
//! VENDORED, MACHINE-GENERATED CODE — DO NOT HAND-EDIT.
//!
//! This is the UNet line-segmentation network, emitted verbatim by `burn-onnx`'s
//! `ModelGen` from the PDF-Extract-Kit ONNX export (`unet.onnx`). It was
//! previously generated into `$OUT_DIR` at build time; it is now committed to the
//! tree so the crate builds with no `burn-onnx` build-dependency, no `.onnx`
//! files, and no `onnx-import` feature.
//!
//! Because it is generated, it does NOT follow the crate's panic-free convention:
//! it contains machine-emitted `.unwrap()`/`.expect()` calls (in the graph
//! `forward` and in `from_file`/`from_bytes`/`Default`). Those are the sanctioned
//! vendored-code exception. Hand-written code (the runtime loader in
//! `crate::weights`, `crate::unet::model`) instead loads these weights panic-free
//! by calling `Model::new(device)` + `load_from(&mut store)` and mapping the
//! error. If the model needs regenerating, re-run `burn-onnx` on the ONNX export
//! and replace this whole file — do not edit it by hand.
#![allow(clippy::all)]
#![allow(clippy::pedantic)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(clippy::approx_constant)]
use burn::prelude::*;
use burn::nn::BatchNorm;
use burn::nn::BatchNormConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::conv::ConvTranspose2d;
use burn::nn::conv::ConvTranspose2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;


#[derive(Module, Debug)]
pub struct Submodule1<B: Backend> {
    conv2d1: Conv2d<B>,
    maxpool2d1: MaxPool2d,
    batchnormalization1: BatchNorm<B>,
    conv2d2: Conv2d<B>,
    maxpool2d2: MaxPool2d,
    batchnormalization2: BatchNorm<B>,
    conv2d3: Conv2d<B>,
    conv2d4: Conv2d<B>,
    conv2d5: Conv2d<B>,
    conv2d6: Conv2d<B>,
    conv2d7: Conv2d<B>,
    conv2d8: Conv2d<B>,
    conv2d9: Conv2d<B>,
    conv2d10: Conv2d<B>,
    conv2d11: Conv2d<B>,
    conv2d12: Conv2d<B>,
    conv2d13: Conv2d<B>,
    conv2d14: Conv2d<B>,
    conv2d15: Conv2d<B>,
    conv2d16: Conv2d<B>,
    conv2d17: Conv2d<B>,
    conv2d18: Conv2d<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule1<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d1 = Conv2dConfig::new([3, 13], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d1 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        let batchnormalization1 = BatchNormConfig::new(16)
            .with_epsilon(0.0010000000474974513f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d2 = Conv2dConfig::new([16, 48], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d2 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        let batchnormalization2 = BatchNormConfig::new(64)
            .with_epsilon(0.0010000000474974513f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d3 = Conv2dConfig::new([64, 64], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d4 = Conv2dConfig::new([64, 64], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([64, 64], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([64, 64], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([64, 64], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([64, 64], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([64, 64], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([64, 64], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([64, 64], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d12 = Conv2dConfig::new([64, 64], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([64, 64], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([64, 64], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d15 = Conv2dConfig::new([64, 64], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d16 = Conv2dConfig::new([64, 64], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([64, 64], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([64, 64], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv2d1,
            maxpool2d1,
            batchnormalization1,
            conv2d2,
            maxpool2d2,
            batchnormalization2,
            conv2d3,
            conv2d4,
            conv2d5,
            conv2d6,
            conv2d7,
            conv2d8,
            conv2d9,
            conv2d10,
            conv2d11,
            conv2d12,
            conv2d13,
            conv2d14,
            conv2d15,
            conv2d16,
            conv2d17,
            conv2d18,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let conv2d1_out1 = self.conv2d1.forward(input.clone());
        let maxpool2d1_out1 = self.maxpool2d1.forward(input);
        let shape1_out1: [i64; 4] = {
            let axes = &conv2d1_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather1_out1 = shape1_out1[2] as i64;
        let gather2_out1 = shape1_out1[3] as i64;
        let unsqueeze1_out1 = [gather1_out1 as i64];
        let unsqueeze2_out1 = [gather2_out1 as i64];
        let concat1_out1: [i64; 2usize] = [&unsqueeze1_out1[..], &unsqueeze2_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let shape3_out1: [i64; 4] = {
            let axes = &maxpool2d1_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice1_out1: [i64; 2] = shape3_out1[0..2].try_into().unwrap();
        let concat2_out1: [i64; 4usize] = [&slice1_out1[..], &concat1_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let resize1_out1 = {
            let target_height = concat2_out1[2] as usize;
            let target_width = concat2_out1[3] as usize;
            burn::tensor::module::interpolate(
                maxpool2d1_out1,
                [target_height, target_width],
                burn::tensor::ops::InterpolateOptions::new(
                        burn::tensor::ops::InterpolateMode::Bilinear,
                    )
                    .with_align_corners(false),
            )
        };
        let concat3_out1 = burn::tensor::Tensor::cat(
            [conv2d1_out1, resize1_out1].into(),
            1,
        );
        let batchnormalization1_out1 = self.batchnormalization1.forward(concat3_out1);
        let relu1_out1 = burn::tensor::activation::relu(batchnormalization1_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1.clone());
        let maxpool2d2_out1 = self.maxpool2d2.forward(relu1_out1);
        let shape4_out1: [i64; 4] = {
            let axes = &conv2d2_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather3_out1 = shape4_out1[2] as i64;
        let gather4_out1 = shape4_out1[3] as i64;
        let unsqueeze3_out1 = [gather3_out1 as i64];
        let unsqueeze4_out1 = [gather4_out1 as i64];
        let concat4_out1: [i64; 2usize] = [&unsqueeze3_out1[..], &unsqueeze4_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let shape6_out1: [i64; 4] = {
            let axes = &maxpool2d2_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice2_out1: [i64; 2] = shape6_out1[0..2].try_into().unwrap();
        let concat5_out1: [i64; 4usize] = [&slice2_out1[..], &concat4_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let resize2_out1 = {
            let target_height = concat5_out1[2] as usize;
            let target_width = concat5_out1[3] as usize;
            burn::tensor::module::interpolate(
                maxpool2d2_out1,
                [target_height, target_width],
                burn::tensor::ops::InterpolateOptions::new(
                        burn::tensor::ops::InterpolateMode::Bilinear,
                    )
                    .with_align_corners(false),
            )
        };
        let concat6_out1 = burn::tensor::Tensor::cat(
            [conv2d2_out1, resize2_out1].into(),
            1,
        );
        let batchnormalization2_out1 = self.batchnormalization2.forward(concat6_out1);
        let relu2_out1 = burn::tensor::activation::relu(batchnormalization2_out1);
        let conv2d3_out1 = self.conv2d3.forward(relu2_out1.clone());
        let relu3_out1 = burn::tensor::activation::relu(conv2d3_out1);
        let conv2d4_out1 = self.conv2d4.forward(relu3_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let conv2d5_out1 = self.conv2d5.forward(relu4_out1);
        let relu5_out1 = burn::tensor::activation::relu(conv2d5_out1);
        let conv2d6_out1 = self.conv2d6.forward(relu5_out1);
        let add1_out1 = conv2d6_out1.add(relu2_out1);
        let relu6_out1 = burn::tensor::activation::relu(add1_out1);
        let conv2d7_out1 = self.conv2d7.forward(relu6_out1.clone());
        let relu7_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let conv2d8_out1 = self.conv2d8.forward(relu7_out1);
        let relu8_out1 = burn::tensor::activation::relu(conv2d8_out1);
        let conv2d9_out1 = self.conv2d9.forward(relu8_out1);
        let relu9_out1 = burn::tensor::activation::relu(conv2d9_out1);
        let conv2d10_out1 = self.conv2d10.forward(relu9_out1);
        let add2_out1 = conv2d10_out1.add(relu6_out1);
        let relu10_out1 = burn::tensor::activation::relu(add2_out1);
        let conv2d11_out1 = self.conv2d11.forward(relu10_out1.clone());
        let relu11_out1 = burn::tensor::activation::relu(conv2d11_out1);
        let conv2d12_out1 = self.conv2d12.forward(relu11_out1);
        let relu12_out1 = burn::tensor::activation::relu(conv2d12_out1);
        let conv2d13_out1 = self.conv2d13.forward(relu12_out1);
        let relu13_out1 = burn::tensor::activation::relu(conv2d13_out1);
        let conv2d14_out1 = self.conv2d14.forward(relu13_out1);
        let add3_out1 = conv2d14_out1.add(relu10_out1);
        let relu14_out1 = burn::tensor::activation::relu(add3_out1);
        let conv2d15_out1 = self.conv2d15.forward(relu14_out1.clone());
        let relu15_out1 = burn::tensor::activation::relu(conv2d15_out1);
        let conv2d16_out1 = self.conv2d16.forward(relu15_out1);
        let relu16_out1 = burn::tensor::activation::relu(conv2d16_out1);
        let conv2d17_out1 = self.conv2d17.forward(relu16_out1);
        let relu17_out1 = burn::tensor::activation::relu(conv2d17_out1);
        let conv2d18_out1 = self.conv2d18.forward(relu17_out1);
        let add4_out1 = conv2d18_out1.add(relu14_out1);
        add4_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule2<B: Backend> {
    conv2d19: Conv2d<B>,
    conv2d20: Conv2d<B>,
    conv2d21: Conv2d<B>,
    conv2d22: Conv2d<B>,
    conv2d23: Conv2d<B>,
    maxpool2d3: MaxPool2d,
    batchnormalization3: BatchNorm<B>,
    conv2d24: Conv2d<B>,
    conv2d25: Conv2d<B>,
    conv2d26: Conv2d<B>,
    conv2d27: Conv2d<B>,
    conv2d28: Conv2d<B>,
    conv2d29: Conv2d<B>,
    conv2d30: Conv2d<B>,
    conv2d31: Conv2d<B>,
    conv2d32: Conv2d<B>,
    conv2d33: Conv2d<B>,
    conv2d34: Conv2d<B>,
    conv2d35: Conv2d<B>,
    conv2d36: Conv2d<B>,
    conv2d37: Conv2d<B>,
    conv2d38: Conv2d<B>,
    conv2d39: Conv2d<B>,
    conv2d40: Conv2d<B>,
    conv2d41: Conv2d<B>,
    conv2d42: Conv2d<B>,
    conv2d43: Conv2d<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule2<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d19 = Conv2dConfig::new([64, 64], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d20 = Conv2dConfig::new([64, 64], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d21 = Conv2dConfig::new([64, 64], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d22 = Conv2dConfig::new([64, 64], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d23 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d3 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        let batchnormalization3 = BatchNormConfig::new(128)
            .with_epsilon(0.0010000000474974513f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d24 = Conv2dConfig::new([128, 128], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d25 = Conv2dConfig::new([128, 128], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d26 = Conv2dConfig::new([128, 128], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 0, 2, 0))
            .with_dilation([2, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d27 = Conv2dConfig::new([128, 128], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 2, 0, 2))
            .with_dilation([1, 2])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d28 = Conv2dConfig::new([128, 128], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d29 = Conv2dConfig::new([128, 128], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d30 = Conv2dConfig::new([128, 128], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 0, 4, 0))
            .with_dilation([4, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d31 = Conv2dConfig::new([128, 128], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 4, 0, 4))
            .with_dilation([1, 4])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d32 = Conv2dConfig::new([128, 128], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d33 = Conv2dConfig::new([128, 128], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d34 = Conv2dConfig::new([128, 128], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(8, 0, 8, 0))
            .with_dilation([8, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d35 = Conv2dConfig::new([128, 128], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 8, 0, 8))
            .with_dilation([1, 8])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d36 = Conv2dConfig::new([128, 128], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d37 = Conv2dConfig::new([128, 128], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d38 = Conv2dConfig::new([128, 128], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(16, 0, 16, 0))
            .with_dilation([16, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d39 = Conv2dConfig::new([128, 128], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 16, 0, 16))
            .with_dilation([1, 16])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d40 = Conv2dConfig::new([128, 128], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d41 = Conv2dConfig::new([128, 128], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d42 = Conv2dConfig::new([128, 128], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 0, 2, 0))
            .with_dilation([2, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d43 = Conv2dConfig::new([128, 128], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 2, 0, 2))
            .with_dilation([1, 2])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv2d19,
            conv2d20,
            conv2d21,
            conv2d22,
            conv2d23,
            maxpool2d3,
            batchnormalization3,
            conv2d24,
            conv2d25,
            conv2d26,
            conv2d27,
            conv2d28,
            conv2d29,
            conv2d30,
            conv2d31,
            conv2d32,
            conv2d33,
            conv2d34,
            conv2d35,
            conv2d36,
            conv2d37,
            conv2d38,
            conv2d39,
            conv2d40,
            conv2d41,
            conv2d42,
            conv2d43,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, add4_out1: Tensor<B, 4>) -> Tensor<B, 4> {
        let relu18_out1 = burn::tensor::activation::relu(add4_out1);
        let conv2d19_out1 = self.conv2d19.forward(relu18_out1.clone());
        let relu19_out1 = burn::tensor::activation::relu(conv2d19_out1);
        let conv2d20_out1 = self.conv2d20.forward(relu19_out1);
        let relu20_out1 = burn::tensor::activation::relu(conv2d20_out1);
        let conv2d21_out1 = self.conv2d21.forward(relu20_out1);
        let relu21_out1 = burn::tensor::activation::relu(conv2d21_out1);
        let conv2d22_out1 = self.conv2d22.forward(relu21_out1);
        let add5_out1 = conv2d22_out1.add(relu18_out1);
        let relu22_out1 = burn::tensor::activation::relu(add5_out1);
        let conv2d23_out1 = self.conv2d23.forward(relu22_out1.clone());
        let maxpool2d3_out1 = self.maxpool2d3.forward(relu22_out1);
        let shape7_out1: [i64; 4] = {
            let axes = &conv2d23_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather5_out1 = shape7_out1[2] as i64;
        let gather6_out1 = shape7_out1[3] as i64;
        let unsqueeze5_out1 = [gather5_out1 as i64];
        let unsqueeze6_out1 = [gather6_out1 as i64];
        let concat7_out1: [i64; 2usize] = [&unsqueeze5_out1[..], &unsqueeze6_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let shape9_out1: [i64; 4] = {
            let axes = &maxpool2d3_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice3_out1: [i64; 2] = shape9_out1[0..2].try_into().unwrap();
        let concat8_out1: [i64; 4usize] = [&slice3_out1[..], &concat7_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let resize3_out1 = {
            let target_height = concat8_out1[2] as usize;
            let target_width = concat8_out1[3] as usize;
            burn::tensor::module::interpolate(
                maxpool2d3_out1,
                [target_height, target_width],
                burn::tensor::ops::InterpolateOptions::new(
                        burn::tensor::ops::InterpolateMode::Bilinear,
                    )
                    .with_align_corners(false),
            )
        };
        let concat9_out1 = burn::tensor::Tensor::cat(
            [conv2d23_out1, resize3_out1].into(),
            1,
        );
        let batchnormalization3_out1 = self.batchnormalization3.forward(concat9_out1);
        let relu23_out1 = burn::tensor::activation::relu(batchnormalization3_out1);
        let conv2d24_out1 = self.conv2d24.forward(relu23_out1.clone());
        let relu24_out1 = burn::tensor::activation::relu(conv2d24_out1);
        let conv2d25_out1 = self.conv2d25.forward(relu24_out1);
        let relu25_out1 = burn::tensor::activation::relu(conv2d25_out1);
        let conv2d26_out1 = self.conv2d26.forward(relu25_out1);
        let relu26_out1 = burn::tensor::activation::relu(conv2d26_out1);
        let conv2d27_out1 = self.conv2d27.forward(relu26_out1);
        let add6_out1 = conv2d27_out1.add(relu23_out1);
        let relu27_out1 = burn::tensor::activation::relu(add6_out1);
        let conv2d28_out1 = self.conv2d28.forward(relu27_out1.clone());
        let relu28_out1 = burn::tensor::activation::relu(conv2d28_out1);
        let conv2d29_out1 = self.conv2d29.forward(relu28_out1);
        let relu29_out1 = burn::tensor::activation::relu(conv2d29_out1);
        let conv2d30_out1 = self.conv2d30.forward(relu29_out1);
        let relu30_out1 = burn::tensor::activation::relu(conv2d30_out1);
        let conv2d31_out1 = self.conv2d31.forward(relu30_out1);
        let add7_out1 = conv2d31_out1.add(relu27_out1);
        let relu31_out1 = burn::tensor::activation::relu(add7_out1);
        let conv2d32_out1 = self.conv2d32.forward(relu31_out1.clone());
        let relu32_out1 = burn::tensor::activation::relu(conv2d32_out1);
        let conv2d33_out1 = self.conv2d33.forward(relu32_out1);
        let relu33_out1 = burn::tensor::activation::relu(conv2d33_out1);
        let conv2d34_out1 = self.conv2d34.forward(relu33_out1);
        let relu34_out1 = burn::tensor::activation::relu(conv2d34_out1);
        let conv2d35_out1 = self.conv2d35.forward(relu34_out1);
        let add8_out1 = conv2d35_out1.add(relu31_out1);
        let relu35_out1 = burn::tensor::activation::relu(add8_out1);
        let conv2d36_out1 = self.conv2d36.forward(relu35_out1.clone());
        let relu36_out1 = burn::tensor::activation::relu(conv2d36_out1);
        let conv2d37_out1 = self.conv2d37.forward(relu36_out1);
        let relu37_out1 = burn::tensor::activation::relu(conv2d37_out1);
        let conv2d38_out1 = self.conv2d38.forward(relu37_out1);
        let relu38_out1 = burn::tensor::activation::relu(conv2d38_out1);
        let conv2d39_out1 = self.conv2d39.forward(relu38_out1);
        let add9_out1 = conv2d39_out1.add(relu35_out1);
        let relu39_out1 = burn::tensor::activation::relu(add9_out1);
        let conv2d40_out1 = self.conv2d40.forward(relu39_out1.clone());
        let relu40_out1 = burn::tensor::activation::relu(conv2d40_out1);
        let conv2d41_out1 = self.conv2d41.forward(relu40_out1);
        let relu41_out1 = burn::tensor::activation::relu(conv2d41_out1);
        let conv2d42_out1 = self.conv2d42.forward(relu41_out1);
        let relu42_out1 = burn::tensor::activation::relu(conv2d42_out1);
        let conv2d43_out1 = self.conv2d43.forward(relu42_out1);
        let add10_out1 = conv2d43_out1.add(relu39_out1);
        add10_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule3<B: Backend> {
    conv2d44: Conv2d<B>,
    conv2d45: Conv2d<B>,
    conv2d46: Conv2d<B>,
    conv2d47: Conv2d<B>,
    conv2d48: Conv2d<B>,
    conv2d49: Conv2d<B>,
    conv2d50: Conv2d<B>,
    conv2d51: Conv2d<B>,
    conv2d52: Conv2d<B>,
    conv2d53: Conv2d<B>,
    conv2d54: Conv2d<B>,
    conv2d55: Conv2d<B>,
    convtranspose2d1: ConvTranspose2d<B>,
    batchnormalization4: BatchNorm<B>,
    conv2d56: Conv2d<B>,
    conv2d57: Conv2d<B>,
    conv2d58: Conv2d<B>,
    conv2d59: Conv2d<B>,
    conv2d60: Conv2d<B>,
    conv2d61: Conv2d<B>,
    conv2d62: Conv2d<B>,
    conv2d63: Conv2d<B>,
    convtranspose2d2: ConvTranspose2d<B>,
    batchnormalization5: BatchNorm<B>,
    conv2d64: Conv2d<B>,
    conv2d65: Conv2d<B>,
    conv2d66: Conv2d<B>,
    conv2d67: Conv2d<B>,
    conv2d68: Conv2d<B>,
    conv2d69: Conv2d<B>,
    conv2d70: Conv2d<B>,
    conv2d71: Conv2d<B>,
    conv2d72: Conv2d<B>,
    conv2d73: Conv2d<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule3<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d44 = Conv2dConfig::new([128, 128], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d45 = Conv2dConfig::new([128, 128], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d46 = Conv2dConfig::new([128, 128], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 0, 4, 0))
            .with_dilation([4, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d47 = Conv2dConfig::new([128, 128], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 4, 0, 4))
            .with_dilation([1, 4])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d48 = Conv2dConfig::new([128, 128], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d49 = Conv2dConfig::new([128, 128], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d50 = Conv2dConfig::new([128, 128], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(8, 0, 8, 0))
            .with_dilation([8, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d51 = Conv2dConfig::new([128, 128], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 8, 0, 8))
            .with_dilation([1, 8])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d52 = Conv2dConfig::new([128, 128], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d53 = Conv2dConfig::new([128, 128], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d54 = Conv2dConfig::new([128, 128], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(16, 0, 16, 0))
            .with_dilation([16, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d55 = Conv2dConfig::new([128, 128], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 16, 0, 16))
            .with_dilation([1, 16])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose2d1 = ConvTranspose2dConfig::new([128, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_padding_out([1, 1])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization4 = BatchNormConfig::new(64)
            .with_epsilon(0.0010000000474974513f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d56 = Conv2dConfig::new([64, 64], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d57 = Conv2dConfig::new([64, 64], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d58 = Conv2dConfig::new([64, 64], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d59 = Conv2dConfig::new([64, 64], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d60 = Conv2dConfig::new([64, 64], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d61 = Conv2dConfig::new([64, 64], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d62 = Conv2dConfig::new([64, 64], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d63 = Conv2dConfig::new([64, 64], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose2d2 = ConvTranspose2dConfig::new([64, 16], [3, 3])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_padding_out([1, 1])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization5 = BatchNormConfig::new(16)
            .with_epsilon(0.0010000000474974513f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d64 = Conv2dConfig::new([16, 16], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d65 = Conv2dConfig::new([16, 16], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d66 = Conv2dConfig::new([16, 16], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d67 = Conv2dConfig::new([16, 16], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d68 = Conv2dConfig::new([16, 16], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d69 = Conv2dConfig::new([16, 16], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d70 = Conv2dConfig::new([16, 16], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d71 = Conv2dConfig::new([16, 16], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d72 = Conv2dConfig::new([16, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d73 = Conv2dConfig::new([128, 3], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv2d44,
            conv2d45,
            conv2d46,
            conv2d47,
            conv2d48,
            conv2d49,
            conv2d50,
            conv2d51,
            conv2d52,
            conv2d53,
            conv2d54,
            conv2d55,
            convtranspose2d1,
            batchnormalization4,
            conv2d56,
            conv2d57,
            conv2d58,
            conv2d59,
            conv2d60,
            conv2d61,
            conv2d62,
            conv2d63,
            convtranspose2d2,
            batchnormalization5,
            conv2d64,
            conv2d65,
            conv2d66,
            conv2d67,
            conv2d68,
            conv2d69,
            conv2d70,
            conv2d71,
            conv2d72,
            conv2d73,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add10_out1: Tensor<B, 4>,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4, Int> {
        let relu43_out1 = burn::tensor::activation::relu(add10_out1);
        let conv2d44_out1 = self.conv2d44.forward(relu43_out1.clone());
        let relu44_out1 = burn::tensor::activation::relu(conv2d44_out1);
        let conv2d45_out1 = self.conv2d45.forward(relu44_out1);
        let relu45_out1 = burn::tensor::activation::relu(conv2d45_out1);
        let conv2d46_out1 = self.conv2d46.forward(relu45_out1);
        let relu46_out1 = burn::tensor::activation::relu(conv2d46_out1);
        let conv2d47_out1 = self.conv2d47.forward(relu46_out1);
        let add11_out1 = conv2d47_out1.add(relu43_out1);
        let relu47_out1 = burn::tensor::activation::relu(add11_out1);
        let conv2d48_out1 = self.conv2d48.forward(relu47_out1.clone());
        let relu48_out1 = burn::tensor::activation::relu(conv2d48_out1);
        let conv2d49_out1 = self.conv2d49.forward(relu48_out1);
        let relu49_out1 = burn::tensor::activation::relu(conv2d49_out1);
        let conv2d50_out1 = self.conv2d50.forward(relu49_out1);
        let relu50_out1 = burn::tensor::activation::relu(conv2d50_out1);
        let conv2d51_out1 = self.conv2d51.forward(relu50_out1);
        let add12_out1 = conv2d51_out1.add(relu47_out1);
        let relu51_out1 = burn::tensor::activation::relu(add12_out1);
        let conv2d52_out1 = self.conv2d52.forward(relu51_out1.clone());
        let relu52_out1 = burn::tensor::activation::relu(conv2d52_out1);
        let conv2d53_out1 = self.conv2d53.forward(relu52_out1);
        let relu53_out1 = burn::tensor::activation::relu(conv2d53_out1);
        let conv2d54_out1 = self.conv2d54.forward(relu53_out1);
        let relu54_out1 = burn::tensor::activation::relu(conv2d54_out1);
        let conv2d55_out1 = self.conv2d55.forward(relu54_out1);
        let add13_out1 = conv2d55_out1.add(relu51_out1);
        let relu55_out1 = burn::tensor::activation::relu(add13_out1);
        let convtranspose2d1_out1 = self.convtranspose2d1.forward(relu55_out1);
        let batchnormalization4_out1 = self
            .batchnormalization4
            .forward(convtranspose2d1_out1);
        let relu56_out1 = burn::tensor::activation::relu(batchnormalization4_out1);
        let conv2d56_out1 = self.conv2d56.forward(relu56_out1.clone());
        let relu57_out1 = burn::tensor::activation::relu(conv2d56_out1);
        let conv2d57_out1 = self.conv2d57.forward(relu57_out1);
        let relu58_out1 = burn::tensor::activation::relu(conv2d57_out1);
        let conv2d58_out1 = self.conv2d58.forward(relu58_out1);
        let relu59_out1 = burn::tensor::activation::relu(conv2d58_out1);
        let conv2d59_out1 = self.conv2d59.forward(relu59_out1);
        let add14_out1 = conv2d59_out1.add(relu56_out1);
        let relu60_out1 = burn::tensor::activation::relu(add14_out1);
        let conv2d60_out1 = self.conv2d60.forward(relu60_out1.clone());
        let relu61_out1 = burn::tensor::activation::relu(conv2d60_out1);
        let conv2d61_out1 = self.conv2d61.forward(relu61_out1);
        let relu62_out1 = burn::tensor::activation::relu(conv2d61_out1);
        let conv2d62_out1 = self.conv2d62.forward(relu62_out1);
        let relu63_out1 = burn::tensor::activation::relu(conv2d62_out1);
        let conv2d63_out1 = self.conv2d63.forward(relu63_out1);
        let add15_out1 = conv2d63_out1.add(relu60_out1);
        let relu64_out1 = burn::tensor::activation::relu(add15_out1);
        let convtranspose2d2_out1 = self.convtranspose2d2.forward(relu64_out1);
        let batchnormalization5_out1 = self
            .batchnormalization5
            .forward(convtranspose2d2_out1);
        let relu65_out1 = burn::tensor::activation::relu(batchnormalization5_out1);
        let conv2d64_out1 = self.conv2d64.forward(relu65_out1.clone());
        let relu66_out1 = burn::tensor::activation::relu(conv2d64_out1);
        let conv2d65_out1 = self.conv2d65.forward(relu66_out1);
        let relu67_out1 = burn::tensor::activation::relu(conv2d65_out1);
        let conv2d66_out1 = self.conv2d66.forward(relu67_out1);
        let relu68_out1 = burn::tensor::activation::relu(conv2d66_out1);
        let conv2d67_out1 = self.conv2d67.forward(relu68_out1);
        let add16_out1 = conv2d67_out1.add(relu65_out1);
        let relu69_out1 = burn::tensor::activation::relu(add16_out1);
        let conv2d68_out1 = self.conv2d68.forward(relu69_out1.clone());
        let relu70_out1 = burn::tensor::activation::relu(conv2d68_out1);
        let conv2d69_out1 = self.conv2d69.forward(relu70_out1);
        let relu71_out1 = burn::tensor::activation::relu(conv2d69_out1);
        let conv2d70_out1 = self.conv2d70.forward(relu71_out1);
        let relu72_out1 = burn::tensor::activation::relu(conv2d70_out1);
        let conv2d71_out1 = self.conv2d71.forward(relu72_out1);
        let add17_out1 = conv2d71_out1.add(relu69_out1);
        let relu73_out1 = burn::tensor::activation::relu(add17_out1);
        let conv2d72_out1 = self.conv2d72.forward(relu73_out1);
        let relu74_out1 = burn::tensor::activation::relu(conv2d72_out1);
        let conv2d73_out1 = self.conv2d73.forward(relu74_out1);
        let shape10_out1: [i64; 4] = {
            let axes = &input.dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather7_out1 = shape10_out1[2] as i64;
        let gather8_out1 = shape10_out1[3] as i64;
        let unsqueeze7_out1 = [gather7_out1 as i64];
        let unsqueeze8_out1 = [gather8_out1 as i64];
        let concat10_out1: [i64; 2usize] = [&unsqueeze7_out1[..], &unsqueeze8_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let shape12_out1: [i64; 4] = {
            let axes = &conv2d73_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice4_out1: [i64; 2] = shape12_out1[0..2].try_into().unwrap();
        let concat11_out1: [i64; 4usize] = [&slice4_out1[..], &concat10_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let resize4_out1 = {
            let target_height = concat11_out1[2] as usize;
            let target_width = concat11_out1[3] as usize;
            burn::tensor::module::interpolate(
                conv2d73_out1,
                [target_height, target_width],
                burn::tensor::ops::InterpolateOptions::new(
                        burn::tensor::ops::InterpolateMode::Bilinear,
                    )
                    .with_align_corners(false),
            )
        };
        let gather9_out1 = shape10_out1[2] as i64;
        let gather10_out1 = shape10_out1[3] as i64;
        let unsqueeze9_out1 = [gather9_out1 as i64];
        let unsqueeze10_out1 = [gather10_out1 as i64];
        let concat12_out1: [i64; 2usize] = [&unsqueeze9_out1[..], &unsqueeze10_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let shape15_out1: [i64; 4] = {
            let axes = &resize4_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice5_out1: [i64; 2] = shape15_out1[0..2].try_into().unwrap();
        let concat13_out1: [i64; 4usize] = [&slice5_out1[..], &concat12_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let resize5_out1 = {
            let target_height = concat13_out1[2] as usize;
            let target_width = concat13_out1[3] as usize;
            burn::tensor::module::interpolate(
                resize4_out1,
                [target_height, target_width],
                burn::tensor::ops::InterpolateOptions::new(
                        burn::tensor::ops::InterpolateMode::Bilinear,
                    )
                    .with_align_corners(false),
            )
        };
        let reducemax1_out1 = { resize5_out1.clone().max_dim(1usize) };
        let sub1_out1 = resize5_out1.sub(reducemax1_out1);
        let exp1_out1 = sub1_out1.exp();
        let reducesum1_out1 = { exp1_out1.clone().sum_dim(1usize) };
        let div1_out1 = exp1_out1.div(reducesum1_out1);
        let argmax_result = div1_out1.argmax(1);
        let argmax1_out1 = argmax_result
            .squeeze_dim::<3usize>(1)
            .cast(burn::tensor::DType::I64);
        let unsqueeze11_out1: Tensor<B, 4, Int> = argmax1_out1.unsqueeze_dims::<4>(&[0]);
        unsqueeze11_out1
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    submodule1: Submodule1<B>,
    submodule2: Submodule2<B>,
    submodule3: Submodule3<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}


extern crate std;

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file(
            "/Users/pohsuanlai/Documents/mineru/mineru-rs/target/release/build/mineru-table-3af5243fd7cca7a4/out/model/unet.bpk",
            &Default::default(),
        )
    }
}

impl<B: Backend> Model<B> {
    /// Load model weights from a burnpack file.
    pub fn from_file<P: AsRef<std::path::Path>>(file: P, device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(file);
        model.load_from(&mut store).expect("Failed to load burnpack file");
        model
    }

    /// Load model weights from in-memory bytes.
    ///
    /// The bytes must be the contents of a `.bpk` file.
    pub fn from_bytes(bytes: Bytes, device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        model.load_from(&mut store).expect("Failed to load burnpack bytes");
        model
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let submodule1 = Submodule1::new(device);
        let submodule2 = Submodule2::new(device);
        let submodule3 = Submodule3::new(device);
        Self {
            submodule1,
            submodule2,
            submodule3,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4, Int> {
        let add4_out1 = self.submodule1.forward(input.clone());
        let add10_out1 = self.submodule2.forward(add4_out1);
        let unsqueeze11_out1 = self.submodule3.forward(add10_out1, input);
        unsqueeze11_out1
    }
}
