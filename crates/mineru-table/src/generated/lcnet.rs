// Generated from ONNX "/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0/models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx" by burn-onnx
//
//! VENDORED, MACHINE-GENERATED CODE — DO NOT HAND-EDIT.
//!
//! This is the PP-LCNet_x1_0 table classifier, emitted verbatim by `burn-onnx`'s
//! `ModelGen` from the PDF-Extract-Kit ONNX export
//! (`PP-LCNet_x1_0_table_cls.onnx`). It was previously generated into `$OUT_DIR`
//! at build time; it is now committed to the tree so the crate builds with no
//! `burn-onnx` build-dependency, no `.onnx` files, and no `onnx-import` feature.
//!
//! Because it is generated, it does NOT follow the crate's panic-free convention:
//! it contains machine-emitted `.unwrap()`/`.expect()` calls (in the graph
//! `forward` and in `from_file`/`from_bytes`/`Default`). Those are the sanctioned
//! vendored-code exception. Hand-written code (the runtime loader in
//! `crate::weights`, `crate::cls`) instead loads these weights panic-free by
//! calling `Model::new(device)` + `load_from(&mut store)` and mapping the error.
//! If the model needs regenerating, re-run `burn-onnx` on the ONNX export and
//! replace this whole file — do not edit it by hand.
#![allow(clippy::all)]
#![allow(clippy::pedantic)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(clippy::approx_constant)]
use burn::prelude::*;
use burn::nn::BatchNorm;
use burn::nn::BatchNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::AdaptiveAvgPool2d;
use burn::nn::pool::AdaptiveAvgPool2dConfig;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    constant122: burn::module::Param<Tensor<B, 4>>,
    constant124: burn::module::Param<Tensor<B, 4>>,
    constant136: burn::module::Param<Tensor<B, 4>>,
    constant138: burn::module::Param<Tensor<B, 4>>,
    constant145: burn::module::Param<Tensor<B, 1>>,
    conv2d1: Conv2d<B>,
    batchnormalization1: BatchNorm<B>,
    conv2d2: Conv2d<B>,
    batchnormalization2: BatchNorm<B>,
    conv2d3: Conv2d<B>,
    batchnormalization3: BatchNorm<B>,
    conv2d4: Conv2d<B>,
    batchnormalization4: BatchNorm<B>,
    conv2d5: Conv2d<B>,
    batchnormalization5: BatchNorm<B>,
    conv2d6: Conv2d<B>,
    batchnormalization6: BatchNorm<B>,
    conv2d7: Conv2d<B>,
    batchnormalization7: BatchNorm<B>,
    conv2d8: Conv2d<B>,
    batchnormalization8: BatchNorm<B>,
    conv2d9: Conv2d<B>,
    batchnormalization9: BatchNorm<B>,
    conv2d10: Conv2d<B>,
    batchnormalization10: BatchNorm<B>,
    conv2d11: Conv2d<B>,
    batchnormalization11: BatchNorm<B>,
    conv2d12: Conv2d<B>,
    batchnormalization12: BatchNorm<B>,
    conv2d13: Conv2d<B>,
    batchnormalization13: BatchNorm<B>,
    conv2d14: Conv2d<B>,
    batchnormalization14: BatchNorm<B>,
    conv2d15: Conv2d<B>,
    batchnormalization15: BatchNorm<B>,
    conv2d16: Conv2d<B>,
    batchnormalization16: BatchNorm<B>,
    conv2d17: Conv2d<B>,
    batchnormalization17: BatchNorm<B>,
    conv2d18: Conv2d<B>,
    batchnormalization18: BatchNorm<B>,
    conv2d19: Conv2d<B>,
    batchnormalization19: BatchNorm<B>,
    conv2d20: Conv2d<B>,
    batchnormalization20: BatchNorm<B>,
    conv2d21: Conv2d<B>,
    batchnormalization21: BatchNorm<B>,
    conv2d22: Conv2d<B>,
    batchnormalization22: BatchNorm<B>,
    conv2d23: Conv2d<B>,
    batchnormalization23: BatchNorm<B>,
    conv2d24: Conv2d<B>,
    batchnormalization24: BatchNorm<B>,
    globalaveragepool1: AdaptiveAvgPool2d,
    conv2d25: Conv2d<B>,
    conv2d26: Conv2d<B>,
    conv2d27: Conv2d<B>,
    batchnormalization25: BatchNorm<B>,
    conv2d28: Conv2d<B>,
    batchnormalization26: BatchNorm<B>,
    globalaveragepool2: AdaptiveAvgPool2d,
    conv2d29: Conv2d<B>,
    conv2d30: Conv2d<B>,
    conv2d31: Conv2d<B>,
    batchnormalization27: BatchNorm<B>,
    globalaveragepool3: AdaptiveAvgPool2d,
    conv2d32: Conv2d<B>,
    linear1: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}


extern crate std;

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file(
            "/Users/pohsuanlai/Documents/mineru/mineru-rs/target/release/build/mineru-table-3af5243fd7cca7a4/out/model/PP-LCNet_x1_0_table_cls.bpk",
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
        let constant122: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                4,
            >::zeros([1, 64, 1, 1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [1, 64, 1, 1].into(),
        );
        let constant124: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                4,
            >::zeros([1, 256, 1, 1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [1, 256, 1, 1].into(),
        );
        let constant136: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                4,
            >::zeros([1, 128, 1, 1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [1, 128, 1, 1].into(),
        );
        let constant138: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                4,
            >::zeros([1, 512, 1, 1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [1, 512, 1, 1].into(),
        );
        let constant145: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                1,
            >::zeros([1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d1 = Conv2dConfig::new([3, 16], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization1 = BatchNormConfig::new(16)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d2 = Conv2dConfig::new([16, 16], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(16)
            .with_bias(false)
            .init(device);
        let batchnormalization2 = BatchNormConfig::new(16)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d3 = Conv2dConfig::new([16, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization3 = BatchNormConfig::new(32)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d4 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(32)
            .with_bias(false)
            .init(device);
        let batchnormalization4 = BatchNormConfig::new(32)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d5 = Conv2dConfig::new([32, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization5 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d6 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(false)
            .init(device);
        let batchnormalization6 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d7 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization7 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d8 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(false)
            .init(device);
        let batchnormalization8 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d9 = Conv2dConfig::new([64, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization9 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d10 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(false)
            .init(device);
        let batchnormalization10 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d11 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization11 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d12 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(false)
            .init(device);
        let batchnormalization12 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d13 = Conv2dConfig::new([128, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization13 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d14 = Conv2dConfig::new([256, 256], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(false)
            .init(device);
        let batchnormalization14 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d15 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization15 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d16 = Conv2dConfig::new([256, 256], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(false)
            .init(device);
        let batchnormalization16 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d17 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization17 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d18 = Conv2dConfig::new([256, 256], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(false)
            .init(device);
        let batchnormalization18 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d19 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization19 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d20 = Conv2dConfig::new([256, 256], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(false)
            .init(device);
        let batchnormalization20 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d21 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization21 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d22 = Conv2dConfig::new([256, 256], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(false)
            .init(device);
        let batchnormalization22 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d23 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization23 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d24 = Conv2dConfig::new([256, 256], [5, 5])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(false)
            .init(device);
        let batchnormalization24 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let globalaveragepool1 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d25 = Conv2dConfig::new([256, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d26 = Conv2dConfig::new([64, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d27 = Conv2dConfig::new([256, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization25 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d28 = Conv2dConfig::new([512, 512], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(512)
            .with_bias(false)
            .init(device);
        let batchnormalization26 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let globalaveragepool2 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d29 = Conv2dConfig::new([512, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d30 = Conv2dConfig::new([128, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d31 = Conv2dConfig::new([512, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization27 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let globalaveragepool3 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d32 = Conv2dConfig::new([512, 1280], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let linear1 = LinearConfig::new(1280, 2).with_bias(true).init(device);
        Self {
            constant122,
            constant124,
            constant136,
            constant138,
            constant145,
            conv2d1,
            batchnormalization1,
            conv2d2,
            batchnormalization2,
            conv2d3,
            batchnormalization3,
            conv2d4,
            batchnormalization4,
            conv2d5,
            batchnormalization5,
            conv2d6,
            batchnormalization6,
            conv2d7,
            batchnormalization7,
            conv2d8,
            batchnormalization8,
            conv2d9,
            batchnormalization9,
            conv2d10,
            batchnormalization10,
            conv2d11,
            batchnormalization11,
            conv2d12,
            batchnormalization12,
            conv2d13,
            batchnormalization13,
            conv2d14,
            batchnormalization14,
            conv2d15,
            batchnormalization15,
            conv2d16,
            batchnormalization16,
            conv2d17,
            batchnormalization17,
            conv2d18,
            batchnormalization18,
            conv2d19,
            batchnormalization19,
            conv2d20,
            batchnormalization20,
            conv2d21,
            batchnormalization21,
            conv2d22,
            batchnormalization22,
            conv2d23,
            batchnormalization23,
            conv2d24,
            batchnormalization24,
            globalaveragepool1,
            conv2d25,
            conv2d26,
            conv2d27,
            batchnormalization25,
            conv2d28,
            batchnormalization26,
            globalaveragepool2,
            conv2d29,
            conv2d30,
            conv2d31,
            batchnormalization27,
            globalaveragepool3,
            conv2d32,
            linear1,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let constant122_out1 = self.constant122.val();
        let constant124_out1 = self.constant124.val();
        let constant136_out1 = self.constant136.val();
        let constant138_out1 = self.constant138.val();
        let constant145_out1 = self.constant145.val();
        let constant146_out1: [i64; 1] = [-1i64];
        let conv2d1_out1 = self.conv2d1.forward(x);
        let batchnormalization1_out1 = self.batchnormalization1.forward(conv2d1_out1);
        let hardsigmoid1_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization1_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul1_out1 = hardsigmoid1_out1.mul(batchnormalization1_out1);
        let conv2d2_out1 = self.conv2d2.forward(mul1_out1);
        let batchnormalization2_out1 = self.batchnormalization2.forward(conv2d2_out1);
        let hardsigmoid2_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization2_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul2_out1 = hardsigmoid2_out1.mul(batchnormalization2_out1);
        let conv2d3_out1 = self.conv2d3.forward(mul2_out1);
        let batchnormalization3_out1 = self.batchnormalization3.forward(conv2d3_out1);
        let hardsigmoid3_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization3_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul3_out1 = hardsigmoid3_out1.mul(batchnormalization3_out1);
        let conv2d4_out1 = self.conv2d4.forward(mul3_out1);
        let batchnormalization4_out1 = self.batchnormalization4.forward(conv2d4_out1);
        let hardsigmoid4_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization4_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul4_out1 = hardsigmoid4_out1.mul(batchnormalization4_out1);
        let conv2d5_out1 = self.conv2d5.forward(mul4_out1);
        let batchnormalization5_out1 = self.batchnormalization5.forward(conv2d5_out1);
        let hardsigmoid5_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization5_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul5_out1 = hardsigmoid5_out1.mul(batchnormalization5_out1);
        let conv2d6_out1 = self.conv2d6.forward(mul5_out1);
        let batchnormalization6_out1 = self.batchnormalization6.forward(conv2d6_out1);
        let hardsigmoid6_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization6_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul6_out1 = hardsigmoid6_out1.mul(batchnormalization6_out1);
        let conv2d7_out1 = self.conv2d7.forward(mul6_out1);
        let batchnormalization7_out1 = self.batchnormalization7.forward(conv2d7_out1);
        let hardsigmoid7_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization7_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul7_out1 = hardsigmoid7_out1.mul(batchnormalization7_out1);
        let conv2d8_out1 = self.conv2d8.forward(mul7_out1);
        let batchnormalization8_out1 = self.batchnormalization8.forward(conv2d8_out1);
        let hardsigmoid8_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization8_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul8_out1 = hardsigmoid8_out1.mul(batchnormalization8_out1);
        let conv2d9_out1 = self.conv2d9.forward(mul8_out1);
        let batchnormalization9_out1 = self.batchnormalization9.forward(conv2d9_out1);
        let hardsigmoid9_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization9_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul9_out1 = hardsigmoid9_out1.mul(batchnormalization9_out1);
        let conv2d10_out1 = self.conv2d10.forward(mul9_out1);
        let batchnormalization10_out1 = self.batchnormalization10.forward(conv2d10_out1);
        let hardsigmoid10_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization10_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul10_out1 = hardsigmoid10_out1.mul(batchnormalization10_out1);
        let conv2d11_out1 = self.conv2d11.forward(mul10_out1);
        let batchnormalization11_out1 = self.batchnormalization11.forward(conv2d11_out1);
        let hardsigmoid11_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization11_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul11_out1 = hardsigmoid11_out1.mul(batchnormalization11_out1);
        let conv2d12_out1 = self.conv2d12.forward(mul11_out1);
        let batchnormalization12_out1 = self.batchnormalization12.forward(conv2d12_out1);
        let hardsigmoid12_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization12_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul12_out1 = hardsigmoid12_out1.mul(batchnormalization12_out1);
        let conv2d13_out1 = self.conv2d13.forward(mul12_out1);
        let batchnormalization13_out1 = self.batchnormalization13.forward(conv2d13_out1);
        let hardsigmoid13_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization13_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul13_out1 = hardsigmoid13_out1.mul(batchnormalization13_out1);
        let conv2d14_out1 = self.conv2d14.forward(mul13_out1);
        let batchnormalization14_out1 = self.batchnormalization14.forward(conv2d14_out1);
        let hardsigmoid14_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization14_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul14_out1 = hardsigmoid14_out1.mul(batchnormalization14_out1);
        let conv2d15_out1 = self.conv2d15.forward(mul14_out1);
        let batchnormalization15_out1 = self.batchnormalization15.forward(conv2d15_out1);
        let hardsigmoid15_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization15_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul15_out1 = hardsigmoid15_out1.mul(batchnormalization15_out1);
        let conv2d16_out1 = self.conv2d16.forward(mul15_out1);
        let batchnormalization16_out1 = self.batchnormalization16.forward(conv2d16_out1);
        let hardsigmoid16_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization16_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul16_out1 = hardsigmoid16_out1.mul(batchnormalization16_out1);
        let conv2d17_out1 = self.conv2d17.forward(mul16_out1);
        let batchnormalization17_out1 = self.batchnormalization17.forward(conv2d17_out1);
        let hardsigmoid17_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization17_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul17_out1 = hardsigmoid17_out1.mul(batchnormalization17_out1);
        let conv2d18_out1 = self.conv2d18.forward(mul17_out1);
        let batchnormalization18_out1 = self.batchnormalization18.forward(conv2d18_out1);
        let hardsigmoid18_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization18_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul18_out1 = hardsigmoid18_out1.mul(batchnormalization18_out1);
        let conv2d19_out1 = self.conv2d19.forward(mul18_out1);
        let batchnormalization19_out1 = self.batchnormalization19.forward(conv2d19_out1);
        let hardsigmoid19_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization19_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul19_out1 = hardsigmoid19_out1.mul(batchnormalization19_out1);
        let conv2d20_out1 = self.conv2d20.forward(mul19_out1);
        let batchnormalization20_out1 = self.batchnormalization20.forward(conv2d20_out1);
        let hardsigmoid20_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization20_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul20_out1 = hardsigmoid20_out1.mul(batchnormalization20_out1);
        let conv2d21_out1 = self.conv2d21.forward(mul20_out1);
        let batchnormalization21_out1 = self.batchnormalization21.forward(conv2d21_out1);
        let hardsigmoid21_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization21_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul21_out1 = hardsigmoid21_out1.mul(batchnormalization21_out1);
        let conv2d22_out1 = self.conv2d22.forward(mul21_out1);
        let batchnormalization22_out1 = self.batchnormalization22.forward(conv2d22_out1);
        let hardsigmoid22_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization22_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul22_out1 = hardsigmoid22_out1.mul(batchnormalization22_out1);
        let conv2d23_out1 = self.conv2d23.forward(mul22_out1);
        let batchnormalization23_out1 = self.batchnormalization23.forward(conv2d23_out1);
        let hardsigmoid23_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization23_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul23_out1 = hardsigmoid23_out1.mul(batchnormalization23_out1);
        let conv2d24_out1 = self.conv2d24.forward(mul23_out1);
        let batchnormalization24_out1 = self.batchnormalization24.forward(conv2d24_out1);
        let hardsigmoid24_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization24_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul24_out1 = hardsigmoid24_out1.mul(batchnormalization24_out1);
        let globalaveragepool1_out1 = self
            .globalaveragepool1
            .forward(mul24_out1.clone());
        let conv2d25_out1 = self.conv2d25.forward(globalaveragepool1_out1);
        let add1_out1 = conv2d25_out1.add(constant122_out1);
        let relu1_out1 = burn::tensor::activation::relu(add1_out1);
        let conv2d26_out1 = self.conv2d26.forward(relu1_out1);
        let add2_out1 = conv2d26_out1.add(constant124_out1);
        let hardsigmoid25_out1 = burn::tensor::activation::hard_sigmoid(
            add2_out1,
            0.16666670143604279,
            0.5,
        );
        let mul25_out1 = mul24_out1.mul(hardsigmoid25_out1);
        let conv2d27_out1 = self.conv2d27.forward(mul25_out1);
        let batchnormalization25_out1 = self.batchnormalization25.forward(conv2d27_out1);
        let hardsigmoid26_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization25_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul26_out1 = hardsigmoid26_out1.mul(batchnormalization25_out1);
        let conv2d28_out1 = self.conv2d28.forward(mul26_out1);
        let batchnormalization26_out1 = self.batchnormalization26.forward(conv2d28_out1);
        let hardsigmoid27_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization26_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul27_out1 = hardsigmoid27_out1.mul(batchnormalization26_out1);
        let globalaveragepool2_out1 = self
            .globalaveragepool2
            .forward(mul27_out1.clone());
        let conv2d29_out1 = self.conv2d29.forward(globalaveragepool2_out1);
        let add3_out1 = conv2d29_out1.add(constant136_out1);
        let relu2_out1 = burn::tensor::activation::relu(add3_out1);
        let conv2d30_out1 = self.conv2d30.forward(relu2_out1);
        let add4_out1 = conv2d30_out1.add(constant138_out1);
        let hardsigmoid28_out1 = burn::tensor::activation::hard_sigmoid(
            add4_out1,
            0.16666670143604279,
            0.5,
        );
        let mul28_out1 = mul27_out1.mul(hardsigmoid28_out1);
        let conv2d31_out1 = self.conv2d31.forward(mul28_out1);
        let batchnormalization27_out1 = self.batchnormalization27.forward(conv2d31_out1);
        let hardsigmoid29_out1 = burn::tensor::activation::hard_sigmoid(
            batchnormalization27_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul29_out1 = hardsigmoid29_out1.mul(batchnormalization27_out1);
        let globalaveragepool3_out1 = self.globalaveragepool3.forward(mul29_out1);
        let conv2d32_out1 = self.conv2d32.forward(globalaveragepool3_out1);
        let hardsigmoid30_out1 = burn::tensor::activation::hard_sigmoid(
            conv2d32_out1.clone(),
            0.1666666716337204,
            0.5,
        );
        let mul30_out1 = hardsigmoid30_out1.mul(conv2d32_out1);
        let mul31_out1 = mul30_out1
            .mul((constant145_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let shape1_out1: [i64; 4] = {
            let axes = &mul31_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice1_out1: [i64; 1] = shape1_out1[0..1].try_into().unwrap();
        let concat1_out1: [i64; 2usize] = [&slice1_out1[..], &constant146_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let reshape1_out1 = mul31_out1.reshape(concat1_out1);
        let linear1_out1 = self.linear1.forward(reshape1_out1);
        let softmax1_out1 = burn::tensor::activation::softmax(linear1_out1, 1);
        softmax1_out1
    }
}
