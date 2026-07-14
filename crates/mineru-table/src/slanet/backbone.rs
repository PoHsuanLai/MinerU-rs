//! SLANet-plus PP-LCNet backbone + CSP-PAN neck.
//!
//! Faithful Burn port of the feed-forward CNN that the SLANet-plus ONNX graph
//! runs before the attention head. It is a **PP-LCNet** backbone (depthwise-
//! separable stages with two squeeze-excite blocks) whose four intermediate taps
//! feed a **CSP-PAN** neck (top-down nearest-neighbour upsampling + bottom-up
//! strided fusion, each fusion a small CSP block of `1×1` + depthwise `5×5`
//! convs). The neck emits a single `[1, 96, 16, 16]` feature map which is
//! flattened to the `[1, 256, 96]` sequence the head attends over.
//!
//! `burn-onnx` codegen cannot import the full SLANet-plus graph (its `Loop` and
//! the surrounding `ConstantOfShape` type inference fail — see [`super::model`]),
//! so the whole network including this backbone is hand-ported.
//!
//! # Weight naming
//!
//! Convolutions and batch-norms are addressed by their ONNX node index so the
//! converted `.safetensors` keys line up with the Burn field paths:
//! `conv.<N>.weight` / `conv.<N>.bias` and
//! `bn.<M>.{weight,bias,running_mean,running_var}`. The `conv`/`bn` `Vec`s are
//! indexed exactly as the ONNX `Conv.N` / `BatchNormalization.M` numbering.
//!
//! Every conv is followed by batch-norm and a `HardSwish` activation *unless* it
//! is one of the two squeeze-excite `1×1` convs (which use ReLU / HardSigmoid) or
//! a bare fusion conv; the forward pass encodes those exceptions explicitly.

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::prelude::Backend;
use burn::tensor::Tensor;

use mineru_burn_common::nn::FrozenBatchNorm2d;

/// BatchNorm epsilon from the ONNX graph (`epsilon=9.999999747378752e-06`).
const BN_EPS: f64 = 1e-5;

/// `(out, in_per_group, kernel, stride, groups, has_bias)` spec for one conv,
/// derived from the ONNX `Conv` node attributes (padding is always `kernel/2`).
type ConvCfg = (usize, usize, usize, usize, usize, bool);

/// The 77 conv specs, indexed by ONNX `Conv.N`.
const CONV_CFGS: [ConvCfg; 77] = [
    // backbone stem + stages
    (16, 3, 3, 2, 1, false),    //0
    (16, 1, 3, 1, 16, false),   //1
    (32, 16, 1, 1, 1, false),   //2
    (32, 1, 3, 2, 32, false),   //3
    (64, 32, 1, 1, 1, false),   //4
    (64, 1, 3, 1, 64, false),   //5
    (64, 64, 1, 1, 1, false),   //6  <- tap0 source (hardswish_6)
    (64, 1, 3, 2, 64, false),   //7
    (128, 64, 1, 1, 1, false),  //8
    (128, 1, 3, 1, 128, false), //9
    (128, 128, 1, 1, 1, false), //10 <- tap1 source (hardswish_10)
    (128, 1, 3, 2, 128, false), //11
    (256, 128, 1, 1, 1, false), //12
    (256, 1, 5, 1, 256, false), //13
    (256, 256, 1, 1, 1, false), //14
    (256, 1, 5, 1, 256, false), //15
    (256, 256, 1, 1, 1, false), //16
    (256, 1, 5, 1, 256, false), //17
    (256, 256, 1, 1, 1, false), //18
    (256, 1, 5, 1, 256, false), //19
    (256, 256, 1, 1, 1, false), //20
    (256, 1, 5, 1, 256, false), //21
    (256, 256, 1, 1, 1, false), //22 <- tap2 source (hardswish_22)
    (256, 1, 5, 2, 256, false), //23  (SE follows on hardswish_23)
    (64, 256, 1, 1, 1, true),   //24  SE reduce
    (256, 64, 1, 1, 1, true),   //25  SE expand
    (512, 256, 1, 1, 1, false), //26
    (512, 1, 5, 1, 512, false), //27  (SE follows on hardswish_25)
    (128, 512, 1, 1, 1, true),  //28  SE reduce
    (512, 128, 1, 1, 1, true),  //29  SE expand
    (512, 512, 1, 1, 1, false), //30 <- tap3 source (hardswish_26)
    // neck feature taps -> 96 channels each
    (96, 64, 1, 1, 1, false),   //31 tap0 (from hardswish_6)  -> hardswish_27
    (96, 128, 1, 1, 1, false),  //32 tap1 (from hardswish_10) -> hardswish_28
    (96, 256, 1, 1, 1, false),  //33 tap2 (from hardswish_22) -> hardswish_29
    (96, 512, 1, 1, 1, false),  //34 tap3 (from hardswish_26) -> hardswish_30
    // CSP-PAN fusion blocks (see forward for wiring)
    (48, 192, 1, 1, 1, false),  //35
    (48, 192, 1, 1, 1, false),  //36
    (48, 48, 1, 1, 1, false),   //37
    (48, 1, 5, 1, 48, false),   //38
    (48, 48, 1, 1, 1, false),   //39
    (96, 96, 1, 1, 1, false),   //40
    (48, 192, 1, 1, 1, false),  //41
    (48, 192, 1, 1, 1, false),  //42
    (48, 48, 1, 1, 1, false),   //43
    (48, 1, 5, 1, 48, false),   //44
    (48, 48, 1, 1, 1, false),   //45
    (96, 96, 1, 1, 1, false),   //46
    (48, 192, 1, 1, 1, false),  //47
    (48, 192, 1, 1, 1, false),  //48
    (48, 48, 1, 1, 1, false),   //49
    (48, 1, 5, 1, 48, false),   //50
    (48, 48, 1, 1, 1, false),   //51
    (96, 96, 1, 1, 1, false),   //52
    (96, 1, 5, 2, 96, false),   //53
    (96, 96, 1, 1, 1, false),   //54
    (48, 192, 1, 1, 1, false),  //55
    (48, 192, 1, 1, 1, false),  //56
    (48, 48, 1, 1, 1, false),   //57
    (48, 1, 5, 1, 48, false),   //58
    (48, 48, 1, 1, 1, false),   //59
    (96, 96, 1, 1, 1, false),   //60
    (96, 1, 5, 2, 96, false),   //61
    (96, 96, 1, 1, 1, false),   //62
    (48, 192, 1, 1, 1, false),  //63
    (48, 192, 1, 1, 1, false),  //64
    (48, 48, 1, 1, 1, false),   //65
    (48, 1, 5, 1, 48, false),   //66
    (48, 48, 1, 1, 1, false),   //67
    (96, 96, 1, 1, 1, false),   //68
    (96, 1, 5, 2, 96, false),   //69
    (96, 96, 1, 1, 1, false),   //70
    (48, 192, 1, 1, 1, false),  //71
    (48, 192, 1, 1, 1, false),  //72
    (48, 48, 1, 1, 1, false),   //73
    (48, 1, 5, 1, 48, false),   //74
    (48, 48, 1, 1, 1, false),   //75
    (96, 96, 1, 1, 1, false),   //76
];

/// Per-conv output-channel count for the matching batch-norm (indexed by ONNX
/// `BatchNormalization.M`). BN follows every conv except the four SE `1×1` convs
/// (24, 25, 28, 29), so this is the conv output channels with those removed, in
/// order.
fn bn_channels() -> Vec<usize> {
    CONV_CFGS
        .iter()
        .enumerate()
        .filter(|(i, _)| !matches!(i, 24 | 25 | 28 | 29))
        .map(|(_, c)| c.0)
        .collect()
}

/// Applies the ONNX `HardSwish`: `x · relu6(x + 3) / 6`.
fn hardswish<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let relu6 = x.clone().add_scalar(3.0).clamp(0.0, 6.0);
    x.mul(relu6).div_scalar(6.0)
}

/// Applies the ONNX `HardSigmoid` with `alpha=1/6, beta=0.5`: `clip(x/6 + 0.5, 0, 1)`.
fn hardsigmoid<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    x.div_scalar(6.0).add_scalar(0.5).clamp(0.0, 1.0)
}

/// Nearest-neighbour upsample of `x` to match `target`'s spatial size.
///
/// The ONNX neck upsamples each top-down feature to its skip partner's `H×W`
/// (an integer ×2 factor here); `interpolate` with nearest mode reproduces the
/// graph's `Resize`.
fn upsample_to<B: Backend>(x: Tensor<B, 4>, target: &Tensor<B, 4>) -> Tensor<B, 4> {
    use burn::tensor::module::interpolate;
    use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
    let [_, _, h, w] = target.dims();
    interpolate(
        x,
        [h, w],
        InterpolateOptions::new(InterpolateMode::Nearest),
    )
}

/// The backbone + neck. Convs and batch-norms are stored in flat index-addressed
/// `Vec`s so the converted checkpoint keys (`conv.<N>.*`, `bn.<M>.*`) map onto the
/// Burn field paths directly.
#[derive(Module, Debug)]
pub struct Backbone<B: Backend> {
    conv: Vec<Conv2d<B>>,
    bn: Vec<FrozenBatchNorm2d<B>>,
}

impl<B: Backend> Backbone<B> {
    /// Builds the zero-initialised backbone; weights are overwritten by loading.
    pub fn new(device: &B::Device) -> Self {
        let conv = CONV_CFGS
            .iter()
            .map(|&(out, in_pg, k, s, g, bias)| {
                let pad = k / 2;
                Conv2dConfig::new([in_pg * g, out], [k, k])
                    .with_stride([s, s])
                    .with_groups(g)
                    .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
                    .with_bias(bias)
                    .init(device)
            })
            .collect();
        let bn = bn_channels()
            .into_iter()
            .map(|c| FrozenBatchNorm2d::init(c, BN_EPS, device))
            .collect();
        Self { conv, bn }
    }

    /// Conv `ci` → BN `bi` → HardSwish.
    fn cbh(&self, x: Tensor<B, 4>, ci: usize, bi: usize) -> Tensor<B, 4> {
        hardswish(self.bn[bi].forward(self.conv[ci].forward(x)))
    }

    /// One PP-LCNet squeeze-excite block: global-avg-pool → `reduce`(ReLU) →
    /// `expand`(HardSigmoid) → channel-wise gate. `rc`/`ec` are the reduce/expand
    /// conv indices.
    fn se(&self, x: Tensor<B, 4>, rc: usize, ec: usize) -> Tensor<B, 4> {
        let pooled = x.clone().mean_dim(3).mean_dim(2); // [N, C, 1, 1]
        let s = burn::tensor::activation::relu(self.conv[rc].forward(pooled));
        let s = hardsigmoid(self.conv[ec].forward(s));
        x.mul(s)
    }

    /// Runs the backbone + neck, returning the final `[1, 96, 16, 16]` feature map.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // --- Backbone (PP-LCNet). BN index runs 0.. skipping the SE convs. ---
        let h = self.cbh(x, 0, 0);
        let h = self.cbh(h, 1, 1);
        let h = self.cbh(h, 2, 2);
        let h = self.cbh(h, 3, 3);
        let h = self.cbh(h, 4, 4);
        let h = self.cbh(h, 5, 5);
        let hs6 = self.cbh(h, 6, 6); // tap0
        let h = self.cbh(hs6.clone(), 7, 7);
        let h = self.cbh(h, 8, 8);
        let h = self.cbh(h, 9, 9);
        let hs10 = self.cbh(h, 10, 10); // tap1
        let h = self.cbh(hs10.clone(), 11, 11);
        let h = self.cbh(h, 12, 12);
        let h = self.cbh(h, 13, 13);
        let h = self.cbh(h, 14, 14);
        let h = self.cbh(h, 15, 15);
        let h = self.cbh(h, 16, 16);
        let h = self.cbh(h, 17, 17);
        let h = self.cbh(h, 18, 18);
        let h = self.cbh(h, 19, 19);
        let h = self.cbh(h, 20, 20);
        let h = self.cbh(h, 21, 21);
        let hs22 = self.cbh(h, 22, 22); // tap2
        let hs23 = self.cbh(hs22.clone(), 23, 23);
        let h = self.se(hs23, 24, 25); // SE block 0 (convs 24/25, no BN)
        let h = self.cbh(h, 26, 24); // BN.24
        let hs25 = self.cbh(h, 27, 25); // BN.25
        let h = self.se(hs25, 28, 29); // SE block 1 (convs 28/29, no BN)
        let hs26 = self.cbh(h, 30, 26); // BN.26  tap3

        // --- Neck: project the four taps to 96 channels (BN 27..30). ---
        let n0 = self.cbh(hs6, 31, 27); // hardswish_27  (highest res)
        let n1 = self.cbh(hs10, 32, 28); // hardswish_28
        let n2 = self.cbh(hs22, 33, 29); // hardswish_29
        let n3 = self.cbh(hs26, 34, 30); // hardswish_30 (lowest res)

        // --- CSP-PAN top-down (upsample + fuse). ---
        // Level A: up(n3) ⊕ n2  -> csp(35..40) -> td_a  (BN 31..36)
        let up = upsample_to(n3.clone(), &n2);
        let cat = Tensor::cat(vec![up, n2.clone()], 1);
        let td_a = self.csp(cat, [35, 36, 37, 38, 39, 40], [31, 32, 33, 34, 35, 36]);
        // Level B: up(td_a) ⊕ n1 -> csp(41..46) -> td_b  (BN 37..42)
        let up = upsample_to(td_a.clone(), &n1);
        let cat = Tensor::cat(vec![up, n1.clone()], 1);
        let td_b = self.csp(cat, [41, 42, 43, 44, 45, 46], [37, 38, 39, 40, 41, 42]);
        // Level C: up(td_b) ⊕ n0 -> csp(47..52) -> td_c  (BN 43..48)  (finest)
        let up = upsample_to(td_b.clone(), &n0);
        let cat = Tensor::cat(vec![up, n0], 1);
        let td_c = self.csp(cat, [47, 48, 49, 50, 51, 52], [43, 44, 45, 46, 47, 48]);

        // --- CSP-PAN bottom-up (downsample + fuse). ---
        // down(td_c) ⊕ td_b -> csp(55..60) -> bu_a  (BN 51..56); conv53/54 downsample (BN 49/50)
        let down = self.cbh(td_c.clone(), 53, 49);
        let down = self.cbh(down, 54, 50);
        let cat = Tensor::cat(vec![down, td_b], 1);
        let bu_a = self.csp(cat, [55, 56, 57, 58, 59, 60], [51, 52, 53, 54, 55, 56]);
        // down(bu_a) ⊕ td_a -> csp(63..68) -> bu_b  (BN 59..64); conv61/62 downsample (BN 57/58)
        let down = self.cbh(bu_a.clone(), 61, 57);
        let down = self.cbh(down, 62, 58);
        let cat = Tensor::cat(vec![down, td_a], 1);
        let bu_b = self.csp(cat, [63, 64, 65, 66, 67, 68], [59, 60, 61, 62, 63, 64]);
        // down(bu_b) ⊕ n3 -> csp(71..76) -> bu_c  (BN 67..72); conv69/70 downsample (BN 65/66)
        let down = self.cbh(bu_b, 69, 65);
        let down = self.cbh(down, 70, 66);
        let cat = Tensor::cat(vec![down, n3], 1);
        self.csp(cat, [71, 72, 73, 74, 75, 76], [67, 68, 69, 70, 71, 72])
    }

    /// One CSP fusion block, matching the neck's repeated 6-conv pattern.
    ///
    /// Splits the input into two `1×1`-projected halves; the second half goes
    /// through `1×1` → depthwise `5×5` → `1×1`; the two halves are concatenated and
    /// fused by a final `1×1`. `c` are the six conv indices, `b` the six BN indices
    /// (all `Conv→BN→HardSwish`).
    fn csp(&self, x: Tensor<B, 4>, c: [usize; 6], b: [usize; 6]) -> Tensor<B, 4> {
        let branch_a = self.cbh(x.clone(), c[0], b[0]); // 1x1 -> 48
        let m = self.cbh(x, c[1], b[1]); // 1x1 -> 48
        let m = self.cbh(m, c[2], b[2]); // 1x1 -> 48
        let m = self.cbh(m, c[3], b[3]); // dw 5x5 -> 48
        let branch_b = self.cbh(m, c[4], b[4]); // 1x1 -> 48
        let cat = Tensor::cat(vec![branch_b, branch_a], 1); // 96
        self.cbh(cat, c[5], b[5]) // 1x1 -> 96
    }

    /// Runs the backbone + neck and flattens to the head's `[1, T, 96]` feature
    /// sequence, where `T = H·W` of the `[1, 96, H, W]` neck output.
    pub fn forward_sequence(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let feat = self.forward(x); // [1, 96, H, W]
        let [n, c, h, w] = feat.dims();
        // [1, 96, H*W] -> transpose -> [1, H*W, 96].
        feat.reshape([n, c, h * w]).swap_dims(1, 2)
    }
}
