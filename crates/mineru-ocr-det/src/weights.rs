//! The safetensors-key → Burn-field remap for PP-OCRv6 *small det*.
//!
//! The shared loader ([`mineru_burn_common::weights::load_weights`]) applies the
//! [`KeyRemap`] built here, then asserts every source key was consumed
//! ([`Coverage::Strict`]). Because this crate's modules store parameters in the
//! checkpoint's own layout and naming (Conv2d weights as `[out, in, kh, kw]` under
//! `convolution.weight`, batch-norm buffers as `weight`/`bias`/`running_mean`/
//! `running_var` via [`mineru_burn_common::nn::FrozenBatchNorm2d`]), most leaf
//! names already match. The remap only bridges *structural* differences:
//!
//! - Squeeze-excitation convs sit at `…squeeze_excitation.convolutions.0` /
//!   `.convolutions.2` in the checkpoint (indices `1`/`3` are parameterless
//!   activations); this crate stores them as named `reduce` / `expand` fields.
//! - The reparameterised depthwise conv appears as a bare conv
//!   (`token_conv.weight` / `token_conv.bias`) in blocks where `stride==1 &&
//!   in==out`; this crate stores that in the `token_conv_rep` field to keep it
//!   distinct from the conv-BN `token_conv` used in the strided blocks. The rule
//!   only rewrites the bare-conv leaves, so `token_conv.convolution.*` /
//!   `token_conv.normalization.*` (the conv-BN form) are left untouched.
//!
//! The top-level prefixes (`model.backbone.` / `model.neck.` / `head.`) already
//! match the Burn module tree ([`crate::model`] nests `backbone`/`neck` under a
//! `model` field and keeps `head` at the top level), so no prefix rewriting is
//! needed. Ordering matters: the specific SE / token_conv rules run and the
//! generic identity of every other key is preserved.

use mineru_burn_common::weights::{Coverage, KeyRemap};

use crate::error::{Error, Result};

/// Builds the HF→Burn key remap for the PP-OCRv6 small-det checkpoint.
///
/// # Errors
/// Returns [`Error::Config`] if any rule's regex is invalid (should not happen).
pub fn key_remap() -> Result<KeyRemap> {
    // Each rule is (regex, replacement). Applied in order.
    let rules: &[(&str, &str)] = &[
        // Squeeze-excitation: Sequential members .0 / .2 -> named reduce / expand.
        // Matches both the backbone (`token_squeeze_excitation.convolutions.*`) and
        // is harmless elsewhere (the neck SE convs are already named conv1/conv2).
        (r"\.convolutions\.0\.", ".reduce."),
        (r"\.convolutions\.2\.", ".expand."),
        // Reparameterised depthwise conv: the bare-conv leaves `token_conv.weight`
        // / `token_conv.bias` go to the `token_conv_rep` field. The conv-BN form
        // `token_conv.convolution.*` / `token_conv.normalization.*` is NOT matched
        // (no `.weight`/`.bias` directly on `token_conv`), so it stays as-is.
        (r"\.token_conv\.(weight|bias)$", ".token_conv_rep.$1"),
    ];

    let mut remap = KeyRemap::new();
    for (from, to) in rules {
        remap = remap
            .rename(*from, *to)
            .map_err(|e| Error::Config(format!("invalid remap rule {from:?}: {e}")))?;
    }
    Ok(remap)
}

/// Coverage policy for loading. Strict: every source key must map to a field, or
/// loading fails with the unmapped list.
pub const COVERAGE: Coverage = Coverage::Strict;

#[cfg(test)]
mod tests {
    use super::*;

    fn remap(key: &str) -> String {
        key_remap()
            .expect("valid remap")
            .apply_str(key)
            .unwrap_or_else(|| key.to_string())
    }

    #[test]
    fn squeeze_excitation_indices() {
        assert_eq!(
            remap(
                "model.backbone.encoder.blocks.0.blocks.0.token_squeeze_excitation.convolutions.0.weight"
            ),
            "model.backbone.encoder.blocks.0.blocks.0.token_squeeze_excitation.reduce.weight"
        );
        assert_eq!(
            remap(
                "model.backbone.encoder.blocks.0.blocks.0.token_squeeze_excitation.convolutions.2.bias"
            ),
            "model.backbone.encoder.blocks.0.blocks.0.token_squeeze_excitation.expand.bias"
        );
    }

    #[test]
    fn reparameterised_depthwise_conv() {
        // Bare conv leaves -> token_conv_rep.
        assert_eq!(
            remap("model.backbone.encoder.blocks.0.blocks.0.token_conv.weight"),
            "model.backbone.encoder.blocks.0.blocks.0.token_conv_rep.weight"
        );
        assert_eq!(
            remap("model.backbone.encoder.blocks.0.blocks.0.token_conv.bias"),
            "model.backbone.encoder.blocks.0.blocks.0.token_conv_rep.bias"
        );
        // Conv-BN form is left untouched.
        assert_eq!(
            remap("model.backbone.encoder.blocks.1.blocks.0.token_conv.convolution.weight"),
            "model.backbone.encoder.blocks.1.blocks.0.token_conv.convolution.weight"
        );
        assert_eq!(
            remap("model.backbone.encoder.blocks.1.blocks.0.token_conv.normalization.running_mean"),
            "model.backbone.encoder.blocks.1.blocks.0.token_conv.normalization.running_mean"
        );
    }

    #[test]
    fn plain_keys_unchanged() {
        for k in [
            "model.backbone.encoder.convolution.stem1.convolution.weight",
            "model.backbone.encoder.convolution.stem1.normalization.running_var",
            "model.neck.input_conv.0.depthwise_convolution.weight",
            "model.neck.insert_conv.0.in_conv.weight",
            "head.conv_down.convolution.weight",
            "head.conv_final.bias",
        ] {
            assert_eq!(remap(k), k, "key should be unchanged: {k}");
        }
    }
}
