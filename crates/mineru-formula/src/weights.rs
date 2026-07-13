//! Checkpoint-key → Burn-field remapping for weight loading.
//!
//! The `unimernet_hf_small_2503` checkpoint is a HuggingFace
//! `VisionEncoderDecoderModel` safetensors file. Its tensor keys are prefixed:
//!
//! - `encoder.…` for the Swin encoder (`UnimerSwinModel`);
//! - `decoder.…` for the MBart decoder (`UnimerMBartForCausalLM`).
//!
//! Burn derives field paths from the Rust struct field names. This crate's modules
//! store their parameters in the checkpoint's own tensor *layout* and leaf *naming*
//! (see [`mineru_burn_common::nn`]: `PtLinear` keeps the `[out, in]` weight and
//! `weight`/`bias` names, `PtLayerNorm` keeps `weight`/`bias`, `FrozenBatchNorm2d`
//! keeps `weight`/`bias`/`running_mean`/`running_var`). So the remap below only
//! bridges *structural* prefix/index differences, never per-leaf renames.
//!
//! The rules were verified against the actual tensor names dumped from the file
//! (763 keys); see [`IGNORED_KEYS`] for the handful of training-only / computed
//! buffers that carry no inference field.

use mineru_burn_common::weights::{Coverage, KeyRemap};

use crate::error::Result;

/// Coverage policy for loading: every source key must map to a field (or be an
/// [`IGNORED_KEYS`] buffer) or loading fails with the unmapped list.
pub const COVERAGE: Coverage = Coverage::Strict;

/// Checkpoint keys (post-remap) that inference intentionally does not load.
///
/// - `…blocks.N.attention.self.relative_position_index` (24 of them): a fixed int64
///   buffer PyTorch `register_buffer`s. This crate recomputes the identical index
///   table at forward time (see [`crate::swin::attention::relative_position_index`]),
///   so there is no parameter field to load it into.
/// - `…projection.norm1.0.num_batches_tracked`: PyTorch stores a BatchNorm batch
///   counter, a training-only int64 scalar that eval-mode (frozen) inference never
///   reads. `FrozenBatchNorm2d` has no field for it.
///
/// These are the only source tensors with no inference field; every real weight is
/// routed through the [`build_remap`] rules below.
pub const IGNORED_KEYS: &[&str] = &[
    // Post-remap Swin block paths: encoder.encoder.layers.N -> encoder.stages.N,
    // .attention.self. -> .attention. .
    "encoder.stages.0.blocks.0.attention.relative_position_index",
    "encoder.stages.0.blocks.1.attention.relative_position_index",
    "encoder.stages.0.blocks.2.attention.relative_position_index",
    "encoder.stages.0.blocks.3.attention.relative_position_index",
    "encoder.stages.0.blocks.4.attention.relative_position_index",
    "encoder.stages.0.blocks.5.attention.relative_position_index",
    "encoder.stages.1.blocks.0.attention.relative_position_index",
    "encoder.stages.1.blocks.1.attention.relative_position_index",
    "encoder.stages.1.blocks.2.attention.relative_position_index",
    "encoder.stages.1.blocks.3.attention.relative_position_index",
    "encoder.stages.1.blocks.4.attention.relative_position_index",
    "encoder.stages.1.blocks.5.attention.relative_position_index",
    "encoder.stages.2.blocks.0.attention.relative_position_index",
    "encoder.stages.2.blocks.1.attention.relative_position_index",
    "encoder.stages.2.blocks.2.attention.relative_position_index",
    "encoder.stages.2.blocks.3.attention.relative_position_index",
    "encoder.stages.2.blocks.4.attention.relative_position_index",
    "encoder.stages.2.blocks.5.attention.relative_position_index",
    "encoder.stages.3.blocks.0.attention.relative_position_index",
    "encoder.stages.3.blocks.1.attention.relative_position_index",
    "encoder.stages.3.blocks.2.attention.relative_position_index",
    "encoder.stages.3.blocks.3.attention.relative_position_index",
    "encoder.stages.3.blocks.4.attention.relative_position_index",
    "encoder.stages.3.blocks.5.attention.relative_position_index",
    // Frozen BatchNorm training counter (no inference field). Post-remap:
    // encoder.embeddings.patch_embeddings.projection.norm1.0 ->
    // encoder.embeddings.projection.norm1.bn .
    "encoder.embeddings.projection.norm1.bn.num_batches_tracked",
];

/// Builds the checkpoint-key remapper for the full UniMerNet model.
///
/// The top-level Burn module exposes the encoder under field `encoder` and the
/// decoder under field `decoder` (see [`crate::model::UniMerNet`]).
///
/// Rules are applied in insertion order; more specific rules precede the general
/// prefix strips they would otherwise be shadowed by.
///
/// # Errors
/// Returns [`crate::Error`] if any remap regex is invalid (they are literals here,
/// so this is effectively infallible but kept in `Result` for the harness API).
pub fn build_remap() -> Result<KeyRemap> {
    let remap = KeyRemap::new()
        // ---- Encoder (Swin) -------------------------------------------------
        // Stem: strip the `patch_embeddings` nesting; the `norm1` Sequential's
        // `.0` member becomes the named `bn` field of `Norm1Seq`.
        .rename(
            r"^encoder\.embeddings\.patch_embeddings\.projection\.norm1\.0\.",
            "encoder.embeddings.projection.norm1.bn.",
        )?
        .rename(
            r"^encoder\.embeddings\.patch_embeddings\.projection\.",
            "encoder.embeddings.projection.",
        )?
        // encoder.embeddings.norm.* already matches (PatchEmbeddings.norm).
        // Stages live at encoder.encoder.layers.N.*; ours at encoder.stages.N.*.
        .rename(r"^encoder\.encoder\.layers\.", "encoder.stages.")?
        // Window attention: HF nests `attention.self.{query,key,value}` and
        // `attention.output.dense`; our fields are `attention.{query,key,value}`
        // and `attention.output`.
        .rename(r"\.attention\.self\.", ".attention.")?
        .rename(r"\.attention\.output\.dense\.", ".attention.output.")?
        // FFN: HF wraps the two projections in `UnimerSwinIntermediate.dense` /
        // `UnimerSwinOutput.dense`; our fields are the bare `intermediate` /
        // `output` PtLinear. (The `.attention.output.dense.` rule above already
        // rewrote the attention-output case, so these only match the FFN.)
        .rename(r"\.blocks\.(\d+)\.intermediate\.dense\.", ".blocks.$1.intermediate.")?
        .rename(r"\.blocks\.(\d+)\.output\.dense\.", ".blocks.$1.output.")?
        // `ce.0.*` / `ce.1.*` (the ConvEnhance ModuleList) map to our `ce` Vec
        // directly — no rule needed. `layernorm_before/after`,
        // `downsample.{norm,reduction}` already match leaf-for-leaf.
        // ---- Decoder (MBart, wrapped) --------------------------------------
        // decoder.model.decoder.* -> decoder.*  (strip the ForCausalLM + Wrapper
        // nesting). decoder.lm_head.* already matches (MBartDecoder.lm_head).
        .rename(r"^decoder\.model\.decoder\.", "decoder.")?;
    Ok(remap)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn remap(key: &str) -> String {
        build_remap()
            .expect("remap builds")
            .apply_str(key)
            .unwrap_or_else(|| key.to_string())
    }

    #[test]
    fn encoder_stem_keys_are_rewritten() {
        assert_eq!(
            remap("encoder.embeddings.patch_embeddings.projection.conv1.weight"),
            "encoder.embeddings.projection.conv1.weight",
        );
        assert_eq!(
            remap("encoder.embeddings.patch_embeddings.projection.conv2.bias"),
            "encoder.embeddings.projection.conv2.bias",
        );
        // norm1 Sequential .0 -> named bn field.
        assert_eq!(
            remap("encoder.embeddings.patch_embeddings.projection.norm1.0.running_mean"),
            "encoder.embeddings.projection.norm1.bn.running_mean",
        );
        // Patch-embedding LayerNorm is already at the right path.
        assert_eq!(
            remap("encoder.embeddings.norm.weight"),
            "encoder.embeddings.norm.weight",
        );
    }

    #[test]
    fn encoder_block_keys_are_rewritten() {
        // attention.self.query -> attention.query.
        assert_eq!(
            remap("encoder.encoder.layers.0.blocks.0.attention.self.query.weight"),
            "encoder.stages.0.blocks.0.attention.query.weight",
        );
        // attention.output.dense -> attention.output.
        assert_eq!(
            remap("encoder.encoder.layers.2.blocks.1.attention.output.dense.bias"),
            "encoder.stages.2.blocks.1.attention.output.bias",
        );
        // relative_position_bias_table is a real param and must remap.
        assert_eq!(
            remap("encoder.encoder.layers.3.blocks.5.attention.self.relative_position_bias_table"),
            "encoder.stages.3.blocks.5.attention.relative_position_bias_table",
        );
        // ce ModuleList index survives (maps into the `ce` Vec).
        assert_eq!(
            remap("encoder.encoder.layers.0.blocks.0.ce.1.proj.weight"),
            "encoder.stages.0.blocks.0.ce.1.proj.weight",
        );
        // FFN + layernorm leaves.
        // FFN wrappers: intermediate.dense / output.dense -> bare intermediate / output.
        assert_eq!(
            remap("encoder.encoder.layers.1.blocks.0.intermediate.dense.weight"),
            "encoder.stages.1.blocks.0.intermediate.weight",
        );
        assert_eq!(
            remap("encoder.encoder.layers.1.blocks.0.output.dense.bias"),
            "encoder.stages.1.blocks.0.output.bias",
        );
        assert_eq!(
            remap("encoder.encoder.layers.1.blocks.0.layernorm_before.bias"),
            "encoder.stages.1.blocks.0.layernorm_before.bias",
        );
        // downsample.
        assert_eq!(
            remap("encoder.encoder.layers.0.downsample.reduction.weight"),
            "encoder.stages.0.downsample.reduction.weight",
        );
    }

    #[test]
    fn decoder_keys_are_rewritten() {
        assert_eq!(
            remap("decoder.model.decoder.layers.0.self_attn.q_proj.weight"),
            "decoder.layers.0.self_attn.q_proj.weight",
        );
        assert_eq!(
            remap("decoder.model.decoder.layer_norm.weight"),
            "decoder.layer_norm.weight",
        );
        assert_eq!(
            remap("decoder.model.decoder.embed_tokens.weight"),
            "decoder.embed_tokens.weight",
        );
        // lm_head already matches.
        assert_eq!(remap("decoder.lm_head.weight"), "decoder.lm_head.weight");
    }

    #[test]
    fn ignored_keys_are_post_remap_names() {
        // Sanity: the ignore entries are exactly what the remap produces for the
        // buffers with no inference field.
        assert_eq!(
            remap("encoder.encoder.layers.0.blocks.0.attention.self.relative_position_index"),
            "encoder.stages.0.blocks.0.attention.relative_position_index",
        );
        assert!(IGNORED_KEYS
            .contains(&"encoder.stages.0.blocks.0.attention.relative_position_index"));
        assert_eq!(
            remap("encoder.embeddings.patch_embeddings.projection.norm1.0.num_batches_tracked"),
            "encoder.embeddings.projection.norm1.bn.num_batches_tracked",
        );
        assert!(IGNORED_KEYS
            .contains(&"encoder.embeddings.projection.norm1.bn.num_batches_tracked"));
    }
}
