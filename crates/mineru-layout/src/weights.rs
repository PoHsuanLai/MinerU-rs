//! The safetensors-key → Burn-field remap for PP-DocLayoutV2.
//!
//! The shared loader ([`mineru_burn_common::weights::load_weights`]) applies the
//! [`mineru_burn_common::weights::KeyRemap`] built here, then asserts every source
//! key was consumed ([`mineru_burn_common::weights::Coverage::Strict`]). The rules
//! rewrite the checkpoint's PyTorch module paths onto this crate's field paths.
//!
//! # Why the rules are shaped this way
//! This crate's modules store parameters in the checkpoint's own tensor layout and
//! naming (see [`crate::nn`]), so most leaf names already match. The remap only has
//! to bridge *structural* differences:
//! - the doubly-nested backbone prefix `model.backbone.model.` → `backbone.`;
//! - `encoder.stages.N` → `encoder.stageN` (named stage fields, no enum/`Vec`);
//! - `Sequential` numeric members `.0`/`.1` on the input projections and
//!   `enc_output` → the named `conv`/`bn` / `linear`/`norm` fields;
//! - the `model.` prefix on the top-level heads that this crate groups under
//!   `decoder.`;
//! - the `self_attn.` / `mlp.` sub-prefixes and the LayoutLMv3 attention naming in
//!   the reading-order head.
//!
//! Ordering matters: more specific rules run before the general prefix strips.

use mineru_burn_common::weights::{Coverage, KeyRemap};

use crate::error::{Error, Result};

/// Builds the full HF→Burn key remap for the checkpoint.
///
/// # Errors
/// Returns [`Error::Config`] if any rule's regex is invalid (should not happen).
pub fn key_remap() -> Result<KeyRemap> {
    // Each rule is (regex, replacement). Applied in order.
    let rules: &[(&str, &str)] = &[
        // ---- Backbone -------------------------------------------------------
        // Block layers are an enum in the module tree, so Burn's derive inserts the
        // variant name (`Plain`/`Light`). Insert the matching segment into source
        // keys, keyed on the (unambiguous) leaf: light layers use conv1/conv2,
        // plain layers use convolution/normalization directly.
        (
            r"^model\.backbone\.model\.encoder\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.(conv1|conv2)\.",
            "backbone.encoder.stage$1.blocks.$2.layers.$3.Light.$4.",
        ),
        (
            r"^model\.backbone\.model\.encoder\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.(convolution|normalization)\.",
            "backbone.encoder.stage$1.blocks.$2.layers.$3.Plain.$4.",
        ),
        // Named stage fields: encoder.stages.N -> encoder.stageN (aggregation,
        // downsample, and anything else in the stage keep their names).
        (r"^model\.backbone\.model\.encoder\.stages\.(\d+)\.", "backbone.encoder.stage$1."),
        // Rest of the backbone: drop the doubly-nested prefix.
        (r"^model\.backbone\.model\.", "backbone."),

        // ---- Input projections (Sequential .0/.1 -> conv/bn) ----------------
        (r"^model\.encoder_input_proj\.(\d+)\.0\.", "encoder_input_proj.$1.conv."),
        (r"^model\.encoder_input_proj\.(\d+)\.1\.", "encoder_input_proj.$1.bn."),
        (r"^model\.decoder_input_proj\.(\d+)\.0\.", "decoder.decoder_input_proj.$1.conv."),
        (r"^model\.decoder_input_proj\.(\d+)\.1\.", "decoder.decoder_input_proj.$1.bn."),

        // ---- Encoder heads / query selection (grouped under decoder) --------
        (r"^model\.enc_output\.0\.", "decoder.enc_output.linear."),
        (r"^model\.enc_output\.1\.", "decoder.enc_output.norm."),
        (r"^model\.enc_score_head\.", "decoder.enc_score_head."),
        (r"^model\.enc_bbox_head\.", "decoder.enc_bbox_head."),

        // ---- Hybrid encoder (AIFI + FPN/PAN) --------------------------------
        // aifi transformer layers: self_attn.* / mlp.* flattened onto fields.
        (r"^model\.encoder\.aifi\.(\d+)\.layers\.(\d+)\.self_attn\.", "encoder.aifi.$1.layers.$2."),
        (r"^model\.encoder\.aifi\.(\d+)\.layers\.(\d+)\.mlp\.", "encoder.aifi.$1.layers.$2."),
        (r"^model\.encoder\.", "encoder."),

        // ---- Decoder proper -------------------------------------------------
        (r"^model\.decoder\.layers\.(\d+)\.self_attn\.", "decoder.layers.$1."),
        (r"^model\.decoder\.layers\.(\d+)\.mlp\.", "decoder.layers.$1."),
        (r"^model\.decoder\.", "decoder."),

        // ---- Reading-order head ---------------------------------------------
        // LayoutLMv3-style attention naming -> this crate's flattened fields.
        (r"^reading_order\.encoder\.layer\.(\d+)\.attention\.self\.query\.", "reading_order.encoder.layer.$1.attn_query."),
        (r"^reading_order\.encoder\.layer\.(\d+)\.attention\.self\.key\.", "reading_order.encoder.layer.$1.attn_key."),
        (r"^reading_order\.encoder\.layer\.(\d+)\.attention\.self\.value\.", "reading_order.encoder.layer.$1.attn_value."),
        (r"^reading_order\.encoder\.layer\.(\d+)\.attention\.output\.dense\.", "reading_order.encoder.layer.$1.attn_out_dense."),
        (r"^reading_order\.encoder\.layer\.(\d+)\.attention\.output\.norm\.", "reading_order.encoder.layer.$1.attn_out_norm."),
        (r"^reading_order\.encoder\.layer\.(\d+)\.intermediate\.dense\.", "reading_order.encoder.layer.$1.intermediate_dense."),
        (r"^reading_order\.encoder\.layer\.(\d+)\.output\.dense\.", "reading_order.encoder.layer.$1.out_dense."),
        (r"^reading_order\.encoder\.layer\.(\d+)\.output\.norm\.", "reading_order.encoder.layer.$1.out_norm."),
        // reading_order.embeddings.*, label_embeddings, label_features_projection,
        // encoder.rel_bias_module.*, relative_head.* already match field names.
    ];

    let mut remap = KeyRemap::new();
    for (from, to) in rules {
        remap = remap
            .rename(*from, *to)
            .map_err(|e| Error::Config(format!("invalid remap rule {from:?}: {e}")))?;
    }
    Ok(remap)
}

/// Coverage policy for loading. Strict once the port is complete: every source
/// key must map to a field, or loading fails with the unmapped list.
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
    fn backbone_stage_and_prefix() {
        // Light block layers get the `Light` enum-variant segment inserted.
        assert_eq!(
            remap("model.backbone.model.encoder.stages.2.blocks.0.layers.0.conv1.convolution.weight"),
            "backbone.encoder.stage2.blocks.0.layers.0.Light.conv1.convolution.weight"
        );
        // Plain block layers get the `Plain` segment.
        assert_eq!(
            remap("model.backbone.model.encoder.stages.0.blocks.0.layers.0.convolution.weight"),
            "backbone.encoder.stage0.blocks.0.layers.0.Plain.convolution.weight"
        );
        // Aggregation / downsample keep their names, just the stage rename.
        assert_eq!(
            remap("model.backbone.model.encoder.stages.1.blocks.0.aggregation.0.convolution.weight"),
            "backbone.encoder.stage1.blocks.0.aggregation.0.convolution.weight"
        );
        assert_eq!(
            remap("model.backbone.model.encoder.stages.1.downsample.convolution.weight"),
            "backbone.encoder.stage1.downsample.convolution.weight"
        );
        assert_eq!(
            remap("model.backbone.model.embedder.stem1.convolution.weight"),
            "backbone.embedder.stem1.convolution.weight"
        );
    }

    #[test]
    fn input_proj_sequential_members() {
        assert_eq!(
            remap("model.encoder_input_proj.0.0.weight"),
            "encoder_input_proj.0.conv.weight"
        );
        assert_eq!(
            remap("model.encoder_input_proj.2.1.running_mean"),
            "encoder_input_proj.2.bn.running_mean"
        );
        assert_eq!(
            remap("model.decoder_input_proj.1.0.weight"),
            "decoder.decoder_input_proj.1.conv.weight"
        );
    }

    #[test]
    fn enc_heads_grouped_under_decoder() {
        assert_eq!(remap("model.enc_output.0.weight"), "decoder.enc_output.linear.weight");
        assert_eq!(remap("model.enc_output.1.bias"), "decoder.enc_output.norm.bias");
        assert_eq!(remap("model.enc_score_head.weight"), "decoder.enc_score_head.weight");
        assert_eq!(
            remap("model.enc_bbox_head.layers.0.weight"),
            "decoder.enc_bbox_head.layers.0.weight"
        );
    }

    #[test]
    fn encoder_aifi_and_fpn() {
        assert_eq!(
            remap("model.encoder.aifi.0.layers.0.self_attn.q_proj.weight"),
            "encoder.aifi.0.layers.0.q_proj.weight"
        );
        assert_eq!(
            remap("model.encoder.aifi.0.layers.0.mlp.fc1.weight"),
            "encoder.aifi.0.layers.0.fc1.weight"
        );
        assert_eq!(
            remap("model.encoder.fpn_blocks.0.conv1.conv.weight"),
            "encoder.fpn_blocks.0.conv1.conv.weight"
        );
        assert_eq!(
            remap("model.encoder.lateral_convs.1.norm.weight"),
            "encoder.lateral_convs.1.norm.weight"
        );
    }

    #[test]
    fn decoder_layers_and_heads() {
        assert_eq!(
            remap("model.decoder.layers.3.self_attn.q_proj.weight"),
            "decoder.layers.3.q_proj.weight"
        );
        assert_eq!(
            remap("model.decoder.layers.0.encoder_attn.sampling_offsets.weight"),
            "decoder.layers.0.encoder_attn.sampling_offsets.weight"
        );
        assert_eq!(
            remap("model.decoder.layers.5.mlp.fc2.bias"),
            "decoder.layers.5.fc2.bias"
        );
        assert_eq!(
            remap("model.decoder.class_embed.2.weight"),
            "decoder.class_embed.2.weight"
        );
        assert_eq!(
            remap("model.decoder.bbox_embed.4.layers.1.weight"),
            "decoder.bbox_embed.4.layers.1.weight"
        );
        assert_eq!(
            remap("model.decoder.query_pos_head.layers.0.weight"),
            "decoder.query_pos_head.layers.0.weight"
        );
    }

    #[test]
    fn reading_order_attention_naming() {
        assert_eq!(
            remap("reading_order.encoder.layer.0.attention.self.query.weight"),
            "reading_order.encoder.layer.0.attn_query.weight"
        );
        assert_eq!(
            remap("reading_order.encoder.layer.2.attention.output.norm.bias"),
            "reading_order.encoder.layer.2.attn_out_norm.bias"
        );
        assert_eq!(
            remap("reading_order.encoder.layer.5.output.dense.weight"),
            "reading_order.encoder.layer.5.out_dense.weight"
        );
        assert_eq!(
            remap("reading_order.embeddings.spatial_proj.weight"),
            "reading_order.embeddings.spatial_proj.weight"
        );
        assert_eq!(
            remap("reading_order.encoder.rel_bias_module.pos_proj.weight"),
            "reading_order.encoder.rel_bias_module.pos_proj.weight"
        );
    }
}
