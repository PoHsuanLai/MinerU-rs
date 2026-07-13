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
        // The block layers are a generic type parameter (not an enum), so their
        // parameter paths already match the checkpoint leaf-for-leaf (a plain
        // stage's layers are `layers.N.convolution.…`, a light stage's are
        // `layers.N.conv1.…` / `layers.N.conv2.…`). The only structural change is
        // the stage rename and the doubly-nested prefix strip.
        //
        // Named stage fields: encoder.stages.N -> encoder.stageN.
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
        // The AIFI transformer lives at `model.encoder.encoder.0.layers.N.*` in the
        // checkpoint (the outer `encoder` is the hybrid encoder, the inner
        // `encoder.0` is its single AIFI stack). This crate stores it as the named
        // `aifi` field, so `encoder.0` -> `aifi`. Within a layer, the attention
        // projections sit under `self_attn.` (q/k/v/out) and the FFN leaves
        // (`fc1`/`fc2`/`self_attn_layer_norm`/`final_layer_norm`) sit directly on
        // the layer. Route `out_proj` -> `o_proj` (this crate's field name), keep
        // q/k/v as-is, and strip the `self_attn.` prefix; the FFN/norm leaves are
        // matched by the trailing `encoder.0` -> `aifi` rewrite.
        (r"^model\.encoder\.encoder\.0\.layers\.(\d+)\.self_attn\.out_proj\.", "encoder.aifi.layers.$1.o_proj."),
        (r"^model\.encoder\.encoder\.0\.layers\.(\d+)\.self_attn\.", "encoder.aifi.layers.$1."),
        (r"^model\.encoder\.encoder\.0\.layers\.(\d+)\.", "encoder.aifi.layers.$1."),
        (r"^model\.encoder\.", "encoder."),

        // ---- Decoder proper -------------------------------------------------
        // Self-attention output projection: checkpoint `self_attn.out_proj` -> this
        // crate's `o_proj` field. Must precede the general `self_attn.` strip below
        // (which keeps q/k/v as-is), or `out_proj` would land on a nonexistent field.
        (r"^model\.decoder\.layers\.(\d+)\.self_attn\.out_proj\.", "decoder.layers.$1.o_proj."),
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

/// Checkpoint keys that inference intentionally does not load.
///
/// `model.denoising_class_embed.weight` is the RT-DETR contrastive-denoising (CDN)
/// class embedding, used only to build noised query groups during training. At
/// inference the denoising branch is disabled, so this crate has no field for it
/// and it is skipped under [`Coverage::Strict`]. Entries are post-remap names; no
/// remap rule touches this key, so it appears unchanged. See the load path in
/// [`crate::LayoutModel::load`].
pub const IGNORED_KEYS: &[&str] = &["model.denoising_class_embed.weight"];

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
        // Light block layers (conv1/conv2) keep their leaf names; only the stage
        // is renamed. No enum-variant segment is inserted.
        assert_eq!(
            remap("model.backbone.model.encoder.stages.2.blocks.0.layers.0.conv1.convolution.weight"),
            "backbone.encoder.stage2.blocks.0.layers.0.conv1.convolution.weight"
        );
        // Plain block layers (convolution/normalization) likewise.
        assert_eq!(
            remap("model.backbone.model.encoder.stages.0.blocks.0.layers.0.convolution.weight"),
            "backbone.encoder.stage0.blocks.0.layers.0.convolution.weight"
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
        // AIFI lives at `model.encoder.encoder.0.layers.N.*`; `encoder.0` -> `aifi`.
        assert_eq!(
            remap("model.encoder.encoder.0.layers.0.self_attn.q_proj.weight"),
            "encoder.aifi.layers.0.q_proj.weight"
        );
        // out_proj -> o_proj (this crate's field name).
        assert_eq!(
            remap("model.encoder.encoder.0.layers.0.self_attn.out_proj.bias"),
            "encoder.aifi.layers.0.o_proj.bias"
        );
        // FFN + norm leaves sit directly on the layer (no `mlp.` prefix).
        assert_eq!(
            remap("model.encoder.encoder.0.layers.0.fc1.weight"),
            "encoder.aifi.layers.0.fc1.weight"
        );
        assert_eq!(
            remap("model.encoder.encoder.0.layers.0.self_attn_layer_norm.weight"),
            "encoder.aifi.layers.0.self_attn_layer_norm.weight"
        );
        assert_eq!(
            remap("model.encoder.encoder.0.layers.0.final_layer_norm.bias"),
            "encoder.aifi.layers.0.final_layer_norm.bias"
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
        // Self-attn output projection: out_proj -> o_proj.
        assert_eq!(
            remap("model.decoder.layers.3.self_attn.out_proj.weight"),
            "decoder.layers.3.o_proj.weight"
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
