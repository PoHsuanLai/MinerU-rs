//! Checkpoint-key → Burn-field remapping for weight loading.
//!
//! The `unimernet_hf_small_2503` checkpoint is a HuggingFace
//! `VisionEncoderDecoderModel` safetensors file. Its tensor keys are prefixed:
//!
//! - `encoder.…` for the Swin encoder (`UnimerSwinModel`);
//! - `decoder.…` for the MBart decoder (`UnimerMBartForCausalLM`).
//!
//! Burn derives field paths from the Rust struct field names. This module builds a
//! [`KeyRemap`] (from the shared harness) that rewrites checkpoint keys onto our
//! field paths, and documents every rule so a mismatch is auditable.
//!
//! # HONESTY / UNCERTAINTY (read before trusting a real load)
//! Real weights were **not** downloaded during this port (the checkpoint is a
//! multi-hundred-MB file, gated behind `#[ignore]`d tests). The rules below are
//! derived from the Python module structure, not verified against the actual
//! tensor names in the file. The following are the **highest-risk** points where
//! the load may leave keys unmapped (surfaced by [`mineru_burn_common::Coverage::Strict`]):
//!
//! 1. **Encoder prefix.** HF Swin nests as `encoder.embeddings.*`,
//!    `encoder.encoder.layers.<i>.blocks.<j>.*`, `encoder.encoder.layers.<i>.downsample.*`.
//!    Our fields are flatter (`embeddings`, `stages.<i>.blocks.<j>`,
//!    `stages.<i>.downsample`). The rules rewrite `encoder.encoder.layers` →
//!    `stages` and drop the doubled prefix. **If the file uses a single `encoder.`
//!    (some exports do), rule 1a must change.**
//! 2. **Stem/BatchNorm.** `encoder.embeddings.patch_embeddings.projection.{conv1,norm1,conv2}`
//!    → `embeddings.projection.{conv1,norm1,conv2}`. BatchNorm running stats
//!    (`running_mean`/`running_var`) must map onto Burn's `RunningState`; this is
//!    the field most likely to mismatch and is called out in the tests.
//! 3. **Attention sub-naming.** HF has `attention.self.{query,key,value}` and
//!    `attention.output.dense`; we flattened to `attention.{query,key,value}` and
//!    `attention.output`. Rules 3a–3b handle this.
//! 4. **Decoder wrapper depth.** `UnimerMBartForCausalLM` stores the decoder at
//!    `model.decoder.*` and the head at `lm_head.*`; under the VED prefix that is
//!    `decoder.model.decoder.*` / `decoder.lm_head.*`. Rules 4a–4b strip to our
//!    `decoder.*` / `lm_head` — but our top-level module nests them under one
//!    struct, so the exact target depends on [`crate::model`]'s field layout.
//!
//! When a real load is attempted, run it under `Coverage::Lenient` first, read the
//! reported unmapped keys, and tighten these rules until `Coverage::Strict` passes.

use mineru_burn_common::weights::KeyRemap;

use crate::error::Result;

/// Builds the checkpoint-key remapper for the full UniMerNet model.
///
/// The rules assume the top-level Burn module exposes the encoder under field
/// `encoder` and the decoder under field `decoder` (see [`crate::model::UniMerNet`]).
///
/// # Errors
/// Returns [`crate::Error`] if any remap regex is invalid (they are literals here,
/// so this is effectively infallible but kept in `Result` for the harness API).
pub fn build_remap() -> Result<KeyRemap> {
    let remap = KeyRemap::new()
        // ---- Encoder (Swin) ----
        // HF: encoder.embeddings.patch_embeddings.projection.* -> our encoder.embeddings.projection.*
        .rename(
            r"^encoder\.embeddings\.patch_embeddings\.projection\.",
            "encoder.embeddings.projection.",
        )?
        // HF: encoder.embeddings.norm.* -> encoder.embeddings.norm.*  (identity, but strip nothing)
        // HF stages live at encoder.encoder.layers.<i>...; ours at encoder.stages.<i>...
        .rename(r"^encoder\.encoder\.layers\.", "encoder.stages.")?
        // attention.self.{q,k,v} -> attention.{q,k,v}; attention.output.dense -> attention.output
        .rename(r"\.attention\.self\.", ".attention.")?
        .rename(r"\.attention\.output\.dense\b", ".attention.output")?
        // ---- Decoder (MBart, wrapped) ----
        // decoder.model.decoder.* -> decoder.*   (strip the ForCausalLM + Wrapper nesting)
        .rename(r"^decoder\.model\.decoder\.", "decoder.")?
        // decoder.lm_head.* -> lm_head.* lives on the decoder struct in our layout:
        .rename(r"^decoder\.lm_head\.", "decoder.lm_head.")?;
    Ok(remap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_stem_key_is_rewritten() {
        let remap = build_remap().expect("remap builds");
        assert_eq!(
            remap
                .apply_str("encoder.embeddings.patch_embeddings.projection.conv1.weight")
                .as_deref(),
            Some("encoder.embeddings.projection.conv1.weight"),
        );
    }

    #[test]
    fn encoder_stage_and_attention_keys_are_rewritten() {
        let remap = build_remap().expect("remap builds");
        // layers.0.blocks.0.attention.self.query.weight -> stages.0.blocks.0.attention.query.weight
        let got = remap
            .apply_str("encoder.encoder.layers.0.blocks.0.attention.self.query.weight")
            .expect("rule matched");
        assert_eq!(
            got,
            "encoder.stages.0.blocks.0.attention.query.weight"
        );

        let got2 = remap
            .apply_str("encoder.encoder.layers.2.blocks.1.attention.output.dense.weight")
            .expect("rule matched");
        assert_eq!(
            got2,
            "encoder.stages.2.blocks.1.attention.output.weight"
        );
    }

    #[test]
    fn decoder_wrapper_prefix_is_stripped() {
        let remap = build_remap().expect("remap builds");
        assert_eq!(
            remap
                .apply_str("decoder.model.decoder.layers.0.self_attn.q_proj.weight")
                .as_deref(),
            Some("decoder.layers.0.self_attn.q_proj.weight"),
        );
    }
}
