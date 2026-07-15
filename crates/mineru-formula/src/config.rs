//! Model hyper-parameters for UniMerNet `unimernet_hf_small_2503`.
//!
//! These mirror the `config.json` of the HuggingFace checkpoint
//! `opendatalab/PDF-Extract-Kit-1.0 :: models/MFR/unimernet_hf_small_2503`
//! (a `vision-encoder-decoder`: a [`SwinConfig`] encoder + an [`MBartConfig`]
//! decoder). Values were read from that config file; see the module tests for the
//! exact numbers so a checkpoint swap that changes them is caught.
//!
//! The Python reference splits these across
//! `unimer_swin/configuration_unimer_swin.py` and
//! `unimer_mbart/configuration_unimer_mbart.py`; we keep one struct per sub-model.

/// Configuration of the Swin-Transformer vision encoder.
///
/// Note the two UniMerNet-specific deviations from vanilla HF Swin:
/// - the patch embedding is an overlapping-conv **stem** (two stride-2 3×3 convs),
///   not a single non-overlapping patch conv;
/// - each Swin layer is wrapped with two depth-wise `ConvEnhance` blocks.
#[derive(Debug, Clone)]
pub struct SwinConfig {
    /// Input image side, `[height, width]`. The processor pads to this; here both
    /// are 420 for `unimernet_hf_small_2503`.
    pub image_size: [usize; 2],
    /// Nominal patch size used only to derive the grid; the real downsampling is
    /// the stem's two stride-2 convs (total stride 4 == `patch_size`).
    pub patch_size: usize,
    /// Number of input channels the encoder expects. The processor produces a
    /// single grayscale channel which the entry point repeats to 3.
    pub num_channels: usize,
    /// Patch-embedding dimensionality (channels after the stem).
    pub embed_dim: usize,
    /// Depth (number of Swin blocks) of each of the four stages.
    pub depths: [usize; 4],
    /// Number of attention heads in each stage.
    pub num_heads: [usize; 4],
    /// Window side length for windowed attention.
    pub window_size: usize,
    /// FFN hidden-dim multiplier (`mlp_ratio * dim`).
    pub mlp_ratio: f64,
    /// Whether q/k/v projections carry a bias.
    pub qkv_bias: bool,
    /// LayerNorm epsilon.
    pub layer_norm_eps: f64,
}

impl Default for SwinConfig {
    /// The `unimernet_hf_small_2503` encoder configuration.
    fn default() -> Self {
        Self {
            image_size: [420, 420],
            patch_size: 4,
            num_channels: 3,
            embed_dim: 96,
            depths: [6, 6, 6, 6],
            num_heads: [3, 6, 12, 24],
            window_size: 5,
            mlp_ratio: 4.0,
            qkv_bias: true,
            layer_norm_eps: 1e-5,
        }
    }
}

impl SwinConfig {
    /// Number of stages (== `depths.len()`).
    pub const NUM_STAGES: usize = 4;

    /// Channel dimension after stage `i` (`embed_dim * 2^i`).
    pub fn stage_dim(&self, stage: usize) -> usize {
        self.embed_dim * (1 << stage)
    }

    /// Final encoder hidden size (channels after the last stage), which is also the
    /// dimension the decoder cross-attends over.
    pub fn hidden_size(&self) -> usize {
        self.stage_dim(Self::NUM_STAGES - 1)
    }
}

/// Configuration of the MBart autoregressive text decoder.
///
/// The one UniMerNet-specific deviation from vanilla MBart is *squeeze attention*:
/// query and key are projected to `d_model / qk_squeeze` (values stay full-width),
/// which shrinks the QK matmul. See the paper (arXiv:2404.15254).
#[derive(Debug, Clone)]
pub struct MBartConfig {
    /// Vocabulary size (== tokenizer length).
    pub vocab_size: usize,
    /// Model / residual-stream width.
    pub d_model: usize,
    /// Squeeze ratio for the query/key projection width.
    pub qk_squeeze: usize,
    /// Number of decoder layers.
    pub decoder_layers: usize,
    /// Number of decoder attention heads (self- and cross-).
    pub decoder_attention_heads: usize,
    /// FFN inner dimension.
    pub decoder_ffn_dim: usize,
    /// Maximum position for the learned positional embedding.
    pub max_position_embeddings: usize,
    /// Whether token embeddings are scaled by `sqrt(d_model)`.
    pub scale_embedding: bool,
    /// LayerNorm epsilon.
    pub layer_norm_eps: f64,
    /// Padding token id.
    pub pad_token_id: usize,
    /// Beginning-of-sequence / decoder-start token id.
    pub bos_token_id: usize,
    /// End-of-sequence token id (generation stops here).
    pub eos_token_id: usize,
    /// The token forced at `max_length` (equals `eos_token_id` here).
    pub forced_eos_token_id: usize,
}

impl Default for MBartConfig {
    /// The `unimernet_hf_small_2503` decoder configuration.
    fn default() -> Self {
        Self {
            vocab_size: 50000,
            d_model: 768,
            qk_squeeze: 2,
            decoder_layers: 8,
            decoder_attention_heads: 16,
            decoder_ffn_dim: 3072,
            max_position_embeddings: 1536,
            scale_embedding: true,
            layer_norm_eps: 1e-5,
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            forced_eos_token_id: 2,
        }
    }
}

impl MBartConfig {
    /// The learned positional embedding is offset by 2 (MBart's padding hack): the
    /// embedding table has `max_position_embeddings + OFFSET` rows and position `p`
    /// indexes row `p + OFFSET`.
    pub const POSITION_OFFSET: usize = 2;

    /// Squeezed q/k projection width (`d_model / qk_squeeze`).
    pub fn squeeze_dim(&self) -> usize {
        self.d_model / self.qk_squeeze
    }

    /// Per-head dimension of the value / output stream (`d_model / heads`).
    pub fn head_dim(&self) -> usize {
        self.d_model / self.decoder_attention_heads
    }

    /// Per-head dimension of the squeezed q/k stream.
    pub fn squeeze_head_dim(&self) -> usize {
        self.squeeze_dim() / self.decoder_attention_heads
    }
}

/// Number of crops decoded together by the batched entry point.
///
/// Matches the Python reference's `batch_size=16`. Every lane in a batch runs until
/// the *longest* one finishes, so oversized batches waste work on short formulas;
/// 16 is the reference's balance point.
const DEFAULT_BATCH_SIZE: usize = 16;

/// Full model configuration: the encoder + decoder pair plus the decode budget.
#[derive(Debug, Clone)]
pub struct UniMerNetConfig {
    /// Swin encoder config.
    pub encoder: SwinConfig,
    /// MBart decoder config.
    pub decoder: MBartConfig,
    /// Maximum number of tokens to generate before forcing a stop. The Python
    /// entry point caps this at 1152/1344 depending on batch size; we use a single
    /// conservative bound here.
    pub max_new_tokens: usize,
    /// How many crops [`crate::FormulaRecognizer::predict_batch`] decodes per batch.
    /// Defaults to [`DEFAULT_BATCH_SIZE`] (16, matching the Python reference).
    pub batch_size: usize,
}

impl Default for UniMerNetConfig {
    fn default() -> Self {
        Self {
            encoder: SwinConfig::default(),
            decoder: MBartConfig::default(),
            max_new_tokens: 0,
            batch_size: DEFAULT_BATCH_SIZE,
        }
    }
}

impl UniMerNetConfig {
    /// Builds the default `unimernet_hf_small_2503` configuration with a 1152-token
    /// generation budget (the small-batch cap in the Python reference).
    pub fn small_2503() -> Self {
        Self {
            max_new_tokens: 1152,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_dims_match_checkpoint() {
        let c = SwinConfig::default();
        assert_eq!(c.embed_dim, 96);
        assert_eq!(c.depths, [6, 6, 6, 6]);
        assert_eq!(c.num_heads, [3, 6, 12, 24]);
        assert_eq!(c.window_size, 5);
        // 96 * 2^3 == 768: encoder output width must equal decoder d_model so
        // cross-attention lines up without a projection.
        assert_eq!(c.hidden_size(), 768);
        assert_eq!(c.hidden_size(), MBartConfig::default().d_model);
    }

    #[test]
    fn decoder_squeeze_dims_are_consistent() {
        let c = MBartConfig::default();
        assert_eq!(c.squeeze_dim(), 384); // 768 / 2
        assert_eq!(c.head_dim(), 48); // 768 / 16
        assert_eq!(c.squeeze_head_dim(), 24); // 384 / 16
        assert_eq!(c.squeeze_head_dim() * c.decoder_attention_heads, c.squeeze_dim());
    }

    #[test]
    fn stage_dims_double_each_stage() {
        let c = SwinConfig::default();
        assert_eq!(c.stage_dim(0), 96);
        assert_eq!(c.stage_dim(1), 192);
        assert_eq!(c.stage_dim(2), 384);
        assert_eq!(c.stage_dim(3), 768);
    }
}
