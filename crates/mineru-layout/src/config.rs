//! Static architecture configuration for PP-DocLayoutV2 / RT-DETR-L.
//!
//! These are the fixed hyperparameters of the released `opendatalab` checkpoint
//! (RT-DETR-L values plus the reading-order sub-config). They are baked in as
//! constants rather than parsed from `config.json`, because the port targets this
//! one checkpoint and hard-coding keeps the module tree statically shaped.

/// Detector hyperparameters (the RT-DETR-L / PP-DocLayoutV2 defaults).
#[derive(Debug, Clone, Copy)]
pub struct DetConfig {
    /// Transformer/embedding width (`d_model`).
    pub d_model: usize,
    /// Number of layout classes (25).
    pub num_labels: usize,
    /// Number of object queries selected from the encoder memory (300).
    pub num_queries: usize,
    /// Number of AIFI transformer encoder layers (1).
    pub encoder_layers: usize,
    /// AIFI FFN hidden dim (1024).
    pub encoder_ffn_dim: usize,
    /// AIFI attention heads (8).
    pub encoder_attention_heads: usize,
    /// Number of decoder layers (6).
    pub decoder_layers: usize,
    /// Decoder FFN hidden dim (1024).
    pub decoder_ffn_dim: usize,
    /// Decoder attention heads (8).
    pub decoder_attention_heads: usize,
    /// Deformable-attention sampling points per (head, level) (4).
    pub decoder_n_points: usize,
    /// Number of feature levels fed to the decoder (3).
    pub num_feature_levels: usize,
    /// Backbone output channels per level, low→high stride `[512, 1024, 2048]`.
    pub encoder_in_channels: [usize; 3],
    /// Feature-map strides `[8, 16, 32]`.
    pub feat_strides: [usize; 3],
    /// Sine positional-encoding temperature (10000).
    pub positional_encoding_temperature: f64,
    /// LayerNorm epsilon (1e-5).
    pub layer_norm_eps: f64,
    /// BatchNorm epsilon (1e-5), also used for the frozen backbone BN.
    pub batch_norm_eps: f64,
}

/// The single detector configuration used by this crate.
pub const DET: DetConfig = DetConfig {
    d_model: 256,
    num_labels: 25,
    num_queries: 300,
    encoder_layers: 1,
    encoder_ffn_dim: 1024,
    encoder_attention_heads: 8,
    decoder_layers: 6,
    decoder_ffn_dim: 1024,
    decoder_attention_heads: 8,
    decoder_n_points: 4,
    num_feature_levels: 3,
    encoder_in_channels: [512, 1024, 2048],
    feat_strides: [8, 16, 32],
    positional_encoding_temperature: 10000.0,
    layer_norm_eps: 1e-5,
    batch_norm_eps: 1e-5,
};

/// Reading-order head hyperparameters (`PPDocLayoutV2ReadingOrderConfig`).
#[derive(Debug, Clone, Copy)]
pub struct ReadingOrderConfig {
    /// Hidden width (512).
    pub hidden_size: usize,
    /// Attention heads (8).
    pub num_attention_heads: usize,
    /// Number of transformer layers (6).
    pub num_hidden_layers: usize,
    /// FFN hidden dim (2048).
    pub intermediate_size: usize,
    /// Token vocabulary size (4: start/pad/end/pred).
    pub vocab_size: usize,
    /// Reading-order category count for the label embedding (20).
    pub num_classes: usize,
    /// Absolute position embedding table size (514).
    pub max_position_embeddings: usize,
    /// 2D (spatial) position embedding table size (1024).
    pub max_2d_position_embeddings: usize,
    /// Per-axis coordinate embedding size (171); x and y use this.
    pub coordinate_size: usize,
    /// Per-axis shape embedding size (170); width and height use this.
    pub shape_size: usize,
    /// token-type vocabulary size (1).
    pub type_vocab_size: usize,
    /// RoPE-style relative-bias embedding dim (16).
    pub relation_bias_embed_dim: usize,
    /// RoPE base for the relative bias (10000).
    pub relation_bias_theta: f64,
    /// Scale applied to relative encodings before the sinusoid (100).
    pub relation_bias_scale: f64,
    /// GlobalPointer head size (64).
    pub global_pointer_head_size: usize,
    /// LayerNorm epsilon (1e-5).
    pub layer_norm_eps: f64,
    /// `pad_token_id` (1), also the `padding_idx` of the position embedding.
    pub pad_token_id: i64,
    /// `start_token_id` (0).
    pub start_token_id: i64,
    /// `end_token_id` (2).
    pub end_token_id: i64,
    /// `pred_token_id` (3).
    pub pred_token_id: i64,
    /// CogView softmax stabilization alpha (32).
    pub cogview_alpha: f64,
}

/// The single reading-order configuration used by this crate.
pub const READING_ORDER: ReadingOrderConfig = ReadingOrderConfig {
    hidden_size: 512,
    num_attention_heads: 8,
    num_hidden_layers: 6,
    intermediate_size: 2048,
    vocab_size: 4,
    num_classes: 20,
    max_position_embeddings: 514,
    max_2d_position_embeddings: 1024,
    coordinate_size: 171,
    shape_size: 170,
    type_vocab_size: 1,
    relation_bias_embed_dim: 16,
    relation_bias_theta: 10000.0,
    relation_bias_scale: 100.0,
    global_pointer_head_size: 64,
    layer_norm_eps: 1e-5,
    pad_token_id: 1,
    start_token_id: 0,
    end_token_id: 2,
    pred_token_id: 3,
    cogview_alpha: 32.0,
};

/// Model input side length (the square 800×800 page image).
pub const INPUT_SIZE: u32 = 800;
