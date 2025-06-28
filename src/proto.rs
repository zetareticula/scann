// ... (previous proto.rs content unchanged)

#[derive(Clone, PartialEq)]
pub struct RetroConfig {
    pub num_tokens: u32,
    pub max_seq_len: u32,
    pub enc_dim: u32,
    pub dec_dim: u32,
    pub enc_depth: u32,
    pub dec_depth: u32,
    pub heads: u32,
    pub dim_head: u32,
    pub chunk_size: u32,
    pub enc_cross_attn_layers: Vec<u32>,
    pub dec_cross_attn_layers: Vec<u32>,
    pub enc_attn_dropout: f32,
    pub enc_ff_dropout: f32,
    pub dec_attn_dropout: f32,
    pub dec_ff_dropout: f32,
    pub pad_id: u32,
    pub use_deepnet: bool,
    pub gated_rmsnorm: bool,
}

impl RetroConfig {
    pub fn new() -> Self {
        RetroConfig {
            num_tokens: 50257,
            max_seq_len: 2048,
            enc_dim: 896,
            dec_dim: 768,
            enc_depth: 2,
            dec_depth: 12,
            heads: 8,
            dim_head: 64,
            chunk_size: 64,
            enc_cross_attn_layers: vec![],
            dec_cross_attn_layers: vec![1, 3, 6, 9],
            enc_attn_dropout: 0.0,
            enc_ff_dropout: 0.0,
            dec_attn_dropout: 0.0,
            dec_ff_dropout: 0.0,
            pad_id: 0,
            use_deepnet: false,
            gated_rmsnorm: false,
        }
    }
}