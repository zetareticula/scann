// Copyright 2025 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! RETRO decoder implementation.

use nalgebra::DMatrix;
use std::error::Error;

use super::{attention, encoder, utils};

pub struct ChunkedCrossAttention {
    chunk_size: u32,
    cross_attn: attention::Attention,
}

impl ChunkedCrossAttention {
    pub fn new(chunk_size: u32, dim: u32, heads: u32, dim_head: u32) -> Self {
        ChunkedCrossAttention {
            chunk_size,
            cross_attn: attention::Attention::new(dim, dim, heads, dim_head, false),
        }
    }

    pub fn forward(
        &self,
        x: &DMatrix<f32>,
        context: &DMatrix<f32>,
        pos_emb: (&DMatrix<f32>, &DMatrix<f32>),
    ) -> Result<DMatrix<f32>, Box<dyn Error>> {
        let chunk_size = self.chunk_size as usize;
        let (q_pos_emb, k_pos_emb) = pos_emb;
        if x.nrows() < chunk_size {
            return Ok(DMatrix::zeros(x.nrows(), x.ncols()));
        }

        let causal_padding = chunk_size - 1;
        let mut x_padded = DMatrix::zeros(x.nrows() + causal_padding, x.ncols());
        x_padded.rows_mut(0, x.nrows()).copy_from(x);

        let num_chunks = x.nrows() / chunk_size;
        let seq_index = num_chunks * chunk_size;
        let x = x_padded.rows(0, seq_index).into_owned();

        let x = x.reshape((num_chunks, chunk_size, x.ncols()));
        let context = context.reshape((num_chunks, context.nrows() / num_chunks, context.ncols()));
        let out = self.cross_attn.forward(&x, Some(&context), Some(&q_pos_emb))?;
        let out = out.reshape((seq_index, x.ncols()));
        Ok(out)
    }
}

pub struct Decoder {
    layers: Vec<(attention::RMSNorm, attention::Attention, Option<ChunkedCrossAttention>, encoder::FeedForward)>,
    rotary_pos_emb: attention::RotaryEmbedding,
    norm_out: attention::RMSNorm,
    chunk_size: u32,
}

impl Decoder {
    pub fn new(dim: u32, depth: u32, heads: u32, dim_head: u32, chunk_size: u32, cross_attn_layers: Vec<u32>) -> Self {
        let mut layers = Vec::new();
        for i in 1..=depth {
            let has_cross_attn = cross_attn_layers.contains(&i);
            layers.push((
                attention::RMSNorm::new(dim),
                attention::Attention::new(dim, dim, heads, dim_head, true),
                if has_cross_attn {
                    Some(ChunkedCrossAttention::new(chunk_size, dim, heads, dim_head))
                } else {
                    None
                },
                encoder::FeedForward::new(dim, 4),
            ));
        }
        Decoder {
            layers,
            rotary_pos_emb: attention::RotaryEmbedding::new(dim_head.min(32)),
            norm_out: attention::RMSNorm::new(dim),
            chunk_size,
        }
    }

    pub fn forward(
        &self,
        x: &DMatrix<f32>,
        encoder: &encoder::Encoder,
        retrieved: Option<&DMatrix<f32>>,
    ) -> Result<DMatrix<f32>, Box<dyn Error>> {
        let seq_len = x.nrows();
        let self_attn_pos_emb = self.rotary_pos_emb.forward(seq_len, 0);
        let mut x = x.clone();
        let mut retrieved_encoded = None;

        for (norm, attn, cross_attn, ff) in &self.layers {
            x = norm.forward(&x)? + &x;
            x = attn.forward(&x, None, Some(&self_attn_pos_emb))?;
            if let (Some(cross_attn), Some(retrieved)) = (cross_attn, retrieved) {
                if retrieved_encoded.is_none() {
                    let num_chunks = seq_len / self.chunk_size as usize;
                    let seq_index = num_chunks * self.chunk_size as usize;
                    let seq_as_context = x.rows(0, seq_index).into_owned();
                    let retrieved_encoded_res = encoder.forward(retrieved, &seq_as_context)?;
                    retrieved_encoded = Some(retrieved_encoded_res);
                }
                let cross_attn_pos_emb = (
                    self.rotary_pos_emb.forward(self.chunk_size as usize, self.chunk_size as usize - 1),
                    self.rotary_pos_emb.forward(self.chunk_size as usize, 0),
                );
                x = cross_attn.forward(&x, retrieved_encoded.as_ref().unwrap(), &cross_attn_pos_emb)? + &x;
            }
            x = ff.forward(&x)? + &x;
        }
        self.norm_out.forward(&x)
    }
}