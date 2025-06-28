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

//! RETRO encoder implementation.

use nalgebra::DMatrix;
use std::error::Error;

use super::{attention, utils};

pub struct FeedForward {
    w1: DMatrix<f32>,
    w2: DMatrix<f32>,
}

impl FeedForward {
    pub fn new(dim: u32, mult: u32) -> Self {
        let inner_dim = dim * mult;
        let normal = Normal::new(0.0, 1.0);
        let mut rng = rand::thread_rng();
        FeedForward {
            w1: DMatrix::from_fn(inner_dim as usize, dim as usize, |_, _| normal.sample(&mut rng) as f32),
            w2: DMatrix::from_fn(dim as usize, inner_dim as usize, |_, _| normal.sample(&mut rng) as f32),
        }
    }

    pub fn forward(&self, x: &DMatrix<f32>) -> Result<DMatrix<f32>, Box<dyn Error>> {
        let hidden = utils::matrix_multiply(x, &self.w1.transpose())?.map(|v| v.max(0.0)); // GELU approximation
        utils::matrix_multiply(&hidden, &self.w2.transpose())
    }
}

pub struct Encoder {
    layers: Vec<(attention::RMSNorm, attention::Attention, Option<attention::Attention>, FeedForward)>,
    rotary_pos_emb: attention::RotaryEmbedding,
    norm_out: attention::RMSNorm,
    project_out: DMatrix<f32>,
}

impl Encoder {
    pub fn new(
        dim: u32,
        context_dim: u32,
        depth: u32,
        heads: u32,
        dim_head: u32,
        cross_attn_layers: Vec<u32>,
    ) -> Self {
        let mut layers = Vec::new();
        for i in 1..=depth {
            let has_cross_attn = cross_attn_layers.contains(&i);
            layers.push((
                attention::RMSNorm::new(dim),
                attention::Attention::new(dim, dim, heads, dim_head, false),
                if has_cross_attn {
                    Some(attention::Attention::new(dim, context_dim, heads, dim_head, false))
                } else {
                    None
                },
                FeedForward::new(dim, 4),
            ));
        }
        Encoder {
            layers,
            rotary_pos_emb: attention::RotaryEmbedding::new(dim_head.min(32)),
            norm_out: attention::RMSNorm::new(dim),
            project_out: DMatrix::from_fn(context_dim as usize, dim as usize, |_, _| 0.0),
        }
    }

    pub fn forward(
        &self,
        x: &DMatrix<f32>,
        chunked_seq: &DMatrix<f32>,
    ) -> Result<DMatrix<f32>, Box<dyn Error>> {
        let chunk_size = chunked_seq.nrows();
        let seq_len = x.nrows();
        let q_pos_emb = self.rotary_pos_emb.forward(chunk_size, 0);
        let k_pos_emb = self.rotary_pos_emb.forward(seq_len, 0);

        let mut x = x.clone();
        for (norm, attn, cross_attn, ff) in &self.layers {
            x = norm.forward(&x)? + &x;
            x = attn.forward(&x, None, Some(&q_pos_emb))?;
            if let Some(cross_attn) = cross_attn {
                x = cross_attn.forward(&x, Some(chunked_seq), Some(&k_pos_emb))?;
            }
            x = ff.forward(&x)? + &x;
        }
        x = self.norm_out.forward(&x)?;
        utils::matrix_multiply(&x, &self.project_out.transpose())
    }
}