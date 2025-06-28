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

//! Attention mechanisms and rotary embeddings for RETRO.

use nalgebra::{DMatrix, DVector};
use std::error::Error;

use super::utils;

pub struct RotaryEmbedding {
    inv_freq: DVector<f32>,
}

impl RotaryEmbedding {
    pub fn new(dim: u32) -> Self {
        let inv_freq = DVector::from_fn(dim as usize / 2, |i, _| {
            1.0 / 10000.0_f32.powf((i * 2) as f32 / dim as f32)
        });
        RotaryEmbedding { inv_freq }
    }

    pub fn forward(&self, max_seq_len: usize, offset: usize) -> DMatrix<f32> {
        let seq = DVector::from_fn(max_seq_len, |i, _| (i + offset) as f32);
        let freqs = seq * &self.inv_freq;
        let mut emb = DMatrix::zeros(max_seq_len, self.inv_freq.len() * 2);
        for i in 0..max_seq_len {
            for j in 0..self.inv_freq.len() {
                let angle = freqs[(i, j)];
                emb[(i, j * 2)] = angle.cos();
                emb[(i, j * 2 + 1)] = angle.sin();
            }
        }
        emb
    }
}

pub struct RMSNorm {
    gamma: DVector<f32>,
    eps: f32,
}

impl RMSNorm {
    pub fn new(dim: u32) -> Self {
        RMSNorm {
            gamma: DVector::from_element(dim as usize, 1.0),
            eps: 1e-8,
        }
    }

    pub fn forward(&self, x: &DMatrix<f32>) -> Result<DMatrix<f32>, Box<dyn Error>> {
        let scale = (x.ncols() as f32).sqrt();
        let norm = x.map(|v| v * v).row_sum().map(|v| (v / x.ncols() as f32).sqrt() * scale);
        let norm_clamped = norm.map(|v| v.max(self.eps));
        Ok(x.component_div(&norm_clamped) * &self.gamma)
    }
}

pub struct Attention {
    heads: u32,
    dim_head: u32,
    scale: f32,
    causal: bool,
    to_q: DMatrix<f32>,
    to_k: DMatrix<f32>,
    to_v: DMatrix<f32>,
    to_out: DMatrix<f32>,
}

impl Attention {
    pub fn new(dim: u32, context_dim: u32, heads: u32, dim_head: u32, causal: bool) -> Self {
        let inner_dim = heads * dim_head;
        let normal = Normal::new(0.0, 1.0);
        let mut rng = rand::thread_rng();
        Attention {
            heads,
            dim_head,
            scale: (dim_head as f32).powf(-0.5),
            causal,
            to_q: DMatrix::from_fn(inner_dim as usize, dim as usize, |_, _| normal.sample(&mut rng) as f32),
            to_k: DMatrix::from_fn(inner_dim as usize, context_dim as usize, |_, _| normal.sample(&mut rng) as f32),
            to_v: DMatrix::from_fn(inner_dim as usize, context_dim as usize, |_, _| normal.sample(&mut rng) as f32),
            to_out: DMatrix::from_fn(dim as usize, inner_dim as usize, |_, _| normal.sample(&mut rng) as f32),
        }
    }

    pub fn forward(
        &self,
        x: &DMatrix<f32>,
        context: Option<&DMatrix<f32>>,
        pos_emb: Option<&DMatrix<f32>>,
    ) -> Result<DMatrix<f32>, Box<dyn Error>> {
        let kv_input = context.unwrap_or(x);
        let inner_dim = self.heads * self.dim_head;

        let q = utils::matrix_multiply(x, &self.to_q.transpose())? * self.scale;
        let k = utils::matrix_multiply(kv_input, &self.to_k.transpose())?;
        let v = utils::matrix_multiply(kv_input, &self.to_v.transpose())?;

        let q = q.reshape((q.nrows(), self.heads as usize, self.dim_head as usize));
        let k = k.reshape((k.nrows(), self.heads as usize, self.dim_head as usize));
        let v = v.reshape((v.nrows(), self.heads as usize, self.dim_head as usize));

        let sim = utils::matrix_multiply(&q, &k.transpose())?;

        if self.causal {
            let mask = DMatrix::from_fn(sim.nrows(), sim.ncols(), |i, j| if j > i { f32::NEG_INFINITY } else { 0.0 });
            sim += &mask;
        }

        let attn = utils::softmax(&sim);
        let out = utils::matrix_multiply(&attn, &v)?;
        let out = out.reshape((out.nrows(), inner_dim as usize));
        utils::matrix_multiply(&out, &self.to_out.transpose())
    }
}