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

//! Token and positional embeddings for RETRO.

use nalgebra::{DMatrix, DVector};
use rand::distributions::{Distribution, Normal};
use std::error::Error;

pub struct TokenEmbedding {
    weights: DMatrix<f32>,
}

impl TokenEmbedding {
    pub fn new(num_tokens: u32, dim: u32) -> Self {
        let normal = Normal::new(0.0, 1.0);
        let mut rng = rand::thread_rng();
        let weights = DMatrix::from_fn(num_tokens as usize, dim as usize, |_, _| {
            normal.sample(&mut rng) as f32
        });
        TokenEmbedding { weights }
    }

    pub fn forward(&self, tokens: &[u32]) -> Result<DMatrix<f32>, Box<dyn Error>> {
        let mut result = DMatrix::zeros(tokens.len(), self.weights.ncols());
        for (i, &token) in tokens.iter().enumerate() {
            if token as usize >= self.weights.nrows() {
                return Err(super::utils::invalid_argument_error(&format!(
                    "Token ID {} exceeds vocabulary size {}",
                    token, self.weights.nrows()
                )));
            }
            result.row_mut(i).copy_from(&self.weights.row(token as usize));
        }
        Ok(result)
    }
}

pub struct PositionalEmbedding {
    weights: DMatrix<f32>,
}

impl PositionalEmbedding {
    pub fn new(max_seq_len: u32, dim: u32) -> Self {
        let normal = Normal::new(0.0, 1.0);
        let mut rng = rand::thread_rng();
        let weights = DMatrix::from_fn(max_seq_len as usize, dim as usize, |_, _| {
            normal.sample(&mut rng) as f32
        });
        PositionalEmbedding { weights }
    }

    pub fn forward(&self, seq_len: usize) -> Result<DMatrix<f32>, Box<dyn Error>> {
        if seq_len > self.weights.nrows() {
            return Err(super::utils::invalid_argument_error(&format!(
                "Sequence length {} exceeds max sequence length {}",
                seq_len, self.weights.nrows()
            )));
        }
        Ok(self.weights.rows(0, seq_len).into_owned())
    }
}