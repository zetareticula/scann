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

//! RETRO model main class.

use nalgebra::DMatrix;
use std::error::Error;

use super::{decoder, embeddings, encoder, utils};
use crate::proto::RetroConfig;
use crate::retrieval::ScannRetriever;

pub struct RETRO {
    token_emb: embeddings::TokenEmbedding,
    pos_emb: embeddings::PositionalEmbedding,
    to_decoder_model_dim: DMatrix<f32>,
    encoder: encoder::Encoder,
    decoder: decoder::Decoder,
    to_logits: DMatrix<f32>,
    seq_len: u32,
    chunk_size: u32,
    pad_id: u32,
    retriever: Option<ScannRetriever>,
}

impl RETRO {
    pub fn new(config: RetroConfig, retriever: Option<ScannRetriever>) -> Self {
        let to_decoder_model_dim = if config.enc_dim != config.dec_dim {
            DMatrix::from_fn(
                config.dec_dim as usize,
                config.enc_dim as usize,
                |_, _| rand::random::<f32>(),
            )
        } else {
            DMatrix::identity(config.enc_dim as usize, config.enc_dim as usize)
        };
        RETRO {
            token_emb: embeddings::TokenEmbedding::new(config.num_tokens, config.enc_dim),
            pos_emb: embeddings::PositionalEmbedding::new(config.max_seq_len, config.enc_dim),
            to_decoder_model_dim,
            encoder: encoder::Encoder::new(
                config.enc_dim,
                config.dec_dim,
                config.enc_depth,
                config.heads,
                config.dim_head,
                config.enc_cross_attn_layers,
            ),
            decoder: decoder::Decoder::new(
                config.dec_dim,
                config.dec_depth,
                config.heads,
                config.dim_head,
                config.chunk_size,
                config.dec_cross_attn_layers,
            ),
            to_logits: DMatrix::from_fn(
                config.num_tokens as usize,
                config.dec_dim as usize,
                |_, _| rand::random::<f32>(),
            ),
            seq_len: config.max_seq_len,
            chunk_size: config.chunk_size,
            pad_id: config.pad_id,
            retriever,
        }
    }

    pub fn forward_without_retrieval(&self, seq: &[u32]) -> Result<DMatrix<f32>, Box<dyn Error>> {
        let embed = self.token_emb.forward(seq)?;
        let pos_emb = self.pos_emb.forward(embed.nrows())?;
        let embed = embed + pos_emb;
        let embed = utils::matrix_multiply(&embed, &self.to_decoder_model_dim.transpose())?;
        let decoded = self.decoder.forward(&embed, &self.encoder, None)?;
        utils::matrix_multiply(&decoded, &self.to_logits.transpose())
    }

    pub fn forward(&self, seq: &[u32], retrieved: Option<&DMatrix<f32>>) -> Result<DMatrix<f32>, Box<dyn Error>> {
        if retrieved.is_none() && self.retriever.is_none() {
            return self.forward_without_retrieval(seq);
        }

        let embed = self.token_emb.forward(seq)?;
        let pos_emb = self.pos_emb.forward(embed.nrows())?;
        let embed = embed + pos_emb;

        let retrieved = if let Some(retrieved) = retrieved {
            retrieved.clone()
        } else if let Some(retriever) = &self.retriever {
            let chunks = retriever.retrieve_chunks(seq, self.chunk_size as usize)?;
            let mut retrieved_data = Vec::new();
            for chunk in chunks {
                let mut chunk_data = Vec::new();
                for neighbor in chunk {
                    chunk_data.push(self.token_emb.forward(&neighbor)?);
                }
                retrieved_data.push(chunk_data);
            }
            DMatrix::from_fn(
                retrieved_data.len(),
                retrieved_data[0].len() * retrieved_data[0][0].ncols(),
                |i, j| retrieved_data[i][j / retrieved_data[0][0].ncols()][0][j % retrieved_data[0][0].ncols()],
            )
        } else {
            return Err(utils::invalid_argument_error("No retrieved data or retriever provided"));
        };

        let embed = utils::matrix_multiply(&embed, &self.to_decoder_model_dim.transpose())?;
        let decoded = self.decoder.forward(&embed, &self.encoder, Some(&retrieved))?;
        utils::matrix_multiply(&decoded, &self.to_logits.transpose())
    }
}