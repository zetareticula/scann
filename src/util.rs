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

//! Shared utility types for the ScaNN library.

use nalgebra::{DMatrix, DVector};
use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub struct ScannError {
    pub message: String,
}

impl fmt::Display for ScannError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for ScannError {}

pub fn invalid_argument_error(msg: &str) -> Box<dyn Error> {
    Box::new(ScannError {
        message: msg.to_string(),
    })
}

pub fn failed_precondition_error(msg: &str) -> Box<dyn Error> {
    Box::new(ScannError {
        message: msg.to_string(),
    })
}

#[derive(Clone)]
pub struct DenseDataset<T> {
    pub data: Vec<Vec<T>>,
    pub dimensionality: usize,
}

impl<T: Clone> DenseDataset<T> {
    pub fn new(data: Vec<Vec<T>>, dimensionality: usize) -> Self {
        DenseDataset { data, dimensionality }
    }

    pub fn set_dimensionality(&mut self, dim: usize) {
        self.dimensionality = dim;
    }

    pub fn reserve(&mut self, size: usize) {
        self.data.reserve(size);
    }

    pub fn append(&mut self, values: &[T], _docid: &str) -> Result<(), Box<dyn Error>> {
        if values.len() != self.dimensionality {
            return Err(invalid_argument_error(&format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimensionality,
                values.len()
            )));
        }
        self.data.push(values.to_vec());
        Ok(())
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn dimensionality(&self) -> usize {
        self.dimensionality
    }
}

#[derive(Clone)]
pub struct DatapointPtr<T> {
    values: Vec<T>,
}

impl<T: Clone> DatapointPtr<T> {
    pub fn new(values: Vec<T>) -> Self {
        DatapointPtr { values }
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }
}

// New: Matrix utilities for RETRO
pub fn dot_product<T: Copy + Into<f32>>(a: &DatapointPtr<T>, b: &DatapointPtr<T>) -> f32 {
    a.values()
        .iter()
        .zip(b.values().iter())
        .map(|(&x, &y)| x.into() * y.into())
        .sum()
}

pub fn matrix_multiply(a: &DMatrix<f32>, b: &DMatrix<f32>) -> Result<DMatrix<f32>, Box<dyn Error>> {
    if a.ncols() != b.nrows() {
        return Err(invalid_argument_error(&format!(
            "Matrix dimension mismatch: {}x{} cannot multiply with {}x{}",
            a.nrows(), a.ncols(), b.nrows(), b.ncols()
        )));
    }
    Ok(a * b)
}

pub fn softmax(x: &DMatrix<f32>) -> DMatrix<f32> {
    let exp_x = x.map(|v| v.exp());
    let sum_exp_x = exp_x.sum();
    exp_x / sum_exp_x
}