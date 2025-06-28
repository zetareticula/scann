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

//! PCA projection implementation for dimensionality reduction.

use super::{failed_precondition_error, invalid_argument_error, proto, utils, ScannError};
use std::error::Error;
use std::sync::Arc;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

// Placeholder for PCA utilities
mod pca_utils {
    use super::*;

    pub fn compute_pca(
        _center: bool,
        _data: &utils::DenseDataset<f32>,
        projected_dims: usize,
        _build_covariance: bool,
        pca_vecs: &mut Vec<utils::DatapointPtr<f32>>,
        eigen_vals: &mut Vec<f32>,
        _parallelization_pool: Option<&ParallelizationPool>,
    ) {
        pca_vecs.clear();
        eigen_vals.clear();
        for _ in 0..projected_dims {
            pca_vecs.push(utils::DatapointPtr::new(vec![0.0; _data.dimensionality]));
            eigen_vals.push(1.0);
        }
    }

    pub fn compute_pca_with_significance_threshold(
        _center: bool,
        _data: &utils::DenseDataset<f32>,
        _significance_threshold: f32,
        _truncation_threshold: f32,
        _build_covariance: bool,
        pca_vecs: &mut Vec<utils::DatapointPtr<f32>>,
        eigen_vals: &mut Vec<f32>,
        _parallelization_pool: Option<&ParallelizationPool>,
    ) {
        pca_vecs.clear();
        eigen_vals.clear();
        pca_vecs.push(utils::DatapointPtr::new(vec![0.0; _data.dimensionality]));
        eigen_vals.push(1.0);
    }
}

// Placeholder for parallelization pool
pub struct ParallelizationPool;

impl ParallelizationPool {
    pub fn new() -> Self {
        ParallelizationPool
    }
}

// Placeholder for random orthogonal projection
pub struct RandomOrthogonalProjection {
    input_dims: usize,
    projected_dims: usize,
    seed: u64,
    directions: Option<Arc<utils::DenseDataset<f32>>>,
}

impl RandomOrthogonalProjection {
    pub fn new(input_dims: usize, projected_dims: usize, seed: u64) -> Self {
        RandomOrthogonalProjection {
            input_dims,
            projected_dims,
            seed,
            directions: None,
        }
    }

    pub fn create(&mut self) {
        let mut data = vec![vec![0.0; self.input_dims]; self.projected_dims];
        for i in 0..self.projected_dims {
            data[i][i] = 1.0;
        }
        self.directions = Some(Arc::new(utils::DenseDataset::new(data, self.input_dims)));
    }

    pub fn get_directions(&self) -> Option<Arc<utils::DenseDataset<f32>>> {
        self.directions.clone()
    }
}

// Placeholder for dot product utility
fn dot_product<T: Copy + Into<f32>, U: Copy + Into<f32>>(a: &utils::DatapointPtr<T>, b: &utils::DatapointPtr<U>) -> f32 {
    a.values()
        .iter()
        .zip(b.values().iter())
        .map(|(&x, &y)| x.into() * y.into())
        .sum()
}

// Placeholder for one-to-many dot product
fn dense_dot_product_distance_one_to_many<T: Copy + Into<f32>, U: Copy + Into<f32>>(
    input: &utils::DatapointPtr<T>,
    dataset: &utils::DenseDataset<U>,
    output: &mut [f32],
) {
    for (i, row) in dataset.data.iter().enumerate() {
        output[i] = dot_product(input, &utils::DatapointPtr::new(row.clone()));
    }
}

pub struct PcaProjection<T> {
    input_dims: i32,
    projected_dims: i32,
    pca_vecs: Option<Arc<utils::DenseDataset<f32>>>,
}

impl<T: Copy + Into<f32> + Send + Sync> PcaProjection<T> {
    pub fn new(input_dims: i32, projected_dims: i32) -> Result<Self, Box<dyn Error>> {
        if input_dims <= 0 {
            return Err(invalid_argument_error("Input dimensionality must be > 0"));
        }
        if projected_dims <= 0 {
            return Err(invalid_argument_error("Projected dimensionality must be > 0"));
        }
        if input_dims < projected_dims {
            return Err(invalid_argument_error(
                "The projected dimensions cannot be larger than input dimensions",
            ));
        }
        Ok(PcaProjection {
            input_dims,
            projected_dims,
            pca_vecs: None,
        })
    }

    pub fn create(&mut self, data: &utils::DenseDataset<f32>, build_covariance: bool, parallelization_pool: Option<&ParallelizationPool>) {
        let mut eigen_vals = Vec::new();
        let mut pca_vecs = Vec::new();
        pca_utils::compute_pca(
            false,
            data,
            self.projected_dims as usize,
            build_covariance,
            &mut pca_vecs,
            &mut eigen_vals,
            parallelization_pool,
        );

        let mut pca_vec_dataset = utils::DenseDataset::new(Vec::new(), data.dimensionality());
        for vec in pca_vecs {
            pca_vec_dataset.append(vec.values(), "").unwrap();
        }
        self.pca_vecs = Some(Arc::new(pca_vec_dataset));
    }

    pub fn create_with_thresholds(
        &mut self,
        data: &utils::DenseDataset<f32>,
        pca_significance_threshold: f32,
        pca_truncation_threshold: f32,
        build_covariance: bool,
        parallelization_pool: Option<&ParallelizationPool>,
    ) {
        let mut eigen_vals = Vec::new();
        let mut pca_vecs = Vec::new();
        pca_utils::compute_pca_with_significance_threshold(
            false,
            data,
            pca_significance_threshold,
            pca_truncation_threshold,
            build_covariance,
            &mut pca_vecs,
            &mut eigen_vals,
            parallelization_pool,
        );

        let mut pca_vec_dataset = utils::DenseDataset::new(Vec::new(), data.dimensionality());
        for vec in pca_vecs {
            pca_vec_dataset.append(vec.values(), "").unwrap();
        }
        self.projected_dims = pca_vecs.len() as i32;
        self.pca_vecs = Some(Arc::new(pca_vec_dataset));
    }

    pub fn create_from_eigenvectors(&mut self, eigenvectors: utils::DenseDataset<f32>) {
        self.pca_vecs = Some(Arc::new(eigenvectors));
    }

    pub fn create_from_serialized(
        &mut self,
        serialized_projection: &proto::SerializedProjection,
    ) -> Result<(), Box<dyn Error>> {
        if serialized_projection.rotation_vec_size() == 0 {
            return Err(invalid_argument_error(
                "Serialized projection rotation matrix is empty in PcaProjection::create_from_serialized.",
            ));
        }
        let mut pca_vecs = utils::DenseDataset::new(
            Vec::new(),
            serialized_projection.rotation_vec()[0].feature_value_float.len(),
        );
        pca_vecs.reserve(serialized_projection.rotation_vec_size());
        for gfv in serialized_projection.rotation_vec() {
            pca_vecs.append(&gfv.feature_value_float, "")?;
        }
        self.pca_vecs = Some(Arc::new(pca_vecs));
        Ok(())
    }

    pub fn random_rotate_projection_matrix(&mut self) {
        let Some(pca_vecs) = &self.pca_vecs else {
            eprintln!("No PCA vectors to rotate.");
            return;
        };
        assert_eq!(pca_vecs.size(), self.projected_dims as usize);
        assert_eq!(pca_vecs.dimensionality(), self.input_dims as usize);

        let mut ortho = RandomOrthogonalProjection::new(
            self.projected_dims as usize,
            self.projected_dims as usize,
            42,
        );
        ortho.create();
        let ortho_vecs = ortho.get_directions().expect("Orthogonal vectors not initialized");
        assert_eq!(ortho_vecs.size(), self.projected_dims as usize);
        assert_eq!(ortho_vecs.dimensionality(), self.projected_dims as usize);

        let mut rotated_matrix = vec![0.0; (self.input_dims * self.projected_dims) as usize];
        let mut col_vec = vec![0.0; self.projected_dims as usize];

        for col_idx in 0..self.input_dims as usize {
            for row_idx in 0..self.projected_dims as usize {
                col_vec[row_idx] = pca_vecs.data[row_idx][col_idx];
            }
            for row_idx in 0..self.projected_dims as usize {
                rotated_matrix[row_idx * self.input_dims as usize + col_idx] = dot_product(
                    &utils::DatapointPtr::new(col_vec.clone()),
                    &utils::DatapointPtr::new(ortho_vecs.data[row_idx].clone()),
                );
            }
        }
        self.pca_vecs = Some(Arc::new(utils::DenseDataset::new(
            rotated_matrix
                .chunks(self.input_dims as usize)
                .map(|chunk| chunk.to_vec())
                .collect(),
            self.input_dims as usize,
        )));
    }

    pub fn project_input<FloatT: Copy + From<f32>>(
        &self,
        input: &utils::DatapointPtr<T>,
        projected: &mut utils::DatapointPtr<FloatT>,
    ) -> Result<(), Box<dyn Error>> {
        if self.pca_vecs.is_none() {
            return Err(failed_precondition_error("First compute the PCA directions."));
        }
        let pca_vecs = self.pca_vecs.as_ref().unwrap();
        projected.values.clear();
        projected.values.resize(self.projected_dims as usize, FloatT::from(0.0));

        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            dense_dot_product_distance_one_to_many(input, pca_vecs, &mut projected.values);
            for val in projected.values.iter_mut() {
                *val = From::from(-f32::from(*val));
            }
        } else {
            for (i, vec) in pca_vecs.data.iter().enumerate() {
                projected.values[i] = From::from(dot_product(input, &utils::DatapointPtr::new(vec.clone())));
            }
        }
        Ok(())
    }

    pub fn get_directions(&self) -> Option<Arc<utils::DenseDataset<f32>>> {
        self.pca_vecs.clone()
    }

    pub fn serialize_to_proto(&self) -> Option<proto::SerializedProjection> {
        let Some(pca_vecs) = &self.pca_vecs else {
            return None;
        };
        let mut result = proto::SerializedProjection::new();
        result.reserve_rotation_vec(pca_vecs.size());
        for eigenvector in &pca_vecs.data {
            *result.add_rotation_vec() = utils::DatapointPtr::new(eigenvector.clone()).to_gfv();
        }
        Some(result)
    }
}

impl<T: Clone> utils::DatapointPtr<T> {
    pub fn to_gfv(&self) -> proto::GenericFeatureVector {
        proto::GenericFeatureVector {
            feature_value_float: self.values.iter().map(|&v| v as f32).collect(),
        }
    }
}