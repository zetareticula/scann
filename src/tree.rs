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

//! K-means tree training options for data partitioning.

use super::proto;

// Placeholder for GmmUtils options
mod gmm_utils {
    #[derive(Clone, PartialEq)]
    pub enum BalancingType {
        Unbalanced,
        GreedyBalanced,
        UnbalancedFloat32,
    }

    #[derive(Clone, PartialEq)]
    pub enum ReassignmentType {
        RandomReassignment,
        PcaSplitting,
    }

    #[derive(Clone, PartialEq)]
    pub enum CenterInitializationType {
        KmeansPlusPlus,
        RandomInitialization,
    }
}

#[derive(Clone)]
pub struct KMeansTreeTrainingOptions {
    pub partitioning_type: proto::PartitioningType,
    pub max_num_levels: i32,
    pub max_leaf_size: i32,
    pub learned_spilling_type: proto::SpillingType,
    pub per_node_spilling_factor: f32,
    pub max_spill_centers: i32,
    pub max_iterations: i32,
    pub convergence_epsilon: f32,
    pub min_cluster_size: i32,
    pub seed: u64,
    pub balancing_type: gmm_utils::BalancingType,
    pub reassignment_type: gmm_utils::ReassignmentType,
    pub center_initialization_type: gmm_utils::CenterInitializationType,
}

impl KMeansTreeTrainingOptions {
    pub fn new() -> Self {
        KMeansTreeTrainingOptions {
            partitioning_type: proto::PartitioningType::Default,
            max_num_levels: 0,
            max_leaf_size: 0,
            learned_spilling_type: proto::SpillingType::Default,
            per_node_spilling_factor: 0.0,
            max_spill_centers: 0,
            max_iterations: 0,
            convergence_epsilon: 0.0,
            min_cluster_size: 0,
            seed: 0,
            balancing_type: gmm_utils::BalancingType::Unbalanced,
            reassignment_type: gmm_utils::ReassignmentType::RandomReassignment,
            center_initialization_type: gmm_utils::CenterInitializationType::KmeansPlusPlus,
        }
    }

    pub fn from_config(config: &proto::PartitioningConfig) -> Self {
        let balancing_type = match config.balancing_type() {
            proto::BalancingType::DefaultUnbalanced => gmm_utils::BalancingType::Unbalanced,
            proto::BalancingType::GreedyBalanced => gmm_utils::BalancingType::GreedyBalanced,
            proto::BalancingType::UnbalancedFloat32 => gmm_utils::BalancingType::UnbalancedFloat32,
        };

        let reassignment_type = match config.trainer_type() {
            proto::TrainerType::DefaultSamplingTrainer | proto::TrainerType::FlumeKmeansTrainer => {
                gmm_utils::ReassignmentType::RandomReassignment
            }
            proto::TrainerType::PcaKmeansTrainer | proto::TrainerType::SamplingPcaKmeansTrainer => {
                gmm_utils::ReassignmentType::PcaSplitting
            }
        };

        let center_initialization_type = match config.single_machine_center_initialization() {
            proto::CenterInitializationType::DefaultKmeansPlusPlus => {
                gmm_utils::CenterInitializationType::KmeansPlusPlus
            }
            proto::CenterInitializationType::RandomInitialization => {
                gmm_utils::CenterInitializationType::RandomInitialization
            }
        };

        KMeansTreeTrainingOptions {
            partitioning_type: config.partitioning_type(),
            max_num_levels: config.max_num_levels(),
            max_leaf_size: config.max_leaf_size(),
            learned_spilling_type: config.database_spilling().spilling_type.clone(),
            per_node_spilling_factor: config.database_spilling().replication_factor,
            max_spill_centers: config.database_spilling().max_spill_centers,
            max_iterations: config.max_clustering_iterations(),
            convergence_epsilon: config.clustering_convergence_tolerance(),
            min_cluster_size: config.min_cluster_size(),
            seed: config.clustering_seed(),
            balancing_type,
            reassignment_type,
            center_initialization_type,
        }
    }
}