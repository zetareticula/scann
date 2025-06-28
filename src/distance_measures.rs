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

//! Distance measure factory for ScaNN.

use super::{proto, utils, ScannError};
use std::error::Error;
use ngalgebra::{DVector, Vector};

pub struct CosineDistance;

impl CosineDistance {
    pub fn new() -> Self {
        CosineDistance
    }
}

impl DistanceMeasure for CosineDistance {
    fn compute_distance<T: Copy + Into<f32>>(&self, a: &utils::DatapointPtr<T>, b: &utils::DatapointPtr<T>) -> f32 {
        let a_vec: Vec<f32> = a.values().iter().map(|&x| x.into()).collect();
        let b_vec: Vec<f32> = b.values().iter().map(|&x| x.into()).collect();
        let a = DVector::from_vec(a_vec);
        let b = DVector::from_vec(b_vec);
        let norm_a = a.norm();
        let norm_b = b.norm();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0; // Max distance if either vector is zero
        }
        1.0 - (a.dot(&b) / (norm_a * norm_b)).max(-1.0).min(1.0)
    }
}


pub trait DistanceMeasure: Send + Sync {
    fn compute_distance<T: Copy + Into<f32>>(&self, a: &utils::DatapointPtr<T>, b: &utils::DatapointPtr<T>) -> f32;
}

// Placeholder implementations for distance measures
macro_rules! define_distance_measure {
    ($name:ident) => {
        pub struct $name;

        impl $name {
            pub fn new() -> Self {
                $name
            }
        }

        impl DistanceMeasure for $name {
            fn compute_distance<T: Copy + Into<f32>>(&self, a: &utils::DatapointPtr<T>, b: &utils::DatapointPtr<T>) -> f32 {
                // Placeholder: Implement actual distance computation
                // For example, DotProductDistance would compute sum(a[i] * b[i])
                let sum: f32 = a
                    .values()
                    .iter()
                    .zip(b.values().iter())
                    .map(|(&x, &y)| x.into() * y.into())
                    .sum();
                sum
            }
        }
    };
}

define_distance_measure!(DotProductDistance);
define_distance_measure!(BinaryDotProductDistance);
define_distance_measure!(AbsDotProductDistance);
define_distance_measure!(L2Distance);
define_distance_measure!(SquaredL2Distance);
define_distance_measure!(NegatedSquaredL2Distance);
define_distance_measure!(L1Distance);
define_distance_measure!(CosineDistance);
define_distance_measure!(BinaryCosineDistance);
define_distance_measure!(GeneralJaccardDistance);
define_distance_measure!(BinaryJaccardDistance);
define_distance_measure!(LimitedInnerProductDistance);
define_distance_measure!(GeneralHammingDistance);
define_distance_measure!(BinaryHammingDistance);
define_distance_measure!(NonzeroIntersectDistance);

pub fn get_distance_measure(config: &proto::DistanceMeasureConfig) -> Result<Box<dyn DistanceMeasure>, Box<dyn Error>> {
    if config.distance_measure().is_empty() {
        return Err(Box::new(ScannError {
            message: "Empty DistanceMeasureConfig proto! Must specify distance_measure.".to_string(),
        }));
    }
    get_distance_measure_by_name(config.distance_measure())
}

pub fn get_distance_measure_by_name(name: &str) -> Result<Box<dyn DistanceMeasure>, Box<dyn Error>> {
    match name {
        "DotProductDistance" => Ok(Box::new(DotProductDistance::new())),
        "BinaryDotProductDistance" => Ok(Box::new(BinaryDotProductDistance::new())),
        "AbsDotProductDistance" => Ok(Box::new(AbsDotProductDistance::new())),
        "L2Distance" => Ok(Box::new(L2Distance::new())),
        "SquaredL2Distance" => Ok(Box::new(SquaredL2Distance::new())),
        "NegatedSquaredL2Distance" => Ok(Box::new(NegatedSquaredL2Distance::new())),
        "L1Distance" => Ok(Box::new(L1Distance::new())),
        "CosineDistance" => Ok(Box::new(CosineDistance::new())),
        "BinaryCosineDistance" => Ok(Box::new(BinaryCosineDistance::new())),
        "GeneralJaccardDistance" => Ok(Box::new(GeneralJaccardDistance::new())),
        "BinaryJaccardDistance" => Ok(Box::new(BinaryJaccardDistance::new())),
        "LimitedInnerProductDistance" => Ok(Box::new(LimitedInnerProductDistance::new())),
        "GeneralHammingDistance" => Ok(Box::new(GeneralHammingDistance::new())),
        "BinaryHammingDistance" => Ok(Box::new(BinaryHammingDistance::new())),
        "NonzeroIntersectDistance" => Ok(Box::new(NonzeroIntersectDistance::new())),
        
        _ => Err(Box::new(ScannError {
            message: format!("Invalid distance_measure: '{}'", name),
        })),
    }
}