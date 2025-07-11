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

//! ScaNN (Scalable Nearest Neighbors) library with RETRO model integration.

pub mod assets;
pub mod distance_measures;
pub mod projection;
pub mod proto;
pub mod retrieval;
pub mod retro;
pub mod serialize;
pub mod trees;
pub mod util;

// Re-export key types
pub use assets::populate_and_save_assets_proto;
pub use distance_measures::{get_distance_measure, DistanceMeasure};
pub use projection::{PcaProjection, RandomOrthogonalProjection};
pub use retrieval::ScannRetriever;
pub use retro::RETRO;
pub use trees::KMeansTreeTrainingOptions;
pub use utils::{DenseDataset, DatapointPtr, ScannError};