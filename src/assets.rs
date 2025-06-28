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

//! Assets serialization for ScaNN.

use super::{proto, ScannError};
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn path_exists<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().exists()
}

pub fn populate_and_save_assets_proto<P: AsRef<Path>>(
    artifacts_dir: P,
) -> Result<proto::ScannAssets, Box<dyn Error>> {
    let artifacts_dir = artifacts_dir.as_ref();
    let mut assets = proto::ScannAssets {
        assets: Vec::new(),
    };

    fn add_if_exists(
        assets: &mut proto::ScannAssets,
        artifacts_dir: &Path,
        filename: &str,
        asset_type: proto::AssetType,
    ) {
        let file_path = artifacts_dir.join(filename);
        if path_exists(&file_path) {
            assets.assets.push(proto::ScannAsset {
                asset_path: file_path.to_string_lossy().into_owned(),
                asset_type,
            });
        }
    }

    add_if_exists(&mut assets, artifacts_dir, "ah_codebook.pb", proto::AssetType::AhCenters);
    add_if_exists(&mut assets, artifacts_dir, "serialized_partitioner.pb", proto::AssetType::Partitioner);
    add_if_exists(&mut assets, artifacts_dir, "datapoint_to_token.npy", proto::AssetType::TokenizationNpy);
    add_if_exists(&mut assets, artifacts_dir, "hashed_dataset.npy", proto::AssetType::AhDatasetNpy);
    add_if_exists(&mut assets, artifacts_dir, "int8_dataset.npy", proto::AssetType::Int8DatasetNpy);
    add_if_exists(&mut assets, artifacts_dir, "int8_multipliers.npy", proto::AssetType::Int8MultipliersNpy);
    add_if_exists(&mut assets, artifacts_dir, "dp_norms.npy", proto::AssetType::Int8NormsNpy);
    add_if_exists(&mut assets, artifacts_dir, "dataset.npy", proto::AssetType::DatasetNpy);

    let output_path = artifacts_dir.join("scann_assets.pbtxt");
    let mut file = File::create(&output_path).map_err(|e| {
        ScannError {
            message: format!("Failed to create file {}: {}", output_path.display(), e),
        }
    })?;
    write!(file, "{}", assets).map_err(|e| {
        ScannError {
            message: format!("Failed to write to file {}: {}", output_path.display(), e),
        }
    })?;

    Ok(assets)
}