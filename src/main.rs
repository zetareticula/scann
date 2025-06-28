use scann::assets::populate_and_save_assets_proto;
use scann::projection::PcaProjection;
use scann::trees::KMeansTreeTrainingOptions;
use scann::utils::{DenseDataset, DatapointPtr};
use scann::proto::PartitioningConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Serialize assets
    let assets = populate_and_save_assets_proto("path/to/artifacts")?;
    println!("Assets found: {}", assets.assets.len());

    // Create and use PCA projection
    let mut pca = PcaProjection::<f32>::new(10, 5)?;
    let data = DenseDataset::new(vec![vec![1.0; 10]; 100], 10);
    pca.create(&data, true, None);
    let input = DatapointPtr::new(vec![1.0; 10]);
    let mut projected = DatapointPtr::new(vec![0.0; 5]);
    pca.project_input(&input, &mut projected)?;
    println!("Projected: {:?}", projected.values());

    // Configure k-means tree training
    let config = PartitioningConfig {
        partitioning_type: scann::proto::PartitioningType::Default,
        max_num_levels: 5,
        max_leaf_size: 100,
        database_spilling: scann::proto::DatabaseSpilling {
            spilling_type: scann::proto::SpillingType::Default,
            replication_factor: 1.0,
            max_spill_centers: 1000,
        },
        max_clustering_iterations: 50,
        clustering_convergence_tolerance: 0.01,
        min_cluster_size: 10,
        clustering_seed: 42,
        balancing_type: scann::proto::BalancingType::GreedyBalanced,
        trainer_type: scann::proto::TrainerType::PcaKmeansTrainer,
        single_machine_center_initialization: scann::proto::CenterInitializationType::RandomInitialization,
    };
    let options = KMeansTreeTrainingOptions::from_config(&config);
    println!("KMeansTreeTrainingOptions: {:?}", format!("{:?}", options));

    Ok(())
}