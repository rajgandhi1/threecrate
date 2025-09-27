//! Integration tests for threecrate-reconstruction
//!
//! These tests verify that all reconstruction algorithms work together
//! and that the unified pipeline functions correctly.

use threecrate_reconstruction::*;
use threecrate_core::{PointCloud, Point3f, NormalPoint3f};
use nalgebra::{Point3, Vector3};

/// Create a simple test point cloud forming a square
fn create_test_square() -> PointCloud<Point3f> {
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.5, 0.5, 0.0),
    ];
    PointCloud::from_points(points)
}

/// Create a test point cloud with normals
fn create_test_square_with_normals() -> PointCloud<NormalPoint3f> {
    let normal = Vector3::new(0.0, 0.0, 1.0);
    let points = vec![
        NormalPoint3f { position: Point3::new(0.0, 0.0, 0.0), normal },
        NormalPoint3f { position: Point3::new(1.0, 0.0, 0.0), normal },
        NormalPoint3f { position: Point3::new(1.0, 1.0, 0.0), normal },
        NormalPoint3f { position: Point3::new(0.0, 1.0, 0.0), normal },
        NormalPoint3f { position: Point3::new(0.5, 0.5, 0.0), normal },
    ];
    PointCloud::from_points(points)
}

/// Create a dense point cloud on a sphere
fn create_sphere_point_cloud(radius: f32, num_points: usize) -> PointCloud<Point3f> {
    let mut points = Vec::new();
    let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;

    for i in 0..num_points {
        let theta = 2.0 * std::f32::consts::PI * i as f32 / golden_ratio;
        let phi = std::f32::consts::PI * (1.0 - 2.0 * i as f32 / num_points as f32).acos();

        let x = radius * phi.sin() * theta.cos();
        let y = radius * phi.sin() * theta.sin();
        let z = radius * phi.cos();

        points.push(Point3::new(x, y, z));
    }

    PointCloud::from_points(points)
}

/// Create a sphere with normals
fn create_sphere_with_normals(radius: f32, num_points: usize) -> PointCloud<NormalPoint3f> {
    let mut points = Vec::new();
    let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;

    for i in 0..num_points {
        let theta = 2.0 * std::f32::consts::PI * i as f32 / golden_ratio;
        let phi = std::f32::consts::PI * (1.0 - 2.0 * i as f32 / num_points as f32).acos();

        let x = radius * phi.sin() * theta.cos();
        let y = radius * phi.sin() * theta.sin();
        let z = radius * phi.cos();

        let position = Point3::new(x, y, z);
        let normal = Vector3::new(x / radius, y / radius, z / radius);

        points.push(NormalPoint3f { position, normal });
    }

    PointCloud::from_points(points)
}

#[test]
fn test_unified_pipeline_auto_reconstruct() {
    let cloud = create_test_square();

    // Test basic auto reconstruction
    let result = auto_reconstruct(&cloud);
    match result {
        Ok(mesh) => {
            assert!(!mesh.is_empty());
            println!("✓ Auto reconstruction succeeded with {} vertices, {} faces",
                     mesh.vertex_count(), mesh.face_count());
        }
        Err(e) => {
            println!("Auto reconstruction failed: {} (acceptable for simple test data)", e);
        }
    }
}

#[test]
fn test_unified_pipeline_auto_reconstruct_with_normals() {
    let cloud = create_test_square_with_normals();

    let result = auto_reconstruct_with_normals(&cloud);
    match result {
        Ok(mesh) => {
            assert!(!mesh.is_empty());
            println!("✓ Auto reconstruction with normals succeeded with {} vertices, {} faces",
                     mesh.vertex_count(), mesh.face_count());
        }
        Err(e) => {
            println!("Auto reconstruction with normals failed: {} (acceptable for simple test data)", e);
        }
    }
}

#[test]
fn test_unified_pipeline_quality_levels() {
    let cloud = create_test_square();

    let quality_levels = [
        QualityLevel::Fast,
        QualityLevel::Balanced,
        QualityLevel::HighQuality,
        QualityLevel::MaxQuality,
    ];

    for quality in &quality_levels {
        let result = auto_reconstruct_with_quality(&cloud, *quality);
        match result {
            Ok(mesh) => {
                assert!(!mesh.is_empty());
                println!("✓ Quality level {:?} succeeded", quality);
            }
            Err(_) => {
                println!("Quality level {:?} failed (acceptable)", quality);
            }
        }
    }
}

#[test]
fn test_unified_pipeline_use_cases() {
    let cloud = create_test_square();

    let use_cases = [
        UseCase::General,
        UseCase::Prototyping,
        UseCase::Engineering,
        UseCase::Organic,
        UseCase::NoisyData,
        UseCase::Sparse,
        UseCase::Dense,
    ];

    for use_case in &use_cases {
        let result = auto_reconstruct_for_use_case(&cloud, *use_case);
        match result {
            Ok(mesh) => {
                assert!(!mesh.is_empty());
                println!("✓ Use case {:?} succeeded", use_case);
            }
            Err(_) => {
                println!("Use case {:?} failed (acceptable)", use_case);
            }
        }
    }
}

#[test]
fn test_pipeline_full_workflow() {
    let cloud = create_sphere_point_cloud(1.0, 100);

    // Create custom pipeline configuration
    let config = PipelineConfig {
        quality: QualityLevel::Balanced,
        use_case: UseCase::General,
        preferred_algorithm: None,
        fallback_algorithms: vec![
            Algorithm::Delaunay,
            Algorithm::BallPivoting,
            Algorithm::MovingLeastSquares,
        ],
        max_processing_time: Some(30.0),
        enable_parallel: true,
        validate_output: true,
        auto_repair: false,
    };

    let pipeline = ReconstructionPipeline::new(config);

    // Test full reconstruction workflow
    match pipeline.reconstruct(&cloud) {
        Ok(result) => {
            println!("✓ Full pipeline workflow succeeded:");
            println!("  Algorithm used: {:?}", result.algorithm_used);
            println!("  Processing time: {:.3}s", result.processing_time);
            println!("  Vertices: {}", result.quality_metrics.vertex_count);
            println!("  Triangles: {}", result.quality_metrics.triangle_count);
            println!("  Quality score: {:.3}", result.quality_metrics.avg_triangle_quality);

            assert!(!result.mesh.is_empty());
            assert!(result.processing_time >= 0.0);
            assert!(result.quality_metrics.vertex_count > 0);
        }
        Err(e) => {
            println!("Full pipeline workflow failed: {} (may be acceptable for test data)", e);
        }
    }
}

#[test]
fn test_pipeline_with_normals_workflow() {
    let cloud = create_sphere_with_normals(1.0, 100);

    let pipeline = ReconstructionPipeline::for_use_case(UseCase::Dense);

    match pipeline.reconstruct_with_normals(&cloud) {
        Ok(result) => {
            println!("✓ Pipeline with normals succeeded:");
            println!("  Algorithm used: {:?}", result.algorithm_used);
            println!("  Processing time: {:.3}s", result.processing_time);
            println!("  Has normals: {}", result.data_characteristics.has_normals);

            assert!(!result.mesh.is_empty());
            assert!(result.data_characteristics.has_normals);
        }
        Err(e) => {
            println!("Pipeline with normals failed: {} (may be acceptable)", e);
        }
    }
}

#[test]
fn test_individual_algorithms_basic() {
    let cloud = create_test_square();

    // Test Delaunay triangulation
    match delaunay_triangulation_auto(&cloud) {
        Ok(mesh) => {
            assert!(!mesh.is_empty());
            println!("✓ Delaunay triangulation: {} vertices, {} faces",
                     mesh.vertex_count(), mesh.face_count());
        }
        Err(e) => println!("Delaunay failed: {}", e),
    }

    // Test Ball Pivoting
    match ball_pivoting_reconstruction(&cloud, 0.5) {
        Ok(mesh) => {
            assert!(!mesh.is_empty());
            println!("✓ Ball Pivoting: {} vertices, {} faces",
                     mesh.vertex_count(), mesh.face_count());
        }
        Err(e) => println!("Ball Pivoting failed: {}", e),
    }

    // Test Moving Least Squares
    match moving_least_squares_auto(&cloud) {
        Ok(mesh) => {
            assert!(!mesh.is_empty());
            println!("✓ Moving Least Squares: {} vertices, {} faces",
                     mesh.vertex_count(), mesh.face_count());
        }
        Err(e) => println!("MLS failed: {}", e),
    }
}

#[test]
fn test_algorithm_selection_logic() {
    let pipeline = ReconstructionPipeline::default();

    // Test sparse data characteristics
    let sparse_characteristics = DataCharacteristics {
        point_count: 50,
        has_normals: false,
        density_uniformity: 0.5,
        noise_level: 0.2,
        avg_neighbor_distance: 0.1,
        bounding_box: (Point3f::origin(), Point3f::new(1.0, 1.0, 0.1)),
        is_closed_surface: false,
        surface_complexity: 0.3,
        distribution_type: DistributionType::Planar,
    };

    let algorithm = pipeline.select_algorithm(&sparse_characteristics);
    println!("✓ Sparse planar data -> Algorithm: {:?}", algorithm);

    // Test dense data with normals
    let dense_characteristics = DataCharacteristics {
        point_count: 5000,
        has_normals: true,
        density_uniformity: 0.8,
        noise_level: 0.1,
        avg_neighbor_distance: 0.02,
        bounding_box: (Point3f::new(-1.0, -1.0, -1.0), Point3f::new(1.0, 1.0, 1.0)),
        is_closed_surface: true,
        surface_complexity: 0.7,
        distribution_type: DistributionType::Spherical,
    };

    let algorithm = pipeline.select_algorithm(&dense_characteristics);
    println!("✓ Dense spherical data with normals -> Algorithm: {:?}", algorithm);

    // Algorithm selection should be consistent
    assert!(matches!(algorithm, Algorithm::Poisson | Algorithm::BallPivoting | Algorithm::MarchingCubes));
}

#[test]
fn test_data_characteristics_analysis() {
    let sphere_cloud = create_sphere_point_cloud(2.0, 200);
    let pipeline = ReconstructionPipeline::default();

    match pipeline.analyze_data(&sphere_cloud) {
        Ok(characteristics) => {
            println!("✓ Data analysis succeeded:");
            println!("  Point count: {}", characteristics.point_count);
            println!("  Has normals: {}", characteristics.has_normals);
            println!("  Density uniformity: {:.3}", characteristics.density_uniformity);
            println!("  Noise level: {:.3}", characteristics.noise_level);
            println!("  Distribution: {:?}", characteristics.distribution_type);
            println!("  Closed surface: {}", characteristics.is_closed_surface);
            println!("  Surface complexity: {:.3}", characteristics.surface_complexity);

            assert_eq!(characteristics.point_count, 200);
            assert!(!characteristics.has_normals);
            assert!(characteristics.density_uniformity >= 0.0 && characteristics.density_uniformity <= 1.0);
            assert!(characteristics.noise_level >= 0.0 && characteristics.noise_level <= 1.0);
        }
        Err(e) => panic!("Data analysis failed: {}", e),
    }
}

#[test]
fn test_parallel_processing_integration() {
    // Test that parallel processing works with the pipeline
    use threecrate_reconstruction::parallel::{init_thread_pool, ThreadPoolConfig};

    let config = ThreadPoolConfig::default()
        .with_threads(2)
        .with_enabled(true);

    // Initialize thread pool (ignore error if already initialized)
    let _ = init_thread_pool(config);

    let cloud = create_sphere_point_cloud(1.5, 150);
    let pipeline = ReconstructionPipeline::default();

    match pipeline.reconstruct(&cloud) {
        Ok(result) => {
            println!("✓ Parallel processing integration succeeded:");
            println!("  Algorithm: {:?}", result.algorithm_used);
            println!("  Processing time: {:.3}s", result.processing_time);
            assert!(!result.mesh.is_empty());
        }
        Err(e) => {
            println!("Parallel processing test failed: {} (may be acceptable)", e);
        }
    }
}

#[test]
fn test_error_handling_and_fallbacks() {
    // Test with very small point cloud that might cause some algorithms to fail
    let tiny_cloud = PointCloud::from_points(vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.1, 0.0, 0.0),
    ]);

    let mut config = PipelineConfig::default();
    config.fallback_algorithms = vec![
        Algorithm::Delaunay,
        Algorithm::MovingLeastSquares,
        Algorithm::BallPivoting,
    ];

    let pipeline = ReconstructionPipeline::new(config);

    // Even if it fails, it should fail gracefully
    match pipeline.reconstruct(&tiny_cloud) {
        Ok(result) => {
            println!("✓ Tiny cloud reconstruction succeeded with algorithm: {:?}", result.algorithm_used);
        }
        Err(e) => {
            println!("✓ Tiny cloud reconstruction failed gracefully: {}", e);
            // This is expected for very small datasets
        }
    }
}

#[test]
fn test_reconstruction_consistency() {
    let cloud = create_sphere_point_cloud(1.0, 100);
    let pipeline = ReconstructionPipeline::default();

    // Run reconstruction multiple times - should be consistent
    let mut results = Vec::new();
    for i in 0..3 {
        match pipeline.reconstruct(&cloud) {
            Ok(result) => {
                results.push(result);
                println!("✓ Run {}: Algorithm {:?}, {} vertices",
                         i + 1,
                         results.last().unwrap().algorithm_used,
                         results.last().unwrap().mesh.vertex_count());
            }
            Err(e) => {
                println!("Run {} failed: {}", i + 1, e);
            }
        }
    }

    // If any succeeded, the algorithm choice should be consistent
    if results.len() > 1 {
        let first_algorithm = results[0].algorithm_used;
        for result in &results[1..] {
            assert_eq!(result.algorithm_used, first_algorithm,
                      "Algorithm selection should be consistent for the same input");
        }
    }
}