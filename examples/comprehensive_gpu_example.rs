use threecrate_gpu::{
    GpuContext, gpu_estimate_normals, gpu_icp, gpu_remove_statistical_outliers,
    gpu_tsdf_integrate, gpu_tsdf_extract_surface, create_tsdf_volume,
    RenderConfig, RenderParams, CameraIntrinsics, point_cloud_to_vertices_colored,
    colored_point_cloud_to_vertices
};
use threecrate_core::{PointCloud, Point3f};
use nalgebra::{Matrix4, Vector3, Point3, Isometry3};
use rand::prelude::*;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ThreeCrate GPU Comprehensive Demo");
    println!("=================================");
    
    // Initialize GPU context
    let gpu_context = GpuContext::new().await?;
    println!("✓ GPU context initialized");
    
    // Generate sample point clouds
    let (source_cloud, target_cloud, depth_image) = generate_sample_data();
    println!("✓ Generated sample point clouds ({} and {} points)", 
             source_cloud.points.len(), target_cloud.points.len());
    
    // 1. GPU Normal Estimation
    println!("\n1. GPU Normal Estimation");
    println!("------------------------");
    let start = Instant::now();
    let mut source_cloud_copy = source_cloud.clone();
    let normals_result = gpu_estimate_normals(&gpu_context, &mut source_cloud_copy, 10).await?;
    let normals_time = start.elapsed();
    println!("✓ Estimated {} normals in {:.2}ms", 
             normals_result.points.len(), normals_time.as_millis());
    
    // 2. GPU Statistical Outlier Removal
    println!("\n2. GPU Statistical Outlier Removal");
    println!("----------------------------------");
    let start = Instant::now();
    let filtered_cloud = gpu_remove_statistical_outliers(&gpu_context, &source_cloud, 10, 1.0).await?;
    let filter_time = start.elapsed();
    println!("✓ Filtered {} -> {} points in {:.2}ms", 
             source_cloud.points.len(), filtered_cloud.points.len(), filter_time.as_millis());
    
    // 3. GPU ICP Registration
    println!("\n3. GPU ICP Registration");
    println!("----------------------");
    let start = Instant::now();
    let transform = gpu_icp(&gpu_context, &source_cloud, &target_cloud, 20, 0.01, 2.0).await?;
    let icp_time = start.elapsed();
    println!("✓ ICP registration completed in {:.2}ms", icp_time.as_millis());
    println!("  Transform: {:?}", transform);
    
    // Apply transformation to source cloud
    let mut transformed_cloud = source_cloud.clone();
    for point in &mut transformed_cloud.points {
        let transformed = transform * Point3::new(point.x, point.y, point.z);
        point.x = transformed.x;
        point.y = transformed.y;
        point.z = transformed.z;
    }
    
    // 4. GPU TSDF Fusion
    println!("\n4. GPU TSDF Fusion");
    println!("------------------");
    let start = Instant::now();
    
    // Create TSDF volume
    let mut tsdf_volume = create_tsdf_volume(
        0.01,  // voxel_size
        0.05,  // truncation_distance
        [128, 128, 128], // resolution
        Point3::new(-0.64, -0.64, -0.64), // origin
    );
    
    // Create camera parameters
    let camera_intrinsics = CameraIntrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
        width: 640,
        height: 480,
        depth_scale: 0.001,
        _padding: 0.0,
    };
    
    let camera_pose = Matrix4::identity();
    
    // Integrate depth image
    let voxels = gpu_tsdf_integrate(
        &gpu_context,
        &mut tsdf_volume,
        &depth_image,
        None, // No color image
        &camera_pose,
        &camera_intrinsics,
    ).await?;
    
    let tsdf_time = start.elapsed();
    println!("✓ TSDF integration completed in {:.2}ms", tsdf_time.as_millis());
    println!("  Volume: {}x{}x{} voxels", 
             tsdf_volume.resolution[0], tsdf_volume.resolution[1], tsdf_volume.resolution[2]);
    
    // Extract surface from TSDF
    let start = Instant::now();
    let surface_cloud = gpu_tsdf_extract_surface(&gpu_context, &tsdf_volume, &voxels, 0.0).await?;
    let extraction_time = start.elapsed();
    println!("✓ Surface extraction completed in {:.2}ms", extraction_time.as_millis());
    println!("  Extracted {} surface points", surface_cloud.points.len());
    
    // 5. Performance Summary
    println!("\n5. Performance Summary");
    println!("---------------------");
    let total_time = normals_time + filter_time + icp_time + tsdf_time + extraction_time;
    println!("  Normal estimation:    {:.2}ms", normals_time.as_millis());
    println!("  Outlier removal:      {:.2}ms", filter_time.as_millis());
    println!("  ICP registration:     {:.2}ms", icp_time.as_millis());
    println!("  TSDF integration:     {:.2}ms", tsdf_time.as_millis());
    println!("  Surface extraction:   {:.2}ms", extraction_time.as_millis());
    println!("  ─────────────────────────────");
    println!("  Total GPU processing: {:.2}ms", total_time.as_millis());
    
    // 6. GPU Point Cloud Rendering with Splatting
    println!("\n6. GPU Point Cloud Rendering with Splatting");
    println!("-------------------------------------------");
    println!("✓ Enhanced splatting rendering pipeline ready");
    
    // Demonstrate different rendering modes
    let render_config = RenderConfig {
        render_params: RenderParams {
            point_size: 8.0,
            alpha_threshold: 0.05,
            enable_splatting: 1.0,
            enable_lighting: 1.0,
            ambient_strength: 0.2,
            diffuse_strength: 0.8,
            specular_strength: 0.3,
            shininess: 64.0,
        },
        enable_multisampling: true,
        enable_depth_test: true,
        enable_alpha_blending: true,
        background_color: [0.05, 0.05, 0.1, 1.0],
    };
    
    // Convert point clouds to vertices with automatic normal estimation
    let filtered_vertices = point_cloud_to_vertices_colored(&filtered_cloud, render_config.render_params.point_size);
    let surface_vertices = colored_point_cloud_to_vertices(&surface_cloud, render_config.render_params.point_size);
    
    println!("✓ Converted {} filtered points to render vertices", filtered_vertices.len());
    println!("✓ Converted {} surface points to render vertices", surface_vertices.len());
    
    // Different rendering modes demonstration
    println!("\n  Rendering Modes:");
    println!("  • Gaussian Splatting: Enabled (smooth circular splats)");
    println!("  • Phong Lighting: Enabled (realistic shading)");
    println!("  • MSAA: {}x (anti-aliasing)", if render_config.enable_multisampling { "4" } else { "1" });
    println!("  • Point Size: {:.1}px", render_config.render_params.point_size);
    println!("  • Alpha Threshold: {:.2}", render_config.render_params.alpha_threshold);
    
    // Camera setup for rendering
    let _view_matrix = Matrix4::look_at_rh(
        &Point3::new(2.0, 2.0, 2.0),
        &Point3::new(0.0, 0.0, 0.0),
        &Vector3::new(0.0, 1.0, 0.0),
    );
    let _proj_matrix = Matrix4::new_perspective(16.0 / 9.0, 45.0f32.to_radians(), 0.1, 100.0);
    let _camera_pos = Vector3::new(2.0, 2.0, 2.0);
    
    println!("✓ Camera matrices configured for 3D perspective rendering");
    println!("  Note: Use 'cargo run --example interactive_gpu_demo' for live rendering");
    
    // Demonstrate different splatting modes
    let _standard_params = RenderParams {
        enable_splatting: 0.0,
        enable_lighting: 0.0,
        ..render_config.render_params
    };
    
    let gaussian_params = RenderParams {
        enable_splatting: 1.0,
        enable_lighting: 1.0,
        ..render_config.render_params
    };
    
    println!("\n  Splatting Comparison:");
    println!("  • Standard Mode: Simple circular points with distance-based brightness");
    println!("  • Gaussian Mode: Smooth gaussian splats with Phong lighting");
    println!("  • Point Size Range: 1.0 - 64.0px (distance-based scaling)");
    println!("  • Lighting: Ambient({:.1}) + Diffuse({:.1}) + Specular({:.1})", 
             gaussian_params.ambient_strength, 
             gaussian_params.diffuse_strength, 
             gaussian_params.specular_strength);
    
    println!("\nGPU acceleration demo completed successfully! 🚀");
    println!("All algorithms executed on GPU with excellent performance.");
    println!("Enhanced point cloud splatting provides high-quality rendering!");
    
    Ok(())
}

fn generate_sample_data() -> (PointCloud<Point3f>, PointCloud<Point3f>, Vec<f32>) {
    let mut rng = thread_rng();
    
    // Generate source point cloud (sphere)
    let mut source_points = Vec::new();
    for _ in 0..1000 {
        let theta = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
        let phi = rng.gen_range(0.0..std::f32::consts::PI);
        let radius = 0.5 + rng.gen_range(0.0..0.1); // Slightly noisy sphere
        
        let x = radius * phi.sin() * theta.cos();
        let y = radius * phi.sin() * theta.sin();
        let z = radius * phi.cos();
        
        source_points.push(Point3f::new(x, y, z));
    }
    
    // Add some outliers to demonstrate filtering
    for _ in 0..50 {
        source_points.push(Point3f::new(
            rng.gen_range(-2.0..2.0),
            rng.gen_range(-2.0..2.0),
            rng.gen_range(-2.0..2.0),
        ));
    }
    
    let mut source_cloud = PointCloud::new();
    source_cloud.points = source_points;
    
    // Generate target point cloud (transformed and noisy sphere)
    let transform = Isometry3::new(
        Vector3::new(0.1, 0.2, 0.05),
        Vector3::new(0.1, 0.05, 0.15),
    );
    
    let mut target_points = Vec::new();
    for point in &source_cloud.points[..1000] { // Exclude outliers from target
        let transformed = transform * Point3::new(point.x, point.y, point.z);
        let noise = Vector3::new(
            rng.gen_range(-0.01..0.01),
            rng.gen_range(-0.01..0.01),
            rng.gen_range(-0.01..0.01),
        );
        let noisy_point = transformed + noise;
        
        target_points.push(Point3f::new(
            noisy_point.x,
            noisy_point.y,
            noisy_point.z,
        ));
    }
    
    let mut target_cloud = PointCloud::new();
    target_cloud.points = target_points;
    
    // Generate synthetic depth image
    let width = 640;
    let height = 480;
    let mut depth_image = vec![0.0f32; width * height];
    
    for y in 0..height {
        for x in 0..width {
            let nx = (x as f32 / width as f32 - 0.5) * 2.0;
            let ny = (y as f32 / height as f32 - 0.5) * 2.0;
            
            // Create a simple depth pattern (sphere in center)
            let r = (nx * nx + ny * ny).sqrt();
            if r < 0.8 {
                let depth = 1000.0 + (1.0 - r) * 200.0; // Depth in mm
                depth_image[y * width + x] = depth;
            }
        }
    }
    
    (source_cloud, target_cloud, depth_image)
} 