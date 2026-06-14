//! Point Cloud Colorization Example
//!
//! Demonstrates colorizing a 3-D point cloud from one or more registered RGB
//! images, as implemented for issue #99.
//!
//! A "registered" image is one whose camera intrinsics and extrinsics (pose)
//! relative to the point cloud coordinate frame are known.  The algorithm
//! projects each point onto the image plane and samples the corresponding
//! pixel color.

use nalgebra::{Isometry3, Translation3, UnitQuaternion};
use threecrate_algorithms::{
    colorize_from_images, colorize_point_cloud, CameraIntrinsics, ColorizationConfig,
    InterpolationMode, RgbImageView,
};
use threecrate_core::{Point3f, PointCloud};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Point Cloud Colorization Example ===\n");

    // ------------------------------------------------------------------
    // 1. Build a synthetic point cloud – a 5×5 grid on the Z=2 plane.
    // ------------------------------------------------------------------
    let cloud = make_grid_cloud(5, 5, 2.0);
    println!("Point cloud: {} points", cloud.len());

    // ------------------------------------------------------------------
    // 2. Create a synthetic 64×64 RGB image with coloured quadrants.
    // ------------------------------------------------------------------
    let (img_data, img_w, img_h) = make_quadrant_image(64, 64);
    let image = RgbImageView::new(&img_data, img_w, img_h)?;

    // ------------------------------------------------------------------
    // 3. Define camera intrinsics.
    //    Focal length chosen so the 5×5 grid just fits inside the image.
    // ------------------------------------------------------------------
    let intrinsics = CameraIntrinsics {
        fx: 32.0,
        fy: 32.0,
        cx: 32.0, // principal point at image centre
        cy: 32.0,
    };

    // ------------------------------------------------------------------
    // 4. Camera pose: identity → camera sits at the origin looking down +Z.
    //    (world_to_camera = identity means no transformation is needed)
    // ------------------------------------------------------------------
    let world_to_camera = Isometry3::identity();

    // ------------------------------------------------------------------
    // 5. Colorize – single image, bilinear interpolation (default).
    // ------------------------------------------------------------------
    let config = ColorizationConfig::default();
    let result = colorize_point_cloud(&cloud, &image, &intrinsics, &world_to_camera, &config)?;

    println!(
        "Single-image colorization (bilinear): colored={}, uncolored={}",
        result.colored_count, result.uncolored_count
    );

    // Print a few sample colors.
    for (_i, pt) in result.cloud.points.iter().enumerate().take(5) {
        println!(
            "  point ({:.1},{:.1},{:.1}) → color {:?}",
            pt.position.x, pt.position.y, pt.position.z, pt.color
        );
    }

    // ------------------------------------------------------------------
    // 6. Nearest-neighbour variant.
    // ------------------------------------------------------------------
    let nn_config = ColorizationConfig {
        interpolation: InterpolationMode::NearestNeighbor,
        ..Default::default()
    };
    let nn_result =
        colorize_point_cloud(&cloud, &image, &intrinsics, &world_to_camera, &nn_config)?;
    println!(
        "\nSingle-image colorization (nearest-neighbour): colored={}, uncolored={}",
        nn_result.colored_count, nn_result.uncolored_count
    );

    // ------------------------------------------------------------------
    // 7. Multi-image colorization.
    //    Two cameras: one at the origin (image A) and one offset by +2 on X
    //    (image B, all-blue).  Each point is colored by the first camera that
    //    sees it; since camera A covers all points in this scene, every point
    //    should get a color from image A.
    // ------------------------------------------------------------------
    let (blue_data, _, _) = make_solid_image(64, 64, [0, 120, 215]);
    let img_blue = RgbImageView::new(&blue_data, 64, 64)?;

    // Camera B: positioned 2 units to the right in world space.
    let cam_b_to_world =
        Isometry3::from_parts(Translation3::new(2.0, 0.0, 0.0), UnitQuaternion::identity());
    let world_to_cam_b = cam_b_to_world.inverse();

    let sources = vec![
        (image, intrinsics, world_to_camera), // Camera A (quadrant image)
        (img_blue, intrinsics, world_to_cam_b), // Camera B (solid blue, fallback)
    ];
    let multi_result = colorize_from_images(&cloud, &sources, &config)?;
    println!(
        "\nMulti-image colorization (2 cameras): colored={}, uncolored={}",
        multi_result.colored_count, multi_result.uncolored_count
    );

    // ------------------------------------------------------------------
    // 8. Custom default color for uncolored points.
    // ------------------------------------------------------------------
    let partial_cloud = PointCloud::from_points(vec![
        Point3f::new(0.0, 0.0, 2.0),    // in view
        Point3f::new(0.0, 0.0, -5.0),   // behind camera
        Point3f::new(1000.0, 0.0, 2.0), // far off to the side
    ]);
    let partial_img_data = make_solid_image(64, 64, [255, 128, 0]).0;
    let partial_img = RgbImageView::new(&partial_img_data, 64, 64)?;
    let custom_config = ColorizationConfig {
        default_color: [50, 50, 50],
        ..Default::default()
    };
    let partial_result = colorize_point_cloud(
        &partial_cloud,
        &partial_img,
        &intrinsics,
        &Isometry3::identity(),
        &custom_config,
    )?;
    println!("\nPartial coverage (3 points, 1 in-view, 2 out-of-view):");
    println!(
        "  colored={}, uncolored={}",
        partial_result.colored_count, partial_result.uncolored_count
    );
    for pt in &partial_result.cloud.points {
        println!(
            "  ({:6.1},{:6.1},{:6.1}) → {:?}",
            pt.position.x, pt.position.y, pt.position.z, pt.color
        );
    }

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

// ---------------------------------------------------------------------------
// Synthetic data helpers
// ---------------------------------------------------------------------------

/// Build a W×H grid of points at depth `z`.
fn make_grid_cloud(w: usize, h: usize, z: f32) -> PointCloud<Point3f> {
    let mut points = Vec::with_capacity(w * h);
    for row in 0..h {
        for col in 0..w {
            let x = col as f32 - (w as f32 - 1.0) / 2.0;
            let y = row as f32 - (h as f32 - 1.0) / 2.0;
            points.push(Point3f::new(x, y, z));
        }
    }
    PointCloud::from_points(points)
}

/// Create a `width × height` image split into four coloured quadrants:
/// top-left = red, top-right = green, bottom-left = blue, bottom-right = yellow.
fn make_quadrant_image(width: u32, height: u32) -> (Vec<u8>, u32, u32) {
    let mut data = Vec::with_capacity((width * height * 3) as usize);
    let hw = width / 2;
    let hh = height / 2;
    for y in 0..height {
        for x in 0..width {
            let color: [u8; 3] = match (x < hw, y < hh) {
                (true, true) => [220, 50, 50],    // top-left     – red
                (false, true) => [50, 180, 50],   // top-right    – green
                (true, false) => [50, 50, 220],   // bottom-left  – blue
                (false, false) => [220, 200, 50], // bottom-right – yellow
            };
            data.extend_from_slice(&color);
        }
    }
    (data, width, height)
}

/// Create a solid-color image.
fn make_solid_image(width: u32, height: u32, color: [u8; 3]) -> (Vec<u8>, u32, u32) {
    let data = color
        .iter()
        .copied()
        .cycle()
        .take((width * height * 3) as usize)
        .collect();
    (data, width, height)
}
