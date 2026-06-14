//! Point cloud colorization from registered RGB images.
//!
//! A "registered" RGB image is one whose camera pose (position and orientation
//! relative to the point cloud coordinate frame) is known.  Given the camera
//! intrinsics and extrinsics, every 3-D point can be projected onto the image
//! plane and the corresponding pixel color assigned to it.
//!
//! # Pipeline
//!
//! 1. Transform each 3-D point from world space into camera space using the
//!    provided world-to-camera isometry.
//! 2. Project the camera-space point onto the image plane with the pinhole
//!    camera model.
//! 3. Discard points that lie behind the camera (negative depth) or whose
//!    projected pixel falls outside the image.
//! 4. Sample the pixel color — nearest-neighbour or bilinear — and write it
//!    into the output [`ColoredPoint3f`].
//!
//! # Multi-image support
//!
//! [`colorize_from_images`] accepts a slice of `(image, intrinsics, pose)`
//! triples.  Each 3-D point is colored by the *first* image that can see it.
//! Images should therefore be supplied in order of preference (e.g. most
//! frontal camera first).

use nalgebra::Isometry3;
use rayon::prelude::*;
use threecrate_core::{ColoredPoint3f, Error, Point3f, PointCloud, Result};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Pinhole camera intrinsic parameters.
#[derive(Debug, Clone, Copy)]
pub struct CameraIntrinsics {
    /// Horizontal focal length in pixels.
    pub fx: f32,
    /// Vertical focal length in pixels.
    pub fy: f32,
    /// Horizontal principal-point offset (pixels from left edge).
    pub cx: f32,
    /// Vertical principal-point offset (pixels from top edge).
    pub cy: f32,
}

/// A borrowed, tightly-packed RGB image (row-major, 3 bytes per pixel).
#[derive(Debug, Clone, Copy)]
pub struct RgbImageView<'a> {
    /// Raw pixel data: `[R, G, B, R, G, B, …]`, row-major.
    pub data: &'a [u8],
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl<'a> RgbImageView<'a> {
    /// Construct a new view, validating that `data` is large enough.
    pub fn new(data: &'a [u8], width: u32, height: u32) -> Result<Self> {
        let expected = (width as usize) * (height as usize) * 3;
        if data.len() < expected {
            return Err(Error::InvalidData(format!(
                "RgbImageView: buffer too small — expected {} bytes for {}×{} RGB image, got {}",
                expected,
                width,
                height,
                data.len()
            )));
        }
        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// Sample pixel color using nearest-neighbour interpolation.
    #[inline]
    fn sample_nearest(&self, u: f32, v: f32) -> Option<[u8; 3]> {
        let x = u.round() as i32;
        let y = v.round() as i32;
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return None;
        }
        let idx = (y as usize * self.width as usize + x as usize) * 3;
        Some([self.data[idx], self.data[idx + 1], self.data[idx + 2]])
    }

    /// Sample pixel color using bilinear interpolation.
    #[inline]
    fn sample_bilinear(&self, u: f32, v: f32) -> Option<[u8; 3]> {
        let w = self.width as i32;
        let h = self.height as i32;

        let x0 = u.floor() as i32;
        let y0 = v.floor() as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        if x0 < 0 || y0 < 0 || x1 >= w || y1 >= h {
            // Fall back to nearest when the bilinear neighbourhood clips the border.
            return self.sample_nearest(u, v);
        }

        let tx = u - u.floor();
        let ty = v - v.floor();

        let fetch = |x: i32, y: i32| -> [f32; 3] {
            let idx = (y as usize * self.width as usize + x as usize) * 3;
            [
                self.data[idx] as f32,
                self.data[idx + 1] as f32,
                self.data[idx + 2] as f32,
            ]
        };

        let c00 = fetch(x0, y0);
        let c10 = fetch(x1, y0);
        let c01 = fetch(x0, y1);
        let c11 = fetch(x1, y1);

        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let r = lerp(lerp(c00[0], c10[0], tx), lerp(c01[0], c11[0], tx), ty);
        let g = lerp(lerp(c00[1], c10[1], tx), lerp(c01[1], c11[1], tx), ty);
        let b = lerp(lerp(c00[2], c10[2], tx), lerp(c01[2], c11[2], tx), ty);

        Some([r.round() as u8, g.round() as u8, b.round() as u8])
    }
}

/// Interpolation strategy when sampling pixel colors.
#[derive(Debug, Clone, Copy, Default)]
pub enum InterpolationMode {
    /// Round projected coordinates to the nearest integer pixel.
    NearestNeighbor,
    /// Bilinear interpolation across the four surrounding pixels.
    #[default]
    Bilinear,
}

/// Configuration for [`colorize_point_cloud`].
#[derive(Debug, Clone)]
pub struct ColorizationConfig {
    /// Color assigned to points that cannot be projected into any image.
    /// Defaults to `[128, 128, 128]` (mid-grey).
    pub default_color: [u8; 3],
    /// Pixel interpolation method.
    pub interpolation: InterpolationMode,
    /// Minimum depth (in camera space, metres) for a point to be considered
    /// visible.  Points closer than this are treated as behind the camera.
    pub min_depth: f32,
}

impl Default for ColorizationConfig {
    fn default() -> Self {
        Self {
            default_color: [128, 128, 128],
            interpolation: InterpolationMode::default(),
            min_depth: 1e-4,
        }
    }
}

/// Summary statistics returned by [`colorize_point_cloud`].
#[derive(Debug, Clone)]
pub struct ColorizationResult {
    /// Colored point cloud.
    pub cloud: PointCloud<ColoredPoint3f>,
    /// Number of points that were successfully colored from an image.
    pub colored_count: usize,
    /// Number of points that received the default color (no image coverage).
    pub uncolored_count: usize,
}

// ---------------------------------------------------------------------------
// Core projection helpers
// ---------------------------------------------------------------------------

/// Project a single world-space point into a camera and return the image-space
/// coordinates `(u, v)`.  Returns `None` if the point is behind the camera.
#[inline]
fn project_point(
    point: &Point3f,
    world_to_camera: &Isometry3<f32>,
    intrinsics: &CameraIntrinsics,
    min_depth: f32,
) -> Option<(f32, f32)> {
    let p_cam = world_to_camera * point;
    let z = p_cam.z;
    if z < min_depth {
        return None;
    }
    let u = intrinsics.fx * p_cam.x / z + intrinsics.cx;
    let v = intrinsics.fy * p_cam.y / z + intrinsics.cy;
    Some((u, v))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Colorize a point cloud from a single registered RGB image.
///
/// # Arguments
///
/// * `cloud`            – Input point cloud (position only).
/// * `image`            – Registered RGB image.
/// * `intrinsics`       – Pinhole camera intrinsics.
/// * `world_to_camera`  – Rigid transform that maps world-space points into
///                        the camera coordinate frame (X right, Y down, Z forward).
/// * `config`           – Colorization options.
///
/// # Returns
///
/// A [`ColorizationResult`] containing the colored cloud and coverage statistics.
pub fn colorize_point_cloud(
    cloud: &PointCloud<Point3f>,
    image: &RgbImageView<'_>,
    intrinsics: &CameraIntrinsics,
    world_to_camera: &Isometry3<f32>,
    config: &ColorizationConfig,
) -> Result<ColorizationResult> {
    let colored_points: Vec<(ColoredPoint3f, bool)> = cloud
        .points
        .par_iter()
        .map(|point| {
            let color = project_and_sample(point, image, intrinsics, world_to_camera, config);
            let hit = color != config.default_color;
            (
                ColoredPoint3f {
                    position: *point,
                    color,
                },
                hit,
            )
        })
        .collect();

    let colored_count = colored_points.iter().filter(|(_, hit)| *hit).count();
    let uncolored_count = colored_points.len() - colored_count;
    let cloud = PointCloud::from_points(colored_points.into_iter().map(|(p, _)| p).collect());

    Ok(ColorizationResult {
        cloud,
        colored_count,
        uncolored_count,
    })
}

/// Colorize a point cloud from multiple registered RGB images.
///
/// Each point is colored by the **first** image in `sources` that can see it.
/// Supply images in decreasing priority order (most preferred camera first).
///
/// # Arguments
///
/// * `cloud`   – Input point cloud.
/// * `sources` – Slice of `(image, intrinsics, world_to_camera)` triples.
/// * `config`  – Colorization options.
pub fn colorize_from_images(
    cloud: &PointCloud<Point3f>,
    sources: &[(RgbImageView<'_>, CameraIntrinsics, Isometry3<f32>)],
    config: &ColorizationConfig,
) -> Result<ColorizationResult> {
    if sources.is_empty() {
        return Err(Error::InvalidData(
            "colorize_from_images: sources slice must not be empty".into(),
        ));
    }

    let colored_points: Vec<(ColoredPoint3f, bool)> = cloud
        .points
        .par_iter()
        .map(|point| {
            for (image, intrinsics, world_to_camera) in sources {
                let color = project_and_sample(point, image, intrinsics, world_to_camera, config);
                if color != config.default_color {
                    return (
                        ColoredPoint3f {
                            position: *point,
                            color,
                        },
                        true,
                    );
                }
            }
            (
                ColoredPoint3f {
                    position: *point,
                    color: config.default_color,
                },
                false,
            )
        })
        .collect();

    let colored_count = colored_points.iter().filter(|(_, hit)| *hit).count();
    let uncolored_count = colored_points.len() - colored_count;
    let cloud = PointCloud::from_points(colored_points.into_iter().map(|(p, _)| p).collect());

    Ok(ColorizationResult {
        cloud,
        colored_count,
        uncolored_count,
    })
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

/// Project `point` into `image` and return the sampled color, or
/// `config.default_color` if the point is not visible.
#[inline]
fn project_and_sample(
    point: &Point3f,
    image: &RgbImageView<'_>,
    intrinsics: &CameraIntrinsics,
    world_to_camera: &Isometry3<f32>,
    config: &ColorizationConfig,
) -> [u8; 3] {
    let Some((u, v)) = project_point(point, world_to_camera, intrinsics, config.min_depth) else {
        return config.default_color;
    };
    let sampled = match config.interpolation {
        InterpolationMode::NearestNeighbor => image.sample_nearest(u, v),
        InterpolationMode::Bilinear => image.sample_bilinear(u, v),
    };
    sampled.unwrap_or(config.default_color)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Isometry3, Translation3, UnitQuaternion};

    /// Identity pose: camera sits at the origin looking down +Z.
    fn identity_pose() -> Isometry3<f32> {
        Isometry3::identity()
    }

    /// Simple 4×4 all-red image.
    fn red_image() -> Vec<u8> {
        vec![255u8, 0, 0].repeat(4 * 4)
    }

    fn intrinsics_4x4() -> CameraIntrinsics {
        // For a 4×4 image, place the principal point at the centre (1.5, 1.5).
        CameraIntrinsics {
            fx: 2.0,
            fy: 2.0,
            cx: 1.5,
            cy: 1.5,
        }
    }

    #[test]
    fn test_single_point_in_view() {
        // A point at (0,0,1) should project to the centre of a 4×4 red image.
        let data = red_image();
        let image = RgbImageView::new(&data, 4, 4).unwrap();
        let intrinsics = intrinsics_4x4();
        let pose = identity_pose();

        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 1.0)]);
        let config = ColorizationConfig::default();
        let result = colorize_point_cloud(&cloud, &image, &intrinsics, &pose, &config).unwrap();

        assert_eq!(result.colored_count, 1);
        assert_eq!(result.cloud.points[0].color, [255, 0, 0]);
    }

    #[test]
    fn test_point_behind_camera() {
        // A point at z=-1 should be invisible and receive the default color.
        let data = red_image();
        let image = RgbImageView::new(&data, 4, 4).unwrap();
        let intrinsics = intrinsics_4x4();
        let pose = identity_pose();

        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, -1.0)]);
        let config = ColorizationConfig {
            default_color: [0, 0, 0],
            ..Default::default()
        };
        let result = colorize_point_cloud(&cloud, &image, &intrinsics, &pose, &config).unwrap();

        assert_eq!(result.uncolored_count, 1);
        assert_eq!(result.cloud.points[0].color, [0, 0, 0]);
    }

    #[test]
    fn test_point_out_of_frame() {
        // A point far to the side should project outside the 4×4 image.
        let data = red_image();
        let image = RgbImageView::new(&data, 4, 4).unwrap();
        let intrinsics = intrinsics_4x4();
        let pose = identity_pose();

        let cloud = PointCloud::from_points(vec![Point3f::new(1000.0, 0.0, 1.0)]);
        let config = ColorizationConfig {
            default_color: [1, 2, 3],
            ..Default::default()
        };
        let result = colorize_point_cloud(&cloud, &image, &intrinsics, &pose, &config).unwrap();

        assert_eq!(result.uncolored_count, 1);
        assert_eq!(result.cloud.points[0].color, [1, 2, 3]);
    }

    #[test]
    fn test_multi_image_first_wins() {
        // Two images: red and blue.  The first (red) should win for a visible point.
        let red = red_image();
        let blue = vec![0u8, 0, 255].repeat(4 * 4);
        let intrinsics = intrinsics_4x4();
        let pose = identity_pose();

        let img_red = RgbImageView::new(&red, 4, 4).unwrap();
        let img_blue = RgbImageView::new(&blue, 4, 4).unwrap();

        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 1.0)]);
        let config = ColorizationConfig::default();
        let sources = vec![(img_red, intrinsics, pose), (img_blue, intrinsics, pose)];
        let result = colorize_from_images(&cloud, &sources, &config).unwrap();
        assert_eq!(result.cloud.points[0].color, [255, 0, 0]);
    }

    #[test]
    fn test_camera_translated_along_x() {
        // Camera moved 1 unit to the right in world space.
        // The world-to-camera transform must account for this translation.
        let data = red_image();
        let image = RgbImageView::new(&data, 4, 4).unwrap();
        let intrinsics = intrinsics_4x4();

        // Camera at world position (1, 0, 0) looking down +Z.
        // world_to_camera = inverse of camera_to_world
        let camera_to_world =
            Isometry3::from_parts(Translation3::new(1.0, 0.0, 0.0), UnitQuaternion::identity());
        let world_to_camera = camera_to_world.inverse();

        // A point at (1, 0, 1) is directly in front of this camera.
        let cloud = PointCloud::from_points(vec![Point3f::new(1.0, 0.0, 1.0)]);
        let config = ColorizationConfig::default();
        let result =
            colorize_point_cloud(&cloud, &image, &intrinsics, &world_to_camera, &config).unwrap();

        assert_eq!(result.colored_count, 1);
        assert_eq!(result.cloud.points[0].color, [255, 0, 0]);
    }

    #[test]
    fn test_rgb_image_view_too_small() {
        let result = RgbImageView::new(&[0u8; 10], 4, 4);
        assert!(result.is_err());
    }
}
