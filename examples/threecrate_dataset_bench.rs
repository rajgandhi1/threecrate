use anyhow::{anyhow, Context, Result};
use image::ImageReader;
use nalgebra::{Isometry3, Translation3, UnitQuaternion};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;
use threecrate_algorithms::{
    estimate_normals, icp_point_to_point, multiscale_icp_point_to_point, voxel_grid_filter,
    IcpScaleLevel, MultiScaleIcpConfig,
};
use threecrate_core::{Point3f, PointCloud};
use threecrate_gpu::{
    gpu_find_k_nearest_batch, gpu_icp, gpu_radius_outlier_removal, gpu_remove_statistical_outliers,
    gpu_voxel_grid_filter, GpuContext,
};
use threecrate_io::read_point_cloud;

#[derive(Debug)]
struct Args {
    task: String,
    dataset: String,
    source: PathBuf,
    target: Option<PathBuf>,
    iterations: usize,
    warmups: usize,
    max_points: Option<usize>,
    voxel_size: f32,
    max_icp_iters: usize,
    convergence: f32,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            task: "icp".to_string(),
            dataset: "dataset".to_string(),
            source: PathBuf::new(),
            target: None,
            iterations: 5,
            warmups: 1,
            max_points: Some(20_000),
            voxel_size: 0.2,
            max_icp_iters: 20,
            convergence: 1e-5,
        }
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let mut source = load_source(&args.source)?;
    limit_points(&mut source, args.max_points);

    let mut target = match &args.target {
        Some(path) => {
            let mut cloud = load_source(path)?;
            limit_points(&mut cloud, args.max_points);
            cloud
        }
        None => transformed_target(&source),
    };
    limit_points(&mut target, args.max_points);

    let gpu_context = if args.task.starts_with("gpu_") {
        Some(pollster::block_on(GpuContext::new())?)
    } else {
        None
    };

    for _ in 0..args.warmups {
        run_task(&args, &source, &target, gpu_context.as_ref())?;
    }

    let mut times = Vec::with_capacity(args.iterations);
    let mut output_points = 0usize;
    let mut detail = String::new();
    for _ in 0..args.iterations {
        let start = Instant::now();
        let outcome = run_task(&args, &source, &target, gpu_context.as_ref())?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        output_points = outcome.output_points;
        detail = outcome.detail;
        times.push(elapsed_ms);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = times[times.len() / 2];
    let min_ms = times[0];
    let mean_ms = times.iter().sum::<f64>() / times.len() as f64;

    println!(
        "library,task,dataset,source_points,target_points,output_points,iterations,median_ms,min_ms,mean_ms,detail"
    );
    println!(
        "{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{}",
        if args.task.starts_with("gpu_") {
            "ThreeCrateGPU"
        } else {
            "ThreeCrateCPU"
        },
        args.task,
        csv_escape(&args.dataset),
        source.len(),
        target.len(),
        output_points,
        args.iterations,
        median_ms,
        min_ms,
        mean_ms,
        csv_escape(&detail)
    );

    Ok(())
}

struct Outcome {
    output_points: usize,
    detail: String,
}

fn run_task(
    args: &Args,
    source: &PointCloud<Point3f>,
    target: &PointCloud<Point3f>,
    gpu_context: Option<&GpuContext>,
) -> Result<Outcome> {
    match args.task.as_str() {
        "read" => {
            let cloud = load_source(&args.source)?;
            Ok(Outcome {
                output_points: cloud.len(),
                detail: if args.source.is_dir() {
                    "read_tum_depth_frame".to_string()
                } else {
                    "read_point_cloud".to_string()
                },
            })
        }
        "voxel" => {
            let filtered = voxel_grid_filter(source, args.voxel_size)?;
            Ok(Outcome {
                output_points: filtered.len(),
                detail: format!("voxel_size={}", args.voxel_size),
            })
        }
        "normals" => {
            let normals = estimate_normals(source, 10)?;
            Ok(Outcome {
                output_points: normals.len(),
                detail: "k=10".to_string(),
            })
        }
        "icp" => {
            let result = icp_point_to_point(
                source,
                target,
                Isometry3::identity(),
                args.max_icp_iters,
                args.convergence,
                None,
            )?;
            Ok(Outcome {
                output_points: result.correspondences.len(),
                detail: format!(
                    "icp_iters={},converged={},mse={:.6}",
                    result.iterations, result.converged, result.mse
                ),
            })
        }
        "multiscale_icp" => {
            let config = MultiScaleIcpConfig {
                levels: vec![
                    IcpScaleLevel {
                        voxel_size: 0.20,
                        max_iterations: args.max_icp_iters.min(10),
                        max_correspondence_distance: Some(0.50),
                    },
                    IcpScaleLevel {
                        voxel_size: 0.10,
                        max_iterations: args.max_icp_iters.min(10),
                        max_correspondence_distance: Some(0.25),
                    },
                    IcpScaleLevel {
                        voxel_size: 0.05,
                        max_iterations: args.max_icp_iters,
                        max_correspondence_distance: Some(0.15),
                    },
                ],
                final_refinement_iterations: args.max_icp_iters,
                final_max_correspondence_distance: Some(0.10),
                convergence_threshold: args.convergence,
            };
            let result =
                multiscale_icp_point_to_point(source, target, Isometry3::identity(), &config)?;
            Ok(Outcome {
                output_points: result.correspondences.len(),
                detail: format!(
                    "levels=3,total_iters={},converged={},mse={:.6}",
                    result.iterations, result.converged, result.mse
                ),
            })
        }
        "gpu_voxel" => {
            let gpu = require_gpu(gpu_context)?;
            let filtered = pollster::block_on(gpu_voxel_grid_filter(gpu, source, args.voxel_size))?;
            Ok(Outcome {
                output_points: filtered.len(),
                detail: format!("voxel_size={}", args.voxel_size),
            })
        }
        "gpu_icp" => {
            let gpu = require_gpu(gpu_context)?;
            let transform = pollster::block_on(gpu_icp(
                gpu,
                source,
                target,
                args.max_icp_iters,
                args.convergence,
                1.0,
            ))?;
            Ok(Outcome {
                output_points: source.len(),
                detail: format!(
                    "translation_norm={:.6},rotation_angle={:.6}",
                    transform.translation.vector.norm(),
                    transform.rotation.angle()
                ),
            })
        }
        "gpu_radius_outlier" => {
            let gpu = require_gpu(gpu_context)?;
            let filtered = pollster::block_on(gpu_radius_outlier_removal(gpu, source, 0.2, 3))?;
            Ok(Outcome {
                output_points: filtered.len(),
                detail: "radius=0.2,min_neighbors=3".to_string(),
            })
        }
        "gpu_statistical_outlier" => {
            let gpu = require_gpu(gpu_context)?;
            let filtered =
                pollster::block_on(gpu_remove_statistical_outliers(gpu, source, 10, 1.0))?;
            Ok(Outcome {
                output_points: filtered.len(),
                detail: "k=10,std_dev_multiplier=1.0".to_string(),
            })
        }
        "gpu_normals" => {
            let gpu = require_gpu(gpu_context)?;
            let normals = pollster::block_on(gpu.compute_normals(&source.points, 10))?;
            Ok(Outcome {
                output_points: normals.len(),
                detail: "k=10".to_string(),
            })
        }
        "gpu_knn" => {
            let gpu = require_gpu(gpu_context)?;
            let query_count = source.len().min(256);
            let results = pollster::block_on(gpu_find_k_nearest_batch(
                gpu,
                &source.points,
                &source.points[..query_count],
                8,
            ))?;
            Ok(Outcome {
                output_points: results.iter().map(Vec::len).sum(),
                detail: format!("queries={},k=8", query_count),
            })
        }
        other => Err(anyhow!(
            "unsupported task: {other}; expected read, voxel, normals, icp, or gpu_*"
        )),
    }
}

fn require_gpu(gpu_context: Option<&GpuContext>) -> Result<&GpuContext> {
    gpu_context.ok_or_else(|| anyhow!("GPU context is required for gpu_* tasks"))
}

fn transformed_target(source: &PointCloud<Point3f>) -> PointCloud<Point3f> {
    let transform = Isometry3::from_parts(
        Translation3::new(0.05, -0.02, 0.01),
        UnitQuaternion::from_euler_angles(0.0, 0.0, 0.02),
    );
    PointCloud::from_points(source.points.iter().map(|p| transform * p).collect())
}

fn load_source(path: &PathBuf) -> Result<PointCloud<Point3f>> {
    if path.is_dir() {
        load_tum_depth_sequence(path)
    } else if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.to_ascii_lowercase().ends_with(".pcd.bin"))
    {
        load_nuscenes_pcd_bin(path)
    } else {
        read_point_cloud(path).with_context(|| format!("failed to read source {}", path.display()))
    }
}

fn load_nuscenes_pcd_bin(path: &PathBuf) -> Result<PointCloud<Point3f>> {
    let bytes =
        fs::read(path).with_context(|| format!("failed to read source {}", path.display()))?;
    const STRIDE: usize = 5 * std::mem::size_of::<f32>();
    if bytes.len() % STRIDE != 0 {
        return Err(anyhow!(
            "nuScenes pcd.bin file size {} is not a multiple of {} bytes",
            bytes.len(),
            STRIDE
        ));
    }

    let points = bytes
        .chunks_exact(STRIDE)
        .map(|chunk| {
            let x = f32::from_le_bytes(chunk[0..4].try_into().unwrap());
            let y = f32::from_le_bytes(chunk[4..8].try_into().unwrap());
            let z = f32::from_le_bytes(chunk[8..12].try_into().unwrap());
            Point3f::new(x, y, z)
        })
        .collect();
    Ok(PointCloud::from_points(points))
}

fn load_tum_depth_sequence(dir: &PathBuf) -> Result<PointCloud<Point3f>> {
    let depth_list = dir.join("depth.txt");
    let first_depth = first_sequence_entry(&depth_list)
        .with_context(|| format!("failed to read {}", depth_list.display()))?;
    let depth_path = dir.join(first_depth);
    let img = ImageReader::open(&depth_path)
        .with_context(|| format!("failed to open {}", depth_path.display()))?
        .decode()
        .with_context(|| format!("failed to decode {}", depth_path.display()))?
        .into_luma16();

    let (width, height) = img.dimensions();
    let fx = 525.0f32;
    let fy = 525.0f32;
    let cx = 319.5f32;
    let cy = 239.5f32;
    let depth_factor = 5000.0f32;
    let mut points = Vec::with_capacity((width * height) as usize / 4);

    for v in 0..height {
        for u in 0..width {
            let depth = img.get_pixel(u, v).0[0];
            if depth == 0 {
                continue;
            }
            let z = depth as f32 / depth_factor;
            let x = (u as f32 - cx) * z / fx;
            let y = (v as f32 - cy) * z / fy;
            points.push(Point3f::new(x, y, z));
        }
    }

    Ok(PointCloud::from_points(points))
}

fn first_sequence_entry(path: &PathBuf) -> Result<String> {
    let file = File::open(path)?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let _timestamp = parts
            .next()
            .ok_or_else(|| anyhow!("missing timestamp in {}", path.display()))?;
        let rel_path = parts
            .next()
            .ok_or_else(|| anyhow!("missing frame path in {}", path.display()))?;
        return Ok(rel_path.to_string());
    }
    Err(anyhow!("no sequence entries found in {}", path.display()))
}

fn limit_points(cloud: &mut PointCloud<Point3f>, max_points: Option<usize>) {
    if let Some(max_points) = max_points {
        cloud.points.truncate(max_points);
    }
}

fn parse_args() -> Result<Args> {
    let mut args = Args::default();
    let mut it = env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .ok_or_else(|| anyhow!("missing value for argument {flag}"))?;
        match flag.as_str() {
            "--task" => args.task = value,
            "--dataset" => args.dataset = value,
            "--source" => args.source = PathBuf::from(value),
            "--target" => args.target = Some(PathBuf::from(value)),
            "--iterations" => args.iterations = value.parse()?,
            "--warmups" => args.warmups = value.parse()?,
            "--max-points" => args.max_points = parse_optional_usize(&value)?,
            "--voxel-size" => args.voxel_size = value.parse()?,
            "--max-icp-iters" => args.max_icp_iters = value.parse()?,
            "--convergence" => args.convergence = value.parse()?,
            other => return Err(anyhow!("unknown argument: {other}")),
        }
    }

    if args.source.as_os_str().is_empty() {
        return Err(anyhow!("--source is required"));
    }
    if args.iterations == 0 {
        return Err(anyhow!("--iterations must be greater than zero"));
    }
    Ok(args)
}

fn parse_optional_usize(value: &str) -> Result<Option<usize>> {
    if value.eq_ignore_ascii_case("all") || value == "0" {
        Ok(None)
    } else {
        Ok(Some(value.parse()?))
    }
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}
