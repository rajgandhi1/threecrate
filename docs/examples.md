# Examples

Runnable examples live in the [`examples/`](../examples/) directory. Run any with:

```bash
cargo run --example basic_usage
cargo run --example ransac_plane_example
cargo run --example global_registration
# ...
```

## Point cloud processing

```rust
use threecrate::prelude::*;

// Load
let cloud = read_point_cloud("scan.ply")?;

// Filter
let cloud = voxel_grid_filter(&cloud, 0.05)?;
let cloud = statistical_outlier_removal(&cloud, 10, 2.0)?;

// Estimate normals
let normals_cloud = estimate_normals(&cloud, 10)?;

// Register two scans
let result = icp(&source, &target, Default::default())?;

// Reconstruct surface
let mesh = auto_reconstruct(&normals_cloud)?;
println!("Reconstructed: {} triangles", mesh.face_count());

write_mesh("output.obj", &mesh)?;
```

## Segmentation

```rust
use threecrate::prelude::*;

// Fit a plane with RANSAC and extract inliers
let result = segment_plane(&cloud, 0.01, 1000)?;
println!("Plane: {:?}, {} inliers", result.model.coefficients, result.inliers.len());

// Euclidean cluster extraction
let config = EuclideanClusterConfig { tolerance: 0.02, min_cluster_size: 100, max_cluster_size: 25000 };
let clusters = extract_euclidean_clusters(&cloud, &config)?;
println!("{} clusters found", clusters.num_clusters());
```

## Mesh simplification and smoothing

```rust
use threecrate::prelude::*;
use threecrate_simplification::{QuadricErrorSimplifier, MeshSimplifier};

let simplified = QuadricErrorSimplifier::new().simplify(&mesh, 0.5)?;

let smooth = smooth_taubin(&mesh, &TaubinSmoothingConfig::default())?;
```

## GPU acceleration

```rust
use threecrate::prelude::*;

let gpu = GpuContext::new().await?;

let filtered   = gpu_voxel_grid_filter(&gpu, &cloud, 0.05).await?;
let registered = gpu_icp(&gpu, &source, &target).await?;
```

## Visualization

```rust
use threecrate::prelude::*;

show_point_cloud(&cloud)?;
show_mesh(&mesh)?;
```

## Python

```python
import numpy as np
import threecrate as tc

cloud = tc.read_point_cloud("scan.ply")
cloud = tc.voxel_downsample(cloud, voxel_size=0.02)
cloud = tc.remove_statistical_outliers(cloud)

# Segment dominant plane
result = tc.segment_plane(cloud, threshold=0.01)
plane_cloud = result.inlier_cloud(cloud)

# Cluster remaining points
clusters = tc.extract_clusters(cloud)
print(f"{len(clusters)} objects found")

# Reconstruct surface
normal_cloud = tc.estimate_normals(cloud)
mesh = tc.poisson_reconstruct(normal_cloud)
mesh = tc.smooth_mesh_taubin(mesh, iterations=10)
mesh = tc.simplify_mesh(mesh, reduction_ratio=0.5)
tc.write_mesh(mesh, "output.ply")
```

For the full Python API see [`threecrate-python/README.md`](../threecrate-python/README.md).
