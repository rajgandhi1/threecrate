---
name: threecrate
description: Process 3D point clouds and meshes (PLY, PCD, XYZ, CSV, OBJ) with the threecrate Python library. Use for inspecting scans, downsampling, removing outliers, estimating normals, aligning scans (ICP, GICP, global registration), finding planes and objects, building meshes (Poisson, ball pivoting), and simplifying or smoothing meshes.
---

# threecrate

threecrate is a fast Rust library for point clouds and meshes with Python bindings (`pip install threecrate`). This skill drives it from short Python scripts.

If the `threecrate` MCP server is connected (tools such as `inspect_point_cloud`, `register`, `reconstruct`), prefer those tools. Use this skill when it isn't.

## Setup

Install into a virtual environment, never the system Python:

```bash
python -m venv .venv
.venv/bin/pip install threecrate numpy
```

Then run scripts with `.venv/bin/python`. On Windows the folder is `Scripts`, not `bin`: use `.venv/Scripts/pip` and `.venv/Scripts/python` in Git Bash or PowerShell, or `.venv\Scripts\python` in cmd.

`orient_normals` and the `poisson_reconstruct` options below need threecrate 0.10.0 or newer, and `analyze_mesh` needs 0.12.0 or newer. Check with `pip show threecrate`.

## Workflow

1. **Inspect first.** Run `scripts/inspect_cloud.py <file>` from this skill's folder. It prints the point count, bounds and median point spacing, plus suggested parameters. Every size parameter below depends on the spacing and the units, so don't guess them.
2. **Clean:** remove outliers, then downsample if the cloud is large or dense.
3. **Do the task:** register, segment or reconstruct.
4. **Write results to new files.** Never overwrite the user's input. Report what changed (point counts, fit error, face counts).

## Choosing parameters

Base sizes on the median spacing `s` from the inspect script:

| Parameter | Start with |
|---|---|
| `voxel_downsample` voxel size | `2*s` (light) to `5*s` (strong) |
| `remove_radius_outliers` radius | `3*s`, `min_neighbors=5` |
| `extract_clusters` tolerance | `3*s` |
| `segment_plane` threshold | `1*s` to `2*s` |
| `ball_pivoting_reconstruct` radius | `2*s` |
| `poisson_reconstruct` `trim_distance` | `2*s` to `3*s`, open scans only |

## Recipes

```python
import numpy as np
import threecrate as tc

cloud = tc.read_point_cloud("scan.ply")              # PLY, PCD, XYZ, CSV
pts = cloud.to_numpy()                               # (N, 3) float32
cloud = tc.PointCloud(pts)                           # and back
s = 0.01  # REPLACE with median_spacing from scripts/inspect_cloud.py. Every size below
          # scales with it, and a value far too small makes the filters delete real points.

# Clean. Statistical removal handles general noise; follow it with radius
# removal to drop isolated strays it leaves behind.
cloud = tc.remove_statistical_outliers(cloud, k_neighbors=20, std_ratio=2.0)
cloud = tc.remove_radius_outliers(cloud, radius=3 * s, min_neighbors=5)  # s: median spacing
cloud = tc.voxel_downsample(cloud, 2 * s)

# Normals (oriented toward a viewpoint above the cloud; fine for one scan)
normals = tc.estimate_normals(cloud, k_neighbors=10)
# Full 3D object? Make normals agree all the way around (outward):
normals = tc.orient_normals(normals, k_neighbors=10)

# Align source to target. Returns a 4x4 matrix (source -> target).
result = tc.icp(source, target, max_iterations=50)             # needs a rough initial alignment
result = tc.gicp(source, target, max_correspondence_distance=0.1)  # more robust
T = result.transformation()
aligned = tc.transform_point_cloud(source, T)
print(result.mse, result.converged)

# No initial alignment needed: feature matching, then ICP
result = tc.global_registration(source, target)
print(result.icp_mse, result.inlier_ratio)

# Floor or wall, then separate objects
plane = tc.segment_plane(cloud, threshold=1.5 * s)
a, b, c, d = plane.plane_coefficients()              # ax + by + cz + d = 0
if c < 0:                                            # the sign varies between runs;
    a, b, c, d = -a, -b, -c, -d                      # make the normal point up
pts = cloud.to_numpy()                               # the same cloud the indices refer to
on_plane = np.zeros(len(pts), dtype=bool)
on_plane[plane.inlier_indices()] = True
rest = pts[~on_plane]
above = rest[rest @ np.array([a, b, c]) + d > 0]     # only what sits on top of the floor
objects = tc.extract_clusters(tc.PointCloud(above), tolerance=3 * s, min_cluster_size=100)
# extract_clusters returns a list of PointClouds, largest first
for i, obj in enumerate(objects):
    tc.write_point_cloud(obj, f"object_{i}.ply")
tc.write_point_cloud(tc.PointCloud(np.vstack([o.to_numpy() for o in objects])), "objects.ply")

# Meshes
quick = tc.reconstruct(cloud)                                 # picks a method
detailed = tc.ball_pivoting_reconstruct(cloud, radius=2 * s)  # keeps detail, may leave holes
open_scan = tc.poisson_reconstruct(normals, depth=7, trim_distance=2.5 * s)  # one-sided scan
mesh = tc.poisson_reconstruct(normals, depth=7)               # smooth, watertight
mesh = tc.simplify_mesh(mesh, reduction_ratio=0.5)            # fails on meshes full of holes
mesh = tc.smooth_mesh_taubin(mesh, iterations=10)
print(mesh.vertex_count, mesh.face_count)

# Health and measuring: holes, pieces, watertight, size, area, volume, printable
report = tc.analyze_mesh(mesh)
print(report["printable"], report["problems"], report["volume"], report["size"])

tc.write_point_cloud(cloud, "scan.clean.ply")
tc.write_mesh(mesh, "scan.mesh.ply")                 # PLY or OBJ
```

## Good to know

- Poisson `depth` runs from 3 to 8. Depth 7 takes about a second; each step up is about 8x slower and finer.
- Poisson always closes the surface. On a scan that only sees one side, set `trim_distance`, or the result is a closed blob.
- `simplify_mesh` raises an error when the mesh is mostly holes (common after ball pivoting). Pass `preserve_boundary=False`, or build the mesh with Poisson instead.
- Point-to-point `icp` can lock onto the wrong match when the clouds start far apart. Use `global_registration` first, then refine with `gicp`.
- Downsample both clouds before registration for speed. The transform still applies to the full clouds.
- Very large clouds (tens of millions of points) need a lot of memory. Downsample before meshing.
