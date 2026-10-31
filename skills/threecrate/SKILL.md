---
name: threecrate
description: Process 3D scans, point clouds and meshes (PLY, PCD, XYZ, CSV, OBJ, STL) with the threecrate Python library. Use to clean a scan (outliers, downsampling, floor, objects), align scans, make a scan or mesh 3D printable (holes, missing base, flipped faces, loose bits), measure size, area, volume or heights, or build and simplify meshes.
---

# threecrate

threecrate is a fast Rust library for point clouds and meshes with Python bindings (`pip install threecrate`). This skill drives it from short Python scripts.

If the `threecrate` MCP server is connected (tools such as `inspect_point_cloud`, `repair_for_print`, `render`), prefer those tools. Use this skill when it isn't.

## Setup

Install into a virtual environment, never the system Python:

```bash
python -m venv .venv
.venv/bin/pip install threecrate numpy
```

Then run scripts with `.venv/bin/python`. On Windows the folder is `Scripts`, not `bin`: use `.venv/Scripts/pip` and `.venv/Scripts/python` in Git Bash or PowerShell, or `.venv\Scripts\python` in cmd.

`analyze_mesh`, `render` and `repair_for_print` need threecrate 0.12.1 or newer (0.12.1 writes printable meshes in millimetres). Check with `pip show threecrate`.

## Workflow

1. **Inspect first.**
   - Point cloud: run `scripts/inspect_cloud.py <file>` from this skill's folder. It prints the point count, bounds, median point spacing `s` and suggested parameters.
   - Mesh: `tc.analyze_mesh(tc.read_mesh(path))` gives size, holes, pieces and whether it is printable.
   - Every size parameter scales with the spacing and the units, so read them off the inspect output.
2. **Open the guide for the task** and follow it:
   - Clean a scan, drop the floor, split out objects, normals: [guides/clean-scan.md](guides/clean-scan.md)
   - Align two or more scans: [guides/align-scans.md](guides/align-scans.md)
   - Make a scan or mesh printable, or build a mesh: [guides/make-printable.md](guides/make-printable.md)
   - Measure size, area, volume, heights or distances: [guides/measure.md](guides/measure.md)
3. **Look at the result.** `tc.render([before, after], "check.png")` draws both side by side from five views. Open the PNG before you report.
4. **Write results to new files.** Keep the user's input untouched. Report what changed, with numbers: point counts, fit error, face counts, sizes with units.

## Choosing parameters

Base sizes on the median spacing `s`:

| Parameter | Start with |
|---|---|
| `voxel_downsample` voxel size | `2*s` (light) to `5*s` (strong) |
| `remove_radius_outliers` radius | `3*s`, `min_neighbors=5` |
| `extract_clusters` tolerance | `3*s` |
| `segment_plane` threshold | `1*s` to `2*s` |
| `ball_pivoting_reconstruct` radius | `2*s` |
| `poisson_reconstruct` `trim_distance` | `2*s` to `3*s`, open scans only |

## Reading and writing

```python
import threecrate as tc

cloud = tc.read_point_cloud("scan.ply")   # PLY, PCD, XYZ, CSV
mesh = tc.read_mesh("part.stl")           # PLY, OBJ, STL
pts = cloud.to_numpy()                    # (N, 3) float32, and tc.PointCloud(pts) back
tc.write_point_cloud(cloud, "scan.clean.ply")
tc.write_mesh(mesh, "part.fixed.stl")
```

Very large clouds (tens of millions of points) need a lot of memory. Downsample first.
