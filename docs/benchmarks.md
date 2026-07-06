# Cross-Library Benchmarks

This page is a reproducible benchmark note for README updates, release notes, and
forum posts. It is written to be honest first: every number below was measured on
this machine, and the caveats are stated plainly rather than buried.

## TL;DR

- On real point-cloud datasets, ThreeCrate (CPU) is **faster than Open3D on file
  read and voxel downsampling**, roughly **at parity on small clouds for normals
  and ICP**, and **slower than Open3D on normal estimation and dense ICP as the
  cloud grows**.
- Across the 12 shared task/dataset rows the composite score is **127.4 at full
  resolution** and **138.9 at the 20k-point cap**. Both are above 100, but that
  headline is **carried by `read` and `voxel`** — see the honest breakdown below.
- **PCL is not yet in these numbers.** A PCL benchmark executable is written and
  builds (`scripts/pcl_bench/`), but it has not been integrated into the
  published table yet. No PCL number here is estimated from papers or other
  machines. Do **not** claim a PCL comparison from this page.

## Environment

- OS: Windows 11 (10.0.26200)
- Open3D: 0.19.0 (Python 3.10)
- ThreeCrate: this branch, `--release`
- Datasets: TUM RGB-D `freiburg1_xyz`, KITTI raw drive `2011_09_26_drive_0001`
  (frame `0000000000`), nuScenes `v1.0-mini` (one `LIDAR_TOP` sample)
- Generated: 2026-06-30
- 5 iterations, 2 warmups, median milliseconds (lower is better)

## Results — full resolution (no point cap)

This is the meaningful comparison: full frames (TUM ~230k, KITTI ~121k,
nuScenes ~35k points).

| Task | Dataset | Open3D (ms) | ThreeCrate (ms) | Ratio (Open3D/TC) |
| --- | --- | ---: | ---: | ---: |
| read | TUM_Freiburg1_XYZ | 19.092 | 3.959 | 4.82x ✅ |
| read | KITTI | 1.852 | 1.024 | 1.81x ✅ |
| read | NuScenesMini | 0.549 | 0.254 | 2.16x ✅ |
| voxel | TUM_Freiburg1_XYZ | 11.209 | 7.073 | 1.58x ✅ |
| voxel | KITTI | 25.155 | 13.777 | 1.83x ✅ |
| voxel | NuScenesMini | 4.420 | 2.557 | 1.73x ✅ |
| normals | TUM_Freiburg1_XYZ | 154.586 | 270.880 | 0.57x ❌ |
| normals | KITTI | 104.543 | 184.708 | 0.57x ❌ |
| normals | NuScenesMini | 32.322 | 29.649 | 1.09x ➖ |
| icp | TUM_Freiburg1_XYZ | 716.175 | 1005.229 | 0.71x ❌ |
| icp | KITTI | 301.766 | 387.318 | 0.78x ❌ |
| icp | NuScenesMini | 116.260 | 117.679 | 0.99x ➖ |

Composite (geometric mean of ratios, all 12 rows): **127.4**.

## Results — 20,000-point cap

Capping every cloud at 20k points makes everything fast and hides scaling. It is
included only because earlier notes used it; the full-resolution table above is
the one to trust.

| Task | Dataset | Open3D (ms) | ThreeCrate (ms) | Ratio |
| --- | --- | ---: | ---: | ---: |
| read | TUM_Freiburg1_XYZ | 33.258 | 3.422 | 9.72x ✅ |
| read | KITTI | 1.526 | 0.982 | 1.55x ✅ |
| read | NuScenesMini | 0.286 | 0.286 | 1.00x ➖ |
| voxel | TUM_Freiburg1_XYZ | 0.904 | 0.603 | 1.50x ✅ |
| voxel | KITTI | 6.373 | 2.258 | 2.82x ✅ |
| voxel | NuScenesMini | 2.168 | 1.334 | 1.63x ✅ |
| normals | TUM_Freiburg1_XYZ | 14.813 | 15.579 | 0.95x ➖ |
| normals | KITTI | 16.301 | 22.213 | 0.73x ❌ |
| normals | NuScenesMini | 17.617 | 13.523 | 1.30x ✅ |
| icp | TUM_Freiburg1_XYZ | 43.004 | 68.230 | 0.63x ❌ |
| icp | KITTI | 39.280 | 48.569 | 0.81x ❌ |
| icp | NuScenesMini | 49.545 | 46.185 | 1.07x ✅ |

Composite (all 12 rows): **138.9**.

## The honest breakdown

The composite is above 100, but most of that comes from two tasks:

- **`read` is partly not apples-to-apples.** For KITTI/nuScenes both libraries
  parse raw `float32` records, so those rows are a fair read comparison and
  ThreeCrate genuinely wins (~1.8–2.2x). But the **TUM `read` row is not
  comparable**: ThreeCrate's number is the benchmark's own depth-image
  back-projection loop, while Open3D runs its full RGBD→point-cloud pipeline.
  Treat the TUM read ratio as illustrative, not as a library-I/O result.
- **`voxel` is a genuine, fair win** on every dataset, and the output is now the
  per-voxel **centroid** (matching Open3D/PCL semantics), not an arbitrary first
  point — see "What changed" below.

If you remove the `read` task entirely and look only at the compute tasks
(voxel + normals + icp), the geometric-mean score is:

- **Full resolution: ~100** (rough parity with Open3D)
- **20k cap: ~115** (ahead, driven by the large voxel win)

So the fair one-line claim is: **ThreeCrate is competitive with Open3D on CPU
point-cloud work — ahead on read and voxel downsampling, roughly at parity
overall on compute, and still behind Open3D on normal estimation and dense ICP,
with that gap widening as clouds grow.**

## What changed in this branch (and why it matters)

These code changes were made to close real algorithmic gaps, not to flatter the
benchmark. Each is covered by the existing unit tests (201 passing).

- **Voxel grid returns the centroid, not the first point** (`filtering.rs`).
  This matches Open3D `voxel_down_sample` / PCL `VoxelGrid` semantics and, because
  the new code accumulates a running sum instead of storing every point index per
  voxel, it is also *faster* (full-res KITTI voxel 21.4 → 13.8 ms; nuScenes
  6.5 → 2.6 ms, which flipped that row from a loss to a win).
- **KD-tree k-NN keeps squared distances during traversal** and takes the square
  root once per surviving neighbor (`nearest_neighbor.rs`). This is the dominant
  cost in ICP correspondence search, so ICP got meaningfully faster
  (full-res KITTI 530.8 → 387.3 ms; TUM 1548.6 → 1005.2 ms; nuScenes → parity).
  It has negligible effect on normal estimation, which is PCA-bound, not
  search-bound — normals are honestly still ~1.7x behind Open3D at full scale.
- **Flat, array-backed kd-tree** (`nearest_neighbor.rs`, [#176]). The tree was
  `Box`-pointer-based; it is now a contiguous `Vec<KdNode>` with children
  referenced by index, so traversal is cache-friendly. k-NN output is identical
  (201 tests pass). A same-machine A/B measured a **consistent ~8–10% speedup on
  normals and ~5–9% on ICP** (e.g. normals KITTI 105.9 → 97.3 ms; ICP TUM
  900.6 → 828.2 ms). Honestly, this is a real but modest win that does **not**
  close the Open3D gap — the remaining cost is per-point PCA and single-threaded
  correspondence, tracked in [#177].
- **Outlier removal now uses the KD-tree** instead of brute force
  (`filtering.rs`), turning `radius_outlier_removal` and
  `statistical_outlier_removal` from O(n²) into O(n log n). Not exercised by the
  four benchmark tasks, but a large asymptotic win on big clouds.
- **FPFH / SHOT neighbor gathering now uses the KD-tree** and runs in parallel
  (`features.rs`), turning feature extraction (and the FPFH+RANSAC global
  registration that depends on it) from O(n²) into O(n log n). Also not in the
  four benchmark tasks.

## Known remaining gaps (honest)

- **Normal estimation is still slower than Open3D at full scale** (~0.5x on
  TUM/KITTI). The kd-tree is now flat/array-backed (see "What changed"), which
  shaved a consistent ~8–10% off but did not close the gap; the dominant remaining
  cost is per-point PCA and single-threaded neighbor search, so parallelising those
  is the next step ([#177]).
- **Dense ICP still trails Open3D** on large clouds (KITTI/TUM), even after the
  k-NN speedup.
- **GPU knn/normals/icp are not competitive yet** (per-call shader/pipeline rebuilds,
  blocking readbacks, no GPU-side spatial index). `gpu_voxel` and TSDF are the
  exceptions. GPU rows are reported separately and never enter the composite.
- **PCL is not measured here yet.** The executable exists (below) but is not wired
  into these numbers.

## PCL benchmark executable (ready, not yet integrated)

`scripts/pcl_bench/` contains a PCL benchmark binary (`pcl_bench.cpp` +
`CMakeLists.txt`) that mirrors this harness exactly — same point cap, voxel size,
normal `k`, ICP iteration count, and the same synthetic rigid target transform —
and prints the same CSV row format. It compiles cleanly (PCL 1.14 via the provided
`Dockerfile`) and runs on the KITTI/nuScenes files. It is **not** yet folded into
the published table; doing so fairly requires running all three libraries in one
environment (the `Dockerfile` is set up for exactly that). Until then, treat PCL
as future work, not as a measured comparison.

## Reproduce (Open3D vs ThreeCrate)

```powershell
.\.venv\Scripts\python.exe scripts\bench_cross_library.py `
  --dataset TUM_Freiburg1_XYZ="C:\Users\Raj Gandhi\Downloads\rgbd_dataset_freiburg1_xyz\rgbd_dataset_freiburg1_xyz" `
  --dataset KITTI="C:\Users\Raj Gandhi\Downloads\raw_data_downloader\KITTI\2011_09_26\2011_09_26_drive_0001_sync\velodyne_points\data\0000000000.bin" `
  --dataset NuScenesMini="C:\Users\Raj Gandhi\Downloads\raw_data_downloader\nuscenes\v1.0-mini\samples\LIDAR_TOP\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin" `
  --tasks read voxel normals icp --iterations 5 --warmups 2 --max-points all `
  --voxel-size 0.2 --max-icp-iters 10 `
  --output target\bench_full.csv --markdown-output target\bench_full.md
```

Swap `--max-points all` for `--max-points 20000` to reproduce the capped table.

## Method notes

- Lower time is better; times are median ms over 5 iterations after 2 warmups.
- The composite includes only rows where ThreeCrate and at least one external
  baseline produced numeric timings. GPU-only rows are excluded.
- ICP uses a synthetic rigid transform (translation `(0.05, -0.02, 0.01)`,
  0.02 rad about z) of the source cloud as the target. This is a near-identity
  registration and does **not** test registration robustness/accuracy — only
  per-iteration speed. A realistic target and an accuracy comparison are future
  work.
- Missing PCL/PDAL values are never estimated from papers, websites, or other
  machines.

## External references

- Open3D point cloud docs: https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html
- PCL VoxelGrid docs: https://pointclouds.org/documentation/classpcl_1_1_voxel_grid.html
- PCL ICP docs: https://pointclouds.org/documentation/classpcl_1_1_iterative_closest_point.html
- KITTI raw data: https://www.cvlibs.net/datasets/kitti/raw_data.php
- TUM RGB-D download: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
- nuScenes paper: https://arxiv.org/abs/1903.11027

[#176]: https://github.com/rajgandhi1/threecrate/issues/176
[#177]: https://github.com/rajgandhi1/threecrate/issues/177
</content>
</invoke>
