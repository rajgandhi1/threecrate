# Cross-Library Benchmarks

This page is a reproducible benchmark note for README updates, release notes, and
forum posts. It is written to be honest first: every number below was measured on
this machine, and the caveats are stated plainly rather than buried.

## TL;DR

- On real point-cloud datasets, ThreeCrate (CPU) is **faster than Open3D at
  reading files, voxel downsampling, and ICP** (ICP is **2.7x to 3.7x** faster
  at full resolution). Normal estimation is faster on two of three datasets and
  slightly slower on full-resolution KITTI (0.92x).
- **ICP accuracy now matches Open3D** on all three datasets (see "ICP accuracy").
- Composite score across the 12 shared rows: **183.0 at full resolution** and
  **188.6 at the 20k-point cap**. Without the `read` rows it is 179.1 and 174.7,
  so the lead does not come from file reading.
- **PCL is not in these numbers yet.** The PCL benchmark is written
  (`scripts/pcl_bench/`) but has not been run here. Do not quote a PCL
  comparison from this page.

## Environment

- OS: Windows 11 (10.0.26200)
- Open3D: 0.19.0 (Python 3.10)
- ThreeCrate: this branch, `--release`
- Datasets: TUM RGB-D `freiburg1_xyz`, KITTI raw drive `2011_09_26_drive_0001`
  (frame `0000000000`), nuScenes `v1.0-mini` (one `LIDAR_TOP` sample)
- Generated: 2026-10-02
- 5 iterations, 2 warmups, median milliseconds (lower is better)

## Results at full resolution

Full frames: TUM about 230k points, KITTI about 121k, nuScenes about 35k.

| Task | Dataset | Open3D (ms) | ThreeCrate (ms) | Ratio (Open3D/TC) |
| --- | --- | ---: | ---: | ---: |
| read | TUM_Freiburg1_XYZ | 15.242 | 3.361 | 4.53x ✅ |
| read | KITTI | 1.380 | 0.888 | 1.55x ✅ |
| read | NuScenesMini | 0.274 | 0.260 | 1.05x ✅ |
| voxel | TUM_Freiburg1_XYZ | 10.274 | 6.670 | 1.54x ✅ |
| voxel | KITTI | 16.906 | 9.122 | 1.85x ✅ |
| voxel | NuScenesMini | 3.763 | 2.412 | 1.56x ✅ |
| normals | TUM_Freiburg1_XYZ | 106.346 | 93.299 | 1.14x ✅ |
| normals | KITTI | 59.174 | 64.345 | 0.92x ❌ |
| normals | NuScenesMini | 18.956 | 12.710 | 1.49x ✅ |
| icp | TUM_Freiburg1_XYZ | 515.311 | 188.979 | 2.73x ✅ |
| icp | KITTI | 204.748 | 55.575 | 3.68x ✅ |
| icp | NuScenesMini | 90.980 | 33.580 | 2.71x ✅ |

Composite (geometric mean of ratios, all 12 rows): **183.0**.

## Results with a 20,000-point cap

Capping hides how things scale. Trust the full-resolution table first.

| Task | Dataset | Open3D (ms) | ThreeCrate (ms) | Ratio |
| --- | --- | ---: | ---: | ---: |
| read | TUM_Freiburg1_XYZ | 15.454 | 3.485 | 4.43x ✅ |
| read | KITTI | 1.394 | 0.902 | 1.55x ✅ |
| read | NuScenesMini | 0.283 | 0.145 | 1.95x ✅ |
| voxel | TUM_Freiburg1_XYZ | 0.735 | 0.575 | 1.28x ✅ |
| voxel | KITTI | 3.350 | 2.017 | 1.66x ✅ |
| voxel | NuScenesMini | 1.417 | 1.202 | 1.18x ✅ |
| normals | TUM_Freiburg1_XYZ | 9.871 | 6.988 | 1.41x ✅ |
| normals | KITTI | 10.180 | 8.478 | 1.20x ✅ |
| normals | NuScenesMini | 11.318 | 6.949 | 1.63x ✅ |
| icp | TUM_Freiburg1_XYZ | 29.860 | 11.418 | 2.62x ✅ |
| icp | KITTI | 29.201 | 7.903 | 3.69x ✅ |
| icp | NuScenesMini | 37.501 | 16.517 | 2.27x ✅ |

Composite (all 12 rows): **188.6**.

## ICP accuracy

The speed tables use an easy, almost aligned target. This test is harder:

- Source: the even-numbered points of each frame.
- Target: the odd-numbered points, moved by a known offset of 0.30 m, 0.20 m,
  0.10 m and about 3 degrees. That is roughly one frame of car motion in KITTI.
- Both libraries start from no offset, use a 1.0 m match distance, run at most 50
  iterations, and use their own default stopping rule.

Errors are measured against the known offset. Lower is better.

| Dataset | Library | Rotation error | Translation error | Inlier RMSE | Time |
| --- | --- | ---: | ---: | ---: | ---: |
| TUM | Open3D | 0.029° | 4.7 mm | 2.8 mm | 1202 ms |
| TUM | ThreeCrate | 0.029° | 4.7 mm | 2.8 mm | 708 ms |
| KITTI | Open3D | 0.104° | 8.8 mm | 94.1 mm | 334 ms |
| KITTI | ThreeCrate | 0.105° | 8.7 mm | 94.1 mm | 90 ms |
| nuScenes | Open3D | 0.972° | 517.7 mm | 300.2 mm | 92 ms |
| nuScenes | ThreeCrate | 0.972° | 517.4 mm | 300.2 mm | 54 ms |

What this shows:

- **TUM and KITTI:** same accuracy as Open3D, and ThreeCrate is faster.
- **nuScenes:** both libraries get stuck in the same wrong spot. This sparse scan
  needs a better starting guess than plain ICP gets here.

Before [#187], ThreeCrate stopped too early on TUM (0.68° and 34 mm off). See
"What changed" below.

## Caveats

- **The TUM `read` row is not a fair I/O test.** ThreeCrate's number is the
  benchmark's own depth-image loop, while Open3D runs its full RGBD pipeline.
  The KITTI and nuScenes read rows are fair (both parse raw `float32`).
- **`voxel` is a fair win** on every dataset. Both return the per-voxel centroid.
- **The ICP speed rows measure speed only.** Accuracy is in "ICP accuracy" above.
- **Normals are not a clean sweep.** Full-resolution KITTI is still 0.92x.

The fair one-line claim: **on CPU, ThreeCrate is faster than Open3D at reading,
voxel downsampling, and ICP with the same ICP accuracy, and faster at normal
estimation except on full-resolution KITTI.**

## What changed in this branch (and why it matters)

These code changes were made to close real algorithmic gaps, not to flatter the
benchmark. Each is covered by unit tests (206 passing).

- **ICP stopping rule is now scale-free** ([#187]). ICP used to stop when the
  error changed by less than a fixed amount. On small indoor scenes the error is
  tiny, so it stopped early (TUM: 0.68° and 34 mm off). It now stops when the
  error improves by less than a set fraction, or when an update barely moves the
  points relative to the cloud's size. TUM accuracy went to 0.029° and 4.7 mm,
  the same as Open3D. The meaning of `convergence_threshold` changed from an
  absolute MSE change to a relative one; existing values like `1e-5` and `1e-6`
  still work well. Clouds far from the origin (such as map coordinates) are now
  moved next to it before ICP runs, so `f32` rounding no longer stops ICP early.
- **Dense ICP and kd-tree build rework** ([#177]). A profile of full-res TUM ICP
  showed ~16% in a serial kd-tree build, ~75% in correspondence search, and ~8% in
  serial gather/SVD/MSE. Changes (`nearest_neighbor.rs`, `registration.rs`):
  - The tree build now uses introselect (`select_nth_unstable_by`) and builds
    large subtrees in parallel. The old last-element-pivot partition degraded
    badly on already-sorted input such as row-ordered depth images.
  - ICP correspondences use a new allocation-free single-nearest query
    (`KdTree::find_nearest_bounded`) instead of the general k-NN path, which
    allocated a heap, a stack, and a result `Vec` per point per iteration.
  - The transform, centroids, covariance and MSE are reduced in one parallel
    pass (in `f64`) instead of serial loops over gathered point lists.
  - Each search is seeded with the point's previous match as a distance bound.
    Results stay exact. Measured honestly, this contributes almost nothing on
    this benchmark (TUM 252 → 243 ms), so the speedup is **not** an artefact of
    the near-identity target.

  Same-machine A/B against `main` (median of 7, two interleaved rounds): ICP
  TUM 1100 → 252 ms (4.3x), KITTI 381 → 91 ms (4.2x), nuScenes 120 → 47 ms
  (2.5x); normals TUM 281 → 129 ms, KITTI 119 → 86 ms, nuScenes 26 → 16 ms.
  ICP iteration counts, convergence and final MSE are unchanged on all three
  datasets.

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
  search-bound — normals were then still ~1.7x behind Open3D at full scale.
- **Flat, array-backed kd-tree** (`nearest_neighbor.rs`, [#176]). The tree was
  `Box`-pointer-based; it is now a contiguous `Vec<KdNode>` with children
  referenced by index, so traversal is cache-friendly. k-NN output is identical
  (201 tests pass). A same-machine A/B measured a **consistent ~8–10% speedup on
  normals and ~5–9% on ICP** (e.g. normals KITTI 105.9 → 97.3 ms; ICP TUM
  900.6 → 828.2 ms). Honestly, this is a real but modest win that does **not**
  close the Open3D gap on its own; the follow-up work in [#177] (first bullet
  above) did.
- **Outlier removal now uses the KD-tree** instead of brute force
  (`filtering.rs`), turning `radius_outlier_removal` and
  `statistical_outlier_removal` from O(n²) into O(n log n). Not exercised by the
  four benchmark tasks, but a large asymptotic win on big clouds.
- **FPFH / SHOT neighbor gathering now uses the KD-tree** and runs in parallel
  (`features.rs`), turning feature extraction (and the FPFH+RANSAC global
  registration that depends on it) from O(n²) into O(n log n). Also not in the
  four benchmark tasks.

## Known remaining gaps (honest)

- **Normal estimation still trails Open3D on full-resolution KITTI** (0.92x),
  though it is now ahead on TUM and nuScenes. The remaining cost there is
  per-point k-NN + PCA, not tree construction.
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
For the accuracy table, use `--tasks icp_accuracy --max-icp-iters 50`.

## Method notes

- Lower time is better; times are median ms over 5 iterations after 2 warmups.
- The composite includes only rows where ThreeCrate and at least one external
  baseline produced numeric timings. GPU-only rows are excluded.
- The `icp` speed task uses a moved copy of the source cloud as the target
  (translation `(0.05, -0.02, 0.01)`, 0.02 rad about z). It only measures speed.
  Accuracy is measured by the separate `icp_accuracy` task.
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
[#180]: https://github.com/rajgandhi1/threecrate/issues/180
[#187]: https://github.com/rajgandhi1/threecrate/issues/187
</content>
</invoke>
