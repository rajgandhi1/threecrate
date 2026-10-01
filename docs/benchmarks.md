# Cross-Library Benchmarks

This page is a reproducible benchmark note for README updates, release notes, and
forum posts. It is written to be honest first: every number below was measured on
this machine, and the caveats are stated plainly rather than buried.

## TL;DR

- On real point-cloud datasets, ThreeCrate (CPU) is **faster than Open3D on file
  read, voxel downsampling, and single-scale ICP** (ICP **2.7x–4.3x** at full
  resolution), and **faster on normal estimation on two of three datasets** — it
  still trails slightly on full-resolution KITTI normals (0.92x).
- Across the 12 shared task/dataset rows the composite score is **208.3 at full
  resolution** and **222.6 at the 20k-point cap**. Unlike the previous run, this
  is **not** carried by `read`: the compute-only score (voxel + normals + icp) is
  **192.4** at full resolution.
- **ICP accuracy** is now measured too (see "ICP accuracy" below). ThreeCrate
  matches Open3D on KITTI, both miss on nuScenes, and ThreeCrate is less accurate
  on TUM because its default stopping rule quits too early on small scenes
  ([#187]).
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
- Generated: 2026-10-01
- 5 iterations, 2 warmups, median milliseconds (lower is better)

## Results — full resolution (no point cap)

This is the meaningful comparison: full frames (TUM ~230k, KITTI ~121k,
nuScenes ~35k points).

| Task | Dataset | Open3D (ms) | ThreeCrate (ms) | Ratio (Open3D/TC) |
| --- | --- | ---: | ---: | ---: |
| read | TUM_Freiburg1_XYZ | 20.827 | 3.756 | 5.54x ✅ |
| read | KITTI | 1.806 | 1.023 | 1.77x ✅ |
| read | NuScenesMini | 0.304 | 0.161 | 1.89x ✅ |
| voxel | TUM_Freiburg1_XYZ | 12.226 | 7.171 | 1.70x ✅ |
| voxel | KITTI | 22.803 | 12.710 | 1.79x ✅ |
| voxel | NuScenesMini | 3.847 | 2.393 | 1.61x ✅ |
| normals | TUM_Freiburg1_XYZ | 160.219 | 123.050 | 1.30x ✅ |
| normals | KITTI | 79.169 | 86.445 | 0.92x ❌ |
| normals | NuScenesMini | 26.156 | 14.831 | 1.76x ✅ |
| icp | TUM_Freiburg1_XYZ | 683.043 | 224.984 | 3.04x ✅ |
| icp | KITTI | 361.661 | 84.937 | 4.26x ✅ |
| icp | NuScenesMini | 108.709 | 40.155 | 2.71x ✅ |

Composite (geometric mean of ratios, all 12 rows): **208.3**.

## Results — 20,000-point cap

Capping every cloud at 20k points makes everything fast and hides scaling. It is
included only because earlier notes used it; the full-resolution table above is
the one to trust.

| Task | Dataset | Open3D (ms) | ThreeCrate (ms) | Ratio |
| --- | --- | ---: | ---: | ---: |
| read | TUM_Freiburg1_XYZ | 17.228 | 4.110 | 4.19x ✅ |
| read | KITTI | 2.014 | 0.950 | 2.12x ✅ |
| read | NuScenesMini | 0.494 | 0.170 | 2.91x ✅ |
| voxel | TUM_Freiburg1_XYZ | 0.743 | 0.578 | 1.29x ✅ |
| voxel | KITTI | 3.576 | 2.104 | 1.70x ✅ |
| voxel | NuScenesMini | 1.524 | 1.201 | 1.27x ✅ |
| normals | TUM_Freiburg1_XYZ | 14.157 | 7.869 | 1.80x ✅ |
| normals | KITTI | 13.973 | 8.578 | 1.63x ✅ |
| normals | NuScenesMini | 15.514 | 7.598 | 2.04x ✅ |
| icp | TUM_Freiburg1_XYZ | 54.843 | 20.260 | 2.71x ✅ |
| icp | KITTI | 42.670 | 11.185 | 3.81x ✅ |
| icp | NuScenesMini | 63.766 | 19.025 | 3.35x ✅ |

Composite (all 12 rows): **222.6**.

## ICP accuracy

The speed tables above use an easy, almost aligned target. This test is harder:

- Source: the even-numbered points of each frame.
- Target: the odd-numbered points, moved by a known offset of 0.30 m, 0.20 m,
  0.10 m and about 3 degrees. That is roughly one frame of car motion in KITTI.
- Both libraries start from no offset, use a 1.0 m match distance, run at most 50
  iterations, and use their own default stopping rule.

Errors are measured against the known offset. Lower is better.

| Dataset | Library | Rotation error | Translation error | Inlier RMSE | Time |
| --- | --- | ---: | ---: | ---: | ---: |
| TUM | Open3D | 0.029° | 4.7 mm | 2.8 mm | 1442 ms |
| TUM | ThreeCrate | 0.665° | 35.0 mm | 8.3 mm | 538 ms |
| KITTI | Open3D | 0.104° | 8.8 mm | 94.1 mm | 422 ms |
| KITTI | ThreeCrate | 0.103° | 10.8 mm | 94.1 mm | 100 ms |
| nuScenes | Open3D | 0.972° | 517.7 mm | 300.2 mm | 127 ms |
| nuScenes | ThreeCrate | 0.987° | 514.1 mm | 299.8 mm | 52 ms |

What this shows:

- **KITTI:** same accuracy, and ThreeCrate is about 4x faster.
- **nuScenes:** both libraries get stuck in the same wrong spot. This sparse scan
  needs a better starting guess than plain ICP gets here.
- **TUM:** ThreeCrate is less accurate. Its default stopping rule looks at the
  absolute change in error, which is tiny on small indoor scenes, so it stops
  after 27 iterations while still 3.5 cm off. With a tighter threshold
  (`--convergence 1e-7`) it reaches 0.032° and 1.9 mm in 699 ms, which matches
  Open3D. We report the default here because that is what users get. Fixing the
  default is tracked in [#187].

## The honest breakdown

The composite is well above 100 and is no longer carried by I/O, but read it
with these caveats:

- **`read` is partly not apples-to-apples.** For KITTI/nuScenes both libraries
  parse raw `float32` records, so those rows are a fair read comparison and
  ThreeCrate genuinely wins (~1.8–2.2x). But the **TUM `read` row is not
  comparable**: ThreeCrate's number is the benchmark's own depth-image
  back-projection loop, while Open3D runs its full RGBD→point-cloud pipeline.
  Treat the TUM read ratio as illustrative, not as a library-I/O result.
- **`voxel` is a genuine, fair win** on every dataset, and the output is now the
  per-voxel **centroid** (matching Open3D/PCL semantics), not an arbitrary first
  point — see "What changed" below.

- **The ICP speed rows measure speed only.** The target is an almost aligned copy
  of the source. Accuracy is covered in "ICP accuracy" above.
- **Normals are not a clean sweep.** Full-resolution KITTI normals are still
  0.92x. That scan is sparse, so per-point PCA costs more than building the tree.

If you remove the `read` task entirely and look only at the compute tasks
(voxel + normals + icp), the geometric-mean score is:

- **Full resolution: 192.4**
- **20k cap: 202.6**

So the fair one-line claim is: **on CPU, ThreeCrate is faster than Open3D on
read, voxel downsampling, and per-iteration ICP throughput, and faster on normal
estimation except full-resolution KITTI. ICP accuracy matches Open3D on KITTI
and trails on TUM until the stopping rule is fixed ([#187]).**

## What changed in this branch (and why it matters)

These code changes were made to close real algorithmic gaps, not to flatter the
benchmark. Each is covered by unit tests (204 passing).

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
- **ICP stops too early on small scenes by default.** On TUM this costs accuracy
  (see "ICP accuracy"). Tracked in [#187].
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
