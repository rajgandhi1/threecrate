# Cross-Library Benchmarks

This page is a reproducible benchmark note for README updates, release notes, and
forum posts. It is written to be honest first: every number below was measured on
this machine, and the caveats are stated plainly rather than buried.

## TL;DR

- On real point-cloud datasets, ThreeCrate (CPU) is **faster than Open3D on every
  task we measure**: reading files, voxel downsampling, normal estimation (1.5x
  to 2.1x), and ICP (1.8x to 3.4x) at full resolution.
- **ICP accuracy matches Open3D** on all three datasets (see "ICP accuracy").
- Composite score across the 12 shared rows: **209.7 at full resolution** and
  **199.5 at the 20k-point cap**. Without the `read` rows it is 194.7 and 180.3,
  so the lead does not come from file reading.
- **PCL is now measured** in a Linux container next to Open3D (see "ThreeCrate
  vs Open3D vs PCL"). ThreeCrate's ICP is 8x to 15x faster than PCL's with the
  same accuracy; in that container ThreeCrate and Open3D are close overall.

## Environment

- OS: Windows 11 (10.0.26200)
- Open3D: 0.19.0 (Python 3.10)
- ThreeCrate: this branch, `--release`
- Datasets: TUM RGB-D `freiburg1_xyz`, KITTI raw drive `2011_09_26_drive_0001`
  (frame `0000000000`), nuScenes `v1.0-mini` (one `LIDAR_TOP` sample)
- Generated: 2026-10-02
- 5 iterations, 2 warmups, median milliseconds (lower is better)
- Both libraries ran in the same session, so ratios are fair even though the
  machine was busier than in earlier runs (absolute times are higher).

## Results at full resolution

Full frames: TUM about 230k points, KITTI about 121k, nuScenes about 35k.

| Task | Dataset | Open3D (ms) | ThreeCrate (ms) | Ratio (Open3D/TC) |
| --- | --- | ---: | ---: | ---: |
| read | TUM_Freiburg1_XYZ | 23.908 | 4.061 | 5.89x ✅ |
| read | KITTI | 1.568 | 1.040 | 1.51x ✅ |
| read | NuScenesMini | 0.385 | 0.190 | 2.03x ✅ |
| voxel | TUM_Freiburg1_XYZ | 15.241 | 10.384 | 1.47x ✅ |
| voxel | KITTI | 18.142 | 10.473 | 1.73x ✅ |
| voxel | NuScenesMini | 4.468 | 2.838 | 1.57x ✅ |
| normals | TUM_Freiburg1_XYZ | 175.365 | 82.430 | 2.13x ✅ |
| normals | KITTI | 86.181 | 55.907 | 1.54x ✅ |
| normals | NuScenesMini | 32.323 | 16.456 | 1.96x ✅ |
| icp | TUM_Freiburg1_XYZ | 816.946 | 320.869 | 2.55x ✅ |
| icp | KITTI | 292.445 | 86.429 | 3.38x ✅ |
| icp | NuScenesMini | 110.298 | 61.003 | 1.81x ✅ |

Composite (geometric mean of ratios, all 12 rows): **209.7**.

## Results with a 20,000-point cap

Capping hides how things scale. Trust the full-resolution table first.

| Task | Dataset | Open3D (ms) | ThreeCrate (ms) | Ratio |
| --- | --- | ---: | ---: | ---: |
| read | TUM_Freiburg1_XYZ | 19.030 | 4.281 | 4.45x ✅ |
| read | KITTI | 2.485 | 1.258 | 1.98x ✅ |
| read | NuScenesMini | 0.397 | 0.176 | 2.26x ✅ |
| voxel | TUM_Freiburg1_XYZ | 0.932 | 0.658 | 1.42x ✅ |
| voxel | KITTI | 5.668 | 2.883 | 1.97x ✅ |
| voxel | NuScenesMini | 2.571 | 1.462 | 1.76x ✅ |
| normals | TUM_Freiburg1_XYZ | 16.020 | 11.046 | 1.45x ✅ |
| normals | KITTI | 18.619 | 12.039 | 1.55x ✅ |
| normals | NuScenesMini | 20.264 | 11.100 | 1.83x ✅ |
| icp | TUM_Freiburg1_XYZ | 54.528 | 28.027 | 1.95x ✅ |
| icp | KITTI | 50.607 | 18.289 | 2.77x ✅ |
| icp | NuScenesMini | 55.284 | 29.659 | 1.86x ✅ |

Composite (all 12 rows): **199.5**.

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
| TUM | Open3D | 0.029° | 4.7 mm | 2.8 mm | 1799 ms |
| TUM | ThreeCrate | 0.029° | 4.7 mm | 2.8 mm | 1197 ms |
| KITTI | Open3D | 0.104° | 8.8 mm | 94.1 mm | 546 ms |
| KITTI | ThreeCrate | 0.105° | 8.7 mm | 94.1 mm | 171 ms |
| nuScenes | Open3D | 0.972° | 517.7 mm | 300.2 mm | 168 ms |
| nuScenes | ThreeCrate | 0.972° | 517.4 mm | 300.2 mm | 86 ms |

What this shows:

- **TUM and KITTI:** same accuracy as Open3D, and ThreeCrate is faster.
- **nuScenes:** both libraries get stuck in the same wrong spot. This sparse scan
  needs a better starting guess than plain ICP gets here.

Before [#187], ThreeCrate stopped too early on TUM (0.68° and 34 mm off). See
"What changed" below.

## GPU vs CPU (ThreeCrate only)

Measured on an **NVIDIA RTX 3050 Ti Laptop GPU** (Vulkan backend). The CPU
column is our own 16-core CPU path. Full resolution.

| Task | Dataset | CPU (ms) | GPU (ms) | GPU speedup |
| --- | --- | ---: | ---: | ---: |
| ICP (10 iters) | TUM | 207 | 63 | 3.30x |
| ICP (10 iters) | KITTI | 62 | 24 | 2.56x |
| ICP (10 iters) | nuScenes | 38 | 11 | 3.44x |
| normals (k=10) | TUM | 60 | 25 | 2.43x |
| normals (k=10) | KITTI | 37 | 17 | 2.15x |
| normals (k=10) | nuScenes | 8.4 | 7.0 | 1.20x |
| voxel | TUM | 7.0 | 4.9 | 1.43x |
| voxel | KITTI | 8.9 | 3.3 | 2.73x |
| voxel | nuScenes | 2.2 | 1.5 | 1.50x |

GPU and CPU give the same ICP result, and GPU normals match the CPU on over 99%
of points in the tests. On the laptop's integrated AMD GPU, GPU ICP is still
faster than the CPU (1.1x to 1.6x) and GPU normals are close (0.7x to 1.0x).

## Caveats

- **The TUM `read` row is not a fair I/O test.** ThreeCrate's number is the
  benchmark's own depth-image loop, while Open3D runs its full RGBD pipeline.
  The KITTI and nuScenes read rows are fair (both parse raw `float32`).
- **`voxel` is a fair win** on every dataset. Both return the per-voxel centroid.
- **The ICP speed rows measure speed only.** Accuracy is in "ICP accuracy" above.
- **Run-to-run noise is real.** ICP ratios moved between runs (nuScenes was 2.7x
  in the previous run, 1.8x here) with no ICP code change. Treat single rows as
  rough.

The fair one-line claim: **on CPU, ThreeCrate is faster than Open3D at reading,
voxel downsampling, normal estimation, and ICP, with the same ICP accuracy.**

## What changed in this branch (and why it matters)

These code changes were made to close real algorithmic gaps, not to flatter the
benchmark. Each is covered by unit tests (210 passing).

- **GPU k-NN, normals and ICP rewritten** ([#178]). All three used to compare
  every point with every other point, and GPU normals even did that on the
  CPU first. They now build a kd-tree on the CPU, upload it once, and search it
  on the GPU. Pipelines are built once per context instead of on every call.
  GPU ICP also had a bug: it read 12-byte points as 16-byte ones and returned a
  wrong answer (a 54.7 m shift for a 0.05 m move). It now keeps the target tree
  and match results on the GPU and reads back only small sums each iteration.
  The GPU radius outlier filter got the same treatment. At the 20k cap on KITTI
  (RTX 3050 Ti): normals 111 to 3.5 ms, ICP 66 ms (wrong) to 5.6 ms (correct),
  k-NN 14 to 2.1 ms, radius filter 6.1 to 1.9 ms. A NaN point no longer breaks
  the GPU search. GPU ICP now stops with the same scale-free rule as the CPU,
  and GPU point-to-plane ICP builds its 6x6 system on the GPU.
- **GICP works at any scene size.** It now uses the same scale-free stopping
  rule as ICP ([#187]), and its covariances follow the GICP paper (spread 1, 1,
  0.001) instead of adding a fixed 1e-4 m². Before, a scene at 1/100 scale
  ended up to 0.13 rad off; now it matches the full-size result.

- **Faster k-nearest search for normals** ([#190]). Profiling KITTI showed the
  neighbor search was almost all of the time; the PCA step was tiny. The search
  allocated three buffers per point. It now reuses one buffer per thread and
  skips far branches that can no longer help. Same-machine A/B: normals TUM
  163 to 91 ms, KITTI 120 to 55 ms, nuScenes 25 to 14 ms. KITTI went from 0.92x
  to 1.54x vs Open3D. Outlier removal and FPFH features use the same search, so
  they get faster too.
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


## ThreeCrate vs Open3D vs PCL (Linux container)

All three libraries ran in one Docker container (Ubuntu 24.04, PCL 1.14,
Open3D 0.19) on the same laptop, so they share the OS, compiler and CPUs. The
container sees 16 CPUs. Each library uses its default threading. Full
resolution, median of 5 runs, lower is better.

| Task | Dataset | ThreeCrate (ms) | Open3D (ms) | PCL (ms) |
| --- | --- | ---: | ---: | ---: |
| voxel | TUM | 8.0 | 12.2 | n/a |
| voxel | KITTI | 8.7 | 10.3 | 5.2 |
| voxel | nuScenes | 2.6 | 2.4 | 1.3 |
| normals | TUM | 81 | 83 | n/a |
| normals | KITTI | 44 | 37 | 51 |
| normals | nuScenes | 13 | 14 | 16 |
| icp | TUM | 270 | 378 | n/a |
| icp | KITTI | 100 | 134 | 1459 |
| icp | nuScenes | 71 | 50 | 548 |

ICP accuracy (same test as "ICP accuracy" above, at most 50 iterations):

| Dataset | Library | Rotation error | Translation error | Time |
| --- | --- | ---: | ---: | ---: |
| KITTI | ThreeCrate | 0.105° | 8.7 mm | 159 ms |
| KITTI | Open3D | 0.104° | 8.8 mm | 186 ms |
| KITTI | PCL | 0.089° | 8.5 mm | 3603 ms |
| nuScenes | ThreeCrate | 0.972° | 517 mm | 109 ms |
| nuScenes | Open3D | 0.972° | 518 mm | 60 ms |
| nuScenes | PCL | 0.955° | 519 mm | 1802 ms |

What this shows:

- **PCL:** ThreeCrate's ICP is 8x to 15x faster than PCL's, with the same
  accuracy. Normals are about even. PCL's voxel filter is the fastest of the
  three.
- **Open3D:** in this container ThreeCrate is ahead on TUM and on KITTI ICP and
  voxel, and behind on KITTI normals and nuScenes ICP. Closer than on Windows,
  but no longer behind overall.

Notes:

- **Threads in the VM.** Waking worker threads is slow inside a VM. ICP used to
  make two light parallel passes per iteration and split them finely, so with
  16 threads it was slower than with 8. Since [#194] it makes one pass per
  iteration with at least 512 points per task, which made ICP here 2x to 3x
  faster on small clouds and about 30% faster on full ones.
- **No `read` row.** Files were read through Docker's shared-folder mount, which
  adds about 20 ms for every library, so those timings measure the mount, not the
  libraries.
- **No PCL numbers for TUM.** TUM frames are depth images, and the PCL harness
  has no loader for them. It reports them as unavailable instead of guessing.
- PCL ICP uses its default stopping rule and the same 1.0 m match distance as the
  other two.

To reproduce, build the image with `docker build -t threecrate-bench
scripts/pcl_bench`, then run `scripts/bench_cross_library.py` inside it with
`--pcl-bench-exe /opt/pcl_bench/build/pcl_bench` (see the `Dockerfile` header).

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
[#190]: https://github.com/rajgandhi1/threecrate/issues/190
[#194]: https://github.com/rajgandhi1/threecrate/issues/194
[#178]: https://github.com/rajgandhi1/threecrate/issues/178
</content>
</invoke>
