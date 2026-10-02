# v0.9.0 Release Notes

## Highlights

- **Faster than Open3D on CPU** on every task we benchmark on Windows (read,
  voxel, normals, ICP), with the same ICP accuracy. In a Linux container the two
  are close, and ThreeCrate's ICP is 8x to 15x faster than PCL's. Full numbers
  in [docs/benchmarks.md](docs/benchmarks.md).
- **ICP is 2.5x to 4x faster** than in 0.8.0. The kd-tree build is parallel and
  no longer slows down on sorted input, nearest-neighbor lookups no longer
  allocate, and each iteration does its sums in one parallel pass.
- **Normal estimation is about 2x faster.** The k-nearest search now reuses one
  buffer per thread. Outlier removal and FPFH features use the same search.
- **GPU k-NN, normals, ICP and radius outlier removal rewritten.** They now
  search a kd-tree kept on the GPU instead of checking every point, and
  pipelines are built once. On an RTX 3050 Ti, GPU ICP is 2.6x to 3.4x and GPU
  normals 1.2x to 2.4x faster than the CPU path.
- **New ICP accuracy benchmark** with a known ground truth, and PCL added to the
  cross-library comparison.

## Fixes

- GPU ICP returned wrong results (it read points in the wrong memory layout).
- ICP stopped too early on small scenes, and could stop early or never stop on
  clouds far from the origin (such as map coordinates).
- GICP failed on small scenes; it now works the same at any scale.
- Clouds with NaN points no longer make kd-tree searches miss neighbors.

## Changes that may affect you

- **`convergence_threshold` in ICP and GICP is now relative.** ICP stops when the
  RMSE improves by less than this fraction, or when an update barely moves the
  points. It used to be an absolute change in MSE. Values like `1e-5` and `1e-6`
  still work well.
- **GPU ICP** (`gpu_icp`, `gpu_icp_point_to_plane`) uses the same relative
  stopping rule.
- **GICP covariances** follow the GICP paper (plane-shaped, spread 1, 1, 0.001)
  instead of adding a fixed 1e-4 m². Results change slightly on most data.
- **ICP max correspondence distance:** a match exactly at the distance is
  accepted, and a negative distance rejects every match.
- **GPU normals** accept `k` from 3 to 31 (it used to be capped at 64).
- **`GpuContext`** gained a private field. Build it with `GpuContext::new()` or
  the new `GpuContext::from_parts(instance, adapter, device, queue)`.
- `threecrate-gpu` now depends on `threecrate-algorithms`.

## New API

- `threecrate_algorithms`: `KdTree::find_nearest_bounded`,
  `KdTree::find_k_nearest_into`, `KdTree::from_points`, `KdTree::flat_nodes`,
  `rigid_transform_from_sums`, `icp_converged`, `LocalFrame`, `CloudScale`.
- `threecrate_gpu`: `GpuContext::from_parts`.

## Python

- Wheels now work on every CPython from 3.8 up (one stable-ABI wheel per
  platform). 0.8.0 only had wheels for Python 3.11.
- The Linux wheel targets manylinux 2.28, so it installs on Ubuntu 20.04+,
  Debian 10+ and RHEL 8+. The 0.8.0 wheel needed glibc 2.38.
- New wheel for Intel Macs (universal2), and a source package for other
  platforms.

## Crates

All crates and the PyPI package are bumped to `0.9.0`: `threecrate`,
`threecrate-core`, `threecrate-algorithms`, `threecrate-io`,
`threecrate-simplification`, `threecrate-gpu`, `threecrate-reconstruction`,
`threecrate-visualization`.

Thanks to @martinfrances107 for the formatting and RNG deprecation fixes.
