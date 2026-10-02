# ThreeCrate Roadmap

This roadmap is written the same way our [benchmarks](docs/benchmarks.md) are:
**honestly.** It says plainly where ThreeCrate already wins, where it still
trails Open3D/PCL, and exactly what work closes each gap. Every item links to a
tracking issue — most are self-contained and a great way to start contributing.

New here? Look for [`good first issue`](https://github.com/rajgandhi1/threecrate/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
and [`help wanted`](https://github.com/rajgandhi1/threecrate/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22).

## Where we stand today

Measured on full-resolution TUM RGB-D, KITTI, and nuScenes-mini frames against
Open3D 0.19 (CPU, same machine). See [docs/benchmarks.md](docs/benchmarks.md)
for the full tables and reproduction command.

| Workload | Status | vs Open3D |
|---|---|---:|
| File read (raw float parse) | ✅ Ahead | 1.5x to 2.0x faster |
| Voxel downsampling (centroid) | ✅ Ahead | 1.5x to 1.7x faster |
| Normal estimation | ✅ Ahead | 1.5x to 2.1x faster |
| ICP speed | ✅ Ahead | 1.8x to 3.4x faster |
| ICP accuracy | ✅ Same as Open3D | all 3 datasets |
| PCL comparison | ✅ Measured | ICP 8x to 15x faster than PCL |

## Near-term: close the honest gaps

These are the concrete, measurable items that move the benchmark and the
credibility story. In rough priority order:

- ~~**Flat-layout kd-tree**~~ — **done** ([#176](https://github.com/rajgandhi1/threecrate/issues/176)).
  The pointer/`Box` tree is now a contiguous, index-referenced `Vec<KdNode>`. k-NN
  results are identical (all 201 algorithm tests pass); a same-machine A/B measured a
  consistent **~8–10% speedup on normal estimation and ~5–9% on ICP**. It does **not**
  close the Open3D gap on its own (normals were still ~0.5x on large clouds), because
  the dominant remaining cost was elsewhere — see the next item.
- ~~**Dense ICP on large clouds**~~ — **done** ([#177](https://github.com/rajgandhi1/threecrate/issues/177)).
  Profiling showed the cost was a serial kd-tree build (whose pivot choice degraded
  on sorted input), a k-NN query that allocated three times per point, and serial
  covariance/MSE loops. With an introselect + parallel build, an allocation-free
  nearest query, and a parallel reduction, ICP went from **0.71x–0.99x to
  2.7x–4.3x** vs Open3D, and normals from 0.57x–1.09x to 0.92x–1.76x.
- ~~**Close the last normals gap**~~: **done** ([#190](https://github.com/rajgandhi1/threecrate/issues/190)).
  The k-nearest search no longer allocates per point. KITTI normals went from
  0.92x to 1.54x vs Open3D.
- ~~**Integrate PCL into the benchmark table**~~: **done** ([#179](https://github.com/rajgandhi1/threecrate/issues/179)).
  All three libraries ran in one Linux container. ThreeCrate's ICP is 8x to 15x
  faster than PCL's with the same accuracy.
- ~~**Too many threads slow down small clouds in VMs**~~: **done** ([#194](https://github.com/rajgandhi1/threecrate/issues/194)).
  ICP now makes one parallel pass per iteration with at least 512 points per
  task. In the Docker VM that made ICP 2x to 3x faster on small clouds.
- ~~**ICP accuracy comparison**~~: **done** ([#180](https://github.com/rajgandhi1/threecrate/issues/180)).
  New `icp_accuracy` benchmark with a known offset.
- ~~**Fix the ICP stopping rule**~~: **done** ([#187](https://github.com/rajgandhi1/threecrate/issues/187)).
  ICP now stops on a relative rule, so it works the same at any scene size. TUM
  accuracy now matches Open3D.

## Medium-term

- ~~**Competitive GPU compute**~~: **done** ([#178](https://github.com/rajgandhi1/threecrate/issues/178)).
  GPU k-NN, normals, ICP and the radius outlier filter now search a kd-tree kept
  on the GPU instead of checking every point, and pipelines are built once. On
  an RTX 3050 Ti, GPU ICP is 2.6x to 3.4x and normals 1.2x to 2.4x faster than
  the CPU.
- ~~Fix the GPU TSDF buffer-cast panic~~ — **done** ([#175](https://github.com/rajgandhi1/threecrate/issues/175)).
  The readback cast a mapped GPU buffer (8-byte aligned) straight into
  `repr(align(16))` structs; now it copies into a correctly aligned `Vec`. All
  TSDF tests pass, no `#[ignore]`.
- **Broader format coverage** and streaming improvements across `threecrate-io`.
- **Python API parity** with the Rust surface (`threecrate-python`).

## Longer-term / exploratory

- **WebAssembly** target for in-browser point-cloud processing.
- **More global-registration and segmentation** algorithms.
- Realistic, published **accuracy** benchmarks (not just speed) across libraries.

## How to help

1. Pick an issue above (or any [`good first issue`](https://github.com/rajgandhi1/threecrate/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)).
2. Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.
3. For perf work, include before/after benchmark numbers — the reproduction
   command is in [docs/benchmarks.md](docs/benchmarks.md).
4. Open a draft PR early; we'd rather help shape it than review it cold.

Have an idea that isn't here? Open a
[discussion](https://github.com/rajgandhi1/threecrate/discussions) or an issue.
