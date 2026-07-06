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
| File read (raw float parse) | ✅ Ahead | 1.8x–2.2x faster |
| Voxel downsampling (centroid) | ✅ Ahead | 1.6x–1.8x faster |
| Normal estimation | ⚠️ Behind at scale | 0.57x–1.09x |
| Single-scale ICP | ⚠️ Behind at scale | 0.71x–0.99x |
| PCL comparison | ⏳ Not yet measured | — |

## Near-term: close the honest gaps

These are the concrete, measurable items that move the benchmark and the
credibility story. In rough priority order:

- ~~**Flat-layout kd-tree**~~ — **done** ([#176](https://github.com/rajgandhi1/threecrate/issues/176)).
  The pointer/`Box` tree is now a contiguous, index-referenced `Vec<KdNode>`. k-NN
  results are identical (all 201 algorithm tests pass); a same-machine A/B measured a
  consistent **~8–10% speedup on normal estimation and ~5–9% on ICP**. It does **not**
  close the Open3D gap on its own — normals are still ~0.5x on large clouds — because
  the dominant remaining cost is per-point PCA and single-threaded correspondence
  search, not tree layout. That work continues in [#177](https://github.com/rajgandhi1/threecrate/issues/177).
- **Dense ICP on large clouds** — at parity on small clouds, still ~0.7x on
  KITTI/TUM even after the flat kd-tree. Parallelising correspondence search and
  per-point PCA (rayon) is the next lever. → [#177](https://github.com/rajgandhi1/threecrate/issues/177)
- **Integrate PCL into the benchmark table** — the PCL harness is written and
  builds ([`scripts/pcl_bench/`](scripts/pcl_bench)); it just needs to be run in a
  shared environment and folded into the published numbers. → [#179](https://github.com/rajgandhi1/threecrate/issues/179)
- **Realistic ICP target + accuracy comparison** — today's benchmark tests
  per-iteration speed against a near-identity transform, not registration
  accuracy. → [#180](https://github.com/rajgandhi1/threecrate/issues/180) *(good first issue)*

## Medium-term

- **Competitive GPU compute** — cache pipelines, async readbacks, and a
  GPU-resident spatial index so GPU knn/normals/icp beat CPU (voxel and TSDF
  already do). → [#178](https://github.com/rajgandhi1/threecrate/issues/178)
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
