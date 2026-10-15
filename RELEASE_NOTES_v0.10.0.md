# v0.10.0 Release Notes

## Highlights

- **Use threecrate from AI agents.** The new [threecrate-mcp](threecrate-mcp/README.md)
  server gives Claude Code, Codex, OpenCode, Cursor and other MCP clients tools to
  inspect, clean, align and mesh point clouds. It works on files inside one
  workspace folder, never overwrites inputs, and returns short summaries instead
  of raw points.
- **New Poisson reconstruction.** Our own solver replaces the
  `poisson_reconstruction` crate. On a unit sphere it is under 0.05% off (it was
  5% to 17% off) and takes about 1 second at the default depth (it took 1.7 to
  55 seconds). It no longer hangs.
- **Consistent normal orientation.** `orient_normals_consistent` (Rust) and
  `orient_normals` (Python) flip normals so they agree all the way around an
  object. Poisson needs this for full 3D objects; without it about half the
  normals pointed inward.

## Fixes

- Poisson reconstruction hung, even on a 100-point cloud (#201).
- `simplify_mesh` silently returned the mesh unchanged when every edge touched a
  hole. It now raises an error that explains why (#202).
- A tiny file claiming billions of points crashed the process. PLY and PCD
  readers now cap the memory they reserve up front, and a PCD file that ends
  early is an error instead of an endless loop (#210).
- Meshes with face indices past the vertex count crashed Python and the MCP
  server. `read_mesh` and `TriangleMesh.from_numpy` now reject them (#211).
- A transform matrix with NaN or infinity hung forever, and a matrix with scale
  was silently turned into a rotation. Both now raise `ValueError` (#212).
- One NaN point crashed voxel downsampling, ground segmentation, Delaunay and
  ball pivoting. NaN and infinite points are now skipped, and a short CSV row is
  an error instead of a crash (#215).
- PointCloud2 parsing checks the buffer layout instead of panicking on
  inconsistent sizes.

## Changes that may affect you

- **`PoissonConfig` has new fields.** It now has `depth` (3 to 8, default 7),
  `smoothing`, `iso_offset`, `trim_distance`, `max_iterations` and `tolerance`.
  The old fields (`iso_level`, `prune_depth`, `full_depth`, `scale`,
  `samples_per_node`, `confidence`, `output_density`, `cg_depth`) are gone, so
  Rust code that sets them needs updating. This is why this release is 0.10.0.
- **Python `poisson_reconstruct`** takes `depth`, `smoothing` and
  `trim_distance`. Call `orient_normals` first for full 3D objects.
- **Python `simplify_mesh`** takes `preserve_boundary` (default `True`). Set it
  to `False` for meshes that are mostly holes.
- **`poisson_reconstruction_with_normals`** now orients the estimated normals
  before solving.
- **Transforms must be rigid.** `transform_point_cloud` and every
  `init_transform` reject matrices that scale, shear or mirror.
- **`voxel_grid_filter`** rejects a NaN voxel size, and one so small that the
  voxel grid would overflow.
- The `poisson_reconstruction` dependency and the second copy of `nalgebra`
  (0.33) are removed.

## Crates

All crates and the PyPI package are bumped to `0.10.0`. `threecrate-mcp` 0.1.0
requires `threecrate>=0.10.0`.
