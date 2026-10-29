# v0.12.0 Release Notes

## Highlights

- **Make a scan printable in one call.** `repair_for_print` takes a point cloud
  or a mesh and returns a watertight mesh ready for a slicer. It removes stray
  points, the floor (if asked) and loose bits, closes holes and a missing base,
  turns flipped faces outward and caps the face count. It returns a report of
  each step and the health before and after (#245).
- **Mesh health and measuring.** `analyze_mesh` reports holes, pieces,
  non-manifold edges, flipped faces, watertightness, size, surface area, volume
  and whether the mesh is printable, with plain-English problems (#243).
- **Preview renders without a GPU.** `render` draws point clouds and meshes from
  five views into one PNG. Holes and inside-out faces show orange, so agents and
  people can check a result by looking at it (#244).
- **Agent skill split into task guides.** The threecrate skill now loads one
  guide per task: clean a scan, align scans, make it printable, measure (#252).
- **Sample scans and agent evals.** `samples/` has a phone-style table scan, a
  LiDAR street patch and a broken STL with known answers, plus a demo script.
  `skills/evals/` has 10 cases for `claude plugin eval` (#246).

## Fixes

- Streaming PLY reads misread binary files with normals, colours or double
  coordinates, and `.PLY` in capitals would not stream (#247).
- `IoRegistry::read_mesh` skipped the face-index check that `read_mesh` does,
  and `register_format_handler` did nothing. Both are fixed (#250).
- The mesh report and the repair counted loose pieces differently. Both now
  join faces through shared edges, via the new `mesh_pieces` (#249).

## Changes that may affect you

- Rust `icp()` now returns `Result<Isometry3<f32>>`. It used to return the
  starting pose when it failed, which looked like success. Add `?` (#248).
- `analyze_mesh` counts faces touching at a single corner as separate pieces.
- Python `TriangleMesh.is_empty` is now true when a mesh has no faces, as in
  Rust.
- The streaming PLY readers ignore `chunk_size`; reads are already buffered.

## Crates

All crates and the PyPI package are bumped to `0.12.0`. threecrate-mcp 0.2.0,
with the new `render` and `repair_for_print` tools, follows this release.
