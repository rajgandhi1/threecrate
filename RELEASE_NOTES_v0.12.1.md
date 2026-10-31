# v0.12.1 Release Notes

## Highlights

- **Printable meshes come out in millimetres.** Slicers read STL as mm, but
  phone and LiDAR scans are in metres, so a 12 cm vase arrived as 0.12 mm.
  `repair_for_print` now always returns millimetres. A new `units` option
  (`"auto"`, `"mm"`, `"cm"`, `"m"`, `"in"`) says what the input is; `"auto"`
  reads anything under 2 units long as metres and reports the step. The
  report has a new `scale` field (#255).
- **Before/after renders work across units.** When two items differ in size by
  more than 50x, `render` frames each row on its own instead of shrinking one
  to a dot (#255).

## Changes that may affect you

- `repair_for_print` on a mesh or cloud under 2 units long now scales it by
  1000. Pass `units="mm"` to keep the old behaviour.
- Rust: `RepairOptions` has a new `units` field and `RepairReport` a new
  `scale` field. Code that builds `RepairOptions` with `..Default::default()`
  is unaffected.

## Crates

All crates and the PyPI package are bumped to `0.12.1`. threecrate-mcp 0.2.1
adds the `units` option to the `repair_for_print` tool.
