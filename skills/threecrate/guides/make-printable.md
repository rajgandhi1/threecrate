# Make it printable

Done when: a new STL is saved, `report["after"]["printable"]` is `True`, you have looked at the before and after render, and you have reported the steps taken and the final size with units.

Works from a point cloud (a phone or LiDAR scan) or a mesh (an STL or OBJ that a slicer rejects).

```python
import threecrate as tc

item = tc.read_mesh("part.stl")           # or tc.read_point_cloud("scan.ply")
before = tc.analyze_mesh(item)            # meshes only: what is wrong
print(before["printable"], before["problems"])

# One step: stray points, floor (if asked), holes, a missing base, flipped
# faces, loose bits and face count.
printable, report = tc.repair_for_print(item, remove_floor=True, target_faces=200_000)
print(report["steps"], report["after"]["printable"], report["after"]["size"])

# Orange in the render = you see the inside: a hole or flipped faces.
# The bottom view shows a missing base.
tc.render([item, printable], "print_check.png")
tc.write_mesh(printable, "part.printable.stl")
```

## Choosing options

- `remove_floor=True` when the scan includes the table or ground the object stood on. Leave it off for a mesh, or a scan with nothing under the object.
- `target_faces`: 100,000 to 300,000 suits most printers. Results are capped at 250,000 faces unless you pass a number.
- `depth` (3 to 8, default 7) sets how fine the rebuilt surface is when a scan has to be rebuilt. Each step up is about 8x slower.
- `report["after"]["printable"]` still `False`: read `report["after"]["problems"]`, then rebuild from points with a higher `depth`.

## Building a mesh by hand

For a mesh that isn't meant for printing, or to control each step:

```python
cloud = tc.read_point_cloud("scan.ply")
s = 0.01  # REPLACE with median_spacing from scripts/inspect_cloud.py
normals = tc.orient_normals(tc.estimate_normals(cloud, k_neighbors=10), k_neighbors=10)

quick = tc.reconstruct(cloud)                                    # picks a method
closed = tc.poisson_reconstruct(normals, depth=7)                # smooth, watertight
open_scan = tc.poisson_reconstruct(normals, depth=7, trim_distance=2.5 * s)  # one-sided scan
detailed = tc.ball_pivoting_reconstruct(cloud, radius=2 * s)     # keeps detail, may leave holes

mesh = tc.simplify_mesh(closed, reduction_ratio=0.5)
mesh = tc.smooth_mesh_taubin(mesh, iterations=10)
print(mesh.vertex_count, mesh.face_count)
```

## Good to know

- Poisson always closes the surface. On a scan that only sees one side, set `trim_distance`, or the result is a closed blob.
- `simplify_mesh` raises an error when the mesh is mostly holes (common after ball pivoting). Pass `preserve_boundary=False`, or build with Poisson.
- STL has no units. Check the size: a part that reads 0.04 across was probably exported in metres, and a slicer will treat it as 0.04 mm.
