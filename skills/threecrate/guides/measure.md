# Measure

Done when: every number you report has a unit, and you have said which axis is up and whether loose bits or noise could change the answer.

Scans carry no units. Guess them from the size: a vase 0.12 across is in metres, a part 40 across is in millimetres. Say which you assumed.

## A mesh: size, area, volume

```python
import threecrate as tc

mesh = tc.read_mesh("part.stl")
r = tc.analyze_mesh(mesh)
print(r["size"], r["min"], r["max"])       # bounding box, in file units
print(r["surface_area"], r["volume"])      # volume is None unless watertight
print(r["components"])                     # faces per piece, largest first
```

- `volume` is `None` when the mesh has holes. Run `repair_for_print` first (see [make-printable.md](make-printable.md)), then measure the result.
- More than one entry in `components` means loose pieces. They stretch the bounding box: remove them with `repair_for_print`, measure again, and report both sizes.

## A point cloud: sizes and heights

```python
import numpy as np
import threecrate as tc

cloud = tc.read_point_cloud("scan.ply")
s = 0.01  # REPLACE with median_spacing from scripts/inspect_cloud.py
cloud = tc.remove_statistical_outliers(cloud, k_neighbors=20, std_ratio=2.0)  # strays inflate sizes
pts = cloud.to_numpy()

# Robust extent: 1st to 99th percentile ignores the last few strays.
lo, hi = np.percentile(pts, [1, 99], axis=0)
print("size", hi - lo)

# Height above the ground: distance from the floor plane, not raw z.
plane = tc.segment_plane(cloud, threshold=1.5 * s)
a, b, c, d = plane.plane_coefficients()
if c < 0:
    a, b, c, d = -a, -b, -c, -d
height = pts @ np.array([a, b, c]) + d    # signed distance, unit normal

# One object at a time: drop the ground (it joins everything standing on it),
# split the rest, then measure each.
on_plane = np.zeros(len(pts), dtype=bool)
on_plane[plane.inlier_indices()] = True
above = pts[~on_plane & (height > 0)]
objects = tc.extract_clusters(tc.PointCloud(above), tolerance=3 * s, min_cluster_size=100)
for i, obj in enumerate(objects):
    p = obj.to_numpy()
    tall = (p @ np.array([a, b, c]) + d).max()
    print(i, len(p), "points, tall", tall, "extent", np.ptp(p, axis=0))
```

## Good to know

- The largest plane is not always the ground: on a street scan it may be a wall. Check the normal `(a, b, c)` points along the up axis.
- Distance between two points: `np.linalg.norm(p1 - p2)`. Pick the points by index or by a cluster's centroid (`p.mean(axis=0)`), and say which.
