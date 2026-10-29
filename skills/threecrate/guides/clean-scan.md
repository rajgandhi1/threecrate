# Clean a scan

Done when: the cleaned cloud is saved to a new file, and you have reported the point count before and after each step.

```python
import numpy as np
import threecrate as tc

cloud = tc.read_point_cloud("scan.ply")
s = 0.01  # REPLACE with median_spacing from scripts/inspect_cloud.py

# 1. Noise. Statistical removal handles general noise; radius removal then
#    drops isolated strays it leaves behind.
cloud = tc.remove_statistical_outliers(cloud, k_neighbors=20, std_ratio=2.0)
cloud = tc.remove_radius_outliers(cloud, radius=3 * s, min_neighbors=5)

# 2. Downsample if the cloud is large or denser than the task needs.
cloud = tc.voxel_downsample(cloud, 2 * s)

# 3. Floor or wall, then the objects on it.
plane = tc.segment_plane(cloud, threshold=1.5 * s)
a, b, c, d = plane.plane_coefficients()   # ax + by + cz + d = 0
if c < 0:                                 # the sign varies between runs;
    a, b, c, d = -a, -b, -c, -d           # make the normal point up
pts = cloud.to_numpy()                    # the same cloud the indices refer to
on_plane = np.zeros(len(pts), dtype=bool)
on_plane[plane.inlier_indices()] = True
rest = pts[~on_plane]
above = rest[rest @ np.array([a, b, c]) + d > 0]   # only what sits on the floor
objects = tc.extract_clusters(tc.PointCloud(above), tolerance=3 * s, min_cluster_size=100)
# extract_clusters returns a list of PointClouds, largest first
for i, obj in enumerate(objects):
    tc.write_point_cloud(obj, f"object_{i}.ply")

# 4. Normals, if a later step needs them (meshing, point-to-plane ICP).
normals = tc.estimate_normals(cloud, k_neighbors=10)   # faces a viewpoint above
normals = tc.orient_normals(normals, k_neighbors=10)   # full 3D object: all outward

tc.write_point_cloud(cloud, "scan.clean.ply")
tc.render([tc.read_point_cloud("scan.ply"), cloud], "clean_check.png")
```

## Good to know

- A filter that removes more than a few percent of points usually means `s` is too small. Check it against the inspect output.
- The largest plane is not always the floor: on a street scan it may be a wall. Check the normal `(a, b, c)` points along the up axis before treating it as ground.
- Ground on a slope or uneven terrain: raise the plane threshold, or split the area into tiles.
