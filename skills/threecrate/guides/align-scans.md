# Align scans

Done when: the aligned source is saved to a new file, and you have reported the fit error and whether it converged. For more than two scans, align each one to the same target.

```python
import threecrate as tc

source = tc.read_point_cloud("scan_a.ply")
target = tc.read_point_cloud("scan_b.ply")
s = 0.01  # REPLACE with median_spacing from scripts/inspect_cloud.py

# Downsample copies for speed; the transform still applies to the full clouds.
src = tc.voxel_downsample(source, 2 * s)
tgt = tc.voxel_downsample(target, 2 * s)

# Scans that start far apart: feature matching first, no initial guess needed.
coarse = tc.global_registration(src, tgt)
print(coarse.icp_mse, coarse.inlier_ratio)
src = tc.transform_point_cloud(src, coarse.transformation())

# Refine. GICP is the robust choice; ICP is fine once the scans already overlap.
fine = tc.gicp(src, tgt, max_correspondence_distance=10 * s)
print(fine.mse, fine.converged)

# Apply both steps to the full-resolution source. Matrices are 4x4, source -> target.
T = fine.transformation() @ coarse.transformation()
aligned = tc.transform_point_cloud(source, T)
tc.write_point_cloud(aligned, "scan_a.aligned.ply")
tc.render([source, aligned], "align_check.png")
```

Point-to-point ICP when the scans already overlap closely:

```python
result = tc.icp(source, target, max_iterations=50)
aligned = tc.transform_point_cloud(source, result.transformation())
```

## Good to know

- Point-to-point `icp` can lock onto the wrong match when the clouds start far apart. Start with `global_registration`.
- A low fit error with a low inlier ratio usually means a wrong match on a small overlap. Check the render.
- Scans in different units (mm and m) never align. Compare the inspect sizes first and scale one with `transform_point_cloud`.
