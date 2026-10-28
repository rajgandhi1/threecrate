# v0.11.0 Release Notes

## Highlights

- **PLY and PCD files load about 20x leaner and 60x faster.** Positions are read
  straight from the file layout. Reading 2 million points used 643 MB (PLY) and
  824 MB (PCD) and took over a second; it now uses 29 MB and 0.02 s.
- **Mesh simplification is fast and keeps the surface intact.** The quadric
  simplifier now updates only the area around each edge it collapses: 50k faces
  take 0.09 s instead of 3 s, and 1M faces about 3 s. It no longer flips faces or
  tears closed meshes open.
- **Auto reconstruction picks a method that works.** Python `reconstruct()` and
  MCP `method="auto"` now use Poisson for 3D objects (estimating and orienting
  normals when needed) and Delaunay for flat clouds. They used to pick ball
  pivoting, which left the mesh full of holes.
- **GPU code handles large clouds.** kNN, normals, ICP and RANSAC work above the
  old limit of about 4.2 million points, use the card's real buffer limits, and
  return errors instead of crashing.

## Fixes

- GPU statistical outlier removal and voxel filtering returned wrong results
  (one kept 20 of 400 points). They now run the CPU versions (#216).
- GPU Euclidean clustering split real clusters and crashed above 65,536 points.
  It now runs the CPU version (#217).
- GPU kNN, normals and ICP crashed above about 4.2 million points (#218).
- The STL reader merged distinct vertices more than about 2147 units from the
  origin, corrupting large CAD parts (#214).
- NDT registration drifted or failed when the scan was far from the origin. It
  now gives the same result at any distance (#223).
- Parallel plane RANSAC kept every try's inliers in memory: 1.5 GB for 1 million
  points. It now uses about 20 MB (#222).
- The public marching cubes left every seam open and pointed every face inward.
  It now shares Poisson's crack-free extraction (#220).
- Moving least squares could not build a mesh without normals, and added a fake
  outer shell with them. Both are fixed (#220).
- Auto reconstruction hid the real error when every method failed (#219).

## Changes that may affect you

- **`MLSSurface::evaluate` now returns positive values outside the surface** and
  negative inside, as its documentation always said. It used to be the opposite.
- **Auto reconstruction** chooses Delaunay for flat clouds and Poisson for
  everything else. The `quality` and `use_case` settings no longer change the
  choice (they still change the fallback list).
- **`QuadricErrorSimplifier`** can land one face below the requested count,
  because one collapse removes two faces at once.
- **`GpuEuclideanClusterConfig::max_neighbors`** is accepted but no longer used.
- **`VolumetricGrid::from_point_cloud`** rejects a grid resolution below 2.

## Crates

All crates and the PyPI package are bumped to `0.11.0`.
