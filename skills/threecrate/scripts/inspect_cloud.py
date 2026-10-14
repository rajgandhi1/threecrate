"""Summarize a point cloud and suggest parameters for threecrate.

Usage: python inspect_cloud.py <file> [<file> ...]

Prints JSON with the point count, bounds, median point spacing and starting
values for voxel sizes and radii. Needs `pip install threecrate numpy`.
"""

import json
import sys

import numpy as np
import threecrate as tc

SAMPLE = 2000  # points used to estimate spacing


def median_spacing(cloud: "tc.PointCloud", points: np.ndarray) -> "float | None":
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) < 2:
        return None
    rng = np.random.default_rng(0)
    sample = finite[rng.choice(len(finite), size=min(SAMPLE, len(finite)), replace=False)]
    tree = tc.KdTree(cloud)
    distances = []
    for query in sample:
        _, dist = tree.knn(query, 2)
        others = [d for d in dist if d > 0]  # the first hit is the point itself
        if others:
            distances.append(others[0])
    return float(np.median(distances)) if distances else None


def inspect(path: str) -> dict:
    cloud = tc.read_point_cloud(path)
    points = cloud.to_numpy()
    finite = points[np.isfinite(points).all(axis=1)]
    info = {"path": path, "points": len(points)}
    if len(finite) == 0:
        return info
    finite = finite.astype(np.float64)
    lo, hi = finite.min(axis=0), finite.max(axis=0)
    info.update(
        min=lo.round(6).tolist(),
        max=hi.round(6).tolist(),
        size=(hi - lo).round(6).tolist(),
        centroid=finite.mean(axis=0).round(6).tolist(),
    )
    s = median_spacing(cloud, points)
    info["median_spacing"] = None if s is None else round(s, 6)
    if s:
        info["suggested"] = {
            "voxel_size_light": round(2 * s, 6),
            "voxel_size_strong": round(5 * s, 6),
            "outlier_radius": round(3 * s, 6),
            "cluster_tolerance": round(3 * s, 6),
            "plane_threshold": round(1.5 * s, 6),
            "ball_pivoting_radius": round(2 * s, 6),
            "poisson_trim_distance": round(2.5 * s, 6),
        }
    return info


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for path in sys.argv[1:]:
        print(json.dumps(inspect(path), indent=2))
