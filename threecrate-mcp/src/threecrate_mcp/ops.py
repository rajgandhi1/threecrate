"""Point cloud and mesh operations behind the MCP tools.

Each function reads its inputs from files, runs a threecrate operation, writes
any result to a file, and returns a small summary dict. Point data never goes
back to the agent: a large scan would flood its context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np
import threecrate as tc

from .workspace import MESH_FORMATS, POINT_CLOUD_WRITE_FORMATS, Workspace, WorkspaceError

# Points sampled when estimating spacing; enough for a stable median.
SPACING_SAMPLE = 2000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _header_point_count(path: Path) -> Optional[int]:
    """The point count a PLY or PCD header declares, read without loading the
    points; None for other formats or an unreadable header."""
    suffix = path.suffix.lower()
    if suffix not in {".ply", ".pcd"}:
        return None
    try:
        with open(path, "rb") as fh:
            for _ in range(200):  # headers are short; give up on anything odd
                line = fh.readline()
                if not line:
                    return None
                words = line.decode("ascii", errors="replace").split()
                if suffix == ".ply" and words[:2] == ["element", "vertex"] and len(words) == 3:
                    return int(words[2])
                if suffix == ".pcd" and words[:1] == ["POINTS"] and len(words) == 2:
                    return int(words[1])
                if words[:1] in (["end_header"], ["DATA"]):
                    return None
    except (OSError, ValueError):
        return None
    return None


def _read_cloud(ws: Workspace, path: str) -> tuple[Path, "tc.PointCloud"]:
    resolved = ws.input_path(path)
    # Refuse oversized files before loading them, when the header says how big.
    declared = _header_point_count(resolved)
    if declared is not None:
        ws.check_size(declared, resolved)
    try:
        cloud = tc.read_point_cloud(str(resolved))
    except Exception as exc:  # threecrate raises plain exceptions for bad files
        raise WorkspaceError(f"could not read {path} as a point cloud: {exc}") from exc
    ws.check_size(len(cloud), resolved)
    if len(cloud) == 0:
        raise WorkspaceError(f"{path} contains no points")
    return resolved, cloud


def _read_mesh(ws: Workspace, path: str) -> tuple[Path, "tc.TriangleMesh"]:
    resolved = ws.input_path(path)
    try:
        mesh = tc.read_mesh(str(resolved))
    except Exception as exc:
        raise WorkspaceError(f"could not read {path} as a mesh: {exc}") from exc
    if mesh.is_empty:
        raise WorkspaceError(f"{path} contains no triangles")
    return resolved, mesh


def _bounds(points: np.ndarray) -> dict:
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) == 0:
        return {"finite_points": 0}
    lo = finite.min(axis=0)
    hi = finite.max(axis=0)
    return {
        "finite_points": int(len(finite)),
        "min": [round(float(v), 6) for v in lo],
        "max": [round(float(v), 6) for v in hi],
        "extent": [round(float(v), 6) for v in hi - lo],
        "centroid": [round(float(v), 6) for v in finite.mean(axis=0)],
    }


def _spacing(cloud: "tc.PointCloud") -> Optional[float]:
    """Median distance from a point to its nearest neighbour, on a sample."""
    points = cloud.to_numpy()
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) < 2:
        return None
    rng = np.random.default_rng(0)
    sample = finite[rng.choice(len(finite), size=min(SPACING_SAMPLE, len(finite)), replace=False)]
    tree = tc.KdTree(cloud)
    distances = []
    for query in sample:
        _, dist = tree.knn(query, 2)
        # The first neighbour is the point itself (distance 0).
        others = [d for d in dist if d > 0]
        if others:
            distances.append(others[0])
    return float(np.median(distances)) if distances else None


def _cloud_summary(cloud: "tc.PointCloud") -> dict:
    return {"points": len(cloud), **_bounds(cloud.to_numpy())}


def _mesh_summary(mesh: "tc.TriangleMesh") -> dict:
    return {
        "vertices": mesh.vertex_count,
        "faces": mesh.face_count,
        **{k: v for k, v in _bounds(mesh.vertices()).items() if k != "finite_points"},
    }


def _write_cloud(cloud: "tc.PointCloud", path: Path) -> None:
    tc.write_point_cloud(cloud, str(path))


def _write_ply_with_normals(positions: np.ndarray, normals: np.ndarray, path: Path) -> None:
    """Binary PLY with x y z nx ny nz (threecrate's writer only stores positions)."""
    data = np.hstack([positions, normals]).astype("<f4")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(data)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "end_header\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(data.tobytes())


def _orient(normals: "tc.NormalPointCloud", orientation: str, k: int) -> "tc.NormalPointCloud":
    """`consistent`: agree all around (outward on closed objects). `viewpoint`:
    keep estimate_normals' choice, facing a point above the cloud (one-sided scans)."""
    if orientation == "consistent":
        return tc.orient_normals(normals, k)
    if orientation == "viewpoint":
        return normals
    raise WorkspaceError(f"unknown normal_orientation {orientation!r}; use 'consistent' or 'viewpoint'")


def _matrix(transform: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise WorkspaceError("transform must be a 4x4 matrix (a list of 4 rows of 4 numbers)")
    if not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise WorkspaceError("the last row of a rigid transform must be [0, 0, 0, 1]")
    return matrix


def _matrix_list(matrix: np.ndarray) -> list[list[float]]:
    return [[round(float(v), 8) for v in row] for row in np.asarray(matrix)]


# ---------------------------------------------------------------------------
# Inspection and conversion
# ---------------------------------------------------------------------------


def inspect_point_cloud(ws: Workspace, path: str) -> dict:
    resolved, cloud = _read_cloud(ws, path)
    spacing = _spacing(cloud)
    summary = {
        "path": ws.relative(resolved),
        "file_bytes": resolved.stat().st_size,
        **_cloud_summary(cloud),
        "median_spacing": None if spacing is None else round(spacing, 6),
    }
    if spacing:
        # Rules of thumb the agent can start from; see the skill for details.
        summary["suggested"] = {
            "voxel_size_light": round(2 * spacing, 6),
            "voxel_size_strong": round(5 * spacing, 6),
            "outlier_radius": round(3 * spacing, 6),
            "cluster_tolerance": round(3 * spacing, 6),
        }
    return summary


def inspect_mesh(ws: Workspace, path: str) -> dict:
    resolved, mesh = _read_mesh(ws, path)
    return {"path": ws.relative(resolved), "file_bytes": resolved.stat().st_size, **_mesh_summary(mesh)}


def convert(
    ws: Workspace,
    path: str,
    output: str,
    kind: Literal["point_cloud", "mesh"] = "point_cloud",
    overwrite: bool = False,
) -> dict:
    if kind == "mesh":
        resolved, mesh = _read_mesh(ws, path)
        out = ws.output_path(output, inputs=[resolved], default_suffix="converted",
                             allowed=MESH_FORMATS, overwrite=overwrite)
        tc.write_mesh(mesh, str(out))
        return {"output": ws.relative(out), **_mesh_summary(mesh)}
    resolved, cloud = _read_cloud(ws, path)
    out = ws.output_path(output, inputs=[resolved], default_suffix="converted",
                         allowed=POINT_CLOUD_WRITE_FORMATS, overwrite=overwrite)
    _write_cloud(cloud, out)
    return {"output": ws.relative(out), "points": len(cloud)}


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def downsample(
    ws: Workspace, path: str, voxel_size: float, output: Optional[str] = None, overwrite: bool = False
) -> dict:
    if not voxel_size > 0:
        raise WorkspaceError("voxel_size must be positive")
    resolved, cloud = _read_cloud(ws, path)
    out = ws.output_path(output, inputs=[resolved], default_suffix="downsampled",
                         allowed=POINT_CLOUD_WRITE_FORMATS, overwrite=overwrite)
    result = tc.voxel_downsample(cloud, voxel_size)
    _write_cloud(result, out)
    return {
        "output": ws.relative(out),
        "points_before": len(cloud),
        "points_after": len(result),
        "voxel_size": voxel_size,
    }


def remove_outliers(
    ws: Workspace,
    path: str,
    method: Literal["statistical", "radius"] = "statistical",
    k_neighbors: int = 20,
    std_ratio: float = 2.0,
    radius: Optional[float] = None,
    min_neighbors: int = 5,
    output: Optional[str] = None,
    overwrite: bool = False,
) -> dict:
    if method == "radius" and (radius is None or not radius > 0):
        raise WorkspaceError("method='radius' needs a positive radius (see inspect_point_cloud's suggestion)")
    resolved, cloud = _read_cloud(ws, path)
    out = ws.output_path(output, inputs=[resolved], default_suffix="clean",
                         allowed=POINT_CLOUD_WRITE_FORMATS, overwrite=overwrite)
    if method == "statistical":
        result = tc.remove_statistical_outliers(cloud, k_neighbors, std_ratio)
        params = {"k_neighbors": k_neighbors, "std_ratio": std_ratio}
    else:
        result = tc.remove_radius_outliers(cloud, radius, min_neighbors)
        params = {"radius": radius, "min_neighbors": min_neighbors}
    _write_cloud(result, out)
    return {
        "output": ws.relative(out),
        "method": method,
        **params,
        "points_before": len(cloud),
        "points_after": len(result),
        "removed": len(cloud) - len(result),
    }


def estimate_normals(
    ws: Workspace,
    path: str,
    k_neighbors: int = 10,
    output: Optional[str] = None,
    overwrite: bool = False,
    normal_orientation: Literal["consistent", "viewpoint"] = "consistent",
) -> dict:
    resolved, cloud = _read_cloud(ws, path)
    out = ws.output_path(output, inputs=[resolved], default_suffix="normals",
                         allowed={".ply"}, default_extension=".ply", overwrite=overwrite)
    normals_cloud = _orient(tc.estimate_normals(cloud, k_neighbors), normal_orientation, k_neighbors)
    _write_ply_with_normals(normals_cloud.positions(), normals_cloud.normals(), out)
    return {
        "output": ws.relative(out),
        "points": len(normals_cloud),
        "k_neighbors": k_neighbors,
        "normal_orientation": normal_orientation,
    }


# ---------------------------------------------------------------------------
# Registration and transforms
# ---------------------------------------------------------------------------


def register(
    ws: Workspace,
    source: str,
    target: str,
    method: Literal["icp", "point_to_plane", "gicp", "global"] = "icp",
    max_iterations: int = 50,
    max_correspondence_distance: float = 1.0,
    voxel_size: Optional[float] = None,
    initial_transform: Optional[Sequence[Sequence[float]]] = None,
    aligned_output: Optional[str] = None,
    overwrite: bool = False,
) -> dict:
    if voxel_size is not None and not voxel_size > 0:
        raise WorkspaceError("voxel_size must be positive")
    if method == "global" and initial_transform is not None:
        raise WorkspaceError("method='global' finds its own starting pose; drop initial_transform")
    init = None if initial_transform is None else _matrix(initial_transform)
    source_path, source_cloud = _read_cloud(ws, source)
    target_path, target_cloud = _read_cloud(ws, target)
    out = None
    if aligned_output is not None:
        out = ws.output_path(aligned_output, inputs=[source_path, target_path], default_suffix="aligned",
                             allowed=POINT_CLOUD_WRITE_FORMATS, overwrite=overwrite)

    # Optional downsampling for speed; the transform still applies to the full clouds.
    src, tgt = source_cloud, target_cloud
    if voxel_size:
        src = tc.voxel_downsample(source_cloud, voxel_size)
        tgt = tc.voxel_downsample(target_cloud, voxel_size)

    if method == "icp":
        result = tc.icp(src, tgt, max_iterations, init)
        metrics = {"mse": result.mse, "iterations": result.iterations, "converged": result.converged}
    elif method == "point_to_plane":
        result = tc.icp_point_to_plane(src, tc.estimate_normals(tgt, 10), max_iterations, init)
        metrics = {"mse": result.mse, "iterations": result.iterations, "converged": result.converged}
    elif method == "gicp":
        result = tc.gicp(src, tgt, max_iterations=max_iterations,
                         max_correspondence_distance=max_correspondence_distance,
                         init_transform=init)
        metrics = {"mse": result.mse, "iterations": result.iterations, "converged": result.converged}
    elif method == "global":
        result = tc.global_registration(src, tgt, icp_max_iterations=max_iterations)
        metrics = {
            "inlier_count": result.inlier_count,
            "inlier_ratio": result.inlier_ratio,
            "mse": result.icp_mse,
            "converged": result.icp_converged,
        }
    else:
        raise WorkspaceError(f"unknown method {method!r}")

    transform = np.asarray(result.transformation(), dtype=np.float64)
    if metrics.get("mse") is not None:
        metrics["rmse"] = round(float(np.sqrt(max(metrics["mse"], 0.0))), 8)
        metrics["mse"] = round(float(metrics["mse"]), 10)

    summary = {
        "method": method,
        "transform": _matrix_list(transform),
        **metrics,
        "source_points": len(source_cloud),
        "target_points": len(target_cloud),
    }
    if voxel_size:
        summary["registered_on"] = {"voxel_size": voxel_size, "source_points": len(src), "target_points": len(tgt)}
    if out is not None:
        _write_cloud(tc.transform_point_cloud(source_cloud, transform), out)
        summary["aligned_output"] = ws.relative(out)
    return summary


def transform(
    ws: Workspace,
    path: str,
    transform: Sequence[Sequence[float]],
    output: Optional[str] = None,
    overwrite: bool = False,
) -> dict:
    matrix = _matrix(transform)
    resolved, cloud = _read_cloud(ws, path)
    out = ws.output_path(output, inputs=[resolved], default_suffix="transformed",
                         allowed=POINT_CLOUD_WRITE_FORMATS, overwrite=overwrite)
    result = tc.transform_point_cloud(cloud, matrix)
    _write_cloud(result, out)
    return {"output": ws.relative(out), **_cloud_summary(result)}


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def segment_plane(
    ws: Workspace,
    path: str,
    distance_threshold: float = 0.01,
    max_iterations: int = 1000,
    inliers_output: Optional[str] = None,
    outliers_output: Optional[str] = None,
    overwrite: bool = False,
) -> dict:
    resolved, cloud = _read_cloud(ws, path)
    # Check every output before computing or writing anything, so a bad path
    # can't leave half the results on disk.
    outputs = {}
    for key, wanted, suffix in (("inliers_output", inliers_output, "plane"),
                                ("outliers_output", outliers_output, "rest")):
        if wanted is not None:
            outputs[key] = ws.output_path(wanted, inputs=[resolved], default_suffix=suffix,
                                          allowed=POINT_CLOUD_WRITE_FORMATS, overwrite=overwrite)
    if len(outputs) == 2 and outputs["inliers_output"] == outputs["outliers_output"]:
        raise WorkspaceError("inliers_output and outliers_output must be different files")
    result = tc.segment_plane(cloud, distance_threshold, max_iterations)
    a, b, c, d = (float(v) for v in result.plane_coefficients())
    inliers = np.asarray(result.inlier_indices(), dtype=np.int64)
    summary = {
        "plane": {"a": round(a, 8), "b": round(b, 8), "c": round(c, 8), "d": round(d, 8)},
        "equation": "a*x + b*y + c*z + d = 0",
        "inliers": int(len(inliers)),
        "inlier_ratio": round(len(inliers) / len(cloud), 6),
    }
    points = cloud.to_numpy()
    mask = np.zeros(len(points), dtype=bool)
    mask[inliers] = True
    for key, keep in (("inliers_output", mask), ("outliers_output", ~mask)):
        if key in outputs:
            _write_cloud(tc.PointCloud(points[keep]), outputs[key])
            summary[key] = ws.relative(outputs[key])
    return summary


def extract_clusters(
    ws: Workspace,
    path: str,
    tolerance: float,
    min_cluster_size: int = 100,
    max_cluster_size: int = 25000,
    output_dir: Optional[str] = None,
    max_files: int = 50,
    overwrite: bool = False,
) -> dict:
    if not tolerance > 0:
        raise WorkspaceError("tolerance must be positive (see inspect_point_cloud's suggestion)")
    resolved, cloud = _read_cloud(ws, path)
    clusters = tc.extract_clusters(cloud, tolerance, min_cluster_size, max_cluster_size)
    summary = {
        "clusters": len(clusters),
        "sizes": [len(c) for c in clusters[:max_files]],
        "points_in_clusters": int(sum(len(c) for c in clusters)),
        "points_total": len(cloud),
    }
    if output_dir is not None:
        folder = ws.output_dir(output_dir)
        # Check every file name first, so an existing file can't stop the
        # call halfway with only some clusters written.
        outs = [
            ws.output_path(str(folder / f"{resolved.stem}.cluster{index:03d}.ply"),
                           inputs=[resolved], default_suffix="cluster",
                           allowed=POINT_CLOUD_WRITE_FORMATS, overwrite=overwrite)
            for index in range(min(len(clusters), max_files))
        ]
        for cluster, out in zip(clusters, outs):
            _write_cloud(cluster, out)
        summary["files"] = [ws.relative(out) for out in outs]
        if len(clusters) > max_files:
            summary["note"] = f"wrote the {max_files} largest clusters; raise max_files for more"
    return summary


# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------


def reconstruct(
    ws: Workspace,
    path: str,
    method: Literal["auto", "poisson", "ball_pivoting", "alpha_shape", "delaunay"] = "auto",
    normals_k: int = 10,
    radius: Optional[float] = None,
    alpha: Optional[float] = None,
    output: Optional[str] = None,
    overwrite: bool = False,
    depth: int = 7,
    trim_distance: Optional[float] = None,
    normal_orientation: Literal["consistent", "viewpoint"] = "consistent",
) -> dict:
    if not 3 <= depth <= 8:
        raise WorkspaceError("depth must be between 3 and 8 (7 takes about a second, 8 about 10-20 seconds)")
    resolved, cloud = _read_cloud(ws, path)
    out = ws.output_path(output, inputs=[resolved], default_suffix="mesh",
                         allowed=MESH_FORMATS, default_extension=".ply", overwrite=overwrite)
    if method == "auto":
        mesh = tc.reconstruct(cloud)
    elif method == "poisson":
        normals = _orient(tc.estimate_normals(cloud, normals_k), normal_orientation, normals_k)
        mesh = tc.poisson_reconstruct(normals, depth=depth, trim_distance=trim_distance)
    elif method == "ball_pivoting":
        if radius is None:
            spacing = _spacing(cloud)
            if not spacing:
                raise WorkspaceError("could not estimate a ball radius; pass radius")
            radius = 2.0 * spacing
        mesh = tc.ball_pivoting_reconstruct(cloud, radius)
    elif method == "alpha_shape":
        mesh = tc.alpha_shape_reconstruct(cloud, 1.0 if alpha is None else alpha)
    elif method == "delaunay":
        mesh = tc.delaunay_triangulate(cloud)
    else:
        raise WorkspaceError(f"unknown method {method!r}")
    if mesh.is_empty:
        raise WorkspaceError(f"{method} produced an empty mesh; try another method or parameters")
    tc.write_mesh(mesh, str(out))
    summary = {"output": ws.relative(out), "method": method, **_mesh_summary(mesh)}
    if method == "ball_pivoting":
        summary["radius"] = round(float(radius), 6)
    if method == "poisson":
        summary["depth"] = depth
    return summary


def simplify_mesh(
    ws: Workspace,
    path: str,
    reduction_ratio: float = 0.5,
    output: Optional[str] = None,
    overwrite: bool = False,
    preserve_boundary: bool = True,
) -> dict:
    if not 0 < reduction_ratio < 1:
        raise WorkspaceError("reduction_ratio must be between 0 and 1 (0.5 removes about half the faces)")
    resolved, mesh = _read_mesh(ws, path)
    out = ws.output_path(output, inputs=[resolved], default_suffix="simplified",
                         allowed=MESH_FORMATS, overwrite=overwrite)
    result = tc.simplify_mesh(mesh, reduction_ratio, preserve_boundary=preserve_boundary)
    tc.write_mesh(result, str(out))
    return {
        "output": ws.relative(out),
        "faces_before": mesh.face_count,
        "faces_after": result.face_count,
        "vertices_after": result.vertex_count,
    }


def smooth_mesh(
    ws: Workspace,
    path: str,
    method: Literal["taubin", "laplacian", "hc"] = "taubin",
    iterations: int = 10,
    output: Optional[str] = None,
    overwrite: bool = False,
) -> dict:
    resolved, mesh = _read_mesh(ws, path)
    out = ws.output_path(output, inputs=[resolved], default_suffix="smooth",
                         allowed=MESH_FORMATS, overwrite=overwrite)
    if method == "taubin":
        result = tc.smooth_mesh_taubin(mesh, iterations)
    elif method == "laplacian":
        result = tc.smooth_mesh_laplacian(mesh, iterations)
    elif method == "hc":
        result = tc.smooth_mesh_hc(mesh, iterations)
    else:
        raise WorkspaceError(f"unknown method {method!r}")
    tc.write_mesh(result, str(out))
    return {"output": ws.relative(out), "method": method, "iterations": iterations, **_mesh_summary(result)}
