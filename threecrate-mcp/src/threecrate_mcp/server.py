"""MCP server exposing threecrate point cloud and mesh tools.

Run with `threecrate-mcp` (or `uvx threecrate-mcp`). It talks MCP over stdio,
so any MCP client (Claude Code, Claude Desktop, Codex, OpenCode, Cursor, ...)
can launch it. Tools read and write files inside one workspace folder (the
current directory, or `--root` / THREECRATE_MCP_ROOT) and return short
summaries rather than point data.
"""

from __future__ import annotations

import argparse
import functools
import os
from functools import lru_cache
from typing import Annotated, Any, Callable, Literal, Optional, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from . import ops
from .workspace import MAX_POINTS_ENV, ROOT_ENV, Workspace

INSTRUCTIONS = """\
Tools for 3D point clouds and meshes (PLY, PCD, XYZ, CSV, OBJ, ...), powered by
threecrate. All paths are relative to the workspace folder. Tools write results
to new files and return summaries, never raw points.

Typical flow: start with inspect_point_cloud to learn the size, units and point
spacing (it also suggests voxel sizes and radii), then clean (remove_outliers,
downsample), then register two scans or reconstruct a mesh. Outputs default to
a new file next to the input; existing files are only replaced with
overwrite=true.
"""

mcp = MCPServer("threecrate", instructions=INSTRUCTIONS)

F = TypeVar("F", bound=Callable[..., Any])


def tool(fn: F) -> F:
    """Register `fn` as an MCP tool whose failures reach the agent.

    The SDK only passes `ToolError` messages to the model; any other exception
    becomes a bare "Error executing tool". Our messages (a path outside the
    workspace, an output that already exists, a threecrate error on a bad file)
    are what the agent needs to fix its call, so every error becomes a ToolError.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except (ToolError, KeyboardInterrupt, SystemExit):
            raise
        # BaseException: a Rust panic arrives as pyo3's PanicException, which
        # is not an Exception and would otherwise take the server down.
        except BaseException as exc:  # noqa: BLE001 - surface every failure to the agent
            raise ToolError(str(exc) or type(exc).__name__) from exc

    mcp.tool()(wrapper)
    return fn


@lru_cache(maxsize=1)
def _ws() -> Workspace:
    return Workspace.from_env()


Path_ = Annotated[str, Field(description="File path, relative to the workspace folder")]
Output = Annotated[
    Optional[str],
    Field(description="Where to write the result. Default: a new file next to the input"),
]
Overwrite = Annotated[bool, Field(description="Replace the output file if it already exists")]
NormalOrientation = Annotated[
    Literal["consistent", "viewpoint"],
    Field(
        description=(
            "consistent: normals agree all around, outward on closed objects (default). "
            "viewpoint: face a point above the cloud; better for one-sided scans of walls or facades"
        )
    ),
]


@tool
def inspect_point_cloud(path: Path_) -> dict:
    """Summarize a point cloud file: point count, bounds, centroid, median point
    spacing, and suggested voxel sizes and radii for the other tools. Call this
    first on any new cloud."""
    return ops.inspect_point_cloud(_ws(), path)


@tool
def inspect_mesh(path: Path_) -> dict:
    """Check a mesh's health and measure it: holes, separate pieces,
    non-manifold or flipped faces, watertight, size, surface area, volume, and
    whether it can be 3D printed as is (with the reasons if not)."""
    return ops.inspect_mesh(_ws(), path)


@tool
def convert(
    path: Path_,
    output: Annotated[str, Field(description="Output path; its extension picks the format")],
    kind: Annotated[Literal["point_cloud", "mesh"], Field(description="What the file holds")] = "point_cloud",
    overwrite: Overwrite = False,
) -> dict:
    """Convert a point cloud (PLY, PCD, XYZ, CSV) or mesh (PLY, OBJ) to another format."""
    return ops.convert(_ws(), path, output, kind, overwrite)


@tool
def downsample(
    path: Path_,
    voxel_size: Annotated[
        float,
        Field(gt=0, description="Voxel edge length in the cloud's units; each voxel keeps the centroid of its points"),
    ],
    output: Output = None,
    overwrite: Overwrite = False,
) -> dict:
    """Reduce point density with a voxel grid. Good before registration or
    reconstruction. 2x the median spacing is a light pass, 5x a strong one."""
    return ops.downsample(_ws(), path, voxel_size, output, overwrite)


@tool
def remove_outliers(
    path: Path_,
    method: Annotated[
        Literal["statistical", "radius"],
        Field(description="statistical: far from neighbours on average; radius: too few neighbours nearby"),
    ] = "statistical",
    k_neighbors: Annotated[int, Field(ge=2, description="Neighbours checked (statistical)")] = 20,
    std_ratio: Annotated[float, Field(gt=0, description="Lower removes more (statistical)")] = 2.0,
    radius: Annotated[Optional[float], Field(gt=0, description="Search radius (radius method, required)")] = None,
    min_neighbors: Annotated[int, Field(ge=1, description="Minimum neighbours within radius (radius method)")] = 5,
    output: Output = None,
    overwrite: Overwrite = False,
) -> dict:
    """Remove stray points (sensor noise, reflections) from a point cloud."""
    return ops.remove_outliers(_ws(), path, method, k_neighbors, std_ratio, radius, min_neighbors, output, overwrite)


@tool
def estimate_normals(
    path: Path_,
    k_neighbors: Annotated[int, Field(ge=3, description="Neighbours used for each normal")] = 10,
    output: Annotated[Optional[str], Field(description="Output .ply path (stores normals as nx ny nz)")] = None,
    overwrite: Overwrite = False,
    normal_orientation: NormalOrientation = "consistent",
) -> dict:
    """Estimate a surface normal for every point and save them in a PLY file.
    Not needed before register or reconstruct; those estimate normals themselves."""
    return ops.estimate_normals(_ws(), path, k_neighbors, output, overwrite, normal_orientation)


@tool
def register(
    source: Annotated[str, Field(description="Cloud to move")],
    target: Annotated[str, Field(description="Cloud to align to")],
    method: Annotated[
        Literal["icp", "point_to_plane", "gicp", "global"],
        Field(
            description=(
                "icp: fast, needs a rough initial alignment. point_to_plane: better on smooth "
                "surfaces. gicp: most robust local method. global: no initial alignment needed "
                "(feature matching + ICP), slower"
            )
        ),
    ] = "icp",
    max_iterations: Annotated[int, Field(ge=1)] = 50,
    max_correspondence_distance: Annotated[
        float, Field(gt=0, description="Ignore matches farther than this (gicp only)")
    ] = 1.0,
    voxel_size: Annotated[
        Optional[float],
        Field(gt=0, description="Downsample both clouds first for speed; the result still applies to the full clouds"),
    ] = None,
    initial_transform: Annotated[
        Optional[list[list[float]]], Field(description="Starting 4x4 transform (not for method='global')")
    ] = None,
    aligned_output: Annotated[
        Optional[str], Field(description="Also save the source cloud moved into the target's frame here")
    ] = None,
    overwrite: Overwrite = False,
) -> dict:
    """Align a source point cloud to a target. Returns the 4x4 rigid transform
    (source to target), fit error (mse, rmse) and convergence."""
    return ops.register(
        _ws(), source, target, method, max_iterations, max_correspondence_distance,
        voxel_size, initial_transform, aligned_output, overwrite,
    )


@tool
def transform(
    path: Path_,
    transform: Annotated[list[list[float]], Field(description="4x4 rigid transform, e.g. from register")],
    output: Output = None,
    overwrite: Overwrite = False,
) -> dict:
    """Apply a 4x4 rigid transform to every point of a cloud."""
    return ops.transform(_ws(), path, transform, output, overwrite)


@tool
def segment_plane(
    path: Path_,
    distance_threshold: Annotated[float, Field(gt=0, description="Max distance from the plane for a point to count")] = 0.01,
    max_iterations: Annotated[int, Field(ge=1, description="RANSAC iterations")] = 1000,
    inliers_output: Annotated[Optional[str], Field(description="Save the plane's points here")] = None,
    outliers_output: Annotated[Optional[str], Field(description="Save everything else here (e.g. objects on a floor)")] = None,
    overwrite: Overwrite = False,
) -> dict:
    """Find the dominant plane (floor, wall, table) with RANSAC. Returns its
    equation and how many points lie on it; can split the cloud in two."""
    return ops.segment_plane(_ws(), path, distance_threshold, max_iterations, inliers_output, outliers_output, overwrite)


@tool
def extract_clusters(
    path: Path_,
    tolerance: Annotated[float, Field(gt=0, description="Max gap between points of one cluster")],
    min_cluster_size: Annotated[int, Field(ge=1)] = 100,
    max_cluster_size: Annotated[int, Field(ge=1)] = 25000,
    output_dir: Annotated[Optional[str], Field(description="Folder to save each cluster as its own PLY")] = None,
    max_files: Annotated[int, Field(ge=1, description="Save at most this many clusters, largest first")] = 50,
    overwrite: Overwrite = False,
) -> dict:
    """Split a cloud into separate objects by distance (Euclidean clustering).
    Often run after segment_plane removes the floor."""
    return ops.extract_clusters(
        _ws(), path, tolerance, min_cluster_size, max_cluster_size, output_dir, max_files, overwrite
    )


@tool
def reconstruct(
    path: Path_,
    method: Annotated[
        Literal["auto", "poisson", "ball_pivoting", "alpha_shape", "delaunay"],
        Field(
            description=(
                "auto: picks a method. poisson: smooth watertight surfaces. ball_pivoting: keeps "
                "detail, may leave holes. alpha_shape: rough hulls. delaunay: near-flat clouds"
            )
        ),
    ] = "auto",
    normals_k: Annotated[int, Field(ge=3, description="Neighbours for the normals Poisson needs")] = 10,
    radius: Annotated[Optional[float], Field(gt=0, description="Ball radius (ball_pivoting; default 2x spacing)")] = None,
    alpha: Annotated[Optional[float], Field(gt=0, description="Alpha (alpha_shape)")] = None,
    depth: Annotated[
        int,
        Field(ge=3, le=8, description="Poisson detail: grid of 2^depth cells per side. 7 takes about 1 s; 8 is finer but 10-20 s"),
    ] = 7,
    trim_distance: Annotated[
        Optional[float],
        Field(gt=0, description="Poisson: drop surface farther than this from any point. Use on open scans (e.g. 2x spacing)"),
    ] = None,
    normal_orientation: NormalOrientation = "consistent",
    output: Annotated[Optional[str], Field(description="Output mesh path (.ply or .obj)")] = None,
    overwrite: Overwrite = False,
) -> dict:
    """Build a triangle mesh from a point cloud. Clean and downsample first for
    better results. Poisson always closes the surface; set trim_distance for
    scans that only see one side."""
    return ops.reconstruct(
        _ws(), path, method, normals_k, radius, alpha, output, overwrite, depth, trim_distance, normal_orientation
    )


@tool
def simplify_mesh(
    path: Path_,
    reduction_ratio: Annotated[float, Field(gt=0, lt=1, description="Share of faces to remove (0.5 removes about half)")] = 0.5,
    preserve_boundary: Annotated[
        bool,
        Field(description="Keep holes and open borders fixed. Set false if the call fails because the mesh is mostly holes"),
    ] = True,
    output: Output = None,
    overwrite: Overwrite = False,
) -> dict:
    """Reduce a mesh's triangle count while keeping its shape (quadric decimation)."""
    return ops.simplify_mesh(_ws(), path, reduction_ratio, output, overwrite, preserve_boundary)


@tool
def smooth_mesh(
    path: Path_,
    method: Annotated[
        Literal["taubin", "laplacian", "hc"],
        Field(description="taubin keeps volume best; laplacian is strongest but shrinks the mesh"),
    ] = "taubin",
    iterations: Annotated[int, Field(ge=1)] = 10,
    output: Output = None,
    overwrite: Overwrite = False,
) -> dict:
    """Smooth a noisy or bumpy mesh, for example after reconstruct. Taubin keeps
    the overall size; Laplacian smooths harder but shrinks the mesh."""
    return ops.smooth_mesh(_ws(), path, method, iterations, output, overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(description="threecrate MCP server (stdio)")
    parser.add_argument("--root", help=f"Workspace folder (default: current folder, or ${ROOT_ENV})")
    parser.add_argument("--max-points", type=int, help=f"Refuse larger clouds (default 50M, or ${MAX_POINTS_ENV})")
    args = parser.parse_args()
    if args.root:
        os.environ[ROOT_ENV] = args.root
    if args.max_points:
        os.environ[MAX_POINTS_ENV] = str(args.max_points)
    _ws()  # fail fast on a bad root
    mcp.run()


if __name__ == "__main__":
    main()
