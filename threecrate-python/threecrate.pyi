"""
Type stubs for the threecrate Python extension module.

These stubs enable IDE autocompletion and static type checking with mypy / pyright.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

class PointCloud:
    """A 3D point cloud holding XYZ positions."""

    @property
    def points(self) -> NDArray[np.float32]:
        """Point positions as a (N, 3) float32 array."""
        ...

    @property
    def is_empty(self) -> bool:
        """True when the cloud contains no points."""
        ...

    def __init__(self, arr: Optional[NDArray] = None) -> None:
        """Create a PointCloud. Optionally accepts a (N, 3) float32 or float64 array."""
        ...

    @staticmethod
    def from_numpy(arr: NDArray) -> PointCloud:
        """Create from a (N, 3) numpy array (float32 or float64)."""
        ...

    def to_numpy(self) -> NDArray[np.float32]:
        """Return point positions as a (N, 3) float32 array."""
        ...

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> NDArray[np.float32]: ...
    def __add__(self, other: PointCloud) -> PointCloud: ...
    def __array__(
        self,
        dtype: Optional[object] = None,
        copy: Optional[object] = None,
    ) -> NDArray[np.float32]: ...
    def __repr__(self) -> str: ...


class NormalPointCloud:
    """A point cloud where each point carries a surface normal.

    Returned by :func:`estimate_normals` and accepted by :func:`poisson_reconstruct`.
    """

    @property
    def is_empty(self) -> bool: ...

    @staticmethod
    def from_numpy(
        positions: NDArray,
        normals: NDArray,
    ) -> NormalPointCloud:
        """Create from separate (N, 3) position and normal arrays (float32 or float64)."""
        ...

    def positions(self) -> NDArray[np.float32]:
        """Point positions as a (N, 3) float32 array."""
        ...

    def normals(self) -> NDArray[np.float32]:
        """Surface normals as a (N, 3) float32 array."""
        ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...


class TriangleMesh:
    """A triangle mesh with vertices and face indices."""

    @property
    def vertex_count(self) -> int: ...
    @property
    def face_count(self) -> int: ...
    @property
    def is_empty(self) -> bool: ...

    def __init__(self) -> None: ...

    @staticmethod
    def from_numpy(
        vertices: NDArray,
        faces: NDArray,
    ) -> TriangleMesh:
        """Create from (N, 3) vertex array (float) and (M, 3) face array (int)."""
        ...

    def vertices(self) -> NDArray[np.float32]:
        """Vertices as a (N, 3) float32 array."""
        ...

    def faces(self) -> NDArray[np.uint32]:
        """Face indices as a (M, 3) uint32 array."""
        ...

    def __repr__(self) -> str: ...


class ColoredPointCloud:
    """A point cloud with per-point RGB colours (XYZ + RGB)."""

    @property
    def is_empty(self) -> bool: ...

    @staticmethod
    def from_numpy(
        positions: NDArray,
        colors: NDArray[np.uint8],
    ) -> ColoredPointCloud:
        """Create from (N, 3) position array (float) and (N, 3) color array (uint8)."""
        ...

    def positions(self) -> NDArray[np.float32]: ...
    def colors(self) -> NDArray[np.uint8]: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...


class ColoredNormalPointCloud:
    """A point cloud with per-point normals and RGB colours."""

    @property
    def is_empty(self) -> bool: ...

    @staticmethod
    def from_numpy(
        positions: NDArray,
        normals: NDArray,
        colors: NDArray[np.uint8],
    ) -> ColoredNormalPointCloud:
        """Create from (N, 3) position/normal arrays (float) and (N, 3) color array (uint8)."""
        ...

    def positions(self) -> NDArray[np.float32]: ...
    def normals(self) -> NDArray[np.float32]: ...
    def colors(self) -> NDArray[np.uint8]: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...


class IcpResult:
    """Result returned by :func:`icp`, :func:`gicp`, :func:`kiss_icp`, :func:`icp_point_to_plane`."""

    mse: float
    """Mean squared error of the final alignment."""
    iterations: int
    """Number of ICP iterations performed."""
    converged: bool
    """Whether the algorithm converged within the iteration limit."""

    def transformation(self) -> NDArray[np.float32]:
        """4×4 rigid transformation matrix (source → target)."""
        ...

    def __repr__(self) -> str: ...


class GlobalRegistrationResult:
    """Result returned by :func:`global_registration`."""

    inlier_count: int
    inlier_ratio: float
    icp_mse: Optional[float]
    icp_converged: Optional[bool]

    def transformation(self) -> NDArray[np.float32]:
        """4×4 rigid transformation matrix (source → target)."""
        ...

    def __repr__(self) -> str: ...


class NdtResult:
    """Result returned by :func:`ndt_registration`."""

    score: float
    iterations: int
    converged: bool

    def transformation(self) -> NDArray[np.float32]:
        """4×4 rigid transformation matrix (source → target)."""
        ...

    def __repr__(self) -> str: ...


class PlaneSegmentationResult:
    """Result returned by :func:`segment_plane`."""

    @property
    def num_inliers(self) -> int: ...

    def plane_coefficients(self) -> NDArray[np.float32]:
        """Plane coefficients [a, b, c, d] (ax + by + cz + d = 0)."""
        ...

    def inlier_indices(self) -> list[int]: ...

    def inlier_cloud(self, cloud: PointCloud) -> PointCloud:
        """Extract a new PointCloud containing only the inlier points."""
        ...

    def __repr__(self) -> str: ...


class KdTree:
    """A KD-tree spatial index built from a point cloud."""

    def __init__(self, cloud: PointCloud) -> None: ...

    def knn(
        self,
        query: NDArray,
        k: int,
    ) -> tuple[list[int], list[float]]:
        """Find the k nearest neighbours.

        Parameters
        ----------
        query:
            1-D array of shape (3,), float32 or float64.
        k:
            Number of neighbours.

        Returns
        -------
        tuple[list[int], list[float]]
            ``(indices, distances)`` ordered nearest first.
        """
        ...

    def radius_search(
        self,
        query: NDArray,
        radius: float,
    ) -> tuple[list[int], list[float]]:
        """Find all neighbours within *radius*.

        Returns
        -------
        tuple[list[int], list[float]]
            ``(indices, distances)`` in unspecified order.
        """
        ...

    def __repr__(self) -> str: ...


class PointCloud2Data:
    """Serialised ``sensor_msgs/PointCloud2`` payload."""

    point_step: int
    row_step: int
    width: int
    height: int
    is_bigendian: bool
    is_dense: bool

    def data(self) -> bytes: ...
    def fields(self) -> list[tuple[str, int, int, int]]: ...
    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def voxel_downsample(cloud: PointCloud, voxel_size: float) -> PointCloud:
    """Downsample using a voxel grid. Each voxel retains one representative point."""
    ...

def remove_statistical_outliers(
    cloud: PointCloud,
    k_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> PointCloud:
    """Remove points whose distance to neighbours is an outlier."""
    ...

def remove_radius_outliers(
    cloud: PointCloud,
    radius: float,
    min_neighbors: int,
) -> PointCloud:
    """Remove points with fewer than *min_neighbors* neighbours within *radius*."""
    ...

# ---------------------------------------------------------------------------
# Normal estimation
# ---------------------------------------------------------------------------

def estimate_normals(
    cloud: PointCloud,
    k_neighbors: int = 10,
) -> NormalPointCloud:
    """Estimate surface normals using K-nearest neighbours."""
    ...

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def icp(
    source: PointCloud,
    target: PointCloud,
    max_iterations: int = 50,
    init_transform: Optional[NDArray] = None,
) -> IcpResult:
    """Point-to-point ICP alignment."""
    ...

def icp_point_to_plane(
    source: PointCloud,
    target: NormalPointCloud,
    max_iterations: int = 50,
    init_transform: Optional[NDArray] = None,
) -> IcpResult:
    """Point-to-plane ICP alignment."""
    ...

def gicp(
    source: PointCloud,
    target: PointCloud,
    max_iterations: int = 50,
    max_correspondence_distance: float = 1.0,
    convergence_threshold: float = 1e-6,
    k_correspondences: int = 20,
    init_transform: Optional[NDArray] = None,
) -> IcpResult:
    """Generalized ICP alignment."""
    ...

def kiss_icp(
    source: PointCloud,
    target: PointCloud,
    voxel_size: float = 1.0,
    max_range: float = 100.0,
    min_range: float = 0.5,
    max_iterations: int = 50,
    init_transform: Optional[NDArray] = None,
) -> IcpResult:
    """KISS-ICP (real-time LiDAR odometry)."""
    ...

def global_registration(
    source: PointCloud,
    target: PointCloud,
    ransac_iterations: int = 50000,
    distance_threshold: float = 0.05,
    inlier_ratio: float = 0.25,
    fpfh_radius: float = 0.25,
    fpfh_k_neighbors: int = 10,
    normal_k_neighbors: int = 10,
    refine_with_icp: bool = True,
    icp_max_iterations: int = 50,
) -> GlobalRegistrationResult:
    """Coarse global registration via FPFH + RANSAC."""
    ...

def global_registration_with_normals(
    source_normals: NormalPointCloud,
    target_normals: NormalPointCloud,
    source: PointCloud,
    target: PointCloud,
    ransac_iterations: int = 50000,
    distance_threshold: float = 0.05,
    inlier_ratio: float = 0.25,
    fpfh_radius: float = 0.25,
    fpfh_k_neighbors: int = 10,
    normal_k_neighbors: int = 10,
    refine_with_icp: bool = True,
    icp_max_iterations: int = 50,
) -> GlobalRegistrationResult:
    """Global registration with pre-computed normals."""
    ...

def ndt_registration(
    source: PointCloud,
    target: PointCloud,
    init_transform: Optional[NDArray] = None,
    resolution: float = 1.0,
    step_size: float = 0.1,
    max_iterations: int = 35,
    epsilon: float = 1e-4,
    min_points_per_voxel: int = 5,
) -> NdtResult:
    """Normal Distributions Transform registration."""
    ...

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_fpfh_features(
    cloud: PointCloud,
    search_radius: float = 0.1,
    k_neighbors: int = 10,
) -> NDArray[np.float32]:
    """Extract FPFH descriptors — returns (N, 33) float32 array."""
    ...

# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_plane(
    cloud: PointCloud,
    threshold: float = 0.01,
    max_iterations: int = 1000,
) -> PlaneSegmentationResult:
    """Fit a plane to the point cloud using RANSAC."""
    ...

def extract_clusters(
    cloud: PointCloud,
    tolerance: float = 0.02,
    min_cluster_size: int = 100,
    max_cluster_size: int = 25000,
) -> list[PointCloud]:
    """Extract Euclidean clusters, ordered largest first."""
    ...

# ---------------------------------------------------------------------------
# Mesh boolean operations
# ---------------------------------------------------------------------------

def mesh_union(mesh_a: TriangleMesh, mesh_b: TriangleMesh) -> TriangleMesh:
    """Boolean union: A ∪ B."""
    ...

def mesh_intersection(mesh_a: TriangleMesh, mesh_b: TriangleMesh) -> TriangleMesh:
    """Boolean intersection: A ∩ B."""
    ...

def mesh_difference(mesh_a: TriangleMesh, mesh_b: TriangleMesh) -> TriangleMesh:
    """Boolean difference: A − B."""
    ...

# ---------------------------------------------------------------------------
# Simplification
# ---------------------------------------------------------------------------

def simplify_mesh(
    mesh: TriangleMesh,
    reduction_ratio: float = 0.5,
) -> TriangleMesh:
    """Simplify using quadric error decimation. reduction_ratio ∈ [0, 1]."""
    ...

# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_mesh_laplacian(
    mesh: TriangleMesh,
    iterations: int = 10,
    lambda_: float = 0.5,
) -> TriangleMesh:
    """Laplacian mesh smoothing."""
    ...

def smooth_mesh_taubin(
    mesh: TriangleMesh,
    iterations: int = 10,
    lambda_: float = 0.5,
    mu: float = -0.53,
) -> TriangleMesh:
    """Taubin (μ|λ) mesh smoothing — preserves volume better than Laplacian."""
    ...

def smooth_mesh_hc(
    mesh: TriangleMesh,
    iterations: int = 10,
    alpha: float = 0.0,
    beta: float = 0.5,
) -> TriangleMesh:
    """HC (Humphrey's Classes) mesh smoothing."""
    ...

# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruct(cloud: PointCloud) -> TriangleMesh:
    """Auto-detect and run the best reconstruction algorithm."""
    ...

def poisson_reconstruct(cloud: NormalPointCloud) -> TriangleMesh:
    """Poisson surface reconstruction — requires normals."""
    ...

def ball_pivoting_reconstruct(
    cloud: PointCloud,
    radius: float = 0.1,
) -> TriangleMesh:
    """Ball Pivoting Algorithm (BPA) surface reconstruction."""
    ...

def alpha_shape_reconstruct(
    cloud: PointCloud,
    alpha: float = 1.0,
) -> TriangleMesh:
    """Alpha-shape surface reconstruction."""
    ...

def delaunay_triangulate(cloud: PointCloud) -> TriangleMesh:
    """Delaunay triangulation — suitable for near-planar clouds."""
    ...

def moving_least_squares_reconstruct(cloud: PointCloud) -> TriangleMesh:
    """Moving Least Squares surface reconstruction."""
    ...

# ---------------------------------------------------------------------------
# Colorization
# ---------------------------------------------------------------------------

def colorize_point_cloud(
    cloud: PointCloud,
    image_data: bytes,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    world_to_camera: NDArray,
) -> ColoredPointCloud:
    """Project a point cloud onto an RGB image to assign per-point colours."""
    ...

# ---------------------------------------------------------------------------
# Point cloud utilities
# ---------------------------------------------------------------------------

def concatenate(clouds: Sequence[PointCloud]) -> PointCloud:
    """Merge a list of point clouds into one."""
    ...

def transform_point_cloud(
    cloud: PointCloud,
    transform: NDArray,
) -> PointCloud:
    """Apply a 4×4 rigid transform to every point in the cloud."""
    ...

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_point_cloud(path: str) -> PointCloud:
    """Read a point cloud from file (PLY, PCD, XYZ, CSV, LAS, LAZ, E57)."""
    ...

def write_point_cloud(cloud: PointCloud, path: str) -> None:
    """Write a point cloud to file (PLY, PCD, XYZ, CSV)."""
    ...

def read_mesh(path: str) -> TriangleMesh:
    """Read a mesh from file (PLY, OBJ)."""
    ...

def write_mesh(mesh: TriangleMesh, path: str) -> None:
    """Write a mesh to file (PLY, OBJ)."""
    ...

# ---------------------------------------------------------------------------
# ROS 2 PointCloud2
# ---------------------------------------------------------------------------

def pointcloud2_to_xyz(
    data: bytes,
    fields: list[tuple[str, int, int, int]],
    point_step: int,
    width: int,
    height: int,
    is_bigendian: bool = False,
    is_dense: bool = True,
) -> PointCloud: ...

def pointcloud2_to_normals(
    data: bytes,
    fields: list[tuple[str, int, int, int]],
    point_step: int,
    width: int,
    height: int,
    is_bigendian: bool = False,
    is_dense: bool = True,
) -> NormalPointCloud: ...

def pointcloud2_to_colored(
    data: bytes,
    fields: list[tuple[str, int, int, int]],
    point_step: int,
    width: int,
    height: int,
    is_bigendian: bool = False,
    is_dense: bool = True,
) -> ColoredPointCloud: ...

def pointcloud2_to_colored_normals(
    data: bytes,
    fields: list[tuple[str, int, int, int]],
    point_step: int,
    width: int,
    height: int,
    is_bigendian: bool = False,
    is_dense: bool = True,
) -> ColoredNormalPointCloud: ...

def xyz_to_pointcloud2(cloud: PointCloud) -> PointCloud2Data: ...
def normals_to_pointcloud2(cloud: NormalPointCloud) -> PointCloud2Data: ...
def colored_to_pointcloud2(cloud: ColoredPointCloud) -> PointCloud2Data: ...
def colored_normals_to_pointcloud2(cloud: ColoredNormalPointCloud) -> PointCloud2Data: ...
