use nalgebra::Isometry3;
use numpy::{
    ndarray::Array1, ndarray::Array2, IntoPyArray,
    PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use threecrate_algorithms::{
    estimate_normals as tc_estimate_normals, extract_euclidean_clusters,
    icp_point_to_plane as tc_icp_point_to_plane, icp_point_to_point_default,
    gicp::{gicp as tc_gicp, GicpConfig},
    kiss_icp::{kiss_icp as tc_kiss_icp, KissIcpConfig},
    radius_outlier_removal, segment_plane as tc_segment_plane,
    smooth_hc, smooth_laplacian, smooth_taubin, statistical_outlier_removal, voxel_grid_filter,
    HcSmoothingConfig, LaplacianSmoothingConfig, TaubinSmoothingConfig,
    // Global registration
    global_registration as tc_global_registration,
    global_registration_with_normals as tc_global_registration_with_normals,
    GlobalRegistrationConfig,
    // NDT registration
    ndt_registration as tc_ndt_registration,
    NdtConfig,
    // Mesh boolean operations
    mesh_union as tc_mesh_union,
    mesh_intersection as tc_mesh_intersection,
    mesh_difference as tc_mesh_difference,
    // FPFH feature extraction
    features::{extract_fpfh_features_with_normals, FpfhConfig, FPFH_DIM},
    // Colorization
    CameraIntrinsics, RgbImageView, ColorizationConfig,
    colorize_point_cloud as tc_colorize_point_cloud,
    // KD-Tree
    KdTree as TcKdTree,
};
use threecrate_core::{
    ColoredNormalPoint3f, ColoredPoint3f, NormalPoint3f, NearestNeighborSearch,
    Point3f, PointCloud, TriangleMesh, Vector3f,
};
use threecrate_io::{
    read_mesh as rs_read_mesh, read_point_cloud as rs_read_pc, write_mesh as rs_write_mesh,
    write_point_cloud as rs_write_pc,
    ros2::{
        self as tc_ros2,
        PointCloud2Info,
        PointField as Ros2PointField,
    },
};
use threecrate_reconstruction::{
    auto_reconstruct, poisson_reconstruction_default,
    ball_pivoting_reconstruction as tc_ball_pivoting,
    alpha_shape_reconstruction as tc_alpha_shape,
    delaunay_triangulation as tc_delaunay,
    moving_least_squares as tc_moving_least_squares,
};
use threecrate_simplification::{MeshSimplifier, QuadricErrorSimplifier};

fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Shared conversion helpers
// ---------------------------------------------------------------------------

/// Convert an ICPResult into the PyIcpResult wrapper.
fn icp_result_to_py(r: threecrate_algorithms::ICPResult) -> PyIcpResult {
    let mat = r.transformation.to_homogeneous();
    PyIcpResult {
        _transformation: [
            [mat[(0, 0)], mat[(0, 1)], mat[(0, 2)], mat[(0, 3)]],
            [mat[(1, 0)], mat[(1, 1)], mat[(1, 2)], mat[(1, 3)]],
            [mat[(2, 0)], mat[(2, 1)], mat[(2, 2)], mat[(2, 3)]],
            [mat[(3, 0)], mat[(3, 1)], mat[(3, 2)], mat[(3, 3)]],
        ],
        mse: r.mse,
        iterations: r.iterations,
        converged: r.converged,
    }
}

/// Convert an Isometry3 into a 4×4 row array suitable for PyIcpResult.
fn isometry_to_mat4(iso: &Isometry3<f32>) -> [[f32; 4]; 4] {
    let mat = iso.to_homogeneous();
    [
        [mat[(0, 0)], mat[(0, 1)], mat[(0, 2)], mat[(0, 3)]],
        [mat[(1, 0)], mat[(1, 1)], mat[(1, 2)], mat[(1, 3)]],
        [mat[(2, 0)], mat[(2, 1)], mat[(2, 2)], mat[(2, 3)]],
        [mat[(3, 0)], mat[(3, 1)], mat[(3, 2)], mat[(3, 3)]],
    ]
}

/// Convert an optional (N, 3) or 4×4 numpy array to `Isometry3<f32>`.
/// If `None`, returns the identity transform.
fn numpy_to_isometry(arr: Option<PyReadonlyArray2<f32>>) -> PyResult<Isometry3<f32>> {
    let arr = match arr {
        None => return Ok(Isometry3::identity()),
        Some(a) => a,
    };
    let a = arr.as_array();
    if a.shape() != [4, 4] {
        return Err(PyValueError::new_err(
            "init_transform must be a 4×4 float32 array",
        ));
    }
    let rot = nalgebra::Matrix3::new(
        a[[0, 0]], a[[0, 1]], a[[0, 2]],
        a[[1, 0]], a[[1, 1]], a[[1, 2]],
        a[[2, 0]], a[[2, 1]], a[[2, 2]],
    );
    let rotation = nalgebra::UnitQuaternion::from_matrix(&rot);
    let translation = nalgebra::Translation3::new(a[[0, 3]], a[[1, 3]], a[[2, 3]]);
    Ok(Isometry3::from_parts(translation, rotation))
}

/// Convert a 1-D length-3 numpy array to `Point3f`.
fn numpy1d_to_point(arr: &PyReadonlyArray1<f32>) -> PyResult<Point3f> {
    let a = arr.as_array();
    if a.len() != 3 {
        return Err(PyValueError::new_err("Query point must be a 1D array of length 3"));
    }
    Ok(Point3f::new(a[0], a[1], a[2]))
}

// ---------------------------------------------------------------------------
// PointCloud
// ---------------------------------------------------------------------------

/// A 3D point cloud holding XYZ positions.
#[pyclass(name = "PointCloud")]
#[derive(Clone)]
pub struct PyPointCloud {
    pub(crate) inner: PointCloud<Point3f>,
}

#[pymethods]
impl PyPointCloud {
    /// Create an empty PointCloud.
    #[new]
    fn new() -> Self {
        Self {
            inner: PointCloud::new(),
        }
    }

    /// Create a PointCloud from a numpy array of shape (N, 3) and dtype float32.
    #[staticmethod]
    fn from_numpy(arr: PyReadonlyArray2<f32>) -> PyResult<Self> {
        let arr = arr.as_array();
        let shape = arr.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(PyValueError::new_err(
                "Array must have shape (N, 3) with dtype float32",
            ));
        }
        let points = (0..shape[0])
            .map(|i| Point3f::new(arr[[i, 0]], arr[[i, 1]], arr[[i, 2]]))
            .collect();
        Ok(Self {
            inner: PointCloud::from_points(points),
        })
    }

    /// Return the point positions as a numpy array of shape (N, 3) and dtype float32.
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let n = self.inner.len();
        let mut data = Array2::<f32>::zeros((n, 3));
        for (i, p) in self.inner.points.iter().enumerate() {
            data[[i, 0]] = p.x;
            data[[i, 1]] = p.y;
            data[[i, 2]] = p.z;
        }
        data.into_pyarray_bound(py)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("PointCloud({} points)", self.inner.len())
    }
}

// ---------------------------------------------------------------------------
// NormalPointCloud
// ---------------------------------------------------------------------------

/// A point cloud where each point carries an estimated surface normal.
/// Returned by `estimate_normals()` and accepted by `poisson_reconstruct()`.
#[pyclass(name = "NormalPointCloud")]
#[derive(Clone)]
pub struct PyNormalPointCloud {
    pub(crate) inner: PointCloud<NormalPoint3f>,
}

#[pymethods]
impl PyNormalPointCloud {
    /// Return point positions as a numpy array of shape (N, 3) float32.
    fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let n = self.inner.len();
        let mut data = Array2::<f32>::zeros((n, 3));
        for (i, p) in self.inner.points.iter().enumerate() {
            data[[i, 0]] = p.position.x;
            data[[i, 1]] = p.position.y;
            data[[i, 2]] = p.position.z;
        }
        data.into_pyarray_bound(py)
    }

    /// Return surface normals as a numpy array of shape (N, 3) float32.
    fn normals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let n = self.inner.len();
        let mut data = Array2::<f32>::zeros((n, 3));
        for (i, p) in self.inner.points.iter().enumerate() {
            data[[i, 0]] = p.normal.x;
            data[[i, 1]] = p.normal.y;
            data[[i, 2]] = p.normal.z;
        }
        data.into_pyarray_bound(py)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("NormalPointCloud({} points)", self.inner.len())
    }
}

// ---------------------------------------------------------------------------
// TriangleMesh
// ---------------------------------------------------------------------------

/// A triangle mesh with vertices and face indices.
#[pyclass(name = "TriangleMesh")]
#[derive(Clone)]
pub struct PyTriangleMesh {
    pub(crate) inner: TriangleMesh,
}

#[pymethods]
impl PyTriangleMesh {
    /// Create an empty TriangleMesh.
    #[new]
    fn new() -> Self {
        Self {
            inner: TriangleMesh::new(),
        }
    }

    #[getter]
    fn vertex_count(&self) -> usize {
        self.inner.vertex_count()
    }

    #[getter]
    fn face_count(&self) -> usize {
        self.inner.face_count()
    }

    /// Return vertices as a numpy array of shape (N, 3) float32.
    fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let n = self.inner.vertices.len();
        let mut data = Array2::<f32>::zeros((n, 3));
        for (i, v) in self.inner.vertices.iter().enumerate() {
            data[[i, 0]] = v.x;
            data[[i, 1]] = v.y;
            data[[i, 2]] = v.z;
        }
        data.into_pyarray_bound(py)
    }

    /// Return face indices as a numpy array of shape (M, 3) uint32.
    fn faces<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u32>> {
        let m = self.inner.faces.len();
        let mut data = Array2::<u32>::zeros((m, 3));
        for (i, f) in self.inner.faces.iter().enumerate() {
            data[[i, 0]] = f[0] as u32;
            data[[i, 1]] = f[1] as u32;
            data[[i, 2]] = f[2] as u32;
        }
        data.into_pyarray_bound(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "TriangleMesh({} vertices, {} faces)",
            self.inner.vertex_count(),
            self.inner.face_count()
        )
    }
}

// ---------------------------------------------------------------------------
// IcpResult
// ---------------------------------------------------------------------------

/// Result returned by `icp()`, `gicp()`, `kiss_icp()`, and `icp_point_to_plane()`.
#[pyclass(name = "IcpResult")]
pub struct PyIcpResult {
    _transformation: [[f32; 4]; 4],
    /// Mean squared error of the final alignment.
    #[pyo3(get)]
    pub mse: f32,
    /// Number of ICP iterations performed.
    #[pyo3(get)]
    pub iterations: usize,
    /// Whether the algorithm converged within the iteration limit.
    #[pyo3(get)]
    pub converged: bool,
}

#[pymethods]
impl PyIcpResult {
    /// Return the 4x4 rigid transformation matrix as a numpy array (float32).
    fn transformation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let data = Array2::from_shape_fn((4, 4), |(i, j)| self._transformation[i][j]);
        data.into_pyarray_bound(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "IcpResult(converged={}, mse={:.6}, iterations={})",
            self.converged, self.mse, self.iterations
        )
    }
}

// ---------------------------------------------------------------------------
// GlobalRegistrationResult
// ---------------------------------------------------------------------------

/// Result returned by `global_registration()`.
///
/// Attributes
/// ----------
/// inlier_count : int
///     Number of RANSAC inlier correspondences.
/// inlier_ratio : float
///     Fraction of correspondences that are inliers (0.0 – 1.0).
/// icp_mse : float or None
///     Final ICP mean squared error when ``refine_with_icp=True``.
/// icp_converged : bool or None
///     Whether ICP converged, when ``refine_with_icp=True``.
#[pyclass(name = "GlobalRegistrationResult")]
pub struct PyGlobalRegistrationResult {
    _transformation: [[f32; 4]; 4],
    #[pyo3(get)]
    pub inlier_count: usize,
    #[pyo3(get)]
    pub inlier_ratio: f32,
    #[pyo3(get)]
    pub icp_mse: Option<f32>,
    #[pyo3(get)]
    pub icp_converged: Option<bool>,
}

#[pymethods]
impl PyGlobalRegistrationResult {
    /// Return the 4×4 rigid transformation (source → target) as a numpy float32 array.
    fn transformation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let data = Array2::from_shape_fn((4, 4), |(i, j)| self._transformation[i][j]);
        data.into_pyarray_bound(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "GlobalRegistrationResult(inliers={}, ratio={:.3})",
            self.inlier_count, self.inlier_ratio
        )
    }
}

// ---------------------------------------------------------------------------
// NdtResult
// ---------------------------------------------------------------------------

/// Result returned by `ndt_registration()`.
///
/// Attributes
/// ----------
/// score : float
///     Final NDT score (higher = better alignment).
/// iterations : int
///     Number of gradient-descent iterations performed.
/// converged : bool
///     Whether the algorithm converged.
#[pyclass(name = "NdtResult")]
pub struct PyNdtResult {
    _transformation: [[f32; 4]; 4],
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub converged: bool,
}

#[pymethods]
impl PyNdtResult {
    /// Return the 4×4 rigid transformation (source → target) as a numpy float32 array.
    fn transformation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let data = Array2::from_shape_fn((4, 4), |(i, j)| self._transformation[i][j]);
        data.into_pyarray_bound(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "NdtResult(converged={}, score={:.6}, iterations={})",
            self.converged, self.score, self.iterations
        )
    }
}

// ---------------------------------------------------------------------------
// PlaneSegmentationResult
// ---------------------------------------------------------------------------

/// Result returned by `segment_plane()`.
///
/// Contains the fitted plane model and the indices of inlier points.
#[pyclass(name = "PlaneSegmentationResult")]
pub struct PyPlaneSegmentationResult {
    /// Plane coefficients [a, b, c, d] (ax + by + cz + d = 0).
    coefficients: [f32; 4],
    inliers: Vec<usize>,
}

#[pymethods]
impl PyPlaneSegmentationResult {
    /// Plane coefficients as a numpy array of shape (4,) float32: [a, b, c, d].
    ///
    /// The plane equation is: a·x + b·y + c·z + d = 0.
    /// The [a, b, c] part is the unit normal of the plane.
    fn plane_coefficients<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let data = Array1::from_vec(self.coefficients.to_vec());
        data.into_pyarray_bound(py)
    }

    /// Indices of the inlier points in the original cloud (sorted).
    fn inlier_indices(&self) -> Vec<usize> {
        self.inliers.clone()
    }

    /// Number of inlier points.
    #[getter]
    fn num_inliers(&self) -> usize {
        self.inliers.len()
    }

    /// Extract a new PointCloud containing only the inlier points.
    fn inlier_cloud(&self, cloud: &PyPointCloud) -> PyPointCloud {
        let points = self
            .inliers
            .iter()
            .map(|&i| cloud.inner.points[i])
            .collect();
        PyPointCloud {
            inner: PointCloud::from_points(points),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PlaneSegmentationResult(inliers={}, normal=[{:.3}, {:.3}, {:.3}])",
            self.inliers.len(),
            self.coefficients[0],
            self.coefficients[1],
            self.coefficients[2],
        )
    }
}

// ---------------------------------------------------------------------------
// KdTree
// ---------------------------------------------------------------------------

/// A KD-tree spatial index built from a point cloud.
///
/// Provides efficient k-nearest-neighbour and radius searches.
///
/// Examples
/// --------
/// >>> tree = threecrate.KdTree(cloud)
/// >>> indices, distances = tree.knn(query, k=5)
/// >>> indices, distances = tree.radius_search(query, radius=0.5)
#[pyclass(name = "KdTree")]
pub struct PyKdTree {
    inner: TcKdTree,
}

#[pymethods]
impl PyKdTree {
    /// Build a KD-tree from a point cloud.
    #[new]
    fn new(cloud: &PyPointCloud) -> PyResult<Self> {
        TcKdTree::new(&cloud.inner.points)
            .map(|t| Self { inner: t })
            .map_err(to_py_err)
    }

    /// Find the `k` nearest neighbours to a query point.
    ///
    /// Parameters
    /// ----------
    /// query : ndarray of shape (3,) float32
    ///     The query position.
    /// k : int
    ///     Number of neighbours to return.
    ///
    /// Returns
    /// -------
    /// tuple[list[int], list[float]]
    ///     ``(indices, distances)`` — point indices in the original cloud and
    ///     their Euclidean distances from the query, ordered nearest first.
    fn knn(
        &self,
        query: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<(Vec<usize>, Vec<f32>)> {
        let q = numpy1d_to_point(&query)?;
        let results = self.inner.find_k_nearest(&q, k);
        Ok((
            results.iter().map(|(i, _)| *i).collect(),
            results.iter().map(|(_, d)| *d).collect(),
        ))
    }

    /// Find all neighbours within `radius` of a query point.
    ///
    /// Parameters
    /// ----------
    /// query : ndarray of shape (3,) float32
    ///     The query position.
    /// radius : float
    ///     Search radius (same units as the point cloud).
    ///
    /// Returns
    /// -------
    /// tuple[list[int], list[float]]
    ///     ``(indices, distances)`` — point indices and their Euclidean distances,
    ///     unordered.
    fn radius_search(
        &self,
        query: PyReadonlyArray1<f32>,
        radius: f32,
    ) -> PyResult<(Vec<usize>, Vec<f32>)> {
        let q = numpy1d_to_point(&query)?;
        let results = self.inner.find_radius_neighbors(&q, radius);
        Ok((
            results.iter().map(|(i, _)| *i).collect(),
            results.iter().map(|(_, d)| *d).collect(),
        ))
    }

    fn __repr__(&self) -> String {
        "KdTree".to_string()
    }
}

// ---------------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------------

/// Downsample a point cloud using a voxel grid.
///
/// Each voxel of the given size keeps one representative point.
#[pyfunction]
fn voxel_downsample(cloud: &PyPointCloud, voxel_size: f32) -> PyResult<PyPointCloud> {
    voxel_grid_filter(&cloud.inner, voxel_size)
        .map(|c| PyPointCloud { inner: c })
        .map_err(to_py_err)
}

/// Remove statistical outliers from a point cloud.
///
/// Points whose mean distance to their `k_neighbors` nearest neighbours deviates
/// more than `std_ratio` standard deviations from the global mean are removed.
#[pyfunction]
#[pyo3(signature = (cloud, k_neighbors = 20, std_ratio = 2.0))]
fn remove_statistical_outliers(
    cloud: &PyPointCloud,
    k_neighbors: usize,
    std_ratio: f32,
) -> PyResult<PyPointCloud> {
    statistical_outlier_removal(&cloud.inner, k_neighbors, std_ratio)
        .map(|c| PyPointCloud { inner: c })
        .map_err(to_py_err)
}

/// Remove radius outliers from a point cloud.
///
/// Points with fewer than `min_neighbors` neighbours within `radius` are removed.
#[pyfunction]
fn remove_radius_outliers(
    cloud: &PyPointCloud,
    radius: f32,
    min_neighbors: usize,
) -> PyResult<PyPointCloud> {
    radius_outlier_removal(&cloud.inner, radius, min_neighbors)
        .map(|c| PyPointCloud { inner: c })
        .map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Normal estimation
// ---------------------------------------------------------------------------

/// Estimate surface normals using K-nearest neighbours.
///
/// Returns a `NormalPointCloud` that can be passed directly to
/// `poisson_reconstruct()`.
#[pyfunction]
#[pyo3(name = "estimate_normals", signature = (cloud, k_neighbors = 10))]
fn py_estimate_normals(
    cloud: &PyPointCloud,
    k_neighbors: usize,
) -> PyResult<PyNormalPointCloud> {
    tc_estimate_normals(&cloud.inner, k_neighbors)
        .map(|c| PyNormalPointCloud { inner: c })
        .map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Align `source` to `target` using point-to-point ICP.
///
/// Returns an `IcpResult` containing the 4x4 transformation matrix,
/// final MSE, iteration count, and convergence flag.
///
/// Args:
///     source: Source point cloud to align.
///     target: Target (reference) point cloud.
///     max_iterations: Maximum ICP iterations (default 50).
///     init_transform: Optional 4×4 float32 numpy array for the initial pose
///         estimate. Defaults to the identity transform.
#[pyfunction]
#[pyo3(signature = (source, target, max_iterations = 50, init_transform = None))]
fn icp(
    source: &PyPointCloud,
    target: &PyPointCloud,
    max_iterations: usize,
    init_transform: Option<PyReadonlyArray2<f32>>,
) -> PyResult<PyIcpResult> {
    let init = numpy_to_isometry(init_transform)?;
    icp_point_to_point_default(&source.inner, &target.inner, init, max_iterations)
        .map(icp_result_to_py)
        .map_err(to_py_err)
}

/// Align `source` to `target` using Generalized ICP (GICP).
///
/// GICP models each point as a local Gaussian distribution estimated from its
/// k nearest neighbours and weights correspondences by the combined covariance.
/// More robust to noise than point-to-point or point-to-plane ICP.
///
/// Args:
///     source: Source point cloud to align.
///     target: Target (reference) point cloud.
///     max_iterations: Maximum ICP iterations (default 50).
///     max_correspondence_distance: Maximum distance for accepting a match (default 1.0).
///     convergence_threshold: Stop when |ΔMSE| is below this value (default 1e-6).
///     k_correspondences: Neighbours used to estimate per-point covariances (default 20).
///     init_transform: Optional 4×4 float32 initial pose. Defaults to identity.
#[pyfunction]
#[pyo3(signature = (
    source, target,
    max_iterations = 50,
    max_correspondence_distance = 1.0,
    convergence_threshold = 1e-6,
    k_correspondences = 20,
    init_transform = None,
))]
fn gicp(
    source: &PyPointCloud,
    target: &PyPointCloud,
    max_iterations: usize,
    max_correspondence_distance: f32,
    convergence_threshold: f32,
    k_correspondences: usize,
    init_transform: Option<PyReadonlyArray2<f32>>,
) -> PyResult<PyIcpResult> {
    let init = numpy_to_isometry(init_transform)?;
    let config = GicpConfig {
        max_iterations,
        max_correspondence_distance,
        convergence_threshold,
        k_correspondences,
    };
    tc_gicp(&source.inner, &target.inner, init, config)
        .map(icp_result_to_py)
        .map_err(to_py_err)
}

/// Align `source` to `target` using KISS-ICP.
///
/// KISS-ICP (Bai et al., IROS 2023) applies range filtering, voxel downsampling,
/// and standard point-to-point ICP with an adaptive correspondence threshold.
/// Designed for real-time LiDAR odometry without per-dataset parameter tuning.
///
/// Args:
///     source: Current LiDAR scan (raw; range filtering + downsampling are applied internally).
///     target: Reference / map point cloud.
///     voxel_size: Downsampling voxel size in metres (default 1.0).
///     max_range: Discard points farther than this from the sensor (default 100.0 m).
///     min_range: Discard points closer than this (removes ego-vehicle noise, default 0.5 m).
///     max_iterations: Maximum ICP iterations (default 50).
///     init_transform: Optional 4×4 float32 initial pose. Defaults to identity.
#[pyfunction]
#[pyo3(signature = (
    source, target,
    voxel_size = 1.0,
    max_range = 100.0,
    min_range = 0.5,
    max_iterations = 50,
    init_transform = None,
))]
fn kiss_icp(
    source: &PyPointCloud,
    target: &PyPointCloud,
    voxel_size: f32,
    max_range: f32,
    min_range: f32,
    max_iterations: usize,
    init_transform: Option<PyReadonlyArray2<f32>>,
) -> PyResult<PyIcpResult> {
    let init = numpy_to_isometry(init_transform)?;
    let config = KissIcpConfig {
        voxel_size,
        max_range,
        min_range,
        max_iterations,
    };
    tc_kiss_icp(&source.inner, &target.inner, init, config)
        .map(icp_result_to_py)
        .map_err(to_py_err)
}

/// Align `source` to `target` using point-to-plane ICP.
///
/// `source` is a plain point cloud (positions only). `target` must be a
/// `NormalPointCloud` — the surface normals at each target point define
/// the local tangent plane used by the algorithm.
///
/// Args:
///     source: Source point cloud (positions only).
///     target: Target NormalPointCloud (positions + normals).
///     max_iterations: Maximum ICP iterations (default 50).
///     init_transform: Optional 4×4 float32 initial pose. Defaults to identity.
#[pyfunction]
#[pyo3(signature = (source, target, max_iterations = 50, init_transform = None))]
fn icp_point_to_plane(
    source: &PyPointCloud,
    target: &PyNormalPointCloud,
    max_iterations: usize,
    init_transform: Option<PyReadonlyArray2<f32>>,
) -> PyResult<PyIcpResult> {
    let init = numpy_to_isometry(init_transform)?;
    let source_cloud = PointCloud::from_points(
        source.inner.points.iter()
            .map(|p| Point3f::new(p.x, p.y, p.z))
            .collect(),
    );
    let target_cloud = PointCloud::from_points(
        target.inner.points.iter()
            .map(|p| Point3f::new(p.position.x, p.position.y, p.position.z))
            .collect(),
    );
    let target_normals: Vec<Vector3f> = target.inner.points.iter()
        .map(|p| Vector3f::new(p.normal.x, p.normal.y, p.normal.z))
        .collect();
    tc_icp_point_to_plane(
        &source_cloud,
        &target_cloud,
        &target_normals,
        init,
        max_iterations,
    )
    .map(icp_result_to_py)
    .map_err(to_py_err)
}

/// Coarse global registration using FPFH feature matching and RANSAC.
///
/// Estimates surface normals, extracts FPFH descriptors from both clouds,
/// builds putative correspondences by nearest-neighbour feature matching, and
/// finds the best rigid transformation using RANSAC. An optional ICP refinement
/// step is run afterwards when `refine_with_icp=True`.
///
/// Use the returned `transformation` as `init_transform` for a subsequent ICP
/// call to achieve fine alignment.
///
/// Args:
///     source: Source point cloud.
///     target: Target point cloud.
///     ransac_iterations: RANSAC iteration count (default 50000).
///     distance_threshold: Max inlier distance in model units (default 0.05).
///     inlier_ratio: Early-exit RANSAC threshold (default 0.25).
///     fpfh_radius: Radius for FPFH feature extraction (default 0.25).
///     fpfh_k_neighbors: Fallback k-NN for FPFH when radius yields too few (default 10).
///     normal_k_neighbors: k-NN for normal estimation (default 10).
///     refine_with_icp: Run ICP after RANSAC (default True).
///     icp_max_iterations: ICP iteration limit when refining (default 50).
#[pyfunction]
#[pyo3(signature = (
    source, target,
    ransac_iterations = 50_000,
    distance_threshold = 0.05,
    inlier_ratio = 0.25,
    fpfh_radius = 0.25,
    fpfh_k_neighbors = 10,
    normal_k_neighbors = 10,
    refine_with_icp = true,
    icp_max_iterations = 50,
))]
fn global_registration(
    source: &PyPointCloud,
    target: &PyPointCloud,
    ransac_iterations: usize,
    distance_threshold: f32,
    inlier_ratio: f32,
    fpfh_radius: f32,
    fpfh_k_neighbors: usize,
    normal_k_neighbors: usize,
    refine_with_icp: bool,
    icp_max_iterations: usize,
) -> PyResult<PyGlobalRegistrationResult> {
    let config = GlobalRegistrationConfig {
        ransac_iterations,
        distance_threshold,
        inlier_ratio,
        fpfh_radius,
        fpfh_k_neighbors,
        normal_k_neighbors,
        refine_with_icp,
        icp_max_iterations,
        icp_distance_threshold: None,
    };
    tc_global_registration(&source.inner, &target.inner, &config)
        .map(|r| {
            let (icp_mse, icp_converged) = r
                .icp_result
                .map(|icp| (Some(icp.mse), Some(icp.converged)))
                .unwrap_or((None, None));
            PyGlobalRegistrationResult {
                _transformation: isometry_to_mat4(&r.transformation),
                inlier_count: r.inlier_count,
                inlier_ratio: r.inlier_ratio,
                icp_mse,
                icp_converged,
            }
        })
        .map_err(to_py_err)
}

/// Global registration when normals are already available.
///
/// Same as `global_registration()` but skips the normal-estimation step.
/// Useful when you have already called `estimate_normals()`.
///
/// Args:
///     source_normals: Source NormalPointCloud (pre-computed normals).
///     target_normals: Target NormalPointCloud.
///     source: Raw source positions (used for optional ICP refinement).
///     target: Raw target positions.
///     (All other parameters identical to `global_registration`.)
#[pyfunction]
#[pyo3(signature = (
    source_normals, target_normals, source, target,
    ransac_iterations = 50_000,
    distance_threshold = 0.05,
    inlier_ratio = 0.25,
    fpfh_radius = 0.25,
    fpfh_k_neighbors = 10,
    normal_k_neighbors = 10,
    refine_with_icp = true,
    icp_max_iterations = 50,
))]
fn global_registration_with_normals(
    source_normals: &PyNormalPointCloud,
    target_normals: &PyNormalPointCloud,
    source: &PyPointCloud,
    target: &PyPointCloud,
    ransac_iterations: usize,
    distance_threshold: f32,
    inlier_ratio: f32,
    fpfh_radius: f32,
    fpfh_k_neighbors: usize,
    normal_k_neighbors: usize,
    refine_with_icp: bool,
    icp_max_iterations: usize,
) -> PyResult<PyGlobalRegistrationResult> {
    let config = GlobalRegistrationConfig {
        ransac_iterations,
        distance_threshold,
        inlier_ratio,
        fpfh_radius,
        fpfh_k_neighbors,
        normal_k_neighbors,
        refine_with_icp,
        icp_max_iterations,
        icp_distance_threshold: None,
    };
    tc_global_registration_with_normals(
        &source_normals.inner,
        &target_normals.inner,
        &source.inner,
        &target.inner,
        &config,
    )
    .map(|r| {
        let (icp_mse, icp_converged) = r
            .icp_result
            .map(|icp| (Some(icp.mse), Some(icp.converged)))
            .unwrap_or((None, None));
        PyGlobalRegistrationResult {
            _transformation: isometry_to_mat4(&r.transformation),
            inlier_count: r.inlier_count,
            inlier_ratio: r.inlier_ratio,
            icp_mse,
            icp_converged,
        }
    })
    .map_err(to_py_err)
}

/// Align `source` to `target` using Normal Distributions Transform (NDT).
///
/// NDT divides the target cloud into voxels and fits a Gaussian distribution
/// to each; the source is aligned by gradient-descent maximisation of the
/// probability of the transformed source points under those distributions.
/// More robust than ICP to large initial misalignments and sparse data.
///
/// Args:
///     source: Source point cloud to align.
///     target: Target (reference) point cloud.
///     init_transform: Optional 4×4 float32 initial pose. Defaults to identity.
///     resolution: Voxel cell side length (default 1.0).
///     step_size: Gradient-descent step size (default 0.1).
///     max_iterations: Maximum iterations (default 35).
///     epsilon: Convergence threshold on transformation change (default 1e-4).
///     min_points_per_voxel: Voxels with fewer points are discarded (default 5).
#[pyfunction]
#[pyo3(signature = (
    source, target,
    init_transform = None,
    resolution = 1.0,
    step_size = 0.1,
    max_iterations = 35,
    epsilon = 1e-4,
    min_points_per_voxel = 5,
))]
fn ndt_registration(
    source: &PyPointCloud,
    target: &PyPointCloud,
    init_transform: Option<PyReadonlyArray2<f32>>,
    resolution: f32,
    step_size: f32,
    max_iterations: usize,
    epsilon: f32,
    min_points_per_voxel: usize,
) -> PyResult<PyNdtResult> {
    let init = numpy_to_isometry(init_transform)?;
    let config = NdtConfig {
        resolution,
        step_size,
        max_iterations,
        epsilon,
        min_points_per_voxel,
    };
    tc_ndt_registration(&source.inner, &target.inner, init, &config)
        .map(|r| PyNdtResult {
            _transformation: isometry_to_mat4(&r.transformation),
            score: r.score,
            iterations: r.iterations,
            converged: r.converged,
        })
        .map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// FPFH feature extraction
// ---------------------------------------------------------------------------

/// Extract FPFH (Fast Point Feature Histograms) descriptors from a point cloud.
///
/// Surface normals are estimated automatically. Each descriptor encodes the
/// local geometry around a point as three 11-bin angular sub-histograms
/// (33 values total).
///
/// Args:
///     cloud: Input point cloud.
///     search_radius: Radius for neighbourhood search during feature extraction (default 0.1).
///     k_neighbors: Fallback k-NN count when radius search yields too few points (default 10).
///
/// Returns
/// -------
/// ndarray of shape (N, 33) float32
///     One 33-dimensional FPFH descriptor per input point.
#[pyfunction]
#[pyo3(signature = (cloud, search_radius = 0.1, k_neighbors = 10))]
fn extract_fpfh_features<'py>(
    py: Python<'py>,
    cloud: &PyPointCloud,
    search_radius: f32,
    k_neighbors: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let cloud_n = tc_estimate_normals(&cloud.inner, k_neighbors).map_err(to_py_err)?;
    let config = FpfhConfig { search_radius, k_neighbors };
    let descs = extract_fpfh_features_with_normals(&cloud_n, &config).map_err(to_py_err)?;

    let n = descs.len();
    let mut data = Array2::<f32>::zeros((n, FPFH_DIM));
    for (i, desc) in descs.iter().enumerate() {
        for (j, &v) in desc.iter().enumerate() {
            data[[i, j]] = v;
        }
    }
    Ok(data.into_pyarray_bound(py))
}

// ---------------------------------------------------------------------------
// Segmentation
// ---------------------------------------------------------------------------

/// Fit a plane to a point cloud using RANSAC.
///
/// Returns a `PlaneSegmentationResult` with the plane equation and inlier indices.
/// Use `result.inlier_cloud(cloud)` to extract the planar points, or
/// filter them out to keep the non-planar remainder.
///
/// Args:
///     cloud: Input point cloud.
///     threshold: Max point-to-plane distance to count as an inlier (default 0.01).
///     max_iterations: Number of RANSAC iterations (default 1000).
#[pyfunction]
#[pyo3(signature = (cloud, threshold = 0.01, max_iterations = 1000))]
fn segment_plane(
    cloud: &PyPointCloud,
    threshold: f32,
    max_iterations: usize,
) -> PyResult<PyPlaneSegmentationResult> {
    tc_segment_plane(&cloud.inner, threshold, max_iterations)
        .map(|r| {
            let c = r.model.coefficients;
            PyPlaneSegmentationResult {
                coefficients: [c.x, c.y, c.z, c.w],
                inliers: r.inliers,
            }
        })
        .map_err(to_py_err)
}

/// Extract Euclidean clusters from a point cloud.
///
/// Grows clusters by BFS: any unvisited point within `tolerance` of a seed
/// is added to the same cluster. Clusters smaller than `min_cluster_size` or
/// larger than `max_cluster_size` are discarded.
///
/// Returns a list of `PointCloud` objects, one per cluster, ordered largest first.
///
/// Args:
///     cloud: Input point cloud.
///     tolerance: Max distance between neighbouring points in the same cluster (default 0.02).
///     min_cluster_size: Minimum points for a valid cluster (default 100).
///     max_cluster_size: Maximum points allowed in a cluster (default 25000).
#[pyfunction]
#[pyo3(signature = (cloud, tolerance = 0.02, min_cluster_size = 100, max_cluster_size = 25000))]
fn extract_clusters(
    cloud: &PyPointCloud,
    tolerance: f32,
    min_cluster_size: usize,
    max_cluster_size: usize,
) -> PyResult<Vec<PyPointCloud>> {
    use threecrate_algorithms::EuclideanClusterConfig;
    let config = EuclideanClusterConfig::new(tolerance, min_cluster_size, max_cluster_size);
    extract_euclidean_clusters(&cloud.inner, &config)
        .map(|result| {
            result
                .clusters
                .iter()
                .map(|indices| {
                    let pts = indices.iter().map(|&i| cloud.inner.points[i]).collect();
                    PyPointCloud {
                        inner: PointCloud::from_points(pts),
                    }
                })
                .collect()
        })
        .map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Mesh boolean operations
// ---------------------------------------------------------------------------

/// Compute the boolean union of two closed triangle meshes (A ∪ B).
///
/// Both meshes must be closed (watertight) with consistently outward-facing
/// normals. Uses a BSP-tree CSG approach.
///
/// Args:
///     mesh_a: First input mesh.
///     mesh_b: Second input mesh.
#[pyfunction]
fn mesh_union(mesh_a: &PyTriangleMesh, mesh_b: &PyTriangleMesh) -> PyResult<PyTriangleMesh> {
    tc_mesh_union(&mesh_a.inner, &mesh_b.inner)
        .map(|m| PyTriangleMesh { inner: m })
        .map_err(to_py_err)
}

/// Compute the boolean intersection of two closed triangle meshes (A ∩ B).
///
/// Both meshes must be closed (watertight) with consistently outward-facing
/// normals. Uses a BSP-tree CSG approach.
///
/// Args:
///     mesh_a: First input mesh.
///     mesh_b: Second input mesh.
#[pyfunction]
fn mesh_intersection(
    mesh_a: &PyTriangleMesh,
    mesh_b: &PyTriangleMesh,
) -> PyResult<PyTriangleMesh> {
    tc_mesh_intersection(&mesh_a.inner, &mesh_b.inner)
        .map(|m| PyTriangleMesh { inner: m })
        .map_err(to_py_err)
}

/// Compute the boolean difference of two closed triangle meshes (A − B).
///
/// Returns the part of `mesh_a` that lies outside `mesh_b`. Both meshes must
/// be closed (watertight) with consistently outward-facing normals.
///
/// Args:
///     mesh_a: Minuend mesh.
///     mesh_b: Subtrahend mesh.
#[pyfunction]
fn mesh_difference(
    mesh_a: &PyTriangleMesh,
    mesh_b: &PyTriangleMesh,
) -> PyResult<PyTriangleMesh> {
    tc_mesh_difference(&mesh_a.inner, &mesh_b.inner)
        .map(|m| PyTriangleMesh { inner: m })
        .map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Mesh simplification
// ---------------------------------------------------------------------------

/// Simplify a triangle mesh using quadric error decimation (Garland-Heckbert).
///
/// `reduction_ratio` is the fraction of faces to remove: 0.0 = no change,
/// 0.5 = half the faces, 1.0 = maximum reduction.
///
/// Args:
///     mesh: Input triangle mesh.
///     reduction_ratio: Fraction of faces to remove, in [0, 1] (default 0.5).
#[pyfunction]
#[pyo3(signature = (mesh, reduction_ratio = 0.5))]
fn simplify_mesh(mesh: &PyTriangleMesh, reduction_ratio: f32) -> PyResult<PyTriangleMesh> {
    if !(0.0..=1.0).contains(&reduction_ratio) {
        return Err(PyValueError::new_err("reduction_ratio must be in [0, 1]"));
    }
    QuadricErrorSimplifier::new()
        .simplify(&mesh.inner, reduction_ratio)
        .map(|m| PyTriangleMesh { inner: m })
        .map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Mesh smoothing
// ---------------------------------------------------------------------------

/// Smooth a mesh with Laplacian smoothing.
///
/// Each vertex is iteratively moved toward the centroid of its one-ring
/// neighbours. Fast but causes mesh shrinkage over many iterations.
/// For volume-preserving smoothing prefer `smooth_mesh_taubin`.
///
/// Args:
///     mesh: Input triangle mesh.
///     iterations: Number of smoothing passes (default 10).
///     lambda_: Per-iteration blend factor in (0, 1] (default 0.5).
#[pyfunction]
#[pyo3(signature = (mesh, iterations = 10, lambda_ = 0.5))]
fn smooth_mesh_laplacian(
    mesh: &PyTriangleMesh,
    iterations: usize,
    lambda_: f32,
) -> PyResult<PyTriangleMesh> {
    let config = LaplacianSmoothingConfig {
        iterations,
        lambda: lambda_,
    };
    smooth_laplacian(&mesh.inner, &config)
        .map(|m| PyTriangleMesh { inner: m })
        .map_err(to_py_err)
}

/// Smooth a mesh with Taubin (μ|λ) smoothing.
///
/// Two alternating Laplacian passes per iteration: a positive λ pass followed
/// by a negative μ pass. Reduces noise without the volume shrinkage of plain
/// Laplacian smoothing.
///
/// Args:
///     mesh: Input triangle mesh.
///     iterations: Number of full (λ + μ) iterations (default 10).
///     lambda_: Positive step factor in (0, 1) (default 0.5).
///     mu: Negative step factor, must be < 0 (default -0.53).
#[pyfunction]
#[pyo3(signature = (mesh, iterations = 10, lambda_ = 0.5, mu = -0.53))]
fn smooth_mesh_taubin(
    mesh: &PyTriangleMesh,
    iterations: usize,
    lambda_: f32,
    mu: f32,
) -> PyResult<PyTriangleMesh> {
    let config = TaubinSmoothingConfig {
        iterations,
        lambda: lambda_,
        mu,
    };
    smooth_taubin(&mesh.inner, &config)
        .map(|m| PyTriangleMesh { inner: m })
        .map_err(to_py_err)
}

/// Smooth a mesh with HC (Humphrey's Classes) smoothing.
///
/// A two-phase algorithm: Laplacian step + backward correction toward original
/// positions. Less shrinkage than Laplacian while still reducing noise.
///
/// Args:
///     mesh: Input triangle mesh.
///     iterations: Number of smoothing iterations (default 10).
///     alpha: Blend toward original positions in [0, 1]; 0 = more smoothing (default 0.0).
///     beta: Balance per-vertex vs neighbour correction in [0, 1] (default 0.5).
#[pyfunction]
#[pyo3(signature = (mesh, iterations = 10, alpha = 0.0, beta = 0.5))]
fn smooth_mesh_hc(
    mesh: &PyTriangleMesh,
    iterations: usize,
    alpha: f32,
    beta: f32,
) -> PyResult<PyTriangleMesh> {
    let config = HcSmoothingConfig {
        iterations,
        alpha,
        beta,
    };
    smooth_hc(&mesh.inner, &config)
        .map(|m| PyTriangleMesh { inner: m })
        .map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Reconstruction
// ---------------------------------------------------------------------------

/// Automatically select and run the best surface reconstruction algorithm.
///
/// Analyses the point cloud characteristics and picks an appropriate algorithm.
/// For best results with noisy or organic data, use `poisson_reconstruct()`
/// after calling `estimate_normals()`.
#[pyfunction]
fn reconstruct(cloud: &PyPointCloud) -> PyResult<PyTriangleMesh> {
    auto_reconstruct(&cloud.inner)
        .map(|m| PyTriangleMesh { inner: m })
        .map_err(to_py_err)
}

/// Run Poisson surface reconstruction on a cloud that already has normals.
///
/// Produces a watertight mesh. Call `estimate_normals()` first if you have a
/// plain `PointCloud`.
#[pyfunction]
fn poisson_reconstruct(cloud: &PyNormalPointCloud) -> PyResult<PyTriangleMesh> {
    poisson_reconstruction_default(&cloud.inner)
        .map(|m| PyTriangleMesh { inner: m })
        .map_err(to_py_err)
}

/// Reconstruct a surface using the Ball Pivoting Algorithm (BPA).
///
/// A ball of `radius` rolls over the point cloud; wherever it touches three
/// points simultaneously a triangle is formed. Works best on uniformly sampled
/// point clouds without large noise.
///
/// Args:
///     cloud: Input point cloud.
///     radius: Ball radius in point-cloud units (default 0.1).
#[pyfunction]
#[pyo3(signature = (cloud, radius = 0.1))]
fn ball_pivoting_reconstruct(
    cloud: &PyPointCloud,
    radius: f32,
) -> PyResult<PyTriangleMesh> {
    tc_ball_pivoting(&cloud.inner, radius)
        .map(|m| PyTriangleMesh { inner: m })
        .map_err(to_py_err)
}

/// Reconstruct a surface using alpha shapes.
///
/// The alpha shape is a generalisation of the convex hull controlled by `alpha`:
/// smaller values produce tighter, more detailed shapes; larger values approach
/// the convex hull.
///
/// Args:
///     cloud: Input point cloud.
///     alpha: Shape parameter (default 1.0). Lower = tighter fit.
#[pyfunction]
#[pyo3(signature = (cloud, alpha = 1.0))]
fn alpha_shape_reconstruct(
    cloud: &PyPointCloud,
    alpha: f32,
) -> PyResult<PyTriangleMesh> {
    tc_alpha_shape(&cloud.inner, alpha)
        .map(|m| PyTriangleMesh { inner: m })
        .map_err(to_py_err)
}

/// Reconstruct a surface using Delaunay triangulation.
///
/// Projects the 3-D point cloud onto its best-fit plane using PCA, runs 2-D
/// Delaunay triangulation, and lifts the result back to 3-D. Fast and exact,
/// but only suitable for nearly planar point sets.
///
/// Args:
///     cloud: Input point cloud.
#[pyfunction]
fn delaunay_triangulate(cloud: &PyPointCloud) -> PyResult<PyTriangleMesh> {
    tc_delaunay(&cloud.inner)
        .map(|m| PyTriangleMesh { inner: m })
        .map_err(to_py_err)
}

/// Reconstruct a surface using Moving Least Squares (MLS).
///
/// Fits local polynomial surfaces to the point cloud and extracts a mesh from
/// the resulting implicit function. Handles noise well and produces smooth
/// surfaces. Slower than Delaunay or Ball Pivoting.
///
/// Args:
///     cloud: Input point cloud.
#[pyfunction]
fn moving_least_squares_reconstruct(cloud: &PyPointCloud) -> PyResult<PyTriangleMesh> {
    tc_moving_least_squares(&cloud.inner)
        .map(|m| PyTriangleMesh { inner: m })
        .map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Colorization
// ---------------------------------------------------------------------------

/// Colorize a point cloud from a registered RGB image.
///
/// Each 3-D point is projected onto the image plane using a pinhole camera
/// model. Points that project outside the image boundary or lie behind the
/// camera are assigned the `default_color` (grey by default).
///
/// Args:
///     cloud: Input point cloud.
///     image_data: Raw RGB image bytes, row-major (R, G, B, R, G, B, …).
///     width: Image width in pixels.
///     height: Image height in pixels.
///     fx: Horizontal focal length in pixels.
///     fy: Vertical focal length in pixels.
///     cx: Horizontal principal-point offset (pixels from left edge).
///     cy: Vertical principal-point offset (pixels from top edge).
///     world_to_camera: 4×4 float32 rigid transform mapping world coordinates
///         to the camera coordinate frame.
///
/// Returns
/// -------
/// ColoredPointCloud
#[pyfunction]
fn colorize_point_cloud(
    cloud: &PyPointCloud,
    image_data: &[u8],
    width: u32,
    height: u32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    world_to_camera: PyReadonlyArray2<f32>,
) -> PyResult<PyColoredPointCloud> {
    let intrinsics = CameraIntrinsics { fx, fy, cx, cy };
    let image = RgbImageView::new(image_data, width, height).map_err(to_py_err)?;
    let pose = numpy_to_isometry(Some(world_to_camera))?;
    let config = ColorizationConfig::default();
    tc_colorize_point_cloud(&cloud.inner, &image, &intrinsics, &pose, &config)
        .map(|r| PyColoredPointCloud { inner: r.cloud })
        .map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------

/// Read a point cloud from a file (PLY, PCD, XYZ, CSV, LAS, LAZ, E57).
///
/// The format is inferred from the file extension.
#[pyfunction]
fn read_point_cloud(path: &str) -> PyResult<PyPointCloud> {
    rs_read_pc(path)
        .map(|c| PyPointCloud { inner: c })
        .map_err(to_py_err)
}

/// Write a point cloud to a file (PLY, PCD, XYZ, CSV).
///
/// The format is inferred from the file extension.
#[pyfunction]
fn write_point_cloud(cloud: &PyPointCloud, path: &str) -> PyResult<()> {
    rs_write_pc(&cloud.inner, path).map_err(to_py_err)
}

/// Read a mesh from a file (PLY, OBJ).
///
/// The format is inferred from the file extension.
#[pyfunction]
fn read_mesh(path: &str) -> PyResult<PyTriangleMesh> {
    rs_read_mesh(path)
        .map(|m| PyTriangleMesh { inner: m })
        .map_err(to_py_err)
}

/// Write a mesh to a file (PLY, OBJ).
///
/// The format is inferred from the file extension.
#[pyfunction]
fn write_mesh(mesh: &PyTriangleMesh, path: &str) -> PyResult<()> {
    rs_write_mesh(&mesh.inner, path).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// PointCloud2 support
// ---------------------------------------------------------------------------

/// A coloured point cloud (XYZ + RGB).
/// Returned by `pointcloud2_to_colored()` and `colorize_point_cloud()`.
#[pyclass(name = "ColoredPointCloud")]
#[derive(Clone)]
pub struct PyColoredPointCloud {
    pub(crate) inner: PointCloud<ColoredPoint3f>,
}

#[pymethods]
impl PyColoredPointCloud {
    /// Point positions as a numpy array of shape (N, 3) float32.
    fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let n = self.inner.len();
        let mut data = Array2::<f32>::zeros((n, 3));
        for (i, p) in self.inner.points.iter().enumerate() {
            data[[i, 0]] = p.position.x;
            data[[i, 1]] = p.position.y;
            data[[i, 2]] = p.position.z;
        }
        data.into_pyarray_bound(py)
    }

    /// Colors as a numpy array of shape (N, 3) uint8 (R, G, B).
    fn colors<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        use numpy::ndarray::Array2 as NdArray2;
        let n = self.inner.len();
        let mut data = NdArray2::<u8>::zeros((n, 3));
        for (i, p) in self.inner.points.iter().enumerate() {
            data[[i, 0]] = p.color[0];
            data[[i, 1]] = p.color[1];
            data[[i, 2]] = p.color[2];
        }
        data.into_pyarray_bound(py)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("ColoredPointCloud({} points)", self.inner.len())
    }
}

/// A coloured point cloud with surface normals (XYZ + normal + RGB).
#[pyclass(name = "ColoredNormalPointCloud")]
#[derive(Clone)]
pub struct PyColoredNormalPointCloud {
    pub(crate) inner: PointCloud<ColoredNormalPoint3f>,
}

#[pymethods]
impl PyColoredNormalPointCloud {
    /// Point positions as a numpy array of shape (N, 3) float32.
    fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let n = self.inner.len();
        let mut data = Array2::<f32>::zeros((n, 3));
        for (i, p) in self.inner.points.iter().enumerate() {
            data[[i, 0]] = p.position.x;
            data[[i, 1]] = p.position.y;
            data[[i, 2]] = p.position.z;
        }
        data.into_pyarray_bound(py)
    }

    /// Surface normals as a numpy array of shape (N, 3) float32.
    fn normals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let n = self.inner.len();
        let mut data = Array2::<f32>::zeros((n, 3));
        for (i, p) in self.inner.points.iter().enumerate() {
            data[[i, 0]] = p.normal.x;
            data[[i, 1]] = p.normal.y;
            data[[i, 2]] = p.normal.z;
        }
        data.into_pyarray_bound(py)
    }

    /// Colors as a numpy array of shape (N, 3) uint8 (R, G, B).
    fn colors<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        use numpy::ndarray::Array2 as NdArray2;
        let n = self.inner.len();
        let mut data = NdArray2::<u8>::zeros((n, 3));
        for (i, p) in self.inner.points.iter().enumerate() {
            data[[i, 0]] = p.color[0];
            data[[i, 1]] = p.color[1];
            data[[i, 2]] = p.color[2];
        }
        data.into_pyarray_bound(py)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("ColoredNormalPointCloud({} points)", self.inner.len())
    }
}

/// The serialised form of a `sensor_msgs/PointCloud2` payload.
///
/// Attributes
/// ----------
/// data : bytes
///     Raw point bytes, row-major.
/// fields : list[tuple[str, int, int, int]]
///     Each tuple is ``(name, offset, datatype, count)``.
///     Datatype values: 1=int8, 2=uint8, 3=int16, 4=uint16, 5=int32,
///     6=uint32, 7=float32, 8=float64.
/// point_step : int
///     Byte stride per point.
/// width : int
///     Points per row (total points for unorganised clouds).
/// height : int
///     Number of rows (1 for unorganised clouds).
/// is_bigendian : bool
/// is_dense : bool
#[pyclass(name = "PointCloud2Data")]
#[derive(Clone)]
pub struct PyPointCloud2Data {
    pub(crate) inner: tc_ros2::PointCloud2Data,
}

#[pymethods]
impl PyPointCloud2Data {
    /// Raw bytes of the point data.
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new_bound(py, &self.inner.data)
    }

    /// Field descriptors as a list of ``(name, offset, datatype, count)`` tuples.
    fn fields(&self) -> Vec<(String, u32, u8, u32)> {
        self.inner
            .info
            .fields
            .iter()
            .map(|f| (f.name.clone(), f.offset, f.datatype, f.count))
            .collect()
    }

    #[getter]
    fn point_step(&self) -> u32 {
        self.inner.info.point_step
    }

    #[getter]
    fn row_step(&self) -> u32 {
        self.inner.info.row_step
    }

    #[getter]
    fn width(&self) -> u32 {
        self.inner.info.width
    }

    #[getter]
    fn height(&self) -> u32 {
        self.inner.info.height
    }

    #[getter]
    fn is_bigendian(&self) -> bool {
        self.inner.info.is_bigendian
    }

    #[getter]
    fn is_dense(&self) -> bool {
        self.inner.info.is_dense
    }

    fn __repr__(&self) -> String {
        format!(
            "PointCloud2Data({}×{} points, point_step={})",
            self.inner.info.width,
            self.inner.info.height,
            self.inner.info.point_step,
        )
    }
}

/// Build a `PointCloud2Info` from the flat Python field representation.
fn build_info(
    fields: Vec<(String, u32, u8, u32)>,
    point_step: u32,
    width: u32,
    height: u32,
    is_bigendian: bool,
    is_dense: bool,
) -> PointCloud2Info {
    PointCloud2Info {
        fields: fields
            .into_iter()
            .map(|(name, offset, datatype, count)| Ros2PointField {
                name,
                offset,
                datatype,
                count,
            })
            .collect(),
        point_step,
        row_step: point_step * width,
        width,
        height,
        is_bigendian,
        is_dense,
    }
}

/// Parse raw `PointCloud2` bytes into an XYZ `PointCloud`.
#[pyfunction]
#[pyo3(signature = (data, fields, point_step, width, height, is_bigendian = false, is_dense = true))]
fn pointcloud2_to_xyz(
    data: &[u8],
    fields: Vec<(String, u32, u8, u32)>,
    point_step: u32,
    width: u32,
    height: u32,
    is_bigendian: bool,
    is_dense: bool,
) -> PyResult<PyPointCloud> {
    let info = build_info(fields, point_step, width, height, is_bigendian, is_dense);
    tc_ros2::pointcloud2_to_xyz(data, &info)
        .map(|c| PyPointCloud { inner: c })
        .map_err(to_py_err)
}

/// Parse raw `PointCloud2` bytes into a `NormalPointCloud`.
///
/// Requires ``normal_x``, ``normal_y``, ``normal_z`` fields in addition to ``x``, ``y``, ``z``.
#[pyfunction]
#[pyo3(signature = (data, fields, point_step, width, height, is_bigendian = false, is_dense = true))]
fn pointcloud2_to_normals(
    data: &[u8],
    fields: Vec<(String, u32, u8, u32)>,
    point_step: u32,
    width: u32,
    height: u32,
    is_bigendian: bool,
    is_dense: bool,
) -> PyResult<PyNormalPointCloud> {
    let info = build_info(fields, point_step, width, height, is_bigendian, is_dense);
    tc_ros2::pointcloud2_to_normals(data, &info)
        .map(|c| PyNormalPointCloud { inner: c })
        .map_err(to_py_err)
}

/// Parse raw `PointCloud2` bytes into a `ColoredPointCloud`.
///
/// Requires an ``rgb`` or ``rgba`` field.  Alpha is discarded.
#[pyfunction]
#[pyo3(signature = (data, fields, point_step, width, height, is_bigendian = false, is_dense = true))]
fn pointcloud2_to_colored(
    data: &[u8],
    fields: Vec<(String, u32, u8, u32)>,
    point_step: u32,
    width: u32,
    height: u32,
    is_bigendian: bool,
    is_dense: bool,
) -> PyResult<PyColoredPointCloud> {
    let info = build_info(fields, point_step, width, height, is_bigendian, is_dense);
    tc_ros2::pointcloud2_to_colored(data, &info)
        .map(|c| PyColoredPointCloud { inner: c })
        .map_err(to_py_err)
}

/// Parse raw `PointCloud2` bytes into a `ColoredNormalPointCloud`.
///
/// Requires ``x``, ``y``, ``z``, ``normal_x``, ``normal_y``, ``normal_z``, and ``rgb``/``rgba``.
#[pyfunction]
#[pyo3(signature = (data, fields, point_step, width, height, is_bigendian = false, is_dense = true))]
fn pointcloud2_to_colored_normals(
    data: &[u8],
    fields: Vec<(String, u32, u8, u32)>,
    point_step: u32,
    width: u32,
    height: u32,
    is_bigendian: bool,
    is_dense: bool,
) -> PyResult<PyColoredNormalPointCloud> {
    let info = build_info(fields, point_step, width, height, is_bigendian, is_dense);
    tc_ros2::pointcloud2_to_colored_normals(data, &info)
        .map(|c| PyColoredNormalPointCloud { inner: c })
        .map_err(to_py_err)
}

/// Serialise a `PointCloud` (XYZ) to `PointCloud2` format.
///
/// Returns a `PointCloud2Data` with 12-byte point step (x, y, z as float32).
#[pyfunction]
fn xyz_to_pointcloud2(cloud: &PyPointCloud) -> PyPointCloud2Data {
    PyPointCloud2Data { inner: tc_ros2::xyz_to_pointcloud2(&cloud.inner) }
}

/// Serialise a `NormalPointCloud` to `PointCloud2` format.
///
/// Returns a `PointCloud2Data` with 24-byte point step.
#[pyfunction]
fn normals_to_pointcloud2(cloud: &PyNormalPointCloud) -> PyPointCloud2Data {
    PyPointCloud2Data { inner: tc_ros2::normals_to_pointcloud2(&cloud.inner) }
}

/// Serialise a `ColoredPointCloud` to `PointCloud2` format.
///
/// Returns a `PointCloud2Data` with 16-byte point step.
#[pyfunction]
fn colored_to_pointcloud2(cloud: &PyColoredPointCloud) -> PyPointCloud2Data {
    PyPointCloud2Data { inner: tc_ros2::colored_to_pointcloud2(&cloud.inner) }
}

/// Serialise a `ColoredNormalPointCloud` to `PointCloud2` format.
///
/// Returns a `PointCloud2Data` with 28-byte point step.
#[pyfunction]
fn colored_normals_to_pointcloud2(cloud: &PyColoredNormalPointCloud) -> PyPointCloud2Data {
    PyPointCloud2Data { inner: tc_ros2::colored_normals_to_pointcloud2(&cloud.inner) }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn threecrate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Types
    m.add_class::<PyPointCloud>()?;
    m.add_class::<PyNormalPointCloud>()?;
    m.add_class::<PyColoredPointCloud>()?;
    m.add_class::<PyColoredNormalPointCloud>()?;
    m.add_class::<PyTriangleMesh>()?;
    m.add_class::<PyIcpResult>()?;
    m.add_class::<PyGlobalRegistrationResult>()?;
    m.add_class::<PyNdtResult>()?;
    m.add_class::<PyPlaneSegmentationResult>()?;
    m.add_class::<PyPointCloud2Data>()?;
    m.add_class::<PyKdTree>()?;

    // Filtering
    m.add_function(wrap_pyfunction!(voxel_downsample, m)?)?;
    m.add_function(wrap_pyfunction!(remove_statistical_outliers, m)?)?;
    m.add_function(wrap_pyfunction!(remove_radius_outliers, m)?)?;

    // Normal estimation
    m.add_function(wrap_pyfunction!(py_estimate_normals, m)?)?;

    // Registration
    m.add_function(wrap_pyfunction!(icp, m)?)?;
    m.add_function(wrap_pyfunction!(icp_point_to_plane, m)?)?;
    m.add_function(wrap_pyfunction!(gicp, m)?)?;
    m.add_function(wrap_pyfunction!(kiss_icp, m)?)?;
    m.add_function(wrap_pyfunction!(global_registration, m)?)?;
    m.add_function(wrap_pyfunction!(global_registration_with_normals, m)?)?;
    m.add_function(wrap_pyfunction!(ndt_registration, m)?)?;

    // Feature extraction
    m.add_function(wrap_pyfunction!(extract_fpfh_features, m)?)?;

    // Segmentation
    m.add_function(wrap_pyfunction!(segment_plane, m)?)?;
    m.add_function(wrap_pyfunction!(extract_clusters, m)?)?;

    // Mesh boolean operations
    m.add_function(wrap_pyfunction!(mesh_union, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_intersection, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_difference, m)?)?;

    // Simplification
    m.add_function(wrap_pyfunction!(simplify_mesh, m)?)?;

    // Smoothing
    m.add_function(wrap_pyfunction!(smooth_mesh_laplacian, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_mesh_taubin, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_mesh_hc, m)?)?;

    // Reconstruction
    m.add_function(wrap_pyfunction!(reconstruct, m)?)?;
    m.add_function(wrap_pyfunction!(poisson_reconstruct, m)?)?;
    m.add_function(wrap_pyfunction!(ball_pivoting_reconstruct, m)?)?;
    m.add_function(wrap_pyfunction!(alpha_shape_reconstruct, m)?)?;
    m.add_function(wrap_pyfunction!(delaunay_triangulate, m)?)?;
    m.add_function(wrap_pyfunction!(moving_least_squares_reconstruct, m)?)?;

    // Colorization
    m.add_function(wrap_pyfunction!(colorize_point_cloud, m)?)?;

    // I/O
    m.add_function(wrap_pyfunction!(read_point_cloud, m)?)?;
    m.add_function(wrap_pyfunction!(write_point_cloud, m)?)?;
    m.add_function(wrap_pyfunction!(read_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(write_mesh, m)?)?;

    // PointCloud2 (ROS 2)
    m.add_function(wrap_pyfunction!(pointcloud2_to_xyz, m)?)?;
    m.add_function(wrap_pyfunction!(pointcloud2_to_normals, m)?)?;
    m.add_function(wrap_pyfunction!(pointcloud2_to_colored, m)?)?;
    m.add_function(wrap_pyfunction!(pointcloud2_to_colored_normals, m)?)?;
    m.add_function(wrap_pyfunction!(xyz_to_pointcloud2, m)?)?;
    m.add_function(wrap_pyfunction!(normals_to_pointcloud2, m)?)?;
    m.add_function(wrap_pyfunction!(colored_to_pointcloud2, m)?)?;
    m.add_function(wrap_pyfunction!(colored_normals_to_pointcloud2, m)?)?;

    Ok(())
}
