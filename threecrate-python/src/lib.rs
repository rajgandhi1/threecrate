use nalgebra::Isometry3;
use numpy::{ndarray::Array2, IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use threecrate_algorithms::{
    estimate_normals as tc_estimate_normals, icp_point_to_point_default, radius_outlier_removal,
    statistical_outlier_removal, voxel_grid_filter,
};
use threecrate_core::{NormalPoint3f, Point3f, PointCloud, TriangleMesh};
use threecrate_io::{
    read_mesh as rs_read_mesh, read_point_cloud as rs_read_pc, write_mesh as rs_write_mesh,
    write_point_cloud as rs_write_pc,
};
use threecrate_reconstruction::{auto_reconstruct, poisson_reconstruction_default};

fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
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
        data.into_pyarray(py)
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
        data.into_pyarray(py)
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
        data.into_pyarray(py)
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
        data.into_pyarray(py)
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
        data.into_pyarray(py)
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

/// Result returned by `icp()`.
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
        data.into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "IcpResult(converged={}, mse={:.6}, iterations={})",
            self.converged, self.mse, self.iterations
        )
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
#[pyfunction]
#[pyo3(signature = (source, target, max_iterations = 50))]
fn icp(
    source: &PyPointCloud,
    target: &PyPointCloud,
    max_iterations: usize,
) -> PyResult<PyIcpResult> {
    icp_point_to_point_default(
        &source.inner,
        &target.inner,
        Isometry3::identity(),
        max_iterations,
    )
    .map(|r| {
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
    })
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
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn threecrate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Types
    m.add_class::<PyPointCloud>()?;
    m.add_class::<PyNormalPointCloud>()?;
    m.add_class::<PyTriangleMesh>()?;
    m.add_class::<PyIcpResult>()?;

    // Filtering
    m.add_function(wrap_pyfunction!(voxel_downsample, m)?)?;
    m.add_function(wrap_pyfunction!(remove_statistical_outliers, m)?)?;
    m.add_function(wrap_pyfunction!(remove_radius_outliers, m)?)?;

    // Normal estimation
    m.add_function(wrap_pyfunction!(py_estimate_normals, m)?)?;

    // Registration
    m.add_function(wrap_pyfunction!(icp, m)?)?;

    // Reconstruction
    m.add_function(wrap_pyfunction!(reconstruct, m)?)?;
    m.add_function(wrap_pyfunction!(poisson_reconstruct, m)?)?;

    // I/O
    m.add_function(wrap_pyfunction!(read_point_cloud, m)?)?;
    m.add_function(wrap_pyfunction!(write_point_cloud, m)?)?;
    m.add_function(wrap_pyfunction!(read_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(write_mesh, m)?)?;

    Ok(())
}
