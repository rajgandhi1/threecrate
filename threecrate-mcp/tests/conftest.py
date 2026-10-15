from pathlib import Path

import numpy as np
import pytest
import threecrate as tc

from threecrate_mcp.workspace import Workspace


def surface_points(n: int = 40, outliers: int = 20, seed: int = 0) -> np.ndarray:
    """A gently curved patch (spacing 0.05) plus a few far-away stray points."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(n) * 0.05, np.arange(n) * 0.05)
    zs = 0.1 * np.sin(xs * 3.0) * np.cos(ys * 2.0)
    patch = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    strays = rng.uniform(-3.0, 5.0, size=(outliers, 3))
    return np.vstack([patch, strays]).astype(np.float32)


def sphere_points(n: int = 1500, radius: float = 1.0) -> np.ndarray:
    """Evenly spread points on a sphere (Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return (radius * np.column_stack([
        np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    ])).astype(np.float32)


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    tc.write_point_cloud(tc.PointCloud(surface_points()), str(tmp_path / "scan.ply"))
    tc.write_point_cloud(tc.PointCloud(sphere_points()), str(tmp_path / "sphere.ply"))
    return Workspace(root=tmp_path.resolve())
