import numpy as np
import pytest
import threecrate as tc

from threecrate_mcp import ops
from threecrate_mcp.workspace import Workspace, WorkspaceError

from .conftest import sphere_points, surface_points


def test_inspect_point_cloud(ws: Workspace):
    info = ops.inspect_point_cloud(ws, "scan.ply")
    assert info["points"] == 40 * 40 + 20
    assert info["median_spacing"] == pytest.approx(0.05, rel=0.2)
    assert info["suggested"]["voxel_size_light"] == pytest.approx(0.1, rel=0.2)
    assert len(info["min"]) == 3 and len(info["max"]) == 3


def test_convert_point_cloud(ws: Workspace):
    result = ops.convert(ws, "scan.ply", "scan.pcd")
    assert result["output"] == "scan.pcd"
    assert ops.inspect_point_cloud(ws, "scan.pcd")["points"] == result["points"]


def test_downsample_writes_new_file(ws: Workspace):
    result = ops.downsample(ws, "scan.ply", 0.15)
    assert result["output"] == "scan.downsampled.ply"
    assert 0 < result["points_after"] < result["points_before"]
    assert (ws.root / "scan.downsampled.ply").exists()


def test_downsample_rejects_bad_voxel(ws: Workspace):
    with pytest.raises(WorkspaceError):
        ops.downsample(ws, "scan.ply", 0.0)


def test_remove_outliers_drops_the_strays(ws: Workspace):
    result = ops.remove_outliers(ws, "scan.ply", method="radius", radius=0.15, min_neighbors=3)
    assert result["removed"] >= 15  # most of the 20 far-away points
    assert result["points_after"] >= 40 * 40 - 5  # the patch itself survives


def test_radius_method_needs_radius(ws: Workspace):
    with pytest.raises(WorkspaceError, match="positive radius"):
        ops.remove_outliers(ws, "scan.ply", method="radius")


def test_estimate_normals_writes_ply_with_normals(ws: Workspace):
    result = ops.estimate_normals(ws, "sphere.ply")
    out = ws.root / result["output"]
    header = out.read_bytes().split(b"end_header")[0]
    assert b"property float nx" in header
    assert ops.inspect_point_cloud(ws, result["output"])["points"] == 1500


def _rigid(yaw: float, t: tuple[float, float, float]) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    m = np.eye(4)
    m[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    m[:3, 3] = t
    return m


@pytest.mark.parametrize("method", ["icp", "point_to_plane", "gicp"])
def test_register_recovers_known_motion(ws: Workspace, method: str):
    # Small enough (about 1.5 cm at the patch edge, under the 5 cm spacing) that
    # point-to-point ICP can't lock onto the wrong grid neighbour.
    truth = _rigid(0.005, (0.01, -0.005, 0.005))
    patch = surface_points(outliers=0)
    moved = (patch @ truth[:3, :3].T + truth[:3, 3]).astype(np.float32)
    tc.write_point_cloud(tc.PointCloud(patch), str(ws.root / "a.ply"))
    tc.write_point_cloud(tc.PointCloud(moved), str(ws.root / "b.ply"))

    result = ops.register(ws, "a.ply", "b.ply", method=method, aligned_output="a.aligned.ply")
    estimate = np.array(result["transform"])
    error = np.abs((patch @ estimate[:3, :3].T + estimate[:3, 3]) - moved).max()
    assert error < 0.01, f"{method} is {error} off"
    assert result["aligned_output"] == "a.aligned.ply"
    assert result["rmse"] < 0.01


def test_transform_round_trip(ws: Workspace):
    shift = _rigid(0.0, (1.0, 2.0, 3.0))
    result = ops.transform(ws, "sphere.ply", shift.tolist())
    assert result["centroid"] == pytest.approx([1.0, 2.0, 3.0], abs=0.02)


def test_transform_rejects_non_rigid_matrix(ws: Workspace):
    bad = np.eye(4)
    bad[3] = [1, 0, 0, 1]
    with pytest.raises(WorkspaceError, match="last row"):
        ops.transform(ws, "sphere.ply", bad.tolist())


def test_segment_plane_splits_cloud(ws: Workspace):
    flat = surface_points(outliers=0)
    flat[:, 2] = 0.0
    blob = np.random.default_rng(1).normal([1.0, 1.0, 0.5], 0.05, size=(300, 3)).astype(np.float32)
    tc.write_point_cloud(tc.PointCloud(np.vstack([flat, blob])), str(ws.root / "table.ply"))

    result = ops.segment_plane(ws, "table.ply", distance_threshold=0.01,
                               inliers_output="floor.ply", outliers_output="objects.ply")
    assert abs(result["plane"]["c"]) > 0.99  # a horizontal plane
    assert result["inliers"] >= 40 * 40 - 5
    objects = ops.inspect_point_cloud(ws, "objects.ply")["points"]
    assert objects + result["inliers"] == 40 * 40 + 300


def test_extract_clusters_finds_two_blobs(ws: Workspace):
    rng = np.random.default_rng(2)
    blobs = np.vstack([
        rng.normal([0, 0, 0], 0.05, size=(400, 3)),
        rng.normal([3, 0, 0], 0.05, size=(300, 3)),
    ]).astype(np.float32)
    tc.write_point_cloud(tc.PointCloud(blobs), str(ws.root / "blobs.ply"))

    result = ops.extract_clusters(ws, "blobs.ply", tolerance=0.2, min_cluster_size=50, output_dir="parts")
    assert result["clusters"] == 2
    assert sorted(result["sizes"], reverse=True) == result["sizes"]
    assert len(result["files"]) == 2


@pytest.mark.parametrize("method", ["auto", "poisson", "ball_pivoting"])
def test_reconstruct_sphere(ws: Workspace, method: str):
    result = ops.reconstruct(ws, "sphere.ply", method=method, output=f"sphere.{method}.ply")
    assert result["faces"] > 100
    assert ops.inspect_mesh(ws, result["output"])["faces"] == result["faces"]


def test_poisson_sphere_is_accurate(ws: Workspace):
    result = ops.reconstruct(ws, "sphere.ply", method="poisson", depth=6)
    mesh = tc.read_mesh(str(ws.root / result["output"]))
    radius = np.linalg.norm(np.asarray(mesh.vertices()), axis=1)
    assert np.abs(radius - 1.0).mean() < 0.02


def test_simplify_and_smooth_mesh(ws: Workspace):
    mesh = ops.reconstruct(ws, "sphere.ply", method="poisson", depth=5)
    simplified = ops.simplify_mesh(ws, mesh["output"], 0.5)
    assert simplified["faces_after"] < simplified["faces_before"]
    smoothed = ops.smooth_mesh(ws, simplified["output"], "taubin", 5)
    assert smoothed["faces"] == simplified["faces_after"]


def test_simplify_explains_a_mesh_full_of_holes(ws: Workspace):
    mesh = ops.reconstruct(ws, "sphere.ply", method="ball_pivoting")
    with pytest.raises(Exception, match="preserve_boundary"):
        ops.simplify_mesh(ws, mesh["output"], 0.5)
    result = ops.simplify_mesh(ws, mesh["output"], 0.5, preserve_boundary=False, overwrite=True)
    assert result["faces_after"] < result["faces_before"]


def test_outputs_never_replace_inputs(ws: Workspace):
    with pytest.raises(WorkspaceError, match="overwrite the input"):
        ops.downsample(ws, "scan.ply", 0.1, output="scan.ply", overwrite=True)


def test_existing_output_needs_overwrite(ws: Workspace):
    ops.downsample(ws, "scan.ply", 0.1)
    with pytest.raises(WorkspaceError, match="already exists"):
        ops.downsample(ws, "scan.ply", 0.1)
    assert ops.downsample(ws, "scan.ply", 0.2, overwrite=True)["voxel_size"] == 0.2


def test_oversized_cloud_is_refused_before_loading(tmp_path):
    # The header claims a billion points but there is no data: the limit must
    # trip on the header, before threecrate tries to read anything.
    (tmp_path / "huge.ply").write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1000000000\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    small = Workspace(root=tmp_path.resolve(), max_points=1000)
    with pytest.raises(WorkspaceError, match="above the limit"):
        ops.inspect_point_cloud(small, "huge.ply")


def test_existing_output_fails_before_the_work(ws: Workspace, monkeypatch):
    (ws.root / "sphere.mesh.ply").write_bytes(b"keep me")

    def boom(*args, **kwargs):
        raise AssertionError("reconstruction ran before the output path was checked")

    monkeypatch.setattr(tc, "poisson_reconstruct", boom)
    with pytest.raises(WorkspaceError, match="already exists"):
        ops.reconstruct(ws, "sphere.ply", method="poisson")
    assert (ws.root / "sphere.mesh.ply").read_bytes() == b"keep me"


def test_segment_plane_checks_outputs_before_writing(ws: Workspace):
    with pytest.raises(WorkspaceError, match="different files"):
        ops.segment_plane(ws, "scan.ply", inliers_output="same.ply", outliers_output="same.ply", overwrite=True)
    assert not (ws.root / "same.ply").exists()

    (ws.root / "rest.ply").write_bytes(b"keep me")
    with pytest.raises(WorkspaceError, match="already exists"):
        ops.segment_plane(ws, "scan.ply", inliers_output="floor.ply", outliers_output="rest.ply")
    assert not (ws.root / "floor.ply").exists()


def test_extract_clusters_writes_nothing_if_any_file_exists(ws: Workspace):
    rng = np.random.default_rng(2)
    blobs = np.vstack([rng.normal([0, 0, 0], 0.05, (400, 3)), rng.normal([3, 0, 0], 0.05, (300, 3))])
    tc.write_point_cloud(tc.PointCloud(blobs.astype(np.float32)), str(ws.root / "blobs.ply"))
    (ws.root / "parts").mkdir()
    (ws.root / "parts" / "blobs.cluster001.ply").write_bytes(b"keep me")
    with pytest.raises(WorkspaceError, match="already exists"):
        ops.extract_clusters(ws, "blobs.ply", tolerance=0.2, min_cluster_size=50, output_dir="parts")
    assert not (ws.root / "parts" / "blobs.cluster000.ply").exists()


def test_register_rejects_bad_voxel_size(ws: Workspace):
    with pytest.raises(WorkspaceError, match="voxel_size must be positive"):
        ops.register(ws, "sphere.ply", "scan.ply", voxel_size=-0.1)


def _saved_normals(path):
    data = path.read_bytes()
    body = data[data.index(b"end_header\n") + len(b"end_header\n"):]
    return np.frombuffer(body, dtype="<f4").reshape(-1, 6)


def test_estimate_normals_orientation(ws: Workspace):
    consistent = ops.estimate_normals(ws, "sphere.ply", output="c.ply")
    viewpoint = ops.estimate_normals(ws, "sphere.ply", output="v.ply", normal_orientation="viewpoint")
    assert consistent["normal_orientation"] == "consistent"

    def outward(name):
        rows = _saved_normals(ws.root / name)
        return float(((rows[:, :3] * rows[:, 3:]).sum(axis=1) > 0).mean())

    assert outward("c.ply") == 1.0  # the sphere is centred at the origin
    assert outward("v.ply") < 0.9  # one viewpoint can't face every side
    assert viewpoint["output"] == "v.ply"


@pytest.mark.parametrize("bad", [
    [[1e39, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],  # inf in float32: used to hang
    [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]],  # scale: used to be dropped silently
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],  # mirror
])
def test_transform_rejects_non_rigid_or_non_finite(ws: Workspace, bad):
    with pytest.raises(ValueError, match="NaN or infinite|must be rigid"):
        ops.transform(ws, "sphere.ply", bad)


def test_inspect_mesh_reports_health_and_measurements(ws: Workspace):
    closed = ops.reconstruct(ws, "sphere.ply", method="poisson", depth=6, output="closed.ply")
    assert closed["watertight"] and closed["printable"] and closed["holes"] == 0
    report = ops.inspect_mesh(ws, "closed.ply")
    assert report["printable"] and report["problems"] == []
    assert report["volume"] == pytest.approx(4 / 3 * np.pi, rel=0.02)  # unit sphere
    assert report["surface_area"] == pytest.approx(4 * np.pi, rel=0.02)
    assert report["size"] == pytest.approx([2, 2, 2], rel=0.02)

    ops.reconstruct(ws, "sphere.ply", method="ball_pivoting", output="open.ply")
    report = ops.inspect_mesh(ws, "open.ply")
    assert not report["printable"] and report["volume"] is None
    assert report["holes_count"] > 0 and len(report["holes"]) <= 10
    assert any("hole" in p for p in report["problems"])


def test_render_draws_meshes_and_clouds_and_can_save(ws: Workspace):
    ops.reconstruct(ws, "sphere.ply", method="poisson", depth=5, output="mesh.ply")
    summary, png = ops.render(ws, ["sphere.ply", "mesh.ply"], size=96, output="compare.png")
    assert png[1:4] == b"PNG"
    assert [row["kind"] for row in summary["rows"]] == ["point_cloud", "mesh"]
    assert (ws.root / "compare.png").read_bytes() == png
    with pytest.raises(WorkspaceError, match="already exists"):
        ops.render(ws, ["mesh.ply"], output="compare.png")
    with pytest.raises(Exception, match="unknown view"):
        ops.render(ws, ["mesh.ply"], views=["sideways"])


def test_repair_for_print_makes_an_open_scan_printable(ws: Workspace):
    # Open-bottomed dome on a floor, plus stray points: a typical table-top scan.
    dome = sphere_points()
    dome = dome[dome[:, 2] > -0.4] + [0, 0, 1.4]
    g = np.stack(np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50)), -1).reshape(-1, 2)
    floor = np.c_[g, np.zeros(len(g))]
    tc.write_point_cloud(tc.PointCloud(np.vstack([dome, floor]).astype(np.float32)), str(ws.root / "scan.xyz"))

    summary, png = ops.repair_for_print(ws, "scan.xyz", remove_floor=True, depth=6, target_faces=5000)
    assert summary["output"] == "scan.printable.stl"
    assert summary["before"] is None and summary["after"]["printable"], summary
    assert summary["after"]["faces"] <= 5000
    assert any(step.startswith("removed the floor") for step in summary["steps"])
    assert png[1:4] == b"PNG"
    mesh = tc.read_mesh(str(ws.root / "scan.printable.stl"))
    assert tc.analyze_mesh(mesh)["watertight"]


def test_repair_for_print_fixes_a_mesh_full_of_holes(ws: Workspace):
    ops.reconstruct(ws, "sphere.ply", method="ball_pivoting", output="holey.ply")
    assert not ops.inspect_mesh(ws, "holey.ply")["printable"]
    summary, _ = ops.repair_for_print(ws, "holey.ply", depth=6, output="fixed.ply")
    assert summary["before"]["holes"] > 0
    assert summary["after"]["printable"], summary


def test_repair_for_print_writes_millimetres(ws: Workspace):
    # A 12 cm dome in metres, as phone scan apps export it.
    dome = sphere_points()
    dome = dome[dome[:, 2] > -0.4] * 0.06
    tc.write_point_cloud(tc.PointCloud(dome.astype(np.float32)), str(ws.root / "vase.xyz"))
    summary, png = ops.repair_for_print(ws, "vase.xyz", depth=6)
    assert 100 < max(summary["after"]["size"]) < 130, summary
    assert "x1000" in summary["units"]
    assert png[1:4] == b"PNG"
    summary, _ = ops.repair_for_print(ws, "vase.xyz", depth=6, units="mm", output="tiny.stl")
    assert max(summary["after"]["size"]) < 1
