"""Make the sample scans in this folder.

    python samples/generate.py

Each file is generated (no third-party data), small, and has known answers,
so demos, tests and agent evals can check results. Needs numpy and threecrate.
Running it again gives the same files.
"""

from pathlib import Path

import numpy as np
import threecrate as tc

HERE = Path(__file__).parent
rng = np.random.default_rng(42)


def vase_points(n: int) -> np.ndarray:
    """A vase 12 cm tall standing on z = 0, scanned from the sides and top: the
    base is never seen, like a phone scan of an object on a table."""
    z = rng.uniform(0.0, 0.12, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = 0.035 + 0.012 * np.sin(z / 0.12 * 1.5 * np.pi)
    side = np.c_[r * np.cos(theta), r * np.sin(theta), z]
    # Closed, slightly domed top.
    m = n // 6
    rt = 0.035 * np.sqrt(rng.uniform(0, 1, m)) * (0.035 + 0.012 * np.sin(1.5 * np.pi)) / 0.035
    tt = rng.uniform(0, 2 * np.pi, m)
    top = np.c_[rt * np.cos(tt), rt * np.sin(tt), 0.12 + 0.004 * (1 - (rt / rt.max()) ** 2)]
    return np.vstack([side, top])


def table_scan() -> np.ndarray:
    """Vase + the table it stands on + scanner noise + stray points."""
    vase = vase_points(18_000)
    g = rng.uniform(-0.12, 0.12, (9_000, 2))
    table = np.c_[g, np.zeros(len(g))]
    cloud = np.vstack([vase, table])
    cloud += rng.normal(0, 0.0004, cloud.shape)  # 0.4 mm noise
    strays = rng.uniform([-0.15, -0.15, 0.0], [0.15, 0.15, 0.2], (60, 3))
    return np.vstack([cloud, strays]).astype(np.float32)


def street_lidar() -> np.ndarray:
    """A LiDAR-style street patch: sloped ground, a wall, a pole and a parked
    car-sized box, in metres (z up)."""
    parts = []
    xy = rng.uniform([-10, -10], [10, 10], (20_000, 2))
    parts.append(np.c_[xy, 0.02 * xy[:, 0]])  # 2% slope
    wy = rng.uniform(-10, 10, 6_000)
    wz = rng.uniform(0, 3, 6_000)
    parts.append(np.c_[np.full_like(wy, 10.0), wy, wz + 0.2])
    t = rng.uniform(0, 2 * np.pi, 2_000)
    pz = rng.uniform(0, 4, 2_000)
    parts.append(np.c_[-3 + 0.1 * np.cos(t), 2 + 0.1 * np.sin(t), pz - 0.06])
    # Car: the faces of a 4.2 x 1.8 x 1.4 m box, sitting on the ground.
    faces = []
    for axis in range(3):
        for side in (0, 1):
            p = rng.uniform([0, 0, 0], [4.2, 1.8, 1.4], (1_000, 3))
            p[:, axis] = [4.2, 1.8, 1.4][axis] * side
            faces.append(p)
    car = np.vstack(faces) + [2.0, -5.0, 0.04]
    parts.append(car[car[:, 2] > 0.05])  # the underside is not visible
    cloud = np.vstack(parts)
    cloud += rng.normal(0, 0.01, cloud.shape)  # 1 cm noise
    return cloud.astype(np.float32)


def broken_part() -> tc.TriangleMesh:
    """A closed rounded part, then broken the ways exported scans often are:
    holes, flipped faces and a floating fragment."""
    n = 6000
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    d = np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    # A rounded block 40 x 30 x 20 mm (superellipsoid), in millimetres.
    shape = np.sign(d) * np.abs(d) ** 0.6 * [20, 15, 10]
    normals = tc.estimate_normals(tc.PointCloud(shape.astype(np.float32)), 10)
    mesh = tc.poisson_reconstruct(tc.orient_normals(normals, 10), depth=6)
    v, f = np.asarray(mesh.vertices()), np.asarray(mesh.faces()).copy()
    centres = v[f].mean(axis=1)
    keep = ~((centres[:, 2] > 8.5) & (np.abs(centres[:, 0]) < 6))  # a hole in the top
    keep &= ~((centres[:, 0] > 18.5) & (np.abs(centres[:, 1]) < 4))  # a hole in one end
    f = f[keep]
    flip = rng.random(len(f)) < 0.05
    f[flip] = f[flip][:, [0, 2, 1]]
    # A small floating fragment 10 mm above.
    frag_v = np.array([[0, 0, 22], [3, 0, 22], [0, 3, 22], [3, 3, 22.5]], dtype=np.float32)
    frag_f = np.array([[0, 1, 2], [1, 3, 2]]) + len(v)
    return tc.TriangleMesh.from_numpy(
        np.vstack([v, frag_v]).astype(np.float32), np.vstack([f, frag_f]).astype(np.int32)
    )


if __name__ == "__main__":
    tc.write_point_cloud(tc.PointCloud(table_scan()), str(HERE / "table_scan.ply"))
    tc.write_point_cloud(tc.PointCloud(street_lidar()), str(HERE / "street_lidar.pcd"))
    tc.write_mesh(broken_part(), str(HERE / "broken_part.stl"))
    for name in ("table_scan.ply", "street_lidar.pcd", "broken_part.stl"):
        print(f"{name}: {(HERE / name).stat().st_size / 1e6:.2f} MB")
    # The broken part must stay broken, and fixable, or the evals mean nothing.
    part = tc.read_mesh(str(HERE / "broken_part.stl"))
    assert not tc.analyze_mesh(part)["printable"]
    assert tc.repair_for_print(part)[1]["after"]["printable"]
