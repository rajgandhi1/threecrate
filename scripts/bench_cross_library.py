"""Cross-library point-cloud benchmarks for real datasets.

Example:
    python scripts/bench_cross_library.py --dataset KITTI=path/to/000000.bin --tasks read voxel normals icp

The script runs ThreeCrate through the Rust example binary and, when available,
Open3D, PDAL, and a user-supplied PCL benchmark executable. Missing libraries or
missing datasets are reported as unavailable instead of replaced with synthetic
data. It also writes a markdown report with a shared-task composite score.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def main() -> int:
    args = parse_args()
    rows: list[dict[str, str]] = []

    for dataset_name, dataset_path in args.dataset:
        path = Path(dataset_path)
        if not path.exists():
            rows.append(unavailable("dataset", "all", dataset_name, f"missing file: {path}"))
            continue

        for task in args.tasks:
            rows.extend(run_threecrate(args, dataset_name, path, task))
            rows.extend(run_open3d(args, dataset_name, path, task))
            rows.extend(run_pdal(args, dataset_name, path, task))
            rows.extend(run_pcl(args, dataset_name, path, task))

    write_csv(args.output, rows)
    write_markdown_report(args.markdown_output, rows, args)
    print_table(rows)
    print_score(rows)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Dataset file to benchmark. Repeat for KITTI, NuScenes, TUM RGB-D, Livox, etc.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["read", "voxel", "icp"],
        choices=[
            "read",
            "voxel",
            "normals",
            "icp",
            "icp_accuracy",
            "multiscale_icp",
            "gpu_voxel",
            "gpu_icp",
            "gpu_radius_outlier",
            "gpu_statistical_outlier",
            "gpu_normals",
            "gpu_knn",
        ],
    )
    parser.add_argument("--output", default="target/cross_library_benchmarks.csv")
    parser.add_argument("--markdown-output", default="target/cross_library_benchmark_report.md")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-points", default="20000")
    parser.add_argument("--voxel-size", type=float, default=0.2)
    parser.add_argument("--max-icp-iters", type=int, default=20)
    parser.add_argument(
        "--max-correspondence-distance",
        type=float,
        default=1.0,
        help="ICP correspondence cutoff for the icp_accuracy task (both libraries).",
    )
    parser.add_argument("--pcl-bench-exe", default=os.environ.get("PCL_BENCH_EXE"))
    argv = [arg for arg in sys.argv[1:] if arg != "`"]
    parsed = parser.parse_args(argv)
    parsed.dataset = [parse_dataset(value) for value in parsed.dataset]
    if not parsed.dataset:
        parser.error("at least one --dataset NAME=PATH is required")
    return parsed


def parse_dataset(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must be NAME=PATH")
    name, path = value.split("=", 1)
    return name, path


def run_threecrate(args: argparse.Namespace, dataset: str, path: Path, task: str) -> list[dict[str, str]]:
    cmd = [
        "cargo",
        "run",
        "--release",
        "-p",
        "threecrate-examples",
        "--bin",
        "threecrate_dataset_bench",
        "--",
        "--task",
        task,
        "--dataset",
        dataset,
        "--source",
        str(path),
        "--iterations",
        str(args.iterations),
        "--warmups",
        str(args.warmups),
        "--max-points",
        str(args.max_points),
        "--voxel-size",
        str(args.voxel_size),
        "--max-icp-iters",
        str(args.max_icp_iters),
        "--max-correspondence-distance",
        str(args.max_correspondence_distance),
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        return [unavailable("ThreeCrate", task, dataset, (exc.stderr or exc.stdout).strip())]
    return list(csv.DictReader(proc.stdout.splitlines()))


def run_open3d(args: argparse.Namespace, dataset: str, path: Path, task: str) -> list[dict[str, str]]:
    if task.startswith("gpu_") or task == "multiscale_icp":
        return [unavailable("Open3D", task, dataset, "ThreeCrate-specific task")]
    if importlib.util.find_spec("open3d") is None:
        return [unavailable("Open3D", task, dataset, "python package open3d is not installed")]
    import numpy as np
    import open3d as o3d

    try:
        source = limit_numpy(open3d_read(path, o3d, np), args.max_points)
        if task == "icp_accuracy":
            source, target = accuracy_pair_numpy(source, np)
        else:
            target = transform_numpy(source, np)
        for _ in range(args.warmups):
            open3d_task(task, source, target, args, o3d, np, path)
        times = []
        output_points = 0
        detail = ""
        for _ in range(args.iterations):
            start = time.perf_counter()
            output_points, detail = open3d_task(task, source, target, args, o3d, np, path)
            times.append((time.perf_counter() - start) * 1000.0)
        times.sort()
        return [result_row("Open3D", task, dataset, len(source), len(target), output_points, times, args.iterations, detail)]
    except Exception as exc:
        return [unavailable("Open3D", task, dataset, str(exc))]


def open3d_read(path: Path, o3d, np):
    if path.is_dir():
        return tum_sequence_to_points(path, o3d, np)
    if path.name.lower().endswith(".pcd.bin"):
        data = np.fromfile(path, dtype=np.float32).reshape((-1, 5))
        return data[:, :3].astype(np.float64)
    if path.suffix.lower() == ".bin":
        data = np.fromfile(path, dtype=np.float32).reshape((-1, 4))
        return data[:, :3].astype(np.float64)
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points)


def open3d_task(task: str, source, target, args: argparse.Namespace, o3d, np, path: Path) -> tuple[int, str]:
    if task == "read":
        points = open3d_read(path, o3d, np)
        return len(points), "read_point_cloud"
    source_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
    if task == "voxel":
        filtered = source_pcd.voxel_down_sample(args.voxel_size)
        return len(filtered.points), f"voxel_size={args.voxel_size}"
    if task == "normals":
        source_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=10))
        return len(source_pcd.normals), "k=10"
    target_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    if task == "icp_accuracy":
        result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            args.max_correspondence_distance,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=args.max_icp_iters),
        )
        rot_err_deg, trans_err_m = transform_error_numpy(np.asarray(result.transformation), np)
        return len(source), (
            f"rot_err_deg={rot_err_deg:.4f},trans_err_m={trans_err_m:.4f},"
            f"fitness={result.fitness:.6f},rmse={result.inlier_rmse:.6f}"
        )
    result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        1.0,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=args.max_icp_iters),
    )
    return len(source), f"fitness={result.fitness:.6f},rmse={result.inlier_rmse:.6f}"


def run_pdal(args: argparse.Namespace, dataset: str, path: Path, task: str) -> list[dict[str, str]]:
    if task.startswith("gpu_") or task == "multiscale_icp":
        return [unavailable("PDAL", task, dataset, "ThreeCrate-specific task")]
    if shutil.which("pdal") is None:
        return [unavailable("PDAL", task, dataset, "pdal CLI is not installed")]
    if task in ("icp", "icp_accuracy"):
        return [unavailable("PDAL", task, dataset, "PDAL is not an ICP baseline")]
    if task == "normals":
        return [unavailable("PDAL", task, dataset, "PDAL normals are not configured in this harness")]
    if path.suffix.lower() not in {".las", ".laz", ".e57", ".ply"}:
        return [unavailable("PDAL", task, dataset, f"PDAL path not configured for {path.suffix}")]
    pipeline = f'["{path.as_posix()}"'
    if task == "voxel":
        pipeline += f',{{"type":"filters.voxelcenternearestneighbor","cell":{args.voxel_size}}}'
    pipeline += "]"
    times = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        subprocess.run(["pdal", "pipeline", "--stdin"], input=pipeline, text=True, capture_output=True, check=True)
        times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    return [result_row("PDAL", task, dataset, "", "", "", times, args.iterations, "pdal pipeline")]


def run_pcl(args: argparse.Namespace, dataset: str, path: Path, task: str) -> list[dict[str, str]]:
    if task.startswith("gpu_") or task == "multiscale_icp":
        return [unavailable("PCL", task, dataset, "ThreeCrate-specific task")]
    if not args.pcl_bench_exe:
        return [unavailable("PCL", task, dataset, "set PCL_BENCH_EXE to a custom PCL benchmark executable")]
    cmd = [
        args.pcl_bench_exe,
        "--task", task,
        "--dataset", dataset,
        "--source", str(path),
        "--iterations", str(args.iterations),
        "--warmups", str(args.warmups),
        "--max-points", str(args.max_points),
        "--voxel-size", str(args.voxel_size),
        "--max-icp-iters", str(args.max_icp_iters),
        "--max-correspondence-distance", str(args.max_correspondence_distance),
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        return [unavailable("PCL", task, dataset, (exc.stderr or exc.stdout).strip())]
    return list(csv.DictReader(proc.stdout.splitlines()))


def limit_numpy(points, max_points: str):
    if max_points.lower() == "all" or max_points == "0":
        return points
    return points[: int(max_points)]


def transform_numpy(points, np):
    theta = 0.02
    rot = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    return points @ rot.T + np.array([0.05, -0.02, 0.01])


# Ground-truth motion for the icp_accuracy task: roughly one KITTI frame (10 Hz)
# of vehicle motion. Must match `accuracy_ground_truth` in
# examples/threecrate_dataset_bench.rs.
ACCURACY_TASKS = ("icp_accuracy",)
ACCURACY_GT_TRANSLATION = (0.30, -0.20, 0.10)
ACCURACY_GT_ROLL_PITCH_YAW = (0.01, -0.015, 0.05)


def accuracy_rotation(np):
    """Rz(yaw) @ Ry(pitch) @ Rx(roll), matching nalgebra's from_euler_angles."""
    roll, pitch, yaw = ACCURACY_GT_ROLL_PITCH_YAW
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(roll), -np.sin(roll)], [0.0, np.sin(roll), np.cos(roll)]])
    ry = np.array([[np.cos(pitch), 0.0, np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0.0, np.cos(pitch)]])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def accuracy_pair_numpy(points, np):
    """Source = even-indexed points; target = odd-indexed points moved by the
    ground truth. Same surfaces, no shared points."""
    source = points[0::2]
    target = points[1::2] @ accuracy_rotation(np).T + np.array(ACCURACY_GT_TRANSLATION)
    return source, target


def transform_error_numpy(transformation, np):
    """Rotation error (degrees) and translation error of a 4x4 estimate."""
    rot_est = transformation[:3, :3]
    cos_angle = (np.trace(rot_est.T @ accuracy_rotation(np)) - 1.0) / 2.0
    rot_err = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))
    trans_err = float(np.linalg.norm(transformation[:3, 3] - np.array(ACCURACY_GT_TRANSLATION)))
    return rot_err, trans_err


def tum_sequence_to_points(path: Path, o3d, np):
    rgb_rel = first_sequence_entry(path / "rgb.txt")
    depth_rel = first_sequence_entry(path / "depth.txt")
    color = o3d.io.read_image(str(path / rgb_rel))
    depth = o3d.io.read_image(str(path / depth_rel))
    rgbd = o3d.geometry.RGBDImage.create_from_tum_format(color, depth)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return np.asarray(pcd.points)


def first_sequence_entry(path: Path) -> str:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            trimmed = line.strip()
            if not trimmed or trimmed.startswith("#"):
                continue
            parts = trimmed.split()
            if len(parts) < 2:
                raise ValueError(f"invalid sequence line in {path}")
            return parts[1]
    raise ValueError(f"no sequence entries found in {path}")


def result_row(library, task, dataset, source_points, target_points, output_points, times, iterations, detail):
    return {
        "library": str(library),
        "task": str(task),
        "dataset": str(dataset),
        "source_points": str(source_points),
        "target_points": str(target_points),
        "output_points": str(output_points),
        "iterations": str(iterations),
        "median_ms": f"{times[len(times) // 2]:.3f}",
        "min_ms": f"{times[0]:.3f}",
        "mean_ms": f"{sum(times) / len(times):.3f}",
        "detail": str(detail),
    }


def unavailable(library: str, task: str, dataset: str, reason: str) -> dict[str, str]:
    return {
        "library": library,
        "task": task,
        "dataset": dataset,
        "source_points": "",
        "target_points": "",
        "output_points": "",
        "iterations": "",
        "median_ms": "n/a",
        "min_ms": "n/a",
        "mean_ms": "n/a",
        "detail": reason.replace("\n", " ")[:300],
    }


def write_csv(path: str, rows: list[dict[str, str]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(unavailable("", "", "", "").keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(path: str, rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    score = composite_score(rows)
    versions = detected_versions()
    lines = [
        "# ThreeCrate Cross-Library Point Cloud Benchmark",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Method",
        "",
        "- Lower time is better. Times are median milliseconds over the configured iterations.",
        f"- Iterations: {args.iterations}; warmups: {args.warmups}; max points: {args.max_points}; voxel size: {args.voxel_size}; max ICP iterations: {args.max_icp_iters}.",
        "- The `icp` speed task uses a synthetic near-identity transform of the same source cloud unless an explicit target is supplied; `icp_accuracy` is described in its own section.",
        "- Composite score includes only tasks where ThreeCrate and at least one non-ThreeCrate baseline produced numeric timings.",
        "- ThreeCrate-specific GPU tasks are reported separately and are not included in the composite score.",
        "- Missing PCL/PDAL values are not filled with numbers from other machines or papers.",
        "",
        "## Environment",
        "",
        f"- OS: {platform.platform()}",
        f"- Python: {platform.python_version()}",
        f"- Open3D: {versions.get('Open3D', 'not installed')}",
        f"- PDAL CLI: {versions.get('PDAL', 'not installed')}",
        f"- PCL benchmark executable: {args.pcl_bench_exe or 'not configured'}",
        "",
        "## Composite Score",
        "",
    ]
    if score["count"]:
        lines.extend(
            [
                f"ThreeCrate score: **{score['score']:.1f}**",
                "",
                "Interpretation: 100 means equal to the fastest runnable external baseline on shared tasks. Above 100 means ThreeCrate is faster on the geometric mean; below 100 means slower.",
                "",
                "| Task | Dataset | ThreeCrate best | External best | Speed ratio |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for item in score["items"]:
            lines.append(
                f"| {item['task']} | {item['dataset']} | {item['three_ms']:.3f} ms | {item['external_ms']:.3f} ms ({item['external_library']}) | {item['ratio']:.2f}x |"
            )
    else:
        lines.append("No shared runnable task had both ThreeCrate and an external numeric baseline.")

    lines.extend(accuracy_report_lines(rows, args))

    lines.extend(
        [
            "",
            "## Raw Results",
            "",
            "| Task | Dataset | Open3D | ThreeCrateCPU | ThreeCrateGPU | PCL | PDAL | Note |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for (task, dataset), libs in grouped_medians(rows):
        note = grouped_note(rows, task, dataset)
        lines.append(
            f"| {task} | {dataset} | {libs.get('Open3D', '')} | {libs.get('ThreeCrateCPU', '')} | {libs.get('ThreeCrateGPU', '')} | {libs.get('PCL', '')} | {libs.get('PDAL', '')} | {note} |"
        )

    lines.extend(
        [
            "",
            "## Scope Notes",
            "",
            "- Open3D comparison uses documented point cloud operations such as voxel downsampling, normal estimation, and registration ICP.",
            "- PCL has native VoxelGrid and ICP APIs, but this Windows harness requires a separately built `PCL_BENCH_EXE` so the exact same dataset/task loop can be timed locally.",
            "- PDAL is relevant for file/pipeline and voxel-style processing, especially LAS/LAZ/E57/PLY data. It is not treated as an ICP baseline here.",
            "- Livox `.lvx` support depends on whether the repo reader can decode the supplied file; converted PLY/PCD/LAS files are easier to compare across libraries.",
            "",
            "## Source References",
            "",
            "- Open3D 0.19 docs: https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html",
            "- PCL VoxelGrid docs: https://pointclouds.org/documentation/classpcl_1_1_voxel_grid.html",
            "- PCL ICP docs: https://pointclouds.org/documentation/classpcl_1_1_iterative_closest_point.html",
            "- PDAL pipeline docs: https://pdal.io/en/2.8.4/pipeline.html",
            "- PDAL voxel-center nearest-neighbor filter: https://pdal.io/en/2.8.4/stages/filters.voxelcenternearestneighbor.html",
            "- KITTI raw data: https://www.cvlibs.net/datasets/kitti/raw_data.php",
            "- TUM RGB-D download: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download",
            "- nuScenes paper: https://arxiv.org/abs/1903.11027",
            "- Livox data summary: https://livox-wiki-en.readthedocs.io/en/latest/data_summary/Livox_data_summary.html",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_detail(detail: str) -> dict[str, str]:
    """Parse a `key=value,key=value` detail string into a dict."""
    fields = {}
    for part in detail.split(","):
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields


def accuracy_report_lines(rows: list[dict[str, str]], args: argparse.Namespace) -> list[str]:
    accuracy_rows = [
        row for row in rows if row["task"] in ACCURACY_TASKS and row["median_ms"] != "n/a"
    ]
    if not accuracy_rows:
        return []
    roll, pitch, yaw = ACCURACY_GT_ROLL_PITCH_YAW
    lines = [
        "",
        "## ICP Accuracy",
        "",
        "Source = even-indexed points, target = odd-indexed points moved by a known "
        f"ground truth (translation {ACCURACY_GT_TRANSLATION} m; roll/pitch/yaw "
        f"{roll}/{pitch}/{yaw} rad). Identity initial guess, max correspondence distance "
        f"{args.max_correspondence_distance} m, at most {args.max_icp_iters} iterations. "
        "Each library uses its own convergence rule. Lower errors are better; fitness is "
        "the fraction of source points with a correspondence inside the cutoff.",
        "",
        "| Dataset | Library | Rotation error (deg) | Translation error (m) | Fitness | Inlier RMSE | Iterations | Time (ms) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(accuracy_rows, key=lambda r: (r["dataset"], r["library"])):
        fields = parse_detail(row["detail"])
        lines.append(
            f"| {row['dataset']} | {row['library']} | {fields.get('rot_err_deg', '')} | "
            f"{fields.get('trans_err_m', '')} | {fields.get('fitness', '')} | {fields.get('rmse', '')} | "
            f"{fields.get('icp_iters', 'n/a')} | {row['median_ms']} |"
        )
    return lines


def detected_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    if importlib.util.find_spec("open3d") is not None:
        import open3d as o3d

        versions["Open3D"] = getattr(o3d, "__version__", "installed")
    pdal = shutil.which("pdal")
    if pdal:
        try:
            proc = subprocess.run([pdal, "--version"], text=True, capture_output=True, check=True)
            versions["PDAL"] = (proc.stdout or proc.stderr).strip().splitlines()[0]
        except Exception:
            versions["PDAL"] = "installed"
    return versions


def composite_score(rows: list[dict[str, str]]) -> dict[str, object]:
    items = []
    for (task, dataset), libs in grouped_numeric(rows):
        # Accuracy runs use different clouds and stopping points per library, so
        # their timings are reported alongside the errors, not in the composite.
        if task in ACCURACY_TASKS:
            continue
        three = {name: ms for name, ms in libs.items() if name.startswith("ThreeCrate")}
        external = {name: ms for name, ms in libs.items() if not name.startswith("ThreeCrate") and name != "dataset"}
        if not three or not external:
            continue
        three_name, three_ms = min(three.items(), key=lambda item: item[1])
        external_name, external_ms = min(external.items(), key=lambda item: item[1])
        if three_ms <= 0.0 or external_ms <= 0.0:
            continue
        ratio = external_ms / three_ms
        items.append(
            {
                "task": task,
                "dataset": dataset,
                "three_library": three_name,
                "three_ms": three_ms,
                "external_library": external_name,
                "external_ms": external_ms,
                "ratio": ratio,
            }
        )
    if not items:
        return {"score": 0.0, "count": 0, "items": []}
    score = 100.0 * math.exp(sum(math.log(item["ratio"]) for item in items) / len(items))
    return {"score": score, "count": len(items), "items": items}


def grouped_medians(rows: list[dict[str, str]]) -> list[tuple[tuple[str, str], dict[str, str]]]:
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["task"], row["dataset"])
        by_key.setdefault(key, {})[row["library"]] = row["median_ms"]
    return sorted(by_key.items())


def grouped_numeric(rows: list[dict[str, str]]) -> list[tuple[tuple[str, str], dict[str, float]]]:
    by_key: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        try:
            median = float(row["median_ms"])
        except ValueError:
            continue
        key = (row["task"], row["dataset"])
        by_key.setdefault(key, {})[row["library"]] = median
    return sorted(by_key.items())


def grouped_note(rows: list[dict[str, str]], task: str, dataset: str) -> str:
    notes = [
        f"{row['library']}: {row['detail']}"
        for row in rows
        if row["task"] == task and row["dataset"] == dataset and row["median_ms"] == "n/a" and row["detail"]
    ]
    return "; ".join(notes)


def print_table(rows: list[dict[str, str]]) -> None:
    print("Task | Dataset | Open3D | ThreeCrateCPU | ThreeCrateGPU | PCL | PDAL | Note")
    print("--- | --- | ---: | ---: | ---: | ---: | ---: | ---")
    for (task, dataset), libs in grouped_medians(rows):
        note = grouped_note(rows, task, dataset)
        if task == "all" or all(
            not libs.get(name)
            for name in ("Open3D", "ThreeCrateCPU", "ThreeCrateGPU", "PCL", "PDAL")
        ):
            note = "dataset missing or no runnable backends"
        print(
            f"{task} | {dataset} | {libs.get('Open3D', '')} | {libs.get('ThreeCrateCPU', '')} | {libs.get('ThreeCrateGPU', '')} | {libs.get('PCL', '')} | {libs.get('PDAL', '')} | {note}"
        )


def print_score(rows: list[dict[str, str]]) -> None:
    score = composite_score(rows)
    if not score["count"]:
        print("\nComposite score: n/a")
        return
    print(f"\nComposite score: {score['score']:.1f} across {score['count']} shared task/dataset rows")


if __name__ == "__main__":
    raise SystemExit(main())
