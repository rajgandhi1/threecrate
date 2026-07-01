// PCL cross-library benchmark executable.
//
// Mirrors the task loop of `examples/threecrate_dataset_bench.rs` and the
// Open3D path in `scripts/bench_cross_library.py` so the same dataset/task can
// be timed under PCL. It deliberately uses the SAME parameters (point cap,
// voxel size, k for normals, ICP iteration count, and the same synthetic rigid
// target transform) as the other two libraries.
//
// Output: a single CSV row on stdout with the header expected by the harness:
//   library,task,dataset,source_points,target_points,output_points,iterations,
//   median_ms,min_ms,mean_ms,detail
//
// Supported tasks: read, voxel, normals, icp. Unsupported inputs (e.g. the TUM
// depth-frame directory, which PCL has no equivalent loader for in this harness)
// exit non-zero so the Python harness records them as "unavailable" rather than
// inventing a number.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/icp.h>
#include <pcl/search/kdtree.h>

using Cloud = pcl::PointCloud<pcl::PointXYZ>;

struct Args {
    std::string task = "read";
    std::string dataset = "dataset";
    std::string source;
    int iterations = 5;
    int warmups = 1;
    long max_points = 20000;  // negative => no cap ("all")
    float voxel_size = 0.2f;
    int max_icp_iters = 10;
    int normals_k = 10;
};

static bool ends_with(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    std::string a = s.substr(s.size() - suffix.size());
    std::string b = suffix;
    std::transform(a.begin(), a.end(), a.begin(), ::tolower);
    std::transform(b.begin(), b.end(), b.begin(), ::tolower);
    return a == b;
}

// Load a raw float32 LiDAR .bin file with `stride` floats per point, using the
// first three (x, y, z). Matches the KITTI (stride 4) and nuScenes .pcd.bin
// (stride 5) loaders in the Rust/Open3D harness.
static bool load_bin(const std::string& path, int stride, Cloud& cloud) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<float> buf(static_cast<size_t>(size) / sizeof(float));
    if (!f.read(reinterpret_cast<char*>(buf.data()), size)) return false;
    size_t n = buf.size() / static_cast<size_t>(stride);
    cloud.clear();
    cloud.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        cloud.emplace_back(buf[i * stride + 0], buf[i * stride + 1], buf[i * stride + 2]);
    }
    return true;
}

static bool load_source(const std::string& path, Cloud& cloud) {
    if (ends_with(path, ".pcd.bin")) return load_bin(path, 5, cloud);
    if (ends_with(path, ".bin")) return load_bin(path, 4, cloud);
    if (ends_with(path, ".pcd")) return pcl::io::loadPCDFile(path, cloud) == 0;
    if (ends_with(path, ".ply")) return pcl::io::loadPLYFile(path, cloud) == 0;
    return false;
}

static void cap_points(Cloud& cloud, long max_points) {
    if (max_points >= 0 && static_cast<long>(cloud.size()) > max_points) {
        cloud.resize(static_cast<size_t>(max_points));
        cloud.width = static_cast<uint32_t>(max_points);
        cloud.height = 1;
    }
}

// Same synthetic rigid transform used by the Rust/Open3D harness:
// translation (0.05, -0.02, 0.01) and a 0.02 rad rotation about z.
static Cloud::Ptr make_target(const Cloud& source) {
    Eigen::Affine3f t = Eigen::Affine3f::Identity();
    t.translation() << 0.05f, -0.02f, 0.01f;
    t.rotate(Eigen::AngleAxisf(0.02f, Eigen::Vector3f::UnitZ()));
    Cloud::Ptr out(new Cloud());
    out->reserve(source.size());
    for (const auto& p : source) {
        Eigen::Vector3f v = t * Eigen::Vector3f(p.x, p.y, p.z);
        out->emplace_back(v.x(), v.y(), v.z());
    }
    return out;
}

struct Outcome {
    size_t output_points = 0;
    std::string detail;
};

static Outcome run_task(const Args& args, const Cloud::Ptr& source, const Cloud::Ptr& target) {
    Outcome o;
    if (args.task == "read") {
        Cloud c;
        load_source(args.source, c);
        cap_points(c, args.max_points);
        o.output_points = c.size();
        o.detail = "read_point_cloud";
    } else if (args.task == "voxel") {
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(source);
        vg.setLeafSize(args.voxel_size, args.voxel_size, args.voxel_size);
        Cloud filtered;
        vg.filter(filtered);
        o.output_points = filtered.size();
        o.detail = "voxel_size=" + std::to_string(args.voxel_size);
    } else if (args.task == "normals") {
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(source);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        ne.setSearchMethod(tree);
        ne.setKSearch(args.normals_k);
        pcl::PointCloud<pcl::Normal> normals;
        ne.compute(normals);
        o.output_points = normals.size();
        o.detail = "k=" + std::to_string(args.normals_k);
    } else if (args.task == "icp") {
        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaximumIterations(args.max_icp_iters);
        icp.setMaxCorrespondenceDistance(1.0);  // matches Open3D's threshold
        Cloud aligned;
        icp.align(aligned);
        o.output_points = source->size();
        // NOTE: deliberately do NOT call icp.getFitnessScore() here — it runs an
        // extra full kd-tree pass over the cloud and would unfairly inflate PCL's
        // measured ICP time relative to Open3D/ThreeCrate, which report fitness
        // from already-computed state.
        o.detail = "converged=" + std::string(icp.hasConverged() ? "true" : "false");
    } else {
        throw std::runtime_error("unsupported task: " + args.task);
    }
    return o;
}

static std::string csv_escape(const std::string& v) {
    if (v.find(',') != std::string::npos || v.find('"') != std::string::npos ||
        v.find('\n') != std::string::npos) {
        std::string out = "\"";
        for (char c : v) {
            if (c == '"') out += "\"\"";
            else out += c;
        }
        out += "\"";
        return out;
    }
    return v;
}

int main(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        if (i + 1 >= argc) {
            std::cerr << "missing value for " << flag << "\n";
            return 2;
        }
        std::string value = argv[++i];
        if (flag == "--task") args.task = value;
        else if (flag == "--dataset") args.dataset = value;
        else if (flag == "--source") args.source = value;
        else if (flag == "--iterations") args.iterations = std::stoi(value);
        else if (flag == "--warmups") args.warmups = std::stoi(value);
        else if (flag == "--max-points") args.max_points = (value == "all" || value == "0") ? -1 : std::stol(value);
        else if (flag == "--voxel-size") args.voxel_size = std::stof(value);
        else if (flag == "--max-icp-iters") args.max_icp_iters = std::stoi(value);
        else { std::cerr << "unknown argument: " << flag << "\n"; return 2; }
    }

    if (args.source.empty()) {
        std::cerr << "--source is required\n";
        return 2;
    }

    Cloud::Ptr source(new Cloud());
    if (!load_source(args.source, *source) || source->empty()) {
        std::cerr << "PCL cannot load source (unsupported format or empty): " << args.source << "\n";
        return 3;
    }
    cap_points(*source, args.max_points);
    Cloud::Ptr target = make_target(*source);

    Outcome outcome;
    try {
        for (int i = 0; i < args.warmups; ++i) outcome = run_task(args, source, target);

        std::vector<double> times;
        times.reserve(args.iterations);
        for (int i = 0; i < args.iterations; ++i) {
            auto start = std::chrono::steady_clock::now();
            outcome = run_task(args, source, target);
            auto end = std::chrono::steady_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        std::sort(times.begin(), times.end());
        double median = times[times.size() / 2];
        double mn = times.front();
        double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

        std::cout << "library,task,dataset,source_points,target_points,output_points,"
                     "iterations,median_ms,min_ms,mean_ms,detail\n";
        std::cout.setf(std::ios::fixed);
        std::cout.precision(3);
        std::cout << "PCL," << args.task << "," << csv_escape(args.dataset) << ","
                  << source->size() << "," << target->size() << "," << outcome.output_points << ","
                  << args.iterations << "," << median << "," << mn << "," << mean << ","
                  << csv_escape(outcome.detail) << "\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 4;
    }
    return 0;
}
