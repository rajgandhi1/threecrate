//! GPU-resident kd-tree shared by the k-NN, normals, and ICP kernels.
//!
//! The tree is built on the CPU with `threecrate_algorithms::KdTree` (fast and
//! parallel) and uploaded once. Kernels then walk it with a small fixed-size
//! stack, so each query costs about `log n` node visits instead of the `n` a
//! brute-force scan needs.
//!
//! Points are moved into a frame centred on the cloud before upload. Far from
//! the origin, `f32` cannot represent small offsets; centring keeps full
//! precision for distances and covariances.

use crate::GpuContext;
use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
use threecrate_algorithms::{KdTree, LocalFrame};
use threecrate_core::{Point3f, Result};

/// WGSL shared by every kernel that searches the tree. Binding 0 of group 0 is
/// the node array; kernels add their own bindings from 1 up.
pub(crate) const KD_TREE_WGSL: &str = r#"
struct Node {
    pos: vec3<f32>,
    axis: u32,
    index: u32,
    left: u32,
    right: u32,
    _pad: u32,
}

const NO_NODE: u32 = 0xffffffffu;
// Median-split trees with u32 node indices are at most 33 levels deep; the
// stack holds one deferred far child per level plus the current near child.
// Kept small: these arrays live in each GPU thread's registers.
const MAX_STACK: u32 = 36u;
const MAX_K: u32 = 32u;

@group(0) @binding(0) var<storage, read> nodes: array<Node>;

// Result of `find_k_nearest`: node positions and squared distances, closest
// first. Only the first `knn_count` entries are valid.
var<private> knn_node: array<u32, MAX_K>;
var<private> knn_dist: array<f32, MAX_K>;
var<private> knn_count: u32;

// Squared distance a candidate must beat: the k-th best once we have k.
fn knn_bound(k: u32, limit_sq: f32) -> f32 {
    if (knn_count < k) {
        return limit_sq;
    }
    return knn_dist[k - 1u];
}

// Find the `k` (<= MAX_K) nodes nearest to `query` that are closer than
// sqrt(limit_sq).
fn find_k_nearest(query: vec3<f32>, k: u32, limit_sq: f32) {
    knn_count = 0u;
    if (k == 0u || arrayLength(&nodes) == 0u) {
        return;
    }
    var stack_node: array<u32, MAX_STACK>;
    var stack_plane: array<f32, MAX_STACK>;
    stack_node[0] = 0u;
    stack_plane[0] = 0.0;
    var len = 1u;

    loop {
        if (len == 0u) {
            break;
        }
        len -= 1u;
        // Re-check on pop: the bound may have shrunk since this was pushed.
        if (stack_plane[len] >= knn_bound(k, limit_sq)) {
            continue;
        }
        let ni = stack_node[len];
        let node = nodes[ni];
        let d = node.pos - query;
        let dist_sq = dot(d, d);

        if (dist_sq < knn_bound(k, limit_sq)) {
            // Insert into the sorted list, dropping the current k-th if full.
            var slot = min(knn_count, k - 1u);
            if (knn_count < k) {
                knn_count += 1u;
            }
            loop {
                if (slot == 0u || knn_dist[slot - 1u] <= dist_sq) {
                    break;
                }
                knn_dist[slot] = knn_dist[slot - 1u];
                knn_node[slot] = knn_node[slot - 1u];
                slot -= 1u;
            }
            knn_dist[slot] = dist_sq;
            knn_node[slot] = ni;
        }

        let axis_dist = query[node.axis] - node.pos[node.axis];
        var near = node.left;
        var far = node.right;
        if (axis_dist > 0.0) {
            near = node.right;
            far = node.left;
        }
        let plane_sq = axis_dist * axis_dist;
        // Push far before near so near is popped first.
        if (far != NO_NODE && plane_sq < knn_bound(k, limit_sq)) {
            stack_node[len] = far;
            stack_plane[len] = plane_sq;
            len += 1u;
        }
        if (near != NO_NODE) {
            stack_node[len] = near;
            stack_plane[len] = 0.0;
            len += 1u;
        }
    }
}

// Number of nodes other than `skip` within sqrt(radius_sq) of `center`,
// stopping early once `stop_at` is reached. A negative `radius_sq` matches
// nothing.
fn count_within(center: vec3<f32>, radius_sq: f32, skip: u32, stop_at: u32) -> u32 {
    var count = 0u;
    if (arrayLength(&nodes) == 0u || radius_sq < 0.0) {
        return 0u;
    }
    var stack_node: array<u32, MAX_STACK>;
    var stack_plane: array<f32, MAX_STACK>;
    stack_node[0] = 0u;
    stack_plane[0] = 0.0;
    var len = 1u;

    loop {
        if (len == 0u || count >= stop_at) {
            break;
        }
        len -= 1u;
        if (stack_plane[len] > radius_sq) {
            continue;
        }
        let ni = stack_node[len];
        let node = nodes[ni];
        let d = node.pos - center;
        if (ni != skip && dot(d, d) <= radius_sq) {
            count += 1u;
        }

        let axis_dist = center[node.axis] - node.pos[node.axis];
        var near = node.left;
        var far = node.right;
        if (axis_dist > 0.0) {
            near = node.right;
            far = node.left;
        }
        let plane_sq = axis_dist * axis_dist;
        // The far side can only hold matches if the split plane is in range.
        if (far != NO_NODE && plane_sq <= radius_sq) {
            stack_node[len] = far;
            stack_plane[len] = plane_sq;
            len += 1u;
        }
        if (near != NO_NODE) {
            stack_node[len] = near;
            stack_plane[len] = 0.0;
            len += 1u;
        }
    }
    return count;
}

struct Nearest {
    node: u32,
    dist_sq: f32,
}

// Nearest node to `query` closer than sqrt(bound_sq). `seed` is a node known to
// be close (e.g. last ICP iteration's match) or NO_NODE; it only speeds up the
// search, the result is still exact.
fn find_nearest(query: vec3<f32>, bound_sq: f32, seed: u32) -> Nearest {
    var best = Nearest(NO_NODE, bound_sq);
    if (arrayLength(&nodes) == 0u) {
        return best;
    }
    if (seed != NO_NODE && seed < arrayLength(&nodes)) {
        let d = nodes[seed].pos - query;
        let dist_sq = dot(d, d);
        if (dist_sq < best.dist_sq) {
            best = Nearest(seed, dist_sq);
        }
    }
    var stack_node: array<u32, MAX_STACK>;
    var stack_plane: array<f32, MAX_STACK>;
    stack_node[0] = 0u;
    stack_plane[0] = 0.0;
    var len = 1u;

    loop {
        if (len == 0u) {
            break;
        }
        len -= 1u;
        if (stack_plane[len] >= best.dist_sq) {
            continue;
        }
        let ni = stack_node[len];
        let node = nodes[ni];
        let d = node.pos - query;
        let dist_sq = dot(d, d);
        if (dist_sq < best.dist_sq) {
            best = Nearest(ni, dist_sq);
        }

        let axis_dist = query[node.axis] - node.pos[node.axis];
        var near = node.left;
        var far = node.right;
        if (axis_dist > 0.0) {
            near = node.right;
            far = node.left;
        }
        let plane_sq = axis_dist * axis_dist;
        if (far != NO_NODE && plane_sq < best.dist_sq) {
            stack_node[len] = far;
            stack_plane[len] = plane_sq;
            len += 1u;
        }
        if (near != NO_NODE) {
            stack_node[len] = near;
            stack_plane[len] = 0.0;
            len += 1u;
        }
    }
    return best;
}
"#;

/// Largest `k` the GPU k-NN search supports (`MAX_K` in [`KD_TREE_WGSL`]).
pub(crate) const MAX_K: usize = 32;

/// One kd-tree node as laid out in the GPU buffer (32 bytes, `Node` in WGSL).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuNode {
    pos: [f32; 3],
    axis: u32,
    index: u32,
    left: u32,
    right: u32,
    _pad: u32,
}

/// `p` as a GPU-friendly `vec4` (w unused).
pub(crate) fn vec4(p: &Point3f) -> [f32; 4] {
    [p.x, p.y, p.z, 0.0]
}

/// `points` as GPU-friendly `vec4`s (in parallel).
pub(crate) fn vec4s(points: &[Point3f]) -> Vec<[f32; 4]> {
    points.par_iter().map(vec4).collect()
}

/// A kd-tree uploaded to the GPU.
pub(crate) struct GpuKdTree {
    pub(crate) nodes: wgpu::Buffer,
    /// Original point index for each node position (to map GPU results back).
    /// Empty when no input point was finite; callers must then skip the GPU.
    pub(crate) node_to_index: Vec<u32>,
    pub(crate) frame: LocalFrame,
}

impl GpuKdTree {
    /// Build a kd-tree over `points`, moved into a frame centred on them, and
    /// upload it. Kernels then work in that frame (see `frame`).
    pub(crate) fn new(gpu: &GpuContext, points: &[Point3f]) -> Result<Self> {
        let frame = LocalFrame::centred_on(points);
        let tree = KdTree::from_points(frame.points_to_local(points))?;

        let gpu_nodes: Vec<GpuNode> = tree
            .flat_nodes()
            .map(|node| GpuNode {
                pos: [node.point.x, node.point.y, node.point.z],
                axis: node.axis as u32,
                index: node.index as u32,
                left: node.left,
                right: node.right,
                _pad: 0,
            })
            .collect();
        let node_to_index = gpu_nodes.iter().map(|n| n.index).collect();

        // wgpu rejects zero-sized buffers. An empty tree still gets one node,
        // but `is_empty` tells callers not to search it.
        let contents: &[GpuNode] = if gpu_nodes.is_empty() {
            &[GpuNode::zeroed()]
        } else {
            &gpu_nodes
        };
        let nodes = gpu.create_buffer_init("KdTree Nodes", contents, wgpu::BufferUsages::STORAGE);

        Ok(Self {
            nodes,
            node_to_index,
            frame,
        })
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.node_to_index.is_empty()
    }
}

/// Indices of `points` sorted along a Morton (Z-order) curve, so points that
/// are close in the list are close in space. GPU threads that run side by side
/// then search the same part of the tree, which keeps them in step and their
/// memory reads cached.
pub(crate) fn morton_order(points: &[Point3f]) -> Vec<u32> {
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for p in points
        .iter()
        .filter(|p| p.coords.iter().all(|c| c.is_finite()))
    {
        for axis in 0..3 {
            lo[axis] = lo[axis].min(p[axis]);
            hi[axis] = hi[axis].max(p[axis]);
        }
    }

    // 10 bits per axis is plenty to group nearby points.
    let cell = |axis: usize, value: f32| -> u32 {
        let extent = hi[axis] - lo[axis];
        if !(extent > 0.0) || !value.is_finite() {
            return 0;
        }
        (((value - lo[axis]) / extent) * 1023.0).clamp(0.0, 1023.0) as u32
    };
    // Spread the low 10 bits of `x` so there are two zero bits between each.
    let spread = |mut x: u32| -> u32 {
        x &= 0x3ff;
        x = (x | (x << 16)) & 0x030000ff;
        x = (x | (x << 8)) & 0x0300f00f;
        x = (x | (x << 4)) & 0x030c30c3;
        (x | (x << 2)) & 0x09249249
    };

    let mut keyed: Vec<(u32, u32)> = points
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let code =
                spread(cell(0, p[0])) | (spread(cell(1, p[1])) << 1) | (spread(cell(2, p[2])) << 2);
            (code, i as u32)
        })
        .collect();
    keyed.par_sort_unstable();
    keyed.into_iter().map(|(_, i)| i).collect()
}
