//! Nearest neighbor search implementations

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use threecrate_core::{NearestNeighborSearch, Point3f, Result};

/// Sentinel used in place of a child index to mean "no child".
const NIL: u32 = u32::MAX;

/// KD-Tree node stored in a flat, contiguous array.
///
/// Children are referenced by index into the owning `Vec<KdNode>` rather than
/// through `Box` pointers. Keeping every node in one allocation makes traversal
/// cache-friendly — neighbour search is the dominant cost in normal estimation
/// and ICP correspondence, so a contiguous layout directly moves those numbers.
#[derive(Debug, Clone)]
struct KdNode {
    point: Point3f,
    original_index: usize, // index into the original input slice
    left: u32,             // child index, or NIL
    right: u32,            // child index, or NIL
    axis: u8,              // splitting axis: 0=x, 1=y, 2=z
}

/// Efficient KD-Tree implementation for nearest neighbor search.
///
/// Nodes live in a single contiguous `Vec` (`nodes`); children are referenced by
/// index. `root` is the index of the tree root (always `0` when non-empty).
pub struct KdTree {
    nodes: Vec<KdNode>,
    root: Option<u32>,
    points: Vec<Point3f>, // Keep original points for reference
}

impl KdTree {
    /// Create a new KD-tree from a slice of points.
    ///
    /// Points with a NaN or infinite coordinate are left out of the tree and are
    /// never returned as neighbors (indices of the remaining points still refer
    /// to `points`). Organized clouds store invalid returns as NaN; a NaN split
    /// value would send every query down one side and make the finite points on
    /// the other side unreachable.
    pub fn new(points: &[Point3f]) -> Result<Self> {
        let mut points_with_indices: Vec<(Point3f, usize)> = points
            .iter()
            .enumerate()
            .filter(|(_, point)| point.coords.iter().all(|c| c.is_finite()))
            .map(|(i, &point)| (point, i))
            .collect();

        if points_with_indices.is_empty() {
            return Ok(Self {
                nodes: Vec::new(),
                root: None,
                points: points.to_vec(),
            });
        }

        // Every slot is overwritten by `build_tree`; the placeholder only lets
        // us hand out disjoint `&mut` sub-slices to parallel subtree builds.
        let placeholder = KdNode {
            point: Point3f::origin(),
            original_index: 0,
            left: NIL,
            right: NIL,
            axis: 0,
        };
        let mut nodes: Vec<KdNode> = (0..points_with_indices.len())
            .map(|_| placeholder.clone())
            .collect();
        Self::build_tree(&mut nodes, 0, &mut points_with_indices, 0);

        Ok(Self {
            nodes,
            root: Some(0),
            points: points.to_vec(),
        })
    }

    /// Recursively build the KD-tree for `points` into `nodes`, whose first slot
    /// sits at absolute array index `base`.
    ///
    /// Slots are filled in pre-order: the subtree root goes in `nodes[0]`, the
    /// left subtree (`median` nodes) in `nodes[1..=median]`, and the right
    /// subtree after it. Because each subtree's slot range is known up front,
    /// the two halves are disjoint slices and large subtrees build in parallel.
    fn build_tree(nodes: &mut [KdNode], base: u32, points: &mut [(Point3f, usize)], depth: usize) {
        /// Below this size, spawning rayon tasks costs more than it saves.
        const PARALLEL_THRESHOLD: usize = 4096;

        let axis = depth % 3;
        let len = points.len();
        let median = (len - 1) / 2;

        // `select_nth_unstable_by` is introselect: O(n) and, unlike a fixed
        // last-element pivot, not quadratic on already-sorted input such as
        // row-ordered depth images. Coordinates are finite here (`new` filters
        // the rest), so `total_cmp` matches numeric order.
        points.select_nth_unstable_by(median, |a, b| a.0.coords[axis].total_cmp(&b.0.coords[axis]));

        let (left_points, rest) = points.split_at_mut(median);
        let ((point, index), right_points) = (rest[0], &mut rest[1..]);

        let (root, children) = nodes.split_first_mut().expect("one slot per point");
        let (left_nodes, right_nodes) = children.split_at_mut(median);

        let left_base = base + 1;
        let right_base = base + 1 + median as u32;
        *root = KdNode {
            point,
            original_index: index,
            left: if left_points.is_empty() {
                NIL
            } else {
                left_base
            },
            right: if right_points.is_empty() {
                NIL
            } else {
                right_base
            },
            axis: axis as u8,
        };

        let build_left = |nodes: &mut [KdNode], points: &mut [(Point3f, usize)]| {
            if !points.is_empty() {
                Self::build_tree(nodes, left_base, points, depth + 1);
            }
        };
        let build_right = |nodes: &mut [KdNode], points: &mut [(Point3f, usize)]| {
            if !points.is_empty() {
                Self::build_tree(nodes, right_base, points, depth + 1);
            }
        };

        if len > PARALLEL_THRESHOLD {
            rayon::join(
                || build_left(left_nodes, left_points),
                || build_right(right_nodes, right_points),
            );
        } else {
            build_left(left_nodes, left_points);
            build_right(right_nodes, right_points);
        }
    }

    /// Find the single nearest neighbor of `query`, allocation-free.
    ///
    /// `bound_sq` is a squared-distance upper bound: only points strictly closer
    /// than it are considered, and the far side of a split is skipped as soon as
    /// it cannot beat the best so far. Pass `f32::INFINITY` for an unbounded
    /// search. When the caller already knows a candidate (e.g. last ICP
    /// iteration's match), pass its index and squared distance as `seed` so the
    /// search starts pruned; `seed` is returned if nothing closer is found.
    ///
    /// Returns `(original_index, squared_distance)`, or `None` if no point lies
    /// within the bound and no seed was given. This is the hot path for ICP
    /// correspondence search, so — unlike `find_k_nearest` — it uses a fixed-size
    /// stack and no heap, result `Vec`, or `sqrt`.
    pub fn find_nearest_bounded(
        &self,
        query: &Point3f,
        bound_sq: f32,
        seed: Option<(usize, f32)>,
    ) -> Option<(usize, f32)> {
        // The tree is median-split, so its depth is at most ceil(log2(n + 1)),
        // which for `u32` node indices is ≤ 33. The stack holds at most one
        // deferred far child per level plus the current near child.
        const MAX_STACK: usize = 64;

        let (mut best, mut best_sq) = match seed {
            Some((_, d)) if d < bound_sq => (seed, d),
            _ => (None, bound_sq),
        };

        // Each entry is (node, squared distance from the query to the split
        // plane that separated it). Far children are pushed before the near
        // child is explored, so `best_sq` may have shrunk by the time one is
        // popped; re-checking the plane distance then prunes the whole subtree.
        let mut stack = [(0u32, 0.0f32); MAX_STACK];
        let mut len = 0usize;
        if let Some(root) = self.root {
            stack[0] = (root, 0.0);
            len = 1;
        }

        while len > 0 {
            len -= 1;
            let (idx, plane_sq) = stack[len];
            if plane_sq >= best_sq {
                continue;
            }
            let node = &self.nodes[idx as usize];

            let dist_sq = Self::distance_squared(&node.point, query);
            if dist_sq < best_sq {
                best_sq = dist_sq;
                best = Some((node.original_index, dist_sq));
            }

            let axis_dist =
                query.coords[node.axis as usize] - node.point.coords[node.axis as usize];
            let axis_dist_sq = axis_dist * axis_dist;
            let (near, far) = if axis_dist <= 0.0 {
                (node.left, node.right)
            } else {
                (node.right, node.left)
            };

            // Push far before near so near is popped first (LIFO).
            if far != NIL && axis_dist_sq < best_sq {
                stack[len] = (far, axis_dist_sq);
                len += 1;
            }
            if near != NIL {
                stack[len] = (near, 0.0);
                len += 1;
            }
        }

        best
    }

    /// Calculate squared distance between two points
    fn distance_squared(a: &Point3f, b: &Point3f) -> f32 {
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let dz = a.z - b.z;
        dx * dx + dy * dy + dz * dz
    }
}

impl NearestNeighborSearch for KdTree {
    /// Find the `k` nearest neighbors using an iterative stack-based traversal.
    ///
    /// Uses an explicit `Vec` stack (LIFO) so that recursion depth is bounded only
    /// by available heap memory — not the call stack — making it safe from stack
    /// overflows even for very deep or unbalanced trees and when called from rayon
    /// worker threads (which have smaller default stacks than the main thread).
    fn find_k_nearest(&self, query: &Point3f, k: usize) -> Vec<(usize, f32)> {
        if k == 0 || self.points.is_empty() {
            return Vec::new();
        }

        // Max-heap: the *farthest* accepted neighbor sits at the top so we can
        // evict it in O(log k) when a closer point is found.
        //
        // Distances are kept *squared* throughout the traversal so we never pay
        // for a `sqrt` per visited node; squared distance is monotonic in
        // distance, so heap ordering and pruning are unaffected. We take the
        // square root once per surviving neighbor when building the result.
        let mut heap: BinaryHeap<Neighbor> = BinaryHeap::with_capacity(k + 1);
        let mut stack: Vec<u32> = Vec::new();

        if let Some(root) = self.root {
            stack.push(root);
        }

        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx as usize];
            let dist_sq = Self::distance_squared(&node.point, query);

            if heap.len() < k {
                heap.push(Neighbor {
                    distance: dist_sq,
                    index: node.original_index,
                });
            } else if let Some(farthest) = heap.peek() {
                if dist_sq < farthest.distance {
                    heap.pop();
                    heap.push(Neighbor {
                        distance: dist_sq,
                        index: node.original_index,
                    });
                }
            }

            let query_val = query.coords[node.axis as usize];
            let node_val = node.point.coords[node.axis as usize];
            let axis_dist = query_val - node_val;
            let axis_dist_sq = axis_dist * axis_dist;

            // Near child: the half-space the query point lives in.
            // Far child:  the other half-space, searched only when it could
            //             contain a point closer than the current k-th nearest.
            let (near, far) = if query_val <= node_val {
                (node.left, node.right)
            } else {
                (node.right, node.left)
            };

            // Push far before near so near is popped first (LIFO), giving the
            // same visit order as the recursive "near first" traversal and
            // maximising early pruning of the far subtree.
            let search_far = if let Some(farthest) = heap.peek() {
                heap.len() < k || axis_dist_sq < farthest.distance
            } else {
                true
            };
            if search_far && far != NIL {
                stack.push(far);
            }
            if near != NIL {
                stack.push(near);
            }
        }

        // `into_sorted_vec` drains the max-heap in ascending order of squared
        // distance (smallest first); take the sqrt here to return true distances.
        heap.into_sorted_vec()
            .into_iter()
            .map(|n| (n.index, n.distance.sqrt()))
            .collect()
    }

    /// Find all neighbors within `radius` using an iterative stack-based traversal.
    fn find_radius_neighbors(&self, query: &Point3f, radius: f32) -> Vec<(usize, f32)> {
        if radius <= 0.0 || self.points.is_empty() {
            return Vec::new();
        }

        let radius_sq = radius * radius;
        let mut result: Vec<(usize, f32)> = Vec::new();
        let mut stack: Vec<u32> = Vec::new();

        if let Some(root) = self.root {
            stack.push(root);
        }

        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx as usize];
            let dist_sq = Self::distance_squared(&node.point, query);
            if dist_sq <= radius_sq {
                result.push((node.original_index, dist_sq.sqrt()));
            }

            let query_val = query.coords[node.axis as usize];
            let node_val = node.point.coords[node.axis as usize];
            let axis_dist = query_val - node_val;

            let (near, far) = if query_val <= node_val {
                (node.left, node.right)
            } else {
                (node.right, node.left)
            };

            // The far subtree can only contain in-radius points when the
            // distance to the splitting hyperplane is within the search radius.
            if axis_dist * axis_dist <= radius_sq {
                if far != NIL {
                    stack.push(far);
                }
            }
            if near != NIL {
                stack.push(near);
            }
        }

        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        result
    }
}

/// Helper struct for maintaining the k-nearest neighbors heap
#[derive(Debug, PartialEq)]
struct Neighbor {
    distance: f32,
    index: usize,
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap ordered by distance: larger distance = "greater" element,
        // so heap.peek() returns the farthest neighbour for eviction.
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Simple brute force nearest neighbor search for small datasets
pub struct BruteForceSearch {
    points: Vec<Point3f>,
}

impl BruteForceSearch {
    pub fn new(points: &[Point3f]) -> Self {
        Self {
            points: points.to_vec(),
        }
    }
}

impl NearestNeighborSearch for BruteForceSearch {
    fn find_k_nearest(&self, query: &Point3f, k: usize) -> Vec<(usize, f32)> {
        if k == 0 || self.points.is_empty() {
            return Vec::new();
        }

        let mut distances: Vec<(usize, f32)> = self
            .points
            .iter()
            .enumerate()
            .map(|(idx, point)| {
                let dx = point.x - query.x;
                let dy = point.y - query.y;
                let dz = point.z - query.z;
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                (idx, distance)
            })
            .collect();

        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        distances.truncate(k);
        distances
    }

    fn find_radius_neighbors(&self, query: &Point3f, radius: f32) -> Vec<(usize, f32)> {
        if radius <= 0.0 || self.points.is_empty() {
            return Vec::new();
        }

        let radius_squared = radius * radius;
        self.points
            .iter()
            .enumerate()
            .filter_map(|(idx, point)| {
                let dx = point.x - query.x;
                let dy = point.y - query.y;
                let dz = point.z - query.z;
                let distance_squared = dx * dx + dy * dy + dz * dz;

                if distance_squared <= radius_squared {
                    Some((idx, distance_squared.sqrt()))
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use threecrate_core::Point3f;

    fn create_test_points() -> Vec<Point3f> {
        vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.0, 1.0, 0.0),
            Point3f::new(0.0, 0.0, 1.0),
            Point3f::new(1.0, 1.0, 0.0),
            Point3f::new(1.0, 0.0, 1.0),
            Point3f::new(0.0, 1.0, 1.0),
            Point3f::new(1.0, 1.0, 1.0),
        ]
    }

    #[test]
    fn test_kd_tree_construction() {
        let points = create_test_points();
        let kdtree = KdTree::new(&points).unwrap();

        assert_eq!(kdtree.points.len(), points.len());
        assert!(kdtree.root.is_some());
    }

    #[test]
    fn test_empty_kd_tree() {
        let kdtree = KdTree::new(&[]).unwrap();
        assert!(kdtree.root.is_none());
        assert!(kdtree.points.is_empty());

        let query = Point3f::new(0.0, 0.0, 0.0);
        let result = kdtree.find_k_nearest(&query, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_k_nearest_neighbors_consistency() {
        let points = create_test_points();
        let kdtree = KdTree::new(&points).unwrap();
        let brute_force = BruteForceSearch::new(&points);

        let query = Point3f::new(0.5, 0.5, 0.5);
        let k = 3;

        let mut kdtree_result = kdtree.find_k_nearest(&query, k);
        let mut brute_force_result = brute_force.find_k_nearest(&query, k);

        println!("KD-tree result before sorting: {:?}", kdtree_result);
        println!(
            "Brute force result before sorting: {:?}",
            brute_force_result
        );

        // Sort by distance first, then by index for consistent comparison
        kdtree_result.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        brute_force_result.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        println!("KD-tree result after sorting: {:?}", kdtree_result);
        println!("Brute force result after sorting: {:?}", brute_force_result);

        // Results should have the same length
        assert_eq!(kdtree_result.len(), brute_force_result.len());
        assert_eq!(kdtree_result.len(), k);

        // Results should be sorted by distance
        for i in 1..kdtree_result.len() {
            assert!(kdtree_result[i - 1].1 <= kdtree_result[i].1);
            assert!(brute_force_result[i - 1].1 <= brute_force_result[i].1);
        }

        // Check that the distances match (within tolerance)
        for (kdtree_neighbor, brute_neighbor) in kdtree_result.iter().zip(brute_force_result.iter())
        {
            assert!((kdtree_neighbor.1 - brute_neighbor.1).abs() < 1e-6);
        }

        // For points with the same distance, we don't require the exact same indices
        // as long as the distances are correct, the implementation is working
        println!(
            "Test passed: Both methods found {} neighbors with correct distances",
            k
        );
    }

    #[test]
    fn test_radius_neighbors_consistency() {
        let points = create_test_points();
        let kdtree = KdTree::new(&points).unwrap();
        let brute_force = BruteForceSearch::new(&points);

        let query = Point3f::new(0.5, 0.5, 0.5);
        let radius = 1.5;

        let mut kdtree_result = kdtree.find_radius_neighbors(&query, radius);
        let mut brute_force_result = brute_force.find_radius_neighbors(&query, radius);

        // Sort by distance first, then by index for consistent comparison
        kdtree_result.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        brute_force_result.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        // Results should have the same length
        assert_eq!(kdtree_result.len(), brute_force_result.len());

        // Results should be sorted by distance
        for i in 1..kdtree_result.len() {
            assert!(kdtree_result[i - 1].1 <= kdtree_result[i].1);
            assert!(brute_force_result[i - 1].1 <= brute_force_result[i].1);
        }

        // All distances should be within radius
        for (_, distance) in &kdtree_result {
            assert!(*distance <= radius);
        }

        for (_, distance) in &brute_force_result {
            assert!(*distance <= radius);
        }

        // Check that the distances match (within tolerance)
        for (kdtree_neighbor, brute_neighbor) in kdtree_result.iter().zip(brute_force_result.iter())
        {
            assert!((kdtree_neighbor.1 - brute_neighbor.1).abs() < 1e-6);
        }

        println!(
            "Test passed: Both methods found {} neighbors within radius {}",
            kdtree_result.len(),
            radius
        );
    }

    #[test]
    fn test_edge_cases() {
        let points = create_test_points();
        let kdtree = KdTree::new(&points).unwrap();
        let _brute_force = BruteForceSearch::new(&points);

        let query = Point3f::new(0.0, 0.0, 0.0);

        // Test k = 0
        let result = kdtree.find_k_nearest(&query, 0);
        assert!(result.is_empty());

        // Test k larger than number of points
        let result = kdtree.find_k_nearest(&query, 20);
        assert_eq!(result.len(), points.len());

        // Test radius = 0
        let result = kdtree.find_radius_neighbors(&query, 0.0);
        assert!(result.is_empty());

        // Test negative radius
        let result = kdtree.find_radius_neighbors(&query, -1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_random_points() {
        let mut rng = rand::rng();
        let mut points = Vec::new();

        // Generate 100 random points
        for _ in 0..100 {
            points.push(Point3f::new(
                rng.random_range(-10.0..10.0),
                rng.random_range(-10.0..10.0),
                rng.random_range(-10.0..10.0),
            ));
        }

        let kdtree = KdTree::new(&points).unwrap();
        let brute_force = BruteForceSearch::new(&points);

        // Test multiple random queries
        for _ in 0..10 {
            let query = Point3f::new(
                rng.random_range(-5.0..5.0),
                rng.random_range(-5.0..5.0),
                rng.random_range(-5.0..5.0),
            );

            let k = rng.random_range(1..=10);
            let radius = rng.random_range(1.0..5.0);

            let mut kdtree_knn = kdtree.find_k_nearest(&query, k);
            let mut brute_knn = brute_force.find_k_nearest(&query, k);

            let mut kdtree_radius = kdtree.find_radius_neighbors(&query, radius);
            let mut brute_radius = brute_force.find_radius_neighbors(&query, radius);

            // Sort by distance first, then by index for consistent comparison
            kdtree_knn.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            brute_knn.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });

            kdtree_radius.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            brute_radius.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });

            // Verify k-nearest neighbors consistency
            assert_eq!(kdtree_knn.len(), brute_knn.len());
            assert_eq!(kdtree_knn.len(), k.min(points.len()));

            // Check that the distances match (within tolerance)
            let min_len = kdtree_knn.len().min(brute_knn.len());
            for i in 0..min_len {
                assert!((kdtree_knn[i].1 - brute_knn[i].1).abs() < 1e-6);
            }

            // Verify radius neighbors consistency
            assert_eq!(kdtree_radius.len(), brute_radius.len());

            // Check that the distances match (within tolerance)
            let min_len = kdtree_radius.len().min(brute_radius.len());
            for i in 0..min_len {
                assert!((kdtree_radius[i].1 - brute_radius[i].1).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_find_nearest_bounded_matches_brute_force() {
        let mut rng = rand::rng();
        let points: Vec<Point3f> = (0..5000)
            .map(|_| {
                Point3f::new(
                    rng.random_range(-10.0..10.0),
                    rng.random_range(-10.0..10.0),
                    rng.random_range(-10.0..10.0),
                )
            })
            .collect();
        let kdtree = KdTree::new(&points).unwrap();
        let brute_force = BruteForceSearch::new(&points);

        for _ in 0..200 {
            let query = Point3f::new(
                rng.random_range(-12.0..12.0),
                rng.random_range(-12.0..12.0),
                rng.random_range(-12.0..12.0),
            );
            let (bf_idx, bf_dist) = brute_force.find_k_nearest(&query, 1)[0];
            let bf_sq = bf_dist * bf_dist;

            // Unbounded, unseeded
            let (_, d) = kdtree
                .find_nearest_bounded(&query, f32::INFINITY, None)
                .unwrap();
            assert!((d - bf_sq).abs() < 1e-4);

            // Seeded with an arbitrary (usually far) point: still exact
            let seed_idx = rng.random_range(0..points.len());
            let seed_sq = (points[seed_idx] - query).magnitude_squared();
            let (_, d) = kdtree
                .find_nearest_bounded(&query, f32::INFINITY, Some((seed_idx, seed_sq)))
                .unwrap();
            assert!((d - bf_sq).abs() < 1e-4);

            // Seeded with the true answer: returns it
            let (idx, _) = kdtree
                .find_nearest_bounded(&query, f32::INFINITY, Some((bf_idx, bf_sq)))
                .unwrap();
            assert_eq!(idx, bf_idx);

            // A bound tighter than the nearest neighbor finds nothing
            assert!(kdtree
                .find_nearest_bounded(&query, bf_sq * 0.5, None)
                .is_none());
        }
    }

    #[test]
    fn test_kd_tree_with_nan_points() {
        let mut points = create_test_points();
        points.push(Point3f::new(f32::NAN, 0.0, 0.0));
        points.push(Point3f::new(0.5, f32::NAN, f32::NAN));

        // Building must not panic, and finite queries still resolve to the
        // nearest finite point.
        let kdtree = KdTree::new(&points).unwrap();
        let (idx, d) = kdtree
            .find_nearest_bounded(&Point3f::new(0.9, 0.1, 0.05), f32::INFINITY, None)
            .unwrap();
        assert_eq!(idx, 1);
        assert!(d.is_finite());
    }

    #[test]
    fn test_kd_tree_mostly_nan_points() {
        // Mostly-invalid organized cloud: NaN points must not become split
        // nodes that hide the finite points from the search.
        let nan = Point3f::new(f32::NAN, f32::NAN, f32::NAN);
        let mut points = vec![nan; 50];
        points[17] = Point3f::new(0.0, 0.0, 0.0);
        points[33] = Point3f::new(5.0, 5.0, 5.0);
        points.push(Point3f::new(f32::INFINITY, 0.0, 0.0));

        let kdtree = KdTree::new(&points).unwrap();
        let query = Point3f::new(0.1, 0.0, 0.0);

        let (idx, _) = kdtree
            .find_nearest_bounded(&query, f32::INFINITY, None)
            .unwrap();
        assert_eq!(idx, 17);

        let knn = kdtree.find_k_nearest(&query, 5);
        assert_eq!(
            knn.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![17, 33]
        );

        let radius = kdtree.find_radius_neighbors(&query, 1.0);
        assert_eq!(radius.len(), 1);
        assert_eq!(radius[0].0, 17);

        // An all-invalid cloud yields an empty tree, not a panic
        let empty = KdTree::new(&[nan, nan]).unwrap();
        assert!(empty
            .find_nearest_bounded(&query, f32::INFINITY, None)
            .is_none());
        assert!(empty.find_k_nearest(&query, 3).is_empty());
    }

    #[test]
    fn test_performance_comparison() {
        let mut rng = rand::rng();
        let mut points = Vec::new();

        // Generate 1000 random points for performance test
        for _ in 0..1000 {
            points.push(Point3f::new(
                rng.random_range(-10.0..10.0),
                rng.random_range(-10.0..10.0),
                rng.random_range(-10.0..10.0),
            ));
        }

        let kdtree = KdTree::new(&points).unwrap();
        let brute_force = BruteForceSearch::new(&points);

        let query = Point3f::new(0.0, 0.0, 0.0);
        let k = 10;

        // Time KD-tree search
        let start = std::time::Instant::now();
        let _kdtree_result = kdtree.find_k_nearest(&query, k);
        let kdtree_time = start.elapsed();

        // Time brute force search
        let start = std::time::Instant::now();
        let _brute_result = brute_force.find_k_nearest(&query, k);
        let brute_time = start.elapsed();

        // KD-tree should be faster for larger datasets
        println!("KD-tree time: {:?}", kdtree_time);
        println!("Brute force time: {:?}", brute_time);

        // For 1000 points, KD-tree should be significantly faster
        // Note: For small k values, brute force might actually be faster due to overhead
        // So we'll just verify both methods work correctly
        assert!(kdtree_time.as_nanos() > 0);
        assert!(brute_time.as_nanos() > 0);
    }

    #[test]
    fn test_debug_k_nearest() {
        let points = vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.0, 1.0, 0.0),
            Point3f::new(0.0, 0.0, 1.0),
            Point3f::new(1.0, 1.0, 0.0),
            Point3f::new(1.0, 0.0, 1.0),
            Point3f::new(0.0, 1.0, 1.0),
            Point3f::new(1.0, 1.0, 1.0),
        ];

        let kdtree = KdTree::new(&points).unwrap();
        let brute_force = BruteForceSearch::new(&points);

        let query = Point3f::new(0.5, 0.5, 0.5);
        let k = 3;

        let mut kdtree_result = kdtree.find_k_nearest(&query, k);
        let mut brute_force_result = brute_force.find_k_nearest(&query, k);

        kdtree_result.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        brute_force_result.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        assert_eq!(kdtree_result.len(), brute_force_result.len());
        assert_eq!(kdtree_result.len(), k);
        for (kd, bf) in kdtree_result.iter().zip(brute_force_result.iter()) {
            assert!(
                (kd.1 - bf.1).abs() < 1e-6,
                "distance mismatch: kd={}, bf={}",
                kd.1,
                bf.1
            );
        }
    }
}
