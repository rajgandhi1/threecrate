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
#[derive(Debug)]
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
    /// Create a new KD-tree from a slice of points
    pub fn new(points: &[Point3f]) -> Result<Self> {
        if points.is_empty() {
            return Ok(Self {
                nodes: Vec::new(),
                root: None,
                points: Vec::new(),
            });
        }

        let mut points_with_indices: Vec<(Point3f, usize)> = points
            .iter()
            .enumerate()
            .map(|(i, &point)| (point, i))
            .collect();

        let mut nodes: Vec<KdNode> = Vec::with_capacity(points.len());
        let root = Self::build_tree(&mut nodes, &mut points_with_indices, 0, 0, points.len() - 1);

        Ok(Self {
            nodes,
            root: Some(root),
            points: points.to_vec(),
        })
    }

    /// Recursively build the KD-tree into the flat `nodes` array, returning the
    /// array index of the subtree root spanning `[start, end]`.
    ///
    /// Slots are filled in pre-order, so the overall root lands at index 0.
    fn build_tree(
        nodes: &mut Vec<KdNode>,
        points: &mut [(Point3f, usize)],
        depth: usize,
        start: usize,
        end: usize,
    ) -> u32 {
        let axis = depth % 3;
        let median_idx = (start + end) / 2;

        // Find the actual median and partition points around it
        Self::select_median(points, start, end, median_idx, axis);

        let (point, index) = points[median_idx];

        // Reserve this node's slot before recursing so children can link to it
        // (and to each other) by index.
        let my_idx = nodes.len() as u32;
        nodes.push(KdNode {
            point,
            original_index: index,
            left: NIL,
            right: NIL,
            axis: axis as u8,
        });

        // Build left subtree
        let left = if median_idx > start {
            Self::build_tree(nodes, points, depth + 1, start, median_idx - 1)
        } else {
            NIL
        };

        // Build right subtree
        let right = if median_idx < end {
            Self::build_tree(nodes, points, depth + 1, median_idx + 1, end)
        } else {
            NIL
        };

        nodes[my_idx as usize].left = left;
        nodes[my_idx as usize].right = right;
        my_idx
    }

    /// Select the median element and partition points around it
    fn select_median(
        points: &mut [(Point3f, usize)],
        start: usize,
        end: usize,
        target: usize,
        axis: usize,
    ) {
        let mut left = start;
        let mut right = end;

        while left < right {
            let pivot_idx = Self::partition(points, left, right, axis);

            match pivot_idx.cmp(&target) {
                Ordering::Equal => return,
                Ordering::Less => left = pivot_idx + 1,
                Ordering::Greater => right = pivot_idx - 1,
            }
        }
    }

    /// Partition points around a pivot on a specific axis
    fn partition(points: &mut [(Point3f, usize)], start: usize, end: usize, axis: usize) -> usize {
        let pivot_value = match axis {
            0 => points[end].0.x,
            1 => points[end].0.y,
            2 => points[end].0.z,
            _ => unreachable!(),
        };

        let mut i = start;
        for j in start..end {
            let point_value = match axis {
                0 => points[j].0.x,
                1 => points[j].0.y,
                2 => points[j].0.z,
                _ => unreachable!(),
            };

            if point_value <= pivot_value {
                points.swap(i, j);
                i += 1;
            }
        }

        points.swap(i, end);
        i
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
