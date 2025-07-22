//! Nearest neighbor search implementations

use threecrate_core::{Point3f, Result, NearestNeighborSearch};

/// KD-Tree implementation for nearest neighbor search
pub struct KdTree {
    points: Vec<Point3f>,
}

impl KdTree {
    pub fn new(points: &[Point3f]) -> Result<Self> {
        Ok(Self {
            points: points.to_vec(),
        })
    }
}

impl NearestNeighborSearch for KdTree {
    fn find_k_nearest(&self, query: &Point3f, k: usize) -> Vec<(usize, f32)> {
        let mut distances: Vec<(usize, f32)> = self.points
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
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);
        distances
    }
    
    fn find_radius_neighbors(&self, query: &Point3f, radius: f32) -> Vec<(usize, f32)> {
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
        let mut distances: Vec<(usize, f32)> = self.points
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
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);
        distances
    }
    
    fn find_radius_neighbors(&self, query: &Point3f, radius: f32) -> Vec<(usize, f32)> {
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