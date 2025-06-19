//! Core traits for 3DCrate

use crate::{point::*, point_cloud::*, mesh::*, transform::Transform3D};

/// Trait for nearest neighbor search functionality
pub trait NearestNeighborSearch {
    /// Find the k nearest neighbors to a query point
    fn find_k_nearest(&self, query: &Point3f, k: usize) -> Vec<(usize, f32)>;
    
    /// Find all neighbors within a given radius
    fn find_radius_neighbors(&self, query: &Point3f, radius: f32) -> Vec<(usize, f32)>;
}

/// Trait for drawable/renderable objects
pub trait Drawable {
    /// Get the bounding box of the object
    fn bounding_box(&self) -> (Point3f, Point3f);
    
    /// Get the center point of the object
    fn center(&self) -> Point3f;
}

/// Trait for objects that can be transformed
pub trait Transformable {
    /// Apply a transformation to the object
    fn transform(&mut self, transform: &Transform3D);
}

impl<T> Drawable for PointCloud<T> 
where 
    T: Clone + Copy,
    Point3f: From<T>,
{
    fn bounding_box(&self) -> (Point3f, Point3f) {
        if self.is_empty() {
            return (Point3f::origin(), Point3f::origin());
        }
        
        let first_point = Point3f::from(self.points[0]);
        let mut min = first_point;
        let mut max = first_point;
        
        for point in &self.points {
            let p = Point3f::from(*point);
            min.x = min.x.min(p.x);
            min.y = min.y.min(p.y);
            min.z = min.z.min(p.z);
            
            max.x = max.x.max(p.x);
            max.y = max.y.max(p.y);
            max.z = max.z.max(p.z);
        }
        
        (min, max)
    }
    
    fn center(&self) -> Point3f {
        let (min, max) = self.bounding_box();
        Point3f::new(
            (min.x + max.x) / 2.0,
            (min.y + max.y) / 2.0,
            (min.z + max.z) / 2.0,
        )
    }
}

impl Drawable for TriangleMesh {
    fn bounding_box(&self) -> (Point3f, Point3f) {
        if self.vertices.is_empty() {
            return (Point3f::origin(), Point3f::origin());
        }
        
        let mut min = self.vertices[0];
        let mut max = self.vertices[0];
        
        for vertex in &self.vertices {
            min.x = min.x.min(vertex.x);
            min.y = min.y.min(vertex.y);
            min.z = min.z.min(vertex.z);
            
            max.x = max.x.max(vertex.x);
            max.y = max.y.max(vertex.y);
            max.z = max.z.max(vertex.z);
        }
        
        (min, max)
    }
    
    fn center(&self) -> Point3f {
        let (min, max) = self.bounding_box();
        Point3f::new(
            (min.x + max.x) / 2.0,
            (min.y + max.y) / 2.0,
            (min.z + max.z) / 2.0,
        )
    }
} 