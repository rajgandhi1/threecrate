//! Headless preview images of point clouds and meshes, drawn on the CPU.
//!
//! Made for checking work, including by AI agents that can read images: the
//! object is framed automatically and drawn from a few fixed angles. Mesh
//! faces seen from the front are blue-grey and faces seen from behind are
//! orange, so holes (you see the inside) and flipped faces stand out. No GPU
//! or window is needed.

use std::io::Write;
use std::path::Path;
use threecrate_core::{Error, Point3f, Result, TriangleMesh, Vector3f};

/// Something to draw.
#[derive(Clone, Copy)]
pub enum Drawable<'a> {
    Points(&'a [Point3f]),
    Mesh(&'a TriangleMesh),
}

/// Direction the camera looks from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    Front,
    Back,
    Left,
    Right,
    Top,
    Bottom,
    /// From the front-right, above: shows three sides at once.
    Angled,
}

impl View {
    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name.to_ascii_lowercase().as_str() {
            "front" => View::Front,
            "back" => View::Back,
            "left" => View::Left,
            "right" | "side" => View::Right,
            "top" => View::Top,
            "bottom" => View::Bottom,
            "angled" | "iso" => View::Angled,
            _ => {
                return Err(Error::InvalidData(format!(
                    "unknown view '{name}'; use front, back, left, right, top, bottom or angled"
                )))
            }
        })
    }

    /// (direction from the object to the camera, up direction), with z up.
    fn camera(self) -> (Vector3f, Vector3f) {
        let z = Vector3f::z();
        match self {
            View::Front => (-Vector3f::y(), z),
            View::Back => (Vector3f::y(), z),
            View::Left => (-Vector3f::x(), z),
            View::Right => (Vector3f::x(), z),
            View::Top => (z, Vector3f::y()),
            View::Bottom => (-z, -Vector3f::y()),
            View::Angled => (Vector3f::new(1.0, -1.0, 0.8).normalize(), z),
        }
    }
}

/// Which axis points up in the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpAxis {
    Z,
    Y,
}

/// Options for [`render_grid`].
#[derive(Debug, Clone)]
pub struct RenderOptions {
    /// Views drawn left to right in each row.
    pub views: Vec<View>,
    /// Size of each tile in pixels.
    pub tile_size: u32,
    pub up: UpAxis,
    /// Side of the square drawn for each point, in pixels.
    pub point_size: u32,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            // Bottom is included because scans most often miss their base.
            views: vec![
                View::Angled,
                View::Front,
                View::Right,
                View::Top,
                View::Bottom,
            ],
            tile_size: 384,
            up: UpAxis::Z,
            point_size: 2,
        }
    }
}

/// An RGB image, row by row.
#[derive(Debug, Clone)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

const BACKGROUND: [u8; 3] = [255, 255, 255];
const FRONT_FACE: [f32; 3] = [110.0, 140.0, 190.0];
const BACK_FACE: [f32; 3] = [235.0, 140.0, 40.0];
const POINT: [f32; 3] = [40.0, 70.0, 130.0];
const DIVIDER: [u8; 3] = [200, 200, 200];

/// Draw each item as one row and each view as one column, all at the same
/// scale, so before/after pairs compare directly.
pub fn render_grid(items: &[Drawable], options: &RenderOptions) -> Result<Image> {
    if items.is_empty() || options.views.is_empty() {
        return Err(Error::InvalidData("nothing to render".to_string()));
    }
    let tile = options.tile_size.clamp(32, 4096);
    // Each view is framed on what it actually shows; every row of a column
    // shares that framing, so before/after stay at the same scale. Items in
    // different units (a scan in metres, its print in millimetres) would
    // shrink one to a dot, so then each row is framed on its own.
    let framing = |group: &[Drawable]| -> Result<(Vector3f, f32, Vec<f32>)> {
        let (center, radius) = frame(group, options.up)?;
        let reach = options
            .views
            .iter()
            .map(|v| view_reach(group, *v, center, options.up))
            .collect();
        Ok((center, radius, reach))
    };
    let radii = items
        .iter()
        .map(|item| frame(std::slice::from_ref(item), options.up).map(|(_, r)| r))
        .collect::<Result<Vec<f32>>>()?;
    let spread = radii.iter().cloned().fold(0.0, f32::max)
        / radii.iter().cloned().fold(f32::INFINITY, f32::min);
    let frames = if spread > 50.0 {
        items
            .iter()
            .map(|item| framing(std::slice::from_ref(item)))
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![framing(items)?; items.len()]
    };

    let (cols, rows) = (options.views.len() as u32, items.len() as u32);
    let mut grid = Image {
        width: cols * tile,
        height: rows * tile,
        pixels: vec![255; (cols * tile * rows * tile * 3) as usize],
    };
    for (r, item) in items.iter().enumerate() {
        let (center, radius, reach) = &frames[r];
        for (c, view) in options.views.iter().enumerate() {
            let image = render_tile(*item, *view, *center, *radius, reach[c], tile, options);
            for y in 0..tile {
                let from = (y * tile * 3) as usize;
                let to = (((r as u32 * tile + y) * grid.width + c as u32 * tile) * 3) as usize;
                grid.pixels[to..to + (tile * 3) as usize]
                    .copy_from_slice(&image.pixels[from..from + (tile * 3) as usize]);
            }
        }
    }
    // Thin dividers between tiles.
    for x in (1..cols).map(|c| c * tile) {
        for y in 0..grid.height {
            set(&mut grid, x, y, DIVIDER);
        }
    }
    for y in (1..rows).map(|r| r * tile) {
        for x in 0..grid.width {
            set(&mut grid, x, y, DIVIDER);
        }
    }
    Ok(grid)
}

/// Encode `image` as a PNG file.
pub fn write_png<P: AsRef<Path>>(image: &Image, path: P) -> Result<()> {
    let file = std::fs::File::create(path)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), image.width, image.height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|e| Error::Io(std::io::Error::other(e)))?;
    writer
        .write_image_data(&image.pixels)
        .map_err(|e| Error::Io(std::io::Error::other(e)))?;
    writer
        .finish()
        .map_err(|e| Error::Io(std::io::Error::other(e)))?;
    Ok(())
}

/// Write `image` as PNG bytes in memory.
pub fn png_bytes(image: &Image) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, image.width, image.height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|e| Error::Io(std::io::Error::other(e)))?;
        writer
            .write_image_data(&image.pixels)
            .map_err(|e| Error::Io(std::io::Error::other(e)))?;
    }
    out.flush()?;
    Ok(out)
}

/// The data's up axis mapped to z.
fn to_z_up(p: &Point3f, up: UpAxis) -> Vector3f {
    match up {
        UpAxis::Z => p.coords,
        UpAxis::Y => Vector3f::new(p.x, -p.z, p.y),
    }
}

/// Centre and radius of everything to draw (finite points only).
fn frame(items: &[Drawable], up: UpAxis) -> Result<(Vector3f, f32)> {
    let mut lo = Vector3f::repeat(f32::INFINITY);
    let mut hi = Vector3f::repeat(f32::NEG_INFINITY);
    let mut any = false;
    for item in items {
        let points: &[Point3f] = match item {
            Drawable::Points(p) => p,
            Drawable::Mesh(m) => &m.vertices,
        };
        for p in points
            .iter()
            .filter(|p| p.coords.iter().all(|c| c.is_finite()))
        {
            let v = to_z_up(p, up);
            lo = lo.inf(&v);
            hi = hi.sup(&v);
            any = true;
        }
    }
    if !any {
        return Err(Error::InvalidData(
            "nothing to render: no finite points".to_string(),
        ));
    }
    let center = (lo + hi) * 0.5;
    Ok((center, ((hi - lo).norm() * 0.5).max(1e-9)))
}

/// Camera axes (right, up, toward the camera) for `view`.
fn camera_axes(view: View) -> (Vector3f, Vector3f, Vector3f) {
    let (eye, up) = view.camera();
    let right = up.cross(&eye).normalize();
    (right, eye.cross(&right), eye)
}

/// Largest distance from `center` across or up the image in `view`, over
/// every finite point of every item.
fn view_reach(items: &[Drawable], view: View, center: Vector3f, up: UpAxis) -> f32 {
    let (right, up_axis, _) = camera_axes(view);
    let mut reach = 0.0f32;
    for item in items {
        let points: &[Point3f] = match item {
            Drawable::Points(p) => p,
            Drawable::Mesh(m) => &m.vertices,
        };
        for p in points
            .iter()
            .filter(|p| p.coords.iter().all(|c| c.is_finite()))
        {
            let v = to_z_up(p, up) - center;
            reach = reach.max(v.dot(&right).abs()).max(v.dot(&up_axis).abs());
        }
    }
    reach.max(1e-9)
}

fn set(image: &mut Image, x: u32, y: u32, color: [u8; 3]) {
    let i = ((y * image.width + x) * 3) as usize;
    image.pixels[i..i + 3].copy_from_slice(&color);
}

fn render_tile(
    item: Drawable,
    view: View,
    center: Vector3f,
    radius: f32,
    reach: f32,
    size: u32,
    options: &RenderOptions,
) -> Image {
    let (right, up, eye) = camera_axes(view);
    // Orthographic, filling the tile with a small margin.
    let scale = size as f32 * 0.46 / reach;
    let half = size as f32 * 0.5;
    let project = |p: &Point3f| {
        let v = to_z_up(p, options.up) - center;
        (
            half + v.dot(&right) * scale,
            half - v.dot(&up) * scale,
            v.dot(&eye),
        )
    };

    let mut image = Image {
        width: size,
        height: size,
        pixels: BACKGROUND.repeat((size * size) as usize),
    };
    // Larger depth = closer to the camera.
    let mut depth = vec![f32::NEG_INFINITY; (size * size) as usize];
    let mut plot = |x: i64, y: i64, z: f32, color: [f32; 3], image: &mut Image| {
        if x < 0 || y < 0 || x >= size as i64 || y >= size as i64 {
            return;
        }
        let i = (y as u32 * size + x as u32) as usize;
        if z > depth[i] {
            depth[i] = z;
            set(
                image,
                x as u32,
                y as u32,
                color.map(|c| c.clamp(0.0, 255.0) as u8),
            );
        }
    };

    match item {
        Drawable::Points(points) => {
            let s = options.point_size.max(1) as i64;
            for p in points
                .iter()
                .filter(|p| p.coords.iter().all(|c| c.is_finite()))
            {
                let (x, y, z) = project(p);
                // Closer points are darker, which reads as depth.
                let shade = 0.55 + 0.45 * (1.0 - (z / radius + 1.0) * 0.5);
                let color = POINT.map(|c| c + (255.0 - c) * (1.0 - shade));
                let (x0, y0) = (x as i64 - s / 2, y as i64 - s / 2);
                for dy in 0..s {
                    for dx in 0..s {
                        plot(x0 + dx, y0 + dy, z, color, &mut image);
                    }
                }
            }
        }
        Drawable::Mesh(mesh) => {
            let n = mesh.vertices.len();
            for face in mesh.faces.iter().filter(|f| f.iter().all(|&v| v < n)) {
                let corners = face.map(|v| mesh.vertices[v]);
                if corners
                    .iter()
                    .any(|p| !p.coords.iter().all(|c| c.is_finite()))
                {
                    continue;
                }
                let [a, b, c] = corners.map(|p| to_z_up(&p, options.up));
                let normal = (b - a).cross(&(c - a));
                let len = normal.norm();
                if len == 0.0 {
                    continue;
                }
                let facing = normal.dot(&eye) / len;
                let base = if facing >= 0.0 { FRONT_FACE } else { BACK_FACE };
                let light = 0.35 + 0.65 * facing.abs();
                let color = base.map(|ch| ch * light + 40.0 * (1.0 - light));
                let [p0, p1, p2] = corners.map(|p| project(&p));
                fill_triangle(p0, p1, p2, |x, y, z| plot(x, y, z, color, &mut image));
            }
        }
    }
    image
}

/// Call `plot` for every pixel centre inside the triangle, with interpolated
/// depth.
fn fill_triangle(
    a: (f32, f32, f32),
    b: (f32, f32, f32),
    c: (f32, f32, f32),
    mut plot: impl FnMut(i64, i64, f32),
) {
    let area = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
    if area.abs() < 1e-12 {
        return;
    }
    let min_x = a.0.min(b.0).min(c.0).floor().max(-1.0) as i64;
    let max_x = a.0.max(b.0).max(c.0).ceil().min(1e6) as i64;
    let min_y = a.1.min(b.1).min(c.1).floor().max(-1.0) as i64;
    let max_y = a.1.max(b.1).max(c.1).ceil().min(1e6) as i64;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let w0 = ((b.0 - px) * (c.1 - py) - (b.1 - py) * (c.0 - px)) / area;
            let w1 = ((c.0 - px) * (a.1 - py) - (c.1 - py) * (a.0 - px)) / area;
            let w2 = 1.0 - w0 - w1;
            if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                plot(x, y, w0 * a.2 + w1 * b.2 + w2 * c.2);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Share of a tile's pixels that are not background.
    fn coverage(image: &Image, col: u32, row: u32, tile: u32) -> f32 {
        let mut drawn = 0;
        for y in row * tile + 1..(row + 1) * tile {
            for x in col * tile + 1..(col + 1) * tile {
                let i = ((y * image.width + x) * 3) as usize;
                if image.pixels[i..i + 3] != BACKGROUND {
                    drawn += 1;
                }
            }
        }
        drawn as f32 / ((tile - 1) * (tile - 1)) as f32
    }

    fn has_color_near(image: &Image, target: [f32; 3]) -> bool {
        image.pixels.chunks_exact(3).any(|p| {
            let d: f32 = (0..3).map(|i| (p[i] as f32 - target[i]).abs()).sum();
            // Shading darkens colours, so compare hue roughly via the ratio.
            d < 120.0 && (p[0] as f32 > p[2] as f32) == (target[0] > target[2])
        })
    }

    /// Unit cube, wound outward.
    fn cube() -> TriangleMesh {
        let v = (0..8)
            .map(|i| Point3f::new((i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32))
            .collect();
        let faces = vec![
            [0, 2, 1],
            [1, 2, 3],
            [4, 5, 6],
            [5, 7, 6],
            [0, 1, 4],
            [1, 5, 4],
            [2, 6, 3],
            [3, 6, 7],
            [0, 4, 2],
            [2, 4, 6],
            [1, 3, 5],
            [3, 7, 5],
        ];
        TriangleMesh::from_vertices_and_faces(v, faces)
    }

    #[test]
    fn grid_has_one_tile_per_item_and_view_and_every_tile_shows_the_object() {
        let cube = cube();
        let cloud: Vec<Point3f> = (0..2000)
            .map(|i| {
                let t = i as f32 * 0.37;
                Point3f::new(t.sin(), t.cos(), (i as f32 / 2000.0) * 2.0 - 1.0)
            })
            .collect();
        let options = RenderOptions {
            tile_size: 128,
            ..Default::default()
        };
        let image =
            render_grid(&[Drawable::Mesh(&cube), Drawable::Points(&cloud)], &options).unwrap();
        assert_eq!((image.width, image.height), (5 * 128, 2 * 128));
        for row in 0..2 {
            for col in 0..5 {
                let c = coverage(&image, col, row, 128);
                assert!(c > 0.02, "tile row {row} col {col} is nearly empty ({c})");
            }
        }
    }

    #[test]
    fn items_in_different_units_each_fill_their_row() {
        // The same cube in metres and in millimetres: neither may shrink to a dot.
        let metres = cube();
        let mut mm = cube();
        mm.vertices.iter_mut().for_each(|v| *v *= 1000.0);
        let options = RenderOptions {
            tile_size: 128,
            ..Default::default()
        };
        let image = render_grid(&[Drawable::Mesh(&metres), Drawable::Mesh(&mm)], &options).unwrap();
        for row in 0..2 {
            let c = coverage(&image, 0, row, 128);
            assert!(c > 0.1, "row {row} is nearly empty ({c})");
        }
    }

    #[test]
    fn closed_mesh_shows_no_back_faces_and_an_open_one_does() {
        let options = RenderOptions {
            views: vec![View::Angled],
            tile_size: 128,
            ..Default::default()
        };
        let closed = cube();
        let image = render_grid(&[Drawable::Mesh(&closed)], &options).unwrap();
        assert!(
            !has_color_near(&image, BACK_FACE),
            "closed cube shows orange"
        );

        let mut open = cube();
        open.faces.retain(|f| f.iter().any(|&v| v < 4)); // remove the top (z = 1)
        let image = render_grid(&[Drawable::Mesh(&open)], &options).unwrap();
        assert!(
            has_color_near(&image, BACK_FACE),
            "the hole should show the orange inside"
        );
    }

    #[test]
    fn png_round_trip_and_bad_input() {
        let cube = cube();
        let image = render_grid(&[Drawable::Mesh(&cube)], &RenderOptions::default()).unwrap();
        let bytes = png_bytes(&image).unwrap();
        assert_eq!(&bytes[1..4], b"PNG");
        assert!(render_grid(&[], &RenderOptions::default()).is_err());
        let nan = [Point3f::new(f32::NAN, 0.0, 0.0)];
        assert!(render_grid(&[Drawable::Points(&nan)], &RenderOptions::default()).is_err());
        assert!(View::parse("sideways").is_err());
        assert_eq!(View::parse("Side").unwrap(), View::Right);
    }
}
