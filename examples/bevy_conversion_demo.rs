//! Bevy Mesh Conversion Demo
//!
//! This example demonstrates the conversion process from TriangleMesh to Bevy Mesh
//! by showing the mesh attributes before and after conversion.
//!
//! Run with: cargo run --package threecrate-examples --bin bevy_conversion_demo --features bevy_interop

#[cfg(feature = "bevy_interop")]
fn main() {
    use bevy::prelude::*;
    use threecrate_core::Point3f;
    use threecrate_reconstruction::marching_cubes::{create_cube_volume, marching_cubes};

    println!("===========================================");
    println!("  Bevy Mesh Conversion Demonstration");
    println!("===========================================\n");

    // Step 1: Generate a mesh using marching cubes
    println!("Step 1: Generating mesh with Marching Cubes...");
    let cube_grid = create_cube_volume(
        Point3f::new(0.0, 0.0, 0.0),
        1.5,
        [20, 20, 20],
        [4.0, 4.0, 4.0],
    );

    let mut triangle_mesh = marching_cubes(&cube_grid, 0.0)
        .expect("Failed to generate mesh");

    println!("✓ Marching Cubes mesh generated");
    println!();

    // Step 2: Show original TriangleMesh details
    println!("Step 2: Original TriangleMesh Details:");
    println!("  Type: threecrate_core::TriangleMesh");
    println!("  Vertices: {}", triangle_mesh.vertex_count());
    println!("  Faces: {}", triangle_mesh.face_count());
    println!("  Has normals: {}", triangle_mesh.normals.is_some());
    println!("  Has colors: {}", triangle_mesh.colors.is_some());

    if let Some(ref normals) = triangle_mesh.normals {
        println!("  Normal count: {}", normals.len());
    }
    println!();

    // Add colors for demonstration
    println!("Step 3: Adding vertex colors...");
    let colors = vec![[255u8, 100, 50]; triangle_mesh.vertex_count()];
    triangle_mesh.set_colors(colors);
    println!("✓ Added {} vertex colors (orange)", triangle_mesh.vertex_count());
    println!();

    // Step 4: Convert to Bevy Mesh
    println!("Step 4: Converting to Bevy Mesh...");
    let bevy_mesh = triangle_mesh.to_bevy_mesh()
        .expect("Failed to convert to Bevy mesh");
    println!("✓ Conversion successful!");
    println!();

    // Step 5: Show Bevy Mesh details
    println!("Step 5: Bevy Mesh Details:");
    println!("  Type: bevy::render::mesh::Mesh");
    println!("  Primitive Topology: TriangleList");

    if let Some(positions) = bevy_mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        if let bevy::render::mesh::VertexAttributeValues::Float32x3(verts) = positions {
            println!("  Position attribute: {} vertices", verts.len());
        }
    }

    if let Some(normals) = bevy_mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
        if let bevy::render::mesh::VertexAttributeValues::Float32x3(norms) = normals {
            println!("  Normal attribute: {} normals", norms.len());
        }
    }

    if let Some(colors) = bevy_mesh.attribute(Mesh::ATTRIBUTE_COLOR) {
        if let bevy::render::mesh::VertexAttributeValues::Float32x4(cols) = colors {
            println!("  Color attribute: {} colors (RGBA)", cols.len());
        }
    }

    if let Some(indices) = bevy_mesh.indices() {
        match indices {
            bevy::render::mesh::Indices::U16(idx) => println!("  Indices: {} (U16)", idx.len()),
            bevy::render::mesh::Indices::U32(idx) => println!("  Indices: {} (U32)", idx.len()),
        }
    }
    println!();

    // Step 6: Launch Bevy viewer
    println!("Step 6: Launching interactive 3D viewer...");
    println!("  The converted Bevy mesh will be displayed in the window");
    println!();
    println!("Controls:");
    println!("  - Left mouse: Rotate camera");
    println!("  - Scroll: Zoom in/out");
    println!("  - ESC: Exit");
    println!();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bevy Mesh Conversion Demo - Orange Cube".to_string(),
                resolution: (1024., 768.).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ConvertedMesh(bevy_mesh))
        .add_systems(Startup, setup_demo_scene)
        .add_systems(Update, (rotate_camera, handle_input))
        .run();
}

#[cfg(feature = "bevy_interop")]
mod demo {
    use bevy::prelude::*;
    use bevy::input::mouse::{MouseMotion, MouseWheel};

    #[derive(Resource)]
    pub struct ConvertedMesh(pub bevy::render::mesh::Mesh);

    #[derive(Component)]
    pub struct CameraController {
        pub rotation_speed: f32,
        pub zoom_speed: f32,
        pub distance: f32,
        pub rotation_x: f32,
        pub rotation_y: f32,
    }

    impl Default for CameraController {
        fn default() -> Self {
            Self {
                rotation_speed: 0.5,
                zoom_speed: 0.5,
                distance: 6.0,
                rotation_x: 0.4,
                rotation_y: 0.8,
            }
        }
    }

    pub fn setup_demo_scene(
        mut commands: Commands,
        mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
        converted_mesh: Res<ConvertedMesh>,
    ) {
        // Add the converted Bevy mesh with vertex colors
        let mesh_handle = meshes.add(converted_mesh.0.clone());
        let material_handle = materials.add(StandardMaterial {
            base_color: Color::WHITE, // Use white to show vertex colors
            metallic: 0.1,
            perceptual_roughness: 0.7,
            ..default()
        });

        commands.spawn((
            Mesh3d(mesh_handle),
            MeshMaterial3d(material_handle),
            Transform::from_xyz(0.0, 0.0, 0.0),
        ));

        // Directional light
        commands.spawn((
            DirectionalLight {
                illuminance: 8000.0,
                shadows_enabled: false,
                ..default()
            },
            Transform::from_xyz(3.0, 5.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
        ));

        // Ambient light
        commands.insert_resource(AmbientLight {
            color: Color::srgb(0.6, 0.6, 0.7),
            brightness: 200.0,
        });

        // Camera
        let controller = CameraController::default();
        commands.spawn((
            Camera3d::default(),
            Transform::from_xyz(
                controller.rotation_y.sin() * controller.rotation_x.cos() * controller.distance,
                controller.rotation_x.sin() * controller.distance,
                controller.rotation_y.cos() * controller.rotation_x.cos() * controller.distance,
            ).looking_at(Vec3::ZERO, Vec3::Y),
            controller,
        ));

        println!("✓ Scene setup complete - Bevy mesh is now rendered!");
        println!("  The orange cube you see is the converted Bevy mesh");
        println!("  with vertex colors from TriangleMesh");
    }

    pub fn rotate_camera(
        mut mouse_motion: EventReader<MouseMotion>,
        mut mouse_wheel: EventReader<MouseWheel>,
        mouse_button: Res<ButtonInput<MouseButton>>,
        mut query: Query<(&mut Transform, &mut CameraController)>,
    ) {
        let (mut transform, mut controller) = query.single_mut();

        if mouse_button.pressed(MouseButton::Left) {
            for motion in mouse_motion.read() {
                controller.rotation_y -= motion.delta.x * 0.01 * controller.rotation_speed;
                controller.rotation_x -= motion.delta.y * 0.01 * controller.rotation_speed;
                controller.rotation_x = controller.rotation_x.clamp(-1.5, 1.5);
            }
        } else {
            mouse_motion.clear();
        }

        for wheel in mouse_wheel.read() {
            controller.distance -= wheel.y * controller.zoom_speed;
            controller.distance = controller.distance.clamp(2.0, 15.0);
        }

        let x = controller.rotation_y.sin() * controller.rotation_x.cos() * controller.distance;
        let y = controller.rotation_x.sin() * controller.distance;
        let z = controller.rotation_y.cos() * controller.rotation_x.cos() * controller.distance;

        *transform = Transform::from_xyz(x, y, z).looking_at(Vec3::ZERO, Vec3::Y);
    }

    pub fn handle_input(
        keyboard: Res<ButtonInput<KeyCode>>,
        mut exit: EventWriter<bevy::app::AppExit>,
    ) {
        if keyboard.just_pressed(KeyCode::Escape) {
            exit.send(bevy::app::AppExit::Success);
        }
    }
}

#[cfg(feature = "bevy_interop")]
use demo::*;

#[cfg(not(feature = "bevy_interop"))]
fn main() {
    println!("This example requires the 'bevy_interop' feature.");
    println!("Run with: cargo run --package threecrate-examples --bin bevy_conversion_demo --features bevy_interop");
}
