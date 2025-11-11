//! Interactive Bevy mesh viewer for threecrate meshes
//!
//! This example demonstrates:
//! 1. Generating a mesh using marching cubes
//! 2. Converting to Bevy mesh format
//! 3. Displaying in a 3D window with camera controls
//!
//! Run with: cargo run --package threecrate-examples --bin bevy_mesh_viewer --features bevy_interop

#[cfg(feature = "bevy_interop")]
fn main() {
    use bevy::prelude::*;
    use threecrate_core::Point3f;
    use threecrate_reconstruction::marching_cubes::{create_cube_volume, marching_cubes};

    println!("Generating mesh with marching cubes...");

    // Generate a cube mesh using a signed distance field
    let cube_grid = create_cube_volume(
        Point3f::new(0.0, 0.0, 0.0),
        2.0, // half-size (cube will be 4x4x4)
        [30, 30, 30],
        [6.0, 6.0, 6.0],
    );

    let triangle_mesh = marching_cubes(&cube_grid, 0.0)
        .expect("Failed to generate mesh");

    println!("Generated mesh:");
    println!("  - Vertices: {}", triangle_mesh.vertex_count());
    println!("  - Faces: {}", triangle_mesh.face_count());
    println!("  - Has normals: {}", triangle_mesh.normals.is_some());

    // Convert to Bevy mesh
    let bevy_mesh = triangle_mesh.to_bevy_mesh()
        .expect("Failed to convert to Bevy mesh");

    println!("\nLaunching Bevy viewer...");
    println!("Controls:");
    println!("  - Left mouse: Rotate camera");
    println!("  - Right mouse: Pan camera");
    println!("  - Scroll: Zoom in/out");
    println!("  - ESC: Exit");

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "ThreeCrate + Bevy Mesh Viewer".to_string(),
                resolution: (1024., 768.).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(GeneratedMesh(bevy_mesh))
        .add_systems(Startup, setup_scene)
        .add_systems(Update, (rotate_camera, handle_input))
        .run();
}

#[cfg(feature = "bevy_interop")]
mod bevy_viewer {
    use bevy::prelude::*;
    use bevy::input::mouse::{MouseMotion, MouseWheel};

    #[derive(Resource)]
    pub struct GeneratedMesh(pub bevy::render::mesh::Mesh);

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
                distance: 8.0,
                rotation_x: 0.3,
                rotation_y: 0.5,
            }
        }
    }

    pub fn setup_scene(
        mut commands: Commands,
        mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
        generated_mesh: Res<GeneratedMesh>,
    ) {
        // Add the generated mesh
        let mesh_handle = meshes.add(generated_mesh.0.clone());
        let material_handle = materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.7, 0.9),
            metallic: 0.3,
            perceptual_roughness: 0.5,
            ..default()
        });

        commands.spawn((
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(material_handle),
            Transform::from_xyz(0.0, 0.0, 0.0),
        ));

        // Add a second mesh with wireframe-like appearance
        let material_handle2 = materials.add(StandardMaterial {
            base_color: Color::srgb(0.1, 0.1, 0.1),
            metallic: 0.8,
            perceptual_roughness: 0.2,
            ..default()
        });

        commands.spawn((
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(material_handle2),
            Transform::from_xyz(0.0, 0.0, 0.0).with_scale(Vec3::splat(1.01)),
        ));

        // Main directional light
        commands.spawn((
            DirectionalLight {
                illuminance: 10000.0,
                shadows_enabled: true,
                ..default()
            },
            Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
        ));

        // Fill light
        commands.spawn((
            PointLight {
                intensity: 500000.0,
                color: Color::srgb(0.8, 0.9, 1.0),
                shadows_enabled: false,
                ..default()
            },
            Transform::from_xyz(-5.0, 3.0, -5.0),
        ));

        // Ambient light
        commands.insert_resource(AmbientLight {
            color: Color::srgb(0.5, 0.5, 0.6),
            brightness: 150.0,
        });

        // Camera with controller
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

        println!("Scene setup complete!");
        println!("Mesh rendered in window");
    }

    pub fn rotate_camera(
        mut mouse_motion: EventReader<MouseMotion>,
        mut mouse_wheel: EventReader<MouseWheel>,
        mouse_button: Res<ButtonInput<MouseButton>>,
        mut query: Query<(&mut Transform, &mut CameraController)>,
    ) {
        let (mut transform, mut controller) = query.single_mut();

        // Handle mouse rotation
        if mouse_button.pressed(MouseButton::Left) {
            for motion in mouse_motion.read() {
                controller.rotation_y -= motion.delta.x * 0.01 * controller.rotation_speed;
                controller.rotation_x -= motion.delta.y * 0.01 * controller.rotation_speed;

                // Clamp vertical rotation
                controller.rotation_x = controller.rotation_x.clamp(-1.5, 1.5);
            }
        } else {
            // Clear the event reader even if not used
            mouse_motion.clear();
        }

        // Handle mouse wheel zoom
        for wheel in mouse_wheel.read() {
            controller.distance -= wheel.y * controller.zoom_speed;
            controller.distance = controller.distance.clamp(2.0, 20.0);
        }

        // Update camera position based on rotation and distance
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
use bevy_viewer::*;

#[cfg(not(feature = "bevy_interop"))]
fn main() {
    println!("This example requires the 'bevy_interop' feature.");
    println!("Run with: cargo run --package threecrate-examples --bin bevy_mesh_viewer --features bevy_interop");
}
