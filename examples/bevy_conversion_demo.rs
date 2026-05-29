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

    println!("===========================================");
    println!("  Bevy Mesh Conversion Demonstration");
    println!("===========================================\n");

    // Step 1: Build a literal cube mesh with sharp edges.
    println!("Step 1: Generating a cube mesh...");
    let mut triangle_mesh = make_cube_mesh(Point3f::new(0.0, 0.0, 0.0), 1.5);
    println!("✓ Cube mesh generated");
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
    println!("  Type: bevy::prelude::Mesh");
    println!("  Primitive Topology: TriangleList");
    println!("  Vertex count : {}", bevy_mesh.count_vertices());
    println!("  Has normals  : {}", bevy_mesh.attribute(Mesh::ATTRIBUTE_NORMAL).is_some());
    println!("  Has colors   : {}", bevy_mesh.attribute(Mesh::ATTRIBUTE_COLOR).is_some());
    println!("  Has indices  : {}", bevy_mesh.indices().is_some());
    println!();

    // Step 6: Launch Bevy viewer
    println!("Step 6: Launching interactive 3D viewer...");
    println!("  The converted Bevy mesh will be displayed in the window");
    println!();
    println!("Controls:");
    println!("  - Left mouse: Rotate cube");
    println!("  - Scroll: Zoom in/out");
    println!("  - ESC: Exit");
    println!();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bevy Mesh Conversion Demo - Orange Cube".to_string(),
                resolution: (1024u32, 768u32).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ConvertedMesh(bevy_mesh))
        .insert_resource(HelpOverlayVisible(true))
        .add_systems(Startup, (setup_demo_scene, setup_help_overlay))
        .add_systems(Update, (rotate_mesh, zoom_camera, handle_input))
        .run();
}

#[cfg(feature = "bevy_interop")]
mod demo {
    use bevy::prelude::*;
    use bevy::input::mouse::{MouseMotion, MouseWheel};

    #[derive(Resource)]
    pub struct ConvertedMesh(pub Mesh);

    #[derive(Component)]
    pub struct SceneMesh;

    #[derive(Component)]
    pub struct HelpOverlay;

    #[derive(Resource)]
    pub struct HelpOverlayVisible(pub bool);

    #[derive(Component)]
    pub struct CameraController {
        pub zoom_speed: f32,
        pub distance: f32,
        pub target: Vec3,
        pub direction: Vec3,
    }

    #[derive(Component)]
    pub struct MeshController {
        pub rotation_speed: f32,
    }

    impl Default for CameraController {
        fn default() -> Self {
            Self {
                zoom_speed: 0.35,
                distance: 8.0,
                target: Vec3::ZERO,
                direction: Vec3::new(0.65, 0.45, 0.62).normalize(),
            }
        }
    }

    impl Default for MeshController {
        fn default() -> Self {
            Self {
                rotation_speed: 0.01,
            }
        }
    }

    pub fn setup_demo_scene(
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
        converted_mesh: Res<ConvertedMesh>,
    ) {
        // Add the converted Bevy mesh with vertex colors
        let mesh_handle = meshes.add(converted_mesh.0.clone());
        let material_handle = materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.39, 0.20), // orange — matches the vertex colors set in main()
            metallic: 0.2,
            perceptual_roughness: 0.5,
            ..default()
        });

        commands.spawn((
            Mesh3d(mesh_handle),
            MeshMaterial3d(material_handle),
            Transform::from_xyz(0.0, 0.0, 0.0),
            SceneMesh,
            MeshController::default(),
        ));

        // Key light
        commands.spawn((
            DirectionalLight {
                illuminance: 15_000.0,
                shadows_enabled: true,
                color: Color::srgb(1.0, 0.97, 0.90),
                ..default()
            },
            Transform::from_xyz(5.0, 8.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ));

        // Fill light
        commands.spawn((
            DirectionalLight {
                illuminance: 4_000.0,
                color: Color::srgb(0.6, 0.75, 1.0),
                shadows_enabled: false,
                ..default()
            },
            Transform::from_xyz(-4.0, 2.0, -4.0).looking_at(Vec3::ZERO, Vec3::Y),
        ));

        // Ambient light
        commands.insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.5, 0.5, 0.6),
            brightness: 80.0,
            ..default()
        });

        // Camera
        let controller = CameraController::default();
        commands.spawn((
            Camera3d::default(),
            Transform::from_translation(controller.direction * controller.distance)
                .looking_at(controller.target, Vec3::Y),
            controller,
        ));

        println!("✓ Scene setup complete - Bevy mesh is now rendered!");
        println!("  The orange cube you see is the converted Bevy mesh");
        println!("  with vertex colors from TriangleMesh");
    }

    pub fn setup_help_overlay(mut commands: Commands) {
        commands.spawn((
            Text::new(
                "Navigation\n\
                 Left drag  Rotate cube\n\
                 Scroll     Zoom camera\n\
                 H          Hide help\n\
                 Esc        Exit",
            ),
            TextFont {
                font_size: 14.0,
                ..default()
            },
            TextColor(Color::srgba(1.0, 1.0, 1.0, 0.92)),
            BackgroundColor(Color::srgba(0.03, 0.03, 0.03, 0.55)),
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(12.0),
                bottom: Val::Px(12.0),
                padding: UiRect::all(Val::Px(10.0)),
                ..default()
            },
            HelpOverlay,
        ));
    }

    pub fn rotate_mesh(
        mut mouse_motion: MessageReader<MouseMotion>,
        mouse_button: Res<ButtonInput<MouseButton>>,
        mut query: Query<(&mut Transform, &MeshController), With<SceneMesh>>,
    ) {
        let Ok((mut transform, controller)) = query.single_mut() else {
            return;
        };

        if mouse_button.pressed(MouseButton::Left) {
            for motion in mouse_motion.read() {
                transform.rotate_y(-motion.delta.x * controller.rotation_speed);
                transform.rotate_x(motion.delta.y * controller.rotation_speed);
            }
        } else {
            mouse_motion.clear();
        }
    }

    pub fn zoom_camera(
        mut mouse_wheel: MessageReader<MouseWheel>,
        mut query: Query<(&mut Transform, &mut CameraController)>,
    ) {
        let Ok((mut transform, mut controller)) = query.single_mut() else {
            return;
        };

        for wheel in mouse_wheel.read() {
            controller.distance -= wheel.y * controller.zoom_speed;
            controller.distance = controller.distance.clamp(3.0, 20.0);
        }

        *transform = Transform::from_translation(controller.direction * controller.distance)
            .looking_at(controller.target, Vec3::Y);
    }

    pub fn handle_input(
        keyboard: Res<ButtonInput<KeyCode>>,
        mut exit: MessageWriter<bevy::app::AppExit>,
        mut help_visible: ResMut<HelpOverlayVisible>,
        mut help_query: Query<&mut Visibility, With<HelpOverlay>>,
    ) {
        if keyboard.just_pressed(KeyCode::Escape) {
            exit.write(bevy::app::AppExit::Success);
        }
        if keyboard.just_pressed(KeyCode::KeyH) {
            help_visible.0 = !help_visible.0;
            let visibility = if help_visible.0 {
                Visibility::Visible
            } else {
                Visibility::Hidden
            };
            for mut v in help_query.iter_mut() {
                *v = visibility;
            }
        }
    }
}

#[cfg(feature = "bevy_interop")]
fn make_cube_mesh(center: threecrate_core::Point3f, half_size: f32) -> threecrate_core::TriangleMesh {
    use threecrate_core::{Point3f, TriangleMesh, Vector3f};

    let (cx, cy, cz) = (center.x, center.y, center.z);
    let (x0, x1) = (cx - half_size, cx + half_size);
    let (y0, y1) = (cy - half_size, cy + half_size);
    let (z0, z1) = (cz - half_size, cz + half_size);

    let mut vertices = Vec::with_capacity(24);
    let mut normals = Vec::with_capacity(24);
    let mut faces = Vec::with_capacity(12);

    let mut add_face = |quad: [(Point3f, Vector3f); 4]| {
        let start = vertices.len();
        for (pos, normal) in quad {
            vertices.push(pos);
            normals.push(normal);
        }
        faces.push([start, start + 1, start + 2]);
        faces.push([start, start + 2, start + 3]);
    };

    add_face([
        (Point3f::new(x0, y0, z1), Vector3f::new(0.0, 0.0, 1.0)),
        (Point3f::new(x1, y0, z1), Vector3f::new(0.0, 0.0, 1.0)),
        (Point3f::new(x1, y1, z1), Vector3f::new(0.0, 0.0, 1.0)),
        (Point3f::new(x0, y1, z1), Vector3f::new(0.0, 0.0, 1.0)),
    ]);
    add_face([
        (Point3f::new(x1, y0, z0), Vector3f::new(0.0, 0.0, -1.0)),
        (Point3f::new(x0, y0, z0), Vector3f::new(0.0, 0.0, -1.0)),
        (Point3f::new(x0, y1, z0), Vector3f::new(0.0, 0.0, -1.0)),
        (Point3f::new(x1, y1, z0), Vector3f::new(0.0, 0.0, -1.0)),
    ]);
    add_face([
        (Point3f::new(x0, y0, z0), Vector3f::new(-1.0, 0.0, 0.0)),
        (Point3f::new(x0, y0, z1), Vector3f::new(-1.0, 0.0, 0.0)),
        (Point3f::new(x0, y1, z1), Vector3f::new(-1.0, 0.0, 0.0)),
        (Point3f::new(x0, y1, z0), Vector3f::new(-1.0, 0.0, 0.0)),
    ]);
    add_face([
        (Point3f::new(x1, y0, z1), Vector3f::new(1.0, 0.0, 0.0)),
        (Point3f::new(x1, y0, z0), Vector3f::new(1.0, 0.0, 0.0)),
        (Point3f::new(x1, y1, z0), Vector3f::new(1.0, 0.0, 0.0)),
        (Point3f::new(x1, y1, z1), Vector3f::new(1.0, 0.0, 0.0)),
    ]);
    add_face([
        (Point3f::new(x0, y1, z1), Vector3f::new(0.0, 1.0, 0.0)),
        (Point3f::new(x1, y1, z1), Vector3f::new(0.0, 1.0, 0.0)),
        (Point3f::new(x1, y1, z0), Vector3f::new(0.0, 1.0, 0.0)),
        (Point3f::new(x0, y1, z0), Vector3f::new(0.0, 1.0, 0.0)),
    ]);
    add_face([
        (Point3f::new(x0, y0, z0), Vector3f::new(0.0, -1.0, 0.0)),
        (Point3f::new(x1, y0, z0), Vector3f::new(0.0, -1.0, 0.0)),
        (Point3f::new(x1, y0, z1), Vector3f::new(0.0, -1.0, 0.0)),
        (Point3f::new(x0, y0, z1), Vector3f::new(0.0, -1.0, 0.0)),
    ]);

    let mut mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
    mesh.set_normals(normals);
    mesh
}

#[cfg(feature = "bevy_interop")]
use demo::*;

#[cfg(not(feature = "bevy_interop"))]
fn main() {
    println!("This example requires the 'bevy_interop' feature.");
    println!("Run with: cargo run --package threecrate-examples --bin bevy_conversion_demo --features bevy_interop");
}
