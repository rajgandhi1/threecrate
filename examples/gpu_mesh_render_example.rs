//! GPU Mesh Rendering Example
//!
//! Loads assets/bunny.obj, converts it to a Bevy mesh, and renders it
//! with PBR lighting using the same Bevy viewer as bevy_mesh_viewer.
//!
//! Controls:
//!   Mouse drag — Orbit camera
//!   Scroll     — Zoom
//!   ESC        — Exit
//!
//! Run with: cargo run -p threecrate-examples --bin gpu_mesh_render_example --features bevy_interop

#[cfg(feature = "bevy_interop")]
fn main() {
    use bevy::prelude::*;
    use threecrate_io::obj::RobustObjReader;

    let obj_path = "assets/bunny.obj";
    println!("Loading {}…", obj_path);
    let obj_data = RobustObjReader::read_obj_file(obj_path).expect("failed to read bunny.obj");
    let mesh = RobustObjReader::obj_data_to_mesh(&obj_data).expect("failed to parse obj");

    println!(
        "Loaded: {} vertices, {} faces",
        mesh.vertex_count(),
        mesh.face_count()
    );

    // Compute bounding sphere so setup_scene can centre and zoom correctly.
    let (cx, cy, cz) = mesh
        .vertices
        .iter()
        .fold((0f32, 0f32, 0f32), |(ax, ay, az), v| {
            (ax + v.x, ay + v.y, az + v.z)
        });
    let n = mesh.vertices.len() as f32;
    let center = Vec3::new(cx / n, cy / n, cz / n);
    let radius = mesh
        .vertices
        .iter()
        .map(|v| (Vec3::new(v.x, v.y, v.z) - center).length())
        .fold(0f32, f32::max);

    println!(
        "Bounding sphere: center={:.2?} radius={:.2}",
        center, radius
    );

    let mut bevy_mesh = mesh.to_bevy_mesh().expect("failed to convert to Bevy mesh");
    // OBJ normal indices are per-face-vertex and don't survive the vertex
    // deduplication in obj_data_to_mesh, so recompute them from geometry.
    bevy_mesh.compute_smooth_normals();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: format!(
                    "ThreeCrate GPU Mesh — bunny.obj ({} verts, {} faces)",
                    mesh.vertex_count(),
                    mesh.face_count()
                ),
                resolution: (1280u32, 720u32).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(viewer::LoadedMesh {
            mesh: bevy_mesh,
            center,
            radius,
        })
        .add_systems(Startup, viewer::setup_scene)
        .add_systems(Update, (viewer::rotate_camera, viewer::handle_input))
        .run();
}

#[cfg(feature = "bevy_interop")]
mod viewer {
    use bevy::input::mouse::{MouseMotion, MouseWheel};
    use bevy::prelude::*;

    #[derive(Resource)]
    pub struct LoadedMesh {
        pub mesh: Mesh,
        pub center: Vec3,
        pub radius: f32,
    }

    #[derive(Component)]
    pub struct CameraController {
        pub rotation_speed: f32,
        pub zoom_speed: f32,
        pub distance: f32,
        pub rotation_x: f32,
        pub rotation_y: f32,
        pub target: Vec3,
    }

    pub fn setup_scene(
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
        loaded: Res<LoadedMesh>,
    ) {
        // Centre the mesh at the world origin via its Transform.
        commands.spawn((
            Mesh3d(meshes.add(loaded.mesh.clone())),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.85, 0.85, 0.85),
                metallic: 0.1,
                perceptual_roughness: 0.6,
                ..default()
            })),
            Transform::from_translation(-loaded.center),
        ));

        commands.spawn((
            DirectionalLight {
                illuminance: 18_000.0,
                shadows_enabled: true,
                color: Color::srgb(1.0, 0.97, 0.90),
                ..default()
            },
            Transform::from_xyz(5.0, 10.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ));
        commands.spawn((
            DirectionalLight {
                illuminance: 5_000.0,
                color: Color::srgb(0.6, 0.75, 1.0),
                shadows_enabled: false,
                ..default()
            },
            Transform::from_xyz(-5.0, 2.0, -5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ));
        commands.insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.5, 0.5, 0.6),
            brightness: 80.0,
            ..default()
        });

        // Place the camera far enough to see the whole mesh.
        let distance = loaded.radius * 2.5;
        let ctrl = CameraController {
            rotation_speed: 0.5,
            zoom_speed: loaded.radius * 0.2,
            distance,
            rotation_x: 0.3,
            rotation_y: 0.6,
            target: Vec3::ZERO,
        };
        let (x, y, z) = cam_pos(&ctrl);
        commands.spawn((
            Camera3d::default(),
            Transform::from_xyz(x, y, z).looking_at(Vec3::ZERO, Vec3::Y),
            ctrl,
        ));
    }

    pub fn rotate_camera(
        mut mouse_motion: MessageReader<MouseMotion>,
        mut mouse_wheel: MessageReader<MouseWheel>,
        mouse_button: Res<ButtonInput<MouseButton>>,
        mut query: Query<(&mut Transform, &mut CameraController)>,
    ) {
        let Ok((mut transform, mut ctrl)) = query.single_mut() else {
            return;
        };

        if mouse_button.pressed(MouseButton::Left) {
            for ev in mouse_motion.read() {
                ctrl.rotation_y -= ev.delta.x * 0.01 * ctrl.rotation_speed;
                ctrl.rotation_x -= ev.delta.y * 0.01 * ctrl.rotation_speed;
                ctrl.rotation_x = ctrl.rotation_x.clamp(-1.4, 1.4);
            }
        } else {
            mouse_motion.clear();
        }

        for ev in mouse_wheel.read() {
            ctrl.distance -= ev.y * ctrl.zoom_speed;
            ctrl.distance = ctrl
                .distance
                .clamp(ctrl.zoom_speed, ctrl.distance.max(ctrl.zoom_speed) * 4.0);
        }

        let (x, y, z) = cam_pos(&ctrl);
        *transform = Transform::from_xyz(x, y, z).looking_at(ctrl.target, Vec3::Y);
    }

    pub fn handle_input(
        keyboard: Res<ButtonInput<KeyCode>>,
        mut exit: MessageWriter<bevy::app::AppExit>,
    ) {
        if keyboard.just_pressed(KeyCode::Escape) {
            exit.write(bevy::app::AppExit::Success);
        }
    }

    fn cam_pos(ctrl: &CameraController) -> (f32, f32, f32) {
        let x = ctrl.rotation_y.sin() * ctrl.rotation_x.cos() * ctrl.distance;
        let y = ctrl.rotation_x.sin() * ctrl.distance;
        let z = ctrl.rotation_y.cos() * ctrl.rotation_x.cos() * ctrl.distance;
        (x, y, z)
    }
}

#[cfg(not(feature = "bevy_interop"))]
fn main() {
    println!("This example requires the 'bevy_interop' feature.");
    println!("Run with: cargo run -p threecrate-examples --bin gpu_mesh_render_example --features bevy_interop");
}
