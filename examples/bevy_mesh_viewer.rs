//! ThreeCrate unified viewer — meshes and point clouds
//!
//! Controls:
//!   1 / 2 / 3  — Switch mesh (Sphere / Torus / Gyroid)
//!   4          — Show point cloud
//!   C          — Cycle material colour (mesh mode only)
//!   R          — Toggle auto-rotate
//!   Mouse drag — Orbit camera
//!   Scroll     — Zoom
//!   ESC        — Exit

#[cfg(feature = "bevy_interop")]
fn main() {
    use bevy::prelude::*;
    use viewer::*;

    println!("Generating initial mesh…");

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "ThreeCrate Viewer".to_string(),
                resolution: (1280u32, 720u32).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ViewerState::default())
        .add_systems(Startup, (setup_scene, setup_ui))
        .add_systems(
            Update,
            (
                update_scene,
                auto_rotate_mesh,
                rotate_camera,
                handle_input,
                update_status_text,
            ),
        )
        .run();
}

#[cfg(feature = "bevy_interop")]
mod viewer {
    use bevy::input::mouse::{MouseMotion, MouseWheel};
    use bevy::mesh::{PrimitiveTopology, VertexAttributeValues};
    use bevy::prelude::*;
    use bevy::asset::RenderAssetUsages;
    use threecrate_core::Point3f;
    use threecrate_reconstruction::marching_cubes::{
        create_sphere_volume, marching_cubes, VolumetricGrid,
    };

    // ── State ─────────────────────────────────────────────────────────────────

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub enum DisplayMode {
        Mesh(MeshType),
        PointCloud,
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub enum MeshType {
        Sphere,
        Torus,
        Gyroid,
    }

    impl MeshType {
        fn label(self) -> &'static str {
            match self {
                MeshType::Sphere => "Sphere",
                MeshType::Torus => "Torus",
                MeshType::Gyroid => "Gyroid",
            }
        }
    }

    const COLORS: &[(f32, f32, f32, &str)] = &[
        (0.25, 0.65, 0.95, "Ocean Blue"),
        (0.30, 0.85, 0.45, "Emerald"),
        (0.95, 0.50, 0.15, "Amber"),
        (0.90, 0.25, 0.55, "Rose"),
        (0.90, 0.90, 0.90, "Silver"),
    ];

    #[derive(Resource)]
    pub struct ViewerState {
        pub mode: DisplayMode,
        pub auto_rotate: bool,
        pub color_index: usize,
        pub scene_dirty: bool,
    }

    impl Default for ViewerState {
        fn default() -> Self {
            Self {
                mode: DisplayMode::Mesh(MeshType::Sphere),
                auto_rotate: true,
                color_index: 0,
                scene_dirty: false,
            }
        }
    }

    // ── Marker components ─────────────────────────────────────────────────────

    #[derive(Component)]
    pub struct SceneMesh;

    #[derive(Component)]
    pub struct StatusText;

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
                distance: 9.0,
                rotation_x: 0.35,
                rotation_y: 0.7,
            }
        }
    }

    // ── Mesh generation ───────────────────────────────────────────────────────

    fn create_torus_volume(
        center: Point3f,
        major_radius: f32,
        minor_radius: f32,
        resolution: [usize; 3],
        size: [f32; 3],
    ) -> VolumetricGrid {
        let origin = Point3f::new(
            center.x - size[0] / 2.0,
            center.y - size[1] / 2.0,
            center.z - size[2] / 2.0,
        );
        let voxel = [
            size[0] / (resolution[0] - 1) as f32,
            size[1] / (resolution[1] - 1) as f32,
            size[2] / (resolution[2] - 1) as f32,
        ];
        let mut grid = VolumetricGrid::new(resolution, voxel, origin);
        for x in 0..resolution[0] {
            for y in 0..resolution[1] {
                for z in 0..resolution[2] {
                    let w = grid.grid_to_world(x, y, z);
                    let xz = ((w.x - center.x).powi(2) + (w.z - center.z).powi(2)).sqrt()
                        - major_radius;
                    let d = (xz.powi(2) + (w.y - center.y).powi(2)).sqrt() - minor_radius;
                    grid.set_value(x, y, z, d).unwrap();
                }
            }
        }
        grid
    }

    fn create_gyroid_volume(resolution: [usize; 3], size: [f32; 3], scale: f32) -> VolumetricGrid {
        let origin = Point3f::new(-size[0] / 2.0, -size[1] / 2.0, -size[2] / 2.0);
        let voxel = [
            size[0] / (resolution[0] - 1) as f32,
            size[1] / (resolution[1] - 1) as f32,
            size[2] / (resolution[2] - 1) as f32,
        ];
        let mut grid = VolumetricGrid::new(resolution, voxel, origin);
        for x in 0..resolution[0] {
            for y in 0..resolution[1] {
                for z in 0..resolution[2] {
                    let w = grid.grid_to_world(x, y, z);
                    let px = w.x * scale;
                    let py = w.y * scale;
                    let pz = w.z * scale;
                    // Gyroid implicit surface: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0
                    let gyroid = px.sin() * py.cos() + py.sin() * pz.cos() + pz.sin() * px.cos();
                    // Intersect with a sphere to bound the surface
                    let sphere = (w.x * w.x + w.y * w.y + w.z * w.z).sqrt() - 2.8;
                    grid.set_value(x, y, z, gyroid.max(sphere)).unwrap();
                }
            }
        }
        grid
    }

    fn build_bevy_mesh(mesh_type: MeshType) -> Mesh {
        let grid = match mesh_type {
            MeshType::Sphere => create_sphere_volume(
                Point3f::new(0.0, 0.0, 0.0),
                2.2,
                [45, 45, 45],
                [6.0, 6.0, 6.0],
            ),
            MeshType::Torus => create_torus_volume(
                Point3f::new(0.0, 0.0, 0.0),
                2.0,
                0.75,
                [55, 55, 55],
                [7.0, 7.0, 7.0],
            ),
            MeshType::Gyroid => create_gyroid_volume([60, 60, 60], [7.0, 7.0, 7.0], 2.2),
        };
        let tri = marching_cubes(&grid, 0.0).expect("marching cubes failed");
        tri.to_bevy_mesh().expect("mesh conversion failed")
    }

    /// Build a rainbow sphere point cloud as a Bevy PointList mesh.
    fn build_point_cloud_mesh() -> Mesh {
        use std::f32::consts::PI;
        const N: usize = 10_000;
        let golden = PI * (3.0 - 5f32.sqrt());

        let mut positions: Vec<[f32; 3]> = Vec::with_capacity(N);
        let mut colors: Vec<[f32; 4]> = Vec::with_capacity(N);

        for i in 0..N {
            // Fibonacci sphere — uniform distribution
            let y_unit = 1.0 - (i as f32 / (N - 1) as f32) * 2.0;
            let r = (1.0 - y_unit * y_unit).sqrt();
            let theta = golden * i as f32;
            let scale = 2.5_f32;
            positions.push([theta.cos() * r * scale, y_unit * scale, theta.sin() * r * scale]);

            // Hue from 0° to 360° across the sphere
            let hue = (i as f32 / N as f32) * 360.0;
            let (cr, cg, cb) = hsv_to_rgb(hue, 1.0, 1.0);
            colors.push([cr, cg, cb, 1.0]);
        }

        let mut mesh = Mesh::new(PrimitiveTopology::PointList, RenderAssetUsages::default());
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
        mesh
    }

    /// HSV → linear RGB (simple, no gamma correction needed for point hues).
    fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
        let h = h % 360.0;
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;
        let (r, g, b) = if h < 60.0 { (c, x, 0.0) }
            else if h < 120.0 { (x, c, 0.0) }
            else if h < 180.0 { (0.0, c, x) }
            else if h < 240.0 { (0.0, x, c) }
            else if h < 300.0 { (x, 0.0, c) }
            else              { (c, 0.0, x) };
        (r + m, g + m, b + m)
    }

    fn make_material(color_index: usize) -> StandardMaterial {
        let (r, g, b, _) = COLORS[color_index];
        StandardMaterial {
            base_color: Color::srgb(r, g, b),
            metallic: 0.45,
            perceptual_roughness: 0.25,
            reflectance: 0.6,
            ..default()
        }
    }

    fn make_point_cloud_material() -> StandardMaterial {
        StandardMaterial {
            unlit: true,
            base_color: Color::WHITE,
            ..default()
        }
    }

    // ── Systems ───────────────────────────────────────────────────────────────

    pub fn setup_scene(
        mut commands: Commands,
        mut meshes: ResMut<Assets<bevy::mesh::Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
    ) {
        commands.spawn((
            Mesh3d(meshes.add(build_bevy_mesh(MeshType::Sphere))),
            MeshMaterial3d(materials.add(make_material(0))),
            Transform::default(),
            SceneMesh,
        ));

        // Key light
        commands.spawn((
            DirectionalLight {
                illuminance: 18_000.0,
                shadows_enabled: true,
                color: Color::srgb(1.0, 0.97, 0.90),
                ..default()
            },
            Transform::from_xyz(5.0, 10.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ));

        // Rim light
        commands.spawn((
            DirectionalLight {
                illuminance: 6_000.0,
                color: Color::srgb(0.6, 0.75, 1.0),
                shadows_enabled: false,
                ..default()
            },
            Transform::from_xyz(-5.0, 2.0, -5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ));

        // Ground bounce
        commands.spawn((
            DirectionalLight {
                illuminance: 2_000.0,
                color: Color::srgb(0.8, 0.85, 0.8),
                shadows_enabled: false,
                ..default()
            },
            Transform::from_xyz(0.0, -8.0, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
        ));

        commands.insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.45, 0.50, 0.60),
            brightness: 100.0,
            ..default()
        });

        // Camera
        let ctrl = CameraController::default();
        let (x, y, z) = camera_pos(&ctrl);
        commands.spawn((
            Camera3d::default(),
            Transform::from_xyz(x, y, z).looking_at(Vec3::ZERO, Vec3::Y),
            ctrl,
        ));
    }

    pub fn setup_ui(mut commands: Commands) {
        let state = ViewerState::default();
        commands.spawn((
            Text::new(status_string(&state)),
            TextFont {
                font_size: 15.0,
                ..default()
            },
            TextColor(Color::srgba(1.0, 1.0, 1.0, 0.9)),
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(12.0),
                left: Val::Px(12.0),
                ..default()
            },
            StatusText,
        ));
    }

    pub fn update_scene(
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
        mut state: ResMut<ViewerState>,
        query: Query<Entity, With<SceneMesh>>,
    ) {
        if !state.scene_dirty {
            return;
        }
        state.scene_dirty = false;

        for e in query.iter() {
            commands.entity(e).despawn();
        }

        match state.mode {
            DisplayMode::Mesh(mesh_type) => {
                println!("Generating {} mesh…", mesh_type.label());
                commands.spawn((
                    Mesh3d(meshes.add(build_bevy_mesh(mesh_type))),
                    MeshMaterial3d(materials.add(make_material(state.color_index))),
                    Transform::default(),
                    SceneMesh,
                ));
            }
            DisplayMode::PointCloud => {
                println!("Generating point cloud…");
                commands.spawn((
                    Mesh3d(meshes.add(build_point_cloud_mesh())),
                    MeshMaterial3d(materials.add(make_point_cloud_material())),
                    Transform::default(),
                    SceneMesh,
                ));
            }
        }
    }

    pub fn auto_rotate_mesh(
        state: Res<ViewerState>,
        time: Res<Time>,
        mut query: Query<&mut Transform, With<SceneMesh>>,
    ) {
        if !state.auto_rotate {
            return;
        }
        for mut t in query.iter_mut() {
            t.rotate_y(0.4 * time.delta_secs());
        }
    }

    pub fn handle_input(
        keyboard: Res<ButtonInput<KeyCode>>,
        mut exit: MessageWriter<bevy::app::AppExit>,
        mut state: ResMut<ViewerState>,
    ) {
        if keyboard.just_pressed(KeyCode::Escape) {
            exit.write(bevy::app::AppExit::Success);
        }
        if keyboard.just_pressed(KeyCode::Digit1) && state.mode != DisplayMode::Mesh(MeshType::Sphere) {
            state.mode = DisplayMode::Mesh(MeshType::Sphere);
            state.scene_dirty = true;
        }
        if keyboard.just_pressed(KeyCode::Digit2) && state.mode != DisplayMode::Mesh(MeshType::Torus) {
            state.mode = DisplayMode::Mesh(MeshType::Torus);
            state.scene_dirty = true;
        }
        if keyboard.just_pressed(KeyCode::Digit3) && state.mode != DisplayMode::Mesh(MeshType::Gyroid) {
            state.mode = DisplayMode::Mesh(MeshType::Gyroid);
            state.scene_dirty = true;
        }
        if keyboard.just_pressed(KeyCode::Digit4) && state.mode != DisplayMode::PointCloud {
            state.mode = DisplayMode::PointCloud;
            state.scene_dirty = true;
        }
        if keyboard.just_pressed(KeyCode::KeyR) {
            state.auto_rotate = !state.auto_rotate;
        }
        if keyboard.just_pressed(KeyCode::KeyC) {
            if matches!(state.mode, DisplayMode::Mesh(_)) {
                state.color_index = (state.color_index + 1) % COLORS.len();
                state.scene_dirty = true;
            }
        }
    }

    pub fn update_status_text(
        state: Res<ViewerState>,
        mut query: Query<&mut Text, With<StatusText>>,
    ) {
        if !state.is_changed() {
            return;
        }
        for mut text in query.iter_mut() {
            **text = status_string(&state);
        }
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
            for motion in mouse_motion.read() {
                ctrl.rotation_y -= motion.delta.x * 0.01 * ctrl.rotation_speed;
                ctrl.rotation_x -= motion.delta.y * 0.01 * ctrl.rotation_speed;
                ctrl.rotation_x = ctrl.rotation_x.clamp(-1.4, 1.4);
            }
        } else {
            mouse_motion.clear();
        }

        for wheel in mouse_wheel.read() {
            ctrl.distance -= wheel.y * ctrl.zoom_speed;
            ctrl.distance = ctrl.distance.clamp(3.0, 22.0);
        }

        let (x, y, z) = camera_pos(&ctrl);
        *transform = Transform::from_xyz(x, y, z).looking_at(Vec3::ZERO, Vec3::Y);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn camera_pos(ctrl: &CameraController) -> (f32, f32, f32) {
        let x = ctrl.rotation_y.sin() * ctrl.rotation_x.cos() * ctrl.distance;
        let y = ctrl.rotation_x.sin() * ctrl.distance;
        let z = ctrl.rotation_y.cos() * ctrl.rotation_x.cos() * ctrl.distance;
        (x, y, z)
    }

    fn status_string(state: &ViewerState) -> String {
        let mode_line = match state.mode {
            DisplayMode::Mesh(t) => {
                let (_, _, _, color_name) = COLORS[state.color_index];
                format!("Mode : Mesh — {} (1·2·3)\nColor: {} (C to cycle)", t.label(), color_name)
            }
            DisplayMode::PointCloud => "Mode : Point Cloud (4)".to_string(),
        };
        format!(
            "ThreeCrate Viewer\n\
             ─────────────────\n\
             {}\n\
             Spin : {} (R to toggle)\n\
             ─────────────────\n\
             Drag   — Orbit camera\n\
             Scroll — Zoom\n\
             ESC    — Exit",
            mode_line,
            if state.auto_rotate { "ON " } else { "OFF" },
        )
    }
}

#[cfg(not(feature = "bevy_interop"))]
fn main() {
    println!("This example requires the 'bevy_interop' feature.");
    println!(
        "Run with: cargo run --package threecrate-examples --bin bevy_mesh_viewer --features bevy_interop"
    );
}
