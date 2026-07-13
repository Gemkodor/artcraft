use glam::{Mat4, Vec3};
use winit::keyboard::KeyCode;

pub struct Camera {
    pub position: Vec3,
    /// Angle horizontal en radians (0 = +X, -90° = -Z).
    pub yaw: f32,
    /// Angle vertical en radians, borné à ±89° pour éviter le gimbal lock.
    pub pitch: f32,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn new(position: Vec3, yaw: f32, pitch: f32, aspect: f32) -> Self {
        Self {
            position,
            yaw,
            pitch,
            aspect,
            fovy: 70f32.to_radians(),
            znear: 0.1,
            zfar: 1000.0,
        }
    }

    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.sin(),
        )
        .normalize()
    }

    pub fn view_proj(&self) -> Mat4 {
        let proj = Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar);
        let view = Mat4::look_to_rh(self.position, self.forward(), Vec3::Y);
        proj * view
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn from_camera(camera: &Camera) -> Self {
        Self {
            view_proj: camera.view_proj().to_cols_array_2d(),
        }
    }
}

/// État des touches de déplacement + sensibilité souris. Les touches sont des
/// positions physiques (WASD sur QWERTY correspond automatiquement à ZQSD sur
/// AZERTY). Le mouvement lui-même est appliqué par la physique du joueur.
pub struct CameraController {
    sensitivity: f32,
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
}

impl CameraController {
    pub fn new(sensitivity: f32) -> Self {
        Self {
            sensitivity,
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
        }
    }

    pub fn process_key(&mut self, key: KeyCode, pressed: bool) -> bool {
        match key {
            KeyCode::KeyW => self.forward = pressed,
            KeyCode::KeyS => self.backward = pressed,
            KeyCode::KeyA => self.left = pressed,
            KeyCode::KeyD => self.right = pressed,
            KeyCode::Space => self.up = pressed,
            KeyCode::ShiftLeft => self.down = pressed,
            _ => return false,
        }
        true
    }

    pub fn process_mouse(&self, camera: &mut Camera, dx: f64, dy: f64) {
        camera.yaw += dx as f32 * self.sensitivity;
        camera.pitch -= dy as f32 * self.sensitivity;
        let limit = 89f32.to_radians();
        camera.pitch = camera.pitch.clamp(-limit, limit);
    }
}
