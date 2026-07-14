//! Cycle jour/nuit : position du soleil, intensité de la lumière du ciel et
//! couleur du ciel (qui sert aussi de couleur de brouillard). Le shader et le
//! meshing n'ont rien à recalculer : l'intensité module simplement le canal
//! "ciel" de la lumière pré-calculée par sommet.

use glam::Vec3;

/// Durée d'un cycle complet jour + nuit, en secondes.
pub const DAY_LENGTH: f32 = 480.0;

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub struct Sky {
    time: f32,
}

impl Sky {
    pub fn new() -> Self {
        // On démarre en début de matinée, soleil déjà levé.
        Self {
            time: DAY_LENGTH * 0.06,
        }
    }

    pub fn advance(&mut self, dt: f32) {
        self.time = (self.time + dt) % DAY_LENGTH;
    }

    /// Angle du soleil sur son orbite (0 = lever, π/2 = zénith).
    fn angle(&self) -> f32 {
        self.time / DAY_LENGTH * std::f32::consts::TAU
    }

    /// 1.0 en plein jour, 0.0 en pleine nuit, transition douce au crépuscule.
    fn day_factor(&self) -> f32 {
        smoothstep(-0.08, 0.25, self.angle().sin())
    }

    /// Direction VERS le soleil (utilisée pour ombrer les faces).
    pub fn sun_dir(&self) -> Vec3 {
        let a = self.angle();
        Vec3::new(a.cos(), a.sin(), 0.3).normalize()
    }

    /// Multiplicateur de la lumière du ciel : la nuit n'est jamais noire
    /// (clair de lune), le plein jour vaut 1.
    pub fn sun_intensity(&self) -> f32 {
        0.13 + 0.87 * self.day_factor()
    }

    /// Couleur du ciel — et du brouillard, pour que le terrain se fonde
    /// dedans à l'horizon.
    pub fn sky_color(&self) -> Vec3 {
        let day = Vec3::new(0.45, 0.70, 1.0);
        let night = Vec3::new(0.012, 0.017, 0.045);
        let base = night.lerp(day, self.day_factor());

        // Teinte orangée quand le soleil frôle l'horizon.
        let sunset = (1.0 - self.angle().sin().abs() / 0.22).clamp(0.0, 1.0);
        base.lerp(Vec3::new(0.95, 0.45, 0.22), sunset * 0.45)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sky_at(fraction: f32) -> Sky {
        let mut sky = Sky::new();
        sky.time = DAY_LENGTH * fraction;
        sky
    }

    #[test]
    fn noon_is_brighter_than_midnight() {
        let noon = sky_at(0.25); // angle = π/2, zénith
        let midnight = sky_at(0.75);
        assert!(noon.sun_intensity() > 0.9);
        assert!(midnight.sun_intensity() < 0.2);
        assert!(noon.sky_color().y > midnight.sky_color().y);
    }
}
