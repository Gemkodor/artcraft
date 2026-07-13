use glam::{IVec3, Vec3};

use crate::camera::CameraController;
use crate::world::World;

/// Hauteur des yeux au-dessus des pieds (comme Minecraft).
pub const EYE_HEIGHT: f32 = 1.62;
/// Demi-largeur de la hitbox (0.6 bloc de large au total).
const HALF_WIDTH: f32 = 0.3;
const HEIGHT: f32 = 1.8;
const GRAVITY: f32 = 28.0;
const JUMP_SPEED: f32 = 8.6;
const WALK_SPEED: f32 = 4.5;
const FLY_SPEED: f32 = 12.0;
const MAX_FALL_SPEED: f32 = 55.0;
const EPS: f32 = 1e-4;
/// Déplacement maximal par sous-étape de collision : en dessous d'un bloc,
/// pour qu'une grosse frame ne fasse jamais traverser un mur (tunneling).
const SUB_STEP: f32 = 0.4;

pub struct Player {
    /// Position des pieds, au centre de la hitbox en X/Z.
    pub pos: Vec3,
    pub vel: Vec3,
    pub on_ground: bool,
    pub flying: bool,
    spawned: bool,
}

fn overlaps_solid(world: &World, min: Vec3, max: Vec3) -> bool {
    let lo = min.floor().as_ivec3();
    let hi = (max - EPS).floor().as_ivec3();
    for x in lo.x..=hi.x {
        for y in lo.y..=hi.y {
            for z in lo.z..=hi.z {
                if world.block_at(IVec3::new(x, y, z)).is_solid() {
                    return true;
                }
            }
        }
    }
    false
}

impl Player {
    pub fn new(pos: Vec3) -> Self {
        Self {
            pos,
            vel: Vec3::ZERO,
            on_ground: false,
            flying: false,
            spawned: false,
        }
    }

    fn aabb(&self) -> (Vec3, Vec3) {
        (
            Vec3::new(self.pos.x - HALF_WIDTH, self.pos.y, self.pos.z - HALF_WIDTH),
            Vec3::new(
                self.pos.x + HALF_WIDTH,
                self.pos.y + HEIGHT,
                self.pos.z + HALF_WIDTH,
            ),
        )
    }

    /// Le bloc donné chevauche-t-il la hitbox ? (utilisé pour interdire de
    /// poser un bloc dans le joueur)
    pub fn intersects_block(&self, block: IVec3) -> bool {
        let (min, max) = self.aabb();
        let bmin = block.as_vec3();
        let bmax = bmin + Vec3::ONE;
        min.x < bmax.x
            && max.x > bmin.x
            && min.y < bmax.y
            && max.y > bmin.y
            && min.z < bmax.z
            && max.z > bmin.z
    }

    pub fn toggle_fly(&mut self) {
        self.flying = !self.flying;
        self.vel = Vec3::ZERO;
    }

    pub fn update(&mut self, world: &World, ctl: &CameraController, yaw: f32, dt: f32) {
        // Pas de physique tant que le terrain sous le joueur n'existe pas
        // (démarrage, ou zone en cours de génération).
        if !world.is_loaded(self.pos) {
            return;
        }
        if !self.spawned {
            let (x, z) = (self.pos.x.floor() as i32, self.pos.z.floor() as i32);
            if let Some(surface) = world.surface_y(x, z) {
                self.pos.y = surface as f32 + 3.0;
            }
            self.spawned = true;
        }
        // Une frame anormalement longue (chargement, fenêtre déplacée) ne
        // doit pas produire un pas de physique géant.
        let dt = dt.clamp(0.0, 0.05);

        let forward = Vec3::new(yaw.cos(), 0.0, yaw.sin());
        let right = forward.cross(Vec3::Y);
        let mut wish = Vec3::ZERO;
        if ctl.forward {
            wish += forward;
        }
        if ctl.backward {
            wish -= forward;
        }
        if ctl.right {
            wish += right;
        }
        if ctl.left {
            wish -= right;
        }
        let wish = wish.normalize_or_zero();

        if self.flying {
            self.vel = wish * FLY_SPEED;
            if ctl.up {
                self.vel.y += FLY_SPEED;
            }
            if ctl.down {
                self.vel.y -= FLY_SPEED;
            }
        } else {
            self.vel.x = wish.x * WALK_SPEED;
            self.vel.z = wish.z * WALK_SPEED;
            self.vel.y = (self.vel.y - GRAVITY * dt).max(-MAX_FALL_SPEED);
            if ctl.up && self.on_ground {
                self.vel.y = JUMP_SPEED;
            }
        }

        self.on_ground = false;
        let delta = self.vel * dt;
        self.move_axis(world, 1, delta.y);
        self.move_axis(world, 0, delta.x);
        self.move_axis(world, 2, delta.z);
    }

    /// Déplace le joueur sur un axe en résolvant les collisions : au premier
    /// chevauchement, on recolle la hitbox contre la face du bloc heurté et
    /// on annule la vitesse sur cet axe.
    fn move_axis(&mut self, world: &World, axis: usize, delta: f32) {
        if delta == 0.0 {
            return;
        }
        let steps = (delta.abs() / SUB_STEP).ceil().max(1.0) as usize;
        let step = delta / steps as f32;

        for _ in 0..steps {
            match axis {
                0 => self.pos.x += step,
                1 => self.pos.y += step,
                _ => self.pos.z += step,
            }
            let (min, max) = self.aabb();
            if !overlaps_solid(world, min, max) {
                continue;
            }
            // La cellule heurtée est forcément sur le bord avant de la
            // hitbox dans le sens du déplacement (le sous-pas < 1 bloc).
            match axis {
                0 => {
                    self.pos.x = if step > 0.0 {
                        (max.x - EPS).floor() - HALF_WIDTH - EPS
                    } else {
                        min.x.floor() + 1.0 + HALF_WIDTH + EPS
                    };
                    self.vel.x = 0.0;
                }
                1 => {
                    if step > 0.0 {
                        self.pos.y = (max.y - EPS).floor() - HEIGHT - EPS;
                    } else {
                        self.pos.y = min.y.floor() + 1.0 + EPS;
                        self.on_ground = true;
                    }
                    self.vel.y = 0.0;
                }
                _ => {
                    self.pos.z = if step > 0.0 {
                        (max.z - EPS).floor() - HALF_WIDTH - EPS
                    } else {
                        min.z.floor() + 1.0 + HALF_WIDTH + EPS
                    };
                    self.vel.z = 0.0;
                }
            }
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Block;

    fn world_with_terrain() -> World {
        let mut world = World::new();
        world.generate_area(1);
        world
    }

    fn idle_controller() -> CameraController {
        CameraController::new(0.002)
    }

    #[test]
    fn player_falls_and_lands_on_surface() {
        let world = world_with_terrain();
        let mut player = Player::new(Vec3::new(8.5, 120.0, 8.5));
        let ctl = idle_controller();
        for _ in 0..600 {
            player.update(&world, &ctl, 0.0, 1.0 / 60.0);
        }
        assert!(player.on_ground, "le joueur doit finir posé au sol");
        // Les pieds reposent sur un bloc solide, et la tête n'est pas dans
        // un bloc.
        let feet = player.pos;
        let below = IVec3::new(feet.x.floor() as i32, (feet.y - 0.5).floor() as i32, feet.z.floor() as i32);
        assert!(world.block_at(below).is_solid());
        assert!(!world.block_at(feet.floor().as_ivec3()).is_solid());
    }

    #[test]
    fn player_cannot_walk_through_walls() {
        let mut world = world_with_terrain();
        let surface = world.surface_y(8, 8).unwrap();

        // Couloir plat en +X avec un mur au bout.
        for dx in 0..5 {
            world.set_block(IVec3::new(8 + dx, surface, 8), Block::Stone);
            for dy in 1..4 {
                world.set_block(IVec3::new(8 + dx, surface + dy, 8), Block::Air);
            }
        }
        for dy in 1..4 {
            world.set_block(IVec3::new(12, surface + dy, 8), Block::Stone);
        }

        let mut player = Player::new(Vec3::new(8.5, surface as f32 + 1.1, 8.5));
        player.spawned = true; // pas de snap au sommet du terrain
        let mut ctl = idle_controller();
        ctl.forward = true;
        // Marche vers +X (yaw = 0) pendant 4 secondes.
        for _ in 0..240 {
            player.update(&world, &ctl, 0.0, 1.0 / 60.0);
        }
        assert!(
            player.pos.x > 11.0,
            "le joueur doit avoir avancé jusqu'au mur (x = {})",
            player.pos.x
        );
        assert!(
            player.pos.x + HALF_WIDTH <= 12.0 + 0.01,
            "le joueur ne doit pas traverser le mur (x = {})",
            player.pos.x
        );
    }

    #[test]
    fn jump_leaves_the_ground_and_comes_back() {
        let world = world_with_terrain();
        let mut player = Player::new(Vec3::new(8.5, 120.0, 8.5));
        let mut ctl = idle_controller();
        for _ in 0..600 {
            player.update(&world, &ctl, 0.0, 1.0 / 60.0);
        }
        let rest_y = player.pos.y;

        ctl.up = true;
        player.update(&world, &ctl, 0.0, 1.0 / 60.0);
        ctl.up = false;
        let mut peak = rest_y;
        for _ in 0..600 {
            player.update(&world, &ctl, 0.0, 1.0 / 60.0);
            peak = peak.max(player.pos.y);
        }
        assert!(peak > rest_y + 1.0, "le saut doit dépasser 1 bloc");
        assert!(player.on_ground, "et retomber au sol");
        assert!((player.pos.y - rest_y).abs() < 0.01);
    }
}
