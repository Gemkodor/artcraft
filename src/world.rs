use std::collections::{HashMap, HashSet};

use glam::{IVec3, Vec3};
use wgpu::util::DeviceExt;

use crate::chunk::{self, Block, CHUNK_HEIGHT, CHUNK_SIZE, Chunk};
use crate::noise::Noise;

/// Rayon en chunks autour de la caméra dans lequel on affiche le terrain.
pub const RENDER_DISTANCE: i32 = 6;
/// Chunks générés (données) par frame au maximum, pour lisser la charge.
const GEN_BUDGET: usize = 8;
/// Meshes construits par frame au maximum — c'est l'opération coûteuse.
const MESH_BUDGET: usize = 2;

pub struct ChunkMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
}

/// Résultat d'un raycast : le bloc touché et la normale de la face d'entrée
/// (le bloc adjacent côté caméra est `block + normal`).
pub struct RayHit {
    pub block: IVec3,
    pub normal: IVec3,
}

/// L'ensemble des chunks chargés. Les données de blocs sont générées dans un
/// rayon RENDER_DISTANCE+1 pour que le meshing d'un chunk puisse toujours
/// consulter ses 4 voisins.
pub struct World {
    noise: Noise,
    chunks: HashMap<(i32, i32), Chunk>,
    meshes: HashMap<(i32, i32), ChunkMesh>,
    /// Chunks dont le mesh doit être reconstruit (bloc modifié).
    dirty: HashSet<(i32, i32)>,
}

impl World {
    pub fn new() -> Self {
        Self {
            noise: Noise::new(1337),
            chunks: HashMap::new(),
            meshes: HashMap::new(),
            dirty: HashSet::new(),
        }
    }

    pub fn meshes(&self) -> impl Iterator<Item = &ChunkMesh> {
        self.meshes.values()
    }

    fn split_coord(pos: IVec3) -> ((i32, i32), (usize, usize, usize)) {
        let size = CHUNK_SIZE as i32;
        let (cx, cz) = (pos.x.div_euclid(size), pos.z.div_euclid(size));
        let local = (
            pos.x.rem_euclid(size) as usize,
            pos.y as usize,
            pos.z.rem_euclid(size) as usize,
        );
        ((cx, cz), local)
    }

    /// Bloc en coordonnées monde ; l'extérieur du monde chargé est de l'air.
    pub fn block_at(&self, pos: IVec3) -> Block {
        if pos.y < 0 || pos.y >= CHUNK_HEIGHT as i32 {
            return Block::Air;
        }
        let (coord, (lx, ly, lz)) = Self::split_coord(pos);
        self.chunks
            .get(&coord)
            .map_or(Block::Air, |c| c.block_local(lx, ly, lz))
    }

    /// Modifie un bloc et marque le chunk (et les voisins si le bloc est en
    /// bordure) pour re-meshing au prochain update.
    pub fn set_block(&mut self, pos: IVec3, block: Block) {
        if pos.y < 0 || pos.y >= CHUNK_HEIGHT as i32 {
            return;
        }
        let (coord, (lx, ly, lz)) = Self::split_coord(pos);
        let Some(chunk) = self.chunks.get_mut(&coord) else {
            return;
        };
        chunk.set_local(lx, ly, lz, block);

        self.dirty.insert(coord);
        if lx == 0 {
            self.dirty.insert((coord.0 - 1, coord.1));
        }
        if lx == CHUNK_SIZE - 1 {
            self.dirty.insert((coord.0 + 1, coord.1));
        }
        if lz == 0 {
            self.dirty.insert((coord.0, coord.1 - 1));
        }
        if lz == CHUNK_SIZE - 1 {
            self.dirty.insert((coord.0, coord.1 + 1));
        }
    }

    /// Parcours voxel par voxel du rayon (algorithme DDA d'Amanatides & Woo) :
    /// on saute de frontière de bloc en frontière de bloc, dans l'ordre exact
    /// de traversée, jusqu'au premier bloc solide.
    pub fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<RayHit> {
        let dir = dir.normalize_or_zero();
        if dir == Vec3::ZERO {
            return None;
        }

        let mut block = origin.floor().as_ivec3();
        let step = IVec3::new(
            if dir.x > 0.0 { 1 } else { -1 },
            if dir.y > 0.0 { 1 } else { -1 },
            if dir.z > 0.0 { 1 } else { -1 },
        );
        // Distance le long du rayon pour traverser un bloc entier sur chaque
        // axe, et distance jusqu'à la première frontière.
        let t_delta = dir.abs().recip();
        let boundary = |o: f32, d: f32, b: i32| -> f32 {
            if d > 0.0 {
                (b as f32 + 1.0 - o) / d
            } else if d < 0.0 {
                (b as f32 - o) / d
            } else {
                f32::INFINITY
            }
        };
        let mut t_max = Vec3::new(
            boundary(origin.x, dir.x, block.x),
            boundary(origin.y, dir.y, block.y),
            boundary(origin.z, dir.z, block.z),
        );

        loop {
            let normal;
            let t;
            if t_max.x <= t_max.y && t_max.x <= t_max.z {
                block.x += step.x;
                t = t_max.x;
                t_max.x += t_delta.x;
                normal = IVec3::new(-step.x, 0, 0);
            } else if t_max.y <= t_max.z {
                block.y += step.y;
                t = t_max.y;
                t_max.y += t_delta.y;
                normal = IVec3::new(0, -step.y, 0);
            } else {
                block.z += step.z;
                t = t_max.z;
                t_max.z += t_delta.z;
                normal = IVec3::new(0, 0, -step.z);
            }
            if t > max_dist {
                return None;
            }
            if self.block_at(block).is_solid() {
                return Some(RayHit { block, normal });
            }
        }
    }

    /// Charge/décharge les chunks en fonction de la position de la caméra.
    /// Appelé à chaque frame ; les budgets étalent le travail pour éviter
    /// les à-coups.
    pub fn update(&mut self, device: &wgpu::Device, camera_pos: Vec3) {
        let ccx = (camera_pos.x.floor() as i32).div_euclid(CHUNK_SIZE as i32);
        let ccz = (camera_pos.z.floor() as i32).div_euclid(CHUNK_SIZE as i32);
        let data_radius = RENDER_DISTANCE + 1;

        // Re-meshing immédiat des chunks modifiés (hors budget : il faut que
        // casser un bloc soit visible à la frame suivante, sans flicker).
        let dirty: Vec<(i32, i32)> = self.dirty.drain().collect();
        for coord in dirty {
            if self.meshes.contains_key(&coord) {
                self.upload_mesh(device, coord);
            }
        }

        // Déchargement (avec une marge d'hystérésis pour ne pas re-générer
        // en boucle à la frontière).
        self.chunks.retain(|&(x, z), _| {
            (x - ccx).abs() <= data_radius + 1 && (z - ccz).abs() <= data_radius + 1
        });
        self.meshes.retain(|&(x, z), _| {
            (x - ccx).abs() <= RENDER_DISTANCE && (z - ccz).abs() <= RENDER_DISTANCE
        });

        // Génération des données manquantes, la plus proche d'abord.
        let mut missing: Vec<(i32, i32)> = Vec::new();
        for x in (ccx - data_radius)..=(ccx + data_radius) {
            for z in (ccz - data_radius)..=(ccz + data_radius) {
                if !self.chunks.contains_key(&(x, z)) {
                    missing.push((x, z));
                }
            }
        }
        missing.sort_by_key(|&(x, z)| (x - ccx).pow(2) + (z - ccz).pow(2));
        for &(x, z) in missing.iter().take(GEN_BUDGET) {
            self.chunks.insert((x, z), Chunk::generate(&self.noise, x, z));
        }

        // Meshing des chunks dont les données et les 4 voisins sont prêts.
        let mut to_mesh: Vec<(i32, i32)> = Vec::new();
        for x in (ccx - RENDER_DISTANCE)..=(ccx + RENDER_DISTANCE) {
            for z in (ccz - RENDER_DISTANCE)..=(ccz + RENDER_DISTANCE) {
                let ready = !self.meshes.contains_key(&(x, z))
                    && self.chunks.contains_key(&(x, z))
                    && self.chunks.contains_key(&(x + 1, z))
                    && self.chunks.contains_key(&(x - 1, z))
                    && self.chunks.contains_key(&(x, z + 1))
                    && self.chunks.contains_key(&(x, z - 1));
                if ready {
                    to_mesh.push((x, z));
                }
            }
        }
        to_mesh.sort_by_key(|&(x, z)| (x - ccx).pow(2) + (z - ccz).pow(2));
        for &coord in to_mesh.iter().take(MESH_BUDGET) {
            self.upload_mesh(device, coord);
        }
    }

    fn upload_mesh(&mut self, device: &wgpu::Device, coord: (i32, i32)) {
        let (vertices, indices) = chunk::build_mesh(&self.chunks, coord.0, coord.1);
        if indices.is_empty() {
            self.meshes.remove(&coord);
            return;
        }
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk_vertex_buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk_index_buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.meshes.insert(
            coord,
            ChunkMesh {
                vertex_buffer,
                index_buffer,
                num_indices: indices.len() as u32,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_world() -> World {
        let mut world = World::new();
        for x in -1..=1 {
            for z in -1..=1 {
                world
                    .chunks
                    .insert((x, z), Chunk::generate(&world.noise, x, z));
            }
        }
        world
    }

    #[test]
    fn raycast_hits_terrain_from_above() {
        let world = test_world();
        let hit = world
            .raycast(Vec3::new(8.5, 200.0, 8.5), Vec3::NEG_Y, 300.0)
            .expect("le rayon vertical doit toucher le sol");
        assert_eq!(hit.normal, IVec3::Y, "on doit toucher une face du dessus");
        assert!(world.block_at(hit.block).is_solid());
        assert!(!world.block_at(hit.block + hit.normal).is_solid());
    }

    #[test]
    fn set_block_breaks_and_marks_dirty() {
        let mut world = test_world();
        let hit = world
            .raycast(Vec3::new(8.5, 200.0, 8.5), Vec3::NEG_Y, 300.0)
            .unwrap();
        world.set_block(hit.block, Block::Air);
        assert_eq!(world.block_at(hit.block), Block::Air);
        assert!(world.dirty.contains(&(0, 0)));
    }

    #[test]
    fn raycast_misses_in_the_sky() {
        let world = test_world();
        assert!(
            world
                .raycast(Vec3::new(8.0, 200.0, 8.0), Vec3::Y, 100.0)
                .is_none()
        );
    }
}
