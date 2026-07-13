//! Le monde : cycle de vie des chunks (génération, meshing, déchargement),
//! accès aux blocs en coordonnées monde, raycast et modification de blocs.
//!
//! La génération et le meshing tournent dans un pool de threads (`worker`).
//! Le thread principal se contente d'envoyer des jobs, d'appliquer les
//! résultats prêts et d'uploader les buffers GPU — des opérations courtes,
//! donc pas de micro-saccades pendant l'exploration.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use glam::{IVec3, Vec3};
use wgpu::util::DeviceExt;

use crate::chunk::{self, Block, CHUNK_HEIGHT, CHUNK_SIZE, Chunk, ChunkNeighbors};
use crate::noise::Noise;
use crate::worker::{Job, JobResult, WorkerPool};

/// Rayon en chunks autour de la caméra dans lequel on affiche le terrain.
pub const RENDER_DISTANCE: i32 = 6;

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
    /// Données de blocs. `Arc` pour pouvoir les prêter aux threads de
    /// meshing sans copie ; `Arc::make_mut` fait un copy-on-write à la
    /// modification si un thread lit encore l'ancienne version.
    chunks: HashMap<(i32, i32), Arc<Chunk>>,
    meshes: HashMap<(i32, i32), ChunkMesh>,
    /// Version incrémentée à chaque modification de bloc d'un chunk : un
    /// mesh calculé pour une version plus ancienne est jeté à l'arrivée.
    versions: HashMap<(i32, i32), u64>,
    /// Chunks à re-mesher immédiatement (bloc modifié : le joueur doit voir
    /// le changement à la frame suivante).
    dirty_now: HashSet<(i32, i32)>,
    /// Chunks à re-mesher dès que possible mais sans bloquer la frame
    /// (voisins d'un bloc modifié).
    dirty_async: HashSet<(i32, i32)>,
    pending_gen: HashSet<(i32, i32)>,
    pending_mesh: HashSet<(i32, i32)>,
    workers: WorkerPool,
}

impl World {
    pub fn new() -> Self {
        let noise = Noise::new(1337);
        Self {
            noise,
            chunks: HashMap::new(),
            meshes: HashMap::new(),
            versions: HashMap::new(),
            dirty_now: HashSet::new(),
            dirty_async: HashSet::new(),
            pending_gen: HashSet::new(),
            pending_mesh: HashSet::new(),
            workers: WorkerPool::new(noise),
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

    /// Le chunk contenant cette position a-t-il ses données générées ?
    pub fn is_loaded(&self, pos: Vec3) -> bool {
        let size = CHUNK_SIZE as i32;
        let cx = (pos.x.floor() as i32).div_euclid(size);
        let cz = (pos.z.floor() as i32).div_euclid(size);
        self.chunks.contains_key(&(cx, cz))
    }

    /// Plus haut bloc solide de la colonne (x, z), s'il existe.
    pub fn surface_y(&self, x: i32, z: i32) -> Option<i32> {
        (0..CHUNK_HEIGHT as i32)
            .rev()
            .find(|&y| self.block_at(IVec3::new(x, y, z)).is_solid())
    }

    #[cfg(test)]
    pub fn generate_area(&mut self, radius: i32) {
        for x in -radius..=radius {
            for z in -radius..=radius {
                self.chunks
                    .insert((x, z), Arc::new(Chunk::generate(&self.noise, x, z)));
            }
        }
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

    /// Modifie un bloc. Le chunk touché sera re-meshé immédiatement, ses
    /// 4 voisins de manière asynchrone (l'éclairage et les faces de bordure
    /// peuvent déborder sur eux).
    pub fn set_block(&mut self, pos: IVec3, block: Block) {
        if pos.y < 0 || pos.y >= CHUNK_HEIGHT as i32 {
            return;
        }
        let (coord, (lx, ly, lz)) = Self::split_coord(pos);
        let Some(chunk) = self.chunks.get_mut(&coord) else {
            return;
        };
        Arc::make_mut(chunk).set_local(lx, ly, lz, block);
        *self.versions.entry(coord).or_insert(0) += 1;

        self.dirty_now.insert(coord);
        for neighbor in [
            (coord.0 + 1, coord.1),
            (coord.0 - 1, coord.1),
            (coord.0, coord.1 + 1),
            (coord.0, coord.1 - 1),
        ] {
            self.dirty_async.insert(neighbor);
        }
    }

    fn version(&self, coord: (i32, i32)) -> u64 {
        self.versions.get(&coord).copied().unwrap_or(0)
    }

    /// Les 4 voisins d'un chunk, si tous sont générés.
    fn neighbors_of(&self, coord: (i32, i32)) -> Option<ChunkNeighbors> {
        Some(ChunkNeighbors {
            east: Arc::clone(self.chunks.get(&(coord.0 + 1, coord.1))?),
            west: Arc::clone(self.chunks.get(&(coord.0 - 1, coord.1))?),
            south: Arc::clone(self.chunks.get(&(coord.0, coord.1 + 1))?),
            north: Arc::clone(self.chunks.get(&(coord.0, coord.1 - 1))?),
        })
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

    /// Fait vivre le monde autour de la caméra. Appelé à chaque frame ; tout
    /// le travail lourd part dans les workers, seuls les uploads GPU et les
    /// re-mesh urgents (bloc modifié) restent ici.
    pub fn update(&mut self, device: &wgpu::Device, camera_pos: Vec3) {
        let ccx = (camera_pos.x.floor() as i32).div_euclid(CHUNK_SIZE as i32);
        let ccz = (camera_pos.z.floor() as i32).div_euclid(CHUNK_SIZE as i32);
        let data_radius = RENDER_DISTANCE + 1;

        // 1. Appliquer les résultats des workers.
        for result in self.workers.drain_results() {
            match result {
                JobResult::Generated { coord, chunk } => {
                    self.pending_gen.remove(&coord);
                    let in_range = (coord.0 - ccx).abs() <= data_radius + 1
                        && (coord.1 - ccz).abs() <= data_radius + 1;
                    if in_range {
                        self.chunks.insert(coord, Arc::new(chunk));
                    }
                }
                JobResult::Meshed {
                    coord,
                    version,
                    vertices,
                    indices,
                } => {
                    self.pending_mesh.remove(&coord);
                    let in_range = (coord.0 - ccx).abs() <= RENDER_DISTANCE
                        && (coord.1 - ccz).abs() <= RENDER_DISTANCE;
                    // Un résultat calculé avant une modification de bloc est
                    // périmé : un job plus récent (ou un re-mesh immédiat)
                    // fournit ou a déjà fourni la bonne version.
                    if in_range && version == self.version(coord) {
                        self.store_mesh(device, coord, vertices, indices);
                    }
                }
            }
        }

        // 2. Re-mesh immédiat des chunks modifiés ce frame (un seul par clic
        // en pratique : le coût reste imperceptible).
        let dirty: Vec<(i32, i32)> = self.dirty_now.drain().collect();
        for coord in dirty {
            if !self.meshes.contains_key(&coord) {
                continue;
            }
            if let (Some(chunk), Some(neighbors)) =
                (self.chunks.get(&coord), self.neighbors_of(coord))
            {
                let (vertices, indices) = chunk::build_mesh(chunk, &neighbors, coord.0, coord.1);
                self.store_mesh(device, coord, vertices, indices);
            }
        }

        // 3. Re-mesh asynchrone des voisins de blocs modifiés.
        let dirty: Vec<(i32, i32)> = self.dirty_async.drain().collect();
        for coord in dirty {
            if self.meshes.contains_key(&coord) {
                self.submit_mesh_job(coord);
            }
        }

        // 4. Déchargement (marge d'hystérésis pour ne pas re-générer en
        // boucle à la frontière).
        self.chunks.retain(|&(x, z), _| {
            (x - ccx).abs() <= data_radius + 1 && (z - ccz).abs() <= data_radius + 1
        });
        self.meshes.retain(|&(x, z), _| {
            (x - ccx).abs() <= RENDER_DISTANCE && (z - ccz).abs() <= RENDER_DISTANCE
        });
        let chunks = &self.chunks;
        self.versions.retain(|coord, _| chunks.contains_key(coord));

        // 5. Demander la génération des données manquantes, la plus proche
        // d'abord (l'ordre d'envoi est l'ordre de traitement des workers).
        let mut missing: Vec<(i32, i32)> = Vec::new();
        for x in (ccx - data_radius)..=(ccx + data_radius) {
            for z in (ccz - data_radius)..=(ccz + data_radius) {
                let coord = (x, z);
                if !self.chunks.contains_key(&coord) && !self.pending_gen.contains(&coord) {
                    missing.push(coord);
                }
            }
        }
        missing.sort_by_key(|&(x, z)| (x - ccx).pow(2) + (z - ccz).pow(2));
        for coord in missing {
            self.pending_gen.insert(coord);
            self.workers.submit(Job::Generate { coord });
        }

        // 6. Demander le meshing des chunks prêts (données + 4 voisins).
        let mut to_mesh: Vec<(i32, i32)> = Vec::new();
        for x in (ccx - RENDER_DISTANCE)..=(ccx + RENDER_DISTANCE) {
            for z in (ccz - RENDER_DISTANCE)..=(ccz + RENDER_DISTANCE) {
                let coord = (x, z);
                if !self.meshes.contains_key(&coord) && !self.pending_mesh.contains(&coord) {
                    to_mesh.push(coord);
                }
            }
        }
        to_mesh.sort_by_key(|&(x, z)| (x - ccx).pow(2) + (z - ccz).pow(2));
        for coord in to_mesh {
            self.submit_mesh_job(coord);
        }
    }

    fn submit_mesh_job(&mut self, coord: (i32, i32)) {
        let (Some(center), Some(neighbors)) = (self.chunks.get(&coord), self.neighbors_of(coord))
        else {
            return;
        };
        self.pending_mesh.insert(coord);
        self.workers.submit(Job::Mesh {
            coord,
            version: self.versions.get(&coord).copied().unwrap_or(0),
            center: Arc::clone(center),
            neighbors,
        });
    }

    /// Upload d'un mesh vers le GPU — rapide, reste sur le thread principal.
    fn store_mesh(
        &mut self,
        device: &wgpu::Device,
        coord: (i32, i32),
        vertices: Vec<crate::mesh::Vertex>,
        indices: Vec<u32>,
    ) {
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
        world.generate_area(1);
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
        assert!(world.dirty_now.contains(&(0, 0)));
        assert_eq!(world.version((0, 0)), 1);
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
