//! Éclairage voxel propagé, calculé au moment du meshing (donc dans les
//! threads de travail).
//!
//! Deux canaux, comme dans Minecraft :
//! - **ciel** : 15 pour toute cellule à l'air libre, puis la lumière se
//!   propage dans les grottes en perdant 1 par bloc traversé (BFS) ;
//! - **émission** : les blocs lumineux rayonnent 15, même décroissance.
//!
//! La grille couvre le chunk central et ses 4 voisins directs (48×48×256) :
//! 16 blocs de marge, soit la portée maximale d'une lumière. Les coins en
//! diagonale ne sont pas transmis et sont traités comme opaques — l'erreur
//! est invisible en pratique et évite de transporter 4 chunks de plus.

use std::collections::VecDeque;

use crate::chunk::{CHUNK_HEIGHT, CHUNK_SIZE, Chunk, ChunkNeighbors};

pub const MAX_LIGHT: u8 = 15;

const MARGIN: i32 = CHUNK_SIZE as i32;
const SIZE: i32 = 3 * CHUNK_SIZE as i32;
const HEIGHT: i32 = CHUNK_HEIGHT as i32;

const FLAG_SOLID: u8 = 1;
const FLAG_KNOWN: u8 = 2;

/// Les 6 directions de propagation.
const DIRS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

pub struct LightGrid {
    sky: Vec<u8>,
    emit: Vec<u8>,
    flags: Vec<u8>,
}

/// Index dans les tableaux plats ; rx/rz sont décalés de MARGIN (≥ 0).
fn index(rx: i32, y: i32, rz: i32) -> usize {
    ((y * SIZE + rz) * SIZE + rx) as usize
}

/// Une cellule empaquetée dans un u32 pour la file du BFS (48 < 2⁶).
fn pack(rx: i32, y: i32, rz: i32) -> u32 {
    (rx as u32) | ((rz as u32) << 6) | ((y as u32) << 12)
}

fn unpack(p: u32) -> (i32, i32, i32) {
    ((p & 63) as i32, (p >> 12) as i32, ((p >> 6) & 63) as i32)
}

impl LightGrid {
    fn in_bounds(lx: i32, ly: i32, lz: i32) -> bool {
        ly >= 0
            && ly < HEIGHT
            && lx >= -MARGIN
            && lx < SIZE - MARGIN
            && lz >= -MARGIN
            && lz < SIZE - MARGIN
    }

    /// Lumière du ciel d'une cellule, en coordonnées locales du chunk
    /// central (lx/lz peuvent déborder dans les voisins).
    pub fn sky(&self, lx: i32, ly: i32, lz: i32) -> u8 {
        if ly >= HEIGHT {
            return MAX_LIGHT; // au-dessus du monde : plein ciel
        }
        if !Self::in_bounds(lx, ly, lz) {
            return 0;
        }
        self.sky[index(lx + MARGIN, ly, lz + MARGIN)]
    }

    pub fn emit(&self, lx: i32, ly: i32, lz: i32) -> u8 {
        if !Self::in_bounds(lx, ly, lz) {
            return 0;
        }
        self.emit[index(lx + MARGIN, ly, lz + MARGIN)]
    }

    pub fn solid(&self, lx: i32, ly: i32, lz: i32) -> bool {
        if !Self::in_bounds(lx, ly, lz) {
            return false;
        }
        self.flags[index(lx + MARGIN, ly, lz + MARGIN)] & FLAG_SOLID != 0
    }

    /// Une cellule est "connue" si ses données de blocs ont été transmises
    /// (faux dans les coins diagonaux et sous le monde).
    pub fn known(&self, lx: i32, ly: i32, lz: i32) -> bool {
        if ly >= HEIGHT {
            return true; // le ciel est connu par définition
        }
        if !Self::in_bounds(lx, ly, lz) {
            return false;
        }
        self.flags[index(lx + MARGIN, ly, lz + MARGIN)] & FLAG_KNOWN != 0
    }
}

/// Calcule la grille de lumière pour un chunk et ses voisins.
pub fn compute(center: &Chunk, neighbors: &ChunkNeighbors) -> LightGrid {
    let volume = (SIZE * SIZE * HEIGHT) as usize;
    let mut sky = vec![0u8; volume];
    let mut emit = vec![0u8; volume];
    let mut flags = vec![0u8; volume];

    // Quel chunk fournit la colonne (rx, rz) de la région ?
    let size = CHUNK_SIZE as i32;
    let chunk_for = |rx: i32, rz: i32| -> Option<(&Chunk, usize, usize)> {
        let (lx, lz) = (rx - MARGIN, rz - MARGIN);
        match (lx.div_euclid(size), lz.div_euclid(size)) {
            (0, 0) => Some((center, lx as usize, lz as usize)),
            (-1, 0) => Some((&neighbors.west, (lx + size) as usize, lz as usize)),
            (1, 0) => Some((&neighbors.east, (lx - size) as usize, lz as usize)),
            (0, -1) => Some((&neighbors.north, lx as usize, (lz + size) as usize)),
            (0, 1) => Some((&neighbors.south, lx as usize, (lz - size) as usize)),
            _ => None, // coin diagonal : non transmis, traité comme opaque
        }
    };

    // Passe 1 : drapeaux, hauteur du premier solide, ensoleillement direct
    // des colonnes et graines d'émission.
    let mut emit_queue: VecDeque<u32> = VecDeque::new();
    // -1 = colonne entièrement vide ; HEIGHT = colonne inconnue (opaque).
    let mut first_solid = vec![HEIGHT; (SIZE * SIZE) as usize];

    for rz in 0..SIZE {
        for rx in 0..SIZE {
            let Some((chunk, cx, cz)) = chunk_for(rx, rz) else {
                continue;
            };
            let mut top = -1;
            for y in (0..HEIGHT).rev() {
                let block = chunk.block_local(cx, y as usize, cz);
                let i = index(rx, y, rz);
                flags[i] |= FLAG_KNOWN;
                if block.is_solid() {
                    flags[i] |= FLAG_SOLID;
                    if top < 0 {
                        top = y;
                    }
                }
                let e = block.emission();
                if e > 0 {
                    emit[i] = e;
                    emit_queue.push_back(pack(rx, y, rz));
                }
                if top < 0 {
                    sky[i] = MAX_LIGHT;
                }
            }
            first_solid[(rz * SIZE + rx) as usize] = top;
        }
    }

    // Passe 2 : graines du BFS ciel. Seules les cellules à l'air libre qui
    // bordent une colonne plus haute peuvent donner de la lumière (entrée de
    // grotte, surplomb) — inutile d'enfiler tout le ciel.
    let mut sky_queue: VecDeque<u32> = VecDeque::new();
    for rz in 0..SIZE {
        for rx in 0..SIZE {
            let h = first_solid[(rz * SIZE + rx) as usize];
            if h >= HEIGHT {
                continue; // colonne inconnue
            }
            let mut highest_neighbor = -1;
            for (dx, _, dz) in DIRS.iter().filter(|d| d.1 == 0) {
                let (nx, nz) = (rx + dx, rz + dz);
                if nx < 0 || nx >= SIZE || nz < 0 || nz >= SIZE {
                    continue;
                }
                let nh = first_solid[(nz * SIZE + nx) as usize];
                if nh < HEIGHT {
                    highest_neighbor = highest_neighbor.max(nh);
                }
            }
            for y in (h + 1)..=highest_neighbor.min(HEIGHT - 1) {
                sky_queue.push_back(pack(rx, y, rz));
            }
        }
    }

    relax(&mut sky, &flags, &mut sky_queue);
    relax(&mut emit, &flags, &mut emit_queue);

    LightGrid { sky, emit, flags }
}

/// BFS de propagation : chaque cellule donne (niveau - 1) à ses voisines
/// non solides, tant que ça les améliore. Converge quel que soit l'ordre.
fn relax(light: &mut [u8], flags: &[u8], queue: &mut VecDeque<u32>) {
    while let Some(p) = queue.pop_front() {
        let (rx, y, rz) = unpack(p);
        let level = light[index(rx, y, rz)];
        if level <= 1 {
            continue;
        }
        for (dx, dy, dz) in DIRS {
            let (nx, ny, nz) = (rx + dx, y + dy, rz + dz);
            if nx < 0 || nx >= SIZE || ny < 0 || ny >= HEIGHT || nz < 0 || nz >= SIZE {
                continue;
            }
            let ni = index(nx, ny, nz);
            if flags[ni] & FLAG_KNOWN == 0 || flags[ni] & FLAG_SOLID != 0 {
                continue;
            }
            if light[ni] + 1 < level {
                light[ni] = level - 1;
                queue.push_back(pack(nx, ny, nz));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Block;
    use crate::noise::Noise;
    use std::sync::Arc;

    fn generate_region() -> (Chunk, ChunkNeighbors) {
        let noise = Noise::new(1);
        (
            Chunk::generate(&noise, 0, 0),
            ChunkNeighbors {
                east: Arc::new(Chunk::generate(&noise, 1, 0)),
                west: Arc::new(Chunk::generate(&noise, -1, 0)),
                south: Arc::new(Chunk::generate(&noise, 0, 1)),
                north: Arc::new(Chunk::generate(&noise, 0, -1)),
            },
        )
    }

    #[test]
    fn skylight_is_full_above_terrain() {
        let (center, neighbors) = generate_region();
        let grid = compute(&center, &neighbors);
        // Trouve la surface de la colonne (8, 8) et vérifie le plein ciel.
        let surface = (0..HEIGHT)
            .rev()
            .find(|&y| center.block_local(8, y as usize, 8).is_solid())
            .unwrap();
        assert_eq!(grid.sky(8, surface + 1, 8), MAX_LIGHT);
        assert!(grid.solid(8, surface, 8));
    }

    #[test]
    fn glow_block_lights_its_surroundings() {
        let (mut center, neighbors) = generate_region();
        // Une poche d'air enterrée avec une lampe au fond.
        center.set_local(8, 4, 8, Block::Glow);
        center.set_local(8, 5, 8, Block::Air);
        center.set_local(8, 6, 8, Block::Air);
        let grid = compute(&center, &neighbors);
        assert_eq!(grid.emit(8, 4, 8), 15, "la lampe elle-même");
        assert_eq!(grid.emit(8, 5, 8), 14, "un bloc au-dessus");
        assert_eq!(grid.emit(8, 6, 8), 13, "deux blocs au-dessus");
    }
}
