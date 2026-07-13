use std::collections::HashMap;

use crate::mesh::Vertex;
use crate::noise::Noise;
use crate::texture::ATLAS_TILES;

pub const CHUNK_SIZE: usize = 16;
pub const CHUNK_HEIGHT: usize = 256;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Block {
    Air,
    Grass,
    Dirt,
    Stone,
}

impl Block {
    pub fn is_solid(self) -> bool {
        self != Block::Air
    }

    /// Tuile de l'atlas pour une face donnée (indice dans FACES).
    /// L'herbe a un dessus vert, des côtés terre+herbe et un dessous terre.
    fn tile(self, face: usize) -> u32 {
        match self {
            Block::Grass => match face {
                2 => 0, // +Y : dessus d'herbe
                3 => 2, // -Y : terre
                _ => 1, // côtés
            },
            Block::Dirt => 2,
            Block::Stone => 3,
            Block::Air => 0,
        }
    }
}

/// Un chunk : colonne de 16×256×16 blocs stockée à plat.
/// On ne dessine jamais "des cubes" : la grille est transformée en un seul
/// mesh ne contenant que les faces exposées à l'air.
pub struct Chunk {
    blocks: Vec<Block>,
}

/// Les 6 faces d'un bloc : normale, puis les 4 coins en sens trigonométrique
/// (vus de l'extérieur), avec le bloc occupant [0,1]³ à sa coordonnée.
#[rustfmt::skip]
const FACES: [([f32; 3], [[f32; 3]; 4]); 6] = [
    // +X (droite)
    ([1.0, 0.0, 0.0], [[1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]),
    // -X (gauche)
    ([-1.0, 0.0, 0.0], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]]),
    // +Y (dessus)
    ([0.0, 1.0, 0.0], [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
    // -Y (dessous)
    ([0.0, -1.0, 0.0], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    // +Z (avant)
    ([0.0, 0.0, 1.0], [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
    // -Z (arrière)
    ([0.0, 0.0, -1.0], [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
];

const FACE_UVS: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

fn index(x: usize, y: usize, z: usize) -> usize {
    (y * CHUNK_SIZE + z) * CHUNK_SIZE + x
}

/// Hauteur du terrain en coordonnées monde : fBm de Perlin 2D à 4 octaves.
fn terrain_height(noise: &Noise, wx: i32, wz: i32) -> usize {
    let n = noise.fbm2(wx as f32 * 0.012, wz as f32 * 0.012, 4);
    ((28.0 + n * 22.0).max(1.0) as usize).min(CHUNK_HEIGHT - 1)
}

/// Grottes en "gruyère" : on creuse là où un fBm de Perlin 3D dépasse un
/// seuil. Plus le seuil est bas, plus les grottes sont vastes.
fn is_cave(noise: &Noise, wx: i32, wy: i32, wz: i32) -> bool {
    let n = noise.fbm3(
        wx as f32 * 0.055,
        wy as f32 * 0.055,
        wz as f32 * 0.055,
        3,
    );
    n > 0.38
}

impl Chunk {
    /// Génère la colonne de terrain du chunk (cx, cz), en coordonnées chunk.
    /// La couche y=0 n'est jamais creusée (bedrock de fortune).
    pub fn generate(noise: &Noise, cx: i32, cz: i32) -> Self {
        let mut blocks = vec![Block::Air; CHUNK_SIZE * CHUNK_SIZE * CHUNK_HEIGHT];
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let wx = cx * CHUNK_SIZE as i32 + x as i32;
                let wz = cz * CHUNK_SIZE as i32 + z as i32;
                let height = terrain_height(noise, wx, wz);
                for y in 0..=height {
                    if y > 0 && is_cave(noise, wx, y as i32, wz) {
                        continue;
                    }
                    blocks[index(x, y, z)] = if y == height {
                        Block::Grass
                    } else if y + 3 >= height {
                        Block::Dirt
                    } else {
                        Block::Stone
                    };
                }
            }
        }
        Self { blocks }
    }

    pub fn block_local(&self, x: usize, y: usize, z: usize) -> Block {
        self.blocks[index(x, y, z)]
    }

    pub fn set_local(&mut self, x: usize, y: usize, z: usize, block: Block) {
        self.blocks[index(x, y, z)] = block;
    }
}

/// Construit le mesh du chunk (cx, cz) en coordonnées monde. Une face n'est
/// émise que si le bloc voisin est de l'air — y compris à travers les
/// frontières de chunks, d'où le besoin des 4 voisins. L'appelant garantit
/// qu'ils sont présents dans la map ; un voisin absent serait traité comme
/// plein (faces cachées, pas de murs fantômes).
pub fn build_mesh(
    chunks: &HashMap<(i32, i32), Chunk>,
    cx: i32,
    cz: i32,
) -> (Vec<Vertex>, Vec<u32>) {
    let size = CHUNK_SIZE as i32;
    let chunk = &chunks[&(cx, cz)];
    let east = chunks.get(&(cx + 1, cz));
    let west = chunks.get(&(cx - 1, cz));
    let south = chunks.get(&(cx, cz + 1));
    let north = chunks.get(&(cx, cz - 1));

    // Voisin d'un bloc en coordonnées locales, éventuellement à ±1 hors du
    // chunk sur un seul axe.
    let block_at = |x: i32, y: i32, z: i32| -> Block {
        if y < 0 || y >= CHUNK_HEIGHT as i32 {
            return Block::Air;
        }
        let y = y as usize;
        match (x, z) {
            (-1, _) => west.map_or(Block::Stone, |c| {
                c.block_local(CHUNK_SIZE - 1, y, z as usize)
            }),
            (x, _) if x == size => east.map_or(Block::Stone, |c| c.block_local(0, y, z as usize)),
            (_, -1) => north.map_or(Block::Stone, |c| {
                c.block_local(x as usize, y, CHUNK_SIZE - 1)
            }),
            (_, z) if z == size => south.map_or(Block::Stone, |c| c.block_local(x as usize, y, 0)),
            _ => chunk.block_local(x as usize, y, z as usize),
        }
    };

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let tile_width = 1.0 / ATLAS_TILES as f32;
    // Léger retrait des UVs à l'intérieur de la tuile pour éviter que les
    // arrondis flottants fassent déborder l'échantillonnage sur la tuile
    // voisine de l'atlas.
    let inset = 0.001;

    for y in 0..CHUNK_HEIGHT as i32 {
        for z in 0..size {
            for x in 0..size {
                let block = chunk.block_local(x as usize, y as usize, z as usize);
                if !block.is_solid() {
                    continue;
                }
                for (face, (normal, corners)) in FACES.iter().enumerate() {
                    let (nx, ny, nz) = (normal[0] as i32, normal[1] as i32, normal[2] as i32);
                    if block_at(x + nx, y + ny, z + nz).is_solid() {
                        continue;
                    }
                    let tile = block.tile(face) as f32;
                    let base = vertices.len() as u32;
                    for (corner, uv) in corners.iter().zip(FACE_UVS) {
                        let u_local = uv[0].clamp(inset, 1.0 - inset);
                        let v_local = uv[1].clamp(inset, 1.0 - inset);
                        vertices.push(Vertex {
                            position: [
                                (cx * size + x) as f32 + corner[0],
                                y as f32 + corner[1],
                                (cz * size + z) as f32 + corner[2],
                            ],
                            uv: [(tile + u_local) * tile_width, v_local],
                            normal: *normal,
                        });
                    }
                    indices.extend_from_slice(&[
                        base,
                        base + 1,
                        base + 2,
                        base,
                        base + 2,
                        base + 3,
                    ]);
                }
            }
        }
    }

    (vertices, indices)
}
