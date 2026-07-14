use std::sync::Arc;

use crate::light;
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
    Sand,
    Plank,
    /// Bloc lumineux : source de lumière propagée (niveau 15).
    Glow,
    Wood,
    Leaves,
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
            Block::Sand => 4,
            Block::Plank => 5,
            Block::Glow => 6,
            Block::Wood => 7,
            Block::Leaves => 8,
            Block::Air => 0,
        }
    }

    /// Tuile utilisée comme icône dans la barre de sélection.
    pub fn icon_tile(self) -> u32 {
        match self {
            Block::Grass => 1,
            Block::Dirt => 2,
            Block::Stone => 3,
            Block::Sand => 4,
            Block::Plank => 5,
            Block::Glow => 6,
            Block::Wood => 7,
            Block::Leaves => 8,
            Block::Air => 0,
        }
    }

    /// Niveau de lumière émis par le bloc (0 à 15).
    pub fn emission(self) -> u8 {
        match self {
            Block::Glow => 15,
            _ => 0,
        }
    }
}

/// Un chunk : colonne de 16×256×16 blocs stockée à plat.
/// On ne dessine jamais "des cubes" : la grille est transformée en un seul
/// mesh ne contenant que les faces exposées à l'air.
/// `Clone` sert au copy-on-write d'`Arc::make_mut` quand on modifie un bloc
/// pendant qu'un thread de meshing lit encore l'ancienne version.
#[derive(Clone)]
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
    let n = noise.fbm3(wx as f32 * 0.055, wy as f32 * 0.055, wz as f32 * 0.055, 3);
    n > 0.38
}

/// Hash entier d'une colonne monde, pour décider déterministiquement de la
/// présence d'un arbre (indépendant du chunk qui fait le calcul).
fn column_hash(wx: i32, wz: i32, salt: u32) -> u32 {
    let mut h = (wx as u32)
        .wrapping_mul(374_761_393)
        .wrapping_add((wz as u32).wrapping_mul(668_265_263))
        .wrapping_add(salt.wrapping_mul(2_246_822_519));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^ (h >> 16)
}

/// Un arbre pousse-t-il sur cette colonne ? Environ 1 colonne sur 60.
fn has_tree(wx: i32, wz: i32) -> bool {
    column_hash(wx, wz, 7) % 61 == 0
}

impl Chunk {
    /// Génère la colonne de terrain du chunk (cx, cz), en coordonnées chunk.
    /// La couche y=0 n'est jamais creusée (bedrock de fortune).
    pub fn generate(noise: &Noise, cx: i32, cz: i32) -> Self {
        let mut blocks = vec![Block::Air; CHUNK_SIZE * CHUNK_SIZE * CHUNK_HEIGHT];
        let (base_x, base_z) = (cx * CHUNK_SIZE as i32, cz * CHUNK_SIZE as i32);

        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let wx = base_x + x as i32;
                let wz = base_z + z as i32;
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

        let mut chunk = Self { blocks };
        chunk.plant_trees(noise, base_x, base_z);
        chunk
    }

    /// Plante les arbres du chunk. On parcourt aussi une marge de 2 colonnes
    /// autour : un arbre voisin peut faire déborder son feuillage ici, et
    /// chaque chunk doit produire le même arbre indépendamment (déterminisme).
    fn plant_trees(&mut self, noise: &Noise, base_x: i32, base_z: i32) {
        const MARGIN: i32 = 2;
        let size = CHUNK_SIZE as i32;

        // N'écrit que si la cible tombe dans ce chunk. `only_air` protège le
        // tronc et le terrain d'être écrasés par le feuillage d'un voisin.
        fn set(
            blocks: &mut [Block],
            base: (i32, i32),
            wx: i32,
            wy: i32,
            wz: i32,
            b: Block,
            only_air: bool,
        ) {
            let (lx, lz) = (wx - base.0, wz - base.1);
            let size = CHUNK_SIZE as i32;
            if lx < 0 || lx >= size || lz < 0 || lz >= size || wy < 0 || wy >= CHUNK_HEIGHT as i32 {
                return;
            }
            let i = index(lx as usize, wy as usize, lz as usize);
            if !only_air || blocks[i] == Block::Air {
                blocks[i] = b;
            }
        }
        let base = (base_x, base_z);

        for wx in (base_x - MARGIN)..(base_x + size + MARGIN) {
            for wz in (base_z - MARGIN)..(base_z + size + MARGIN) {
                if !has_tree(wx, wz) {
                    continue;
                }
                let surface = terrain_height(noise, wx, wz) as i32;
                // Pas d'arbre au bord d'une grotte affleurante.
                if is_cave(noise, wx, surface, wz) {
                    continue;
                }
                let trunk_h = 4 + (column_hash(wx, wz, 13) % 3) as i32;
                let base_y = surface + 1;

                // Feuillage : deux couronnes larges puis un chapeau étroit.
                for dy in (trunk_h - 2)..=(trunk_h + 1) {
                    let radius: i32 = if dy < trunk_h { 2 } else { 1 };
                    for dx in -radius..=radius {
                        for dz in -radius..=radius {
                            // Coins tronqués pour arrondir la silhouette.
                            if dx.abs() == radius && dz.abs() == radius {
                                continue;
                            }
                            set(
                                &mut self.blocks,
                                base,
                                wx + dx,
                                base_y + dy,
                                wz + dz,
                                Block::Leaves,
                                true,
                            );
                        }
                    }
                }
                // Tronc par-dessus le feuillage.
                for dy in 0..trunk_h {
                    set(
                        &mut self.blocks,
                        base,
                        wx,
                        base_y + dy,
                        wz,
                        Block::Wood,
                        false,
                    );
                }
            }
        }
    }

    pub fn block_local(&self, x: usize, y: usize, z: usize) -> Block {
        self.blocks[index(x, y, z)]
    }

    pub fn set_local(&mut self, x: usize, y: usize, z: usize, block: Block) {
        self.blocks[index(x, y, z)] = block;
    }
}

/// Les 4 chunks voisins d'un chunk à mesher. Des `Arc` (pointeurs partagés,
/// comptés par référence) : le meshing tourne dans des threads de travail et
/// a besoin de lire ces données sans les copier ni les verrouiller.
pub struct ChunkNeighbors {
    pub east: Arc<Chunk>,  // +X
    pub west: Arc<Chunk>,  // -X
    pub south: Arc<Chunk>, // +Z
    pub north: Arc<Chunk>, // -Z
}

/// Construit le mesh du chunk (cx, cz) en coordonnées monde. Une face n'est
/// émise que si le bloc voisin est de l'air — y compris à travers les
/// frontières de chunks, d'où le besoin des 4 voisins.
pub fn build_mesh(
    chunk: &Chunk,
    neighbors: &ChunkNeighbors,
    cx: i32,
    cz: i32,
) -> (Vec<Vertex>, Vec<u32>) {
    let size = CHUNK_SIZE as i32;

    // Voisin d'un bloc en coordonnées locales, éventuellement à ±1 hors du
    // chunk sur un seul axe.
    let block_at = |x: i32, y: i32, z: i32| -> Block {
        if y < 0 || y >= CHUNK_HEIGHT as i32 {
            return Block::Air;
        }
        let y = y as usize;
        match (x, z) {
            (-1, _) => neighbors.west.block_local(CHUNK_SIZE - 1, y, z as usize),
            (x, _) if x == size => neighbors.east.block_local(0, y, z as usize),
            (_, -1) => neighbors.north.block_local(x as usize, y, CHUNK_SIZE - 1),
            (_, z) if z == size => neighbors.south.block_local(x as usize, y, 0),
            _ => chunk.block_local(x as usize, y, z as usize),
        }
    };

    // La lumière propagée (ciel + émission) sur toute la région ; l'occlusion
    // ambiante et la lumière par sommet sont dérivées de cette grille.
    let light = light::compute(chunk, neighbors);

    // Facteur d'assombrissement par niveau d'occlusion (0 = coin bouché).
    const AO_CURVE: [f32; 4] = [0.35, 0.62, 0.82, 1.0];

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
                    let n = [normal[0] as i32, normal[1] as i32, normal[2] as i32];
                    if block_at(x + n[0], y + n[1], z + n[2]).is_solid() {
                        continue;
                    }
                    // La cellule d'air devant la face : c'est elle qui est
                    // éclairée, pas le bloc lui-même.
                    let f = [x + n[0], y + n[1], z + n[2]];

                    let tile = block.tile(face) as f32;
                    let base = vertices.len() as u32;
                    let mut corner_ao = [0u8; 4];

                    for (ci, (corner, uv)) in corners.iter().zip(FACE_UVS).enumerate() {
                        // Direction du coin (±1 par axe), projetée sur le
                        // plan de la face : les deux offsets tangents vers
                        // les cellules qui bordent ce coin.
                        let r = [
                            corner[0] as i32 * 2 - 1,
                            corner[1] as i32 * 2 - 1,
                            corner[2] as i32 * 2 - 1,
                        ];
                        let mut t1 = [0i32; 3];
                        let mut t2 = [0i32; 3];
                        let mut tangents = (0..3).filter(|&a| n[a] == 0);
                        let (a1, a2) = (tangents.next().unwrap(), tangents.next().unwrap());
                        t1[a1] = r[a1];
                        t2[a2] = r[a2];

                        let s1 = [f[0] + t1[0], f[1] + t1[1], f[2] + t1[2]];
                        let s2 = [f[0] + t2[0], f[1] + t2[1], f[2] + t2[2]];
                        let c = [
                            f[0] + t1[0] + t2[0],
                            f[1] + t1[1] + t2[1],
                            f[2] + t1[2] + t2[2],
                        ];

                        // Occlusion ambiante façon voxel : 2 côtés bouchés
                        // = coin totalement occlus, sinon on compte.
                        let (o1, o2, oc) = (
                            light.solid(s1[0], s1[1], s1[2]),
                            light.solid(s2[0], s2[1], s2[2]),
                            light.solid(c[0], c[1], c[2]),
                        );
                        let ao = if o1 && o2 {
                            0
                        } else {
                            3 - (o1 as u8 + o2 as u8 + oc as u8)
                        };
                        corner_ao[ci] = ao;

                        // Lumière lissée : moyenne des 4 cellules d'air qui
                        // touchent ce coin (c'est ce qui donne les dégradés).
                        let (mut sky_sum, mut emit_sum, mut count) = (0u32, 0u32, 0u32);
                        for cell in [f, s1, s2, c] {
                            if light.known(cell[0], cell[1], cell[2])
                                && !light.solid(cell[0], cell[1], cell[2])
                            {
                                sky_sum += light.sky(cell[0], cell[1], cell[2]) as u32;
                                emit_sum += light.emit(cell[0], cell[1], cell[2]) as u32;
                                count += 1;
                            }
                        }
                        let count = count.max(1) as f32;
                        let ao_factor = AO_CURVE[ao as usize];
                        let sky = sky_sum as f32 / count / light::MAX_LIGHT as f32 * ao_factor;
                        let emit = emit_sum as f32 / count / light::MAX_LIGHT as f32 * ao_factor;

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
                            light: [sky, emit],
                        });
                    }

                    // Le quad est coupé selon la diagonale qui suit le
                    // dégradé d'AO, sinon les coins sombres "bavent" le long
                    // d'une seule diagonale (anisotropie).
                    let quad: [u32; 6] =
                        if corner_ao[0] + corner_ao[2] > corner_ao[1] + corner_ao[3] {
                            [0, 1, 2, 0, 2, 3]
                        } else {
                            [1, 2, 3, 1, 3, 0]
                        };
                    indices.extend(quad.iter().map(|i| base + i));
                }
            }
        }
    }

    (vertices, indices)
}
