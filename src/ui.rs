//! Géométrie de l'interface 2D : viseur et hotbar. Tout est exprimé en
//! pixels puis converti en NDC (coordonnées d'écran normalisées [-1, 1]),
//! donc indépendant de la résolution.

use crate::chunk::Block;
use crate::texture::ATLAS_TILES;

/// Contenu initial de la barre de sélection (touches 1-N). Elle est ensuite
/// modifiable : l'inventaire (touche E) assigne un bloc au slot sélectionné.
pub const HOTBAR: [Block; 8] = [
    Block::Grass,
    Block::Dirt,
    Block::Stone,
    Block::Sand,
    Block::Plank,
    Block::Glow,
    Block::Wood,
    Block::Leaves,
];

/// Tous les blocs plaçables, dans l'ordre d'affichage de l'inventaire.
pub const ALL_BLOCKS: [Block; 31] = [
    Block::Grass,
    Block::Dirt,
    Block::Stone,
    Block::Sand,
    Block::Plank,
    Block::Glow,
    Block::Wood,
    Block::Leaves,
    Block::Cobble,
    Block::MossyCobble,
    Block::Brick,
    Block::StoneBrick,
    Block::Sandstone,
    Block::Snow,
    Block::Ice,
    Block::Obsidian,
    Block::Gravel,
    Block::CoalBlock,
    Block::SteelBlock,
    Block::GoldBlock,
    Block::DiamondBlock,
    Block::Bookshelf,
    Block::DesertSand,
    Block::DesertStone,
    Block::SnowyGrass,
    Block::JungleLitter,
    Block::Cactus,
    Block::PineWood,
    Block::PineNeedles,
    Block::JungleWood,
    Block::JungleLeaves,
];

/// Grille de l'inventaire : dimensions et position, partagées entre le
/// rendu et la détection de clic pour qu'ils ne divergent jamais.
const INV_COLS: usize = 8;
const INV_CELL: f32 = 56.0;
const INV_GAP: f32 = 6.0;

fn inv_rows() -> usize {
    ALL_BLOCKS.len().div_ceil(INV_COLS)
}

/// Rectangle (x, y, taille) de la cellule `i`, en pixels depuis le HAUT de
/// la fenêtre (le repère de la souris).
fn inventory_cell_rect(width: u32, height: u32, i: usize) -> (f32, f32, f32) {
    let grid_w = INV_COLS as f32 * (INV_CELL + INV_GAP) - INV_GAP;
    let grid_h = inv_rows() as f32 * (INV_CELL + INV_GAP) - INV_GAP;
    let x0 = (width as f32 - grid_w) / 2.0;
    let y0 = (height as f32 - grid_h) / 2.0;
    let (col, row) = (i % INV_COLS, i / INV_COLS);
    (
        x0 + col as f32 * (INV_CELL + INV_GAP),
        y0 + row as f32 * (INV_CELL + INV_GAP),
        INV_CELL,
    )
}

/// Bloc sous le curseur (coordonnées souris, origine en haut à gauche).
pub fn inventory_block_at(width: u32, height: u32, cursor: (f32, f32)) -> Option<Block> {
    for (i, block) in ALL_BLOCKS.iter().enumerate() {
        let (x, y, size) = inventory_cell_rect(width, height, i);
        if cursor.0 >= x && cursor.0 < x + size && cursor.1 >= y && cursor.1 < y + size {
            return Some(*block);
        }
    }
    None
}

/// Sommets de l'inventaire : un panneau sombre plein écran (qui signale
/// aussi que le jeu est "en pause" de visée), puis une cellule + icône par
/// bloc.
pub fn inventory_vertices(width: u32, height: u32) -> Vec<UiVertex> {
    let mut verts = Vec::new();
    let mut quad = quad_emitter(width, height, &mut verts);

    // Voile sombre sur tout l'écran.
    quad(
        0.0,
        0.0,
        width as f32,
        height as f32,
        None,
        [0.0, 0.0, 0.0, 0.45],
    );

    for (i, block) in ALL_BLOCKS.iter().enumerate() {
        let (x, y_top, size) = inventory_cell_rect(width, height, i);
        // Conversion repère souris (origine haut) → repère UI (origine bas).
        let y = height as f32 - y_top - size;
        quad(x, y, size, size, None, [0.10, 0.10, 0.12, 0.85]);
        quad(
            x + 5.0,
            y + 5.0,
            size - 10.0,
            size - 10.0,
            Some(block.icon_tile()),
            [1.0, 1.0, 1.0, 1.0],
        );
    }
    // Rend l'emprunt de `verts` par l'émetteur explicite avant de le
    // retourner (le type opaque pourrait avoir un destructeur).
    drop(quad);
    verts
}

/// Nombre de sommets produits par `inventory_vertices` (constant).
pub const INVENTORY_VERTEX_COUNT: u32 = ((1 + 2 * ALL_BLOCKS.len()) * 6) as u32;

/// Sommet UI : position en NDC, UV dans l'atlas (négatif = couleur unie),
/// couleur RGBA (sert aussi de teinte pour les icônes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

/// Les 12 sommets (2 quads) du viseur, en NDC, pour une fenêtre donnée.
pub fn crosshair_vertices(width: u32, height: u32) -> [[f32; 2]; 12] {
    // Demi-longueur et demi-épaisseur des branches, en pixels.
    let (len, thick) = (9.0, 1.5);
    let (lx, tx) = (len / width as f32 * 2.0, thick / width as f32 * 2.0);
    let (ly, ty) = (len / height as f32 * 2.0, thick / height as f32 * 2.0);
    [
        // Branche horizontale.
        [-lx, -ty],
        [lx, -ty],
        [lx, ty],
        [-lx, -ty],
        [lx, ty],
        [-lx, ty],
        // Branche verticale.
        [-tx, -ly],
        [tx, -ly],
        [tx, ly],
        [-tx, -ly],
        [tx, ly],
        [-tx, ly],
    ]
}

/// Nombre de sommets produits par `hotbar_vertices` (constant : il y a
/// toujours exactement un slot sélectionné).
pub const HOTBAR_VERTEX_COUNT: u32 = (HOTBAR.len() * 12 + 6) as u32;

/// Sommets de la hotbar : par slot, un cadre blanc si sélectionné, un fond
/// sombre, et l'icône du bloc (tuile de l'atlas).
pub fn hotbar_vertices(
    width: u32,
    height: u32,
    selected: usize,
    hotbar: &[Block],
) -> Vec<UiVertex> {
    let (slot, gap, margin) = (46.0, 5.0, 14.0);
    let n = hotbar.len() as f32;
    let x0 = (width as f32 - (n * slot + (n - 1.0) * gap)) / 2.0;

    let mut verts = Vec::with_capacity(HOTBAR_VERTEX_COUNT as usize);
    let mut quad = quad_emitter(width, height, &mut verts);

    for (i, block) in hotbar.iter().enumerate() {
        let x = x0 + i as f32 * (slot + gap);
        if i == selected {
            quad(
                x - 3.0,
                margin - 3.0,
                slot + 6.0,
                slot + 6.0,
                None,
                [0.95, 0.95, 0.95, 0.9],
            );
        }
        quad(x, margin, slot, slot, None, [0.05, 0.05, 0.05, 0.62]);
        quad(
            x + 5.0,
            margin + 5.0,
            slot - 10.0,
            slot - 10.0,
            Some(block.icon_tile()),
            [1.0, 1.0, 1.0, 1.0],
        );
    }
    drop(quad);
    verts
}

/// Fabrique un émetteur de quads UI : rectangle en pixels (origine en bas à
/// gauche) → 6 sommets NDC dans `verts`, en couleur unie (`tile: None`) ou
/// texturé depuis une tuile de l'atlas.
fn quad_emitter<'a>(
    width: u32,
    height: u32,
    verts: &'a mut Vec<UiVertex>,
) -> impl FnMut(f32, f32, f32, f32, Option<u32>, [f32; 4]) + 'a {
    let (w, h) = (width as f32, height as f32);
    move |x: f32, y: f32, qw: f32, qh: f32, tile: Option<u32>, color: [f32; 4]| {
        let (u0, v0, u1, v1) = match tile {
            Some(t) => {
                let tw = 1.0 / ATLAS_TILES as f32;
                let inset = 0.02;
                (
                    (t as f32 + inset) * tw,
                    inset,
                    (t as f32 + 1.0 - inset) * tw,
                    1.0 - inset,
                )
            }
            None => (-1.0, -1.0, -1.0, -1.0),
        };
        let (xa, ya) = (x / w * 2.0 - 1.0, y / h * 2.0 - 1.0);
        let (xb, yb) = ((x + qw) / w * 2.0 - 1.0, (y + qh) / h * 2.0 - 1.0);
        // v0 est le haut de la tuile, donc associé au bord haut du quad (yb).
        let corners = [
            ([xa, ya], [u0, v1]),
            ([xb, ya], [u1, v1]),
            ([xb, yb], [u1, v0]),
            ([xa, ya], [u0, v1]),
            ([xb, yb], [u1, v0]),
            ([xa, yb], [u0, v0]),
        ];
        for (position, uv) in corners {
            verts.push(UiVertex {
                position,
                uv,
                color,
            });
        }
    }
}
