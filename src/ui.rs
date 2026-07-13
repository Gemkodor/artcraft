//! Géométrie de l'interface 2D : viseur et hotbar. Tout est exprimé en
//! pixels puis converti en NDC (coordonnées d'écran normalisées [-1, 1]),
//! donc indépendant de la résolution.

use crate::chunk::Block;
use crate::texture::ATLAS_TILES;

/// Les blocs plaçables, dans l'ordre de la barre de sélection (touches 1-N).
pub const HOTBAR: [Block; 5] = [
    Block::Grass,
    Block::Dirt,
    Block::Stone,
    Block::Sand,
    Block::Plank,
];

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
pub fn hotbar_vertices(width: u32, height: u32, selected: usize) -> Vec<UiVertex> {
    let (w, h) = (width as f32, height as f32);
    let (slot, gap, margin) = (46.0, 5.0, 14.0);
    let n = HOTBAR.len() as f32;
    let x0 = (w - (n * slot + (n - 1.0) * gap)) / 2.0;

    let mut verts = Vec::with_capacity(HOTBAR_VERTEX_COUNT as usize);
    let mut quad = |x: f32, y: f32, qw: f32, qh: f32, tile: Option<u32>, color: [f32; 4]| {
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
    };

    for (i, block) in HOTBAR.iter().enumerate() {
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
    verts
}
