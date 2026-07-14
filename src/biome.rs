//! Biomes : chaque colonne du monde a un climat, déterminé par deux bruits
//! de Perlin basse fréquence (température et humidité), qui choisit les
//! blocs de surface et la végétation. Même seed → mêmes biomes, et chaque
//! chunk peut calculer le biome de n'importe quelle colonne sans dépendre
//! de ses voisins.

use crate::chunk::Block;
use crate::noise::Noise;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Biome {
    Plains,
    Desert,
    Taiga,
    Jungle,
}

/// Fréquence des bruits climatiques : ~1/0.005 = des biomes de l'ordre de
/// 200 blocs de large.
const BIOME_SCALE: f32 = 0.005;

pub fn biome_at(noise: &Noise, wx: i32, wz: i32) -> Biome {
    // Deux échantillons décorrélés du même bruit, par grand décalage de
    // coordonnées (régions très éloignées du champ = indépendantes).
    let temperature = noise.fbm2(
        wx as f32 * BIOME_SCALE + 20_000.0,
        wz as f32 * BIOME_SCALE + 20_000.0,
        3,
    );
    let humidity = noise.fbm2(
        wx as f32 * BIOME_SCALE - 20_000.0,
        wz as f32 * BIOME_SCALE - 20_000.0,
        3,
    );

    // Le fBm se concentre autour de 0 : des seuils trop larges rendraient
    // les biomes extrêmes introuvables en pratique.
    if temperature < -0.15 {
        Biome::Taiga
    } else if temperature > 0.12 {
        if humidity < 0.0 {
            Biome::Desert
        } else {
            Biome::Jungle
        }
    } else {
        Biome::Plains
    }
}

impl Biome {
    /// Le bloc qui affleure en surface.
    pub fn surface_block(self) -> Block {
        match self {
            Biome::Plains => Block::Grass,
            Biome::Desert => Block::DesertSand,
            Biome::Taiga => Block::SnowyGrass,
            Biome::Jungle => Block::JungleLitter,
        }
    }

    /// Les 3 couches sous la surface.
    pub fn subsurface_block(self) -> Block {
        match self {
            Biome::Desert => Block::DesertSand,
            _ => Block::Dirt,
        }
    }

    /// La roche profonde.
    pub fn deep_block(self) -> Block {
        match self {
            Biome::Desert => Block::DesertStone,
            _ => Block::Stone,
        }
    }

    /// 1 colonne sur N porte un arbre (ou un cactus).
    pub fn vegetation_density(self) -> u32 {
        match self {
            Biome::Plains => 61,
            Biome::Desert => 89,
            Biome::Taiga => 71,
            Biome::Jungle => 31,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn biomes_are_deterministic_and_varied() {
        let noise = Noise::new(1337);
        // Déterminisme : deux appels identiques donnent le même biome.
        assert_eq!(biome_at(&noise, 123, -456), biome_at(&noise, 123, -456));

        // Sur une grande zone, on doit rencontrer plusieurs biomes.
        let mut seen = std::collections::HashSet::new();
        for x in (-2000..2000).step_by(97) {
            for z in (-2000..2000).step_by(97) {
                seen.insert(format!("{:?}", biome_at(&noise, x, z)));
            }
        }
        assert!(
            seen.len() >= 3,
            "au moins 3 biomes différents attendus, trouvés : {seen:?}"
        );
    }
}
