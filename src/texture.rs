pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth_texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Nombre de tuiles dans l'atlas, côte à côte horizontalement.
pub const ATLAS_TILES: u32 = 9;

/// Résolution "logique" des tuiles procédurales, dessinées en code.
const TILE: u32 = 16;
/// Résolution réelle des tuiles dans l'atlas GPU : celle des PNG du dossier
/// assets/ (pack Minetest en 128×128). Les tuiles procédurales sont
/// agrandies ×8 sans changer d'aspect.
const TILE_PX: u32 = 128;

fn hash(x: u32, y: u32, seed: u32) -> u32 {
    let mut h = x
        .wrapping_mul(374_761_393)
        .wrapping_add(y.wrapping_mul(668_265_263))
        .wrapping_add(seed.wrapping_mul(2_246_822_519));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^ (h >> 16)
}

fn shade(base: [i32; 3], variation: i32) -> [u8; 3] {
    [
        (base[0] + variation).clamp(0, 255) as u8,
        (base[1] + variation).clamp(0, 255) as u8,
        (base[2] + variation).clamp(0, 255) as u8,
    ]
}

const GRASS: [i32; 3] = [96, 160, 56];
const DIRT: [i32; 3] = [134, 96, 67];
const STONE: [i32; 3] = [125, 125, 125];
const SAND: [i32; 3] = [220, 205, 160];
const PLANK: [i32; 3] = [162, 127, 78];
const GLOW: [i32; 3] = [235, 198, 120];
const BARK: [i32; 3] = [104, 80, 48];
const LEAVES: [i32; 3] = [56, 116, 38];

/// Couleur d'un texel de l'atlas. Tuiles : 0 = dessus d'herbe, 1 = côté
/// d'herbe (terre + bande d'herbe irrégulière en haut), 2 = terre,
/// 3 = pierre, 4 = sable, 5 = planches, 6 = bloc lumineux, 7 = écorce,
/// 8 = feuilles.
fn atlas_pixel(tile: u32, x: u32, y: u32) -> [u8; 3] {
    let v = (hash(x, y, tile) % 48) as i32 - 24;
    match tile {
        0 => shade(GRASS, v),
        1 => {
            let grass_depth = 2 + hash(x, 0, 7) % 3;
            if y < grass_depth {
                shade(GRASS, v)
            } else {
                shade(DIRT, v)
            }
        }
        2 => shade(DIRT, v),
        3 => shade(STONE, v / 2),
        4 => shade(SAND, v / 3),
        5 => {
            // Planches : rainures horizontales toutes les 4 lignes, joints
            // verticaux décalés d'une planche à l'autre.
            if y % 4 == 3 {
                shade(PLANK, -40 + v / 4)
            } else if (x + (y / 4) * 7) % 16 == 0 {
                shade(PLANK, -28 + v / 4)
            } else {
                shade(PLANK, v / 3)
            }
        }
        6 => {
            // Bloc lumineux : taches claires groupées, façon glowstone.
            if hash(x / 3, y / 3, 42) % 3 == 0 {
                shade(GLOW, 20 + v / 3)
            } else {
                shade(GLOW, -35 + v / 2)
            }
        }
        7 => {
            // Écorce : stries verticales.
            if hash(x, 0, 21) % 4 == 0 {
                shade(BARK, -30 + v / 3)
            } else {
                shade(BARK, v / 3)
            }
        }
        _ => {
            // Feuilles : trous sombres épars dans le feuillage.
            if hash(x, y, 33) % 6 == 0 {
                shade(LEAVES, -55 + v / 3)
            } else {
                shade(LEAVES, v)
            }
        }
    }
}

/// Tuile procédurale rendue à la résolution de l'atlas : chaque texel 16×16
/// est répété (agrandissement "nearest"), le rendu est identique à l'écran.
fn procedural_tile(tile: u32) -> Vec<u8> {
    let scale = TILE_PX / TILE;
    let mut out = Vec::with_capacity((TILE_PX * TILE_PX * 4) as usize);
    for y in 0..TILE_PX {
        for x in 0..TILE_PX {
            let [r, g, b] = atlas_pixel(tile, x / scale, y / scale);
            out.extend_from_slice(&[r, g, b, 255]);
        }
    }
    out
}

/// Fond alpha-blendé par-dessus une image de base (pour les overlays du
/// pack, comme la frange d'herbe sur le côté du bloc de terre).
fn blend_over(base: &mut image::RgbaImage, overlay: &image::RgbaImage) {
    for (x, y, o) in overlay.enumerate_pixels() {
        let a = o[3] as u32;
        if a > 0 {
            let d = base.get_pixel_mut(x, y);
            for c in 0..3 {
                d[c] = ((o[c] as u32 * a + d[c] as u32 * (255 - a)) / 255) as u8;
            }
        }
    }
}

/// Remplace la transparence par une couleur de fond : notre rendu de blocs
/// est opaque, un pixel transparent afficherait des données indéfinies.
fn flatten_alpha(img: &mut image::RgbaImage, background: [u8; 3]) {
    for pixel in img.pixels_mut() {
        let a = pixel[3] as u32;
        if a < 255 {
            for c in 0..3 {
                pixel[c] = ((pixel[c] as u32 * a + background[c] as u32 * (255 - a)) / 255) as u8;
            }
            pixel[3] = 255;
        }
    }
}

/// Charge la tuile depuis le dossier assets/ (pack Minetest), si un fichier
/// lui correspond. `None` (fichier absent ou illisible) fait retomber
/// l'appelant sur la version procédurale : le jeu marche toujours, même
/// avec un dossier assets incomplet.
fn asset_tile(tile: u32) -> Option<Vec<u8>> {
    let file = |name: &str| -> Option<image::RgbaImage> {
        let path = std::path::Path::new("assets").join(name);
        let img = image::open(&path).ok()?.to_rgba8();
        Some(if img.dimensions() == (TILE_PX, TILE_PX) {
            img
        } else {
            image::imageops::resize(&img, TILE_PX, TILE_PX, image::imageops::FilterType::Nearest)
        })
    };

    let img = match tile {
        0 => file("default_grass.png")?,
        1 => {
            // Côté du bloc d'herbe : le pack fournit la frange d'herbe en
            // overlay transparent, à composer sur la terre.
            let mut dirt = file("default_dirt.png")?;
            blend_over(&mut dirt, &file("default_grass_side.png")?);
            dirt
        }
        2 => file("default_dirt.png")?,
        3 => file("default_stone.png")?,
        4 => file("default_sand.png")?,
        5 => file("default_wood.png")?,
        6 => file("default_meselamp.png")?,
        7 => file("default_tree.png")?,
        8 => {
            // Les feuilles du pack sont ajourées (transparence) ; on les
            // aplatit sur un vert sombre en attendant de vrais blocs
            // transparents dans le moteur.
            let mut leaves = file("default_leaves.png")?;
            flatten_alpha(&mut leaves, [24, 48, 16]);
            leaves
        }
        _ => return None,
    };
    Some(img.into_raw())
}

/// Construit l'atlas et l'envoie au GPU. `use_assets` choisit la source :
/// les PNG du dossier assets/ ou les tuiles calculées en code (et toute
/// tuile manquante côté assets retombe sur sa version procédurale).
pub fn create_atlas_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    use_assets: bool,
) -> wgpu::TextureView {
    let width = TILE_PX * ATLAS_TILES;
    let mut pixels = vec![0u8; (width * TILE_PX * 4) as usize];
    let row_bytes = (TILE_PX * 4) as usize;

    for tile in 0..ATLAS_TILES {
        let data = if use_assets { asset_tile(tile) } else { None };
        let data = data.unwrap_or_else(|| procedural_tile(tile));
        for y in 0..TILE_PX as usize {
            let src = y * row_bytes;
            let dst = (y * width as usize + (tile * TILE_PX) as usize) * 4;
            pixels[dst..dst + row_bytes].copy_from_slice(&data[src..src + row_bytes]);
        }
    }

    let size = wgpu::Extent3d {
        width,
        height: TILE_PX,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("atlas_texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(TILE_PX),
        },
        size,
    );

    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
