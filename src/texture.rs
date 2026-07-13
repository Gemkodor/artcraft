pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> wgpu::TextureView {
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

/// Nombre de tuiles 16×16 dans l'atlas, côte à côte horizontalement.
pub const ATLAS_TILES: u32 = 4;

const TILE: u32 = 16;

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

/// Couleur d'un texel de l'atlas. Tuiles : 0 = dessus d'herbe, 1 = côté
/// d'herbe (terre + bande d'herbe irrégulière en haut), 2 = terre, 3 = pierre.
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
        _ => shade(STONE, v / 2),
    }
}

/// Texture atlas générée procéduralement (pas encore d'assets sur disque) :
/// toutes les tuiles de blocs dans une seule image de 64×16.
pub fn create_atlas_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let width = TILE * ATLAS_TILES;
    let mut pixels = Vec::with_capacity((width * TILE * 4) as usize);
    for y in 0..TILE {
        for x in 0..width {
            let [r, g, b] = atlas_pixel(x / TILE, x % TILE, y);
            pixels.extend_from_slice(&[r, g, b, 255]);
        }
    }

    let size = wgpu::Extent3d {
        width,
        height: TILE,
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
            rows_per_image: Some(TILE),
        },
        size,
    );

    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
