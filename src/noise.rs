//! Bruit de Perlin 2D/3D écrit à la main, avec fBm (somme d'octaves).
//! Un gradient pseudo-aléatoire est associé à chaque coin de cellule via un
//! hash entier ; le bruit est l'interpolation lissée des produits scalaires
//! gradient·offset. Déterministe pour une seed donnée.

#[derive(Copy, Clone)]
pub struct Noise {
    seed: u32,
}

/// Interpolation quintique de Perlin : dérivées nulles aux bornes,
/// pas d'artefacts de grille visibles contrairement à un lerp simple.
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

const SQRT_HALF: f32 = std::f32::consts::FRAC_1_SQRT_2;

#[rustfmt::skip]
const GRAD2: [[f32; 2]; 8] = [
    [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
    [SQRT_HALF, SQRT_HALF], [-SQRT_HALF, SQRT_HALF],
    [SQRT_HALF, -SQRT_HALF], [-SQRT_HALF, -SQRT_HALF],
];

/// Gradients 3D classiques de Perlin : les 12 arêtes du cube.
fn grad3(h: u32, x: f32, y: f32, z: f32) -> f32 {
    match h % 12 {
        0 => x + y,
        1 => -x + y,
        2 => x - y,
        3 => -x - y,
        4 => x + z,
        5 => -x + z,
        6 => x - z,
        7 => -x - z,
        8 => y + z,
        9 => -y + z,
        10 => y - z,
        _ => -y - z,
    }
}

impl Noise {
    pub fn new(seed: u32) -> Self {
        Self { seed }
    }

    /// Hash entier d'un coin de cellule. `salt` décorrèle les octaves du fBm.
    fn hash(&self, x: i32, y: i32, z: i32, salt: u32) -> u32 {
        let mut h = (x as u32).wrapping_mul(374_761_393)
            ^ (y as u32).wrapping_mul(668_265_263)
            ^ (z as u32).wrapping_mul(2_246_822_519)
            ^ self.seed.wrapping_mul(3_266_489_917)
            ^ salt.wrapping_mul(2_654_435_761);
        h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
        h ^ (h >> 16)
    }

    /// Bruit de Perlin 2D, valeurs approximativement dans [-1, 1].
    fn perlin2(&self, x: f32, z: f32, salt: u32) -> f32 {
        let (xi, zi) = (x.floor() as i32, z.floor() as i32);
        let (xf, zf) = (x - xi as f32, z - zi as f32);

        let dot = |cx: i32, cz: i32, dx: f32, dz: f32| {
            let g = GRAD2[(self.hash(cx, 0, cz, salt) & 7) as usize];
            g[0] * dx + g[1] * dz
        };

        let n00 = dot(xi, zi, xf, zf);
        let n10 = dot(xi + 1, zi, xf - 1.0, zf);
        let n01 = dot(xi, zi + 1, xf, zf - 1.0);
        let n11 = dot(xi + 1, zi + 1, xf - 1.0, zf - 1.0);

        let (u, w) = (fade(xf), fade(zf));
        lerp(lerp(n00, n10, u), lerp(n01, n11, u), w)
    }

    /// Bruit de Perlin 3D, valeurs approximativement dans [-1, 1].
    fn perlin3(&self, x: f32, y: f32, z: f32, salt: u32) -> f32 {
        let (xi, yi, zi) = (x.floor() as i32, y.floor() as i32, z.floor() as i32);
        let (xf, yf, zf) = (x - xi as f32, y - yi as f32, z - zi as f32);

        let dot = |cx: i32, cy: i32, cz: i32, dx: f32, dy: f32, dz: f32| {
            grad3(self.hash(cx, cy, cz, salt), dx, dy, dz)
        };

        let n000 = dot(xi, yi, zi, xf, yf, zf);
        let n100 = dot(xi + 1, yi, zi, xf - 1.0, yf, zf);
        let n010 = dot(xi, yi + 1, zi, xf, yf - 1.0, zf);
        let n110 = dot(xi + 1, yi + 1, zi, xf - 1.0, yf - 1.0, zf);
        let n001 = dot(xi, yi, zi + 1, xf, yf, zf - 1.0);
        let n101 = dot(xi + 1, yi, zi + 1, xf - 1.0, yf, zf - 1.0);
        let n011 = dot(xi, yi + 1, zi + 1, xf, yf - 1.0, zf - 1.0);
        let n111 = dot(xi + 1, yi + 1, zi + 1, xf - 1.0, yf - 1.0, zf - 1.0);

        let (u, v, w) = (fade(xf), fade(yf), fade(zf));
        lerp(
            lerp(lerp(n000, n100, u), lerp(n010, n110, u), v),
            lerp(lerp(n001, n101, u), lerp(n011, n111, u), v),
            w,
        )
    }

    /// fBm 2D : somme d'octaves de fréquence doublée et d'amplitude divisée
    /// par deux, normalisée pour rester dans [-1, 1].
    pub fn fbm2(&self, mut x: f32, mut z: f32, octaves: u32) -> f32 {
        let (mut sum, mut amp, mut norm) = (0.0, 1.0, 0.0);
        for octave in 0..octaves {
            sum += self.perlin2(x, z, octave) * amp;
            norm += amp;
            amp *= 0.5;
            x *= 2.0;
            z *= 2.0;
        }
        sum / norm
    }

    /// fBm 3D, même principe.
    pub fn fbm3(&self, mut x: f32, mut y: f32, mut z: f32, octaves: u32) -> f32 {
        let (mut sum, mut amp, mut norm) = (0.0, 1.0, 0.0);
        for octave in 0..octaves {
            sum += self.perlin3(x, y, z, octave) * amp;
            norm += amp;
            amp *= 0.5;
            x *= 2.0;
            y *= 2.0;
            z *= 2.0;
        }
        sum / norm
    }
}
