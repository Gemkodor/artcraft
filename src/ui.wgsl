// UI 2D (hotbar) : quads en coordonnées d'écran (NDC), soit en couleur unie
// (uv négatif), soit texturés depuis l'atlas de blocs.

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@group(0) @binding(0) var t_atlas: texture_2d<f32>;
@group(0) @binding(1) var s_atlas: sampler;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (in.uv.x < 0.0) {
        return in.color;
    }
    // textureSampleLevel : pas de dérivées implicites, donc utilisable
    // dans une branche.
    return textureSampleLevel(t_atlas, s_atlas, in.uv, 0.0) * in.color;
}
