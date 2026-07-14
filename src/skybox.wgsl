// Soleil et lune : deux quads texturés dessinés AVANT le terrain, sans
// écrire la profondeur — le monde les recouvre naturellement.

struct Globals {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    sun: vec4<f32>,
    sky_color: vec4<f32>,
    fog: vec4<f32>,
};

@group(1) @binding(0)
var<uniform> globals: Globals;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = globals.view_proj * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

@group(0) @binding(0) var t_sky: texture_2d<f32>;
@group(0) @binding(1) var s_sky: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_sky, s_sky, in.uv);
}
