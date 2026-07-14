// Shader du terrain : projection caméra, éclairage pré-calculé par sommet
// (ciel + émission, occlusion ambiante incluse) modulé par le cycle
// jour/nuit, et brouillard atmosphérique fondu dans la couleur du ciel.

struct Globals {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    // xyz : direction VERS le soleil ; w : intensité de la lumière du ciel.
    sun: vec4<f32>,
    sky_color: vec4<f32>,
    // x : début du brouillard, y : fin (en blocs).
    fog: vec4<f32>,
};

@group(1) @binding(0)
var<uniform> globals: Globals;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) light: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) light: vec2<f32>,
    @location(3) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = globals.view_proj * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    out.normal = in.normal;
    out.light = in.light;
    out.world_pos = in.position;
    return out;
}

@group(0) @binding(0) var t_diffuse: texture_2d<f32>;
@group(0) @binding(1) var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let albedo = textureSample(t_diffuse, s_diffuse, in.uv);

    // Directionnalité : les faces tournées vers le soleil sont un peu plus
    // claires, ça donne du relief sans assombrir les grottes davantage.
    let face_shade = 0.72 + 0.28 * max(dot(normalize(in.normal), globals.sun.xyz), 0.0);

    // Lumière du ciel (blanche, modulée par le jour/nuit) contre lumière
    // émise (chaude, constante) : on garde la plus forte par canal.
    let sky = in.light.x * face_shade * globals.sun.w;
    let warm = vec3<f32>(1.0, 0.82, 0.55);
    let light_rgb = max(vec3<f32>(sky, sky, sky), warm * in.light.y);

    // Plancher pour ne jamais être dans le noir absolu.
    let lit = albedo.rgb * max(light_rgb, vec3<f32>(0.035, 0.035, 0.045));

    // Brouillard : le terrain lointain se fond dans la couleur du ciel.
    let dist = length(in.world_pos - globals.camera_pos.xyz);
    let fog_amount = smoothstep(globals.fog.x, globals.fog.y, dist);
    let final_rgb = mix(lit, globals.sky_color.rgb, fog_amount);

    return vec4<f32>(final_rgb, 1.0);
}
