// Shader du terrain : projection caméra, puis éclairage combinant la lumière
// pré-calculée par sommet (ciel + émission, occlusion ambiante incluse) et
// une légère directionnalité solaire pour distinguer les faces.

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

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
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    out.normal = in.normal;
    out.light = in.light;
    return out;
}

@group(0) @binding(0) var t_diffuse: texture_2d<f32>;
@group(0) @binding(1) var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let albedo = textureSample(t_diffuse, s_diffuse, in.uv);

    // Directionnalité : les faces tournées vers le soleil sont un peu plus
    // claires, ça donne du relief sans assombrir les grottes davantage.
    let sun_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let face_shade = 0.72 + 0.28 * max(dot(normalize(in.normal), sun_dir), 0.0);

    // Lumière du ciel (blanche, directionnelle) contre lumière émise
    // (chaude, omnidirectionnelle) : on garde la plus forte par canal.
    let sky = in.light.x * face_shade;
    let warm = vec3<f32>(1.0, 0.82, 0.55);
    let light_rgb = max(vec3<f32>(sky, sky, sky), warm * in.light.y);

    // Plancher pour ne jamais être dans le noir absolu.
    let lit = max(light_rgb, vec3<f32>(0.035, 0.035, 0.045));
    return vec4<f32>(albedo.rgb * lit, 1.0);
}
