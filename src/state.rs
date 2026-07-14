use std::sync::Arc;

use bytemuck::Zeroable;
use glam::Vec3;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::camera::{Camera, CameraController};
use crate::chunk::Block;
use crate::mesh::Vertex;
use crate::player::{self, Player};
use crate::sky::Sky;
use crate::texture;
use crate::ui::{self, HOTBAR, UiVertex};
use crate::world::{self, World};

/// Portée de la main du joueur, en blocs.
const REACH: f32 = 6.0;

/// Uniforms partagés par la scène : caméra, soleil, ciel et brouillard.
/// La disposition doit correspondre au struct `Globals` de shader.wgsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    sun: [f32; 4],
    sky_color: [f32; 4],
    fog: [f32; 4],
}

/// Le brouillard se termine juste avant la limite des chunks chargés, pour
/// masquer leur apparition.
const FOG_END: f32 = (world::RENDER_DISTANCE * 16 - 12) as f32;
const FOG_START: f32 = FOG_END - 45.0;

/// Sommet des quads soleil/lune : position monde + UV dans la texture ciel.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyVertex {
    position: [f32; 3],
    uv: [f32; 2],
}

/// Les 12 sommets (2 quads) du soleil et de la lune, placés loin de la
/// caméra dans leur direction respective (la lune est à l'opposé du soleil).
fn sky_vertices(camera_pos: Vec3, sun_dir: Vec3) -> [SkyVertex; 12] {
    const DIST: f32 = 420.0;
    const SIZE: f32 = 42.0;

    let mut verts = [SkyVertex {
        position: [0.0; 3],
        uv: [0.0; 2],
    }; 12];

    for (i, (dir, tile)) in [(sun_dir, 0.0f32), (-sun_dir, 1.0)].into_iter().enumerate() {
        // Base orthonormée du quad, face à la caméra. Près du zénith,
        // l'axe Y ne peut plus servir de référence "haut".
        let up_ref = if dir.y.abs() > 0.98 { Vec3::Z } else { Vec3::Y };
        let right = dir.cross(up_ref).normalize() * SIZE;
        let up = right.cross(dir).normalize() * SIZE;
        let center = camera_pos + dir * DIST;

        let (u0, u1) = (tile * 0.5 + 0.002, (tile + 1.0) * 0.5 - 0.002);
        let corners = [
            (center - right - up, [u0, 1.0]),
            (center + right - up, [u1, 1.0]),
            (center + right + up, [u1, 0.0]),
            (center - right - up, [u0, 1.0]),
            (center + right + up, [u1, 0.0]),
            (center - right + up, [u0, 0.0]),
        ];
        for (j, (pos, uv)) in corners.into_iter().enumerate() {
            verts[i * 6 + j] = SkyVertex {
                position: pos.to_array(),
                uv,
            };
        }
    }
    verts
}

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    crosshair_pipeline: wgpu::RenderPipeline,
    crosshair_buffer: wgpu::Buffer,
    ui_pipeline: wgpu::RenderPipeline,
    ui_buffer: wgpu::Buffer,
    sky_pipeline: wgpu::RenderPipeline,
    sky_buffer: wgpu::Buffer,
    sky_bind_group: wgpu::BindGroup,
    inventory_buffer: wgpu::Buffer,
    world: World,
    player: Player,
    sky: Sky,
    selected: usize,
    /// Barre de sélection modifiable : l'inventaire assigne un bloc au slot
    /// sélectionné.
    hotbar: [Block; 8],
    inventory_open: bool,
    /// Dernière position connue de la souris (repère fenêtre, origine en
    /// haut à gauche), pour les clics dans l'inventaire.
    cursor_pos: (f32, f32),
    depth_view: wgpu::TextureView,
    texture_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    use_assets: bool,
    texture_bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    pub camera: Camera,
    pub controller: CameraController,
}

impl State {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("aucun GPU compatible trouvé");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("échec de création du device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let depth_view = texture::create_depth_texture(&device, config.width, config.height);

        // Texture + sampler (filtrage "nearest" pour le look pixelisé).
        // Les PNG du dossier assets/ sont utilisés s'il existe, sinon les
        // textures procédurales ; la touche T bascule entre les deux.
        let use_assets = std::path::Path::new("assets").is_dir();
        let texture_view = texture::create_atlas_texture(&device, &queue, use_assets);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texture_bind_group"),
            layout: &texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Le joueur apparaît au-dessus du terrain ; la physique le pose au
        // sol dès que le chunk est généré. La caméra suit ses yeux.
        let player = Player::new(Vec3::new(8.5, 90.0, 8.5));
        let camera = Camera::new(
            player.pos + Vec3::Y * player::EYE_HEIGHT,
            45f32.to_radians(),
            -10f32.to_radians(),
            config.width as f32 / config.height as f32,
        );
        let controller = CameraController::new(0.002);

        let sky = Sky::new();
        let globals = Globals::zeroed();
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("globals_buffer"),
            contents: bytemuck::bytes_of(&globals),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                // Le fragment shader lit le soleil et le brouillard.
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bind_group"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&texture_layout, &camera_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        // Pipeline du viseur : dessiné par-dessus la scène, sans test de
        // profondeur (mais le format doit correspondre à la render pass).
        let crosshair_shader = device.create_shader_module(wgpu::include_wgsl!("crosshair.wgsl"));
        let crosshair_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("crosshair_layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let crosshair_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("crosshair_pipeline"),
            layout: Some(&crosshair_layout),
            vertex: wgpu::VertexState {
                module: &crosshair_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &crosshair_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let crosshair_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("crosshair_buffer"),
            contents: bytemuck::cast_slice(&ui::crosshair_vertices(config.width, config.height)),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // Pipeline UI (hotbar) : quads en NDC, alpha blending, échantillonne
        // l'atlas de blocs pour les icônes.
        let ui_shader = device.create_shader_module(wgpu::include_wgsl!("ui.wgsl"));
        let ui_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui_layout"),
            bind_group_layouts: &[&texture_layout],
            push_constant_ranges: &[],
        });
        let ui_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ui_pipeline"),
            layout: Some(&ui_layout),
            vertex: wgpu::VertexState {
                module: &ui_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<UiVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &ui_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let hotbar = HOTBAR;
        let ui_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ui_buffer"),
            contents: bytemuck::cast_slice(&ui::hotbar_vertices(
                config.width,
                config.height,
                0,
                &hotbar,
            )),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let inventory_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("inventory_buffer"),
            contents: bytemuck::cast_slice(&ui::inventory_vertices(config.width, config.height)),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // Soleil et lune : leur propre petite texture et un pipeline dédié,
        // dessiné avant le terrain sans écrire la profondeur.
        let sky_view = texture::create_sky_texture(&device, &queue);
        let sky_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky_bind_group"),
            layout: &texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&sky_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let sky_shader = device.create_shader_module(wgpu::include_wgsl!("skybox.wgsl"));
        let sky_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky_layout"),
            bind_group_layouts: &[&texture_layout, &camera_layout],
            push_constant_ranges: &[],
        });
        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky_pipeline"),
            layout: Some(&sky_layout),
            vertex: wgpu::VertexState {
                module: &sky_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SkyVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &sky_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let sky_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sky_buffer"),
            contents: bytemuck::cast_slice(&sky_vertices(Vec3::ZERO, Vec3::Y)),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            crosshair_pipeline,
            crosshair_buffer,
            ui_pipeline,
            ui_buffer,
            sky_pipeline,
            sky_buffer,
            sky_bind_group,
            inventory_buffer,
            world: World::new(),
            player,
            sky,
            selected: 0,
            hotbar,
            inventory_open: false,
            cursor_pos: (0.0, 0.0),
            depth_view,
            texture_layout,
            sampler,
            use_assets,
            texture_bind_group,
            camera_buffer,
            camera_bind_group,
            camera,
            controller,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = texture::create_depth_texture(&self.device, width, height);
        self.camera.aspect = width as f32 / height as f32;
        self.queue.write_buffer(
            &self.crosshair_buffer,
            0,
            bytemuck::cast_slice(&ui::crosshair_vertices(width, height)),
        );
    }

    /// Clic gauche : casse le bloc visé.
    pub fn break_block(&mut self) {
        let hit = self
            .world
            .raycast(self.camera.position, self.camera.forward(), REACH);
        if let Some(hit) = hit {
            self.world.set_block(hit.block, Block::Air);
        }
    }

    /// Clic droit : pose le bloc sélectionné sur la face visée.
    pub fn place_block(&mut self) {
        let hit = self
            .world
            .raycast(self.camera.position, self.camera.forward(), REACH);
        if let Some(hit) = hit {
            let target = hit.block + hit.normal;
            // Jamais dans la hitbox du joueur.
            if self.world.block_at(target) == Block::Air && !self.player.intersects_block(target) {
                self.world.set_block(target, self.hotbar[self.selected]);
            }
        }
    }

    pub fn inventory_open(&self) -> bool {
        self.inventory_open
    }

    /// Touche E : ouvre/ferme l'inventaire. Renvoie le nouvel état pour que
    /// main.rs libère ou capture la souris en conséquence.
    pub fn toggle_inventory(&mut self) -> bool {
        self.inventory_open = !self.inventory_open;
        self.inventory_open
    }

    pub fn close_inventory(&mut self) {
        self.inventory_open = false;
    }

    pub fn set_cursor(&mut self, x: f32, y: f32) {
        self.cursor_pos = (x, y);
    }

    /// Clic dans l'inventaire : assigne le bloc cliqué au slot sélectionné
    /// de la hotbar et ferme l'inventaire. Renvoie true si un bloc a été
    /// choisi (main.rs recapture alors la souris).
    pub fn inventory_click(&mut self) -> bool {
        let block = ui::inventory_block_at(self.config.width, self.config.height, self.cursor_pos);
        if let Some(block) = block {
            self.hotbar[self.selected] = block;
            self.inventory_open = false;
            return true;
        }
        false
    }

    /// Touches 1-8.
    pub fn select_slot(&mut self, slot: usize) {
        if slot < self.hotbar.len() {
            self.selected = slot;
        }
    }

    /// Molette : +1 vers le bas, -1 vers le haut.
    pub fn scroll_slot(&mut self, dir: i32) {
        let n = self.hotbar.len() as i32;
        self.selected = ((self.selected as i32 + dir).rem_euclid(n)) as usize;
    }

    pub fn toggle_fly(&mut self) {
        self.player.toggle_fly();
    }

    /// Touche T : bascule entre les textures du dossier assets/ et les
    /// textures procédurales. L'atlas est reconstruit et le bind group
    /// remplacé ; les meshes et UVs ne changent pas.
    pub fn toggle_textures(&mut self) {
        self.use_assets = !self.use_assets;
        let view = texture::create_atlas_texture(&self.device, &self.queue, self.use_assets);
        self.texture_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texture_bind_group"),
            layout: &self.texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
    }

    pub fn update(&mut self, dt: f32) {
        self.player
            .update(&self.world, &self.controller, self.camera.yaw, dt);
        self.camera.position = self.player.pos + Vec3::Y * player::EYE_HEIGHT;
        self.world.update(&self.device, self.camera.position);
        self.sky.advance(dt);

        let (pos, sun, color) = (
            self.camera.position,
            self.sky.sun_dir(),
            self.sky.sky_color(),
        );
        let globals = Globals {
            view_proj: self.camera.view_proj().to_cols_array_2d(),
            camera_pos: [pos.x, pos.y, pos.z, 0.0],
            sun: [sun.x, sun.y, sun.z, self.sky.sun_intensity()],
            sky_color: [color.x, color.y, color.z, 1.0],
            fog: [FOG_START, FOG_END, 0.0, 0.0],
        };
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&globals));
        self.queue.write_buffer(
            &self.sky_buffer,
            0,
            bytemuck::cast_slice(&sky_vertices(pos, sun)),
        );
        self.queue.write_buffer(
            &self.ui_buffer,
            0,
            bytemuck::cast_slice(&ui::hotbar_vertices(
                self.config.width,
                self.config.height,
                self.selected,
                &self.hotbar,
            )),
        );
        if self.inventory_open {
            self.queue.write_buffer(
                &self.inventory_buffer,
                0,
                bytemuck::cast_slice(&ui::inventory_vertices(
                    self.config.width,
                    self.config.height,
                )),
            );
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // La couleur du ciel suit le cycle jour/nuit ; le
                        // brouillard du shader utilise la même valeur pour
                        // que l'horizon soit invisible.
                        load: wgpu::LoadOp::Clear({
                            let c = self.sky.sky_color();
                            wgpu::Color {
                                r: c.x as f64,
                                g: c.y as f64,
                                b: c.z as f64,
                                a: 1.0,
                            }
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            // Soleil et lune d'abord : sans écriture de profondeur, le
            // terrain dessiné ensuite les recouvre.
            pass.set_pipeline(&self.sky_pipeline);
            pass.set_bind_group(0, &self.sky_bind_group, &[]);
            pass.set_bind_group(1, &self.camera_bind_group, &[]);
            pass.set_vertex_buffer(0, self.sky_buffer.slice(..));
            pass.draw(0..12, 0..1);

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.texture_bind_group, &[]);
            pass.set_bind_group(1, &self.camera_bind_group, &[]);
            for mesh in self.world.meshes() {
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
            }

            if !self.inventory_open {
                pass.set_pipeline(&self.crosshair_pipeline);
                pass.set_vertex_buffer(0, self.crosshair_buffer.slice(..));
                pass.draw(0..12, 0..1);
            }

            pass.set_pipeline(&self.ui_pipeline);
            pass.set_bind_group(0, &self.texture_bind_group, &[]);
            pass.set_vertex_buffer(0, self.ui_buffer.slice(..));
            pass.draw(0..ui::HOTBAR_VERTEX_COUNT, 0..1);

            if self.inventory_open {
                pass.set_vertex_buffer(0, self.inventory_buffer.slice(..));
                pass.draw(0..ui::INVENTORY_VERTEX_COUNT, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }
}
