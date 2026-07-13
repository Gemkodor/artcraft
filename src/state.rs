use std::sync::Arc;

use glam::Vec3;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::camera::{Camera, CameraController, CameraUniform};
use crate::chunk::Block;
use crate::mesh::Vertex;
use crate::player::{self, Player};
use crate::texture::{self, ATLAS_TILES};
use crate::world::World;

/// Portée de la main du joueur, en blocs.
const REACH: f32 = 6.0;

/// Les blocs plaçables, dans l'ordre de la barre de sélection (touches 1-5).
pub const HOTBAR: [Block; 5] = [
    Block::Grass,
    Block::Dirt,
    Block::Stone,
    Block::Sand,
    Block::Plank,
];

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UiVertex {
    position: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

/// Sommets de la hotbar : par slot, un cadre blanc si sélectionné, un fond
/// sombre, et l'icône du bloc (tuile de l'atlas). Nombre de sommets constant.
fn hotbar_vertices(width: u32, height: u32, selected: usize) -> Vec<UiVertex> {
    let (w, h) = (width as f32, height as f32);
    let (slot, gap, margin) = (46.0, 5.0, 14.0);
    let n = HOTBAR.len() as f32;
    let x0 = (w - (n * slot + (n - 1.0) * gap)) / 2.0;

    let mut verts = Vec::with_capacity(66);
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
            verts.push(UiVertex { position, uv, color });
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

/// Les 12 sommets (2 quads) du viseur, en NDC, pour une fenêtre donnée.
fn crosshair_vertices(width: u32, height: u32) -> [[f32; 2]; 12] {
    // Demi-longueur et demi-épaisseur des branches, en pixels.
    let (len, thick) = (9.0, 1.5);
    let (lx, tx) = (len / width as f32 * 2.0, thick / width as f32 * 2.0);
    let (ly, ty) = (len / height as f32 * 2.0, thick / height as f32 * 2.0);
    [
        // Branche horizontale.
        [-lx, -ty], [lx, -ty], [lx, ty],
        [-lx, -ty], [lx, ty], [-lx, ty],
        // Branche verticale.
        [-tx, -ly], [tx, -ly], [tx, ly],
        [-tx, -ly], [tx, ly], [-tx, ly],
    ]
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
    world: World,
    player: Player,
    selected: usize,
    depth_view: wgpu::TextureView,
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
        let texture_view = texture::create_atlas_texture(&device, &queue);
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

        let camera_uniform = CameraUniform::from_camera(&camera);
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera_buffer"),
            contents: bytemuck::bytes_of(&camera_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
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
            contents: bytemuck::cast_slice(&crosshair_vertices(config.width, config.height)),
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
        let ui_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ui_buffer"),
            contents: bytemuck::cast_slice(&hotbar_vertices(config.width, config.height, 0)),
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
            world: World::new(),
            player,
            selected: 0,
            depth_view,
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
            bytemuck::cast_slice(&crosshair_vertices(width, height)),
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
            if self.world.block_at(target) == Block::Air && !self.player.intersects_block(target)
            {
                self.world.set_block(target, HOTBAR[self.selected]);
            }
        }
    }

    /// Touches 1-5.
    pub fn select_slot(&mut self, slot: usize) {
        if slot < HOTBAR.len() {
            self.selected = slot;
        }
    }

    /// Molette : +1 vers le bas, -1 vers le haut.
    pub fn scroll_slot(&mut self, dir: i32) {
        let n = HOTBAR.len() as i32;
        self.selected = ((self.selected as i32 + dir).rem_euclid(n)) as usize;
    }

    pub fn toggle_fly(&mut self) {
        self.player.toggle_fly();
    }

    pub fn update(&mut self, dt: f32) {
        self.player
            .update(&self.world, &self.controller, self.camera.yaw, dt);
        self.camera.position = self.player.pos + Vec3::Y * player::EYE_HEIGHT;
        self.world.update(&self.device, self.camera.position);
        let uniform = CameraUniform::from_camera(&self.camera);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
        self.queue.write_buffer(
            &self.ui_buffer,
            0,
            bytemuck::cast_slice(&hotbar_vertices(
                self.config.width,
                self.config.height,
                self.selected,
            )),
        );
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
                        // Bleu ciel.
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.45,
                            g: 0.7,
                            b: 1.0,
                            a: 1.0,
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

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.texture_bind_group, &[]);
            pass.set_bind_group(1, &self.camera_bind_group, &[]);
            for mesh in self.world.meshes() {
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
            }

            pass.set_pipeline(&self.crosshair_pipeline);
            pass.set_vertex_buffer(0, self.crosshair_buffer.slice(..));
            pass.draw(0..12, 0..1);

            pass.set_pipeline(&self.ui_pipeline);
            pass.set_bind_group(0, &self.texture_bind_group, &[]);
            pass.set_vertex_buffer(0, self.ui_buffer.slice(..));
            pass.draw(0..66, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }
}
