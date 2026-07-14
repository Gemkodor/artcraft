mod camera;
mod chunk;
mod light;
mod mesh;
mod noise;
mod player;
mod sky;
mod state;
mod texture;
mod ui;
mod worker;
mod world;

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{
    DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use state::State;

#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
    last_frame: Option<Instant>,
    mouse_grabbed: bool,
}

impl App {
    fn set_mouse_grab(&mut self, grab: bool) {
        let Some(window) = &self.window else { return };
        if grab {
            // Locked n'est pas supporté partout (notamment sous Windows) :
            // on retombe sur Confined si besoin.
            let result = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
            if result.is_ok() {
                window.set_cursor_visible(false);
                self.mouse_grabbed = true;
            }
        } else {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            window.set_cursor_visible(true);
            self.mouse_grabbed = false;
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title(
                    "Artcraft — clic : casser · droit : poser · 1-8 : bloc · F : vol · T : textures · Échap : libérer",
                ))
                .expect("échec de création de la fenêtre"),
        );
        self.state = Some(pollster::block_on(State::new(window.clone())));
        self.window = Some(window);
        self.last_frame = Some(Instant::now());
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    let pressed = event.state == ElementState::Pressed;
                    match code {
                        KeyCode::Escape if pressed => self.set_mouse_grab(false),
                        KeyCode::KeyF if pressed => state.toggle_fly(),
                        KeyCode::KeyT if pressed => state.toggle_textures(),
                        KeyCode::Digit1 if pressed => state.select_slot(0),
                        KeyCode::Digit2 if pressed => state.select_slot(1),
                        KeyCode::Digit3 if pressed => state.select_slot(2),
                        KeyCode::Digit4 if pressed => state.select_slot(3),
                        KeyCode::Digit5 if pressed => state.select_slot(4),
                        KeyCode::Digit6 if pressed => state.select_slot(5),
                        KeyCode::Digit7 if pressed => state.select_slot(6),
                        KeyCode::Digit8 if pressed => state.select_slot(7),
                        _ => {
                            state.controller.process_key(code, pressed);
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if self.mouse_grabbed {
                    let y = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
                    };
                    if y < 0.0 {
                        state.scroll_slot(1);
                    } else if y > 0.0 {
                        state.scroll_slot(-1);
                    }
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => match button {
                MouseButton::Left if !self.mouse_grabbed => self.set_mouse_grab(true),
                MouseButton::Left => state.break_block(),
                MouseButton::Right if self.mouse_grabbed => state.place_block(),
                _ => {}
            },
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = self
                    .last_frame
                    .map(|t| (now - t).as_secs_f32())
                    .unwrap_or(0.0);
                self.last_frame = Some(now);

                state.update(dt);
                match state.render() {
                    Ok(()) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = self.window.as_ref().unwrap().inner_size();
                        state.resize(size.width, size.height);
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => eprintln!("erreur de rendu : {e:?}"),
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if !self.mouse_grabbed {
            return;
        }
        if let (Some(state), DeviceEvent::MouseMotion { delta: (dx, dy) }) =
            (&mut self.state, event)
        {
            state.controller.process_mouse(&mut state.camera, dx, dy);
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("échec de création de l'event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("event loop error");
}
