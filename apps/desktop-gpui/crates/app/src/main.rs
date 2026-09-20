//! Bootstrap: window, key bindings, global hotkey, diagnostics.

mod app;
mod assets;
mod input;
mod theme;
mod upload;
mod views;

use std::time::{Duration, Instant};

use global_hotkey::hotkey::{CMD_OR_CTRL, Code, HotKey, Modifiers};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};
use gpui::{
    AppContext, Application, Bounds, KeyBinding, Size, TitlebarOptions, WindowBounds,
    WindowDecorations, WindowOptions, px, size,
};

use crate::app::{StarlingApp, ToggleRecording};

fn main() {
    let started = Instant::now();
    let diagnostics = std::env::var("STARLING_DIAGNOSTICS")
        .map(|value| value == "1")
        .unwrap_or(false);

    Application::new()
        .with_assets(assets::Assets)
        .run(move |cx| {
            cx.bind_keys([KeyBinding::new(
                "secondary-shift-space",
                ToggleRecording,
                None,
            )]);
            input::bind_keys(cx);

            let options = WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(Bounds::centered(
                    None,
                    size(px(1180.), px(760.)),
                    cx,
                ))),
                window_min_size: Some(Size {
                    width: px(920.),
                    height: px(620.),
                }),
                titlebar: Some(TitlebarOptions {
                    title: Some("Starling".into()),
                    appears_transparent: false,
                    traffic_light_position: None,
                }),
                window_decorations: Some(WindowDecorations::Server),
                window_background: gpui::WindowBackgroundAppearance::Opaque,
                focus: true,
                ..Default::default()
            };

            let window = cx.open_window(options, |_window, cx| {
                cx.new(|cx| {
                    let mut app = StarlingApp::new(started, diagnostics, cx);
                    app.init(cx);
                    app
                })
            });

            match window {
                Ok(handle) => {
                    if let Err(err) = handle.update(cx, |app, window, _cx| {
                        window.focus(&app.root_focus);
                    }) {
                        eprintln!("Could not focus the Starling window: {err}");
                    }
                    let window = handle.clone();
                    cx.spawn(async move |cx| {
                        loop {
                            gpui::Timer::after(Duration::from_millis(150)).await;
                            for event in GlobalHotKeyEvent::receiver().try_iter() {
                                if event.state() == HotKeyState::Pressed {
                                    cx.update(|cx| {
                                        let _ = window.update(cx, |app, window, cx| {
                                            window.activate_window();
                                            app.toggle_recording(cx);
                                        });
                                    })
                                    .ok();
                                }
                            }
                        }
                    })
                    .detach();
                }
                Err(err) => {
                    eprintln!("Could not open the Starling window: {err}");
                }
            }

            setup_global_hotkey();
        });
}

fn setup_global_hotkey() {
    match GlobalHotKeyManager::new() {
        Ok(manager) => {
            let hotkey = HotKey::new(Some(CMD_OR_CTRL | Modifiers::SHIFT), Code::Space);
            match manager.register(hotkey) {
                Ok(()) => {
                    std::mem::forget(manager);
                }
                Err(err) => {
                    eprintln!("Global shortcut unavailable: {err}");
                }
            }
        }
        Err(err) => {
            eprintln!("Global shortcut unavailable: {err}");
        }
    }
}
