//! Bootstrap: window, key bindings, global hotkey, diagnostics.

mod app;
mod assets;
mod editor;
mod input;
mod live_stream;
mod processing;
mod staging;
mod store;
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
            editor::bind_keys(cx);

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

            // #362: closing the window quits the app (a zero-window gpui
            // app would otherwise keep running headless). Quitting is what
            // stops the engine sidecar cleanly — through the app entity's
            // own quit hook (app.rs `register_quit_hook`, the single owner
            // of engine shutdown; the server's --parent-pid watchdog is
            // the crash backstop). Process-lifetime hook: forgotten on
            // purpose, like the hotkey manager below.
            std::mem::forget(cx.on_window_closed(|cx| cx.quit()));

            match window {
                Ok(handle) => {
                    if let Err(err) = handle.update(cx, |app, window, _cx| {
                        window.focus(&app.root_focus);
                    }) {
                        eprintln!("Could not focus the Starling window: {err}");
                    }
                    // No engine shutdown here: the app entity's own quit
                    // hook (app.rs `register_quit_hook`) owns it, so quit
                    // stops the sidecar exactly once. (`WindowHandle` is
                    // `Copy`; the hotkey loop below takes its own copy.)
                    cx.spawn(async move |cx| {
                        loop {
                            gpui::Timer::after(Duration::from_millis(150)).await;
                            for event in GlobalHotKeyEvent::receiver().try_iter() {
                                if event.state() == HotKeyState::Pressed {
                                    cx.update(|cx| {
                                        let _ = handle.update(cx, |app, window, cx| {
                                            window.activate_window();
                                            // Same guarded entry as the
                                            // in-app binding (#209, #214.1):
                                            // the global path must not start
                                            // recording behind an open modal
                                            // or eat a repeat burst either.
                                            app.hotkey_toggle_recording(cx);
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
