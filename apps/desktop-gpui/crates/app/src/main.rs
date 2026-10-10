//! Bootstrap: window, key bindings, the recording shortcut, diagnostics.

mod activation;
mod app;
mod assets;
mod cues;
mod delivery;
mod editor;
mod engine_link;
mod host_link;
mod input;
mod mic;
mod overlay;
mod portal;
mod processing;
mod remote_take;
mod shortcut;
mod slider;
mod staging;
mod store;
mod system_check;
mod theme;
mod upkeep;
mod upload;
mod views;

use std::time::Instant;

use gpui::{
    AppContext, Application, Bounds, Size, TitlebarOptions, WindowBounds, WindowDecorations,
    WindowOptions, px, size,
};

use crate::app::StarlingApp;

fn main() {
    // #220: the same executable is the recording service. The app starts
    // it as `starling-gpui --runtime-host …` (see `host_link`), so app and
    // service are always one build; nothing below runs in that process.
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() == Some("--runtime-host") {
        std::process::exit(starling_runtime_host::cli::run(args.collect()));
    }
    let started = Instant::now();
    let diagnostics = std::env::var("STARLING_DIAGNOSTICS")
        .map(|value| value == "1")
        .unwrap_or(false);

    Application::new()
        .with_assets(assets::Assets)
        .run(move |cx| {
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
                // The id the desktop portal knows Starling by (#221).
                app_id: Some(portal::APP_ID.to_string()),
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
            // purpose.
            // The dictation overlay (#221) is a window too: only the main
            // window going away quits.
            std::mem::forget(cx.on_window_closed(|cx| {
                if cx
                    .windows()
                    .into_iter()
                    .all(overlay::Overlay::is_overlay_window)
                {
                    cx.quit();
                }
            }));

            match window {
                Ok(handle) => {
                    if let Err(err) = handle.update(cx, |app, window, _cx| {
                        window.focus(&app.root_focus);
                    }) {
                        eprintln!("Could not focus the Starling window: {err}");
                    }
                    // No engine shutdown here: the app entity's own quit
                    // hook (app.rs `register_quit_hook`) owns it, so quit
                    // stops the sidecar exactly once.
                    //
                    // #221: the system-wide recording shortcut is created
                    // here, on the main thread (macOS requires it), and
                    // owned by the app from then on. Its events never
                    // raise or focus this window: the app being dictated
                    // into keeps focus.
                    let shortcuts = shortcut::GlobalShortcuts::new();
                    if let Err(err) = handle.update(cx, |app, window, cx| {
                        app.install_global_shortcuts(shortcuts, cx);
                        app.track_window_focus(window, cx);
                    }) {
                        eprintln!("Could not set up the recording shortcut: {err}");
                    }
                }
                Err(err) => {
                    eprintln!("Could not open the Starling window: {err}");
                }
            }
        });
}

