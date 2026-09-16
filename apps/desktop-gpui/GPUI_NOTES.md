# gpui 0.2.2 API cheat-sheet (verified against docs.rs / zed source 2025-10)

NOTE from the coordinator: this sheet was produced by a research agent against
docs.rs for exactly 0.2.2. Items listed in the UNVERIFIED section were not
confirmed — verify against the vendored source under
`~/.cargo/registry/src/*/gpui-0.2.2/` before relying on them. One correction to
the appendix example: `ToggleRecording` comes from the local `actions!(...)`
macro, not from the `gpui` import list.

## 1. Bootstrap: Application, App, open_window, WindowOptions

Two distinct types: `Application` = bootstrap (`main`); `App` = the context you get inside `.run(...)` (and `Context<'_, T>` derefs to it).

```rust
// Application
pub fn new() -> Self
pub fn headless() -> Self                       // no GUI (SSH etc.)
pub fn with_assets(self, asset_source: impl AssetSource) -> Self
pub fn run<F>(self, on_finish_launching: F) where F: 'static + FnOnce(&mut App)
// also: with_http_client(Arc<dyn HttpClient>), on_open_urls, on_reopen

// App
pub fn open_window<V: 'static + Render>(
    &mut self,
    options: WindowOptions,
    build_root_view: impl FnOnce(&mut Window, &mut App) -> Entity<V>,
) -> Result<WindowHandle<V>>
```

`WindowOptions` — all 14 public fields:

```rust
pub struct WindowOptions {
    pub window_bounds:      Option<WindowBounds>,
    pub titlebar:           Option<TitlebarOptions>,
    pub focus:              bool,
    pub show:               bool,
    pub kind:               WindowKind,
    pub is_movable:         bool,
    pub is_resizable:       bool,
    pub is_minimizable:     bool,
    pub display_id:         Option<DisplayId>,
    pub window_background:  WindowBackgroundAppearance,
    pub app_id:             Option<String>,
    pub window_min_size:    Option<Size<Pixels>>,
    pub window_decorations: Option<WindowDecorations>,  // Wayland; "may be ignored"
    pub tabbing_identifier: Option<String>,             // macOS native tabs
}

pub enum WindowBounds { Windowed(Bounds<Pixels>), Maximized(Bounds<Pixels>), Fullscreen(Bounds<Pixels>) }
WindowBounds::centered(size: Size<Pixels>, cx: &App) -> Self
Bounds::new(origin: Point<T>, size: Size<T>)
pub const fn size<T>(width: T, height: T) -> Size<T>
pub const fn px(pixels: f32) -> Pixels

pub struct TitlebarOptions {
    pub title: Option<SharedString>,
    pub appears_transparent: bool,                  // macOS & Windows ONLY
    pub traffic_light_position: Option<Point<Pixels>>, // macOS only
}
pub enum WindowDecorations { Server, Client }       // Server = WM titlebar (default), Client = CSD
pub enum WindowBackgroundAppearance { Opaque, Transparent, Blurred }
```

- Fixed 1180x760: `window_bounds: Some(WindowBounds::Windowed(Bounds::new(point(px(0.), px(0.)), size(px(1180.), px(760.)))))` or `WindowBounds::centered(...)`. Min size: `window_min_size: Some(size(px(920.), px(620.)))`.
- Title: `TitlebarOptions { title: Some("Starling".into()), .. }`; runtime `window.set_window_title(&mut self, title: &str)`.
- Background `#171813`: paint via the root view `.size_full().bg(rgb(0x171813))` with `WindowBackgroundAppearance::Opaque`.

## 2. Entity / context model (0.2: `Context<'a, T>` has a lifetime)

```rust
fn render(&mut self, window: &mut Window, cx: &mut Context<'_, Self>) -> impl IntoElement; // trait Render

// AppContext (on App and Context)
pub fn new<U: 'static>(&mut self, build_entity: impl FnOnce(&mut Context<'_, U>) -> U) -> Entity<U>;

// Context<'a, T>
pub fn notify(&mut self);
pub fn emit<Evt>(&mut self, event: Evt) where T: EventEmitter<Evt>, Evt: 'static;
pub fn subscribe<T2, Evt>(&mut self, entity: &Entity<T2>,
    on_event: impl FnMut(&mut T, Entity<T2>, &Evt, &mut Context<'_, T>) + 'static) -> Subscription;
pub fn observe<W>(&mut self, entity: &Entity<W>,
    on_notify: impl FnMut(&mut T, Entity<W>, &mut Context<'_, T>) + 'static) -> Subscription;
pub fn listener<E: ?Sized>(&self,
    f: impl Fn(&mut T, &E, &mut Window, &mut Context<'_, T>) + 'static,
) -> impl Fn(&E, &mut Window, &mut App) + 'static;

// Entity<T>
pub fn update<R, C: AppContext>(&self, cx: &mut C,
    update: impl FnOnce(&mut T, &mut Context<'_, T>) -> R) -> C::Result<R>;
pub fn downgrade(&self) -> WeakEntity<T>;
pub fn read<'a>(&self, cx: &'a App) -> &'a T;

// WeakEntity<T>
pub fn update<C, R>(&self, cx: &mut C, update: impl FnOnce(&mut T, &mut Context<'_, T>) -> R) -> Result<R>;
pub fn upgrade(&self) -> Option<Entity<T>>;
```

**`spawn` closure signature (0.2): closure receives WeakEntity as FIRST arg plus `&mut AsyncApp`:**

```rust
pub fn spawn<AsyncFn, R>(&self, f: AsyncFn) -> Task<R>
where T: 'static, AsyncFn: AsyncFnOnce(WeakEntity<T>, &mut AsyncApp) -> R + 'static, R: 'static;
// App::spawn: AsyncFnOnce(&mut AsyncApp); Window::spawn: AsyncFnOnce(&mut AsyncWindowContext)

cx.spawn(async move |this, cx| {
    loop {
        Timer::after(Duration::from_secs(5)).await;
        this.update(cx, |this, cx| { this.status = check(); cx.notify(); }).ok();
    }
}).detach();   // dropping a Task cancels it; Task::detach verified
```

Also on `Window`: `window.observe(&entity, cx, |entity, window, app| ...)`, `window.on_next_frame(cb)`.

## 3. Async & timers (no tokio needed)

```rust
fn background_spawn<R>(&self, future: impl Future<Output = R> + Send + 'static) -> Task<R> where R: Send + 'static;
fn background_executor(&self) -> &BackgroundExecutor;
// BackgroundExecutor: spawn(future), timer(Duration) -> Task<()>, block(future), now() -> Instant

pub fn after(duration: Duration) -> Timer;       // gpui::Timer = async_io::Timer re-export
pub fn at(instant: Instant) -> Timer;
pub fn interval(period: Duration) -> Timer;      // also a Stream (Item = Instant)
```

`smol ^2.0` is a direct gpui dependency; `futures ^0.3` also available transitively. Polling pattern: `cx.spawn` + `Timer::after` loop + `this.update(...)` + `cx.notify()`. CPU/blocking work → `cx.background_spawn`.

## 4. Styling on div() (trait `Styled`; `fn style(&mut self) -> &mut StyleRefinement`)

`div()` returns `Div`; `.id(impl Into<ElementId>)` returns `Stateful<Div>` (needed for `on_click`, `overflow_y_scroll`, `tooltip`).

- Flex: `flex()`, `flex_row()`, `flex_col()`, `flex_1()`, `items_center()` (+family), `justify_center()` (+family), `gap(...)` + `gap_0..gap_128`, fractions, negatives.
- Spacing: `p/px/py/pt/...(...)`, absolute `px(px(28.))` or t-shirt `px_4`; `w/h/size(...)`, `w_full()/h_full()`, `min_w/max_w(...)`.
- Overflow: `overflow_hidden()`; `overflow_y_scroll()/overflow_x_scroll()` on `StatefulInteractiveElement` — **after** `.id(...)`. `gpui::ScrollHandle`, `UniformListScrollHandle`.
- Borders: `border_1()`, `border_color(impl Into<Hsla>)` (per-side variants UNVERIFIED).
- Corners/shadows: `rounded(...)` + `rounded_none/xs/sm/md/lg/xl/2xl/3xl/full()`; `shadow_2xs..2xl()` and `shadow(Vec<BoxShadow>)`.
- Visual: `opacity(f32)`, `bg(impl Into<Fill>)` — `rgb(0x171813)`, `rgba(...)`, `hsla(...)`; `visible()`.
- Text: `text_color`, `text_size(px(20.))`, `line_height(...)`, `font_family(impl Into<SharedString>)` (inherits to children), `font_weight(FontWeight)`, `italic()`, `text_align`/`text_left()/text_center()/text_right()`, `truncate()`.
- **No letter-spacing/tracking anywhere.** **No `.z_index()`** — paint order = element order; `Window::paint_layer(bounds, |window| ...)` to paint above.
- Positioning: `absolute()`, `relative()`, `top/left/bottom/right/inset(...)` + t-shirt variants.
- Helpers: `cursor_pointer()`, `hover(|style| style...)`, `group(...)`, `group_hover(name, f)`, `occlude()`, `key_context(...)`, `track_focus(&FocusHandle)`.
- Serif: `.font_family("Georgia")` resolves via fontconfig at runtime; fallback chain for missing fonts UNVERIFIED (only installed families resolve).

## 5. Events, focus, actions

```rust
// StatefulInteractiveElement (requires .id()):
fn on_click(self, listener: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static) -> Self;
fn on_hover(self, listener: impl Fn(&bool, &mut Window, &mut App) + 'static) -> Self;
fn overflow_y_scroll(self) -> Self;
fn tooltip(self, build_tooltip: impl Fn(&mut Window, &mut App) -> AnyView + 'static) -> Self;

// InteractiveElement (no .id() needed):
fn on_mouse_down(self, button: MouseButton, listener: impl Fn(&MouseDownEvent, &mut Window, &mut App) + 'static) -> Self;
fn on_key_down(self, listener: impl Fn(&KeyDownEvent, &mut Window, &mut App) + 'static) -> Self;
fn on_action<A: Action>(self, listener: impl Fn(&A, &mut Window, &mut App) + 'static) -> Self;
fn id(self, id: impl Into<ElementId>) -> Stateful<Self>;
fn track_focus(self, focus_handle: &FocusHandle) -> Self;
fn key_context<C, E>(self, key_context: C) -> Self where C: TryInto<KeyContext, Error = E>;

// Focus
cx.focus_handle() -> FocusHandle   // on App via deref
impl Focusable for X { fn focus_handle(&self, cx: &App) -> FocusHandle { self.focus.clone() } }
// handle.focus(&mut Window), .is_focused(&Window)

// Actions
actions!(starling, [ToggleRecording, ShowSettings]);
cx.bind_keys([KeyBinding::new("secondary-shift-space", ToggleRecording, None)]);
// keystroke syntax "[secondary-][ctrl-][alt-][shift-][cmd-][fn-]key[->key_char]"
// secondary = cmd on macOS, ctrl elsewhere. Third arg = key-context string or None (matches anywhere).
cx.on_action::<A: Action>(impl Fn(&A, &mut App));   // app-level
element.on_action(cx.listener(|this, action: &ToggleRecording, window, cx| ...)); // needs track_focus subtree
```

`ClickEvent::Mouse(MouseClickEvent) | Keyboard(KeyboardClickEvent)` with `.modifiers()/.position()/.click_count()/.is_keyboard()`. `KeyDownEvent { keystroke: Keystroke, is_held: bool }`; `Keystroke { modifiers, key: String, key_char: Option<String> }`.

## 6. Built-in elements

- Text: `.child("hi")` / `.child(format!(...))`; `StyledText::new(...).with_highlights(...)` / `.with_runs(Vec<TextRun>)`.
- **Text input NOT shipped in 0.2.2.** Build one: `div().id(...).track_focus(&focus)`, implement `InputHandler`/`EntityInputHandler`, `Window::handle_input(&mut self, &FocusHandle, impl InputHandler, &App)`, key events via `.on_key_down`.
- SVG: `svg().path("icons/foo.svg")` + your `AssetSource`; color via `.text_color(...)` (Svg implements Styled). AssetSource:

```rust
pub trait AssetSource: 'static + Send + Sync {
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>>;
    fn list(&self, path: &str) -> Result<Vec<SharedString>>;
}
Application::new().with_assets(Assets).run(|cx| { ... });
```

- canvas (custom paint):

```rust
pub fn canvas<T>(
    prepaint: impl 'static + FnOnce(Bounds<Pixels>, &mut Window, &mut App) -> T,
    paint:    impl 'static + FnOnce(Bounds<Pixels>, T, &mut Window, &mut App),
) -> Canvas<T>
// window.paint_quad(quad), window.paint_path(path) inside paint
```

For waveform bars, plain divs are simpler than canvas.

- UniformList (virtualized):

```rust
pub fn uniform_list<R>(id: impl Into<ElementId>, item_count: usize,
    f: impl 'static + Fn(Range<usize>, &mut Window, &mut App) -> Vec<R>) -> UniformList
// .track_scroll(&UniformListScrollHandle), .with_width_from_item(...)
```

- `img(source)` exists (AssetSource paths or Arc<RenderImage>).

## 7. Utility APIs

```rust
cx.write_to_clipboard(ClipboardItem::new_string(s.into()));
cx.read_from_clipboard() -> Option<ClipboardItem>;   // .text() -> Option<String>
cx.open_url("https://example.com");

pub struct PathPromptOptions { pub files: bool, pub directories: bool, pub multiple: bool, pub prompt: Option<SharedString> }
let rx = cx.prompt_for_paths(PathPromptOptions { files: true, directories: false, multiple: false, prompt: None });
// -> Receiver<Result<Option<Vec<PathBuf>>>>, await in cx.spawn

// Animation
pub struct Animation { pub duration: Duration, pub oneshot: bool, pub easing: Rc<dyn Fn(f32) -> f32> }
Animation::new(Duration::from_millis(1000)).repeat().with_easing(pulsating_between(0.4, 1.0));
// AnimationExt (blanket for IntoElement):
el.with_animation(id, animation, |el, delta| el.opacity(delta))
// easings: ease_in_out(f32), pulsating_between(min,max), linear, quadratic, bounce, ease_out_quint, phi

// UI tick: with_animation loop, or window.on_next_frame re-arming itself / window.request_animation_frame(),
// or Timer::after loop + cx.notify().
```

## 8. Linux (Wayland/X11) gotchas in 0.2.2

- Build deps: `libfontconfig-dev libwayland-dev libx11-xcb-dev libxkbcommon-x11-dev`; runtime Vulkan loader. Backends: x11rb 0.13 + wayland-client 0.31, both in default features.
- `TitlebarOptions.appears_transparent` is macOS/Windows only; on Linux use `window_decorations: WindowDecorations::Server` (normal WM titlebar — closest parity with Electron's frame) or `Client` (CSD, you draw it; Wayland-only).
- Fonts via zed's font-kit fork + fontconfig (dlopen); families resolve only when installed.
- No declared MSRV; needs recent stable (public APIs use edition-2024 `AsyncFnOnce`).

## UNVERIFIED — check `~/.cargo/registry/src/*/gpui-0.2.2/` before relying

- `point(x, y)` free fn; `rgb()` exact signature; `MouseButton` variants; `WindowKind` variants; `WindowOptions::default()` per-field values.
- Per-side border helpers (`border_t_1`, `border_b`, ...); `.child()/.children()` signatures; `items_center`/`justify_between` siblings.
- `App::observe`/`App::subscribe` variants; `AsyncApp::background_spawn`; `Receiver` future details for prompt_for_paths; `AssetSource` Result alias.
- Styled-level font-fallback builder / `FontFallbacks` constructor; Linux fallback chain for missing families.
- `fill(bounds, color)` / PaintQuad helper signatures.

## Appendix: verified-as-far-as-possible main.rs for 0.2.2

```rust
use gpui::{
    actions, div, px, rgb, size, App, Application, Bounds, ClickEvent, Context, FocusHandle,
    Focusable, KeyBinding, Render, TitlebarOptions, Timer, Window, WindowBounds, WindowOptions,
};
use std::time::Duration;

actions!(hello, [ToggleRecording]);

struct Hello { counter: usize, focus: FocusHandle }

impl Focusable for Hello {
    fn focus_handle(&self, _cx: &App) -> FocusHandle { self.focus.clone() }
}

impl Render for Hello {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("root")
            .track_focus(&self.focus)
            .key_context("Main")
            .on_action(cx.listener(|this, _: &ToggleRecording, _window, cx| { this.counter += 1; cx.notify(); }))
            .size_full().flex().flex_col().items_center().justify_center().gap_2()
            .bg(rgb(0x171813)).text_color(rgb(0xe8e5dc)).text_size(px(20.)).font_family("Georgia")
            .child(format!("count: {}", self.counter))
            .child(
                div()
                    .id("inc").px(px(28.)).py_1().rounded(px(12.))
                    .bg(rgb(0x2e3140)).cursor_pointer()
                    .hover(|style| style.bg(rgb(0x3b3f52)))
                    .on_click(cx.listener(|this, _click: &ClickEvent, _window, cx| { this.counter += 1; cx.notify(); }))
                    .child("click me (or secondary-shift-space)"),
            )
    }
}

fn main() {
    Application::new().run(|cx: &mut App| {
        cx.bind_keys([KeyBinding::new("secondary-shift-space", ToggleRecording, None)]);
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::centered(size(px(1180.), px(760.)), cx)),
            window_min_size: Some(size(px(920.), px(620.))),
            titlebar: Some(TitlebarOptions { title: Some("Starling".into()), appears_transparent: true, traffic_light_position: None }),
            focus: true,
            ..Default::default()
        };
        cx.open_window(options, |_window, cx| cx.new(|cx| Hello { counter: 0, focus: cx.focus_handle() })).unwrap();
    });
}
