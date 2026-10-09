//! A whole-percent horizontal slider (the playback "Lower volume" level).
//! Clicking the track jumps there; a drag keeps tracking outside the
//! element until the button is released.

use gpui::{
    div, fill, point, prelude::*, px, relative, size, App, Bounds, Context, CursorStyle,
    DispatchPhase, Element, GlobalElementId, IntoElement, LayoutId, MouseButton, MouseDownEvent,
    MouseMoveEvent, MouseUpEvent, Pixels, Style, Window,
};

use crate::theme;

pub struct LevelSlider {
    value: u8,
    dragging: bool,
    /// Captured in [`SliderTrack`]'s prepaint; mouse x maps against it.
    track_bounds: Option<Bounds<Pixels>>,
}

impl LevelSlider {
    pub fn new(value: u8) -> Self {
        LevelSlider {
            value,
            dragging: false,
            track_bounds: None,
        }
    }

    pub fn value(&self) -> u8 {
        self.value
    }

    pub fn set_value(&mut self, value: u8, cx: &mut Context<Self>) {
        if self.value != value {
            self.value = value;
            cx.notify();
        }
    }

    fn set_from_x(&mut self, x: Pixels, cx: &mut Context<Self>) {
        let Some(bounds) = self.track_bounds else {
            return;
        };
        if bounds.size.width <= Pixels::ZERO {
            return;
        }
        let fraction = ((x - bounds.origin.x) / bounds.size.width).clamp(0.0, 1.0);
        self.set_value((fraction * 100.0).round() as u8, cx);
    }

    fn on_mouse_down(&mut self, event: &MouseDownEvent, _: &mut Window, cx: &mut Context<Self>) {
        self.dragging = true;
        self.set_from_x(event.position.x, cx);
        cx.notify();
    }
}

impl Render for LevelSlider {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("level-slider")
            .cursor(CursorStyle::ResizeLeftRight)
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_mouse_down))
            .w_full()
            // Taller than the 4px track so a click lands reliably.
            .h(px(26.))
            .child(SliderTrack {
                slider: cx.entity(),
                value: self.value,
                dragging: self.dragging,
            })
    }
}

/// A 4px rail, inked up to the value, with a 12px thumb. A custom element
/// so it can hand its bounds to the slider.
struct SliderTrack {
    slider: gpui::Entity<LevelSlider>,
    value: u8,
    dragging: bool,
}

impl IntoElement for SliderTrack {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for SliderTrack {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<gpui::ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut style = Style::default();
        style.size.width = relative(1.).into();
        style.size.height = px(26.).into();
        (window.request_layout(style, [], cx), ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        self.slider.update(cx, |slider, _| {
            slider.track_bounds = Some(bounds);
        });
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        _cx: &mut App,
    ) {
        let track_height = px(4.);
        let track_top = bounds.origin.y + (bounds.size.height - track_height) / 2.;
        let thumb = px(12.);
        let fill_width = bounds.size.width * (self.value as f32 / 100.);
        window.paint_quad(fill(
            gpui::Bounds::new(
                point(bounds.origin.x, track_top),
                size(bounds.size.width, track_height),
            ),
            theme::SETTINGS_LINE,
        ));
        window.paint_quad(fill(
            gpui::Bounds::new(
                point(bounds.origin.x, track_top),
                size(fill_width, track_height),
            ),
            theme::SETTINGS_INK,
        ));
        window.paint_quad(fill(
            gpui::Bounds::new(
                point(
                    (bounds.origin.x + fill_width - thumb / 2.).max(bounds.origin.x),
                    bounds.origin.y + (bounds.size.height - thumb) / 2.,
                ),
                size(thumb, thumb),
            ),
            theme::SETTINGS_INK,
        ));
        // Window-level, so a drag keeps tracking once the pointer leaves
        // the element.
        if self.dragging {
            let slider = self.slider.clone();
            window.on_mouse_event(move |event: &MouseMoveEvent, phase, _, cx| {
                if phase == DispatchPhase::Bubble {
                    slider.update(cx, |slider, cx| slider.set_from_x(event.position.x, cx));
                }
            });
            let slider = self.slider.clone();
            window.on_mouse_event(move |event: &MouseUpEvent, phase, _, cx| {
                if phase == DispatchPhase::Bubble && event.button == MouseButton::Left {
                    slider.update(cx, |slider, cx| {
                        slider.dragging = false;
                        cx.notify();
                    });
                }
            });
        }
    }
}
