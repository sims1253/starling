//! Live stream pumping, app side (#357): the worker itself lives in the
//! runtime host crate ([`starling_runtime_host::live`]); this applies the
//! newest preview to the UI.

use std::sync::Arc;

use gpui::Context;

use crate::app::StarlingApp;
use starling_runtime_host::live::pump::{Handoff, Live, StreamPump};
use starling_runtime_host::live::stream::{LiveStream, Partial, StreamOptions};
use starling_runtime_host::live::trace::StreamTrace;

impl StarlingApp {
    /// Opens the take's live stream at `endpoint` and starts its worker on
    /// the take's `feed` (its audio as the recording service sends it,
    /// #220) at the device `rate`, with the cadence from the settings. An
    /// error is the reason no live text will show; the take records
    /// regardless.
    pub(crate) fn start_stream_pump(
        &mut self,
        feed: Arc<crate::host_link::TakeFeed>,
        rate: u32,
        endpoint: &str,
        cx: &mut Context<Self>,
    ) -> Result<(), String> {
        let trace = StreamTrace::from_env();
        let options = StreamOptions {
            cadence: self.live_preview.effective(),
            trace: trace.clone(),
        };
        let stream = LiveStream::start(endpoint, &options)?;
        let endpoint = endpoint.to_string();
        let (pump, live) = StreamPump::start(
            Box::new(feed),
            rate,
            stream,
            Box::new(move || LiveStream::start(&endpoint, &options)),
            trace.clone(),
        )?;
        let Live {
            mut previews,
            mut degradation,
        } = live;
        self.stream_generation = self.stream_generation.wrapping_add(1);
        let generation = self.stream_generation;
        self.stream_pump = Some(pump);
        self.stream_trace = trace;
        // A foreground task, not a render callback: it runs while the
        // main window is minimized or covered. It ends when the worker
        // does (the senders drop with it).
        cx.spawn(async move |this, cx| {
            loop {
                let applied = tokio::select! {
                    biased;
                    changed = previews.changed() => {
                        if changed.is_err() {
                            break;
                        }
                        let Some(partial) = previews.borrow_and_update().clone() else {
                            continue;
                        };
                        this.update(cx, |app, cx| app.show_stream_partial(generation, partial, cx))
                    }
                    changed = degradation.changed() => {
                        if changed.is_err() {
                            break;
                        }
                        let Some(reason) = degradation.borrow_and_update().clone() else {
                            continue;
                        };
                        this.update(cx, |app, cx| app.stream_degraded(generation, reason, cx))
                    }
                };
                if applied.is_err() {
                    break;
                }
            }
        })
        .detach();
        Ok(())
    }

    /// The newest preview of the running take: into the staging draft
    /// (which keeps the user's edits, #297), or the direct-mode line.
    fn show_stream_partial(&mut self, generation: u64, partial: Partial, cx: &mut Context<Self>) {
        // A preview that was in flight when its take stopped is never
        // shown: the final replaces the draft's live text.
        if generation != self.stream_generation || self.stream_pump.is_none() {
            return;
        }
        if let Some(trace) = self.stream_trace.as_ref() {
            trace.displayed(&partial);
        }
        if self.staging.is_some() {
            self.staging_partial(partial, cx);
        } else {
            self.live_partial = partial.text;
            cx.notify();
        }
    }

    /// Live text stopped for the rest of the running take: say why now.
    fn stream_degraded(&mut self, generation: u64, reason: String, cx: &mut Context<Self>) {
        if generation != self.stream_generation || self.stream_pump.is_none() {
            return;
        }
        self.stream_degradation = Some(reason);
        cx.notify();
    }

    /// Stops the take's worker and hands its audio and connection to the
    /// stop path; empty when the take had no stream.
    pub(crate) fn finish_stream_pump(&mut self) -> Handoff<LiveStream> {
        self.stream_trace = None;
        self.stream_pump
            .take()
            .map(StreamPump::finish)
            .unwrap_or_default()
    }
}

