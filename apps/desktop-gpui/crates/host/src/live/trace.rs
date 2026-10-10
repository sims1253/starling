//! The take's stream timeline (#226).

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError};
use std::sync::Arc;
use std::time::Instant;

use serde_json::{json, Value};

use super::stream::Partial;

/// Trace lines queued for the writer before new ones are dropped (and
/// counted): a stalled trace sink must not hold up the UI or the stream.
const TRACE_BACKLOG: usize = 1024;

/// The take's stream timeline (#226), on when `STARLING_STREAM_TRACE` is
/// set: `1` or `stderr` writes to stderr, anything else is a file the
/// lines are appended to. One JSON object per line, tagged with the take
/// and `ms` since it started: `start` (with the wall clock), every server frame (`partial`
/// with the server's `covered_s`/`audio_s`, `final` with its stop path),
/// every preview the UI applied (`display`), `stream_failed`, `reconnect`,
/// `replay_released` and `stop`. Partial age at display is
/// `display.ms - 1000 * covered_s`, measured from the take's start rather
/// than the microphone's.
///
/// Lines are written by the take's own writer thread: the UI thread and
/// the stream's reader only queue them, so a slow or stalled sink cannot
/// freeze the window. A full queue drops lines; the next line written
/// carries how many (`dropped`).
pub struct StreamTrace {
    /// Tags every line: a take's `final` can land after the next take
    /// started.
    take: String,
    started: Instant,
    lines: SyncSender<String>,
    dropped: AtomicU64,
}

impl StreamTrace {
    pub fn from_env() -> Option<Arc<StreamTrace>> {
        let target = std::env::var("STARLING_STREAM_TRACE").ok()?;
        let out: Box<dyn Write + Send> = match target.trim() {
            "" | "0" => return None,
            "1" | "stderr" => Box::new(std::io::stderr()),
            path => match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                Ok(file) => Box::new(file),
                Err(err) => {
                    eprintln!("STARLING_STREAM_TRACE: cannot open {path}: {err}");
                    return None;
                }
            },
        };
        // Wall time too, to line the take up with outside events.
        let unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_millis() as u64);
        static TAKES: AtomicU64 = AtomicU64::new(0);
        let take = format!(
            "{}-{}",
            std::process::id(),
            TAKES.fetch_add(1, Ordering::Relaxed)
        );
        // No writer, no trace: saying so on stderr from the UI thread
        // could wait behind an earlier take's stalled stderr writer.
        let trace = StreamTrace::writing_to(take, out, TRACE_BACKLOG).ok()?;
        trace.log("start", json!({ "unix_ms": unix_ms }));
        Some(Arc::new(trace))
    }

    /// A trace whose lines `out` receives on its own thread, at most
    /// `backlog` of them waiting. The thread ends once the trace and its
    /// queued lines are gone.
    fn writing_to(
        take: String,
        mut out: Box<dyn Write + Send>,
        backlog: usize,
    ) -> std::io::Result<StreamTrace> {
        let (lines, queued) = std::sync::mpsc::sync_channel::<String>(backlog);
        std::thread::Builder::new()
            .name("starling-stream-trace".into())
            .spawn(move || {
                for line in queued {
                    let _ = out.write_all(line.as_bytes());
                }
            })?;
        Ok(StreamTrace {
            take,
            started: Instant::now(),
            lines,
            dropped: AtomicU64::new(0),
        })
    }

    #[cfg(test)]
    pub fn discard() -> StreamTrace {
        StreamTrace::writing_to("test".into(), Box::new(std::io::sink()), TRACE_BACKLOG)
            .expect("a trace writer thread")
    }

    pub fn log(&self, event: &str, mut fields: Value) {
        let ms = (self.started.elapsed().as_secs_f64() * 10_000.0).round() / 10.0;
        if let Value::Object(map) = &mut fields {
            map.insert("ev".into(), event.into());
            map.insert("take".into(), self.take.clone().into());
            map.insert("ms".into(), ms.into());
        }
        let dropped = self.dropped.swap(0, Ordering::Relaxed);
        if dropped > 0 {
            if let Value::Object(map) = &mut fields {
                map.insert("dropped".into(), dropped.into());
            }
        }
        // One write per line: every take appends to the same file through
        // its own handle, and a take's final can land during the next take.
        let line = format!("{fields}\n");
        match self.lines.try_send(line) {
            Ok(()) => {}
            // This line and the ones it was to report are lost.
            Err(TrySendError::Full(_)) => {
                self.dropped.fetch_add(dropped + 1, Ordering::Relaxed);
            }
            Err(TrySendError::Disconnected(_)) => {}
        }
    }

    /// A text frame from the server.
    pub fn received(&self, frame: &str) {
        let Ok(payload) = serde_json::from_str::<Value>(frame) else {
            return;
        };
        let trace = payload.get("trace");
        let field = |key: &str| trace.and_then(|trace| trace.get(key)).cloned();
        let words = payload
            .get("text")
            .and_then(Value::as_str)
            .map(|text| text.split_ascii_whitespace().count());
        match payload.get("type").and_then(Value::as_str) {
            Some("partial") => self.log(
                "partial",
                json!({
                    "words": words,
                    "stable_words": payload.get("stable_words"),
                    "covered_s": field("covered_s"),
                    "audio_s": field("audio_s"),
                }),
            ),
            Some("final") => {
                // The stop's own calls, with their audio spans, show
                // whether the stop decoded only the tail.
                let stop = field("stop");
                let stop_t0 = stop
                    .as_ref()
                    .and_then(|stop| stop.get("t0_ms"))
                    .and_then(Value::as_f64);
                let stop_calls: Vec<Value> = field("calls")
                    .and_then(|calls| calls.as_array().cloned())
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|call| {
                        let t0 = call.get("t0_ms").and_then(Value::as_f64);
                        matches!((t0, stop_t0), (Some(t0), Some(stop_t0)) if t0 >= stop_t0)
                    })
                    .collect();
                self.log(
                    "final",
                    json!({
                        "words": words,
                        "audio_s": field("audio_s"),
                        "stop": stop,
                        "stop_calls": stop_calls,
                        "by_kind": field("by_kind"),
                        "totals": field("totals"),
                    }),
                )
            }
            Some("error") => self.log("error", json!({ "message": payload.get("message") })),
            _ => {}
        }
    }

    /// A preview the UI applied.
    pub fn displayed(&self, partial: &Partial) {
        self.log(
            "display",
            json!({
                "words": partial.text.split_ascii_whitespace().count(),
                "covered_s": partial.covered_s,
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::time::Duration;

    #[test]
    fn a_stalled_trace_sink_drops_lines_instead_of_blocking() {
        /// A sink that blocks every write until the gate opens, and keeps
        /// what it was given.
        struct Gated(Arc<(Mutex<bool>, std::sync::Condvar)>, Arc<Mutex<Vec<u8>>>);
        impl Write for Gated {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                let (open, opened) = &*self.0;
                let mut open = open.lock().unwrap();
                while !*open {
                    open = opened.wait(open).unwrap();
                }
                self.1.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        /// Opens the gate when dropped, so a failing assertion cannot
        /// leave the writer blocked.
        struct Opens(Arc<(Mutex<bool>, std::sync::Condvar)>);
        impl Drop for Opens {
            fn drop(&mut self) {
                *self.0.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
                self.0.1.notify_all();
            }
        }
        let gate = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
        let opens = Opens(Arc::clone(&gate));
        let written = Arc::new(Mutex::new(Vec::new()));
        let trace = Arc::new(
            StreamTrace::writing_to(
                "test".into(),
                Box::new(Gated(Arc::clone(&gate), Arc::clone(&written))),
                4,
            )
            .unwrap(),
        );
        // With the sink stalled, 100 lines are logged without waiting on
        // it (a blocking log would never finish while the gate is shut).
        let (done, finished) = std::sync::mpsc::channel();
        {
            let trace = Arc::clone(&trace);
            std::thread::spawn(move || {
                for line in 0..100 {
                    trace.log("display", json!({ "line": line }));
                }
                let _ = done.send(());
            });
        }
        assert!(
            finished.recv_timeout(Duration::from_secs(10)).is_ok(),
            "logging waited on the sink"
        );
        drop(opens);
        // The queue may still be full for a moment: a stop that is dropped
        // too is counted on the next one.
        let mut logged = 100;
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            trace.log("stop", json!({}));
            logged += 1;
            if trace.dropped.load(Ordering::Relaxed) == 0 {
                break;
            }
            assert!(Instant::now() < deadline, "the writer never drained");
            std::thread::sleep(Duration::from_millis(5));
        }
        drop(trace);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !String::from_utf8_lossy(&written.lock().unwrap()).contains("\"stop\"")
            && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(5));
        }
        let text = String::from_utf8(written.lock().unwrap().clone()).unwrap();
        let lines: Vec<Value> = text.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
        let stop = lines.last().unwrap();
        assert_eq!(stop["ev"], "stop");
        // Every line is either written or counted on a later one.
        let dropped: u64 = lines.iter().filter_map(|line| line["dropped"].as_u64()).sum();
        assert!(dropped > 0);
        assert_eq!(lines.len() as u64 + dropped, logged);
    }
}
