//! The event side of the envelope: ids, timestamps, per-stream sequence
//! numbers, and the bounded fan-out to subscribers.
//!
//! Per-stream `seq` is strictly increasing (envelope.schema.json): a stream
//! is the correlation channel named by `corr`, or the direction stream
//! (`__commands__` / `__events__`) when absent. The runtime allocates
//! sequence numbers for **both** directions from one frontier per stream,
//! so a take's commands and events interleave monotonically the way the I0
//! fixture traces do (e.g. `take_77`: `mode.routeFrozen` seq 409, then
//! `capture.start` 410, `capture.started` 411, …).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use time::OffsetDateTime;

use crate::channel::{bounded, Receiver, RecvError, Sender, TrySendError};
use crate::protocol::Event;

/// A serialized event: the typed payload plus its envelope fields.
#[derive(Debug, Clone, PartialEq)]
pub struct EventMessage {
    pub id: String,
    pub ts: String,
    pub corr: Option<String>,
    pub seq: u64,
    pub event: Event,
}

impl EventMessage {
    /// The envelope's wire form.
    pub fn to_value(&self) -> serde_json::Value {
        let mut object = serde_json::Map::new();
        object.insert("v".into(), 1.into());
        object.insert("id".into(), self.id.clone().into());
        object.insert("ts".into(), self.ts.clone().into());
        if let Some(corr) = &self.corr {
            object.insert("corr".into(), corr.clone().into());
        }
        object.insert("seq".into(), self.seq.into());
        object.insert("type".into(), self.event.type_name().into());
        object.insert("payload".into(), self.event.payload_value());
        serde_json::Value::Object(object)
    }

    pub fn type_name(&self) -> &'static str {
        self.event.type_name()
    }
}

/// A pending command envelope traveling to a machine actor.
#[derive(Debug, Clone)]
pub struct CommandMessage {
    pub id: String,
    pub ts: String,
    pub corr: Option<String>,
    pub seq: Option<u64>,
    pub command: crate::protocol::Command,
}

/// RFC 3339 UTC timestamp with millisecond precision (`Z` form).
pub fn now_ts() -> String {
    OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string())
}

/// Short unique message id: `prefix_<12 hex>`.
pub fn new_id(prefix: &str) -> String {
    let uuid = uuid::Uuid::new_v4();
    format!("{}_{:x}", prefix, uuid.as_simple())
}

/// One frontier per correlation stream, shared by the command and event
/// directions so both interleave monotonically.
#[derive(Default)]
struct SeqFrontiers {
    next: HashMap<String, u64>,
}

/// Allocates sequence numbers and emits events to every subscriber.
pub struct EventBus {
    frontiers: Mutex<SeqFrontiers>,
    subscribers: Mutex<Vec<Sender<EventMessage>>>,
    capacity: usize,
}

impl EventBus {
    pub fn new(capacity: usize) -> EventBus {
        EventBus {
            frontiers: Mutex::new(SeqFrontiers::default()),
            subscribers: Mutex::new(Vec::new()),
            capacity,
        }
    }

    fn stream(corr: Option<&str>) -> String {
        corr.map(str::to_string)
            .unwrap_or_else(|| "__events__".to_string())
    }

    /// Allocates the next sequence number for `corr`'s stream (the command
    /// side uses this before handing an envelope to a machine).
    pub fn next_seq(&self, corr: Option<&str>) -> u64 {
        let mut frontiers = self.frontiers.lock().expect("seq frontier lock");
        let next = frontiers.next.entry(Self::stream(corr)).or_insert(0);
        *next += 1;
        *next
    }

    /// The last allocated sequence number for `corr`'s stream (0 when none).
    pub fn last_seq(&self, corr: Option<&str>) -> u64 {
        self.frontiers
            .lock()
            .expect("seq frontier lock")
            .next
            .get(&Self::stream(corr))
            .copied()
            .unwrap_or(0)
    }

    /// Adds a subscriber; returns its bounded receiving end. Subscribers
    /// that fall `capacity` behind apply backpressure to the machines (a
    /// dropped event would silently corrupt the projection, which is the
    /// failure mode this runtime exists to prevent).
    pub fn subscribe(&self) -> EventSub {
        let (sender, receiver) = bounded(self.capacity);
        self.subscribers
            .lock()
            .expect("subscriber lock")
            .push(sender);
        EventSub {
            receiver: Arc::new(receiver),
        }
    }

    /// Serializes and emits an event on `corr`'s stream. Blocks while any
    /// subscriber queue is full (backpressure), returns
    /// [`RecvError::Closed`] when every subscriber has gone away.
    pub fn emit(&self, event: Event, corr: Option<&str>) -> Result<EventMessage, RecvError> {
        let message = EventMessage {
            id: new_id("evt"),
            ts: now_ts(),
            corr: corr.map(str::to_string),
            seq: self.next_seq(corr),
            event,
        };
        let subscribers = self.subscribers.lock().expect("subscriber lock").clone();
        let mut closed = 0;
        for subscriber in &subscribers {
            // try_send first (the common case), then bounded backpressure.
            match subscriber.try_send(message.clone()) {
                Ok(()) => {}
                Err(TrySendError::Full(value)) => {
                    subscriber.send_blocking(value)?;
                }
                Err(TrySendError::Closed(_)) => closed += 1,
            }
        }
        if closed == subscribers.len() && !subscribers.is_empty() {
            return Err(RecvError::Closed);
        }
        Ok(message)
    }
}

/// A subscriber's bounded view of the runtime's event stream. The handle
/// is cheap to clone (the queue itself stays single-consumer).
#[derive(Clone)]
pub struct EventSub {
    receiver: Arc<Receiver<EventMessage>>,
}

impl EventSub {
    /// Blocks until the next event or the runtime closes.
    pub fn recv(&self) -> Result<EventMessage, RecvError> {
        self.receiver.recv()
    }

    /// Waits up to `timeout` for the next event ([`RecvError::Timeout`] on
    /// idle — the normal poll result, not an error state).
    pub fn recv_timeout(&self, timeout: Duration) -> Result<EventMessage, RecvError> {
        self.receiver.recv_timeout(timeout)
    }

    /// Non-blocking poll.
    pub fn try_recv(&self) -> Result<EventMessage, RecvError> {
        self.receiver.try_recv()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Event;

    #[test]
    fn seq_is_strictly_monotonic_per_stream() {
        let bus = EventBus::new(8);
        assert_eq!(bus.next_seq(Some("take_1")), 1);
        assert_eq!(bus.next_seq(Some("take_1")), 2);
        assert_eq!(bus.next_seq(Some("take_2")), 1);
        assert_eq!(bus.next_seq(None), 1);
        assert_eq!(bus.next_seq(Some("take_1")), 3);
    }

    #[test]
    fn emit_assigns_seq_and_reaches_subscribers() {
        let bus = EventBus::new(8);
        let sub = bus.subscribe();
        let msg = bus
            .emit(Event::JobsQueued, Some("job-1"))
            .expect("emit with subscriber");
        assert_eq!(msg.seq, 1);
        assert_eq!(msg.corr.as_deref(), Some("job-1"));
        assert_eq!(msg.type_name(), "jobs.queued");
        assert_eq!(sub.recv_timeout(Duration::from_millis(200)), Ok(msg));
    }

    #[test]
    fn emit_backpressures_when_subscriber_is_full() {
        let bus = Arc::new(EventBus::new(2));
        let _sub = bus.subscribe();
        assert!(bus.emit(Event::JobsQueued, None).is_ok());
        assert!(bus.emit(Event::JobsQueued, None).is_ok());
        // Third emit parks until the subscriber drains (or goes away).
        let parked_bus = Arc::clone(&bus);
        let parked = std::thread::spawn(move || parked_bus.emit(Event::JobsQueued, None));
        std::thread::sleep(Duration::from_millis(30));
        assert!(!parked.is_finished());
        drop(_sub); // dropping the subscriber releases the producer
        let _ = parked.join().expect("emit thread");
    }
}
