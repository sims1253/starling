//! Bounded in-process channels (E17 §2: "all queues are bounded with explicit
//! rejection events").
//!
//! `std::sync::mpsc` is unbounded, which is exactly the G01 failure mode the
//! runtime exists to remove, and no queueing dependency is in
//! `starling-dictation`'s allowed set — so this is a small bounded MPMC
//! channel built from a `Mutex` + `VecDeque` + two `Condvar`s. Senders are
//! cloneable (fan-in); each receiver is single-owned per queue.
//!
//! Every producer in the runtime faces a real capacity: [`Sender::try_send`]
//! reports [`TrySendError::Full`] instead of growing, and
//! [`Sender::send_blocking`] applies backpressure instead of allocating.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// Why a [`Sender::try_send`] failed. The value is returned so callers never
/// lose ownership silently.
#[derive(Debug, PartialEq, Eq)]
pub enum TrySendError<T> {
    /// The queue is at capacity — the bounded rejection.
    Full(T),
    /// Every receiver was dropped.
    Closed(T),
}

/// A bounded queue's shared state. `closed` is checked under the mutex so a
/// `send_blocking` parked on `not_full` wakes when receivers go away.
struct Chan<T> {
    queue: Mutex<ChanState<T>>,
    not_empty: Condvar,
    not_full: Condvar,
}

struct ChanState<T> {
    items: VecDeque<T>,
    capacity: usize,
    senders: usize,
    receivers: usize,
}

impl<T> ChanState<T> {
    fn is_full(&self) -> bool {
        self.items.len() >= self.capacity
    }
}

/// The producing end of a bounded channel. Cloning adds a sender; the queue
/// is closed when the last sender drops.
pub struct Sender<T> {
    chan: Arc<Chan<T>>,
}

/// The consuming end of a bounded channel. The queue reports
/// [`RecvError::Closed`] once the last sender drops and the queue drains.
pub struct Receiver<T> {
    chan: Arc<Chan<T>>,
}

/// Why a [`Receiver::recv`] failed.
#[derive(Debug, PartialEq, Eq)]
pub enum RecvError {
    /// Every sender was dropped and the queue is empty.
    Closed,
    /// No item was ready within the timeout.
    Timeout,
}

/// Creates a bounded channel holding at most `capacity` items.
///
/// # Panics
///
/// Panics if `capacity` is zero — a zero-capacity queue can never transfer
/// anything and would deadlock every sender; refusing it at construction is
/// cheaper than diagnosing the hang later.
pub fn bounded<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    assert!(capacity > 0, "bounded channel capacity must be > 0");
    let chan = Arc::new(Chan {
        queue: Mutex::new(ChanState {
            items: VecDeque::with_capacity(capacity),
            capacity,
            senders: 1,
            receivers: 1,
        }),
        not_empty: Condvar::new(),
        not_full: Condvar::new(),
    });
    (
        Sender {
            chan: Arc::clone(&chan),
        },
        Receiver { chan },
    )
}

impl<T> Sender<T> {
    /// Enqueues `value` if the queue has room; never blocks.
    pub fn try_send(&self, value: T) -> Result<(), TrySendError<T>> {
        let mut state = self.chan.queue.lock().expect("channel mutex poisoned");
        if state.receivers == 0 {
            return Err(TrySendError::Closed(value));
        }
        if state.is_full() {
            return Err(TrySendError::Full(value));
        }
        state.items.push_back(value);
        drop(state);
        self.chan.not_empty.notify_one();
        Ok(())
    }

    /// Enqueues `value`, blocking while the queue is full (backpressure, not
    /// growth). Returns [`RecvError::Closed`] (not the value) once every
    /// receiver is gone, so long-running producers can shut down cleanly.
    pub fn send_blocking(&self, value: T) -> Result<(), RecvError> {
        let mut state = self.chan.queue.lock().expect("channel mutex poisoned");
        loop {
            if state.receivers == 0 {
                return Err(RecvError::Closed);
            }
            if !state.is_full() {
                state.items.push_back(value);
                drop(state);
                self.chan.not_empty.notify_one();
                return Ok(());
            }
            let (guard, _) = self
                .chan
                .not_full
                .wait_timeout(state, Duration::from_millis(500))
                .expect("channel mutex poisoned");
            state = guard;
        }
    }

    /// Whether a receiver is still connected.
    pub fn is_closed(&self) -> bool {
        self.chan
            .queue
            .lock()
            .expect("channel mutex poisoned")
            .receivers
            == 0
    }
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        let mut state = self.chan.queue.lock().expect("channel mutex poisoned");
        state.senders += 1;
        drop(state);
        Sender {
            chan: Arc::clone(&self.chan),
        }
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        let mut state = self.chan.queue.lock().expect("channel mutex poisoned");
        state.senders -= 1;
        let senders_left = state.senders;
        drop(state);
        if senders_left == 0 {
            // Wake any blocked receiver: the queue can only drain from here.
            self.chan.not_empty.notify_all();
        }
    }
}

impl<T> Receiver<T> {
    /// Blocks until an item is available or all senders drop.
    pub fn recv(&self) -> Result<T, RecvError> {
        let mut state = self.chan.queue.lock().expect("channel mutex poisoned");
        loop {
            if let Some(item) = state.items.pop_front() {
                drop(state);
                self.chan.not_full.notify_one();
                return Ok(item);
            }
            if state.senders == 0 {
                return Err(RecvError::Closed);
            }
            state = self
                .chan
                .not_empty
                .wait(state)
                .expect("channel mutex poisoned");
        }
    }

    /// Like [`Receiver::recv`] with a deadline. [`RecvError::Timeout`] is a
    /// normal result for polls (the capture actor's progress ticker relies
    /// on it), not an error condition.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<T, RecvError> {
        let deadline = std::time::Instant::now() + timeout;
        let mut state = self.chan.queue.lock().expect("channel mutex poisoned");
        loop {
            if let Some(item) = state.items.pop_front() {
                drop(state);
                self.chan.not_full.notify_one();
                return Ok(item);
            }
            if state.senders == 0 {
                return Err(RecvError::Closed);
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return Err(RecvError::Timeout);
            }
            let (guard, timed_out) = self
                .chan
                .not_empty
                .wait_timeout(state, remaining)
                .expect("channel mutex poisoned");
            state = guard;
            if timed_out.timed_out() && state.items.is_empty() {
                return Err(RecvError::Timeout);
            }
        }
    }

    /// Non-blocking poll.
    pub fn try_recv(&self) -> Result<T, RecvError> {
        let mut state = self.chan.queue.lock().expect("channel mutex poisoned");
        match state.items.pop_front() {
            Some(item) => {
                drop(state);
                self.chan.not_full.notify_one();
                Ok(item)
            }
            None if state.senders == 0 => Err(RecvError::Closed),
            None => Err(RecvError::Timeout),
        }
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        let mut state = self.chan.queue.lock().expect("channel mutex poisoned");
        state.receivers = 0;
        state.items.clear();
        drop(state);
        // Wake any producer parked on a full queue: it must observe Closed.
        self.chan.not_full.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_beyond_capacity_and_recovers() {
        let (tx, rx) = bounded::<u32>(2);
        assert_eq!(tx.try_send(1), Ok(()));
        assert_eq!(tx.try_send(2), Ok(()));
        assert_eq!(tx.try_send(3), Err(TrySendError::Full(3)));
        assert_eq!(rx.recv(), Ok(1));
        assert_eq!(tx.try_send(3), Ok(()));
        assert_eq!(rx.recv(), Ok(2));
        assert_eq!(rx.recv(), Ok(3));
    }

    #[test]
    fn recv_times_out_when_idle() {
        let (tx, rx) = bounded::<u32>(2);
        let _ = &tx;
        assert_eq!(
            rx.recv_timeout(Duration::from_millis(20)),
            Err(RecvError::Timeout)
        );
    }

    #[test]
    fn closed_when_last_sender_drops() {
        let (tx, rx) = bounded::<u32>(2);
        let tx2 = tx.clone();
        assert_eq!(tx.try_send(7), Ok(()));
        drop(tx);
        assert!(!tx2.is_closed());
        drop(tx2);
        assert_eq!(rx.recv(), Ok(7));
        assert_eq!(rx.recv(), Err(RecvError::Closed));
    }

    #[test]
    fn send_blocking_reports_closed_receivers() {
        let (tx, rx) = bounded::<u32>(1);
        drop(rx);
        assert_eq!(tx.send_blocking(9), Err(RecvError::Closed));
        assert_eq!(tx.try_send(9), Err(TrySendError::Closed(9)));
    }

    #[test]
    fn blocking_send_applies_backpressure_then_delivers() {
        let (tx, rx) = bounded::<u32>(1);
        assert_eq!(tx.try_send(1), Ok(()));
        let producer = std::thread::spawn(move || {
            // Parks until the consumer makes room.
            tx.send_blocking(2).expect("send_blocking");
        });
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(rx.recv(), Ok(1));
        producer.join().expect("producer thread");
        assert_eq!(rx.recv(), Ok(2));
    }
}
