//! Per-connection rate limiting (E17 §1 Mode B: "message size/rate
//! limits"). A sliding-window counter over frame arrivals: at most `max`
//! client→host frames within any `per` window. Over the limit the
//! connection is closed with `rate_limited` — a flooding client is
//! disconnected, not queued.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// The default command budget: 512 frames per second per connection.
/// A UI issues commands at human speed and polls snapshots on user
/// interaction; anything sustained near this rate is a runaway loop, and
/// 512 keeps honest bursts (reconnect + resnapshot + replay) far from the
/// ceiling.
pub const DEFAULT_RATE_MAX: u32 = 512;
pub const DEFAULT_RATE_WINDOW: Duration = Duration::from_secs(1);

/// The maximum number of simultaneous client connections. The design has
/// two clients (GPUI and the Electron comparison adapter); 16 leaves room
/// for tooling and a stuck reconnect without opening an unbounded fd set.
pub const DEFAULT_MAX_CONNECTIONS: usize = 16;

/// The per-connection outbound event queue depth before the host decides
/// the client stopped reading (see `SlowConsumer` in [`crate::frame`]).
pub const DEFAULT_OUTBOUND_CAPACITY: usize = 1024;

/// A rate limit: `max` frames per `per` window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateLimit {
    pub max: u32,
    pub per: Duration,
}

impl Default for RateLimit {
    fn default() -> Self {
        RateLimit {
            max: DEFAULT_RATE_MAX,
            per: DEFAULT_RATE_WINDOW,
        }
    }
}

impl RateLimit {
    pub fn new(max: u32, per: Duration) -> RateLimit {
        RateLimit { max, per }
    }
}

/// Sliding-window admission for one connection.
#[derive(Debug)]
pub struct SlidingWindow {
    limit: RateLimit,
    arrivals: VecDeque<Instant>,
}

impl SlidingWindow {
    pub fn new(limit: RateLimit) -> SlidingWindow {
        SlidingWindow {
            limit,
            arrivals: VecDeque::new(),
        }
    }

    /// Records a frame arrival and reports whether it was within budget.
    /// Callers pass `now` so tests (and the host's single clock) decide
    /// time, not the limiter.
    pub fn allow(&mut self, now: Instant) -> bool {
        let cutoff = now.checked_sub(self.limit.per);
        match cutoff {
            Some(cutoff) => {
                while self
                    .arrivals
                    .front()
                    .is_some_and(|arrival| *arrival < cutoff)
                {
                    self.arrivals.pop_front();
                }
            }
            // The window predates the monotonic clock's zero: every
            // arrival this limiter has ever counted is younger than the
            // window, so there is nothing to expire.
            None => {}
        }
        if self.arrivals.len() >= self.limit.max as usize {
            // Still record the violation's arrival: the connection is
            // closing anyway, and the count is diagnostic.
            self.arrivals.push_back(now);
            return false;
        }
        self.arrivals.push_back(now);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_up_to_max_within_the_window() {
        let mut window = SlidingWindow::new(RateLimit::new(3, Duration::from_secs(1)));
        let t0 = Instant::now();
        assert!(window.allow(t0));
        assert!(window.allow(t0 + Duration::from_millis(10)));
        assert!(window.allow(t0 + Duration::from_millis(20)));
        assert!(
            !window.allow(t0 + Duration::from_millis(30)),
            "the fourth frame inside one window is over budget"
        );
    }

    #[test]
    fn old_arrivals_expire_and_budget_returns() {
        let mut window = SlidingWindow::new(RateLimit::new(2, Duration::from_millis(100)));
        let t0 = Instant::now();
        assert!(window.allow(t0));
        assert!(window.allow(t0 + Duration::from_millis(10)));
        assert!(!window.allow(t0 + Duration::from_millis(20)));
        // Past the window the early arrivals no longer count.
        assert!(window.allow(t0 + Duration::from_millis(130)));
        assert!(window.allow(t0 + Duration::from_millis(140)));
        assert!(!window.allow(t0 + Duration::from_millis(150)));
    }

    #[test]
    fn a_window_longer_than_clock_history_admits_freshly() {
        // `now - per` underflows only when the process just booted with a
        // huge window; the limiter must not wedge (clear-and-admit is the
        // conservative reading: nothing has provably expired, but nothing
        // has been counted either).
        let mut window = SlidingWindow::new(RateLimit::new(2, Duration::from_secs(10_000)));
        let t0 = Instant::now();
        assert!(window.allow(t0));
        assert!(window.allow(t0));
        assert!(!window.allow(t0));
    }
}
