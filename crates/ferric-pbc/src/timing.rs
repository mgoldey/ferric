//! Stage timers for the periodic builds and SCF (Performance plan item 0,
//! `reference/pbc/FINDINGS.md` "Performance plan (research) — 2026-09-25").
//!
//! Every later optimisation in that plan is stated as a RATIO of stage times,
//! so the stages must be measured, not estimated. The timers here are
//! observation only: they read two clocks at stage boundaries and never touch
//! an integral, a matrix or a loop order, so every energy is bit-identical
//! with or without them (`tests/pbc_timings.rs`). There is no per-integral
//! timing: the finest grain is one J/K/XC build call.
//!
//! * [`StageClock`] — wall clock plus process CPU time
//!   (`CLOCK_PROCESS_CPUTIME_ID`: every thread of the process, so under rayon
//!   `cpu_s / wall_s` is the effective parallelism; `None` off Unix).
//! * [`PbcTimings`] — an ordered list of LEAF stages (no stage contains
//!   another, so `Σ stages <= wall_s` holds by construction), the component's
//!   own total `wall_s`/`cpu_s`, and counters (triplets, pairs, G vectors,
//!   chunks, aux dropped) copied from what the builds already count.
//! * [`CallClock`] — an accumulator for the per-SCF-iteration builders
//!   (J, K, XC), which only get `&self`/`&mut self` of a borrowed source:
//!   atomics, so the source can stay `Clone` and `Sync`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Process CPU seconds (user + system, all threads), or `None` where the
/// clock is unavailable. An observability helper: it never fails a run.
pub fn process_cpu_seconds() -> Option<f64> {
    #[cfg(unix)]
    {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: `ts` is a valid, writable timespec; clock_gettime only
        // writes it and reports failure through the return code.
        let rc = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
        if rc == 0 {
            return Some(ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9);
        }
        None
    }
    #[cfg(not(unix))]
    {
        None
    }
}

/// A started stage: wall `Instant` and the process CPU time at start.
#[derive(Debug, Clone, Copy)]
pub struct StageClock {
    t0: Instant,
    cpu0: Option<f64>,
}

impl StageClock {
    /// Start timing now.
    pub fn start() -> Self {
        Self {
            t0: Instant::now(),
            cpu0: process_cpu_seconds(),
        }
    }

    /// `(wall seconds, CPU seconds)` since [`StageClock::start`].
    pub fn elapsed(&self) -> (f64, Option<f64>) {
        let wall = self.t0.elapsed().as_secs_f64();
        let cpu = match (self.cpu0, process_cpu_seconds()) {
            (Some(a), Some(b)) => Some((b - a).max(0.0)),
            _ => None,
        };
        (wall, cpu)
    }
}

/// One leaf stage: accumulated wall/CPU seconds over `calls` timed calls.
#[derive(Debug, Clone, PartialEq)]
pub struct StageTiming {
    pub name: &'static str,
    pub wall_s: f64,
    /// `None` when the process CPU clock is unavailable.
    pub cpu_s: Option<f64>,
    pub calls: u64,
}

fn add_opt(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x + y),
        (x, None) => x,
        (None, y) => y,
    }
}

/// Stage timings and counters of one component (hcore, an RS-GDF build, a
/// KS driver) or of a whole run (a caller merging components).
///
/// Invariant: `stages` are LEAVES (disjoint intervals of this component's
/// work), so `stage_wall_sum() <= wall_s` up to clock resolution once
/// `wall_s` is set by [`PbcTimings::finish`] around all of them. What is not
/// in a stage (small bookkeeping) is `wall_s − stage_wall_sum()`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PbcTimings {
    /// Total wall seconds of the component (0 until [`PbcTimings::finish`]).
    pub wall_s: f64,
    /// Total process CPU seconds of the component.
    pub cpu_s: Option<f64>,
    /// Leaf stages in first-recorded order.
    pub stages: Vec<StageTiming>,
    /// Counters in first-recorded order.
    pub counters: Vec<(&'static str, u64)>,
}

impl PbcTimings {
    /// Add `(wall, cpu, calls)` to stage `name` (created on first use).
    pub fn add(&mut self, name: &'static str, wall_s: f64, cpu_s: Option<f64>, calls: u64) {
        if let Some(s) = self.stages.iter_mut().find(|s| s.name == name) {
            s.wall_s += wall_s;
            s.cpu_s = add_opt(s.cpu_s, cpu_s);
            s.calls += calls;
        } else {
            self.stages.push(StageTiming {
                name,
                wall_s,
                cpu_s,
                calls,
            });
        }
    }

    /// Close a stage started with `clock` (one call).
    pub fn stop(&mut self, name: &'static str, clock: &StageClock) {
        let (w, c) = clock.elapsed();
        self.add(name, w, c, 1);
    }

    /// Add an already-accumulated stage (e.g. a [`CallClock`] snapshot).
    pub fn add_stage(&mut self, s: &StageTiming) {
        self.add(s.name, s.wall_s, s.cpu_s, s.calls);
    }

    /// Set counter `name` to `v` (overwrites).
    pub fn set_counter(&mut self, name: &'static str, v: u64) {
        if let Some(c) = self.counters.iter_mut().find(|c| c.0 == name) {
            c.1 = v;
        } else {
            self.counters.push((name, v));
        }
    }

    /// Set the component total from the clock started before its first stage.
    pub fn finish(&mut self, clock: &StageClock) {
        let (w, c) = clock.elapsed();
        self.wall_s = w;
        self.cpu_s = c;
    }

    /// Append `other`'s stages (summed by name) and counters (overwritten
    /// by name). Totals are NOT merged: the absorbing record's
    /// [`PbcTimings::finish`] measures its own span.
    pub fn absorb(&mut self, other: &PbcTimings) {
        for s in &other.stages {
            self.add_stage(s);
        }
        for &(n, v) in &other.counters {
            self.set_counter(n, v);
        }
    }

    /// Stage `name`, if recorded.
    pub fn stage(&self, name: &str) -> Option<&StageTiming> {
        self.stages.iter().find(|s| s.name == name)
    }

    /// Counter `name`, if recorded.
    pub fn counter(&self, name: &str) -> Option<u64> {
        self.counters.iter().find(|c| c.0 == name).map(|c| c.1)
    }

    /// `Σ stages wall_s`.
    pub fn stage_wall_sum(&self) -> f64 {
        self.stages.iter().map(|s| s.wall_s).sum()
    }
}

/// Accumulated time of a builder called once per SCF iteration (J, K, XC).
/// Interior-mutable (atomics) because the builders borrow their source
/// immutably; `Clone` copies the current totals.
#[derive(Debug, Default)]
pub struct CallClock {
    wall_ns: AtomicU64,
    cpu_ns: AtomicU64,
    calls: AtomicU64,
}

impl Clone for CallClock {
    fn clone(&self) -> Self {
        Self {
            wall_ns: AtomicU64::new(self.wall_ns.load(Ordering::Relaxed)),
            cpu_ns: AtomicU64::new(self.cpu_ns.load(Ordering::Relaxed)),
            calls: AtomicU64::new(self.calls.load(Ordering::Relaxed)),
        }
    }
}

impl CallClock {
    /// Run `f` and add its wall/CPU time and one call.
    pub fn time<T>(&self, f: impl FnOnce() -> T) -> T {
        let c = StageClock::start();
        let out = f();
        self.record(&c);
        out
    }

    /// Add the time since `clock` started as one call.
    pub fn record(&self, clock: &StageClock) {
        let (w, cpu) = clock.elapsed();
        self.wall_ns
            .fetch_add((w * 1e9).round() as u64, Ordering::Relaxed);
        if let Some(cpu) = cpu {
            self.cpu_ns
                .fetch_add((cpu * 1e9).round() as u64, Ordering::Relaxed);
        }
        self.calls.fetch_add(1, Ordering::Relaxed);
    }

    /// The totals so far as a stage named `name`.
    pub fn timing(&self, name: &'static str) -> StageTiming {
        StageTiming {
            name,
            wall_s: self.wall_ns.load(Ordering::Relaxed) as f64 * 1e-9,
            cpu_s: process_cpu_seconds().map(|_| self.cpu_ns.load(Ordering::Relaxed) as f64 * 1e-9),
            calls: self.calls.load(Ordering::Relaxed),
        }
    }

    /// Calls timed so far.
    pub fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_stages_sum_within_the_total_and_merge_by_name() {
        let total = StageClock::start();
        let mut t = PbcTimings::default();
        for _ in 0..2 {
            let c = StageClock::start();
            let _ = std::hint::black_box((0..10_000).map(|i| i as f64).sum::<f64>());
            t.stop("a", &c);
        }
        t.set_counter("n", 3);
        t.set_counter("n", 4);
        t.finish(&total);
        assert_eq!(t.stages.len(), 1);
        assert_eq!(t.stage("a").unwrap().calls, 2);
        assert_eq!(t.counter("n"), Some(4));
        assert!(t.stage_wall_sum() <= t.wall_s);
        let clock = CallClock::default();
        let v = clock.time(|| 7);
        assert_eq!((v, clock.calls()), (7, 1));
        assert!(clock.timing("x").wall_s >= 0.0);
    }
}
