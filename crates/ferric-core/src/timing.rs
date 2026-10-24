//! Opt-in stage timing, shared across the workspace.
//!
//! Gated by the SAME `FERRIC_TIMING` env var the RPA/GW pipeline already used
//! (`ferric_rpa::timing` is now a re-export shim over this module), so one flag
//! turns on instrumentation everywhere rather than each crate inventing its own.
//!
//! Two shapes:
//!
//! - [`Stage`](crate::timing::Stage) — a one-shot scoped timer that prints `[timing] <label> <ms>` to
//!   stderr at `.end()`. Right for a stage that runs ONCE per job (the RPA
//!   eigensolve, a 3-index tensor build).
//! - [`PhaseAccumulator`](crate::timing::PhaseAccumulator) — sums repeated phases across many iterations and
//!   prints ONE table at the end. Right for the SCF loop, where a per-`.end()`
//!   print would emit `n_iter × n_phase` lines interleaved with the SCF
//!   convergence table and drown it.
//!
//! WHY THIS EXISTS AT ALL: a predecessor instrument (`FERRIC_PHASE_PROF`) was
//! used to produce the per-component serial/parallel split in
//! `wiki/perf-handoff-2026-07-25.md`, but it was never committed — it lived in
//! a scratch worktree and vanished with it, so that document's headline finding
//! (a ~38% serial fraction in RHF) became unreproducible. This module is
//! committed precisely so the next measurement does not start from zero.
//!
//! PLACEMENT RULE (load-bearing — violating it changes what you measure):
//! a timer must bracket a WHOLE-STAGE call and must never sit inside a rayon
//! `par_iter`/`into_par_iter` closure or a per-chunk inner loop. Timing inside
//! a parallel region adds a clock read per chunk (thousands of calls) and can
//! suppress the very parallelism under measurement.

use std::sync::OnceLock;
use std::time::Instant;

/// `FERRIC_TIMING` descriptor: opt-in stage timing (env-only debug toggle).
/// Routed through the shared `parse_toggle` so `FERRIC_TIMING=0` means OFF,
/// matching every other ferric toggle. Resolved once and cached in a
/// `OnceLock` since it gates a hot-path timer.
static TIMING: crate::config::ConfigVar<bool> = crate::config::ConfigVar {
    env_name: "FERRIC_TIMING",
    default: false,
    parse: crate::config::parse_toggle,
    validate: crate::config::accept_any,
};

/// Whether stage timing is enabled for this process.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| TIMING.toggle())
}

/// A scoped one-shot stage timer. Construct with `start`, call `end` when the
/// stage finishes. No-op when `FERRIC_TIMING` is unset (no clock read, no
/// allocation beyond the `&'static str` label).
#[derive(Debug)]
pub struct Stage {
    label: &'static str,
    t0: Option<Instant>,
}

impl Stage {
    #[inline]
    pub fn start(label: &'static str) -> Self {
        Stage {
            label,
            t0: if enabled() {
                Some(Instant::now())
            } else {
                None
            },
        }
    }

    #[inline]
    pub fn end(self) {
        if let Some(t0) = self.t0 {
            eprintln!(
                "[timing] {:<48} {:>9.1} ms",
                self.label,
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
    }
}

/// A single phase's running total inside a [`PhaseAccumulator`].
#[derive(Debug, Clone)]
struct PhaseTotal {
    label: &'static str,
    secs: f64,
    calls: u64,
}

/// Accumulates repeated phase timings across loop iterations and prints one
/// summary table on [`PhaseAccumulator::report`].
///
/// Phases are keyed by their `&'static str` label and reported in FIRST-SEEN
/// order, which for a loop body is source order — so the table reads top-to-
/// bottom like the loop it describes rather than being re-sorted by cost.
///
/// Entirely inert when `FERRIC_TIMING` is unset: [`PhaseAccumulator::time`]
/// runs the closure directly with no clock read, and `report` prints nothing.
#[derive(Debug, Default)]
pub struct PhaseAccumulator {
    phases: Vec<PhaseTotal>,
}

impl PhaseAccumulator {
    pub fn new() -> Self {
        Self { phases: Vec::new() }
    }

    /// Run `f`, adding its elapsed time to `label`'s running total. Returns
    /// `f`'s value untouched, so this wraps an existing call without changing
    /// the expression's type or its arithmetic.
    #[inline]
    pub fn time<T>(&mut self, label: &'static str, f: impl FnOnce() -> T) -> T {
        if !enabled() {
            return f();
        }
        let t0 = Instant::now();
        let out = f();
        self.add(label, t0.elapsed().as_secs_f64());
        out
    }

    /// Fold `dt` seconds into `label`'s running total, creating the row on
    /// first sight (first-seen order is the report order).
    fn add(&mut self, label: &'static str, dt: f64) {
        match self.phases.iter_mut().find(|p| p.label == label) {
            Some(p) => {
                p.secs += dt;
                p.calls += 1;
            }
            None => self.phases.push(PhaseTotal {
                label,
                secs: dt,
                calls: 1,
            }),
        }
    }

    /// Begin timing `label`; the returned token stops it on
    /// [`PhaseAccumulator::stop`]. Use this instead of [`PhaseAccumulator::time`]
    /// when the region is a long `if`/`else` chain holding mutable borrows and
    /// `?`, where a closure would fight the borrow checker for no benefit.
    /// Returns `None` when timing is off, so the disabled path allocates and
    /// reads no clock.
    #[inline]
    pub fn begin(&self, label: &'static str) -> Option<(&'static str, Instant)> {
        if enabled() {
            Some((label, Instant::now()))
        } else {
            None
        }
    }

    /// End a region opened by [`PhaseAccumulator::begin`], folding its elapsed
    /// time into that label's running total. A `None` token is a no-op.
    #[inline]
    pub fn stop(&mut self, token: Option<(&'static str, Instant)>) {
        let Some((label, t0)) = token else { return };
        self.add(label, t0.elapsed().as_secs_f64());
    }

    /// Total accumulated seconds across every phase.
    pub fn total_secs(&self) -> f64 {
        self.phases.iter().map(|p| p.secs).sum()
    }

    /// Accumulated seconds for one label (`None` if never timed). Exposed so
    /// tests can assert on phases without parsing stderr.
    pub fn secs_for(&self, label: &str) -> Option<f64> {
        self.phases
            .iter()
            .find(|p| p.label == label)
            .map(|p| p.secs)
    }

    /// Print the summary table to stderr. No-op when `FERRIC_TIMING` is unset
    /// or nothing was timed.
    ///
    /// `title` names the accumulating solver (e.g. `"RHF"`). The percentage
    /// column is each phase's share of the ACCUMULATED (timed) total, not of
    /// whole-job wall time: anything the caller did not bracket is outside
    /// this table by construction. The caller's own wall-clock is the
    /// denominator to use for a whole-job claim.
    pub fn report(&self, title: &str) {
        if !enabled() || self.phases.is_empty() {
            return;
        }
        let total = self.total_secs();
        eprintln!("[timing] {title} phase totals (accumulated):");
        eprintln!(
            "[timing]   {:<32} {:>10} {:>8} {:>7}",
            "phase", "sec", "calls", "%"
        );
        for p in &self.phases {
            let pct = if total > 0.0 {
                100.0 * p.secs / total
            } else {
                0.0
            };
            eprintln!(
                "[timing]   {:<32} {:>10.3} {:>8} {:>6.1}%",
                p.label, p.secs, p.calls, pct
            );
        }
        eprintln!("[timing]   {:<32} {:>10.3}", "TOTAL (timed phases)", total);
    }
    /// Wrap this accumulator in a guard that prints its table (via
    /// [`PhaseAccumulator::report`]) when the guard is DROPPED.
    ///
    /// This is how a solver with many exits reports on every one of them —
    /// converged, max-iterations, diverged, stalled, and `?` error returns
    /// alike — without a `report` call at each `return`, where a forgotten one
    /// would silently drop the table on exactly the non-converged runs being
    /// diagnosed. The guard derefs to the accumulator, so `time`/`begin`/`stop`
    /// are called on it directly. Inert when `FERRIC_TIMING` is unset (the
    /// drop-time `report` returns immediately).
    pub fn report_on_drop(self, title: &'static str) -> ReportOnDrop {
        ReportOnDrop { acc: self, title }
    }
}

/// A [`PhaseAccumulator`] that reports when it goes out of scope. Built by
/// [`PhaseAccumulator::report_on_drop`].
#[derive(Debug)]
pub struct ReportOnDrop {
    acc: PhaseAccumulator,
    title: &'static str,
}

impl std::ops::Deref for ReportOnDrop {
    type Target = PhaseAccumulator;
    fn deref(&self) -> &PhaseAccumulator {
        &self.acc
    }
}

impl std::ops::DerefMut for ReportOnDrop {
    fn deref_mut(&mut self) -> &mut PhaseAccumulator {
        &mut self.acc
    }
}

impl Drop for ReportOnDrop {
    fn drop(&mut self) {
        self.acc.report(self.title);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `time` must return the closure's value unchanged whether or not timing
    /// is enabled — it wraps a call, it does not replace it. (This test runs
    /// under whatever `FERRIC_TIMING` the environment has; it asserts the
    /// pass-through contract that holds in BOTH states.)
    #[test]
    fn time_is_value_transparent() {
        let mut acc = PhaseAccumulator::new();
        let v = acc.time("probe", || 6.0_f64 * 7.0);
        assert_eq!(v, 42.0);
        let s = acc.time("probe2", || String::from("kept"));
        assert_eq!(s, "kept");
    }

    /// Repeated labels accumulate into ONE row rather than appending a row per
    /// call — otherwise a 400-iteration SCF would print 400 rows per phase.
    /// Only meaningful when timing is on, so it is asserted conditionally
    /// rather than by mutating the process environment (which races other
    /// tests — see the mutex-cannot-isolate-env memory).
    #[test]
    fn repeated_labels_accumulate_into_one_row() {
        let mut acc = PhaseAccumulator::new();
        for _ in 0..5 {
            acc.time("same", || std::hint::black_box(1_u64));
        }
        if enabled() {
            assert_eq!(acc.phases.len(), 1, "labels must coalesce");
            assert_eq!(acc.phases[0].calls, 5);
        } else {
            assert!(acc.phases.is_empty(), "must not record when disabled");
            assert_eq!(acc.total_secs(), 0.0);
        }
    }

    /// `time` must run its closure EXACTLY once in both states: the SCF wraps
    /// stateful calls (a DIIS step mutates its history), so a second
    /// evaluation would change the answer, not just the timing. The `FnOnce`
    /// bound already forbids it at compile time; this pins the contract if the
    /// bound is ever loosened.
    #[test]
    fn time_runs_the_closure_exactly_once() {
        let mut acc = PhaseAccumulator::new();
        let mut calls = 0;
        acc.time("once", || calls += 1);
        assert_eq!(calls, 1);
    }

    /// The drop guard is the accumulator: timing through it records (when
    /// enabled) exactly as the bare type does.
    #[test]
    fn report_guard_derefs_to_the_accumulator() {
        let mut guard = PhaseAccumulator::new().report_on_drop("probe");
        let t = guard.begin("region");
        guard.stop(t);
        assert_eq!(guard.secs_for("region").is_some(), enabled());
    }

    /// The two tests above that only bite when timing is ON
    /// (`repeated_labels_accumulate_into_one_row`, and the enabled branch of
    /// `time_runs_the_closure_exactly_once`) would be vacuous in a plain
    /// `cargo test`, because `enabled()` is fixed per process by a `OnceLock`.
    /// So an untimed run re-executes this test binary with `FERRIC_TIMING=1`
    /// over the timing tests and requires them to pass there too. (A timed
    /// run already IS that child, so it does nothing.)
    #[test]
    fn timed_contract_holds_in_a_timed_child() {
        if enabled() {
            return;
        }
        let exe = std::env::current_exe().expect("test binary path");
        let out = std::process::Command::new(exe)
            .args(["timing::tests::", "--test-threads=1"])
            .env("FERRIC_TIMING", "1")
            .output()
            .expect("spawn timed child");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "timing tests fail under FERRIC_TIMING=1:\n{stdout}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        for name in [
            "repeated_labels_accumulate_into_one_row ... ok",
            "time_runs_the_closure_exactly_once ... ok",
        ] {
            assert!(
                stdout.contains(name),
                "timed child did not run `{name}`:\n{stdout}"
            );
        }
    }
}
