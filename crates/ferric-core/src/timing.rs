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
    /// Start timing the stage `label`; reads the clock only when timing is
    /// [`enabled`].
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

    /// Print `[timing] <label> <elapsed> ms` to stderr; prints nothing when
    /// timing was disabled at [`Stage::start`].
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

/// CPU seconds consumed so far by every thread of this process
/// (`CLOCK_PROCESS_CPUTIME_ID`), or `None` where that clock is unavailable.
/// One syscall; read only at phase boundaries, never inside a parallel loop.
#[cfg(unix)]
fn process_cpu_secs() -> Option<f64> {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable timespec for the duration of the call.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    (rc == 0).then(|| ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9)
}

/// No process CPU clock is read on this platform; the `busy` column shows `-`.
#[cfg(not(unix))]
fn process_cpu_secs() -> Option<f64> {
    None
}

/// An open region returned by [`PhaseAccumulator::begin`]: its label, wall
/// start, and the process CPU clock at the start.
#[derive(Debug, Clone, Copy)]
pub struct PhaseToken {
    label: &'static str,
    t0: Instant,
    cpu0: Option<f64>,
}

impl PhaseToken {
    fn open(label: &'static str) -> Self {
        PhaseToken {
            label,
            t0: Instant::now(),
            cpu0: process_cpu_secs(),
        }
    }

    /// Wall seconds and process CPU seconds since [`PhaseToken::open`].
    fn close(self) -> (f64, Option<f64>) {
        let secs = self.t0.elapsed().as_secs_f64();
        let cpu = match (self.cpu0, process_cpu_secs()) {
            (Some(a), Some(b)) => Some(b - a),
            _ => None,
        };
        (secs, cpu)
    }
}

/// A single phase's running total inside a [`PhaseAccumulator`].
///
/// `cpu` is the process CPU time spent while the phase ran, summed over all
/// threads, so `cpu / secs` is the mean number of busy cores. It is `None`
/// for rows whose CPU time is not attributable to them alone (overlapped
/// phases, timed inside one arm of a concurrent region).
#[derive(Debug, Clone)]
struct PhaseTotal {
    label: &'static str,
    secs: f64,
    cpu: Option<f64>,
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
    /// Phases that ran CONCURRENTLY inside another timed phase (e.g. the RI-J
    /// build and the XC digest inside the SCF's `jk_xc_wall`). Reported below
    /// the table and kept out of `TOTAL`, which would otherwise count the
    /// overlapped wall time twice.
    overlapped: Vec<PhaseTotal>,
    /// Sub-phases timed INSIDE another timed phase (e.g. the RI-J and DF-K
    /// builds inside the SCF's `jk_build`). Reported below the table and kept
    /// out of `TOTAL`, since their parent already counts that wall time.
    nested: Vec<PhaseTotal>,
}

/// Fold `secs` (and `cpu`) into `label`'s row of `rows`, creating it on first
/// sight. A row's CPU total stays known only while every call reported one.
fn fold_row(rows: &mut Vec<PhaseTotal>, label: &'static str, secs: f64, cpu: Option<f64>) {
    match rows.iter_mut().find(|p| p.label == label) {
        Some(p) => {
            p.secs += secs;
            p.cpu = p.cpu.zip(cpu).map(|(a, b)| a + b);
            p.calls += 1;
        }
        None => rows.push(PhaseTotal {
            label,
            secs,
            cpu,
            calls: 1,
        }),
    }
}

impl PhaseAccumulator {
    /// An accumulator with no phases recorded (same as `Default`).
    pub fn new() -> Self {
        Self {
            phases: Vec::new(),
            overlapped: Vec::new(),
            nested: Vec::new(),
        }
    }

    /// End a region opened by [`PhaseAccumulator::begin`] as a NESTED
    /// sub-phase of an enclosing timed phase: shown in the report under its
    /// parent, excluded from [`PhaseAccumulator::total_secs`] (the parent
    /// already counts it). A `None` token is a no-op.
    #[inline]
    pub fn stop_nested(&mut self, token: Option<PhaseToken>) {
        let Some(tok) = token else { return };
        let (secs, cpu) = tok.close();
        fold_row(&mut self.nested, tok.label, secs, cpu);
    }

    /// Accumulated seconds for one nested sub-phase (`None` if never timed).
    pub fn nested_secs_for(&self, label: &str) -> Option<f64> {
        self.nested
            .iter()
            .find(|p| p.label == label)
            .map(|p| p.secs)
    }

    /// Fold `dt` (measured by the caller, e.g. inside one arm of a
    /// `rayon::join`) into `label`'s OVERLAPPED total: shown in the report,
    /// excluded from [`PhaseAccumulator::total_secs`]. No-op when timing is off.
    pub fn add_overlapped(&mut self, label: &'static str, dt: std::time::Duration) {
        if !enabled() {
            return;
        }
        fold_row(&mut self.overlapped, label, dt.as_secs_f64(), None);
    }

    /// Run `f`, adding its elapsed time to `label`'s running total. Returns
    /// `f`'s value untouched, so this wraps an existing call without changing
    /// the expression's type or its arithmetic.
    #[inline]
    pub fn time<T>(&mut self, label: &'static str, f: impl FnOnce() -> T) -> T {
        if !enabled() {
            return f();
        }
        let tok = PhaseToken::open(label);
        let out = f();
        self.stop(Some(tok));
        out
    }

    /// Begin timing `label`; the returned token stops it on
    /// [`PhaseAccumulator::stop`]. Use this instead of [`PhaseAccumulator::time`]
    /// when the region is a long `if`/`else` chain holding mutable borrows and
    /// `?`, where a closure would fight the borrow checker for no benefit.
    /// Returns `None` when timing is off, so the disabled path allocates and
    /// reads no clock.
    #[inline]
    pub fn begin(&self, label: &'static str) -> Option<PhaseToken> {
        enabled().then(|| PhaseToken::open(label))
    }

    /// End a region opened by [`PhaseAccumulator::begin`], folding its elapsed
    /// time into that label's running total (first-seen order is the report
    /// order). A `None` token is a no-op.
    #[inline]
    pub fn stop(&mut self, token: Option<PhaseToken>) {
        let Some(tok) = token else { return };
        let (secs, cpu) = tok.close();
        fold_row(&mut self.phases, tok.label, secs, cpu);
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
    /// denominator to use for a whole-job claim. The `busy` column is the
    /// process CPU time over the phase's wall time: the mean number of cores
    /// the whole process kept busy while the phase ran.
    ///
    /// The whole table is formatted into one `String` and handed to a locked
    /// stderr in a single `write_all`, so tables from solves running in
    /// parallel (e.g. free-atom guess SCFs under `par_iter`) never interleave
    /// line by line. The write error is ignored rather than unwrapped, because
    /// this runs inside `Drop` (see [`PhaseAccumulator::report_on_drop`]) and
    /// a panic there during an unwind would abort the process.
    pub fn report(&self, title: &str) {
        use std::fmt::Write as _;
        use std::io::Write as _;
        if !enabled() || self.phases.is_empty() {
            return;
        }
        let total = self.total_secs();
        // `fmt::Write` for `String` is infallible; the results are discarded.
        let mut out = String::new();
        let _ = writeln!(out, "[timing] {title} phase totals (accumulated):");
        let _ = writeln!(
            out,
            "[timing]   {:<32} {:>10} {:>8} {:>7} {:>6}",
            "phase", "sec", "calls", "%", "busy"
        );
        for p in &self.phases {
            let pct = if total > 0.0 {
                100.0 * p.secs / total
            } else {
                0.0
            };
            let _ = writeln!(
                out,
                "[timing]   {:<32} {:>10.3} {:>8} {:>6.1}% {:>6}",
                p.label,
                p.secs,
                p.calls,
                pct,
                busy_cell(p)
            );
        }
        let total_cpu: Option<f64> = self.phases.iter().map(|p| p.cpu).sum();
        let total_row = PhaseTotal {
            label: "TOTAL (timed phases)",
            secs: total,
            cpu: total_cpu,
            calls: 0,
        };
        let _ = writeln!(
            out,
            "[timing]   {:<32} {:>10.3} {:>8} {:>7} {:>6}",
            total_row.label,
            total,
            "",
            "",
            busy_cell(&total_row)
        );
        side_table(
            &mut out,
            "nested (inside a phase above, not in TOTAL)",
            '>',
            &self.nested,
        );
        side_table(
            &mut out,
            "overlapped (concurrent, not in TOTAL)",
            '~',
            &self.overlapped,
        );
        let _ = std::io::stderr().lock().write_all(out.as_bytes());
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

/// The `busy` cell for a row: CPU seconds over wall seconds, `-` when the
/// CPU time is unknown or the phase took no measurable wall time.
fn busy_cell(p: &PhaseTotal) -> String {
    match p.cpu {
        Some(cpu) if p.secs > 0.0 => format!("{:.2}", cpu / p.secs),
        _ => "-".to_string(),
    }
}

/// Append a titled side table (phases outside TOTAL) to `out`; nothing when
/// `phases` is empty. `mark` prefixes each label (`>` nested, `~` overlapped).
fn side_table(out: &mut String, title: &str, mark: char, phases: &[PhaseTotal]) {
    use std::fmt::Write as _;
    if phases.is_empty() {
        return;
    }
    let _ = writeln!(out, "[timing]   {title}:");
    for p in phases {
        let _ = writeln!(
            out,
            "[timing]   {mark}{:<31} {:>10.3} {:>8} {:>7} {:>6}",
            p.label,
            p.secs,
            p.calls,
            "",
            busy_cell(p)
        );
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

    /// A nested sub-phase is recorded (when enabled) but never counted in
    /// `TOTAL`: its parent phase already holds that wall time, so adding it
    /// would double-count.
    #[test]
    fn nested_sub_phases_stay_out_of_the_total() {
        let mut acc = PhaseAccumulator::new();
        let outer = acc.begin("parent");
        let inner = acc.begin("child");
        std::thread::sleep(std::time::Duration::from_millis(2));
        acc.stop_nested(inner);
        acc.stop(outer);
        if enabled() {
            let parent = acc.secs_for("parent").expect("parent timed");
            let child = acc.nested_secs_for("child").expect("child timed");
            assert!(child > 0.0 && child <= parent);
            assert_eq!(
                acc.secs_for("child"),
                None,
                "child must not be a top-level row"
            );
            assert_eq!(acc.total_secs(), parent);
        } else {
            assert_eq!(acc.nested_secs_for("child"), None);
            assert_eq!(acc.total_secs(), 0.0);
        }
    }

    /// A phase's CPU column counts the work done inside it: a region that
    /// spins one thread until the process CPU clock has advanced by `SPIN`
    /// must record at least `SPIN` of CPU time. Other test threads in this
    /// process can only add to it, so the lower bound holds under a parallel
    /// harness. The wall time is at least the CPU time of the spinning thread
    /// for the same reason.
    #[test]
    fn busy_column_counts_cpu_spent_in_the_phase() {
        const SPIN: f64 = 0.02;
        let mut acc = PhaseAccumulator::new();
        acc.time("spin", || {
            let Some(start) = process_cpu_secs() else {
                return;
            };
            while process_cpu_secs().is_some_and(|now| now - start < SPIN) {
                std::hint::black_box(0_u64);
            }
        });
        if enabled() && process_cpu_secs().is_some() {
            let row = &acc.phases[0];
            let cpu = row.cpu.expect("CPU time recorded");
            assert!(cpu >= SPIN, "cpu {cpu} < spun {SPIN}");
            assert!(row.secs > 0.0);
            assert_ne!(busy_cell(row), "-");
        } else if !enabled() {
            assert!(acc.phases.is_empty());
        }
    }

    /// `enabled()` is fixed per process, so in an untimed run the ON-only
    /// assertions above (coalescing; single evaluation in `time`'s enabled
    /// branch) never execute. Re-run the timing tests in a child with
    /// `FERRIC_TIMING=1` so a plain `cargo test` still checks them. No-op
    /// when this process is already timed.
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
            "nested_sub_phases_stay_out_of_the_total ... ok",
            "busy_column_counts_cpu_spent_in_the_phase ... ok",
        ] {
            assert!(
                stdout.contains(name),
                "timed child did not run `{name}`:\n{stdout}"
            );
        }
    }
}
