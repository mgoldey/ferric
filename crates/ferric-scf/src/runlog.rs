//! Machine-readable JSON run logs (JSON Lines), streamed as the run happens.
//!
//! # Why this exists
//!
//! A number with no surviving artifact is not a measurement. This module exists
//! because a load-bearing SCF result (a 27-atom PBE/6-31G run reported as "173
//! iterations, converged, E = -390.3794234093") could not be checked afterwards:
//! its stdout log had been truncated to 275 bytes and two sibling logs were
//! empty. The run itself was gone. Turning logging ON BY DEFAULT and writing
//! each record the moment it is produced is the fix.
//!
//! # Format: JSON Lines, not one JSON document
//!
//! Each record is one self-contained JSON object on one line, `write`n and
//! `flush`ed immediately. This is deliberate and is the whole point:
//!
//! * A run that is OOM-killed, hits a wall-clock limit, or is Ctrl-C'd at
//!   iteration 90 of 100 leaves a file whose first 90 iteration records are
//!   complete and parseable. A single buffered JSON document serialized at exit
//!   would leave nothing — which is exactly the failure this module exists to
//!   prevent.
//! * There is no closing bracket to miss, so a truncated file is still valid
//!   JSONL up to its last newline. A partial final line is the ONLY thing a
//!   reader must tolerate (`serde_json::from_str` on it will fail; skip it).
//!
//! Read one with e.g. `jq -c 'select(.record=="scf_iter")' ferric-run-*.json`.
//!
//! # Observation, never participation
//!
//! Logging must not move a number. Two properties enforce that:
//!
//! * Every entry point takes already-computed values and only formats them.
//!   Nothing here is consulted by a convergence gate, and no method computes a
//!   quantity that exists only for the log.
//! * Every failure is swallowed after one warning to stderr. An unopenable
//!   path, a full disk, a broken pipe — none of them fail a calculation. See
//!   [`RunLog::emit`].
//!
//! The regression tests `logging_does_not_move_the_energy_*` in
//! `tests/runlog_bit_identity.rs` assert `f64::to_bits()` equality of SCF
//! energies with the log on and off.
//!
//! # Scope
//!
//! SCF (RHF/UHF/ROHF and their KS-DFT counterparts, which are the same solvers)
//! plus the convergence ladder. Post-SCF methods (MP2/RPA/CC/GW/CI) are NOT
//! covered yet; the schema is open (`record` is a free-form tag and each record
//! type carries only its own fields) precisely so they can be added without
//! changing anything here.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Schema version of the emitted records.
///
/// Bump this when a field changes meaning or is removed. ADDING a field is
/// backwards compatible for any reader that ignores unknown keys (which a
/// `jq`/`json.loads` reader does by construction), so it does not need a bump.
pub const SCHEMA_VERSION: u32 = 1;

/// Where a run log writes, and how.
///
/// This is a *process-global* sink rather than a field on `RhfConfig` on
/// purpose. `RhfConfig` is cloned per ladder rung (see
/// [`crate::ladder::default_ladder_from`]), is constructed by library callers
/// that know nothing about files, and is threaded through five solvers — a
/// field there would have to survive every clone and every hand-rolled struct
/// literal, and a missed one would silently drop records. A global has exactly
/// one install point and cannot be lost by a clone.
struct Sink {
    /// The open file. A `Mutex` because solvers may emit from different
    /// threads; contention is irrelevant at one record per SCF iteration.
    file: Mutex<std::fs::File>,
    path: PathBuf,
    /// Set once, after the first write failure, so a full disk warns once
    /// instead of once per iteration.
    warned: AtomicBool,
    /// Monotonic record counter, so a reader can detect a gap.
    seq: AtomicU64,
    /// Run start, for the `t` (seconds since run start) field on every record.
    started: Instant,
}

static SINK: OnceLock<Option<Sink>> = OnceLock::new();

/// Handle to the installed run log. Cheap to obtain and to hold; all methods
/// are no-ops when no log is installed.
///
/// Every method is infallible by design — see the module docs on why a logging
/// failure may never fail a calculation.
#[derive(Debug, Clone, Copy)]
pub struct RunLog;

/// The installed run log, or `None` when logging is off or was never
/// initialized (the default for a library caller that never calls
/// [`init`]).
#[must_use]
pub fn log() -> Option<RunLog> {
    SINK.get().and_then(|s| s.as_ref()).map(|_| RunLog)
}

/// Path of the installed run log, if any. For the "wrote log to ..." line the
/// CLI prints.
#[must_use]
pub fn path() -> Option<PathBuf> {
    SINK.get().and_then(|s| s.as_ref()).map(|s| s.path.clone())
}

/// Install a run log at `path`, truncating any existing file.
///
/// Returns `false` (after one stderr warning) if the file cannot be opened —
/// the run then proceeds with logging off, exactly as if it had never been
/// requested. Calling this more than once is a no-op after the first call
/// (`OnceLock`), which keeps a library caller from silently repointing a CLI's
/// log mid-run.
///
/// Not called automatically: a library caller gets no log unless it asks, and
/// the CLI asks on every run (see `ferric_cli`'s `[output] json` handling).
pub fn init(path: &Path) -> bool {
    let mut opened = false;
    SINK.get_or_init(|| match std::fs::File::create(path) {
        Ok(f) => {
            opened = true;
            Some(Sink {
                file: Mutex::new(f),
                path: path.to_path_buf(),
                warned: AtomicBool::new(false),
                seq: AtomicU64::new(0),
                started: Instant::now(),
            })
        }
        Err(e) => {
            // WARN, never swallow. A log that silently does not exist is the
            // exact failure this module was built to prevent: a load-bearing
            // measurement earlier in this project's history could not be
            // checked because its log had been truncated to 275 bytes and
            // nobody noticed until the number was needed.
            //
            // This is a WARNING and not an error on purpose -- logging is
            // observation, and an unwritable path must never fail a
            // calculation that would otherwise succeed. The run continues
            // unlogged, loudly.
            eprintln!(
                "[ferric] warning: could not open run log {}: {e}; \
                 continuing without one. THIS RUN WILL LEAVE NO \
                 MACHINE-READABLE RECORD.",
                path.display()
            );
            None
        }
    });
    opened || log().is_some()
}

/// Which ladder rung is currently running, +1, or 0 for "not in a ladder".
///
/// The `+1` offset lets 0 mean "no rung" without a second flag, since rung
/// indices are 0-based.
static CURRENT_RUNG: AtomicU64 = AtomicU64::new(0);

/// The ladder rung whose SCF is running right now, if any.
///
/// Read by the per-iteration emit sites in the solvers, so an `scf_iter`
/// record says which rung it belongs to. Without this, a laddered run's
/// iteration records would be an undifferentiated stream and "iteration 40"
/// would be ambiguous across five rungs — the exact ambiguity that made a
/// reported "iterations = 100" get misread as a whole-run total rather than
/// rung 4 hitting its own hardcoded cap.
///
/// A plain global (not thread-local): the ladder is strictly sequential, one
/// rung at a time, and the solver it calls emits from whichever thread it
/// happens to be on.
#[must_use]
pub fn current_rung() -> Option<usize> {
    match CURRENT_RUNG.load(Ordering::Relaxed) {
        0 => None,
        n => Some((n - 1) as usize),
    }
}

/// Scope guard that tags every `scf_iter` record emitted while it is alive
/// with `rung`. Restores the previous value on drop, so a nested ladder (the
/// DF-guess pre-stage inside a rung, say) cannot leave a stale tag behind.
///
/// Constructed unconditionally and cheaply (one relaxed atomic store) whether
/// or not a log is installed, so the ladder needs no `if logging` branch.
pub struct RungScope(u64);

impl RungScope {
    /// Tag records with `rung` until this guard drops.
    #[must_use]
    pub fn enter(rung: usize) -> Self {
        let prev = CURRENT_RUNG.swap(rung as u64 + 1, Ordering::Relaxed);
        Self(prev)
    }
}

impl Drop for RungScope {
    fn drop(&mut self) {
        CURRENT_RUNG.store(self.0, Ordering::Relaxed);
    }
}

/// Nesting depth of [`SubSolveScope`] guards: >0 means the SCF currently
/// running is an INTERNAL one (a free-atom SCF for the SAD/MINAO guess, say),
/// not the molecular SCF the user asked for.
static SUB_SOLVE_DEPTH: AtomicU64 = AtomicU64::new(0);

/// Whether the SCF running right now is an internal sub-solve rather than the
/// run's own SCF.
///
/// This matters because the MINAO/SAD initial guess runs a full `solve_rhf` or
/// `solve_uhf` per ELEMENT (see [`crate::guess`]), inside the ladder's rung
/// scope. Without this distinction a water/STO-3G run's log opened with four
/// `scf_iter` records labelled `uhf` at rung 0 — free hydrogen and free oxygen
/// — before the molecular RHF ever started. A reader counting iterations, or
/// reading the first energy as the run's, would be wrong on both counts.
///
/// Sub-solve iterations are still LOGGED, under `record: "guess_scf_iter"`:
/// they are real work, they can fail, and their cost is part of the run. They
/// are simply not the same record type as the SCF whose energy is the answer.
#[must_use]
pub fn in_sub_solve() -> bool {
    SUB_SOLVE_DEPTH.load(Ordering::Relaxed) > 0
}

/// Scope guard marking everything inside it as an internal sub-solve (see
/// [`in_sub_solve`]). Nests correctly: a counter, not a flag.
pub struct SubSolveScope;

impl SubSolveScope {
    /// Mark the enclosing scope as an internal sub-solve until the guard drops.
    #[must_use]
    pub fn enter() -> Self {
        SUB_SOLVE_DEPTH.fetch_add(1, Ordering::Relaxed);
        Self
    }
}

impl Drop for SubSolveScope {
    fn drop(&mut self) {
        SUB_SOLVE_DEPTH.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Explicitly install *no* run log, so a later [`init`] cannot turn one on.
///
/// This is what `[output] json = false` resolves to. Without it, a library
/// call made after the CLI decided not to log could still install a sink.
pub fn disable() {
    SINK.get_or_init(|| None);
}

impl RunLog {
    /// Write one record: `{"record": <kind>, ...fields}` plus the automatic
    /// `seq`/`t` envelope, then flush.
    ///
    /// EVERY failure path here is a warn-once-and-continue. This is the single
    /// most important property of the module: a logging error must not
    /// propagate, panic, or change control flow in the solver that called it.
    /// The `Mutex` is unpoisonable in practice (nothing here panics while
    /// holding it) but is recovered rather than unwrapped anyway, because a
    /// panic elsewhere must not turn into a second panic here.
    fn emit(self, kind: &str, fields: serde_json::Value) {
        let Some(sink) = SINK.get().and_then(|s| s.as_ref()) else {
            return;
        };
        let mut obj = match fields {
            serde_json::Value::Object(m) => m,
            other => {
                let mut m = serde_json::Map::new();
                m.insert("value".into(), other);
                m
            }
        };
        let seq = sink.seq.fetch_add(1, Ordering::SeqCst);
        obj.insert("record".into(), serde_json::Value::from(kind));
        obj.insert("seq".into(), serde_json::Value::from(seq));
        obj.insert("t".into(), json_f64(sink.started.elapsed().as_secs_f64()));
        let mut line = match serde_json::to_string(&serde_json::Value::Object(obj)) {
            Ok(s) => s,
            Err(e) => {
                self.warn_once(sink, &format!("could not serialize a {kind} record: {e}"));
                return;
            }
        };
        line.push('\n');
        let mut f = match sink.file.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        // write_all + flush, per record. `flush` on a `File` is a no-op at the
        // std level (there is no user-space buffer to drain — `write_all`
        // already issued the `write(2)`), so this costs nothing beyond the
        // syscall we must make anyway; it is kept so that swapping the sink for
        // a `BufWriter` later cannot silently reintroduce end-of-run buffering,
        // which is the failure mode this module exists to prevent.
        if let Err(e) = f.write_all(line.as_bytes()).and_then(|()| f.flush()) {
            drop(f);
            self.warn_once(sink, &format!("could not write a {kind} record: {e}"));
        }
    }

    fn warn_once(self, sink: &Sink, msg: &str) {
        if !sink.warned.swap(true, Ordering::SeqCst) {
            eprintln!(
                "[warning] JSON run log {}: {msg}; further log errors are suppressed \
                 and the calculation continues",
                sink.path.display()
            );
        }
    }

    /// Header record: what this run is. Emitted once, before any solver runs.
    ///
    /// `config` and `molecule` are opaque JSON objects assembled by the caller
    /// (the CLI knows the TOML; this crate does not), so adding a knob to the
    /// CLI never requires touching this signature.
    pub fn run_start(self, config: serde_json::Value, molecule: serde_json::Value) {
        self.emit(
            "run_start",
            serde_json::json!({
                "schema": SCHEMA_VERSION,
                "ferric_version": env!("CARGO_PKG_VERSION"),
                "git_sha": git_sha(),
                "timestamp": timestamp_rfc3339(),
                "config": config,
                "molecule": molecule,
            }),
        );
    }

    /// One SCF iteration.
    ///
    /// Every argument is a quantity the solver ALREADY computed for its own
    /// convergence decision or its existing `FERRIC_SCF_TRACE` line — nothing
    /// here is computed for the log's benefit. See
    /// [`crate::rhf::ConvergenceSignals`] and the `scf_converged` gate.
    ///
    /// `err_max` is the DIIS commutator max element (a diagnostic, never a
    /// gate); `grad_rms` its RMS, which only `solve_rhf` forms — `None`
    /// elsewhere rather than a fabricated zero.
    #[allow(clippy::too_many_arguments)]
    pub fn scf_iter(
        self,
        method: &str,
        rung: Option<usize>,
        iter: usize,
        energy: f64,
        de: f64,
        dp_rms: f64,
        dp_max: f64,
        err_max: f64,
        grad_rms: Option<f64>,
    ) {
        // An SCF run for the initial guess (free-atom SAD/MINAO) gets its own
        // record type so it can never be mistaken for the molecular SCF whose
        // energy is the run's answer. See `in_sub_solve`.
        let kind = if in_sub_solve() {
            "guess_scf_iter"
        } else {
            "scf_iter"
        };
        self.emit(
            kind,
            serde_json::json!({
                "method": method,
                "rung": rung,
                "iter": iter,
                "energy": json_f64(energy),
                "de": json_f64(de),
                "dp_rms": json_f64(dp_rms),
                "dp_max": json_f64(dp_max),
                "err_max": json_f64(err_max),
                "grad_rms": grad_rms.map(json_f64),
            }),
        );
    }

    /// One ladder rung's outcome.
    ///
    /// `max_iter` is recorded because the ladder OVERRIDES the caller's
    /// configured `max_iter` with a hardcoded per-rung cap (60/60/60/80/100 —
    /// see [`crate::ladder::default_ladder_from`]). Reading a reported
    /// "iterations = 100" as a whole-run total rather than as rung 4 hitting
    /// its own cap has already produced a wrong diagnosis, so the cap and the
    /// count that hit it travel together.
    #[allow(clippy::too_many_arguments)]
    pub fn ladder_rung(
        self,
        rung: usize,
        tricks: &[&str],
        max_iter: usize,
        iters: usize,
        exit: &str,
        energy: f64,
        converged: bool,
    ) {
        self.emit(
            "ladder_rung",
            serde_json::json!({
                "rung": rung,
                "tricks": tricks,
                "max_iter": max_iter,
                "iterations": iters,
                "exit": exit,
                "energy": json_f64(energy),
                "converged": converged,
            }),
        );
    }

    /// Terminal record: the run's result.
    ///
    /// `extra` carries method-specific fields (correlation energy, ΔE, …) so
    /// post-SCF methods can be added without changing this signature.
    pub fn run_end(self, energy: f64, converged: bool, exit: &str, extra: serde_json::Value) {
        let started = SINK
            .get()
            .and_then(|s| s.as_ref())
            .map(|s| s.started.elapsed().as_secs_f64());
        self.emit(
            "run_end",
            serde_json::json!({
                "energy": json_f64(energy),
                "converged": converged,
                "exit": exit,
                "wall_s": started.map(json_f64),
                "cpu_s": cpu_seconds().map(json_f64),
                "peak_rss_bytes": peak_rss_bytes(),
                "extra": extra,
            }),
        );
    }

    /// The METHOD'S ANSWER -- the number the run was launched to produce.
    ///
    /// `run_end` carries the SCF energy, which for a correlated method is the
    /// REFERENCE, not the result: a `kind="rimp2"` run's `run_end.energy` is
    /// the RHF energy it was built on. Before this record existed, every
    /// correlated energy went to stdout ONLY (71 print sites across 18 method
    /// kinds), so a machine reading the log got the iteration trace and not
    /// the answer.
    ///
    /// `total` is the headline number a user would quote. `components` carries
    /// the decomposition that makes it checkable -- `e_corr`, `e_os`/`e_ss`,
    /// a reference energy -- because a total alone cannot be reconciled
    /// against a reference implementation.
    ///
    /// Emit this ONCE per run, after the method finishes. A method that has
    /// not been wired up yet must call [`RunLog::result_unlogged`] instead of
    /// staying silent: a consumer has to be able to tell "this method does not
    /// log its result yet" from "this run died before producing one".
    pub fn result(self, kind: &str, total: f64, components: serde_json::Value) {
        self.emit(
            "result",
            serde_json::json!({
                "kind": kind,
                "total": json_f64(total),
                "components": components,
            }),
        );
    }

    /// Declare that `kind` produced a result this module does not yet record.
    ///
    /// The point is that SILENCE IS AMBIGUOUS. A log with no `result` record
    /// could mean the method is unwired, or that the run crashed before
    /// finishing; those demand opposite responses from whatever reads the log.
    /// This makes the first case explicit and leaves the second as the only
    /// remaining reading of a missing record.
    pub fn result_unlogged(self, kind: &str) {
        self.emit(
            "result_unlogged",
            serde_json::json!({
                "kind": kind,
                "why": "this method's result is printed to stdout but not yet \
                        wired into the run log; run_end carries only its SCF \
                        reference energy",
            }),
        );
    }

    /// A free-form record, for a caller that wants to note something the
    /// typed helpers do not cover (a warning, a stage boundary, a method this
    /// module does not yet model). `kind` becomes the `record` tag.
    pub fn note(self, kind: &str, fields: serde_json::Value) {
        self.emit(kind, fields);
    }
}

/// JSON-encode an `f64`, mapping non-finite values to a string rather than
/// silently losing them.
///
/// JSON has no NaN or infinity. `serde_json` encodes both as `null`, which
/// would make "the gradient was NaN" and "this method does not report a
/// gradient" indistinguishable in the log — and a NaN is exactly the thing a
/// post-mortem reader most needs to see. `"NaN"` / `"inf"` / `"-inf"` are
/// unambiguous and survive a round trip.
///
/// `dp_rms`/`dp_max` are genuinely `+inf` on SCF iteration 1 (the density
/// monitor has no previous density yet), so this is a routine path, not an
/// error path.
fn json_f64(v: f64) -> serde_json::Value {
    if v.is_nan() {
        serde_json::Value::from("NaN")
    } else if v.is_infinite() {
        serde_json::Value::from(if v > 0.0 { "inf" } else { "-inf" })
    } else {
        serde_json::Value::from(v)
    }
}

/// Build-time git SHA, if the build environment supplied one.
///
/// `FERRIC_GIT_SHA` is read with `option_env!`, so a plain `cargo build`
/// (which sets nothing) reports `null` rather than a wrong or stale SHA.
/// Shelling out to `git` at runtime was rejected: the binary may run far from
/// its source tree, on a different machine, long after the checkout moved.
fn git_sha() -> Option<&'static str> {
    option_env!("FERRIC_GIT_SHA")
}

/// Wall-clock timestamp as an RFC-3339 UTC string.
///
/// Hand-rolled from `SystemTime` (civil-from-days, Howard Hinnant's algorithm)
/// rather than adding a `chrono`/`time` dependency for one line of output.
fn timestamp_rfc3339() -> String {
    let now = std::time::SystemTime::now();
    let Ok(dur) = now.duration_since(std::time::UNIX_EPOCH) else {
        return "1970-01-01T00:00:00Z".to_string();
    };
    let secs = dur.as_secs();
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (h, mi, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let (y, mo, d) = civil_from_days(days as i64);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// Days-since-1970 → (year, month, day). Hinnant's `civil_from_days`.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Total (user + system) CPU seconds for this process, from
/// `/proc/self/stat`. `None` off Linux or on any parse failure — an
/// observability helper never fails a run.
fn cpu_seconds() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/stat").ok()?;
    // Field 14 (utime) and 15 (stime), 1-based, AFTER the comm field — which
    // may itself contain spaces and parentheses, so split at the LAST ')'.
    let rest = s.rsplit_once(')')?.1;
    let f: Vec<&str> = rest.split_whitespace().collect();
    // `rest` starts at field 3 (state), so utime/stime are indices 11 and 12.
    let utime = f.get(11)?.parse::<u64>().ok()?;
    let stime = f.get(12)?.parse::<u64>().ok()?;
    // USER_HZ is 100 on every Linux target ferric builds for.
    Some((utime + stime) as f64 / 100.0)
}

/// Peak resident set size in bytes, from `/proc/self/status`'s `VmHWM`.
///
/// `VmHWM` (high-water mark), not `VmRSS`: a post-mortem reader wants the peak
/// the run actually reached, not whatever happened to be resident at exit.
/// `None` off Linux or on any parse failure.
fn peak_rss_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

/// Default log path for a run whose input file is `input`: the input's path
/// with its extension replaced by `.ferric.jsonl`, so a run's log lands next
/// to the input that produced it and two different inputs in one directory
/// never collide.
///
/// Falls back to `ferric-run.jsonl` in the current directory when `input` has
/// no usable stem.
#[must_use]
pub fn default_path_for_input(input: &Path) -> PathBuf {
    match input.file_stem() {
        Some(stem) => {
            let mut p = input.to_path_buf();
            p.set_file_name(format!("{}.ferric.jsonl", stem.to_string_lossy()));
            p
        }
        None => PathBuf::from("ferric-run.jsonl"),
    }
}

/// Short names of the convergence "tricks" a rung's config enables, for the
/// `ladder_rung` record.
///
/// Reports only what differs from a plain DIIS rung, so rung 0 of the default
/// ladder reports `["diis"]` and the escalation is readable at a glance.
#[must_use]
pub fn rung_tricks(config: &crate::rhf::RhfConfig) -> Vec<&'static str> {
    let mut v = Vec::new();
    match config.diis_flavor {
        crate::diis::DiisFlavor::Adiis => v.push("adiis"),
        _ => v.push("diis"),
    }
    if config.level_shift > 0.0 {
        v.push("level_shift");
    }
    if config.newton_trigger > 0.0 {
        v.push("soscf");
    }
    if config.smearing_sigma.is_some() {
        v.push("smearing");
    }
    if config.mom_after_iter > 0 {
        v.push("mom");
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_f64_distinguishes_nan_from_a_missing_value() {
        // The whole point: `null` must mean "not reported", never "NaN".
        assert_eq!(json_f64(f64::NAN), serde_json::Value::from("NaN"));
        assert_eq!(json_f64(f64::INFINITY), serde_json::Value::from("inf"));
        assert_eq!(json_f64(f64::NEG_INFINITY), serde_json::Value::from("-inf"));
        assert!(json_f64(1.5).is_number());
        assert_ne!(json_f64(f64::NAN), serde_json::Value::Null);
    }

    #[test]
    fn json_f64_round_trips_a_full_precision_energy() {
        // A log that loses the last digits of an energy cannot be used to
        // check a reported energy, which is what this module is for.
        let e = -390.379_423_409_312_7_f64;
        let s = serde_json::to_string(&json_f64(e)).unwrap();
        let back: f64 = serde_json::from_str(&s).unwrap();
        assert_eq!(back.to_bits(), e.to_bits(), "energy did not round-trip");
    }

    #[test]
    fn civil_from_days_matches_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1)); // leap-year boundary
        assert_eq!(civil_from_days(20_000), (2024, 10, 4));
    }

    #[test]
    fn timestamp_has_the_expected_shape() {
        let t = timestamp_rfc3339();
        assert_eq!(t.len(), 20, "{t}");
        assert!(t.ends_with('Z'), "{t}");
        assert!(t.starts_with("20"), "{t}");
    }

    #[test]
    fn default_path_lands_next_to_the_input() {
        assert_eq!(
            default_path_for_input(Path::new("/a/b/water-rhf.toml")),
            PathBuf::from("/a/b/water-rhf.ferric.jsonl")
        );
        assert_eq!(
            default_path_for_input(Path::new("run.toml")),
            PathBuf::from("run.ferric.jsonl")
        );
    }

    #[test]
    fn rung_tricks_names_the_escalation() {
        let ladder = crate::ladder::default_ladder();
        let names: Vec<Vec<&str>> = ladder.iter().map(|r| rung_tricks(&r.config)).collect();
        assert_eq!(names[0], vec!["diis"]);
        assert_eq!(names[1], vec!["adiis"]);
        assert!(names[2].contains(&"level_shift"), "{:?}", names[2]);
        assert!(names[3].contains(&"soscf"), "{:?}", names[3]);
        assert!(names[4].contains(&"smearing"), "{:?}", names[4]);
    }

    /// `log()` is `None` in a process that never installed a sink — which is
    /// every library caller and every test but the dedicated writer tests.
    /// If this ever returns `Some`, the emit calls in the solvers are doing
    /// real I/O in every unit test in the workspace.
    #[test]
    fn no_sink_means_no_log() {
        // Cannot assert globally (another test in this binary may install
        // one), but `emit` on a `RunLog` with no sink must be a silent no-op
        // rather than a panic.
        RunLog.note("unit_test", serde_json::json!({"ok": true}));
    }
}
