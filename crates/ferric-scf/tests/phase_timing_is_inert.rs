//! The opt-in phase timers inside `solve_rhf` must not change the calculation,
//! and the summary table must appear on EVERY exit, not only the converged one.
//!
//! A timer that perturbs what it measures is worse than no timer: it would make
//! every profiled number a measurement of the instrument. A timer that stays
//! silent on a non-converged exit is nearly as bad, because those are exactly
//! the runs someone is profiling.
//!
//! WHY CHILD PROCESSES: `ferric_core::timing::enabled()` caches its answer in a
//! `OnceLock`, so the flag is fixed for a process's whole life and cannot be
//! flipped mid-test. Mutating the env var from a test also races every other
//! test in the binary. So the on/off comparison re-runs THIS test binary as a
//! child, once with `FERRIC_TIMING=1` and once with it removed, and compares
//! what the two children print. The `child_*` tests are no-ops unless the
//! parent sets `CHILD_ENV`.

use std::process::Command;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::result::ScfExit;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;

/// Water/STO-3G RHF, `energy_conv` 1e-10, default config.
///
/// MEASURED, not copied from a reference table: origin/main 6e7e7e44a with
/// rhf.rs UNMODIFIED (before any timer was added), 2026-10-24, gave bits
/// 0xc052bda43279da86 = -74.96314680003897 Ha in 8 iterations; the
/// instrumented build reproduces the same bits untimed. The 2026-09-21 source
/// branch pinned -74.963_146_800_0, which still agrees to 1e-10 despite the
/// libint-precision (PR #226) and guess changes since.
const WATER_STO3G_RHF: f64 = -74.963_146_800_038_97;
/// Iteration count of the same run, pinned exactly.
const WATER_STO3G_ITERS: usize = 8;

/// Marks a process as a child spawned by one of the parent tests below.
const CHILD_ENV: &str = "FERRIC_PHASE_TIMING_CHILD";
/// The header line `PhaseAccumulator::report("RHF")` prints.
const TABLE_HEADER: &str = "[timing] RHF phase totals";

fn is_child() -> bool {
    std::env::var_os(CHILD_ENV).is_some()
}

fn run_water(cfg: &RhfConfig) -> ScfResult {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, cfg).unwrap()
}

fn converged_cfg() -> RhfConfig {
    RhfConfig {
        energy_conv: 1e-10,
        ..Default::default()
    }
}

/// Re-run this test binary as a child running only `test`, with
/// `FERRIC_TIMING=1` when `timed`, else with `FERRIC_TIMING` removed.
/// Returns (stdout, stderr); panics if the child failed.
fn spawn_child(test: &str, timed: bool) -> (String, String) {
    let exe = std::env::current_exe().expect("test binary path");
    let mut cmd = Command::new(exe);
    cmd.args(["--exact", test, "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, "1");
    if timed {
        cmd.env("FERRIC_TIMING", "1");
    } else {
        cmd.env_remove("FERRIC_TIMING");
    }
    let out = cmd.output().expect("spawn child test process");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "child `{test}` (timed={timed}) failed\n--- stdout\n{stdout}\n--- stderr\n{stderr}"
    );
    assert!(
        stdout.contains("1 passed"),
        "child `{test}` did not run exactly one test:\n{stdout}"
    );
    (stdout, stderr)
}

/// The energy and iteration count must match the pinned reference regardless
/// of `FERRIC_TIMING` (this test runs under whatever the environment says; the
/// two-process test below covers both states explicitly). 1e-12 rather than
/// `==` so a 1-ulp BLAS difference across machines does not fail it.
#[test]
fn timing_flag_does_not_change_the_energy() {
    let r = run_water(&converged_cfg());
    assert!(r.converged, "reference SCF must converge");
    assert_eq!(
        r.iterations, WATER_STO3G_ITERS,
        "iteration count moved; if ONLY under FERRIC_TIMING=1, the phase timers \
         are perturbing the solve"
    );
    assert!(
        (r.energy - WATER_STO3G_RHF).abs() < 1e-12,
        "RHF energy {:.14} deviates from the pinned {WATER_STO3G_RHF:.14}; if this \
         fails ONLY under FERRIC_TIMING=1, the phase timers are perturbing the \
         calculation and must be reverted to pure bracketing",
        r.energy
    );
}

/// Child half of the A/B: print the result bit-exactly on stdout.
#[test]
fn child_print_water_result() {
    if !is_child() {
        return;
    }
    let r = run_water(&converged_cfg());
    println!(
        "PHASE_AB energy_bits={:#018x} iters={} converged={}",
        r.energy.to_bits(),
        r.iterations,
        r.converged
    );
}

/// Two processes, timing ON and OFF: the energy must be BIT-identical and the
/// iteration count equal, the timed run must print exactly one RHF table
/// (MINAO's free-atom SCFs may print their own nested tables first, so
/// "at least one"), and the untimed run must print no timing output at all.
#[test]
fn timing_on_and_off_are_bit_identical_across_processes() {
    if is_child() {
        return;
    }
    let pick = |stdout: &str| -> String {
        // libtest prints `test <name> ... ` on the same line before the
        // child's own output, so match the marker anywhere in the line.
        let line = stdout
            .lines()
            .find_map(|l| l.find("PHASE_AB ").map(|i| &l[i..]))
            .unwrap_or_else(|| panic!("no PHASE_AB line in child stdout:\n{stdout}"));
        line.to_string()
    };
    let (out_on, err_on) = spawn_child("child_print_water_result", true);
    let (out_off, err_off) = spawn_child("child_print_water_result", false);
    assert_eq!(
        pick(&out_on),
        pick(&out_off),
        "FERRIC_TIMING=1 changed the energy bits or the iteration count"
    );
    assert!(
        err_on.contains(TABLE_HEADER),
        "timed run printed no RHF phase table:\n{err_on}"
    );
    for label in [
        "setup:prepare",
        "jk_build",
        "diis_extrapolate",
        "diagonalize",
    ] {
        assert!(
            err_on.contains(label),
            "timed table lacks `{label}`:\n{err_on}"
        );
    }
    assert!(
        !err_off.contains("[timing]"),
        "untimed run printed timing output:\n{err_off}"
    );
}

/// Child for the non-converged exits: one MaxIter run, then one Diverged run,
/// each preceded by a marker on stderr so the parent can attribute tables.
/// hcore guess (`use_sad_guess: false`) so no nested free-atom SCF prints a
/// table of its own and muddies the count.
#[test]
fn child_nonconverged_exits() {
    if !is_child() {
        return;
    }
    eprintln!("PHASE_CASE maxiter");
    let r = run_water(&RhfConfig {
        max_iter: 3,
        use_sad_guess: false,
        ..Default::default()
    });
    assert_eq!(r.exit, ScfExit::MaxIter, "case must hit max_iter");
    eprintln!("PHASE_CASE diverged");
    // A -inf tolerance counts every iteration as "energy rising", so the
    // 3-iteration divergence streak fires deterministically at iter 3, ahead
    // of the convergence test.
    let r = run_water(&RhfConfig {
        divergence_tol: Some(f64::NEG_INFINITY),
        use_sad_guess: false,
        ..Default::default()
    });
    assert_eq!(r.exit, ScfExit::Diverged, "case must diverge");
    eprintln!("PHASE_CASE end");
}

/// MaxIter and Diverged exits must each print exactly one RHF table. A report
/// wired only to the converged `return` (the original design) fails this.
#[test]
fn nonconverged_exits_still_print_the_table() {
    if is_child() {
        return;
    }
    let (_, err) = spawn_child("child_nonconverged_exits", true);
    let section = |from: &str, to: &str| -> &str {
        let a = err
            .find(from)
            .unwrap_or_else(|| panic!("no `{from}` in:\n{err}"));
        let b = err
            .find(to)
            .unwrap_or_else(|| panic!("no `{to}` in:\n{err}"));
        &err[a..b]
    };
    for (case, from, to) in [
        ("MaxIter", "PHASE_CASE maxiter", "PHASE_CASE diverged"),
        ("Diverged", "PHASE_CASE diverged", "PHASE_CASE end"),
    ] {
        let n = section(from, to).matches(TABLE_HEADER).count();
        assert_eq!(n, 1, "{case} exit printed {n} RHF tables, want 1:\n{err}");
    }
}

/// The accumulator itself must be value-transparent and must coalesce repeated
/// labels, asserted directly on the type, independent of any SCF run.
#[test]
fn accumulator_is_value_transparent_and_coalesces() {
    use ferric_core::timing::PhaseAccumulator;
    let mut acc = PhaseAccumulator::new();
    let got = acc.time("phase_a", || 6.0_f64 * 7.0);
    assert_eq!(got, 42.0);
    for _ in 0..4 {
        acc.time("phase_a", || std::hint::black_box(0_u64));
    }
    if ferric_core::timing::enabled() {
        assert!(acc.secs_for("phase_a").is_some_and(|t| t >= 0.0));
        assert!(acc.secs_for("never_timed").is_none());
    } else {
        assert!(acc.secs_for("phase_a").is_none());
        assert_eq!(acc.total_secs(), 0.0);
    }
}
