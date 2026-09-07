//! Phase 1: decompose the per-grid-point 3c1e cost into setup vs arithmetic.
//!
//! See `scripts/queue/out/phase1_prereg.md` for the go/no-go rule, which was
//! written and timestamped BEFORE any number here was produced.
//!
//! # What is being measured
//!
//! The COSX A-build pays, per grid point `g`:
//!
//!   (a) `scf_engine_set_point_charges` -> `libint2::Engine::set_params`
//!   (b) a full shell-pair sweep of `compute_1e_block`
//!
//! A batched kernel amortizes (a) over a batch of points and cannot touch (b),
//! so the ceiling on any batching speedup is `1 / (1 - f_a)` with
//! `f_a = a / (a + b)`.
//!
//! # Estimator
//!
//! libint2 accumulates all point charges into one scratch buffer, so a
//! `k`-charge call yields the *sum* over charges -- useless as an A-matrix, but
//! perfect as a *cost probe*: it runs exactly the work that `k` separate points
//! would run inside `compute()`, while paying the `set_params` cost only once.
//! Timing one `set_params(k charges)` plus one full sweep gives
//!
//!   T(k) = a + k * b
//!
//! whose intercept is the per-point setup and whose slope is the per-point
//! arithmetic. `f_a = a / (a + b)` is then the k=1 split. This uses the same
//! code path as the real build and needs no C++ instrumentation.
//!
//! Run with `--ignored --nocapture`; it is a measurement, not an assertion.

use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi::{self, CAtom};
use std::os::raw::c_int;
use std::time::Instant;

/// One `set_params(k charges)` + one full shell-pair sweep. Returns seconds.
///
/// `sink` accumulates a value from every block so the optimizer cannot elide
/// the sweep -- a dead-code-eliminated sweep would report `b = 0` and fake a
/// GO verdict, which is precisely the artifact this guards against.
fn time_one_sweep(eng: &mut Engine, prep: &PreparedBasis, charges: &[CAtom], sink: &mut f64) -> f64 {
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let t0 = Instant::now();
    let rc = unsafe {
        ffi::scf_engine_set_point_charges(eng.handle_mut(), charges.as_ptr(), charges.len() as c_int)
    };
    assert!(rc >= 0, "set_point_charges failed: {rc}");
    for s1 in 0..nsh {
        for s2 in 0..=s1 {
            let block = eng.compute_1e_block(prep, s1, s2);
            let n = dims[s1] * dims[s2];
            *sink += block[..n.min(block.len())].iter().sum::<f64>();
        }
    }
    t0.elapsed().as_secs_f64()
}

/// Pseudo-random probe points on a sphere-ish shell around the molecule,
/// deterministic so the measurement is reproducible.
fn probe_points(n: usize) -> Vec<CAtom> {
    let mut v = Vec::with_capacity(n);
    let mut s = 0x243f_6a88_85a3_08d3_u64;
    for _ in 0..n {
        let mut nxt = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64) / ((1u64 << 53) as f64)
        };
        v.push(CAtom {
            atomic_number: 1.0,
            x: 8.0 * (nxt() - 0.5),
            y: 8.0 * (nxt() - 0.5),
            z: 8.0 * (nxt() - 0.5),
        });
    }
    v
}

/// Resolve a repo-relative testdata path against the crate manifest dir, so the
/// harness works both under `cargo test` and when the binary is run directly.
fn testdata(rel: &str) -> String {
    format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"))
}

fn run_case(label: &str, xyz: &str, basis: &str) {
    let mol = Molecule::load_xyz(&testdata(xyz)).expect("xyz");
    let bs = ferric_core::basis::bundled(basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let nbf = prep.nbasis();
    let nsh = prep.nshells();

    let mut eng = Engine::new_1e(ffi::OP_NUCLEAR, &prep, 1e-14).expect("engine");
    let mut sink = 0.0_f64;

    // k values spanning an order of magnitude so the linear fit is well posed.
    let ks: [usize; 6] = [1, 2, 4, 8, 16, 32];
    let pts = probe_points(64);

    // Warm up: first call pays page-faults and core-ints table setup.
    for _ in 0..2 {
        time_one_sweep(&mut eng, &prep, &pts[..4], &mut sink);
    }

    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    let mut report: Vec<(usize, f64)> = Vec::new();
    for &k in &ks {
        // Repeat enough to get out of timer noise; take the MIN (least
        // contaminated by scheduler noise), which is the standard robust
        // estimator for a deterministic compute kernel.
        let reps = if k <= 4 { 7 } else { 5 };
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t = time_one_sweep(&mut eng, &prep, &pts[..k], &mut sink);
            best = best.min(t);
        }
        xs.push(k as f64);
        ys.push(best);
        report.push((k, best));
    }

    // Ordinary least squares: y = a + b*x.
    let n = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|v| v * v).sum();
    let sxy: f64 = xs.iter().zip(&ys).map(|(u, v)| u * v).sum();
    let b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let a = (sy - b * sx) / n;

    // R^2 so a bad fit cannot masquerade as a clean split.
    let ybar = sy / n;
    let ss_tot: f64 = ys.iter().map(|v| (v - ybar).powi(2)).sum();
    let ss_res: f64 = xs
        .iter()
        .zip(&ys)
        .map(|(u, v)| (v - (a + b * u)).powi(2))
        .sum();
    let r2 = 1.0 - ss_res / ss_tot;

    let f_a = a / (a + b);
    let ceiling = 1.0 / (1.0 - f_a);

    println!("=== {label} / {basis}: nbf={nbf} nsh={nsh} ===");
    for (k, t) in &report {
        println!("  k={k:3}  T={t:.6} s   T/k={:.6e} s", t / (*k as f64));
    }
    println!("  fit: a(setup) = {a:.6e} s   b(arith/point) = {b:.6e} s   R^2 = {r2:.6}");
    println!("  f_a = a/(a+b) = {f_a:.4}   batching ceiling = 1/(1-f_a) = {ceiling:.2}x");
    println!("  sink (anti-DCE) = {sink:.6e}");
}

#[test]
#[ignore = "measurement, not an assertion; run with --ignored --nocapture"]
fn phase1_setup_vs_arithmetic_split() {
    for (label, xyz) in [
        ("water", "testdata/molecules/water.xyz"),
        ("butane", "testdata/molecules/alkane_4.xyz"),
    ] {
        for basis in ["cc-pvdz", "cc-pvtz"] {
            run_case(label, xyz, basis);
        }
    }
}
