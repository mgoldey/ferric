//! Phase 1 cross-check: is the T(k) intercept really the *whole* setup cost?
//!
//! The primary estimator (`cosx_phase1_split.rs`) fits `T(k) = a + k*b` over one
//! `set_params` plus one sweep. It reports `a` tiny. Before accepting a NO-GO on
//! that basis, this file rules out the two ways that intercept could LIE:
//!
//! 1. **Setup is lazy.** If `set_params` deferred its real work into the first
//!    `compute()`, the per-charge cost `b` would silently contain setup work
//!    that a batched kernel *could* amortize. Probe: time `set_params` alone,
//!    with no sweep at all. If it is a large fraction of a full point's cost,
//!    the intercept is wrong.
//!
//! 2. **The k-sweep amortizes something a real batch could not.** Probe: time
//!    the *actual* COSX access pattern -- k separate `set_params`+sweep pairs,
//!    one point each -- and compare against one k-charge call. The ratio is the
//!    true, end-to-end ceiling on batching, measured rather than inferred, and
//!    it makes no assumption about where the cost sits.
//!
//! Probe 2 is the load-bearing one: it is a direct measurement of the speedup a
//! perfect batched kernel could deliver, with zero model in between.
//!
//! Run with `--ignored --nocapture`.

use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi::{self, CAtom};
use std::hint::black_box;
use std::os::raw::c_int;
use std::time::Instant;

fn testdata(rel: &str) -> String {
    format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"))
}

fn set_charges(eng: &mut Engine, charges: &[CAtom]) {
    let rc = unsafe {
        ffi::scf_engine_set_point_charges(eng.handle_mut(), charges.as_ptr(), charges.len() as c_int)
    };
    assert!(rc >= 0, "set_point_charges failed: {rc}");
}

fn sweep(eng: &mut Engine, prep: &PreparedBasis, sink: &mut f64) {
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    for s1 in 0..nsh {
        for s2 in 0..=s1 {
            let block = eng.compute_1e_block(prep, s1, s2);
            let n = dims[s1] * dims[s2];
            *sink += black_box(block[..n.min(block.len())].iter().sum::<f64>());
        }
    }
}

fn probe_points(n: usize) -> Vec<CAtom> {
    let mut v = Vec::with_capacity(n);
    let mut s = 0x9e37_79b9_7f4a_7c15_u64;
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

fn run_case(label: &str, xyz: &str, basis: &str) {
    let mol = Molecule::load_xyz(&testdata(xyz)).expect("xyz");
    let bs = ferric_core::basis::bundled(basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let mut eng = Engine::new_1e(ffi::OP_NUCLEAR, &prep, 1e-14).expect("engine");
    let mut sink = 0.0_f64;
    let pts = probe_points(32);
    const K: usize = 32;

    for _ in 0..3 {
        set_charges(&mut eng, &pts[..4]);
        sweep(&mut eng, &prep, &mut sink);
    }

    // Probe 1: set_params ALONE, no sweep. Repeat many times; min of means.
    let mut t_setup = f64::INFINITY;
    for _ in 0..7 {
        let t0 = Instant::now();
        for _ in 0..200 {
            set_charges(&mut eng, &pts[..1]);
        }
        t_setup = t_setup.min(t0.elapsed().as_secs_f64() / 200.0);
    }

    // Probe 2a: the REAL COSX pattern -- K independent single-point builds.
    let mut t_serial = f64::INFINITY;
    for _ in 0..5 {
        let t0 = Instant::now();
        for p in pts.iter().take(K) {
            set_charges(&mut eng, std::slice::from_ref(p));
            sweep(&mut eng, &prep, &mut sink);
        }
        t_serial = t_serial.min(t0.elapsed().as_secs_f64());
    }

    // Probe 2b: the best a batched kernel could ever do on THIS engine --
    // one set_params for all K charges, one sweep. (The result is the summed
    // A-matrix, not per-point, so this is a cost floor, not a usable build.)
    let mut t_batch = f64::INFINITY;
    for _ in 0..5 {
        let t0 = Instant::now();
        set_charges(&mut eng, &pts[..K]);
        sweep(&mut eng, &prep, &mut sink);
        t_batch = t_batch.min(t0.elapsed().as_secs_f64());
    }

    let per_pt_serial = t_serial / K as f64;
    let per_pt_batch = t_batch / K as f64;
    let measured_ceiling = t_serial / t_batch;
    let setup_frac = t_setup / per_pt_serial;

    println!("=== {label} / {basis}: nbf={} nsh={} ===", prep.nbasis(), prep.nshells());
    println!("  probe1  set_params alone      = {t_setup:.6e} s");
    println!("  probe2a serial  K={K} pts      = {t_serial:.6e} s  ({per_pt_serial:.6e} s/pt)");
    println!("  probe2b batched K={K} charges  = {t_batch:.6e} s  ({per_pt_batch:.6e} s/pt)");
    println!("  => setup fraction of a real point  f_a = {setup_frac:.4}");
    println!("  => MEASURED end-to-end batching ceiling = {measured_ceiling:.2}x");
    println!("  sink = {sink:.6e}");
}

#[test]
#[ignore = "measurement, not an assertion; run with --ignored --nocapture"]
fn phase1_crosscheck_setup_is_not_the_cost() {
    for (label, xyz) in [
        ("water", "testdata/molecules/water.xyz"),
        ("butane", "testdata/molecules/alkane_4.xyz"),
    ] {
        for basis in ["cc-pvdz", "cc-pvtz"] {
            run_case(label, xyz, basis);
        }
    }
}
