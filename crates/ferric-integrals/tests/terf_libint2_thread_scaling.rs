//! #234 measurement: thread scaling of 3-centre integrals through libint2's
//! `GenericGmEval` (terf via the terf-patched prefix, erfc) against plain
//! Coulomb, which never goes through `GenericGmEval`.
//!
//! Stock `GenericGmEval::eval` copy-constructs the core evaluator on every
//! primitive quartet. For erfc that is an atomic refcount inc/dec on the
//! process-wide Boys engine plus a heap alloc/free of a scratch vector; for the
//! terf patch's `terf_gm_eval` it is a heap alloc/free of a vector it never
//! uses. If that copy contends, the per-thread time of a FIXED amount of work
//! (every thread does the same full sweep with its own engine, no shared
//! output) grows with the thread count for erfc/terf but not for Coulomb.
//!
//! Prints, per operator and thread count, the best-of-`REPS` wall time of one
//! sweep per thread, the inflation against 1 thread, the aggregate speed-up,
//! and an FNV-1a hash of every integral of the sweep (bitwise; compare it
//! between builds to show a header change is bit-identical on this workload).
//!
//! PRECONDITION: a quiet box (no other heavy job), release build, one test at
//! a time, OPENBLAS_NUM_THREADS=1. Feature `libint2_terf` against a
//! terf-patched libint2 with FERRIC_LIBINT2_TERF set, like the other
//! terf_libint2_* tests. Run:
//! `cargo test --release -p ferric-integrals --features libint2_terf --test
//! terf_libint2_thread_scaling -- --ignored --nocapture --test-threads 1`
//! Knobs: FERRIC_SCALING_THREADS (default 1,2,4,6), FERRIC_SCALING_REPS (3),
//! FERRIC_SCALING_MOLECULE (benzene), FERRIC_SCALING_SWEEPS (1).
#![cfg(feature = "libint2_terf")]

use ferric_core::{basis, mol::Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::{Engine, TerfLibint2Braket};
use ferric_integrals::operator::Operator;
use std::sync::Barrier;
use std::time::Instant;

fn env_or<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn table_dir() -> String {
    std::env::var("FERRIC_TERF_TABLE_DIR")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../terf-tables").into())
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Coulomb,
    Erfc,
    Terf,
}

const R0: f64 = 2.0;
const OMEGA: f64 = 0.4;

fn make_engine(op: Op, obs: &PreparedBasis, dfbs: &PreparedBasis) -> Engine {
    match op {
        Op::Coulomb => Engine::new_3center(Operator::coulomb(), obs, dfbs, 0.0).unwrap(),
        Op::Erfc => Engine::new_3center(Operator::erfc(OMEGA), obs, dfbs, 0.0).unwrap(),
        Op::Terf => Engine::new_terf_libint2(
            R0,
            R0.recip() / 2f64.sqrt(),
            TerfLibint2Braket::XsXx,
            obs.max_nprim().max(dfbs.max_nprim()) as usize,
            obs.max_l().max(dfbs.max_l()) as i32,
            0.0,
            Some(table_dir().as_str()),
        )
        .unwrap(),
    }
}

/// One full (P|s1 s2), s2 <= s1, sweep; returns the FNV-1a 64 of every value.
fn sweep(eng: &mut Engine, obs: &PreparedBasis, dfbs: &PreparedBasis) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let (nd, no) = (dfbs.shell_dims().len(), obs.shell_dims().len());
    for p in 0..nd {
        for s1 in 0..no {
            for s2 in 0..=s1 {
                let Some(block) = eng.compute_eri3(obs, dfbs, p, s1, s2) else {
                    continue;
                };
                for v in block {
                    for b in v.to_bits().to_le_bytes() {
                        h ^= u64::from(b);
                        h = h.wrapping_mul(0x0000_0100_0000_01b3);
                    }
                }
            }
        }
    }
    h
}

/// Every thread builds its own engine, waits at the barrier, then runs
/// `sweeps` full sweeps. Returns the slowest thread's seconds per sweep and
/// the hash every thread must agree on.
fn run(op: Op, n: usize, sweeps: usize, obs: &PreparedBasis, dfbs: &PreparedBasis) -> (f64, u64) {
    let barrier = Barrier::new(n);
    let res: Vec<(f64, u64)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..n)
            .map(|_| {
                s.spawn(|| {
                    let mut eng = make_engine(op, obs, dfbs);
                    barrier.wait();
                    let t = Instant::now();
                    let mut h = 0;
                    for _ in 0..sweeps {
                        h = sweep(&mut eng, obs, dfbs);
                    }
                    (t.elapsed().as_secs_f64() / sweeps as f64, h)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let h0 = res[0].1;
    assert!(res.iter().all(|r| r.1 == h0), "{op:?}: threads disagree");
    (res.iter().map(|r| r.0).fold(0.0, f64::max), h0)
}

#[test]
#[ignore = "timing: quiet box, release, --test-threads 1 (see module doc)"]
fn thread_scaling_of_generic_gm_eval_operators() {
    let threads: Vec<usize> = std::env::var("FERRIC_SCALING_THREADS")
        .unwrap_or_else(|_| "1,2,4,6".into())
        .split(',')
        .map(|t| t.trim().parse().unwrap())
        .collect();
    let reps: usize = env_or("FERRIC_SCALING_REPS", 3);
    let sweeps: usize = env_or("FERRIC_SCALING_SWEEPS", 1);
    let molname = std::env::var("FERRIC_SCALING_MOLECULE").unwrap_or_else(|_| "benzene".into());
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{molname}.xyz",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    println!(
        "SCALING {molname}/cc-pvdz+cc-pvdz-ri, {} obs x {} aux shells, sweeps/thread {sweeps}, best of {reps}, \
         libint2 boys.h override: {:?}",
        obs.shell_dims().len(),
        dfbs.shell_dims().len(),
        option_env!("FERRIC_LIBINT2_BOYS_OVERRIDE")
    );
    for op in [Op::Coulomb, Op::Erfc, Op::Terf] {
        // best[k] over reps, thread counts interleaved within each rep
        let mut best = vec![f64::INFINITY; threads.len()];
        let mut hash = None;
        for _ in 0..reps {
            for (k, &n) in threads.iter().enumerate() {
                let (t, h) = run(op, n, sweeps, &obs, &dfbs);
                assert_eq!(
                    *hash.get_or_insert(h),
                    h,
                    "{op:?}: hash changed with threads"
                );
                best[k] = best[k].min(t);
            }
        }
        let t1 = best[0];
        for (k, &n) in threads.iter().enumerate() {
            println!(
                "SCALING op={op:?} threads={n} s_per_sweep={:.4} inflation={:.2} speedup={:.2} hash={:016x}",
                best[k],
                best[k] / t1,
                n as f64 * t1 / best[k] * (threads[0] as f64).recip(),
                hash.unwrap()
            );
        }
    }
}
