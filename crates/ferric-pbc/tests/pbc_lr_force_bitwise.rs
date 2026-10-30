//! END-TO-END bit-identity of the Gamma RS-GDF force's LR (G ≠ 0) derivative
//! pass (`rsgdf::deriv`: orbital / aux-centre / metric pieces) and of the full
//! force built on it, across rayon thread counts.
//!
//! The LR pass evaluates its G chunks in parallel (pair-FT derivative
//! producer, GEMMs and per-G contractions per chunk) and replays the
//! accumulation into the shared force rows serially in chunk order
//! (`ferric_pbc::ordered`), so the result is bit-identical to the serial
//! pass at any thread count.
//!
//! The CI-run tests compare, IN PROCESS on the machine running them, the force
//! (`grad` and its fit parts `fit_orb_lr`, `fit_aux_lr`, `fit_metric_lr`, as
//! FNV-1a hashes of `f64::to_bits`) at 2 and 6 rayon threads against the same
//! code's 1-thread result, with the range split off / on and the SR column
//! rotation on. Absolute bits depend on the libint build and the CPU's BLAS
//! dispatch, so none is pinned there. The comparison against the FROZEN serial
//! LR pass is `rsgdf::deriv::lr_tests` (in-crate oracle). The hashes recorded
//! from the pre-parallel code on the development box (all five cases, rotation
//! on / off, split on / off) are `dev_box_goldens`, `--ignored` only. The
//! cases run with a small `[memory]` budget so the LR pass really splits into
//! many chunks (asserted).

//!
//! The 1/2/6-thread split case and the cc-pVDZ split+rotated case run weekly
//! in `validation_pbc_lr_force_bitwise.rs` (shared harness:
//! `lr_force_bitwise_support/mod.rs`); here the STO-3G unsplit case at 2/6
//! threads and the STO-3G SPLIT case at 2 threads are the per-commit
//! canaries of the LR derivative pass's thread invariance.

mod common;
#[path = "lr_force_bitwise_support/mod.rs"]
mod support;

use support::*;

/// Prints the goldens (run once on the serial pass).
#[test]
#[ignore]
fn print_goldens() {
    for c in &CASES {
        let g = force(&setup(c), c);
        let nc = g.fit.as_ref().unwrap().n_g_chunks;
        eprintln!("GOLDEN {}: {:016x?} chunks {nc}", c.name, hashes(&g));
    }
}

#[test]
fn lr_force_sto3g_unsplit_is_thread_invariant() {
    check(0, &[2, 6]);
}

/// Canary of `lr_force_sto3g_split_is_thread_invariant` (weekly): the
/// range-split LR passes at 2 threads vs 1.
#[test]
fn lr_force_sto3g_split_two_threads_canary() {
    check(1, &[2]);
}

/// The hashes recorded from the serial (pre-parallel) pass on the development
/// box, all five cases at 1 / 2 / 6 threads.
#[test]
#[ignore = "dev-box goldens (libint 2.13.1, this CPU): run with --ignored on the recording machine"]
fn dev_box_goldens() {
    for c in &CASES {
        let fx = setup(c);
        for &n in &THREADS {
            let g = in_pool(n, || force(&fx, c));
            assert_eq!(hashes(&g), c.golden, "{} at {n} threads", c.name);
        }
    }
}
