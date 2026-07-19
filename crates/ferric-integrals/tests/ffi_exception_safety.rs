//! FFI exception-safety regression test for the libint2 shim (`shim/shim.cc`).
//!
//! Every `scf_compute_*` function wraps its libint2 call(s) in `try { ... }
//! catch (...)` and returns a negative sentinel (`SCF_EINTERNAL = -3`, or `-1`
//! for `scf_compute_dipole`) on any C++ exception, because a throw that unwinds
//! across the `extern "C"` boundary into Rust is undefined behavior (the Rust
//! declarations are `extern "C"`, not `extern "C-unwind"`). See the FFI
//! exception-safety convention in CLAUDE.md and Lane 1 item 1 of
//! docs/reliability-audit-2026-07-06.md.
//!
//! This test drives libint2 into a *genuine* throwing situation and confirms the
//! throw is caught at the C ABI boundary and surfaced as a clean Rust panic (via
//! the `assert!(written >= 0)` in `Engine::compute_1e_block`) instead of UB / a
//! process abort.
//!
//! Trigger: this libint2 build is compiled with `LIBINT_MAX_AM 6` (i functions;
//! see /home/matt/.local/include/libint2/config.h). Asking an engine to compute
//! integrals over a shell with angular momentum L > 6 (here L = 7, a "k" shell)
//! makes libint2 throw `Engine::lmax_exceeded` (a `std::logic_error` subclass;
//! see engine.h:849 and the `throw Engine::lmax_exceeded(...)` in
//! engine.impl.h:610). Pre-fix that throw was UB; post-fix the shim catches it
//! and returns SCF_EINTERNAL, which the Rust wrapper turns into a panic.
//!
//! Note on scope: the throw may occur either at engine *construction* (which the
//! creation wrapper already caught pre-audit — returns null → `Err`) or at
//! *compute* time (the path this audit item fixed). Both are exercised here and
//! both must be a clean Rust-visible failure, never a crash. The panic path is
//! the one that proves the *compute* try/catch specifically.

use ferric_core::basis::{BasisSet, Shell};
use ferric_core::mol::{Atom, Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use std::collections::HashMap;

fn one_atom_mol() -> Molecule {
    Molecule {
        atoms: vec![Atom {
            symbol: "H".to_string(),
            z: 1,
            x: 0.0,
            y: 0.0,
            zpos: 0.0,
            ghost: false,
            n_core_ecp: 0,
        }],
        charge: 0,
        multiplicity: 2,
    }
}

/// Build a one-atom basis with a single shell of angular momentum `l`.
fn single_shell_basis(z: i32, l: i32) -> BasisSet {
    let mut shells: HashMap<i32, Vec<Shell>> = HashMap::new();
    shells.insert(
        z,
        vec![Shell {
            l,
            pure: true,
            exponents: vec![1.0],
            coefficients: vec![1.0],
        }],
    );
    BasisSet {
        name: format!("single-l{l}"),
        shells,
        ecps: HashMap::new(),
    }
}

/// Sanity anchor: an in-range shell (L = 2, a d shell) computes an overlap block
/// cleanly. This guards against the "trigger" test passing for the wrong reason
/// (e.g. a panic from an unrelated setup bug rather than from the caught throw).
#[test]
fn in_range_shell_overlap_is_clean() {
    unsafe { ffi::scf_libint_init() };
    let mol = one_atom_mol();
    let bs = single_shell_basis(1, 2); // d shell, well within LIBINT_MAX_AM 6
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let mut eng = Engine::new_1e(ffi::OP_OVERLAP, &prep, 1e-14).unwrap();
    let block = eng.compute_1e_block(&prep, 0, 0);
    // A d-shell self-overlap block is 5x5 (spherical) = 25 entries, and the
    // diagonal is 1 for unit-self-overlap-normalized shells. We only assert it
    // is the right size and finite — the numbers themselves are covered by the
    // oneelectron.rs unit tests; here we only need "the happy path still works".
    assert_eq!(block.len(), 25, "d-shell overlap block must be 5x5");
    assert!(block.iter().all(|v| v.is_finite()), "overlap must be finite");
}

/// The load-bearing test: a shell above the compiled `LIBINT_MAX_AM` must NOT
/// cause UB. libint2 throws `lmax_exceeded`; the shim must catch it. We accept
/// either resolution as "safe":
///   (a) engine creation returns null → `Engine::new_1e` returns `Err`, or
///   (b) engine creation succeeds and `compute_1e_block` hits the shim's caught
///       exception → negative status → the wrapper's `assert!` panics.
/// A process abort / segfault (the pre-fix UB) fails the test by never returning.
#[test]
fn over_max_am_shell_is_caught_not_ub() {
    unsafe { ffi::scf_libint_init() };
    let mol = one_atom_mol();
    let bs = single_shell_basis(1, 7); // L = 7 (k shell) > LIBINT_MAX_AM 6

    // PreparedBasis construction itself does not call libint2 compute; it should
    // succeed (it just records shell metadata).
    let prep = PreparedBasis::new(&mol, &bs).expect("PreparedBasis::new should not fail on an L=7 shell");

    match Engine::new_1e(ffi::OP_OVERLAP, &prep, 1e-14) {
        Err(_) => {
            // Path (a): libint2 threw at construction, the creation wrapper
            // caught it and returned null → clean Err. Exception never crossed
            // the ABI as an unwind. This is safe.
        }
        Ok(mut eng) => {
            // Path (b): construction succeeded; the throw must happen at compute
            // and be caught by the compute-path try/catch, surfacing as a panic.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = eng.compute_1e_block(&prep, 0, 0);
            }));
            assert!(
                result.is_err(),
                "compute over an L=7 shell must panic (caught libint2 throw \
                 → SCF_EINTERNAL → assert), not return normally or abort"
            );
        }
    }
}
