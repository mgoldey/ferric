//! STEP 4 gate: does the libint2-native terf path reproduce the MD path?
//!
//! Only builds/runs under FERRIC_LIBINT2_TERF against a patched libint2.
//! These are NOT expected to be bit-identical: libint2's generated recurrences
//! sum in a different order from the hand-rolled McMurchie-Davidson driver. So
//! the bar is a stated TOLERANCE, fixed here before any number was measured.
#![cfg(feature = "libint2_terf")]

use ferric_core::{basis, mol::Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::ffi;
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int, c_void};

#[test]
fn libint2_terf_matches_md_within_tolerance() {
    let Ok(dir) = std::env::var("FERRIC_TERF_TABLE_DIR") else {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR unset");
        return;
    };
    let mol = Molecule::load_xyz(concat!(
        env!("CARGO_MANIFEST_DIR"), "/../../testdata/molecules/water.xyz")).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let cdir = CString::new(dir).unwrap();
    let r0 = 2.0_f64;
    let omega = 1.0 / (r0 * 2.0_f64.sqrt());

    let (mut worst_rel, mut worst_at) = (0.0f64, (0usize, 0usize, 0usize));
    let mut checked = 0usize;
    let nsh_o = obs.shell_dims().len();
    let nsh_d = dfbs.shell_dims().len();
    for p in (0..nsh_d).step_by((nsh_d / 6).max(1)) {
        for s1 in (0..nsh_o).step_by((nsh_o / 4).max(1)) {
            for s2 in 0..=s1 {
                let (mut ma, mut mr) = (0.0f64, 0.0f64);
                // SAFETY: valid bases, in-bounds shells, two live out-params.
                let n = unsafe {
                    ffi::scf_terf_libint2_vs_md_eri3(
                        obs.handle() as *const c_void, dfbs.handle() as *const c_void,
                        p as c_int, s1 as c_int, s2 as c_int, r0, omega,
                        cdir.as_ptr() as *const c_char, &mut ma, &mut mr)
                };
                assert!(n > 0, "gate status {n} at ({p},{s1},{s2})");
                if mr > worst_rel { worst_rel = mr; worst_at = (p, s1, s2); }
                checked += 1;
            }
        }
    }
    assert!(checked > 10, "only {checked} blocks compared");
    eprintln!("libint2 terf vs MD: {checked} blocks, worst rel {worst_rel:.3e} at {worst_at:?}");
    // Tolerance fixed BEFORE measuring: recurrence-order differences should sit
    // near double rounding on a well-conditioned block. 1e-10 is loose enough
    // for reassociation, tight enough that a wrong kernel cannot pass.
    // STATUS 2026-09-15: FAILING at worst rel 1.989e2 -- a ~200x scale error,
    // not a rounding difference, so the kernel is wrong and this gate is doing
    // its job. Prime suspect: PREFACTOR DOUBLE-COUNTING. libint2 multiplies the
    // core-eval output by `pfac = K12 * sqrt_gammapq * oogammapq` AFTER the
    // evaluator returns (engine.impl.h:1288, :1417), whereas terf_aux already
    // folds in `phi_over_theta = sqrt(phi2/rho)`. The MD driver applies its own
    // `2*pi^2.5/(p*q*sqrt(p+q))` separately. The hook must return values in
    // libint2's convention -- i.e. WITHOUT any prefactor libint2 will apply
    // itself -- so the next step is to work out which factors belong where and
    // strip the duplicates. Compare against erfc_coulomb_gm_eval, which returns
    // raw Gm and lets libint2 scale.
    assert!(worst_rel < 1e-10,
        "libint2 terf disagrees with MD by {worst_rel:.3e} at {worst_at:?}");
}
