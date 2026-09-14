//! The operator-split anchor for the 4-center terfc engine: `terf + terfc`
//! must reconstruct the plain Coulomb quartet, element-wise.
//!
//! What this DOES prove: the terf and terfc 4-center paths are both live, both
//! use the tables, and their decomposition of the Coulomb kernel is exact.
//!
//! What this does NOT prove: the correctness of the MD contraction itself. terf
//! and terfc share `compute_cart_eri4`, so a sign, prefactor or layout error
//! cancels in the sum and the identity still holds. That job belongs to
//! `eri4_vs_libint.rs`, which checks the same machinery against libint2 -- an
//! independent construction.
//!
//! Because of that cancellation, a bare `terf + terfc == coulomb` assertion is
//! a candidate for "a passing test measuring inertness": it holds trivially if
//! `compute_cart_eri4` returned zeros for terf (then terfc == coulomb). So this
//! test ALSO asserts that terf carries real weight -- a meaningful fraction of
//! the Coulomb block -- which is what makes the identity load-bearing.

use ferric_core::{basis, mol::Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_integrals::operator::Operator;
use std::os::raw::{c_int, c_void};

fn tables_available() -> bool {
    std::env::var("FERRIC_TERF_TABLE_DIR").is_ok()
}

fn water() -> PreparedBasis {
    let mol = Molecule::load_xyz(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/water.xyz"
    ))
    .unwrap();
    PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap()
}

/// Call one of the 4-center terf/terfc entry points through the raw FFI.
/// `Engine::compute_quartet` has no terfc branch yet, so this is the only way
/// to reach the new path from Rust.
fn eri4_via(
    eng: &mut Engine,
    prep: &PreparedBasis,
    q: [usize; 4],
    complement: bool,
) -> Option<Vec<f64>> {
    let dims = prep.shell_dims();
    let n: usize = q.iter().map(|&s| dims[s]).product();
    let mut out = vec![0.0f64; n];
    let h = eng.handle_mut();
    // SAFETY: `h` is a live terf/terfc engine handle, `prep.handle()` a valid
    // scf_basis, the shell indices index prep.shell_dims(), and `out` is sized
    // n1*n2*n3*n4 as the shim contract requires. Status checked before reading.
    let written = unsafe {
        let obs = prep.handle() as *const c_void;
        let (a, b, c, d) = (q[0] as c_int, q[1] as c_int, q[2] as c_int, q[3] as c_int);
        if complement {
            ffi::scf_compute_terf_eri4(h, obs, a, b, c, d, out.as_mut_ptr())
        } else {
            ffi::scf_compute_terfc_eri4(h, obs, a, b, c, d, out.as_mut_ptr())
        }
    };
    if written < 0 {
        return None;
    }
    assert_eq!(written as usize, n, "shim wrote an unexpected block size");
    Some(out)
}

#[test]
fn terf4_plus_terfc4_reconstructs_the_coulomb_quartet() {
    if !tables_available() {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR unset (terf tables are untracked data)");
        return;
    }
    let prep = water();
    let r0 = 2.0_f64;

    let mut e_terf = Engine::new_2center(Operator::terf(r0), &prep, 0.0).unwrap();
    let mut e_terfc = Engine::new_2center(Operator::terfc(r0), &prep, 0.0).unwrap();
    let mut e_coul = Engine::new_2e(Operator::coulomb(), &prep, 0.0).unwrap();

    let dims = prep.shell_dims();
    let nsh = dims.len();
    // A handful of quartets spanning s/p/d on distinct atoms.
    let mut quartets: Vec<[usize; 4]> = Vec::new();
    for &a in &[0usize, nsh / 3, nsh / 2] {
        for &c in &[0usize, nsh / 2, nsh - 1] {
            quartets.push([a, nsh / 2, c, nsh - 1]);
        }
    }
    quartets.retain(|q| q.iter().all(|&s| s < nsh));

    let mut checked = 0usize;
    let mut worst_resid = 0.0f64;
    let mut best_terf_weight = 0.0f64;

    for q in quartets {
        let (Some(terf), Some(terfc)) = (
            eri4_via(&mut e_terf, &prep, q, true),
            eri4_via(&mut e_terfc, &prep, q, false),
        ) else {
            continue;
        };
        let coul = e_coul
            .compute_quartet(&prep, q[0], q[1], q[2], q[3])
            .map(|b| b.to_vec())
            .unwrap_or_else(|| vec![0.0; terf.len()]);
        assert_eq!(terf.len(), coul.len(), "block size mismatch at {q:?}");

        let scale = coul.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if scale < 1e-12 {
            continue; // nothing to resolve in this block
        }
        let terf_weight = terf.iter().fold(0.0f64, |m, v| m.max(v.abs())) / scale;
        best_terf_weight = best_terf_weight.max(terf_weight);

        for i in 0..coul.len() {
            let resid = (terf[i] + terfc[i] - coul[i]).abs() / scale;
            worst_resid = worst_resid.max(resid);
            assert!(
                resid < 1e-12,
                "terf4 + terfc4 != coulomb4 at {q:?} element {i}: \
                 terf={:.17e} terfc={:.17e} coul={:.17e} resid={resid:.3e}",
                terf[i],
                terfc[i],
                coul[i]
            );
        }
        checked += 1;
    }

    assert!(
        checked >= 3,
        "only {checked} quartets compared -- the identity proved almost nothing"
    );
    // THE ANTI-INERTNESS ASSERTION. Without this, an all-zero terf block would
    // satisfy the identity trivially (terfc would simply equal coulomb) and the
    // test would pass while measuring nothing.
    assert!(
        best_terf_weight > 1e-3,
        "terf contributes only {best_terf_weight:.3e} of the Coulomb block -- \
         the identity is holding trivially, so it is not evidence the terf \
         4-center path computes anything"
    );
    eprintln!(
        "terf4+terfc4==coulomb4: {checked} quartets, worst resid {worst_resid:.3e}, \
         max terf weight {best_terf_weight:.3e}"
    );
}
