//! `trunc_thresh = 0` means full rank: every static dielectric mode is kept,
//! including one whose λ(0) is exactly 1.0.
//!
//! The old filter `|λ − 1| > trunc_thresh` dropped such a mode at
//! `trunc_thresh = 0`. The energy cannot see the drop (the mode is a null
//! vector of Π(iω) at every ω and contributes ln 1 + 1 − 1 = 0), so this test
//! pins what CAN see it: the returned rank and eigenpotentials.

use ferric_mp2::rimp2::RpaIntermediates;
use ferric_rpa::{run_pdep_rpa_from_parts, PdepRpaConfig};
use ndarray::Array2;

/// naux = 6, nov = 6 with aux row 0 of B identically zero, so e₀ is an exact
/// null vector of Π and ε̃(0) has an exact eigenvalue 1.0 there.
///
/// The EXACT 1.0 relies on LAPACK returning the decoupled row/column of ε̃(0)
/// untouched (it does with the OpenBLAS LAPACK here). A LAPACK that rounds it
/// to 1 ± ε would make the old filter keep it too; the setup asserts in the
/// test turn that into a loud failure rather than a silently vacuous pass.
fn inter_with_a_null_aux_row() -> RpaIntermediates {
    let (naux, nocc, nvir) = (6, 2, 3);
    let b_ov = Array2::from_shape_fn((naux, nocc * nvir), |(p, k)| {
        if p == 0 {
            0.0
        } else {
            0.3 * ((p * p) as f64 * 1.3 + (k * k) as f64 * 0.7 + (p * k) as f64).sin()
        }
    });
    RpaIntermediates {
        b_ov,
        v_inv_sqrt: Array2::eye(naux),
        nocc,
        nvir,
        nocc_total: nocc,
        first_occ: 0,
        naux,
    }
}

#[test]
fn zero_trunc_thresh_keeps_an_exact_unit_eigenvalue_mode() {
    let inter = inter_with_a_null_aux_row();
    let cfg = PdepRpaConfig::default().with_trunc_thresh(0.0);
    let res = run_pdep_rpa_from_parts(&inter, &[-1.0, -0.8], &[0.2, 0.5, 0.9], &cfg).unwrap();
    assert_eq!(
        res.n_eigenpotentials, inter.naux,
        "full rank keeps every mode"
    );
    assert!(
        res.eigenvalues_static.contains(&1.0),
        "setup: the null aux row must give an EXACT λ = 1.0, got {:?}",
        res.eigenvalues_static
    );
    assert_eq!(res.eigenvalues_static.len(), inter.naux);
    let near_one = res
        .eigenvalues_static
        .iter()
        .filter(|l| (*l - 1.0).abs() < 1e-12)
        .count();
    assert_eq!(near_one, 1, "setup: B rows 1..6 must be full rank");
    assert_eq!(res.eigenpotentials.dim(), (inter.naux, inter.naux));
    assert!(
        res.e_rpa.is_finite() && res.e_rpa < 0.0,
        "E_c {}",
        res.e_rpa
    );
}
