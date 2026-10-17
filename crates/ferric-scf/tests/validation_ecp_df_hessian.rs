//! VALIDATION tier: analytic Hessians with an ECP and density-fitted J/K
//! (issue #406): the slow routes against FD of ferric's fitted gradient, and
//! the PySCF comparisons.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test validation_ecp_df_hessian \
//!     -- --ignored --test-threads=2
//! ```
//!
//! References: `scripts/validation/gen_ecp_df_hessian.py` ->
//! `testdata/reference/validation/ecp_df_hessian/`. PySCF `density_fit(aux)`
//! with ferric's own basis, ECP and aux JSON, exact-equivalent geometry:
//! `hessian_analytic` (`mf.Hessian()`, full aux response; RHF and UHF),
//! `hessian_fd` (central FD of PySCF's analytic DF gradient; the ONLY PySCF
//! reference for ROHF, which has no PySCF Hessian), `hessian_exact_jk` (the
//! exact-integral Hessian: the negative control, its distance from the fitted
//! one is `exact_vs_ri_jk_hessian_max_abs`).
//!
//! CONDITIONING. `def2-universal-jkfit` on iodine has an RI-K metric of cond
//! 4.9e8 (HI) to 1.4e9 (CH3I) and 6e8 to 1.1e9 (HIH, HOI), above the code's
//! `RIK_COND_WARN` = 1e8 where accuracy is "no longer established". The `jkfit`
//! tests below are therefore evidence AT that limit (the agreement is
//! nevertheless at the floor), not within the validated range; the `rifit`
//! tests (`def2-svp-rifit`, probe-accepted) are the within-range evidence.
//!
//! Bars: see `ecp_df_common` (FD) and `TOL_PYSCF`; the PySCF-analytic bar is
//! raised to twice the generator's measured PySCF analytic-vs-FD gap where
//! that gap exceeds it (PySCF's own ECP second-derivative roughness).

mod ecp_df_common;

use std::path::PathBuf;

use ecp_df_common::*;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ndarray::Array2;

#[test]
#[ignore = "validation: slow FD-of-gradient / PySCF comparison on iodine"]
fn rhf_ri_j_ecp_fd() {
    check_against_fd(
        "HOI RHF RI-J",
        HOI,
        1,
        Reference::Rhf,
        Fit::J(AUX_OK),
        &[1, 8],
    );
}

#[test]
#[ignore = "validation: slow FD-of-gradient / PySCF comparison on iodine"]
fn rhf_ri_k_ecp_fd() {
    check_against_fd(
        "HOI RHF RI-K",
        HOI,
        1,
        Reference::Rhf,
        Fit::K(AUX_OK),
        &[2, 7],
    );
}

/// jkfit on iodine: cond ~1e9, above the validated range (see module docs).
#[test]
#[ignore = "validation: slow FD-of-gradient / PySCF comparison on iodine"]
fn rhf_ri_jk_ecp_fd_above_the_conditioning_limit() {
    check_against_fd(
        "HOI RHF RI-JK jkfit (cond>1e8)",
        HOI,
        1,
        Reference::Rhf,
        Fit::Jk(AUX_HIGH_COND),
        &[0, 4, 8],
    );
}

#[test]
#[ignore = "validation: slow FD-of-gradient / PySCF comparison on iodine"]
fn uhf_ri_j_ecp_fd() {
    check_against_fd(
        "HIH UHF RI-J",
        HIH,
        2,
        Reference::Uhf,
        Fit::J(AUX_OK),
        &[0, 5, 8],
    );
}

#[test]
#[ignore = "validation: slow FD-of-gradient / PySCF comparison on iodine"]
fn uhf_ri_k_ecp_fd() {
    check_against_fd(
        "HIH UHF RI-K",
        HIH,
        2,
        Reference::Uhf,
        Fit::K(AUX_OK),
        &[0, 5, 8],
    );
}

#[test]
#[ignore = "validation: slow FD-of-gradient / PySCF comparison on iodine"]
fn uhf_ri_jk_ecp_fd_above_the_conditioning_limit() {
    check_against_fd(
        "HIH UHF RI-JK jkfit (cond>1e8)",
        HIH,
        2,
        Reference::Uhf,
        Fit::Jk(AUX_HIGH_COND),
        &[0, 5, 8],
    );
}

#[test]
#[ignore = "validation: slow FD-of-gradient / PySCF comparison on iodine"]
fn rohf_ri_k_ecp_fd() {
    check_against_fd(
        "HIH ROHF RI-K",
        HIH,
        2,
        Reference::Rohf,
        Fit::K(AUX_OK),
        &[0, 5, 8],
    );
}

#[test]
#[ignore = "validation: slow FD-of-gradient / PySCF comparison on iodine"]
fn rohf_ri_jk_ecp_fd() {
    check_against_fd(
        "HIH ROHF RI-JK",
        HIH,
        2,
        Reference::Rohf,
        Fit::Jk(AUX_OK),
        &[0, 5, 8],
    );
}

// ---------------------------------------------------------------------------
// PySCF
// ---------------------------------------------------------------------------

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .to_path_buf()
}

fn matrix(v: &serde_json::Value, key: &str) -> Option<Array2<f64>> {
    let rows = v[key].as_array()?;
    let n = rows.len();
    Some(Array2::from_shape_fn((n, n), |(i, j)| {
        rows[i][j].as_f64().unwrap()
    }))
}

/// `system` is the reference name, `stem` the geometry file, `aux` its fit.
fn check_pyscf(system: &str, stem: &str, mult: usize, r: Reference, aux: &'static str) {
    if !has_deriv2() {
        eprintln!("{system}: skipped, libint2 without second derivatives");
        return;
    }
    let path = root().join(format!(
        "testdata/reference/validation/ecp_df_hessian/{system}_def2-svp.json"
    ));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing reference {} ({e})", path.display()));
    let j: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(j["aux_basis"].as_str(), Some(aux), "{system}: aux basis");
    let xyz = root().join(format!("testdata/molecules/validation/ecp/{stem}.xyz"));
    let mut mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, mult).unwrap();
    let bs = basis::bundled("def2-svp").unwrap();
    mol.apply_ecp(&bs);
    let fit = Fit::Jk(aux);
    let s = solve(&mol, &bs, r, fit);
    let e_ref = j["energy"].as_f64().unwrap();
    assert!(
        (s.scf.energy - e_ref).abs() < 2.5e-7,
        "{system}: state/energy {} vs {e_ref}",
        s.scf.energy
    );
    let h = hessian(&s, r, fit);
    let h_fd = matrix(&j, "hessian_fd").unwrap();
    let d_fd = max_diff(&h, &h_fd);
    eprintln!("{system}: vs PySCF FD {d_fd:.2e} (bar {TOL_PYSCF:.0e})");
    assert!(d_fd < TOL_PYSCF, "{system}: vs PySCF FD {d_fd:.3e}");

    let Some(h_an) = matrix(&j, "hessian_analytic") else {
        eprintln!("{system}: no PySCF analytic Hessian for {r:?}; FD reference only");
        return;
    };
    let gap = j["pyscf_fd_vs_analytic_hessian_max_abs"].as_f64().unwrap();
    let bar = TOL_PYSCF.max(2.0 * gap);
    let d_an = max_diff(&h, &h_an);
    eprintln!(
        "{system}: vs PySCF analytic {d_an:.2e} (bar {bar:.1e}; PySCF FD-vs-analytic {gap:.1e})"
    );
    assert!(d_an < bar, "{system}: vs PySCF analytic {d_an:.3e}");

    // Negative control: the exact-integral Hessian of the exact SCF misses this
    // reference by the fitting signal, which is above the bar.
    let signal = j["exact_vs_ri_jk_hessian_max_abs"].as_f64().unwrap();
    let ex = solve(&mol, &bs, r, Fit::Exact);
    let miss = max_diff(&hessian(&ex, r, Fit::Exact), &h_an);
    eprintln!(
        "  {system}: exact-integral Hessian misses by {miss:.3e} (PySCF fit signal {signal:.2e})"
    );
    assert!(
        miss > MUST_MISS * bar && miss > 3.0 * d_an,
        "{system}: control vacuous"
    );
}

#[test]
#[ignore = "validation: slow FD-of-gradient / PySCF comparison on iodine"]
fn hi_ri_jk_ecp_matches_pyscf_above_the_conditioning_limit() {
    check_pyscf("hi", "hi", 1, Reference::Rhf, AUX_HIGH_COND);
}

#[test]
#[ignore = "validation: slow FD-of-gradient / PySCF comparison on iodine"]
fn ch3i_ri_jk_ecp_matches_pyscf_above_the_conditioning_limit() {
    check_pyscf("ch3i", "ch3i", 1, Reference::Rhf, AUX_HIGH_COND);
}

#[test]
#[ignore = "validation: slow FD-of-gradient / PySCF comparison on iodine"]
fn hi_ri_jk_ecp_matches_pyscf_rifit() {
    check_pyscf("hi_rifit", "hi", 1, Reference::Rhf, AUX_OK);
}

#[test]
#[ignore = "validation: slow FD-of-gradient / PySCF comparison on iodine"]
fn ch3i_ri_jk_ecp_matches_pyscf_rifit() {
    check_pyscf("ch3i_rifit", "ch3i", 1, Reference::Rhf, AUX_OK);
}

#[test]
#[ignore = "validation: slow FD-of-gradient / PySCF comparison on iodine"]
fn ch2i_uhf_ri_jk_ecp_matches_pyscf() {
    check_pyscf("ch2i_uhf", "ch2i", 2, Reference::Uhf, AUX_OK);
}

#[test]
#[ignore = "validation: slow FD-of-gradient / PySCF comparison on iodine"]
fn ch2i_rohf_ri_jk_ecp_matches_pyscf_fd() {
    check_pyscf("ch2i_rohf", "ch2i", 2, Reference::Rohf, AUX_OK);
}
