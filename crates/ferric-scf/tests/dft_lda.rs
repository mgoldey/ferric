//! LDA validation against PySCF reference energies.
//!
//! For each {molecule, cc-pVDZ, LDA(VWN)} tuple, runs ferric with
//! Becke-Lebedev (75, 110) grid and conv_tol=1e-10, then compares total
//! energy to the PySCF reference loaded from testdata/reference/.
//!
//! # Reference alignment
//!
//! References are generated with three ferric-matching settings:
//!   * `mf.density_fit(auxbasis="def2-universal-jkfit")` — matches ferric's RI-J.
//!   * `mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust` — matches
//!     ferric's Becke (1988) size correction (PySCF's default is Treutler).
//!   * Lebedev-110 angular quadrature uses the canonical Lebedev-Laikov
//!     parameters (b3 at p=0.3956894730559419, c-orbit at p=0.4783690288121502).
//!
//! With all three aligned, observed errors are:
//!
//! * H2    ≈ 1.0e-6 Ha
//! * H2O   ≈ 2.0e-6 Ha
//! * CH4   ≈ 3.0e-6 Ha
//!
//! These are single-digit µHa — essentially the noise floor of a (75,110)
//! Becke-Lebedev grid. PySCF's own electron-count integration on this grid
//! is only 7.5e-6 accurate, so this is the irreducible grid error.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Ref {
    e_total: f64,
    // Present in the reference JSON for provenance; not asserted on.
    #[serde(default)]
    #[allow(dead_code)]
    xc: String,
    #[serde(default)]
    #[allow(dead_code)]
    label: String,
    #[serde(default)]
    converged: bool,
}

/// Tolerance is set just above the measured worst-case (CH4 ≈ 3.0e-6 Ha) to
/// guard against drift in either the SCF implementation or grid construction.
/// PySCF itself only integrates electrons on this (75,110) grid to ~7e-6,
/// so 1e-5 is roughly the noise floor — tightening further would just guard
/// roundoff. Do NOT silently widen this; any drift indicates a real regression.
const TOL: f64 = 1e-5;

fn ref_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/reference")
        .join(name)
}

fn run_case(label: &str, xyz: &str, expected_file: &str) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();

    let cfg = RhfConfig {
        xc: Some("LDA".into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        // LDA has no exact exchange — no df_k_aux needed.
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    };
    let res = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).unwrap();

    let r: Ref =
        serde_json::from_str(&fs::read_to_string(ref_path(expected_file)).unwrap()).unwrap();
    assert!(
        r.converged,
        "PySCF reference {} not converged",
        expected_file
    );

    let err = (res.energy - r.e_total).abs();
    eprintln!(
        "[{label}] ferric = {:.10} Ha,  PySCF = {:.10} Ha,  err = {err:.2e}",
        res.energy, r.e_total
    );
    assert!(
        err < TOL,
        "LDA E_total mismatch for {label}: err = {err:.2e} (ferric={:.10}, pyscf={:.10})",
        res.energy,
        r.e_total
    );
}

#[test]
fn lda_h2() {
    run_case("H2", "2\nH2\nH 0 0 0\nH 0 0 0.74\n", "h2_cc-pvdz_lda.json");
}

#[test]
fn lda_water() {
    run_case(
        "H2O",
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
        "h2o_cc-pvdz_lda.json",
    );
}

#[test]
fn lda_methane() {
    let xyz = "5\nCH4\nC 0 0 0\n\
               H 0.6276 0.6276 0.6276\n\
               H -0.6276 -0.6276 0.6276\n\
               H -0.6276 0.6276 -0.6276\n\
               H 0.6276 -0.6276 -0.6276\n";
    run_case("CH4", xyz, "methane_cc-pvdz_lda.json");
}
