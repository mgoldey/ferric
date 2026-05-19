//! LDA validation against PySCF reference energies.
//!
//! For each {molecule, cc-pVDZ, LDA(VWN)} tuple, runs ferric with
//! Becke-Lebedev (75, 110) grid and conv_tol=1e-10, then compares total
//! energy to the PySCF reference loaded from testdata/reference/.
//!
//! # Known deviation
//!
//! The reference JSON files were generated with PySCF's standard RKS
//! (exact-J Coulomb, Treutler-Ahlrichs radii-adjusted Becke partition).
//! Ferric uses RI-J (def2-universal-jkfit) and the original Becke (1988)
//! size correction. The combined effect of these two differences is:
//!
//! * H2    ≈ 3e-4 Ha
//! * H2O   ≈ 1.2e-3 Ha
//! * CH4   ≈ 5e-4 Ha
//!
//! All three are below the 2e-3 Ha test guard. Once ferric has a native
//! RI-J-aware reference generator, these can be tightened to 1e-6 Ha.

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
    #[serde(default)]
    xc: String,
    #[serde(default)]
    label: String,
    #[serde(default)]
    converged: bool,
}

/// Tolerance accounts for two sources of systematic deviation vs PySCF:
///   1. RI-J approximation (def2-universal-jkfit) vs PySCF's exact Coulomb: ~4e-5 Ha
///   2. Becke (1988) size correction vs PySCF's Treutler radii adjustment: ~1e-3 Ha
/// The combined worst-case is ~1.5e-3 Ha; 2e-3 gives a comfortable margin.
const TOL: f64 = 2e-3;

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
