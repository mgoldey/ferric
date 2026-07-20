//! B3LYP validation against PySCF reference energies.
//!
//! Exercises the plain-hybrid path through `KsXc` + RhfConfig: α·K (ω=0) with
//! a single libxc handle (HYB_GGA_XC_B3LYP). RI-K via `df_k_aux` is required
//! to match PySCF's `density_fit(auxbasis="def2-universal-jkfit")` reference.
//! Generated with `scripts/gen_pyscf_dft_refs.py b3lyp`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ladder::{default_ladder_from, solve_rhf_ladder};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Ref {
    e_total: f64,
    #[serde(default)]
    converged: bool,
}

/// B3LYP adds α·K (α=0.20) on top of the hybrid GGA semilocal piece. The
/// extra exchange integration carries the same grid noise as LDA/PBE; the
/// RI-K aux contributes a small (~µHa) error that's well below this guard.
const TOL: f64 = 5e-5;

fn ref_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/reference")
        .join(name)
}

fn run_case(label: &str, xyz: &str, expected_file: &str) {
    run_case_basis(label, xyz, "cc-pvdz", expected_file);
}

fn run_case_basis(label: &str, xyz: &str, basis_name: &str, expected_file: &str) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();

    // SCF ladder (matches the CLI's actual ksdft path) -- see dft_lda.rs's
    // module doc; PBE/NH3 needed this for correctness (plain DIIS landed a
    // wrong SCF solution), so it's applied uniformly across all 4 XC test
    // files rather than only where a failure happened to be observed.
    let cfg = RhfConfig {
        xc: Some("B3LYP".into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    };
    let ladder = default_ladder_from(&cfg);
    let lr = solve_rhf_ladder(&ctx, &mol, &obs, op, &bounds, &ladder).unwrap();
    let res = lr.result;

    let r: Ref =
        serde_json::from_str(&fs::read_to_string(ref_path(expected_file)).unwrap()).unwrap();
    assert!(r.converged, "PySCF reference {} not converged", expected_file);

    let err = (res.energy - r.e_total).abs();
    eprintln!(
        "[{label}] ferric = {:.10} Ha,  PySCF = {:.10} Ha,  err = {err:.2e}, ladder_converged={}, rung={}",
        res.energy, r.e_total, lr.converged, lr.rung_reached
    );
    assert!(lr.converged, "{label}: SCF ladder did not converge on any rung");
    assert!(
        err < TOL,
        "B3LYP E_total mismatch for {label}: err = {err:.2e} (ferric={:.10}, pyscf={:.10})",
        res.energy, r.e_total
    );
}

#[test]
fn b3lyp_h2() {
    run_case("H2", "2\nH2\nH 0 0 0\nH 0 0 0.74\n", "h2_cc-pvdz_b3lyp.json");
}

#[test]
fn b3lyp_water() {
    run_case(
        "H2O",
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
        "h2o_cc-pvdz_b3lyp.json",
    );
}

#[test]
fn b3lyp_methane() {
    let xyz = "5\nCH4\nC 0 0 0\n\
               H 0.6276 0.6276 0.6276\n\
               H -0.6276 -0.6276 0.6276\n\
               H -0.6276 0.6276 -0.6276\n\
               H 0.6276 -0.6276 -0.6276\n";
    run_case("CH4", xyz, "methane_cc-pvdz_b3lyp.json");
}

const NH3_XYZ: &str = "4\nNH3\n\
    N 0.000000 0.000000 0.116489\n\
    H 0.000000 0.939731 -0.271808\n\
    H 0.813831 -0.469865 -0.271808\n\
    H -0.813831 -0.469865 -0.271808\n";

/// Fourth molecule (widens past H2/H2O/CH4): NH3, C3v.
#[test]
fn b3lyp_nh3() {
    run_case("NH3", NH3_XYZ, "nh3_cc-pvdz_b3lyp.json");
}

/// Second basis (widens past cc-pVDZ-only) across all four molecules.
#[test]
fn b3lyp_h2_def2svp() {
    run_case_basis("H2", "2\nH2\nH 0 0 0\nH 0 0 0.74\n", "def2-svp", "h2_def2-svp_b3lyp.json");
}

#[test]
fn b3lyp_water_def2svp() {
    run_case_basis(
        "H2O",
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
        "def2-svp",
        "h2o_def2-svp_b3lyp.json",
    );
}

#[test]
fn b3lyp_methane_def2svp() {
    let xyz = "5\nCH4\nC 0 0 0\n\
               H 0.6276 0.6276 0.6276\n\
               H -0.6276 -0.6276 0.6276\n\
               H -0.6276 0.6276 -0.6276\n\
               H 0.6276 -0.6276 -0.6276\n";
    run_case_basis("CH4", xyz, "def2-svp", "methane_def2-svp_b3lyp.json");
}

#[test]
fn b3lyp_nh3_def2svp() {
    run_case_basis("NH3", NH3_XYZ, "def2-svp", "nh3_def2-svp_b3lyp.json");
}
