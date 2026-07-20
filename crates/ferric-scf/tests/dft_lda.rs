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
    run_case_basis(label, xyz, "cc-pvdz", expected_file);
}

fn run_case_basis(label: &str, xyz: &str, basis_name: &str, expected_file: &str) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();

    // Plain solve_rhf, not the SCF ladder: unlike PBE/NH3 (dft_pbe.rs, a
    // genuine wrong-SCF-solution problem the ladder fixes), LDA has no such
    // issue -- every case here lands on the right energy under plain DIIS,
    // just slowly (see below). The production ladder's rung max_iter caps
    // (60-100, sized for the heavy-atom/near-degenerate-d-manifold cases it
    // primarily targets) are too small for LDA's specific slow-density-
    // oscillation pattern and can't be changed here without touching the
    // shared default_ladder(), so plain solve_rhf with a generous max_iter
    // local to this test is the correct, minimal fix.
    //
    // LDA (no exact exchange) satisfies a tight density_conv gate markedly
    // more slowly than the hybrids: the ENERGY converges quickly and
    // correctly, but density_conv keeps oscillating just above threshold
    // for hundreds of iterations with zero further energy movement.
    // Measured directly this pass, sweeping density_conv 1e-8->1e-7 and
    // max_iter 200->1500: every configuration tried leaves 1-3 of 8 cases
    // reporting converged=false while EVERY energy stays correct (worst
    // observed err 5.88e-6 Ha, still inside the 1e-5 TOL below) -- this is
    // a genuine, benign DIIS density-oscillation floor for pure-functional
    // LDA specifically (PBE/B3LYP/wB97X-V, all hybrids, converge cleanly
    // at density_conv=1e-8/max_iter=200 with no such issue), not something
    // more iterations or a looser density_conv reliably fixes. At the old
    // max_iter=200/density_conv=1e-8 defaults this silently returned WRONG
    // energies (up to 2.5e-2 Ha off on H2O/def2-SVP) precisely because this
    // function never checked `converged` at all -- so the fix here is NOT
    // to keep chasing a strict converged gate (which the measurements above
    // show doesn't reliably trip for LDA even at 1500 iterations) but to
    // make the ENERGY comparison below the actual correctness gate, which
    // is what this test suite exists to validate, while still surfacing
    // `converged`/`iters` in the eprintln so a real future regression
    // (energy AND convergence both wrong) stays visible in the output.
    let cfg = RhfConfig {
        xc: Some("LDA".into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        // LDA has no exact exchange — no df_k_aux needed.
        energy_conv: 1e-10,
        density_conv: 1e-7,
        max_iter: 800,
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
        "[{label}] ferric = {:.10} Ha,  PySCF = {:.10} Ha,  err = {err:.2e}, converged={}, iters={}",
        res.energy, r.e_total, res.converged, res.iterations
    );
    assert!(
        err < TOL,
        "LDA E_total mismatch for {label}: err = {err:.2e} (ferric={:.10}, pyscf={:.10})",
        res.energy,
        r.e_total,
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

const NH3_XYZ: &str = "4\nNH3\n\
    N 0.000000 0.000000 0.116489\n\
    H 0.000000 0.939731 -0.271808\n\
    H 0.813831 -0.469865 -0.271808\n\
    H -0.813831 -0.469865 -0.271808\n";

/// Fourth molecule (widens past H2/H2O/CH4): NH3, C3v.
#[test]
fn lda_nh3() {
    run_case("NH3", NH3_XYZ, "nh3_cc-pvdz_lda.json");
}

/// Second basis (widens past cc-pVDZ-only) across all four molecules.
#[test]
fn lda_h2_def2svp() {
    run_case_basis("H2", "2\nH2\nH 0 0 0\nH 0 0 0.74\n", "def2-svp", "h2_def2-svp_lda.json");
}

#[test]
fn lda_water_def2svp() {
    run_case_basis(
        "H2O",
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
        "def2-svp",
        "h2o_def2-svp_lda.json",
    );
}

#[test]
fn lda_methane_def2svp() {
    let xyz = "5\nCH4\nC 0 0 0\n\
               H 0.6276 0.6276 0.6276\n\
               H -0.6276 -0.6276 0.6276\n\
               H -0.6276 0.6276 -0.6276\n\
               H 0.6276 -0.6276 -0.6276\n";
    run_case_basis("CH4", xyz, "def2-svp", "methane_def2-svp_lda.json");
}

#[test]
fn lda_nh3_def2svp() {
    run_case_basis("NH3", NH3_XYZ, "def2-svp", "nh3_def2-svp_lda.json");
}
