use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{QuadratureConfig, QuadratureScheme};
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig, RhfResult};
use ferric_scf::screening::SchwarzBounds;
use serde::Deserialize;

#[derive(Deserialize)]
struct RpaRef {
    e_corr: f64,
}

fn load_ref(path: &str) -> f64 {
    let s = std::fs::read_to_string(path).unwrap_or_else(|_| panic!("missing ref: {path}"));
    let r: RpaRef = serde_json::from_str(&s).unwrap();
    r.e_corr
}

fn setup(
    xyz: &str,
    obs_name: &str,
    dfbs_name: &str,
) -> (Molecule, PreparedBasis, PreparedBasis, Operator, RhfResult) {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz(xyz).unwrap();
    let obs_bs = basis::bundled(obs_name).unwrap();
    let dfbs_bs = basis::bundled(dfbs_name).unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    (mol, obs, dfbs, op, rhf)
}

fn pyscf_compat_config(n_quad: usize) -> PdepRpaConfig {
    let mut cfg = PdepRpaConfig::default();
    cfg.quadrature = QuadratureConfig {
        scheme: QuadratureScheme::GaussLegendre,
        n_points: n_quad,
        u0: 0.5,
    };
    cfg.frozen_core = 0;
    cfg
}

#[test]
fn h2_sto3g_rpa_energy_sign() {
    let (mol, obs, dfbs, op, rhf) = setup("../../testdata/molecules/h2.xyz", "sto-3g", "sto-3g");
    let result = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &PdepRpaConfig::default()).unwrap();
    assert!(result.e_rpa < 0.0, "E_c should be negative, got {}", result.e_rpa);
    assert!(result.n_eigenpotentials > 0);
}

#[test]
fn h2_sto3g_pdep_rpa_matches_pyscf() {
    // Reference uses STO-3G/STO-3G-RI to match ferric's RI basis exactly.
    let e_ref = load_ref("../../testdata/reference/h2_sto-3g_rpa.json");
    let (mol, obs, dfbs, op, rhf) = setup("../../testdata/molecules/h2.xyz", "sto-3g", "sto-3g");
    let cfg = pyscf_compat_config(40);
    let result = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let diff = (result.e_rpa - e_ref).abs();
    println!("H2/STO-3G  ferric={:.10} pyscf={:.10}  diff={:.2e}", result.e_rpa, e_ref, diff);
    assert!(diff < 1e-6, "H2/STO-3G PDEP-RPA differs by {:.2e}", diff);
}

#[test]
fn h2o_cc_pvdz_pdep_rpa_matches_pyscf() {
    let e_ref = load_ref("../../testdata/reference/h2o_cc-pvdz_rpa.json");
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/water.xyz", "cc-pvdz", "cc-pvdz-ri");
    let cfg = pyscf_compat_config(40);
    let result = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let diff = (result.e_rpa - e_ref).abs();
    println!("H2O/cc-pVDZ ferric={:.10} pyscf={:.10}  diff={:.2e}", result.e_rpa, e_ref, diff);
    assert!(diff < 1e-6, "H2O/cc-pVDZ PDEP-RPA differs by {:.2e}", diff);
}

#[test]
fn h2o_aug_cc_pvdz_pdep_rpa_matches_pyscf() {
    let e_ref = load_ref("../../testdata/reference/h2o_aug-cc-pvdz_rpa.json");
    let (mol, obs, dfbs, op, rhf) = setup(
        "../../testdata/molecules/water.xyz",
        "aug-cc-pvdz",
        "aug-cc-pvdz-rifit",
    );
    let cfg = pyscf_compat_config(40);
    let result = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let diff = (result.e_rpa - e_ref).abs();
    println!("H2O/aug-cc-pVDZ ferric={:.10} pyscf={:.10}  diff={:.2e}", result.e_rpa, e_ref, diff);
    assert!(diff < 1e-6, "H2O/aug-cc-pVDZ PDEP-RPA differs by {:.2e}", diff);
}

#[test]
fn h2o_aug_cc_pvtz_pdep_rpa_matches_pyscf() {
    let e_ref = load_ref("../../testdata/reference/h2o_aug-cc-pvtz_rpa.json");
    let (mol, obs, dfbs, op, rhf) = setup(
        "../../testdata/molecules/water.xyz",
        "aug-cc-pvtz",
        "aug-cc-pvtz-rifit",
    );
    let cfg = pyscf_compat_config(40);
    let result = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let diff = (result.e_rpa - e_ref).abs();
    println!("H2O/aug-cc-pVTZ ferric={:.10} pyscf={:.10}  diff={:.2e}", result.e_rpa, e_ref, diff);
    assert!(diff < 1e-6, "H2O/aug-cc-pVTZ PDEP-RPA differs by {:.2e}", diff);
}

#[test]
fn h2_sto3g_pdep_rpa_vs_ri_drpa() {
    // Sanity: PDEP-RPA == RI-dRPA when no truncation is applied (PDEP keeps all naux modes).
    let (mol, obs, dfbs, op, rhf) = setup("../../testdata/molecules/h2.xyz", "sto-3g", "sto-3g");
    let mut cfg = PdepRpaConfig::default();
    cfg.davidson_conv_thresh = 1e-10;
    let result = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let e_diag = result.e_rpa_dft_diag.expect("diagnostic should be present");
    let diff = (result.e_rpa - e_diag).abs();
    assert!(diff < 1e-8, "PDEP-RPA vs RI-dRPA differ by {:.2e}", diff);
}

#[test]
fn h2o_cc_pvdz_quadrature_convergence() {
    // 20 vs 40 GL points should both match PySCF to ≤1e-5.
    let e_ref = load_ref("../../testdata/reference/h2o_cc-pvdz_rpa.json");
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/water.xyz", "cc-pvdz", "cc-pvdz-ri");
    for &n in &[20usize, 40] {
        let cfg = pyscf_compat_config(n);
        let result = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        let diff = (result.e_rpa - e_ref).abs();
        println!("H2O/cc-pVDZ n_quad={} diff={:.2e}", n, diff);
        assert!(diff < 1e-5, "n_quad={}: |ΔE| = {:.2e} > 1e-5", n, diff);
    }
}
