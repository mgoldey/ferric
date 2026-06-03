use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{QuadratureConfig, QuadratureScheme};
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::ScfResult;
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
) -> (Molecule, PreparedBasis, PreparedBasis, Operator, ScfResult) {
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
    PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: n_quad,
            u0: 0.5,
        },
        frozen_core: 0,
        // Disable PDEP truncation: compare full-basis dielectric to PySCF's full RI-RPA.
        trunc_thresh: 0.0,
        davidson_conv_thresh: 1e-10,
        ..Default::default()
    }
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
    let cfg = PdepRpaConfig {
        trunc_thresh: 0.0,
        davidson_conv_thresh: 1e-10,
        run_diagnostics: true,
        ..Default::default()
    };
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

#[test]
fn h2o_cc_pvdz_pdep_truncation_convergence() {
    // PDEP test: can we recover full RPA energy with M ≈ 3 × N_atoms eigenpotentials?
    // H2O has 3 atoms → target M ≈ 9. Full naux=84.
    let e_ref = load_ref("../../testdata/reference/h2o_cc-pvdz_rpa.json");
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/water.xyz", "cc-pvdz", "cc-pvdz-ri");

    println!("\nPDEP truncation study (H2O/cc-pVDZ, naux=84):");
    println!("PySCF (full):      {:.10}", e_ref);

    for thresh in &[1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-10] {
        let mut cfg = pyscf_compat_config(40);
        cfg.trunc_thresh = *thresh;
        cfg.davidson_conv_thresh = 1e-10;
        let result = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        let diff = result.e_rpa - e_ref;
        println!(
            "  trunc={:.0e}: M={:3} E_c={:.10}  diff={:+.2e}  λ_max={:.3} λ_min={:.3e}",
            thresh, result.n_eigenpotentials, result.e_rpa, diff,
            result.eigenvalues_static.first().copied().unwrap_or(0.0),
            result.eigenvalues_static.last().copied().unwrap_or(0.0),
        );
    }
}

#[test]
fn h2o_aug_cc_pvtz_timing_comparison() {
    use std::time::Instant;
    let (mol, obs, dfbs, op, rhf) = setup(
        "../../testdata/molecules/water.xyz",
        "aug-cc-pvtz",
        "aug-cc-pvtz-rifit",
    );
    println!("\nH2O/aug-cc-pVTZ timing (naux=198, 40 GL points):");

    // Full RI-RPA (no truncation)
    let mut cfg_full = pyscf_compat_config(40);
    cfg_full.trunc_thresh = 0.0;
    let t0 = Instant::now();
    let r_full = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_full).unwrap();
    let dt_full = t0.elapsed();
    println!("  full (M=naux={}):  E_c={:.10}  t={:.2}s",
        r_full.n_eigenpotentials, r_full.e_rpa, dt_full.as_secs_f64());

    // PDEP truncated
    for &th in &[1e-1, 1e-2, 1e-3, 1e-4] {
        let mut cfg = pyscf_compat_config(40);
        cfg.trunc_thresh = th;
        let t0 = Instant::now();
        let r = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        let dt = t0.elapsed();
        let diff = (r.e_rpa - r_full.e_rpa).abs();
        println!("  trunc={:.0e}: M={:3} E_c={:.10}  diff_vs_full={:.2e}  t={:.2}s ({:.0}% of full)",
            th, r.n_eigenpotentials, r.e_rpa, diff,
            dt.as_secs_f64(), 100.0 * dt.as_secs_f64() / dt_full.as_secs_f64());
    }
}

#[test]
#[ignore]
fn benzene_cc_pvdz_timing() {
    use std::time::Instant;
    let (mol, obs, dfbs, op, rhf) = setup(
        "../../testdata/molecules/benzene.xyz",
        "cc-pvdz",
        "cc-pvdz-ri",
    );
    println!("\nBenzene/cc-pVDZ PDEP-RPA timing (40 GL points):");

    let mut cfg = pyscf_compat_config(40);
    cfg.trunc_thresh = 1e-4;
    cfg.frozen_core = 6;

    let t0 = Instant::now();
    let r = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let dt = t0.elapsed();
    println!("  trunc=1e-4 frozen_core=6: M={} E_c={:.10} t={:.2}s",
        r.n_eigenpotentials, r.e_rpa, dt.as_secs_f64());
}
