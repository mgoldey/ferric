//! End-to-end geometry-optimization smoke tests across all four headline
//! KS-DFT functionals. Each test runs optimize_geometry starting from r_HH
//! = 1.40 Bohr and checks that:
//!   * the optimizer converges within the step budget
//!   * the final bond length is within a literature-reasonable range

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::operator::Operator;
use ferric_scf::optimize::{optimize_geometry, OptimizeConfig};
use ferric_scf::rhf::RhfConfig;

fn h2_mol() -> Molecule {
    Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap()
}

fn opt_cfg() -> OptimizeConfig {
    OptimizeConfig {
        max_steps: 25,
        g_max_thresh: 1.0e-4,
        g_rms_thresh: 5.0e-5,
        e_conv: 1e-7,
        trust_radius: 0.2,
    }
}

fn rhf_cfg_for(xc: &str, hybrid: bool) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: if hybrid { Some("def2-universal-jkfit".into()) } else { None },
        energy_conv: 1e-9,
        density_conv: 1e-7,
        ..Default::default()
    }
}

fn run_h2_opt(xc: &str, hybrid: bool, r_lo: f64, r_hi: f64) {
    let mol = h2_mol();
    let op = Operator::coulomb();
    let cfg = rhf_cfg_for(xc, hybrid);
    let opt = opt_cfg();
    let ctx = ParallelContext::default();
    let result = optimize_geometry(&ctx, &mol, "sto-3g", op, &cfg, &opt).unwrap();
    let r_hh = (result.mol.atoms[1].zpos - result.mol.atoms[0].zpos).abs();
    eprintln!(
        "[{xc}] converged={} steps={} E={:.8} r_HH={:.4} Bohr",
        result.converged, result.steps, result.energy, r_hh,
    );
    assert!(result.converged, "[{xc}] H2 opt did not converge");
    assert!(r_lo <= r_hh && r_hh <= r_hi,
            "[{xc}] H2 r_HH = {r_hh:.4} Bohr outside expected [{r_lo}, {r_hi}]");
}

#[test]
fn h2_lda_opt() { run_h2_opt("LDA",   false, 1.30, 1.50); }

#[test]
fn h2_pbe_opt() { run_h2_opt("PBE",   false, 1.30, 1.50); }

#[test]
fn h2_b3lyp_opt() { run_h2_opt("B3LYP", true, 1.30, 1.50); }

#[test]
fn h2_wb97xv_opt() { run_h2_opt("wB97X-V", true, 1.25, 1.50); }
