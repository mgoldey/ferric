//! Scratch investigation (NOT a permanent gate): isolate ethylene's ~2x
//! oscillator-strength discrepancy (BSE-TDA[G0W0@HF]/cc-pVDZ f=0.635 vs
//! CC3/CBS literature f=0.338±0.005, Chrayteh/Blondel/Loos/Jacquemin JCTC
//! 2021, arXiv:2011.08509, Table 7) between: (a) bare CIS-TDA (no GW
//! screening) vs full BSE-TDA (G0W0-screened) on the SAME cc-pVDZ basis, to
//! see whether GW screening itself inflates f; (b) basis-set size (cc-pVDZ
//! vs aug-cc-pVDZ) at the CIS-TDA level, to see whether the effect is a
//! basis-incompleteness artifact like the literature's own TZVP->CBS trend
//! (Table 7 of the same paper: f drops 0.365->0.338, ~8%, NOT ~2x, when
//! going aug-cc-pVDZ->CBS).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme, SternheimerConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_gw::bse::{run_bse_tda, run_cis_tda};

const HA_TO_EV: f64 = 27.211386245988_f64;

const C2H4_XYZ: &str = "6\nethylene\nC 0.000000 0.000000 0.669500\nC 0.000000 0.000000 -0.669500\nH 0.000000 0.922832 1.237695\nH 0.000000 -0.922832 1.237695\nH 0.000000 0.922832 -1.237695\nH 0.000000 -0.922832 -1.237695\n";

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        quadrature: QuadratureConfig { scheme: QuadratureScheme::GaussLegendre, n_points: 16, u0: 0.5 },
        eigensolver_conv_thresh: 1e-7,
        eigensolver_max_vecs: 0,
        trunc_thresh: 0.0,
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
        memory_budget_bytes: None,
        need_inv_dielectric_freq: false,
    }
}

fn run_at_basis(obs_name: &str, dfbs_name: &str) {
    let mol = Molecule::parse_xyz(C2H4_XYZ, 0, 1).expect("parse C2H4");
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(dfbs_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    eprintln!("\n=== basis = {obs_name} (nbasis={}) ===", obs.nbasis());

    // Bare CIS-TDA (no GW, no screening) -- isolates the kernel/formula.
    let cis = run_cis_tda(&mol, &obs, &dfbs, op, &rhf, 0).unwrap();
    eprintln!("CIS-TDA (bare HF, no GW):");
    for (n, (&om, &f)) in cis.omega.iter().zip(cis.oscillator_strength.iter()).take(4).enumerate() {
        eprintln!("  n={} Omega={:.4} eV  f={:.5}", n + 1, om * HA_TO_EV, f);
    }

    // Full BSE-TDA (G0W0@HF screened) -- the pilot's actual path.
    let bse = run_bse_tda(&mol, &obs, &dfbs, op, &rhf, &pdep_cfg(), 0).unwrap();
    eprintln!("BSE-TDA (G0W0@HF screened):");
    for (n, (&om, &f)) in bse.omega.iter().zip(bse.oscillator_strength.iter()).take(4).enumerate() {
        eprintln!("  n={} Omega={:.4} eV  f={:.5}", n + 1, om * HA_TO_EV, f);
    }
}

#[test]
#[ignore = "scratch investigation, slow; run --release --ignored --nocapture"]
fn c2h4_osc_strength_gw_vs_bare_and_basis_scan() {
    run_at_basis("cc-pvdz", "cc-pvdz-ri");
    run_at_basis("def2-tzvp", "def2-tzvp-rifit");
    run_at_basis("aug-cc-pvdz", "aug-cc-pvdz-rifit");
    run_at_basis("aug-cc-pvtz", "aug-cc-pvtz-rifit");
}
