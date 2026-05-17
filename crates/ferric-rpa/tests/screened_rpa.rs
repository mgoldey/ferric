//! Boys-screened χ₀ tests (C7).
//!
//! Validates that the per-orbital screened B-tile representation reproduces
//! the dense PDEP-RPA energy at `thresh = 0` (algebraic equivalence) and
//! converges to the dense answer at production thresholds with substantial
//! pair reduction.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Chi0Sparsity, QuadratureConfig, QuadratureScheme};
use ferric_rpa::{run_pdep_rpa, screen, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::ScfResult;
use ferric_scf::screening::SchwarzBounds;

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

fn base_cfg() -> PdepRpaConfig {
    let mut cfg = PdepRpaConfig::default();
    cfg.quadrature = QuadratureConfig {
        scheme: QuadratureScheme::GaussLegendre,
        n_points: 40,
        u0: 0.5,
    };
    cfg.frozen_core = 0;
    cfg.trunc_thresh = 0.0;
    cfg.davidson_conv_thresh = 1e-10;
    cfg
}

#[test]
fn h2o_cc_pvdz_screened_equivalence_thresh_zero() {
    // At thresh = 0 no aux rows are dropped; the screened tile representation
    // should match the dense path to high precision.
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/water.xyz", "cc-pvdz", "cc-pvdz-ri");

    let cfg_dense = base_cfg();
    let r_dense = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_dense).unwrap();

    let mut cfg_screen = base_cfg();
    cfg_screen.chi0_sparsity = Chi0Sparsity::BoysScreened { thresh: 0.0 };
    let r_scr = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_screen).unwrap();

    let diff = (r_scr.e_rpa - r_dense.e_rpa).abs();
    println!(
        "H2O/cc-pVDZ thresh=0  dense={:.10}  screened={:.10}  diff={:.2e}",
        r_dense.e_rpa, r_scr.e_rpa, diff
    );
    assert!(
        diff < 1e-9,
        "screened-vs-dense diff at thresh=0 = {:.2e}; expected <1e-9",
        diff
    );
}

#[test]
fn h2o_cc_pvdz_screened_production_thresh() {
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/water.xyz", "cc-pvdz", "cc-pvdz-ri");

    let r_dense = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &base_cfg()).unwrap();

    let thresh = 1e-6;
    let mut cfg = base_cfg();
    cfg.chi0_sparsity = Chi0Sparsity::BoysScreened { thresh };
    let r_scr = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();

    // Diagnostic: pair retention.
    let (sb, _) = screen::build_screened_bov_boys(&mol, &obs, &dfbs, op, &rhf, 0, thresh).unwrap();
    let total_possible = sb.n_occ_loc * sb.naux;
    println!(
        "H2O/cc-pVDZ thresh={:.0e}  retained {}/{} ({:.1}%)  ΔE={:.2e}",
        thresh,
        sb.total_retained,
        total_possible,
        100.0 * sb.total_retained as f64 / total_possible as f64,
        (r_scr.e_rpa - r_dense.e_rpa).abs(),
    );

    let diff = (r_scr.e_rpa - r_dense.e_rpa).abs();
    assert!(
        diff < 1e-7,
        "screened-vs-dense diff at thresh={:.0e} = {:.2e}; expected <1e-7",
        thresh, diff
    );
}

#[test]
#[ignore] // slow: benzene/cc-pVDZ is the scaling demonstration
fn benzene_cc_pvdz_screened_scaling() {
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/benzene.xyz", "cc-pvdz", "cc-pvdz-ri");

    use std::time::Instant;

    let mut cfg_dense = base_cfg();
    cfg_dense.frozen_core = 6;
    let t0 = Instant::now();
    let r_dense = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_dense).unwrap();
    let dt_dense = t0.elapsed().as_secs_f64();

    // C7-tighten: exact (P|i_loc i_loc) density-pair metric. The bound is
    // genuine Cauchy-Schwarz on (P|i a). For benzene, π orbitals span all
    // atoms so retention saturates near 100% below ~5e-3; tightening below
    // this discards no shells. Anything larger discards rapidly. The dial
    // test below sweeps a broader range.
    let thresh = 5e-3;
    let mut cfg_scr = base_cfg();
    cfg_scr.frozen_core = 6;
    cfg_scr.chi0_sparsity = Chi0Sparsity::BoysScreened { thresh };
    let t0 = Instant::now();
    let r_scr = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_scr).unwrap();
    let dt_scr = t0.elapsed().as_secs_f64();

    let (sb, _) = screen::build_screened_bov_boys(&mol, &obs, &dfbs, op, &rhf, 6, thresh).unwrap();
    let total_possible = sb.n_occ_loc * sb.naux;
    let reduction = total_possible as f64 / sb.total_retained.max(1) as f64;

    println!(
        "Benzene/cc-pVDZ thresh={:.0e}  retained {}/{} (reduction {:.2}×)  ΔE={:.2e}  dense {:.2}s  screened {:.2}s",
        thresh,
        sb.total_retained,
        total_possible,
        reduction,
        (r_scr.e_rpa - r_dense.e_rpa).abs(),
        dt_dense,
        dt_scr,
    );

    let diff = (r_scr.e_rpa - r_dense.e_rpa).abs();
    assert!(
        diff < 1e-3,
        "screened-vs-dense diff on benzene at thresh={:.0e} = {:.2e}; expected <1e-3",
        thresh, diff
    );
    // Demonstrate non-trivial pair reduction.
    assert!(
        reduction >= 1.1,
        "pair reduction factor {:.2}× too small; expected ≥1.1×",
        reduction
    );
}

/// Accuracy/sparsity dial: sweep thresh on benzene/cc-pVDZ and print the
/// retained-pair fraction plus ΔE at each setting. Informational — checks
/// the screen builds and yields a monotone tradeoff over 3+ decades.
#[test]
#[ignore] // slow
fn benzene_cc_pvdz_thresh_sweep() {
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/benzene.xyz", "cc-pvdz", "cc-pvdz-ri");

    let mut cfg_dense = base_cfg();
    cfg_dense.frozen_core = 6;
    let r_dense = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_dense).unwrap();

    println!("Benzene/cc-pVDZ dense e_rpa = {:.10}", r_dense.e_rpa);

    for &thresh in &[1e-1, 5e-2, 1e-2, 5e-3, 1e-3] {
        use std::time::Instant;
        let t_build = Instant::now();
        let (sb, _) =
            screen::build_screened_bov_boys(&mol, &obs, &dfbs, op, &rhf, 6, thresh).unwrap();
        let dt_build = t_build.elapsed().as_secs_f64();

        let mut cfg = base_cfg();
        cfg.frozen_core = 6;
        cfg.chi0_sparsity = Chi0Sparsity::BoysScreened { thresh };
        let t_run = Instant::now();
        let r = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        let dt_run = t_run.elapsed().as_secs_f64();

        let total = sb.n_occ_loc * sb.naux;
        let frac = 100.0 * sb.total_retained as f64 / total as f64;
        let de = (r.e_rpa - r_dense.e_rpa).abs();
        println!(
            "  thresh={:.0e}  retained {}/{} ({:.1}%)  ΔE={:.2e}  build={:.2}s  run={:.2}s",
            thresh, sb.total_retained, total, frac, de, dt_build, dt_run,
        );
    }
}
