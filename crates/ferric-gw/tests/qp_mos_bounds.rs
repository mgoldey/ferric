//! Regression test for the qp_mos out-of-bounds panic (found 2026-07-18):
//! `GwConfig.qp_mos`'s upper bound was never validated against the actual MO
//! count, so a caller-supplied range like `qp_mos = [0, 999]` on a small
//! molecule would panic deep inside a rayon closure in sigma.rs/cohsex.rs
//! (run_gw) or u_sigma.rs/u_cohsex.rs (run_u_gw) instead of surfacing a
//! clean error. Both entry points now fail fast with a `FerricError::General`
//! before any RPA work runs, so these tests are cheap (H2/STO-3G, no actual
//! GW solve reached).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::{run_gw, run_u_gw, GwConfig, GwMethod};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme, SternheimerConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, UhfConfig};

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        quadrature: QuadratureConfig { scheme: QuadratureScheme::GaussLegendre, n_points: 8, u0: 0.5 },
        eigensolver_conv_thresh: 1e-6,
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

#[test]
fn run_gw_rejects_out_of_range_qp_mos_instead_of_panicking() {
    // H2/STO-3G: 2 basis functions, so nmo = 2. qp_mos upper bound of 999 is
    // wildly out of range -- must Err, not panic, and must do so before any
    // real RPA work (this test would take much longer if it reached the
    // solve, since pdep_cfg's n_points=8 quadrature would actually run).
    let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    let gw_cfg = GwConfig {
        method: GwMethod::G0W0,
        qp_mos: Some(0..999),
        ..Default::default()
    };
    let err = run_gw(&mol, &obs, &dfbs, op, &rhf, &pdep_cfg(), &gw_cfg, None).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("qp_mos") && msg.contains("exceeds"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn run_u_gw_rejects_out_of_range_qp_mos_instead_of_panicking() {
    // OH radical/STO-3G (doublet, open-shell) exercises run_u_gw's own
    // separate check (u_sigma.rs/u_cohsex.rs share the same class of bug
    // as sigma.rs/cohsex.rs, fixed identically).
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let uhf = solve_uhf(&ctx, &mol, &obs, &bounds, &UhfConfig::default()).unwrap();

    let gw_cfg = GwConfig {
        method: GwMethod::G0W0,
        qp_mos: Some(0..999),
        ..Default::default()
    };
    let err = run_u_gw(&mol, &obs, &dfbs, op, &uhf, &pdep_cfg(), &gw_cfg).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("qp_mos") && msg.contains("exceeds"),
        "unexpected error message: {msg}"
    );
}
