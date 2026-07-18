//! evGW0@PBE / evGW@PBE for H2O/cc-pVDZ — smoke tests for the KS-reference
//! plumbing added to `sigma::run_evgw0`/`sigma::run_evgw` (the evGW@KS gap
//! documented in docs/open-work-triage item #37).
//!
//! No PySCF/MOLGW numerical reference exists yet for evGW0/evGW@KS
//! specifically (only G0W0@PBE has one, in `g0w0_pbe_h2o.rs`), so these are
//! graded "smoke" per docs/VALIDATION.md's convention: they assert the run
//! converges to a physically sane quasiparticle energy and that the KS
//! static shift (Σx − vxc) is actually being applied (result differs from
//! the HF-reference run), rather than inventing a fabricated reference
//! number.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::vxc_mo::vxc_diagonal_mo;
use ferric_gw::{run_gw, GwConfig, GwMethod};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const HA: f64 = 27.211_386_245_988;

fn h2o_setup() -> (Molecule, PreparedBasis, PreparedBasis, Operator) {
    let xyz = "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    (mol, obs, dfbs, Operator::coulomb())
}

/// evGW0@PBE must converge to a finite, physically sane HOMO IP (positive,
/// within a broad chemical window) and must differ measurably from the
/// mean-field (Koopmans) IP — proving Σc AND the Σx−vxc static shift are
/// both actually applied, not silently dropped back to 0.0.
#[test]
fn evgw0_pbe_h2o_homo_ip_is_sane_and_ks_shifted() {
    let ctx = ParallelContext::default();
    let (mol, obs, dfbs, op) = h2o_setup();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();

    let cfg = RhfConfig { xc: Some("pbe".into()), ..Default::default() };
    let scf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).unwrap();
    let nocc = (mol.nelec() as usize) / 2;
    let homo_abs = nocc - 1;

    let pdep_cfg = PdepRpaConfig {
        quadrature: QuadratureConfig { scheme: QuadratureScheme::GaussLegendre, n_points: 16, u0: 0.5 },
        eigensolver_conv_thresh: 1e-7,
        trunc_thresh: 0.0,
        ..Default::default()
    };
    let gcfg = GwConfig {
        method: GwMethod::EvGw0,
        qp_mos: Some(homo_abs..homo_abs + 1),
        max_ev_iter: 20,
        ev_conv_thresh: 1e-4,
        ..Default::default()
    };

    let (vxc_diag, _) = vxc_diagonal_mo(&mol, &basis::bundled("cc-pvdz").unwrap(), "pbe", &scf).unwrap();

    // KS-referenced evGW0.
    let res_ks = run_gw(&mol, &obs, &dfbs, op, &scf, &pdep_cfg, &gcfg, Some(&vxc_diag)).unwrap();
    let loc = res_ks.mo_indices.iter().position(|&i| i == homo_abs).unwrap();
    let ip_ks = -res_ks.eps_qp[loc] * HA;

    assert!(res_ks.outer_converged, "evGW0@PBE outer loop must converge within max_ev_iter");
    assert!(res_ks.qp_converged[loc], "evGW0@PBE HOMO Newton QP solve must converge");
    assert!(
        ip_ks > 3.0 && ip_ks < 25.0,
        "evGW0@PBE HOMO IP {ip_ks:.3} eV is outside a physically sane window for H2O"
    );

    // Unshifted (HF-labeled-reference, vxc_diag=None) run for comparison —
    // proves the static shift actually changes the result, i.e. the plumbing
    // is wired through and not silently ignored.
    let res_unshifted = run_gw(&mol, &obs, &dfbs, op, &scf, &pdep_cfg, &gcfg, None).unwrap();
    let ip_unshifted = -res_unshifted.eps_qp[loc] * HA;
    assert!(
        (ip_ks - ip_unshifted).abs() > 0.5,
        "evGW0@PBE KS-shifted IP ({ip_ks:.3} eV) should differ substantially from the \
         unshifted run ({ip_unshifted:.3} eV) — Σx−vxc for a PBE reference is large \
         (~several eV); a near-zero difference would mean vxc_diag is not being applied"
    );

    // Sanity vs. the already-validated G0W0@PBE starting point (see
    // g0w0_pbe_h2o.rs, PySCF-anchored to <0.1 eV): evGW0 self-consistency
    // should move the HOMO IP by a modest amount, not diverge wildly.
    let gcfg_g0w0 = GwConfig { method: GwMethod::G0W0, qp_mos: Some(homo_abs..homo_abs + 1), ..Default::default() };
    let res_g0w0 = run_gw(&mol, &obs, &dfbs, op, &scf, &pdep_cfg, &gcfg_g0w0, Some(&vxc_diag)).unwrap();
    let ip_g0w0 = -res_g0w0.eps_qp[loc] * HA;
    assert!(
        (ip_ks - ip_g0w0).abs() < 2.0,
        "evGW0@PBE HOMO IP ({ip_ks:.3} eV) should stay within ~2 eV of its G0W0@PBE \
         starting point ({ip_g0w0:.3} eV), not diverge"
    );
}

/// evGW@PBE (self-consistent in both G and W) smoke test: same sanity checks
/// as evGW0@PBE above, on a reduced max_ev_iter budget (evGW rebuilds PDEP
/// every outer iteration, so it is far more expensive per step).
#[test]
fn evgw_pbe_h2o_homo_ip_is_sane_and_ks_shifted() {
    let ctx = ParallelContext::default();
    let (mol, obs, dfbs, op) = h2o_setup();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();

    let cfg = RhfConfig { xc: Some("pbe".into()), ..Default::default() };
    let scf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).unwrap();
    let nocc = (mol.nelec() as usize) / 2;
    let homo_abs = nocc - 1;

    let pdep_cfg = PdepRpaConfig {
        quadrature: QuadratureConfig { scheme: QuadratureScheme::GaussLegendre, n_points: 16, u0: 0.5 },
        eigensolver_conv_thresh: 1e-7,
        trunc_thresh: 0.0,
        ..Default::default()
    };
    let gcfg = GwConfig {
        method: GwMethod::EvGw,
        qp_mos: Some(homo_abs..homo_abs + 1),
        max_ev_iter: 4,
        ev_conv_thresh: 1e-3,
        ..Default::default()
    };

    let (vxc_diag, _) = vxc_diagonal_mo(&mol, &basis::bundled("cc-pvdz").unwrap(), "pbe", &scf).unwrap();

    let res_ks = run_gw(&mol, &obs, &dfbs, op, &scf, &pdep_cfg, &gcfg, Some(&vxc_diag)).unwrap();
    let loc = res_ks.mo_indices.iter().position(|&i| i == homo_abs).unwrap();
    let ip_ks = -res_ks.eps_qp[loc] * HA;

    assert!(res_ks.qp_converged[loc], "evGW@PBE HOMO Newton QP solve must converge");
    assert!(
        ip_ks > 3.0 && ip_ks < 25.0,
        "evGW@PBE HOMO IP {ip_ks:.3} eV is outside a physically sane window for H2O"
    );

    let res_unshifted = run_gw(&mol, &obs, &dfbs, op, &scf, &pdep_cfg, &gcfg, None).unwrap();
    let ip_unshifted = -res_unshifted.eps_qp[loc] * HA;
    assert!(
        (ip_ks - ip_unshifted).abs() > 0.5,
        "evGW@PBE KS-shifted IP ({ip_ks:.3} eV) should differ substantially from the \
         unshifted run ({ip_unshifted:.3} eV) — Σx−vxc for a PBE reference is large \
         (~several eV); a near-zero difference would mean vxc_diag is not being applied"
    );

    let gcfg_g0w0 = GwConfig { method: GwMethod::G0W0, qp_mos: Some(homo_abs..homo_abs + 1), ..Default::default() };
    let res_g0w0 = run_gw(&mol, &obs, &dfbs, op, &scf, &pdep_cfg, &gcfg_g0w0, Some(&vxc_diag)).unwrap();
    let ip_g0w0 = -res_g0w0.eps_qp[loc] * HA;
    assert!(
        (ip_ks - ip_g0w0).abs() < 2.0,
        "evGW@PBE HOMO IP ({ip_ks:.3} eV) should stay within ~2 eV of its G0W0@PBE \
         starting point ({ip_g0w0:.3} eV), not diverge"
    );
}
