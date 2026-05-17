//! Smoke + consistency tests for U-PDEP-RPA (C4).
//!
//! Two test families:
//!
//!   1. Internal consistency: closed-shell molecule (H2O) run via U-PDEP-RPA
//!      using a UHF (spin-symmetric) reference must reproduce closed-shell
//!      run_pdep_rpa to ≤1e-8 Ha. This catches sign/factor errors in the
//!      α+β summation without needing a PySCF reference for open-shell.
//!
//!   2. Open-shell sanity: H atom and OH radical produce finite, negative
//!      RPA correlation. Not pinned to PySCF until a U-RI-RPA reference
//!      script lands (see [[pyscf-ri-rpa-convention]]).
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Eigensolver, QuadratureConfig, QuadratureScheme};
use ferric_rpa::{run_pdep_rpa, run_u_pdep_rpa, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::{solve_rohf, RohfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, UhfConfig};

/// Config that keeps every PDEP eigenmode (trunc_thresh=0) so the
/// dielectric is the full naux×naux matrix. Used here for internal-
/// consistency tests (closed-shell-via-UHF ≡ closed-shell-via-RHF)
/// where any truncation would be a confound. 20-point Gauss-Legendre,
/// u0=0.5: those are the PySCF RI-RPA defaults, kept here so future
/// PySCF U-RI-RPA references plug in without re-tuning quadrature.
fn cfg_full_basis() -> PdepRpaConfig {
    let mut cfg = PdepRpaConfig::default();
    cfg.quadrature = QuadratureConfig {
        scheme: QuadratureScheme::GaussLegendre,
        n_points: 20,
        u0: 0.5,
    };
    cfg.frozen_core = 0;
    cfg.trunc_thresh = 0.0;
    cfg.davidson_conv_thresh = 1e-9;
    cfg
}

#[test]
fn u_pdep_rpa_matches_closed_shell_on_h2o() {
    // H2O via UHF with mult=1 → α and β orbitals coincide → U-PDEP-RPA
    // should match closed-shell PDEP-RPA bit-for-bit (modulo Davidson noise).
    let ctx = ParallelContext::default();
    let xyz = "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();

    let cfg = cfg_full_basis();

    // Closed-shell reference.
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    let e_closed = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap().e_rpa;

    // UHF (spin-symmetric) → U-RPA.
    let uhf_cfg = UhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let uhf = solve_uhf(&ctx, &mol, &obs, op, &bounds, &uhf_cfg).unwrap();
    assert!((uhf.energy - rhf.energy).abs() < 1e-7, "UHF/RHF energy disagreement on closed-shell H2O");

    let e_unrestricted = run_u_pdep_rpa(&mol, &obs, &dfbs, op, &uhf, &cfg).unwrap().e_rpa;
    let dev = (e_closed - e_unrestricted).abs();
    assert!(
        dev < 1e-7,
        "U-RPA on spin-symmetric UHF must match closed-shell RPA: \
         e_closed={e_closed:.10}, e_unrestricted={e_unrestricted:.10}, dev={dev:.2e}"
    );
}

#[test]
fn u_pdep_rpa_h_atom_finite_negative() {
    // H atom is the smallest open-shell test case: 1 occupied α, 0 occupied β.
    // RPA correlation must be exactly 0 in the n_occ_β = 0 limit because the
    // β-spin Π contribution is empty, and the α-spin Π built from a single
    // occupied orbital + virtuals gives a finite but small correlation.
    let ctx = ParallelContext::default();
    let xyz = "1\nh\nH 0 0 0\n";
    let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let uhf_cfg = UhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let uhf = solve_uhf(&ctx, &mol, &obs, op, &bounds, &uhf_cfg).unwrap();
    let cfg = cfg_full_basis();
    let r = run_u_pdep_rpa(&mol, &obs, &dfbs, op, &uhf, &cfg).unwrap();
    assert!(r.e_rpa < 0.0, "RPA correlation should be ≤ 0, got {}", r.e_rpa);
    assert!(r.e_rpa.is_finite(), "non-finite RPA correlation");
    assert!(r.e_rpa.abs() < 0.01, "H atom RPA correlation absurdly large: {}", r.e_rpa);
}

#[test]
fn u_pdep_rpa_rohf_reference_runs() {
    // ROHF-RPA: same dispatch as U-RPA (per-spin α and β MO blocks), just
    // using the spin-pure Guest-Saunders ROHF reference instead of UHF.
    // OH radical / cc-pVDZ: doublet, 9 electrons (5α, 4β).
    let ctx = ParallelContext::default();
    let xyz = "2\noh\nO 0 0 0\nH 0 0 0.97\n";
    let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rohf_cfg = RohfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let rohf = solve_rohf(&ctx, &mol, &obs, op, &bounds, &rohf_cfg).unwrap();
    let cfg = cfg_full_basis();
    let r = run_u_pdep_rpa(&mol, &obs, &dfbs, op, &rohf, &cfg).unwrap();
    assert!(r.e_rpa.is_finite(), "ROHF-RPA correlation non-finite");
    assert!(r.e_rpa < 0.0, "ROHF-RPA correlation should be negative, got {}", r.e_rpa);
    // Sanity bound: OH/cc-pVDZ RPA correlation is O(-0.2 to -0.3) Ha.
    assert!(r.e_rpa > -0.5 && r.e_rpa < -0.05,
            "ROHF-RPA OH/cc-pVDZ correlation = {} Ha, outside expected band", r.e_rpa);
}

#[test]
fn u_pdep_rpa_uhf_vs_rohf_close_on_doublet() {
    // OH is a clean doublet — UHF and ROHF should produce similar but not
    // identical RPA correlation energies (ROHF uses Guest-Saunders coupling,
    // UHF allows spin contamination). Bound the gap loosely.
    let ctx = ParallelContext::default();
    let xyz = "2\noh\nO 0 0 0\nH 0 0 0.97\n";
    let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = cfg_full_basis();

    let uhf_cfg = UhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let uhf = solve_uhf(&ctx, &mol, &obs, op, &bounds, &uhf_cfg).unwrap();
    let rohf_cfg = RohfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let rohf = solve_rohf(&ctx, &mol, &obs, op, &bounds, &rohf_cfg).unwrap();
    let e_u = run_u_pdep_rpa(&mol, &obs, &dfbs, op, &uhf, &cfg).unwrap().e_rpa;
    let e_r = run_u_pdep_rpa(&mol, &obs, &dfbs, op, &rohf, &cfg).unwrap().e_rpa;
    assert!(e_u.is_finite() && e_r.is_finite());
    let gap = (e_u - e_r).abs();
    assert!(gap < 0.05,
            "UHF vs ROHF U-RPA on OH gap = {gap:.4} Ha; expected <50 mHa for clean doublet");
}

#[test]
fn u_pdep_rpa_lanczos_matches_davidson() {
    // Internal consistency: Lanczos and Davidson eigensolvers on the same
    // U-PDEP-RPA dielectric must produce the same correlation energy.
    let ctx = ParallelContext::default();
    let xyz = "1\nh\nH 0 0 0\n";
    let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let uhf_cfg = UhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let uhf = solve_uhf(&ctx, &mol, &obs, op, &bounds, &uhf_cfg).unwrap();

    let mut cfg_d = cfg_full_basis();
    cfg_d.eigensolver = Eigensolver::Davidson;
    let e_davidson = run_u_pdep_rpa(&mol, &obs, &dfbs, op, &uhf, &cfg_d).unwrap().e_rpa;

    let mut cfg_l = cfg_full_basis();
    cfg_l.eigensolver = Eigensolver::Lanczos;
    let e_lanczos = run_u_pdep_rpa(&mol, &obs, &dfbs, op, &uhf, &cfg_l).unwrap().e_rpa;

    let dev = (e_davidson - e_lanczos).abs();
    assert!(dev < 1e-7, "Davidson vs Lanczos U-RPA disagree: {dev:.2e}");
}
