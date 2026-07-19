//! Property checks for the DFT grid + Vxc machinery (T15 from the original plan).
//!
//! Existing tests cover ∫ρ = N_e (density_grid.rs), ρ ≥ 0 (density_grid.rs),
//! σ = |∇ρ|² (density_grid.rs), and V_xc Hermiticity for LDA & PBE (vxc_lda.rs).
//! This file fills the remaining gaps:
//!
//! 1. V_xc Hermiticity for the hybrid (B3LYP) and range-separated hybrid
//!    (wB97X-V) families — the semilocal Vxc piece is identical structurally
//!    to plain GGA, but the test surface ought to include the families we
//!    actually validate against PySCF.
//! 2. FD-vs-analytic ∇χ at a fixed point — `eval_basis_and_grad_on_points`
//!    returns ∂χ/∂r evaluated at the *electron* position. Confirm against a
//!    central-difference of χ over the electron coordinate so the analytic
//!    derivative formula doesn't silently drift relative to the value-only path.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::ao_grid::{eval_basis_and_grad_on_points, eval_basis_on_points};
use ferric_dft::density_on_grid::eval_density_closed;
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_dft::libxc::xc_def_from_name;
use ferric_dft::vxc::semilocal_vxc_closed;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn build_h2o() -> Molecule {
    Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
        0,
        1,
    )
    .unwrap()
}

fn max_asym(vxc: &ndarray::Array2<f64>) -> f64 {
    let n = vxc.nrows();
    let mut asym = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let a = (vxc[(i, j)] - vxc[(j, i)]).abs();
            if a > asym {
                asym = a;
            }
        }
    }
    asym
}

fn vxc_at(xc_name: &str) -> f64 {
    let mol = build_h2o();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    let grid = build_atomic_grid(&mol, &AtomicGridConfig::default());
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi) = eval_basis_and_grad_on_points(&mol, &bs, &pts).unwrap();
    let dens = eval_density_closed(&rhf.density_total, &chi, &dchi);

    let xc = xc_def_from_name(xc_name).unwrap();
    let (e_xc, vxc) = semilocal_vxc_closed(&grid, &chi, &dchi, &dens, None, &xc);
    eprintln!("{xc_name} E_xc(H2O, cc-pVDZ) = {e_xc:.6} Ha (semilocal piece only)");
    max_asym(&vxc)
}

#[test]
fn vxc_b3lyp_h2o_is_hermitian() {
    let asym = vxc_at("B3LYP");
    assert!(asym < 1e-12, "V_xc(B3LYP) not Hermitian: max asym = {asym:.2e}");
}

#[test]
fn vxc_wb97xv_h2o_is_hermitian() {
    let asym = vxc_at("wB97X-V");
    assert!(asym < 1e-12, "V_xc(wB97X-V) not Hermitian: max asym = {asym:.2e}");
}

/// Central-difference ∂χ_μ/∂r at a fixed electron position vs the analytic
/// derivative returned by `eval_basis_and_grad_on_points`. This protects the
/// per-shell ∂/∂x_e formulas against silent drift relative to χ itself —
/// distinct from the cross-check between `eval_shell` and
/// `eval_shell_and_grad` (which verifies the value branch, not the gradient).
#[test]
fn analytic_dchi_dr_matches_central_difference_ccpvdz() {
    let mol = build_h2o();
    let bs = basis::bundled("cc-pvdz").unwrap();

    let r0 = [0.27_f64, 0.13, 0.41];
    let h = 1e-5_f64;

    let (_, dchi_ana) = eval_basis_and_grad_on_points(&mol, &bs, &[r0]).unwrap();
    let nbf = dchi_ana.shape()[1];

    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for axis in 0..3 {
        let mut r_plus = r0;
        r_plus[axis] += h;
        let mut r_minus = r0;
        r_minus[axis] -= h;

        let chi_plus = eval_basis_on_points(&mol, &bs, &[r_plus]).unwrap();
        let chi_minus = eval_basis_on_points(&mol, &bs, &[r_minus]).unwrap();

        for mu in 0..nbf {
            let fd = (chi_plus[(mu, 0)] - chi_minus[(mu, 0)]) / (2.0 * h);
            let ana = dchi_ana[(axis, mu, 0)];
            let diff = (fd - ana).abs();
            if diff > max_abs {
                max_abs = diff;
            }
            let denom = ana.abs().max(1e-10);
            let rel = diff / denom;
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }
    eprintln!("max |∂χ_ana − ∂χ_FD| = {max_abs:.3e}, max rel = {max_rel:.3e}");
    // Central difference at h=1e-5: leading error ~h² ≈ 1e-10, plus
    // double-precision round-off ~1e-11. 1e-8 is a comfortable bound.
    assert!(max_abs < 1e-8, "analytic ∇χ off from FD by {max_abs:.3e}");
}

/// Same FD-vs-analytic χ-gradient check on a basis that exercises f-shells,
/// since f-shell ∂/∂x formulas are the youngest piece of the AO code (P1.2).
#[test]
fn analytic_dchi_dr_matches_central_difference_def2_tzvp() {
    let mol = build_h2o();
    let bs = basis::bundled("def2-tzvp").unwrap();

    let r0 = [0.27_f64, 0.13, 0.41];
    let h = 1e-5_f64;

    let (_, dchi_ana) = eval_basis_and_grad_on_points(&mol, &bs, &[r0]).unwrap();
    let nbf = dchi_ana.shape()[1];

    let mut max_abs = 0.0_f64;
    for axis in 0..3 {
        let mut r_plus = r0;
        r_plus[axis] += h;
        let mut r_minus = r0;
        r_minus[axis] -= h;

        let chi_plus = eval_basis_on_points(&mol, &bs, &[r_plus]).unwrap();
        let chi_minus = eval_basis_on_points(&mol, &bs, &[r_minus]).unwrap();

        for mu in 0..nbf {
            let fd = (chi_plus[(mu, 0)] - chi_minus[(mu, 0)]) / (2.0 * h);
            let ana = dchi_ana[(axis, mu, 0)];
            let diff = (fd - ana).abs();
            if diff > max_abs {
                max_abs = diff;
            }
        }
    }
    eprintln!("def2-tzvp max |∂χ_ana − ∂χ_FD| = {max_abs:.3e}");
    assert!(max_abs < 1e-8, "def2-tzvp ∇χ FD mismatch = {max_abs:.3e}");
}
