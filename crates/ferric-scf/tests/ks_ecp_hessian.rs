//! The analytic moving-grid KS Hessian with an effective core potential
//! (RKS and UKS): the ECP skeleton `Σ D ∂²V_ECP` and `∂V_ECP/∂x` in the
//! first-order Fock matrices, the same two terms the RHF/UHF ECP Hessian
//! adds (`src/hessian_ecp.rs`, `ecp_hessian.rs`).
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test ks_ecp_hessian
//! ```
//!
//! # Anchor
//!
//! The full Hessian against the Richardson central difference of the
//! production KS gradient (the same SCF at every displaced geometry, every
//! SCF at `fd_reference_config`). The gradient's ECP term is validated against
//! PySCF and ORCA (#206), so this tests the ASSEMBLY of the ECP terms into the
//! KS Hessian. The independent reference is PySCF's analytic RKS Hessian with
//! an ECP on a shared grid (`validation_ks_fixed_grid_hessian.rs`, HI case).
//!
//! # Artifact hypothesis, written before running
//!
//! * Both ECP terms present: the miss is the FD error of the all-electron
//!   cases (~1e-10) plus the ECP skeleton's stencil error (the RHF suite
//!   bounds `|S(h) − S(2h)|` by 1e-8).
//! * The skeleton or `∂V_ECP/∂x` missing: the miss is the size of that term
//!   (1e-2 to 1e-1 for iodine). Both negative controls are run here, so the
//!   bar is shown to separate them.
//!
//! # Mutations (each measured to fail a named test)
//!
//! * `extra_skeleton` dropped from `rks_fixed_grid_hessian_core`'s one-electron
//!   skeleton (`ks.rs`): `rks_ecp_hessian_matches_fd_of_the_production_gradient`;
//! * `add_extra_first_order` dropped from the same function: same test;
//! * the same two in `uks.rs`: `uks_ecp_hessian_matches_fd_of_the_production_gradient`.

mod ks_moving_grid_common;
mod uks_hessian_common;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::operator::Operator;
use ferric_scf::hessian::rij_support::with_ecp_first_order_dropped;
use ferric_scf::hessian::{rks_moving_grid_hessian, uks_moving_grid_hessian};
use ferric_scf::hessian_ecp::ecp_skeleton_hessian;
use ndarray::Array2;

/// HI tilted off the axes, so every Cartesian block is populated.
const HI: &str = "2\nHI tilted\nH 0.30 -0.40 0.20\nI 0.05 0.15 1.80\n";
/// H-I-H doublet (odd electron count), tilted (the RHF/UHF ECP suite's).
const HIH: &str = "3\nHIH tilted\nH 0.30 -0.40 0.20\nI 0.00 0.00 1.61\nH -1.10 0.70 2.90\n";
const BASIS: &str = "def2-svp";

/// Analytic vs FD of the production gradient, Ha/Bohr². Derivation: the
/// all-electron moving-grid bar is 2e-9 (`TOL_HESS_FD`, set by the FD
/// reference's SCF noise across software stacks); the ECP skeleton adds its
/// fourth-order stencil error, which the RHF ECP suite bounds by 1e-8
/// (`TOL_STEP`) and which measures far below that here. Measured 1.2e-10
/// (LDA), 8.6e-11 (PBE) for RKS HI and 5.8e-10 for UKS H-I-H; the bar is the
/// stencil bound, 17x the worst measured miss.
const TOL: f64 = 1e-8;
/// A negative control must miss by this many bars.
const MUST_MISS: f64 = 10.0;

fn ecp_mol(xyz: &str, mult: usize) -> Molecule {
    let mut mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    mol.apply_ecp(&basis::bundled(BASIS).unwrap());
    assert!(
        mol.atoms.iter().any(|a| a.n_core_ecp > 0),
        "iodine carries no ECP in {BASIS}"
    );
    mol
}

fn report(what: &str, miss: f64, no_skel: f64, no_first: f64, h: &Array2<f64>) {
    eprintln!(
        "{what}: analytic vs FD of the production gradient {miss:.3e} (max |H| {:.3e}); \
         without the ECP skeleton {no_skel:.3e}, without dV_ECP/dx in F^x {no_first:.3e}",
        ks_moving_grid_common::max_abs(h)
    );
    assert!(miss < TOL, "{what}: {miss:.3e} >= {TOL:e}");
    assert!(
        no_skel > MUST_MISS * TOL,
        "{what}: skeleton control {no_skel:.3e}"
    );
    assert!(
        no_first > MUST_MISS * TOL,
        "{what}: first-order control {no_first:.3e}"
    );
}

#[test]
fn rks_ecp_hessian_matches_fd_of_the_production_gradient() {
    use ks_moving_grid_common as c;
    if !c::has_deriv2() {
        return;
    }
    let (ctx, op) = (ParallelContext::default(), Operator::coulomb());
    for xc in ["LDA", "PBE"] {
        let cfg = c::fd_reference_config(&c::config(xc));
        let p = c::scf(&ecp_mol(HI, 1), BASIS, &cfg);
        let h =
            rks_moving_grid_hessian(&ctx, &p.mol, &p.prep, op, &p.bounds, &p.res, &cfg).unwrap();
        let fd = c::fd_of_production_gradient(&p, BASIS, &cfg, xc);
        let skel = ecp_skeleton_hessian(&p.mol, &p.bs, p.res.density_r()).unwrap();
        let no_first = with_ecp_first_order_dropped(|| {
            rks_moving_grid_hessian(&ctx, &p.mol, &p.prep, op, &p.bounds, &p.res, &cfg)
        })
        .unwrap();
        report(
            &format!("RKS {xc} HI/{BASIS}"),
            c::max_abs_diff(&h, &fd),
            c::max_abs_diff(&(&h - &skel), &fd),
            c::max_abs_diff(&no_first, &fd),
            &h,
        );
        assert!(c::sum_rule(&h) < 1e-6, "sum rule {:.2e}", c::sum_rule(&h));
    }
}

#[test]
fn uks_ecp_hessian_matches_fd_of_the_production_gradient() {
    use uks_hessian_common as u;
    if !u::has_deriv2() {
        return;
    }
    let (ctx, op) = (ParallelContext::default(), Operator::coulomb());
    let xc = "PBE";
    let cfg = u::fd_reference_config(&u::config(xc));
    let p = u::scf(&ecp_mol(HIH, 2), BASIS, &cfg);
    let h = uks_moving_grid_hessian(&ctx, &p.mol, &p.prep, op, &p.bounds, &p.res, &cfg).unwrap();
    let (fd, worst) = u::fd_of_production_gradient(&p, BASIS, &cfg, xc);
    let (da, db) = u::spin_densities(&p);
    let skel = ecp_skeleton_hessian(&p.mol, &p.bs, &(&da + &db)).unwrap();
    let no_first = with_ecp_first_order_dropped(|| {
        uks_moving_grid_hessian(&ctx, &p.mol, &p.prep, op, &p.bounds, &p.res, &cfg)
    })
    .unwrap();
    eprintln!("worst displaced-SCF err_max {worst:.2e}");
    report(
        &format!("UKS {xc} HIH/{BASIS}"),
        u::max_abs_diff(&h, &fd),
        u::max_abs_diff(&(&h - &skel), &fd),
        u::max_abs_diff(&no_first, &fd),
        &h,
    );
}
