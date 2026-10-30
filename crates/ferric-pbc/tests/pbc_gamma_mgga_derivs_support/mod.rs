//! Setup of `pbc_gamma_mgga_derivs.rs`: every non-test item of the test
//! file, made `pub`.
#![allow(dead_code, unused_imports)]

pub use crate::common::*;
pub use crate::support::*;
pub use ferric_pbc::dense_aft::ExxDiv;
pub use ferric_pbc::dft::{gamma_rks, PeriodicGridConfig};
pub use ferric_pbc::gamma_mgga::{gamma_rks_mgga_gradient, GammaMggaDerivConfig};
pub use ferric_pbc::grad::{gamma_rks_gradient_with, GammaGradConfig};
pub use ferric_pbc::hcore::periodic_hcore;
pub use ferric_pbc::kdense_aft::KDenseAftEri;
pub use ferric_pbc::kdft::{solve_krks, KRksConfig};
pub use ferric_pbc::kgrad::KGradJk;
pub use ferric_pbc::kpts::KPointMesh;
pub use ferric_pbc::kxcgrad::{kpoint_rks_gradient, KRksGradConfig, KXcGrid, KXcMutation};
pub use ferric_pbc::kxcstress::KXcStressMutation;
pub use ferric_pbc::lattice::Cell;
pub use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig};
pub use ferric_pbc::stress::{gamma_rks_stress, GammaStressConfig};
pub use ferric_pbc::uhf::GammaUhfIntegrals;

/// Per-commit FD rows: the two force components and two strain components
/// on which every ledger mutant misses (the full three-functional matrix and
/// the h sweeps are in `validation_pbc_mgga_derivs.rs`).
pub const FORCE_ROW: [(usize, usize); 2] = [(0, 2), (1, 1)];

pub const STRAIN_ROW: [(usize, usize); 2] = [(0, 0), (2, 2)];

/// Gamma vs 1x1x1 on the two-orbital cell (two separately converged SCFs):
/// measured 1.7e-12 (2026-10-22); the SCAN vs r2SCAN force gap there is 1.5e-3.
pub const H4_GLUE_BAR: f64 = 1e-10;

/// `solve_krks` + `kpoint_rks_gradient` at 1x1x1 on `g`'s cell, grid and
/// hcore: `(force, energy)`.
pub fn one_point_force(g: &G, f: &str) -> (ferric_pbc::kxcgrad::KRksGradient, f64) {
    let mesh = KPointMesh::gamma_centred(&g.cell, [1, 1, 1]).unwrap();
    let mut kc = KRksConfig::new(&g.cell, ExxDiv::Ewald, f);
    kc.krhf.hcore = hcfg();
    kc.krhf.scf.energy_conv = 1e-13;
    kc.krhf.scf.grad_conv = 1e-10;
    kc.krhf.scf.max_iter = 300;
    kc.grid = grid();
    let k = solve_krks(&g.cell, &g.prep, None, &mesh, &kc).unwrap();
    let hk =
        ferric_pbc::hcore::kpoint::periodic_hcore_kpts(&g.cell, &g.prep, &mesh, &hcfg()).unwrap();
    let eri = KDenseAftEri::build(
        &g.cell,
        &g.prep,
        &mesh,
        &hk.s,
        ExxDiv::Ewald,
        &Default::default(),
    )
    .unwrap();
    let kg = kpoint_rks_gradient(
        &g.cell,
        &g.prep,
        &mesh,
        &hcfg(),
        &hk,
        KGradJk::Dense(&eri),
        &k.scf,
        f,
        &KRksGradConfig {
            grad: dcfg().grad,
            grid: KXcGrid::Atom(grid()),
            exxdiv: Some(ExxDiv::Ewald),
            ..Default::default()
        },
    )
    .unwrap();
    (kg, k.scf.energy)
}
