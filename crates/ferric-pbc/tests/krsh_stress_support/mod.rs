//! Stress helpers of the range-separated k-point stress tests
//! (`pbc_krsh_stress.rs` per commit, `validation_krsh_stress.rs` weekly), on
//! top of the force fixtures (`krsh_force_support`).
#![allow(dead_code)]

use super::common::*;
use super::support::*;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdft::solve_krks;
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::KScfResult;
use ferric_pbc::kstress::{KStressConfig, KStressJk, KStressMutation};
use ferric_pbc::kxcgrad::KXcGrid;
use ferric_pbc::kxcstress::{
    kpoint_null_xc_stress_rsh, kpoint_rks_stress, KRksStress, KRksStressConfig,
};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsh::RshParams;
use ferric_pbc::stress::Mat3;

pub const ALL9: [(usize, usize); 9] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
];

pub fn scfg(exxdiv: ExxDiv, m: Option<KStressMutation>) -> KRksStressConfig {
    KRksStressConfig {
        stress: KStressConfig {
            budget_bytes: Some(AMPLE),
            mutation: m,
            ..Default::default()
        },
        grid: KXcGrid::Atom(grid_cfg()),
        exxdiv: Some(exxdiv),
        ..Default::default()
    }
}

/// The production RSH stress of a fixture on its own functional's kernels.
pub fn stress_m(
    fx: &Fx,
    m: Option<KStressMutation>,
) -> Result<KRksStress, ferric_core::FerricError> {
    let eri = eri_of(fx, Some(rsh_of(&fx.f)));
    kpoint_rks_stress(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &fx.hk,
        KStressJk::Dense(&eri),
        &fx.res.scf,
        &fx.f,
        &scfg(fx.exxdiv, m),
    )
}

pub fn stress(fx: &Fx) -> KRksStress {
    stress_m(fx, None).unwrap()
}

/// Zero-XC stress on the anchor SCF of `zero_xc` (exchange scale `a`, range
/// separation `rsh` of the kernels).
pub fn null_stress(
    fx: &Fx,
    scf: &KScfResult,
    eri: &ferric_pbc::kdense_aft::KDenseAftEri,
    a: f64,
    rsh: Option<RshParams>,
    m: Option<KStressMutation>,
) -> KRksStress {
    kpoint_null_xc_stress_rsh(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &fx.hk,
        KStressJk::Dense(eri),
        scf,
        a,
        rsh,
        &scfg(fx.exxdiv, m),
    )
    .unwrap()
}

/// Fresh SCF energy (per cell) on `cell` (the mesh rebuilt on it with the
/// same integer numerators, the shared explicit grid cutoff).
pub fn energy_cell(cell: &Cell, mesh_n: [usize; 3], f: &str, exxdiv: ExxDiv) -> f64 {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(cell, mesh_n).unwrap();
    solve_krks(cell, &prep, None, &mesh, &krks_cfg(cell, f, exxdiv))
        .unwrap()
        .scf
        .energy
}

pub fn eps_at(i: usize, j: usize, h: f64) -> Mat3 {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    e
}

/// Central FD of the energy under the single strain component `(i, j)`.
pub fn fd_strain(
    cell: &Cell,
    (i, j): (usize, usize),
    h: f64,
    mesh_n: [usize; 3],
    f: &str,
    exxdiv: ExxDiv,
) -> f64 {
    let ep = energy_cell(&cell.strained(&eps_at(i, j, h)).unwrap(), mesh_n, f, exxdiv);
    let em = energy_cell(
        &cell.strained(&eps_at(i, j, -h)).unwrap(),
        mesh_n,
        f,
        exxdiv,
    );
    (ep - em) / (2.0 * h)
}

pub fn mdiff(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - b[i][j]).abs());
        }
    }
    m
}

pub fn mmax(a: &Mat3) -> f64 {
    a.iter().flatten().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// FD step and bar of the stress FD anchors. Measured for the RSH stress
/// (`validation_krsh_stress.rs`): worst all-nine |analytic - FD| 5.6e-9 at
/// h = 2.5e-5 (the (0, 0) component: FD-floor noise there; at larger h its
/// energy steps where the atom grid's hard neighbour_cutoff is crossed,
/// mgoldey/ferric#495, identically for the global hybrid PBE0); other
/// components <= 1.3e-9.
/// The bar is ~6x the worst miss and >= 1e5 below the smallest ledger mutant
/// (3.45e-3).
pub const FD_H: f64 = 2.5e-5;
pub const STRESS_FD_BAR: f64 = 3e-8;
