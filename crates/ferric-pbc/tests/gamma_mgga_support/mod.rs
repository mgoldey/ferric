//! Shared fixtures of the Gamma-point meta-GGA derivative tests
//! (`pbc_gamma_mgga_derivs.rs` per commit, `validation_pbc_mgga_derivs.rs`
//! weekly): a converged Gamma RKS on the dense-AFT J, the routed force /
//! stress (`ferric_pbc::gamma_mgga`), and central FD of the Gamma energy.
#![allow(dead_code)]

use super::common::*;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::dft::{gamma_rks, GammaRksConfig, PeriodicGridConfig};
use ferric_pbc::gamma_mgga::{
    gamma_rks_mgga_gradient, gamma_rks_mgga_stress, GammaMggaDerivConfig,
};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::kdense_aft::KDenseAftConfig;
use ferric_pbc::kgrad::KGradConfig;
use ferric_pbc::kstress::KStressConfig;
use ferric_pbc::kxcgrad::{KRksGradient, KXcMutation};
use ferric_pbc::kxcstress::{KRksStress, KXcStressMutation};
use ferric_pbc::lattice::Cell;
use ferric_pbc::stress::Mat3;
use ferric_pbc::uhf::GammaUhfIntegrals;
use ferric_scf::result::ScfResult;
use ndarray::Array2;

pub const OMEGA: f64 = 0.8;
pub const AMPLE: usize = 1 << 31;
pub const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
/// Explicit neighbour cutoff (the default depends on the cell, which strain
/// changes) large enough that no image crosses it under the FD steps.
pub const CUTOFF: f64 = 14.0;
pub const MGGA: [&str; 3] = ["SCAN", "R2SCAN", "TPSS"];
/// Central-difference step (Bohr for the force, strain for the stress).
pub const H: f64 = 1e-4;
/// Force FD bar: the measured floor is the h^2 truncation of the
/// bond-direction third derivative, 2.3e-9 at h = 1e-4 for all three
/// meta-GGAs (9.2e-9 at h = 2e-4), so 1e-7 is 43x above it.
pub const FD_BAR: f64 = 1e-7;
/// Stress (dE/d eps) FD bar: measured floor 3.0e-8 at h = 1e-4 (SCAN, TPSS).
pub const STRESS_FD_BAR: f64 = 3e-7;
pub const MUTANT_BAR: f64 = 1e-5;
/// Routed vs native (PBE) and Gamma vs 1x1x1 (glue) bar.
pub const GLUE_BAR: f64 = 1e-9;

pub fn grid() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(CUTOFF);
    g
}

pub fn cell_of(pos: &[[f64; 3]]) -> Cell {
    Cell::new(hydrogens(pos), cubic(4.0)).expect("cell")
}

pub fn hcfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig::with_omega(OMEGA)
}

pub fn rks_cfg(functional: &str) -> GammaRksConfig {
    let mut c = GammaRksConfig::new(functional);
    c.grid = grid();
    c.exxdiv = ExxDiv::Ewald;
    c.scf.energy_conv = 1e-13;
    c.scf.density_conv = 1e-10;
    c.scf.max_iter = 300;
    c
}

/// A converged Gamma RKS on the dense-AFT J.
pub struct G {
    pub cell: Cell,
    pub prep: ferric_integrals::basis_bridge::PreparedBasis,
    pub hc: PeriodicHcore,
    pub eri: DenseAftEri,
    pub cfg: GammaRksConfig,
    pub scf: ScfResult,
}

pub fn gamma(cell: Cell, functional: &str) -> G {
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let hc = periodic_hcore(&cell, &prep, &hcfg()).unwrap();
    let cfg = rks_cfg(functional);
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        cfg.exxdiv,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let r = gamma_rks(&cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &cfg)
        .unwrap_or_else(|e| panic!("gamma_rks {functional}: {e}"));
    assert!(r.scf.converged, "{functional}");
    G {
        cell,
        prep,
        hc,
        eri,
        cfg,
        scf: r.scf,
    }
}

pub fn energy(cell: Cell, functional: &str) -> f64 {
    gamma(cell, functional).scf.energy
}

pub fn dcfg() -> GammaMggaDerivConfig {
    GammaMggaDerivConfig {
        kdense: KDenseAftConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        grad: KGradConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        stress: KStressConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        ..Default::default()
    }
}

pub fn try_force(
    g: &G,
    dft: &GammaRksConfig,
    scf: &ScfResult,
    c: &GammaMggaDerivConfig,
) -> Result<KRksGradient, ferric_core::FerricError> {
    gamma_rks_mgga_gradient(&g.cell, &g.prep, &hcfg(), scf, dft, c)
}

pub fn force(g: &G, m: Option<KXcMutation>) -> KRksGradient {
    let c = GammaMggaDerivConfig {
        xc_grad_mutation: m,
        ..dcfg()
    };
    try_force(g, &g.cfg, &g.scf, &c).expect("gamma_rks_mgga_gradient")
}

pub fn stress(g: &G, m: Option<KXcStressMutation>) -> KRksStress {
    let c = GammaMggaDerivConfig {
        xc_stress_mutation: m,
        ..dcfg()
    };
    gamma_rks_mgga_stress(&g.cell, &g.prep, &hcfg(), &g.scf, &g.cfg, &c)
        .expect("gamma_rks_mgga_stress")
}

pub fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

pub fn strained(cell: &Cell, i: usize, j: usize, h: f64) -> Cell {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    cell.strained(&e).expect("strained cell")
}

pub fn diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
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

pub const FORCE_COMPS: [(usize, usize); 3] = [(0, 0), (0, 2), (1, 1)];
pub const STRAIN_COMPS: [(usize, usize); 3] = [(0, 0), (2, 2), (0, 2)];

/// Central FD of the Gamma SCF energy along atom displacements `comps`.
pub fn fd_force(f: &str, h: f64, comps: &[(usize, usize)]) -> Vec<f64> {
    comps
        .iter()
        .map(|&(a, x)| {
            (energy(cell_of(&moved(&H2, a, x, h)), f) - energy(cell_of(&moved(&H2, a, x, -h)), f))
                / (2.0 * h)
        })
        .collect()
}

/// Central FD of the Gamma SCF energy under strain `comps` (atoms move with
/// the lattice).
pub fn fd_stress(f: &str, h: f64, comps: &[(usize, usize)]) -> Vec<f64> {
    let cell = cell_of(&H2);
    comps
        .iter()
        .map(|&(i, j)| {
            (energy(strained(&cell, i, j, h), f) - energy(strained(&cell, i, j, -h), f)) / (2.0 * h)
        })
        .collect()
}

pub fn worst_force(g: &Array2<f64>, fd: &[f64], comps: &[(usize, usize)]) -> f64 {
    comps
        .iter()
        .zip(fd)
        .fold(0.0_f64, |m, (&(a, x), v)| m.max((g[(a, x)] - v).abs()))
}

pub fn worst_stress(s: &Mat3, fd: &[f64], comps: &[(usize, usize)]) -> f64 {
    comps
        .iter()
        .zip(fd)
        .fold(0.0_f64, |m, (&(i, j), v)| m.max((s[i][j] - v).abs()))
}
