//! Shared fixtures of the range-separated k-point force and stress tests
//! (`pbc_krsh_force.rs` / `pbc_krsh_stress.rs` per commit,
//! `validation_krsh_force.rs` / `validation_krsh_stress.rs` weekly).
#![allow(dead_code)]

use super::common::*;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::PeriodicGridConfig;
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kdft::{solve_krks, KRksConfig, KRksResult};
use ferric_pbc::kgrad::{KGradConfig, KGradJk, KGradMutation};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{solve_krks_injected, KPointInjection, KPointXc, KScfResult};
use ferric_pbc::kxcgrad::{kpoint_rks_gradient, KRksGradConfig, KRksGradient, KXcGrid};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsh::{resolve_periodic_functional_rsh, RshParams};
use ndarray::Array2;
use num_complex::Complex64;

/// libxc name of CAM-B3LYP (the friendly spelling does not resolve).
pub const CAM: &str = "HYB_GGA_XC_CAM_B3LYP";

pub const OMEGA: f64 = 0.8;
pub const AMPLE: usize = 1 << 31;
pub const CUTOFF: f64 = 10.0;
pub const PRECISION: f64 = 1e-12;
/// FD bar: the same as the global-hybrid force (`pbc_kxcgrad_hybrid.rs`),
/// whose h^2 truncation at h = 1e-4 on this geometry is 2.3e-9; the RSH
/// floor is measured in `cam_b3lyp_fd_residual_follows_h_squared`.
pub const FD_BAR: f64 = 1e-7;
pub const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
pub const A: f64 = 4.0;

pub fn rsh_of(f: &str) -> RshParams {
    resolve_periodic_functional_rsh(f).unwrap().2.unwrap()
}

pub fn grid_cfg() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(CUTOFF);
    g
}

pub fn cell_of(pos: &[[f64; 3]]) -> Cell {
    Cell::new(hydrogens(pos), cubic(A)).unwrap()
}

pub fn krks_cfg(cell: &Cell, f: &str, exxdiv: ExxDiv) -> KRksConfig {
    let mut cfg = KRksConfig::new(cell, exxdiv, f);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.krhf.scf.energy_conv = 1e-13;
    cfg.krhf.scf.grad_conv = 1e-10;
    cfg.krhf.scf.max_iter = 200;
    cfg.krhf.dense.precision = PRECISION;
    cfg.krhf.dense.budget_bytes = Some(AMPLE);
    cfg.grid = grid_cfg();
    cfg
}

pub struct Fx {
    pub cell: Cell,
    pub prep: ferric_integrals::basis_bridge::PreparedBasis,
    pub mesh: KPointMesh,
    pub hk: PeriodicHcoreK,
    pub res: KRksResult,
    pub f: String,
    pub exxdiv: ExxDiv,
}

pub fn fixture_cell(cell: Cell, mesh_n: [usize; 3], f: &str, exxdiv: ExxDiv) -> Fx {
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(&cell, mesh_n).unwrap();
    let cfg = krks_cfg(&cell, f, exxdiv);
    let res = solve_krks(&cell, &prep, None, &mesh, &cfg).unwrap();
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &cfg.krhf.hcore).unwrap();
    Fx {
        cell,
        prep,
        mesh,
        hk,
        res,
        f: f.into(),
        exxdiv,
    }
}

pub fn fixture(pos: &[[f64; 3]], mesh_n: [usize; 3], f: &str, exxdiv: ExxDiv) -> Fx {
    fixture_cell(cell_of(pos), mesh_n, f, exxdiv)
}

pub fn gcfg(exxdiv: ExxDiv, m: Option<KGradMutation>) -> KRksGradConfig {
    KRksGradConfig {
        grad: KGradConfig {
            budget_bytes: Some(AMPLE),
            mutation: m,
            ..Default::default()
        },
        grid: KXcGrid::Atom(grid_cfg()),
        exxdiv: Some(exxdiv),
        ..Default::default()
    }
}

/// The dense kernels exactly as `solve_krks` built them (functional's rsh).
pub fn eri_of(fx: &Fx, rsh: Option<RshParams>) -> KDenseAftEri {
    let dc = KDenseAftConfig {
        precision: PRECISION,
        budget_bytes: Some(AMPLE),
        rsh,
        ..Default::default()
    };
    KDenseAftEri::build(&fx.cell, &fx.prep, &fx.mesh, &fx.hk.s, fx.exxdiv, &dc).unwrap()
}

pub fn force_m(
    fx: &Fx,
    m: Option<KGradMutation>,
) -> Result<KRksGradient, ferric_core::FerricError> {
    let eri = eri_of(fx, Some(rsh_of(&fx.f)));
    kpoint_rks_gradient(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &fx.hk,
        KGradJk::Dense(&eri),
        &fx.res.scf,
        &fx.f,
        &gcfg(fx.exxdiv, m),
    )
}

pub fn force(fx: &Fx) -> KRksGradient {
    force_m(fx, None).unwrap()
}

pub fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

pub fn energy(pos: &[[f64; 3]], mesh_n: [usize; 3], f: &str, exxdiv: ExxDiv) -> f64 {
    let cell = cell_of(pos);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(&cell, mesh_n).unwrap();
    solve_krks(&cell, &prep, None, &mesh, &krks_cfg(&cell, f, exxdiv))
        .unwrap()
        .scf
        .energy
}

pub fn fd(
    pos: &[[f64; 3]],
    a: usize,
    x: usize,
    h: f64,
    mesh_n: [usize; 3],
    f: &str,
    e: ExxDiv,
) -> f64 {
    (energy(&moved(pos, a, x, h), mesh_n, f, e) - energy(&moved(pos, a, x, -h), mesh_n, f, e))
        / (2.0 * h)
}

pub fn diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

pub fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// Largest `|c_lr - c_sr| e^{-K^2/4w^2}` over the exchange K set of a mesh
/// (`K = G + q`, `K != 0`, inside the dense K sphere): how much of the
/// attenuated (erf) piece the walks actually see.
pub fn max_erf_weight(cell: &Cell, mesh: &KPointMesh, r: RshParams) -> f64 {
    let mut w = 0.0_f64;
    for iq in 0..mesh.nk() {
        let q = mesh.q_class(iq).0;
        for g in cell.gvectors(3.0).unwrap() {
            let k = [g[0] + q[0], g[1] + q[1], g[2] + q[2]];
            let k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
            if k2 > 1e-20 {
                w = w.max((r.c_lr - r.c_sr).abs() * (-k2 / (4.0 * r.omega * r.omega)).exp());
            }
        }
    }
    w
}

/// A zero XC (`E_xc = 0`, `V_xc = 0`) for the equal-fraction anchors.
pub struct ZeroXc(pub usize);
impl KPointXc for ZeroXc {
    fn build(
        &mut self,
        dm: &[Array2<Complex64>],
    ) -> Result<(f64, Vec<Array2<Complex64>>), ferric_core::FerricError> {
        Ok((
            0.0,
            dm.iter().map(|_| Array2::zeros((self.0, self.0))).collect(),
        ))
    }
}

/// Zero-XC k SCF on dense kernels with range separation `rsh` and exchange
/// scale `a`; returns (fixture pieces, SCF, kernels).
pub fn zero_xc(exxdiv: ExxDiv, rsh: Option<RshParams>, a: f64) -> (Fx, KScfResult, KDenseAftEri) {
    let cell = cell_of(&H2);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    let hk =
        periodic_hcore_kpts(&cell, &prep, &mesh, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let dc = KDenseAftConfig {
        precision: PRECISION,
        budget_bytes: Some(AMPLE),
        rsh,
        ..Default::default()
    };
    let eri = KDenseAftEri::build(&cell, &prep, &mesh, &hk.s, exxdiv, &dc).unwrap();
    let mut sc = ferric_pbc::kscf::KScfConfig::default();
    sc.energy_conv = 1e-13;
    sc.grad_conv = 1e-10;
    sc.max_iter = 200;
    let inj = KPointInjection {
        s: hk.s.clone(),
        h: hk.h.clone(),
        vnn: hk.enn,
        jk: Box::new(eri.jk_builder()),
    };
    let scf = solve_krks_injected(&cell, &mesh, &sc, inj, &mut ZeroXc(prep.nbasis()), a).unwrap();
    assert!(scf.converged);
    let dummy = KRksResult {
        e_xc: 0.0,
        exact_exchange_fraction: a,
        n_grid_points: 0,
        electrons_on_grid: 0.0,
        scf: scf.clone(),
    };
    (
        Fx {
            cell,
            prep,
            mesh,
            hk,
            res: dummy,
            f: String::new(),
            exxdiv,
        },
        scf,
        eri,
    )
}

/// The explicit diag(n) supercell of `pos` in the `A` cube (copy-major,
/// copy 0 first).
pub fn supercell(pos: &[[f64; 3]], n: [usize; 3]) -> Cell {
    let mut atoms = Vec::new();
    for m0 in 0..n[0] {
        for m1 in 0..n[1] {
            for m2 in 0..n[2] {
                for p in pos {
                    atoms.push([
                        p[0] + m0 as f64 * A,
                        p[1] + m1 as f64 * A,
                        p[2] + m2 as f64 * A,
                    ]);
                }
            }
        }
    }
    let lat = [
        [A * n[0] as f64, 0.0, 0.0],
        [0.0, A * n[1] as f64, 0.0],
        [0.0, 0.0, A * n[2] as f64],
    ];
    Cell::new(hydrogens(&atoms), lat).unwrap()
}
