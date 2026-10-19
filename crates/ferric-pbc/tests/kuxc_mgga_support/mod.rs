//! Shared setup of `pbc_kuxc_mgga.rs` (always-on) and
//! `validation_pbc_kuxc_mgga.rs` (weekly): the k-point UKS meta-GGA (SCAN /
//! r2SCAN) force and stress fixtures, the fixed-density XC finite differences
//! and the SCF-energy finite differences.
#![allow(dead_code)]

use crate::common::*;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::{PeriodicGrid, PeriodicGridConfig, PeriodicXcConfig};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::KDenseAftEri;
use ferric_pbc::kdft::{solve_krks, solve_krks_on_grid, KRksConfig, KRksResult};
use ferric_pbc::kgrad::{KGradConfig, KGradJk};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kstress::{KStressConfig, KStressJk};
use ferric_pbc::kuks::{
    solve_kuks, solve_kuks_on_grid, KPeriodicXcPolarized, KUksConfig, KUksResult,
};
use ferric_pbc::kuscf::KPointXcPolarized;
use ferric_pbc::kuxcgrad::{kpoint_uks_gradient, KUSpinMutation, KUksGradConfig, KUksGradient};
use ferric_pbc::kuxcstress::{kpoint_uks_stress, KUStressMutation, KUksStress, KUksStressConfig};
use ferric_pbc::kxcgrad::{
    kpoint_rks_gradient, KRksGradConfig, KRksGradient, KXcGrid, KXcMutation,
};
use ferric_pbc::kxcstress::{kpoint_rks_stress, KRksStress, KRksStressConfig, KXcStressMutation};
use ferric_pbc::lattice::Cell;
use ferric_pbc::stress::Mat3;
use ndarray::Array2;
use num_complex::Complex64;

pub const OMEGA: f64 = 0.8;
pub const AMPLE: usize = 1 << 31;
/// Explicit neighbour cutoff shared by every energy, force and stress here
/// (the default depends on the cell and puts one-step discontinuities into the
/// energy that FD would see).
pub const CUTOFF: f64 = 14.0;
/// H2 singlet (the closed-shell limit).
pub const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
/// H3 doublet (N_alpha = 2, N_beta = 1) in the a = 4 cube: the polarized cell
/// of the k-UKS meta-GGA energy pins (#479).
pub const H3: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [0.3, 1.9, 0.4]];
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
pub const FUNCTIONALS: [&str; 2] = ["SCAN", "R2SCAN"];

pub fn atom_grid() -> KXcGrid {
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(CUTOFF);
    KXcGrid::Atom(g)
}

/// `pos` in the a = 4 cube with multiplicity `mult`.
pub fn cube_cell(pos: &[[f64; 3]], mult: usize) -> Cell {
    let mut mol = hydrogens(pos);
    mol.multiplicity = mult;
    Cell::new(mol, cubic(4.0)).expect("cell")
}

/// The four-H triclinic cell of `common` as a triplet (N_alpha = 3, N_beta =
/// 1): the second polarized cell of #479, non-orthogonal lattice.
pub fn tri_cell(pos: &[[f64; 3]]) -> Cell {
    let mut mol = hydrogens(pos);
    mol.multiplicity = 3;
    Cell::new(mol, TRI_A).expect("triclinic cell")
}

/// Multiplicity of an H_n cell: singlet for even n, doublet for odd n.
pub fn default_mult(n: usize) -> usize {
    if n % 2 == 1 {
        2
    } else {
        1
    }
}

pub fn kuks_cfg(cell: &Cell, f: &str, grid: &KXcGrid) -> KUksConfig {
    let mut cfg = KUksConfig::new(cell, ExxDiv::None, f);
    cfg.kuhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.kuhf.scf.energy_conv = 1e-13;
    cfg.kuhf.scf.grad_conv = 1e-10;
    cfg.kuhf.scf.max_iter = 300;
    if let KXcGrid::Atom(g) = grid {
        cfg.grid = g.clone();
    }
    cfg
}

pub fn krks_cfg(cell: &Cell, f: &str, grid: &KXcGrid) -> KRksConfig {
    let mut cfg = KRksConfig::new(cell, ExxDiv::None, f);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.krhf.scf.energy_conv = 1e-13;
    cfg.krhf.scf.grad_conv = 1e-10;
    cfg.krhf.scf.max_iter = 300;
    if let KXcGrid::Atom(g) = grid {
        cfg.grid = g.clone();
    }
    cfg
}

pub fn uks_scf(cell: &Cell, n: [usize; 3], f: &str, grid: &KXcGrid) -> KUksResult {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let cfg = kuks_cfg(cell, f, grid);
    let r = match grid {
        KXcGrid::Atom(_) => solve_kuks(cell, &prep, None, &mesh, &cfg),
        KXcGrid::Uniform(u) => {
            let g = PeriodicGrid::uniform(cell, *u).unwrap();
            solve_kuks_on_grid(cell, &prep, None, &mesh, &g, &cfg)
        }
    }
    .unwrap_or_else(|e| panic!("solve_kuks {f} {n:?}: {e}"));
    assert!(r.scf.converged);
    r
}

pub fn rks_scf(cell: &Cell, n: [usize; 3], f: &str, grid: &KXcGrid) -> KRksResult {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let cfg = krks_cfg(cell, f, grid);
    match grid {
        KXcGrid::Atom(_) => solve_krks(cell, &prep, None, &mesh, &cfg),
        KXcGrid::Uniform(u) => {
            let g = PeriodicGrid::uniform(cell, *u).unwrap();
            solve_krks_on_grid(cell, &prep, None, &mesh, &g, &cfg)
        }
    }
    .unwrap_or_else(|e| panic!("solve_krks {f} {n:?}: {e}"))
}

/// A converged k-UKS state with everything its force and stress need.
pub struct Fx {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub mesh: KPointMesh,
    pub n: [usize; 3],
    pub hk: PeriodicHcoreK,
    pub eri: KDenseAftEri,
    pub res: KUksResult,
    pub grid: KXcGrid,
    pub functional: String,
}

pub fn fixture(cell: Cell, n: [usize; 3], f: &str, grid: KXcGrid) -> Fx {
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
    let res = uks_scf(&cell, n, f, &grid);
    let hk =
        periodic_hcore_kpts(&cell, &prep, &mesh, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let eri = KDenseAftEri::build(
        &cell,
        &prep,
        &mesh,
        &hk.s,
        ExxDiv::None,
        &Default::default(),
    )
    .unwrap();
    Fx {
        cell,
        prep,
        mesh,
        n,
        hk,
        eri,
        res,
        grid,
        functional: f.to_string(),
    }
}

pub fn gcfg(grid: &KXcGrid, xm: Option<KXcMutation>, sm: Option<KUSpinMutation>) -> KUksGradConfig {
    KUksGradConfig {
        base: KRksGradConfig {
            grad: KGradConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
            grid: grid.clone(),
            mutation: xm,
            ..Default::default()
        },
        spin_mutation: sm,
    }
}

pub fn scfg(
    grid: &KXcGrid,
    xm: Option<KXcStressMutation>,
    sm: Option<KUStressMutation>,
) -> KUksStressConfig {
    KUksStressConfig {
        base: KRksStressConfig {
            stress: KStressConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
            grid: grid.clone(),
            mutation: xm,
            ..Default::default()
        },
        spin_mutation: sm,
    }
}

pub fn try_force_as(fx: &Fx, f: &str, cfg: &KUksGradConfig) -> Result<KUksGradient, FerricError> {
    kpoint_uks_gradient(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &fx.hk,
        KGradJk::Dense(&fx.eri),
        &fx.res.scf,
        f,
        cfg,
    )
}

pub fn force(fx: &Fx, xm: Option<KXcMutation>, sm: Option<KUSpinMutation>) -> KUksGradient {
    try_force_as(fx, &fx.functional, &gcfg(&fx.grid, xm, sm))
        .unwrap_or_else(|e| panic!("kpoint_uks_gradient {xm:?} {sm:?}: {e}"))
}

pub fn try_stress_as(fx: &Fx, f: &str, cfg: &KUksStressConfig) -> Result<KUksStress, FerricError> {
    kpoint_uks_stress(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &fx.hk,
        KStressJk::Dense(&fx.eri),
        &fx.res.scf,
        f,
        cfg,
    )
}

pub fn stress(fx: &Fx, xm: Option<KXcStressMutation>, sm: Option<KUStressMutation>) -> KUksStress {
    try_stress_as(fx, &fx.functional, &scfg(&fx.grid, xm, sm))
        .unwrap_or_else(|e| panic!("kpoint_uks_stress {xm:?} {sm:?}: {e}"))
}

/// The closed-shell twin of a singlet fixture: the k-RKS force and stress
/// from an independent closed-shell SCF on the same cell, mesh and grid.
pub fn rks_twin(fx: &Fx) -> (KRksGradient, KRksStress) {
    let r = rks_scf(&fx.cell, fx.n, &fx.functional, &fx.grid);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let g = kpoint_rks_gradient(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &hcfg,
        &fx.hk,
        KGradJk::Dense(&fx.eri),
        &r.scf,
        &fx.functional,
        &KRksGradConfig {
            grad: KGradConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
            grid: fx.grid.clone(),
            ..Default::default()
        },
    )
    .unwrap();
    let s = kpoint_rks_stress(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &hcfg,
        &fx.hk,
        KStressJk::Dense(&fx.eri),
        &r.scf,
        &fx.functional,
        &KRksStressConfig {
            stress: KStressConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
            grid: fx.grid.clone(),
            ..Default::default()
        },
    )
    .unwrap();
    (g, s)
}

// ------------------------------------------------------------ small helpers

pub fn diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

pub fn amax(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
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

pub fn madd(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = *a;
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] += b[i][j];
        }
    }
    o
}

/// XC part of a force: X1 + X3 + X4.
pub fn xc_force(g: &KUksGradient) -> Array2<f64> {
    &(&g.xc_ao + &g.xc_point) + &g.xc_weight
}

/// XC part of a stress (`dE_xc/dε`, Hartree): AO strain + weight response.
pub fn xc_stress(s: &KUksStress) -> Mat3 {
    madd(&s.xc_ao, &s.xc_weight)
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

/// `E_xc[D_α, D_β]` at FIXED AO density matrices on `cell`'s own grid of the
/// kind `grid` (rebuilt for this geometry, as the SCF would).
pub fn exc_fixed_d(
    cell: &Cell,
    n: [usize; 3],
    f: &str,
    grid: &KXcGrid,
    da: &[Array2<Complex64>],
    db: &[Array2<Complex64>],
) -> f64 {
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let g = match grid {
        KXcGrid::Atom(c) => PeriodicGrid::build(cell, c).unwrap(),
        KXcGrid::Uniform(u) => PeriodicGrid::uniform(cell, *u).unwrap(),
    };
    let mut xc = KPeriodicXcPolarized::new(
        cell,
        &pyscf_sto3g_h(),
        f,
        &g,
        &mesh,
        &PeriodicXcConfig::default(),
    )
    .unwrap();
    xc.build_polarized(da, db).unwrap().0
}

/// Central FD of the fixed-density `E_xc` under atom displacement for the
/// listed `(atom, axis)` components of `pos` (cell built by `mk`); the other
/// entries stay zero.
pub fn fd_xc_force(
    mk: &dyn Fn(&[[f64; 3]]) -> Cell,
    pos: &[[f64; 3]],
    fx: &Fx,
    h: f64,
    comps: &[(usize, usize)],
) -> Array2<f64> {
    let (da, db) = (&fx.res.scf.density_alpha, &fx.res.scf.density_beta);
    let mut out = Array2::<f64>::zeros((pos.len(), 3));
    for &(a, x) in comps {
        let ep = exc_fixed_d(
            &mk(&moved(pos, a, x, h)),
            fx.n,
            &fx.functional,
            &fx.grid,
            da,
            db,
        );
        let em = exc_fixed_d(
            &mk(&moved(pos, a, x, -h)),
            fx.n,
            &fx.functional,
            &fx.grid,
            da,
            db,
        );
        out[(a, x)] = (ep - em) / (2.0 * h);
    }
    out
}

/// Central FD of the fixed-density `E_xc` under cell strain for the listed
/// components (Hartree, the `dE/dε` of `xc_stress`); the others stay zero.
pub fn fd_xc_stress(fx: &Fx, h: f64, comps: &[(usize, usize)]) -> Mat3 {
    let (da, db) = (&fx.res.scf.density_alpha, &fx.res.scf.density_beta);
    let mut out = [[0.0; 3]; 3];
    for &(i, j) in comps {
        let ep = exc_fixed_d(
            &strained(&fx.cell, i, j, h),
            fx.n,
            &fx.functional,
            &fx.grid,
            da,
            db,
        );
        let em = exc_fixed_d(
            &strained(&fx.cell, i, j, -h),
            fx.n,
            &fx.functional,
            &fx.grid,
            da,
            db,
        );
        out[i][j] = (ep - em) / (2.0 * h);
    }
    out
}

/// Largest `|a - b|` over the listed force components.
pub fn diff_at(a: &Array2<f64>, b: &Array2<f64>, comps: &[(usize, usize)]) -> f64 {
    comps
        .iter()
        .fold(0.0_f64, |m, &(i, j)| m.max((a[(i, j)] - b[(i, j)]).abs()))
}

/// Largest `|a - b|` over the listed tensor components.
pub fn mdiff_at(a: &Mat3, b: &Mat3, comps: &[(usize, usize)]) -> f64 {
    comps
        .iter()
        .fold(0.0_f64, |m, &(i, j)| m.max((a[i][j] - b[i][j]).abs()))
}

/// Every (atom, axis) of an n-atom cell.
pub fn all_comps(n: usize) -> Vec<(usize, usize)> {
    (0..n).flat_map(|a| (0..3).map(move |x| (a, x))).collect()
}

/// Fixed-density XC force check of `fx` (cell `mk(pos)`) over `comps`:
/// `(|xc - FD|, largest |FD|, the FD)`.
pub fn fixed_d_force_check(
    mk: &dyn Fn(&[[f64; 3]]) -> Cell,
    pos: &[[f64; 3]],
    fx: &Fx,
    h: f64,
    comps: &[(usize, usize)],
) -> (f64, f64, Array2<f64>) {
    let fd = fd_xc_force(mk, pos, fx, h, comps);
    let ok = diff_at(&xc_force(&force(fx, None, None)), &fd, comps);
    (ok, amax(&fd), fd)
}

/// The XC-side force defects (spin and tau) and their misses against `fd`.
pub fn force_xc_mutant_misses(
    fx: &Fx,
    fd: &Array2<f64>,
    comps: &[(usize, usize)],
) -> Vec<(String, f64)> {
    use KUSpinMutation as U;
    let xm = [KXcMutation::TauNoAo, KXcMutation::TauDoubled];
    let sm = [
        U::TauSpinSwap,
        U::TauDropBeta,
        U::TauTotalBothChannels,
        U::XcSpinSwap,
    ];
    xm.iter()
        .map(|m| (format!("{m:?}"), Some(*m), None))
        .chain(sm.iter().map(|m| (format!("{m:?}"), None, Some(*m))))
        .map(|(n, x, s)| (n, diff_at(&xc_force(&force(fx, x, s)), fd, comps)))
        .collect()
}

/// The XC-side stress defects and their misses against `fd`.
pub fn stress_xc_mutant_misses(fx: &Fx, fd: &Mat3, comps: &[(usize, usize)]) -> Vec<(String, f64)> {
    use KUStressMutation as U;
    let xm = [
        KXcStressMutation::NoTauStrain,
        KXcStressMutation::TauDoubled,
    ];
    let sm = [
        U::TauSpinSwap,
        U::TauDropBeta,
        U::TauTotalBothChannels,
        U::XcSpinSwap,
    ];
    xm.iter()
        .map(|m| (format!("{m:?}"), Some(*m), None))
        .chain(sm.iter().map(|m| (format!("{m:?}"), None, Some(*m))))
        .map(|(n, x, s)| (n, mdiff_at(&xc_stress(&stress(fx, x, s)), fd, comps)))
        .collect()
}

/// Singlet k-UKS vs k-RKS per part: `(label, |UKS - RKS|)` rows.
pub fn singlet_rows(fx: &Fx) -> Vec<(&'static str, f64)> {
    let u = force(fx, None, None);
    let us = stress(fx, None, None);
    let (r, rs) = rks_twin(fx);
    assert!(amax(&u.xc_ao) > 1e-3 && mmax(&us.xc_ao) > 1e-3);
    vec![
        ("force xc_ao", diff(&u.xc_ao, &r.xc_ao)),
        ("force xc_point", diff(&u.xc_point, &r.xc_point)),
        ("force xc_weight", diff(&u.xc_weight, &r.xc_weight)),
        ("force total", diff(&u.grad, &r.grad)),
        ("stress xc_ao", mdiff(&us.xc_ao, &rs.xc_ao)),
        ("stress xc_weight", mdiff(&us.xc_weight, &rs.xc_weight)),
        ("stress de_deps", mdiff(&us.de_deps, &rs.de_deps)),
    ]
}

/// Central FD of the converged k-UKS energy under atom displacement for the
/// listed `(atom, axis)` components.
pub fn fd_scf_force(
    mk: &dyn Fn(&[[f64; 3]]) -> Cell,
    pos: &[[f64; 3]],
    n: [usize; 3],
    f: &str,
    grid: &KXcGrid,
    h: f64,
    comps: &[(usize, usize)],
) -> Vec<f64> {
    comps
        .iter()
        .map(|&(a, x)| {
            let ep = uks_scf(&mk(&moved(pos, a, x, h)), n, f, grid).scf.energy;
            let em = uks_scf(&mk(&moved(pos, a, x, -h)), n, f, grid).scf.energy;
            (ep - em) / (2.0 * h)
        })
        .collect()
}

/// Central FD of the converged k-UKS energy under cell strain (`dE/dε`).
pub fn fd_scf_stress(
    cell: &Cell,
    n: [usize; 3],
    f: &str,
    grid: &KXcGrid,
    h: f64,
    comps: &[(usize, usize)],
) -> Vec<f64> {
    comps
        .iter()
        .map(|&(i, j)| {
            let ep = uks_scf(&strained(cell, i, j, h), n, f, grid).scf.energy;
            let em = uks_scf(&strained(cell, i, j, -h), n, f, grid).scf.energy;
            (ep - em) / (2.0 * h)
        })
        .collect()
}

/// `Σ_k Σ |D_α − D_β|`: proof that a fixture is genuinely polarized.
pub fn spin_asymmetry(fx: &Fx) -> f64 {
    fx.res
        .scf
        .density_alpha
        .iter()
        .zip(&fx.res.scf.density_beta)
        .map(|(a, b)| (a - b).iter().map(|z| z.norm()).sum::<f64>())
        .sum()
}
