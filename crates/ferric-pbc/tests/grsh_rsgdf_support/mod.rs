//! Shared fixtures of the Gamma-point range-separated hybrid derivatives on
//! RS-GDF J/K (`pbc_grsh_rsgdf_derivs.rs` per commit,
//! `validation_grsh_rsgdf_derivs.rs` weekly): a Gamma RKS / UKS / ROKS SCF on
//! the Gamma RS-GDF J (`RsGdf`) plus the `k = 0` attenuated exchange
//! (`GammaRshExchange::build_rsgdf`), its routed force and stress, and the
//! INDEPENDENT 1x1x1 k-point SCF + force + stress on `KRsGdf::build_rsh`.
//!
//! Cells: the closed-shell triclinic H2 (`rs::TRI2_ATOMS` in `TRI_A`) and the
//! open-shell H3 doublet (`support::H3D` in `TRI_A`), STO-3G with the
//! prototype's ET-sp aux, RS-GDF split ω₀ = 1, metric cut 1e-6 (the cut that
//! makes the ω = 0.33 attenuated fit FD-clean).
#![allow(dead_code)]

use crate::common::*;
use crate::rs::{et_sp_aux, gdf_cfg, Mat3, AMPLE, OMEGA_HCORE};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::{DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::dft::{GammaRksConfig, GammaUksConfig, PeriodicGridConfig};
use ferric_pbc::gamma_rsh::{
    gamma_rks_rsh_gradient, gamma_rks_rsh_stress, gamma_roks_rsh_gradient, gamma_uks_rsh_gradient,
    GammaRshExchange,
};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::kdense_aft::KDenseAftConfig;
use ferric_pbc::kdft::{solve_krks_keep, KRksConfig};
use ferric_pbc::kgrad::KGradConfig;
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kroks::{solve_kroks_keep, KRoksConfig};
use ferric_pbc::kscf::{KJkKind, KRetained};
use ferric_pbc::kstress::KStressConfig;
use ferric_pbc::kuks::{solve_kuks_keep, KUksConfig};
use ferric_pbc::kuxcgrad::{kpoint_roks_gradient, kpoint_uks_gradient, KUksGradConfig};
use ferric_pbc::kxcgrad::{kpoint_rks_gradient, KRksGradConfig, KRksGradient, KXcGrid};
use ferric_pbc::kxcstress::{kpoint_rks_stress, KRksStress, KRksStressConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rohf::GammaRoksConfig;
use ferric_pbc::rsgdf::kpoint::KRsGdfConfig;
use ferric_pbc::rsgdf::RsGdf;
use ferric_pbc::rsh::{resolve_periodic_functional_rsh, RshParams};
use ferric_pbc::uhf::GammaUhfIntegrals;
use ndarray::Array2;

/// libxc name of CAM-B3LYP (the friendly spelling does not resolve).
pub const CAM: &str = "HYB_GGA_XC_CAM_B3LYP";
/// The closed-shell triclinic H2 (`run_kgrad_anchor.TRI2`).
pub const H2: [[f64; 3]; 2] = [[0.13, 0.25, 0.31], [0.02, 0.27, 1.66]];
/// The open-shell H3 doublet.
pub const H3D: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.3, 0.2, 1.5], [0.3, 0.2, 2.9]];
/// Metric cut of the fits (FD-clean at ω = 0.33).
pub const LINDEP: f64 = 1e-6;

/// Which SCF.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sp {
    Rks,
    Uks,
    Roks,
}

impl Sp {
    pub fn atoms(self) -> &'static [[f64; 3]] {
        match self {
            Sp::Rks => &H2,
            _ => &H3D,
        }
    }
}

/// Hydrogens at `pos` in `TRI_A` (doublet for an odd count).
pub fn cell_at(pos: &[[f64; 3]]) -> Cell {
    let mut mol = hydrogens(pos);
    mol.multiplicity = if pos.len() % 2 == 1 { 2 } else { 1 };
    Cell::new(mol, TRI_A).unwrap()
}

pub fn grid_cfg() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(14.0);
    g
}

pub fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig::with_omega(OMEGA_HCORE)
}

pub fn rsh_of(f: &str) -> RshParams {
    resolve_periodic_functional_rsh(f).unwrap().2.unwrap()
}

pub fn diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

pub fn max_abs(a: &Array2<f64>) -> f64 {
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

// ------------------------------------------------------------------ Gamma

/// A Gamma SCF config of either spin.
#[derive(Clone)]
pub enum GCfg {
    R(GammaRksConfig),
    U(GammaUksConfig),
    Ro(GammaRoksConfig),
}

/// Which J/K the Gamma run uses: everything on RS-GDF (production), all
/// dense AFT, or a deliberately MIXED pair (the J of one, the exchange of
/// the other) that the routed Fock guard must refuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Jk {
    RsGdf,
    Dense,
    /// Dense Gamma J under the RS-GDF exchange.
    DenseJRsGdfK,
    /// RS-GDF Gamma J under the dense exchange.
    RsGdfJDenseK,
}

/// A converged Gamma RSH run kept for its derivatives.
pub struct G {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub hc: PeriodicHcore,
    /// The exchange the derivative is handed (for the mixed `Jk` cases, NOT
    /// the one the SCF ran on).
    pub ex: GammaRshExchange,
    pub cfg: GCfg,
    pub scf: ferric_scf::result::ScfResult,
}

fn kcfg(exxdiv: ExxDiv) -> KRsGdfConfig {
    gdf_cfg(LINDEP, exxdiv)
}

fn dense_k(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    f: &str,
    x: ExxDiv,
) -> GammaRshExchange {
    let dc = KDenseAftConfig {
        precision: 1e-12,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    };
    GammaRshExchange::build_dense(cell, prep, &hc.s, x, &dc, rsh_of(f)).unwrap()
}

fn rsgdf_k(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: &PreparedBasis,
    hc: &PeriodicHcore,
    f: &str,
    x: ExxDiv,
) -> GammaRshExchange {
    GammaRshExchange::build_rsgdf(cell, prep, aux, &hc.s, x, &kcfg(x), rsh_of(f)).unwrap()
}

/// Gamma RSH SCF of `f` (spin `sp`) at `pos` on the J/K `jk`.
pub fn gamma_run(cell: Cell, sp: Sp, f: &str, exxdiv: ExxDiv, jk: Jk) -> G {
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).unwrap();
    let aux = PreparedBasis::new(cell.mol(), &et_sp_aux()).unwrap();
    let rs_j = matches!(jk, Jk::RsGdf | Jk::RsGdfJDenseK);
    let (gdf_j, dense_j);
    if rs_j {
        gdf_j = Some(RsGdf::build(&cell, &prep, &aux, &hc.s, &kcfg(exxdiv).gdf).unwrap());
        dense_j = None;
    } else {
        gdf_j = None;
        dense_j = Some(
            DenseAftEri::build(
                &cell,
                &prep,
                &hc.s,
                exxdiv,
                1e-12,
                DEFAULT_DENSE_AFT_MAX_BYTES,
            )
            .unwrap(),
        );
    }
    let ints = match (&gdf_j, &dense_j) {
        (Some(g), _) => GammaUhfIntegrals::RsGdf(g),
        (_, Some(e)) => GammaUhfIntegrals::DenseAft(e),
        _ => unreachable!(),
    };
    // The exchange the SCF runs on: the one matching its J for the
    // production cases, the OTHER one for the mixed cases (the derivative is
    // handed the matching one below, so its J and the SCF's differ).
    let scf_k_rsgdf = matches!(jk, Jk::RsGdf | Jk::RsGdfJDenseK);
    let scf_ex = if scf_k_rsgdf {
        rsgdf_k(&cell, &prep, &aux, &hc, f, exxdiv)
    } else {
        dense_k(&cell, &prep, &hc, f, exxdiv)
    };
    let (cfg, scf) = match sp {
        Sp::Rks => {
            let mut c = GammaRksConfig::new(f);
            c.grid = grid_cfg();
            c.exxdiv = exxdiv;
            c.scf.energy_conv = 1e-13;
            c.scf.density_conv = 1e-10;
            c.scf.max_iter = 300;
            let r = ferric_pbc::dft::gamma_rks_rsh(&cell, &prep, &hc, ints, &scf_ex, &c).unwrap();
            (GCfg::R(c), r.scf)
        }
        Sp::Uks => {
            let mut c = GammaUksConfig::new(f);
            c.grid = grid_cfg();
            c.exxdiv = exxdiv;
            c.scf.energy_conv = 1e-13;
            c.scf.density_conv = 1e-10;
            c.scf.max_iter = 300;
            let r = ferric_pbc::dft::gamma_uks_rsh(&cell, &prep, &hc, ints, &scf_ex, &c).unwrap();
            (GCfg::U(c), r.scf)
        }
        Sp::Roks => {
            let mut c = GammaRoksConfig::new(f);
            c.grid = grid_cfg();
            c.exxdiv = exxdiv;
            c.scf.energy_conv = 1e-13;
            c.scf.density_conv = 1e-10;
            let r = ferric_pbc::rohf::gamma_roks_rsh(&cell, &prep, &hc, ints, &scf_ex, &c).unwrap();
            (GCfg::Ro(c), r.scf)
        }
    };
    assert!(
        scf.converged,
        "Gamma {sp:?} {f} {exxdiv:?} {jk:?}: not converged"
    );
    let ex = match jk {
        Jk::RsGdf | Jk::Dense => scf_ex,
        // The derivative gets the exchange of the OTHER source.
        Jk::DenseJRsGdfK => rsgdf_k(&cell, &prep, &aux, &hc, f, exxdiv),
        Jk::RsGdfJDenseK => dense_k(&cell, &prep, &hc, f, exxdiv),
    };
    G {
        cell,
        prep,
        hc,
        ex,
        cfg,
        scf,
    }
}

/// The Gamma SCF energy alone (FD oracle: fresh build + SCF per point).
pub fn gamma_energy(cell: Cell, sp: Sp, f: &str, exxdiv: ExxDiv) -> f64 {
    gamma_run(cell, sp, f, exxdiv, Jk::RsGdf).scf.energy
}

pub fn kgrad_cfg() -> KGradConfig {
    KGradConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

pub fn kstress_cfg() -> KStressConfig {
    KStressConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

/// The routed Gamma force of `g` (`gamma_{rks,uks,roks}_rsh_gradient`).
pub fn gamma_force(g: &G) -> Result<KRksGradient, FerricError> {
    let (c, p, h) = (&g.cell, &g.prep, &hcore_cfg());
    match &g.cfg {
        GCfg::R(d) => gamma_rks_rsh_gradient(c, p, h, &g.scf, d, &g.ex, &kgrad_cfg()),
        GCfg::U(d) => gamma_uks_rsh_gradient(c, p, h, &g.scf, d, &g.ex, &kgrad_cfg()),
        GCfg::Ro(d) => gamma_roks_rsh_gradient(c, p, h, &g.scf, d, &g.ex, &kgrad_cfg()),
    }
}

/// The routed Gamma RKS stress of `g`.
pub fn gamma_stress(g: &G) -> Result<KRksStress, FerricError> {
    match &g.cfg {
        GCfg::R(d) => gamma_rks_rsh_stress(
            &g.cell,
            &g.prep,
            &hcore_cfg(),
            &g.scf,
            d,
            &g.ex,
            &kstress_cfg(),
        ),
        _ => panic!("the open-shell Gamma RSH stress is not routed here"),
    }
}

// ------------------------------------------------------------- 1x1x1 k

/// An independent 1x1x1 k-point SCF on `KRsGdf::build_rsh` (or the dense
/// AFT kernels), kept.
pub struct K1 {
    pub rsgdf: bool,
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub aux: PreparedBasis,
    pub mesh: KPointMesh,
    pub hk: PeriodicHcoreK,
    pub rcfg: KRsGdfConfig,
    pub kept: KRetained,
    pub scf: KScf,
}

pub enum KScf {
    R(ferric_pbc::kscf::KScfResult),
    U(ferric_pbc::kuscf::KUScfResult),
    Ro(ferric_pbc::kroks::KRoScfResult),
}

pub fn k1_run(cell: Cell, sp: Sp, f: &str, exxdiv: ExxDiv) -> K1 {
    k1_run_on(cell, sp, f, exxdiv, true)
}

/// [`k1_run`] on RS-GDF (`rsgdf`) or the dense-AFT J/K.
pub fn k1_run_on(cell: Cell, sp: Sp, f: &str, exxdiv: ExxDiv, rsgdf: bool) -> K1 {
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let aux = PreparedBasis::new(cell.mol(), &et_sp_aux()).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcore_cfg()).unwrap();
    let rcfg = kcfg(exxdiv);
    let jk = if rsgdf {
        KJkKind::RsGdf
    } else {
        KJkKind::Dense
    };
    let dense = KDenseAftConfig {
        precision: 1e-12,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    };
    let auxr = rsgdf.then_some(&aux);
    let (scf, kept) = match sp {
        Sp::Rks => {
            let mut c = KRksConfig::new(&cell, exxdiv, f);
            c.krhf.hcore = hcore_cfg();
            c.krhf.scf.energy_conv = 1e-13;
            c.krhf.scf.grad_conv = 1e-10;
            c.krhf.scf.max_iter = 300;
            c.krhf.jk = jk;
            c.krhf.dense = dense;
            c.krhf.rsgdf = rcfg;
            c.grid = grid_cfg();
            let (r, k) = solve_krks_keep(&cell, &prep, auxr, &mesh, &c).unwrap();
            (KScf::R(r.scf), k)
        }
        Sp::Uks => {
            let mut c = KUksConfig::new(&cell, exxdiv, f);
            c.kuhf.hcore = hcore_cfg();
            c.kuhf.scf.energy_conv = 1e-13;
            c.kuhf.scf.grad_conv = 1e-10;
            c.kuhf.scf.max_iter = 300;
            c.kuhf.jk = jk;
            c.kuhf.dense = dense;
            c.kuhf.rsgdf = rcfg;
            c.grid = grid_cfg();
            let (r, k) = solve_kuks_keep(&cell, &prep, auxr, &mesh, &c).unwrap();
            (KScf::U(r.scf), k)
        }
        Sp::Roks => {
            let mut c = KRoksConfig::new(&cell, exxdiv, f);
            c.rohf.hcore = hcore_cfg();
            c.rohf.scf.energy_conv = 1e-13;
            c.rohf.scf.grad_conv = 1e-10;
            c.rohf.jk = jk;
            c.rohf.dense = dense;
            c.rohf.rsgdf = rcfg;
            c.grid = grid_cfg();
            let (r, k) = solve_kroks_keep(&cell, &prep, auxr, &mesh, &c).unwrap();
            (KScf::Ro(r.scf), k)
        }
    };
    let ok = match &scf {
        KScf::R(r) => r.converged,
        KScf::U(r) => r.converged,
        KScf::Ro(r) => r.converged,
    };
    assert!(ok, "1x1x1 {sp:?} {f} {exxdiv:?}: not converged");
    K1 {
        rsgdf,
        cell,
        prep,
        aux,
        mesh,
        hk,
        rcfg,
        kept,
        scf,
    }
}

fn krks_grad_cfg(exxdiv: ExxDiv) -> KRksGradConfig {
    KRksGradConfig {
        grad: kgrad_cfg(),
        grid: KXcGrid::Atom(grid_cfg()),
        exxdiv: Some(exxdiv),
        ..Default::default()
    }
}

/// The 1x1x1 k-point force of `k`.
pub fn k1_force(k: &K1, f: &str, exxdiv: ExxDiv) -> KRksGradient {
    let jk = k.kept.grad_jk(k.rsgdf.then_some(&k.aux), &k.rcfg).unwrap();
    let (c, p, m, h) = (&k.cell, &k.prep, &k.mesh, &hcore_cfg());
    let uc = KUksGradConfig {
        base: krks_grad_cfg(exxdiv),
        spin_mutation: None,
    };
    match &k.scf {
        KScf::R(r) => kpoint_rks_gradient(c, p, m, h, &k.hk, jk, r, f, &krks_grad_cfg(exxdiv)),
        KScf::U(r) => kpoint_uks_gradient(c, p, m, h, &k.hk, jk, r, f, &uc),
        KScf::Ro(r) => kpoint_roks_gradient(c, p, m, h, &k.hk, jk, r, f, &uc),
    }
    .unwrap()
}

/// The 1x1x1 k-point RKS stress of `k`.
pub fn k1_stress(k: &K1, f: &str, exxdiv: ExxDiv) -> KRksStress {
    let KScf::R(r) = &k.scf else {
        panic!("closed shell only")
    };
    let jk = k
        .kept
        .stress_jk(k.rsgdf.then_some(&k.aux), &k.rcfg)
        .unwrap();
    let cfg = KRksStressConfig {
        stress: kstress_cfg(),
        grid: KXcGrid::Atom(grid_cfg()),
        exxdiv: Some(exxdiv),
        ..Default::default()
    };
    kpoint_rks_stress(
        &k.cell,
        &k.prep,
        &k.mesh,
        &hcore_cfg(),
        &k.hk,
        jk,
        r,
        f,
        &cfg,
    )
    .unwrap()
}

// -------------------------------------------------------------------- FD

pub fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

/// Central FD of the Gamma RS-GDF RSH energy along atom `a`, axis `x`.
pub fn fd_force(sp: Sp, f: &str, exxdiv: ExxDiv, (a, x, h): (usize, usize, f64)) -> f64 {
    let pos = sp.atoms();
    let e = |p: Vec<[f64; 3]>| gamma_energy(cell_at(&p), sp, f, exxdiv);
    (e(moved(pos, a, x, h)) - e(moved(pos, a, x, -h))) / (2.0 * h)
}

pub fn eps_at(i: usize, j: usize, h: f64) -> Mat3 {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    e
}

/// Central FD of the Gamma RS-GDF RSH energy under strain `(i, j)`.
pub fn fd_strain(sp: Sp, f: &str, exxdiv: ExxDiv, (i, j, h): (usize, usize, f64)) -> f64 {
    let cell = cell_at(sp.atoms());
    let e = |s: f64| gamma_energy(cell.strained(&eps_at(i, j, s)).unwrap(), sp, f, exxdiv);
    (e(h) - e(-h)) / (2.0 * h)
}
