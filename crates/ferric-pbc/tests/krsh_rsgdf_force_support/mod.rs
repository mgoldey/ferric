//! Shared fixtures of the range-separated k-point force on RS-GDF J/K
//! (`pbc_krsh_rsgdf_force.rs` per commit, `validation_krsh_rsgdf_force.rs`
//! weekly): the prototype's triclinic H2 (`reference/pbc/rsgdf_rsh_force_proto.md`)
//! with STO-3G and the ET-sp aux, the attenuated fit alone at a FIXED density,
//! and the closed-shell RKS SCF + force on RS-GDF and on dense AFT.
#![allow(dead_code)]

use super::common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::PeriodicGridConfig;
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdft::{solve_krks_keep, KRksConfig, KRksResult};
use ferric_pbc::kgrad::{
    kpoint_fit_exchange_gradient_fixed_d, KGradConfig, KGradMutation, KRsGdfGradSource,
};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{solve_krhf_injected, KJkKind, KPointInjection, KRetained, KScfConfig};
use ferric_pbc::kxcgrad::{kpoint_rks_gradient, KRksGradConfig, KRksGradient, KXcGrid};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use ndarray::Array2;
use num_complex::Complex64 as C64;
use std::collections::HashMap;

/// `run_kgrad_anchor.TRI2`: the first two `TRI_MOVED` hydrogens in `TRI_A`.
pub const TRI2_ATOMS: [[f64; 3]; 2] = [[0.13, 0.25, 0.31], [0.02, 0.27, 1.66]];
/// Off-axis H2 for the larger cubic cells (HSE06 needs a > ~6 Bohr so that
/// the q = 0 class carries K vectors inside its cut, ω = 0.11).
pub const H2_BIG: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
pub const MESH: [usize; 3] = [1, 1, 2];
/// RS-GDF Ewald split of the full-Coulomb fit (the prototype's `w0`).
pub const OMEGA0: f64 = 1.0;
pub const OMEGA_HCORE: f64 = 0.8;
pub const AMPLE: usize = 1 << 31;
/// The production metric cut (`DEFAULT_RSGDF_LINDEP`).
pub const LINDEP_DEFAULT: f64 = 1e-10;
/// The cut that makes the ω = 0.33 attenuated fit FD-clean (prototype note,
/// "Conditioning of the attenuated fit").
pub const LINDEP_FD: f64 = 1e-6;
/// libxc name of CAM-B3LYP (the friendly spelling does not resolve).
pub const CAM: &str = "HYB_GGA_XC_CAM_B3LYP";

/// The prototype's ET-sp aux on H: s(4.7, 1.9, 0.75, 0.3) + p(1.25, 0.5).
pub fn et_sp_aux() -> BasisSet {
    let mut shells = Vec::new();
    for (l, a) in [(0, 4.7), (0, 1.9), (0, 0.75), (0, 0.3), (1, 1.25), (1, 0.5)] {
        shells.push(Shell {
            l,
            pure: false,
            exponents: vec![a],
            coefficients: vec![1.0],
        });
    }
    let mut m = HashMap::new();
    m.insert(1, shells);
    BasisSet {
        name: "et-sp-aux-H".into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

pub fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig::with_omega(OMEGA_HCORE)
}

pub fn gdf_cfg(lindep: f64, exxdiv: ExxDiv) -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: OMEGA0,
            lindep,
            exxdiv,
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        mutation: None,
    }
}

pub fn tri2(pos: &[[f64; 3]]) -> Cell {
    Cell::new(hydrogens(pos), TRI_A).unwrap()
}

pub fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

/// Every atom shifted by `d` (a rigid translation: the energy is invariant).
pub fn translated(pos: &[[f64; 3]], d: [f64; 3]) -> Vec<[f64; 3]> {
    pos.iter()
        .map(|r| [r[0] + d[0], r[1] + d[1], r[2] + d[2]])
        .collect()
}

pub fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

pub fn diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// Cell, bases, mesh, hcore of one geometry (STO-3G, ET-sp aux).
pub struct Sys {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub aux: PreparedBasis,
    pub mesh: KPointMesh,
    pub hk: PeriodicHcoreK,
}

pub fn sys(cell: Cell) -> Sys {
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let aux = PreparedBasis::new(cell.mol(), &et_sp_aux()).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, MESH).unwrap();
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcore_cfg()).unwrap();
    Sys {
        cell,
        prep,
        aux,
        mesh,
        hk,
    }
}

pub fn plain_fit(s: &Sys, cfg: &KRsGdfConfig) -> KRsGdf {
    KRsGdf::build(&s.cell, &s.prep, &s.aux, &s.mesh, &s.hk.s, cfg).unwrap()
}

/// The attenuated fit alone at `omega` (cut `gcut`, `None` = production).
pub fn att_fit(s: &Sys, cfg: &KRsGdfConfig, omega: f64, gcut: Option<f64>) -> KRsGdf {
    KRsGdf::build_attenuated_only(&s.cell, &s.prep, &s.aux, &s.mesh, &s.hk.s, cfg, omega, gcut)
        .unwrap()
        .expect("the attenuated fit has K vectors")
}

/// The production Gaussian cut `2ω√ln(1/ε)` of `cfg`.
pub fn prod_gcut(cfg: &KRsGdfConfig, omega: f64) -> f64 {
    2.0 * omega * (1.0 / cfg.gdf.precision).ln().sqrt()
}

/// The converged k-RHF density on the plain fit (`exxdiv = none`), held fixed
/// by the fixed-density anchors.
pub fn fixed_density(s: &Sys, gdf: &KRsGdf) -> Vec<Array2<C64>> {
    let sc = KScfConfig {
        energy_conv: 1e-12,
        grad_conv: 1e-9,
        max_iter: 200,
        ..Default::default()
    };
    let inj = KPointInjection {
        s: s.hk.s.clone(),
        h: s.hk.h.clone(),
        vnn: s.hk.enn,
        jk: Box::new(gdf.jk_builder_with_madelung(0.0)),
    };
    let r = solve_krhf_injected(&s.cell, &s.mesh, &sc, inj).unwrap();
    assert!(r.converged);
    r.densities
}

/// `E_x = −(1/4N_k) Σ_k Re tr[D(k) K(k)]` of one fit (no Madelung term).
pub fn ex_energy(gdf: &KRsGdf, d: &[Array2<C64>]) -> f64 {
    let (nk, n) = (d.len(), d[0].nrows());
    let mut j: Vec<Array2<C64>> = (0..nk).map(|_| Array2::zeros((n, n))).collect();
    let mut k = j.clone();
    gdf.contract(d, 0.0, &mut j, &mut k).unwrap();
    let mut e = 0.0;
    for (dk, kk) in d.iter().zip(&k) {
        for a in 0..n {
            for b in 0..n {
                e += (dk[(a, b)] * kk[(b, a)]).re;
            }
        }
    }
    -0.25 * e / nk as f64
}

/// The fixed-density exchange force of one fit (`attenuated = Some((ω, cut))`
/// for an attenuated fit).
pub fn fit_force(
    s: &Sys,
    gdf: &KRsGdf,
    cfg: &KRsGdfConfig,
    d: &[Array2<C64>],
    attenuated: Option<(f64, f64)>,
) -> Result<Array2<f64>, ferric_core::FerricError> {
    kpoint_fit_exchange_gradient_fixed_d(
        &s.cell,
        &s.prep,
        &s.mesh,
        &hcore_cfg(),
        &s.hk.s,
        KRsGdfGradSource {
            gdf,
            cfg,
            aux: &s.aux,
        },
        d,
        attenuated,
    )
}

/// The attenuated exchange energy at FIXED `d` with atom `a` moved by `h`
/// along `x` (production cut).
pub fn att_energy_at(
    cell_of: &dyn Fn(&[[f64; 3]]) -> Cell,
    pos: &[[f64; 3]],
    cfg: &KRsGdfConfig,
    omega: f64,
    d: &[Array2<C64>],
) -> f64 {
    let s = sys(cell_of(pos));
    ex_energy(&att_fit(&s, cfg, omega, None), d)
}

// ------------------------------------------------------------ KS SCF + force

pub fn grid_cfg() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(10.0);
    g
}

/// Closed-shell KRKS config on RS-GDF (`aux = true`) or dense AFT.
pub fn krks_cfg(cell: &Cell, f: &str, rsgdf: Option<f64>, exxdiv: ExxDiv) -> KRksConfig {
    let mut cfg = KRksConfig::new(cell, exxdiv, f);
    cfg.krhf.hcore = hcore_cfg();
    cfg.krhf.scf.energy_conv = 1e-13;
    cfg.krhf.scf.grad_conv = 1e-10;
    cfg.krhf.scf.max_iter = 300;
    cfg.krhf.dense.precision = 1e-12;
    cfg.krhf.dense.budget_bytes = Some(AMPLE);
    if let Some(lindep) = rsgdf {
        cfg.krhf.jk = KJkKind::RsGdf;
        cfg.krhf.rsgdf = gdf_cfg(lindep, exxdiv);
    }
    cfg.grid = grid_cfg();
    cfg
}

/// A converged KRKS run kept for its force.
pub struct Ks {
    pub sys: Sys,
    pub cfg: KRksConfig,
    pub res: KRksResult,
    pub kept: KRetained,
    pub f: String,
    pub exxdiv: ExxDiv,
    pub rsgdf: bool,
}

/// KRKS of functional `f` on `cell` (RS-GDF with metric cut `lindep`, or
/// dense AFT for `None`).
pub fn ks_run(cell: Cell, f: &str, lindep: Option<f64>, exxdiv: ExxDiv) -> Ks {
    let s = sys(cell);
    let cfg = krks_cfg(&s.cell, f, lindep, exxdiv);
    let aux = lindep.map(|_| &s.aux);
    let (res, kept) = solve_krks_keep(&s.cell, &s.prep, aux, &s.mesh, &cfg).unwrap();
    assert!(res.scf.converged, "{f}: SCF not converged");
    Ks {
        sys: s,
        cfg,
        res,
        kept,
        f: f.into(),
        exxdiv,
        rsgdf: lindep.is_some(),
    }
}

pub fn ks_energy(cell: Cell, f: &str, lindep: Option<f64>, exxdiv: ExxDiv) -> f64 {
    ks_run(cell, f, lindep, exxdiv).res.scf.energy
}

pub fn ks_force_m(
    ks: &Ks,
    m: Option<KGradMutation>,
) -> Result<KRksGradient, ferric_core::FerricError> {
    let aux = ks.rsgdf.then_some(&ks.sys.aux);
    let jk = ks.kept.grad_jk(aux, &ks.cfg.krhf.rsgdf)?;
    kpoint_rks_gradient(
        &ks.sys.cell,
        &ks.sys.prep,
        &ks.sys.mesh,
        &ks.cfg.krhf.hcore,
        &ks.kept.hk,
        jk,
        &ks.res.scf,
        &ks.f,
        &KRksGradConfig {
            grad: KGradConfig {
                budget_bytes: Some(AMPLE),
                mutation: m,
                ..Default::default()
            },
            grid: KXcGrid::Atom(grid_cfg()),
            exxdiv: Some(ks.exxdiv),
            ..Default::default()
        },
    )
}

pub fn ks_force(ks: &Ks) -> KRksGradient {
    ks_force_m(ks, None).unwrap()
}

/// Central FD of the KRKS energy (fresh SCF per displacement).
pub fn ks_fd(
    cell_of: &dyn Fn(&[[f64; 3]]) -> Cell,
    pos: &[[f64; 3]],
    (a, x, h): (usize, usize, f64),
    f: &str,
    lindep: Option<f64>,
    exxdiv: ExxDiv,
) -> f64 {
    let e = |p: Vec<[f64; 3]>| ks_energy(cell_of(&p), f, lindep, exxdiv);
    (e(moved(pos, a, x, h)) - e(moved(pos, a, x, -h))) / (2.0 * h)
}

// ------------------------------------------------ zero-XC RSH SCF + force

/// A zero XC (`E_xc = 0`, `V_xc = 0`): the RSH exchange-only energy, free of
/// the XC grid (whose FD noise is the same on dense AFT, see the validation
/// file's header).
pub struct ZeroXc(pub usize);
impl ferric_pbc::kscf::KPointXc for ZeroXc {
    fn build(
        &mut self,
        dm: &[Array2<C64>],
    ) -> Result<(f64, Vec<Array2<C64>>), ferric_core::FerricError> {
        Ok((
            0.0,
            dm.iter().map(|_| Array2::zeros((self.0, self.0))).collect(),
        ))
    }
}

/// HSE06-like and CAM-B3LYP-like fractions.
pub fn rsh_params(c_sr: f64, c_lr: f64, omega: f64) -> ferric_pbc::rsh::RshParams {
    ferric_pbc::rsh::RshParams { omega, c_sr, c_lr }
}

/// A converged zero-XC k SCF on the RS-GDF RSH J/K (`KRsGdf::build_rsh`,
/// exchange scale 1, the mixed Madelung constant for ewald).
pub struct Zx {
    pub sys: Sys,
    pub cfg: KRsGdfConfig,
    pub gdf: KRsGdf,
    pub scf: ferric_pbc::kscf::KScfResult,
    pub rsh: ferric_pbc::rsh::RshParams,
    pub exxdiv: ExxDiv,
}

pub fn zx_run(cell: Cell, rsh: ferric_pbc::rsh::RshParams, lindep: f64, exxdiv: ExxDiv) -> Zx {
    let s = sys(cell);
    let cfg = gdf_cfg(lindep, exxdiv);
    let gdf = KRsGdf::build_rsh(&s.cell, &s.prep, &s.aux, &s.mesh, &s.hk.s, &cfg, rsh).unwrap();
    let sc = KScfConfig {
        energy_conv: 1e-13,
        grad_conv: 1e-10,
        max_iter: 300,
        ..Default::default()
    };
    let inj = KPointInjection {
        s: s.hk.s.clone(),
        h: s.hk.h.clone(),
        vnn: s.hk.enn,
        jk: Box::new(gdf.jk_builder()),
    };
    let scf = ferric_pbc::kscf::solve_krks_injected(
        &s.cell,
        &s.mesh,
        &sc,
        inj,
        &mut ZeroXc(s.prep.nbasis()),
        1.0,
    )
    .unwrap();
    assert!(scf.converged, "zero-XC RSH SCF not converged");
    Zx {
        sys: s,
        cfg,
        gdf,
        scf,
        rsh,
        exxdiv,
    }
}

pub fn zx_force_m(
    z: &Zx,
    m: Option<KGradMutation>,
) -> Result<KRksGradient, ferric_core::FerricError> {
    ferric_pbc::kxcgrad::kpoint_null_xc_gradient_rsh(
        &z.sys.cell,
        &z.sys.prep,
        &z.sys.mesh,
        &hcore_cfg(),
        &z.sys.hk,
        ferric_pbc::kgrad::KGradJk::RsGdf(KRsGdfGradSource {
            gdf: &z.gdf,
            cfg: &z.cfg,
            aux: &z.sys.aux,
        }),
        &z.scf,
        1.0,
        Some(z.rsh),
        &KRksGradConfig {
            grad: KGradConfig {
                budget_bytes: Some(AMPLE),
                mutation: m,
                ..Default::default()
            },
            exxdiv: Some(z.exxdiv),
            ..Default::default()
        },
    )
}

/// Central FD of the zero-XC RSH energy (fresh build + SCF per displacement).
pub fn zx_fd(
    cell_of: &dyn Fn(&[[f64; 3]]) -> Cell,
    pos: &[[f64; 3]],
    (a, x, h): (usize, usize, f64),
    rsh: ferric_pbc::rsh::RshParams,
    lindep: f64,
    exxdiv: ExxDiv,
) -> f64 {
    let e = |p: Vec<[f64; 3]>| zx_run(cell_of(&p), rsh, lindep, exxdiv).scf.energy;
    (e(moved(pos, a, x, h)) - e(moved(pos, a, x, -h))) / (2.0 * h)
}

// ------------------------------------------------ zero-XC RSH stress

pub type Mat3 = [[f64; 3]; 3];

pub fn zx_stress_m(
    z: &Zx,
    m: Option<ferric_pbc::kstress::KStressMutation>,
) -> Result<ferric_pbc::kxcstress::KRksStress, ferric_core::FerricError> {
    ferric_pbc::kxcstress::kpoint_null_xc_stress_rsh(
        &z.sys.cell,
        &z.sys.prep,
        &z.sys.mesh,
        &hcore_cfg(),
        &z.sys.hk,
        ferric_pbc::kstress::KStressJk::RsGdf(KRsGdfGradSource {
            gdf: &z.gdf,
            cfg: &z.cfg,
            aux: &z.sys.aux,
        }),
        &z.scf,
        1.0,
        Some(z.rsh),
        &ferric_pbc::kxcstress::KRksStressConfig {
            stress: ferric_pbc::kstress::KStressConfig {
                budget_bytes: Some(AMPLE),
                mutation: m,
                ..Default::default()
            },
            exxdiv: Some(z.exxdiv),
            ..Default::default()
        },
    )
}

pub fn eps_at(i: usize, j: usize, h: f64) -> Mat3 {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    e
}

/// Central FD of the zero-XC RSH energy under strain component `(i, j)`.
pub fn zx_fd_strain(
    cell: &Cell,
    (i, j, h): (usize, usize, f64),
    rsh: ferric_pbc::rsh::RshParams,
    lindep: f64,
    exxdiv: ExxDiv,
) -> f64 {
    let e = |s: f64| {
        zx_run(
            cell.strained(&eps_at(i, j, s)).unwrap(),
            rsh,
            lindep,
            exxdiv,
        )
        .scf
        .energy
    };
    (e(h) - e(-h)) / (2.0 * h)
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
