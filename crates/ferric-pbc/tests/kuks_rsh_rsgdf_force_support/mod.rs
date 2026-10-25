//! Shared fixtures of the OPEN-SHELL range-separated k-point force on RS-GDF
//! J/K (`pbc_kuks_rsh_rsgdf_force.rs` per commit,
//! `validation_kuks_rsh_rsgdf_force.rs` weekly). The RS-GDF settings (STO-3G,
//! the ET-sp aux, Ewald split `ω_0 = 1`, 1x1x2 mesh, hcore `ω = 0.8`) are the
//! closed-shell file's (`krsh_rsgdf_force_support`, included as `rs`) and the
//! open-shell cells (H3 doublet in `TRI_A`) are the dense-AFT UKS file's
//! (`kuks_rsh_force_support`), so an RS-GDF row here and its dense row there
//! are the same physical problem.
#![allow(dead_code)]

use crate::common::*;
use crate::rs::{gdf_cfg, sys, Sys, AMPLE, OMEGA_HCORE};
use ferric_core::FerricError;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::PeriodicGridConfig;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::KDenseAftConfig;
use ferric_pbc::kgrad::{KGradConfig, KGradJk, KGradMutation, KRsGdfGradSource};
use ferric_pbc::kroks::{solve_kroks_injected, solve_kroks_keep, KRoScfResult, KRoksConfig};
use ferric_pbc::kscf::{KJkKind, KPointInjection, KRetained, KScfConfig};
use ferric_pbc::kuks::{solve_kuks_keep, KUksConfig};
use ferric_pbc::kuscf::{solve_kuks_injected, KPointXcPolarized, KUScfResult};
use ferric_pbc::kuxcgrad::{
    kpoint_null_xc_roks_gradient_rsh, kpoint_null_xc_uks_gradient_rsh, kpoint_roks_gradient,
    kpoint_uks_gradient, KUSpinMutation, KUksGradConfig, KUksGradient,
};
use ferric_pbc::kxcgrad::{KRksGradConfig, KXcGrid};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsh::{resolve_periodic_functional_rsh, RshParams};
use ndarray::Array2;
use num_complex::Complex64 as C64;

/// libxc name of CAM-B3LYP (the friendly spelling does not resolve).
pub const CAM: &str = "HYB_GGA_XC_CAM_B3LYP";

/// The H3 doublet of the dense UKS file (`kuks_rsh_force_support::H3D`).
pub const H3D: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.3, 0.2, 1.5], [0.3, 0.2, 2.9]];

/// Neighbour cutoff of the KS grids (the dense UKS file's `CUTOFF`).
pub const CUTOFF: f64 = 14.0;

/// Exactness-anchor bar (the task's `≤ 1e-10`).
pub const ANCHOR_TOL: f64 = 1e-10;

/// A CAM-type separation whose attenuated piece is live on `TRI_A` (the
/// closed-shell RS-GDF file's TRI2 row).
pub fn cam_type() -> RshParams {
    RshParams {
        omega: 0.33,
        c_sr: 0.19,
        c_lr: 0.65,
    }
}

/// An HSE-type separation (`c_lr = 0`, `ω = 0.11`).
pub fn hse_type() -> RshParams {
    RshParams {
        omega: 0.11,
        c_sr: 0.25,
        c_lr: 0.0,
    }
}

pub fn rsh_of(f: &str) -> RshParams {
    resolve_periodic_functional_rsh(f).unwrap().2.unwrap()
}

/// Which lattice a fixture uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lat {
    /// The triclinic `TRI_A`.
    Tri,
    /// A 6.5 Bohr cube (HSE's `ω = 0.11` has q = 0 K vectors inside its cut).
    Cube65,
}

/// Hydrogens at `pos` with multiplicity `mult` on the lattice `lat`.
pub fn cell_of(lat: Lat, mult: usize, pos: &[[f64; 3]]) -> Cell {
    let mut mol = hydrogens(pos);
    mol.multiplicity = mult;
    let a = match lat {
        Lat::Tri => TRI_A,
        Lat::Cube65 => cubic(6.5),
    };
    Cell::new(mol, a).unwrap()
}

pub fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

pub fn diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

pub fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// `Σ_k Σ |D_α − D_β|`.
pub fn asym(da: &[Array2<C64>], db: &[Array2<C64>]) -> f64 {
    da.iter()
        .zip(db)
        .map(|(a, b)| (a - b).iter().map(|z| z.norm()).sum::<f64>())
        .sum()
}

/// `(N_α, N_β)` of a cell.
pub fn nocc(cell: &Cell) -> (usize, usize) {
    let ne = cell.mol().nelec() as usize;
    let nb = (ne + 1 - cell.mol().multiplicity) / 2;
    (ne - nb, nb)
}

pub fn gcfg(
    exxdiv: ExxDiv,
    m: Option<KGradMutation>,
    sm: Option<KUSpinMutation>,
    grid: Option<PeriodicGridConfig>,
) -> KUksGradConfig {
    KUksGradConfig {
        base: KRksGradConfig {
            grad: KGradConfig {
                budget_bytes: Some(AMPLE),
                mutation: m,
                ..Default::default()
            },
            grid: grid.map_or(KXcGrid::default(), KXcGrid::Atom),
            exxdiv: Some(exxdiv),
            ..Default::default()
        },
        spin_mutation: sm,
    }
}

// ------------------------------------------------ zero-XC open-shell SCF

/// A zero polarized XC (`E_xc = 0`, `V_xc,σ = 0`).
pub struct ZeroXcPol(pub usize);
impl KPointXcPolarized for ZeroXcPol {
    #[allow(clippy::type_complexity)]
    fn build_polarized(
        &mut self,
        da: &[Array2<C64>],
        _db: &[Array2<C64>],
    ) -> Result<(f64, Vec<Array2<C64>>, Vec<Array2<C64>>), FerricError> {
        let z = || da.iter().map(|_| Array2::zeros((self.0, self.0))).collect();
        Ok((0.0, z(), z()))
    }
}

/// UKS or ROKS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spin {
    Uks,
    Roks,
}

/// The converged open-shell SCF of either kind.
pub enum OScf {
    U(KUScfResult),
    Ro(KRoScfResult),
}

impl OScf {
    pub fn energy(&self) -> f64 {
        match self {
            OScf::U(r) => r.energy,
            OScf::Ro(r) => r.energy,
        }
    }
    pub fn asym(&self) -> f64 {
        match self {
            OScf::U(r) => asym(&r.density_alpha, &r.density_beta),
            OScf::Ro(r) => asym(&r.density_alpha, &r.density_beta),
        }
    }
}

/// A converged zero-XC open-shell k SCF on RS-GDF J/K: `rsh = Some` is the
/// range-separated build (`KRsGdf::build_rsh`, exchange scale `a = 1`, the
/// mixed Madelung constant for ewald); `None` the plain fit with fraction `a`.
pub struct OZx {
    pub sys: Sys,
    pub cfg: KRsGdfConfig,
    pub gdf: KRsGdf,
    pub scf: OScf,
    pub rsh: Option<RshParams>,
    pub a: f64,
    pub exxdiv: ExxDiv,
}

pub fn zx_open(
    cell: Cell,
    spin: Spin,
    rsh: Option<RshParams>,
    a: f64,
    lindep: f64,
    exxdiv: ExxDiv,
) -> OZx {
    let s = sys(cell);
    let cfg = gdf_cfg(lindep, exxdiv);
    let gdf = match rsh {
        Some(r) => KRsGdf::build_rsh(&s.cell, &s.prep, &s.aux, &s.mesh, &s.hk.s, &cfg, r),
        None => KRsGdf::build(&s.cell, &s.prep, &s.aux, &s.mesh, &s.hk.s, &cfg),
    }
    .unwrap();
    let sc = KScfConfig {
        energy_conv: 1e-13,
        grad_conv: 1e-10,
        max_iter: 600,
        ..Default::default()
    };
    let inj = KPointInjection {
        s: s.hk.s.clone(),
        h: s.hk.h.clone(),
        vnn: s.hk.enn,
        jk: Box::new(gdf.jk_builder()),
    };
    let (na, nb) = nocc(&s.cell);
    let scf = match spin {
        Spin::Uks => {
            let mut z = ZeroXcPol(s.prep.nbasis());
            OScf::U(solve_kuks_injected(&s.cell, &s.mesh, &sc, inj, na, nb, &mut z, a).unwrap())
        }
        Spin::Roks => {
            OScf::Ro(solve_kroks_injected(&s.cell, &s.mesh, &sc, inj, na, nb, None, a).unwrap())
        }
    };
    let ok = match &scf {
        OScf::U(r) => r.converged,
        OScf::Ro(r) => r.converged,
    };
    assert!(
        ok,
        "zero-XC open-shell RS-GDF SCF ({spin:?}, {rsh:?}) not converged"
    );
    OZx {
        sys: s,
        cfg,
        gdf,
        scf,
        rsh,
        a,
        exxdiv,
    }
}

pub fn zx_open_force_m(
    z: &OZx,
    m: Option<KGradMutation>,
    sm: Option<KUSpinMutation>,
) -> Result<KUksGradient, FerricError> {
    let jk = KGradJk::RsGdf(KRsGdfGradSource {
        gdf: &z.gdf,
        cfg: &z.cfg,
        aux: &z.sys.aux,
    });
    let hc = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);
    let c = gcfg(z.exxdiv, m, sm, None);
    let s = &z.sys;
    match &z.scf {
        OScf::U(r) => kpoint_null_xc_uks_gradient_rsh(
            &s.cell, &s.prep, &s.mesh, &hc, &s.hk, jk, r, z.a, z.rsh, &c,
        ),
        OScf::Ro(r) => kpoint_null_xc_roks_gradient_rsh(
            &s.cell, &s.prep, &s.mesh, &hc, &s.hk, jk, r, z.a, z.rsh, &c,
        ),
    }
}

pub fn zx_open_force(z: &OZx) -> KUksGradient {
    zx_open_force_m(z, None, None).unwrap()
}

/// Central FD of the zero-XC open-shell RS-GDF energy (fresh build + SCF per
/// displacement).
#[allow(clippy::too_many_arguments)]
pub fn zx_open_fd(
    (lat, mult): (Lat, usize),
    pos: &[[f64; 3]],
    (a, x, h): (usize, usize, f64),
    spin: Spin,
    rsh: RshParams,
    lindep: f64,
    exxdiv: ExxDiv,
) -> f64 {
    let e = |p: Vec<[f64; 3]>| {
        zx_open(cell_of(lat, mult, &p), spin, Some(rsh), 1.0, lindep, exxdiv)
            .scf
            .energy()
    };
    (e(moved(pos, a, x, h)) - e(moved(pos, a, x, -h))) / (2.0 * h)
}

/// The production-wired defects of the open-shell RS-GDF RSH force
/// (spin bookkeeping of both fit passes and the M-term, the attenuated fit's
/// scale and `c0`, the M-term constant).
pub fn ledger_mutants() -> Vec<(&'static str, Option<KGradMutation>, Option<KUSpinMutation>)> {
    use KGradMutation as M;
    vec![
        ("ExchDropBeta", Some(M::ExchDropBeta), None),
        ("ExchBetaUsesAlpha", Some(M::ExchBetaUsesAlpha), None),
        (
            "ExchClosedShellFactor",
            Some(M::ExchClosedShellFactor),
            None,
        ),
        ("MTermTotalDensity", Some(M::MTermTotalDensity), None),
        ("PulayKSwap", None, Some(KUSpinMutation::PulayKSwap)),
        ("RshFitNoScale", Some(M::RshFitNoScale), None),
        ("RshFitKeepC0", Some(M::RshFitKeepC0), None),
        ("RshMadelungUnmixed", Some(M::RshMadelungUnmixed), None),
        ("RshMadelungScaled", Some(M::RshMadelungScaled), None),
        ("RshPlainMadelung", Some(M::RshPlainMadelung), None),
    ]
}

/// The defects that only reassign spins: invisible on a singlet.
pub const SPIN_ONLY: [&str; 3] = ["ExchBetaUsesAlpha", "MTermTotalDensity", "PulayKSwap"];

// ------------------------------------------------ KS SCF + force

pub fn grid_cfg() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(CUTOFF);
    g
}

fn scf_tight(sc: &mut KScfConfig) {
    sc.energy_conv = 1e-13;
    sc.grad_conv = 1e-10;
    sc.max_iter = 300;
}

/// A converged open-shell KS run on RS-GDF (`lindep = Some`) or dense AFT,
/// kept for its force.
pub struct OKs {
    pub sys: Sys,
    pub rcfg: KRsGdfConfig,
    pub scf: OScf,
    pub kept: KRetained,
    pub f: String,
    pub exxdiv: ExxDiv,
    pub rsgdf: bool,
}

pub fn ks_open(cell: Cell, spin: Spin, f: &str, lindep: Option<f64>, exxdiv: ExxDiv) -> OKs {
    let s = sys(cell);
    let rcfg = gdf_cfg(lindep.unwrap_or(1e-10), exxdiv);
    let dense = KDenseAftConfig {
        precision: 1e-12,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    };
    let jk = if lindep.is_some() {
        KJkKind::RsGdf
    } else {
        KJkKind::Dense
    };
    let aux = lindep.map(|_| &s.aux);
    let (scf, kept) = match spin {
        Spin::Uks => {
            let mut c = KUksConfig::new(&s.cell, exxdiv, f);
            c.kuhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);
            scf_tight(&mut c.kuhf.scf);
            c.kuhf.dense = dense;
            c.kuhf.jk = jk;
            c.kuhf.rsgdf = rcfg;
            c.grid = grid_cfg();
            let (r, k) = solve_kuks_keep(&s.cell, &s.prep, aux, &s.mesh, &c)
                .unwrap_or_else(|e| panic!("solve_kuks {f}: {e}"));
            (OScf::U(r.scf), k)
        }
        Spin::Roks => {
            let mut c = KRoksConfig::new(&s.cell, exxdiv, f);
            c.rohf.hcore = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);
            c.rohf.scf.energy_conv = 1e-13;
            c.rohf.scf.grad_conv = 1e-10;
            c.rohf.dense = dense;
            c.rohf.jk = jk;
            c.rohf.rsgdf = rcfg;
            c.grid = grid_cfg();
            let (r, k) = solve_kroks_keep(&s.cell, &s.prep, aux, &s.mesh, &c)
                .unwrap_or_else(|e| panic!("solve_kroks {f}: {e}"));
            (OScf::Ro(r.scf), k)
        }
    };
    let ok = match &scf {
        OScf::U(r) => r.converged,
        OScf::Ro(r) => r.converged,
    };
    assert!(ok, "{f} {spin:?} not converged");
    OKs {
        sys: s,
        rcfg,
        scf,
        kept,
        f: f.into(),
        exxdiv,
        rsgdf: lindep.is_some(),
    }
}

pub fn ks_open_force_m(
    ks: &OKs,
    m: Option<KGradMutation>,
    sm: Option<KUSpinMutation>,
) -> Result<KUksGradient, FerricError> {
    let aux = ks.rsgdf.then_some(&ks.sys.aux);
    let jk = ks.kept.grad_jk(aux, &ks.rcfg)?;
    let hc = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);
    let c = gcfg(ks.exxdiv, m, sm, Some(grid_cfg()));
    let s = &ks.sys;
    match &ks.scf {
        OScf::U(r) => kpoint_uks_gradient(
            &s.cell,
            &s.prep,
            &s.mesh,
            &hc,
            &ks.kept.hk,
            jk,
            r,
            &ks.f,
            &c,
        ),
        OScf::Ro(r) => kpoint_roks_gradient(
            &s.cell,
            &s.prep,
            &s.mesh,
            &hc,
            &ks.kept.hk,
            jk,
            r,
            &ks.f,
            &c,
        ),
    }
}

pub fn ks_open_force(ks: &OKs) -> KUksGradient {
    ks_open_force_m(ks, None, None).unwrap_or_else(|e| panic!("{} force: {e}", ks.f))
}

/// Central FD of the open-shell KS energy (fresh SCF per displacement).
#[allow(clippy::too_many_arguments)]
pub fn ks_open_fd(
    (lat, mult): (Lat, usize),
    pos: &[[f64; 3]],
    (a, x, h): (usize, usize, f64),
    spin: Spin,
    f: &str,
    lindep: Option<f64>,
    exxdiv: ExxDiv,
) -> f64 {
    let e = |p: Vec<[f64; 3]>| {
        ks_open(cell_of(lat, mult, &p), spin, f, lindep, exxdiv)
            .scf
            .energy()
    };
    (e(moved(pos, a, x, h)) - e(moved(pos, a, x, -h))) / (2.0 * h)
}
