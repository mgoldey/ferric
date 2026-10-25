//! Shared setup of `pbc_kks_ecp_derivs.rs` (always-on) and
//! `validation_kks_ecp_derivs.rs` (weekly): HI / H2I cells with the LANL2DZ
//! ECP on I (the fixtures of `pbc_kpoint_ecp.rs`), one SCF + force + stress
//! harness over the four k-point Kohn-Sham / ROHF routes (RKS, UKS, ROKS,
//! ROHF), dense AFT or RS-GDF, and the finite-difference helpers.
// LANL2DZ exponents such as 0.318 are basis data, not approximations of 1/pi.
#![allow(clippy::approx_constant, dead_code)]

use crate::common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_core::ecp::{EcpDef, EcpShell, EcpTerm};
use ferric_core::mol::{Atom, Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::PeriodicGridConfig;
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::KDenseAftEri;
use ferric_pbc::kdft::{solve_krks, KRksConfig, KRksResult};
use ferric_pbc::kgrad::{KGradConfig, KGradJk, KGradMutation, KRsGdfGradSource};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kroks::{solve_krohf, solve_kroks, KRohfConfig, KRoksConfig, KRoksResult};
use ferric_pbc::kscf::KJkKind;
use ferric_pbc::kstress::{KStressConfig, KStressJk, KStressMutation};
use ferric_pbc::kuks::{solve_kuks, KUksConfig, KUksResult};
use ferric_pbc::kuxcgrad::{
    kpoint_rohf_gradient, kpoint_roks_gradient, kpoint_uks_gradient, KUksGradConfig,
};
use ferric_pbc::kuxcstress::{
    kpoint_rohf_stress, kpoint_roks_stress, kpoint_uks_stress, KUksStressConfig,
};
use ferric_pbc::kxcgrad::{kpoint_rks_gradient, KRksGradConfig, KRksGradient, KXcGrid};
use ferric_pbc::kxcstress::{kpoint_rks_stress, KRksStress, KRksStressConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use ferric_pbc::stress::Mat3;
use ndarray::Array2;
use std::collections::HashMap;

pub const OMEGA: f64 = 0.8;
pub const AMPLE: usize = 1 << 31;
/// Explicit shared neighbour cutoff of the atom grid. The energy has one-step
/// jumps where a neighbour atom crosses the cutoff, which break FD when a step
/// straddles one (`pbc_kxcgrad.rs`). MEASURED on these cells
/// (`pbc_kks_ecp_derivs.rs::probe_all_nine_strains`, 1x1x3, |analytic − FD|
/// at h = 1.25e-5 / 2.5e-5 / 5e-5):
/// RKS PBE HI, cutoff 14: eps_xx 1.0e-9 / 6.0e-7 / 2.3e-5 (not h^2: a jump
/// inside ±2.5e-5); cutoff 12: eps_xx 1.1e-9 / 4.3e-9 / 9.6e-7; cutoff 16:
/// every component h^2, <= 4.3e-9 at 2.5e-5.
/// UKS PBE H2I, cutoff 14: eps_xx 9.4e-10 / 2.1e-8 / 8.9e-6; cutoff 16:
/// eps_yy 2.1e-9 / 7.9e-6 / 4.0e-6, eps_xy 2.9e-10 / 4.3e-10 / 4.0e-6;
/// cutoff 20: five components miss 1.05e-5 / 5.2e-6 / 2.6e-6 (∝ 1/h: a jump AT
/// the reference geometry). Jumps get denser as the cutoff grows (more
/// point–atom distances near it). At h = 1.25e-5 every component of the
/// cutoff-14 and -16 probes is <= 2.1e-9, so the strain anchors difference at
/// 1.25e-5 (`validation_kks_ecp_derivs.rs::strain_fd`); a straddled jump
/// fails the anchor rather than being retried.
pub const CUTOFF: f64 = 16.0;
pub const HI_A: [[f64; 3]; 3] = [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]];
/// HI off the common axis (`pbc_kpoint_ecp.rs`'s `MOVED`): H, I.
pub const HI_POS: [[f64; 3]; 2] = [[0.3, 0.2, 0.4], [0.9, -0.4, 3.3]];
/// H2I doublet (`pbc_kpoint_ecp.rs`): H, H, I; 5 alpha / 4 beta valence.
pub const H2I_POS: [[f64; 3]; 3] = [[0.3, 0.2, 0.4], [2.0, 0.6, 1.0], [0.9, -0.4, 3.3]];
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

// ------------------------------------------------------------ basis / cells

fn shell(l: i32, exps: &[f64], coefs: &[f64]) -> Shell {
    let lf = l as f64;
    let mut s = 0.0;
    for (a, ca) in exps.iter().zip(coefs) {
        for (b, cb) in exps.iter().zip(coefs) {
            s += ca * cb * (2.0 * (a * b).sqrt() / (a + b)).powf(lf + 1.5);
        }
    }
    Shell {
        l,
        pure: false,
        exponents: exps.to_vec(),
        coefficients: coefs.iter().map(|c| c / s.sqrt()).collect(),
    }
}

pub fn lanl2dz_i_ecp() -> EcpDef {
    let ch = |l: i32, t: &[(i32, f64, f64)]| EcpShell {
        angular_momentum: l,
        terms: t
            .iter()
            .map(|&(n, z, d)| EcpTerm {
                coef: d,
                r_exp: n,
                gexp: z,
            })
            .collect(),
    };
    EcpDef {
        n_core: 46,
        shells: vec![
            ch(
                3,
                &[
                    (0, 1.0715702, -0.0747621),
                    (1, 44.1936028, -30.0811224),
                    (2, 12.9367609, -75.3722721),
                    (2, 3.1956412, -22.0563758),
                    (2, 0.8589806, -1.6979585),
                ],
            ),
            ch(
                0,
                &[
                    (0, 127.9202670, 2.9380036),
                    (1, 78.6211465, 41.2471267),
                    (2, 36.5146237, 287.8680095),
                    (2, 9.9065681, 114.3758506),
                    (2, 1.9420086, 37.6547714),
                ],
            ),
            ch(
                1,
                &[
                    (0, 13.0035304, 2.2222630),
                    (1, 76.0331404, 39.4090831),
                    (2, 24.1961684, 177.4075002),
                    (2, 6.4053433, 77.9889462),
                    (2, 1.5851786, 25.7547641),
                ],
            ),
            ch(
                2,
                &[
                    (0, 40.4278108, 7.0524360),
                    (1, 28.9084375, 33.3041635),
                    (2, 15.6268936, 186.9453875),
                    (2, 4.1442856, 71.9688361),
                    (2, 0.9377235, 9.3630657),
                ],
            ),
        ],
    }
}

/// H STO-3G + compact I s+p, LANL2DZ ECP on I (`pbc_kpoint_ecp.rs`).
pub fn hi_basis() -> BasisSet {
    let mut shells = HashMap::new();
    shells.insert(
        1,
        vec![shell(
            0,
            &[3.42525091, 0.62391373, 0.1688554],
            &[0.15432897, 0.53532814, 0.44463454],
        )],
    );
    shells.insert(
        53,
        vec![shell(0, &[0.4653], &[1.0]), shell(1, &[0.318], &[1.0])],
    );
    let mut ecps = HashMap::new();
    ecps.insert(53, lanl2dz_i_ecp());
    BasisSet {
        name: "HI-compact".into(),
        shells,
        ecps,
    }
}

/// Even-tempered s/p aux on H and I (`pbc_kpoint_ecp.rs`'s `hi_aux`): forces
/// and stress are exact derivatives of the FITTED energy whatever the aux.
pub fn hi_aux() -> BasisSet {
    let sh = |l: i32, a: f64| Shell {
        l,
        pure: false,
        exponents: vec![a],
        coefficients: vec![1.0],
    };
    let mut m = HashMap::new();
    m.insert(
        1,
        vec![
            sh(0, 4.7),
            sh(0, 1.9),
            sh(0, 0.75),
            sh(0, 0.3),
            sh(1, 1.25),
            sh(1, 0.5),
        ],
    );
    m.insert(
        53,
        vec![
            sh(0, 2.0),
            sh(0, 0.8),
            sh(0, 0.3),
            sh(1, 1.2),
            sh(1, 0.5),
            sh(1, 0.2),
        ],
    );
    BasisSet {
        name: "hi-aux".into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

fn atom(symbol: &str, z: i32, r: [f64; 3]) -> Atom {
    Atom {
        symbol: symbol.into(),
        z,
        x: r[0],
        y: r[1],
        zpos: r[2],
        ghost: false,
        n_core_ecp: 0,
    }
}

/// The species of a cell with `n` atoms: HI (2) or H2I (3).
fn species(n: usize) -> &'static [(&'static str, i32)] {
    match n {
        2 => &[("H", 1), ("I", 53)],
        3 => &[("H", 1), ("H", 1), ("I", 53)],
        _ => panic!("HI or H2I only"),
    }
}

/// A cell of `pos` (HI singlet / H2I doublet), repeated at each z copy of a
/// diag(1, 1, `copies`) supercell, through `apply_ecp`.
pub fn ecp_cell_copies(pos: &[[f64; 3]], copies: usize) -> Cell {
    let bs = hi_basis();
    let sp = species(pos.len());
    let mut atoms = Vec::new();
    for c in 0..copies {
        let tz = c as f64 * HI_A[2][2];
        for ((sym, z), r) in sp.iter().zip(pos) {
            atoms.push(atom(sym, *z, [r[0], r[1], r[2] + tz]));
        }
    }
    // H2I: one unpaired electron per copy (high spin, as the k-mesh state).
    let mut mol = Molecule {
        atoms,
        charge: 0,
        multiplicity: if pos.len() == 3 { copies + 1 } else { 1 },
    };
    mol.apply_ecp(&bs);
    let mut lat = HI_A;
    lat[2][2] *= copies as f64;
    Cell::new(mol, lat).expect("ECP cell")
}

pub fn ecp_cell(pos: &[[f64; 3]]) -> Cell {
    ecp_cell_copies(pos, 1)
}

pub fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

pub fn eps_at(i: usize, j: usize, h: f64) -> Mat3 {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    e
}

// ------------------------------------------------------------ routes

/// The k-point route under test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Rks(&'static str),
    Uks(&'static str),
    Roks(&'static str),
    Rohf,
}

impl Route {
    pub fn functional(self) -> Option<&'static str> {
        match self {
            Route::Rks(f) | Route::Uks(f) | Route::Roks(f) => Some(f),
            Route::Rohf => None,
        }
    }
    fn hybrid(self) -> bool {
        matches!(self.functional(), Some(f) if f.contains('0') || f.contains("B3"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Jk {
    Dense,
    Gdf,
}

/// The run settings: route, J/K, mesh, exxdiv and grid.
#[derive(Clone, Debug)]
pub struct Spec {
    pub route: Route,
    pub jk: Jk,
    pub mesh: [usize; 3],
    pub exxdiv: ExxDiv,
    pub grid: PeriodicGridConfig,
}

impl Spec {
    pub fn new(route: Route, jk: Jk, mesh: [usize; 3]) -> Self {
        let mut grid = PeriodicGridConfig::with_size(20, 50);
        grid.neighbour_cutoff = Some(CUTOFF);
        Spec {
            route,
            jk,
            mesh,
            exxdiv: if route.hybrid() || route == Route::Rohf {
                ExxDiv::Ewald
            } else {
                ExxDiv::None
            },
            grid,
        }
    }
}

pub fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig::with_omega(OMEGA)
}

pub fn gdf_cfg(exxdiv: ExxDiv) -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 1.0,
            exxdiv,
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        mutation: None,
    }
}

pub enum ScfOut {
    R(KRksResult),
    U(KUksResult),
    Ro(KRoksResult),
}

impl ScfOut {
    pub fn energy(&self) -> f64 {
        match self {
            ScfOut::R(r) => r.scf.energy,
            ScfOut::U(r) => r.scf.energy,
            ScfOut::Ro(r) => r.scf.energy,
        }
    }
    pub fn converged(&self) -> bool {
        match self {
            ScfOut::R(r) => r.scf.converged,
            ScfOut::U(r) => r.scf.converged,
            ScfOut::Ro(r) => r.scf.converged,
        }
    }
}

/// A converged k SCF and everything its derivatives need.
pub struct Fx {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub aux: PreparedBasis,
    pub mesh: KPointMesh,
    pub hk: PeriodicHcoreK,
    pub spec: Spec,
    pub scf: ScfOut,
}

fn rohf_cfg(cell: &Cell, s: &Spec) -> KRohfConfig {
    let mut c = KRohfConfig::for_cell(cell, s.exxdiv);
    c.hcore = hcore_cfg();
    c.scf.energy_conv = 1e-13;
    c.scf.grad_conv = 1e-10;
    if s.jk == Jk::Gdf {
        c.jk = KJkKind::RsGdf;
        c.rsgdf = gdf_cfg(s.exxdiv);
    }
    c
}

/// Fresh SCF on `cell` for `s`.
pub fn run_scf(cell: &Cell, s: &Spec) -> ScfOut {
    let bs = hi_basis();
    let prep = prep_for(cell, &bs);
    let aux = (s.jk == Jk::Gdf).then(|| prep_for(cell, &hi_aux()));
    let mesh = KPointMesh::gamma_centred(cell, s.mesh).unwrap();
    let out = match s.route {
        Route::Rks(f) => {
            let mut c = KRksConfig::new(cell, s.exxdiv, f);
            c.krhf.hcore = hcore_cfg();
            c.krhf.scf.energy_conv = 1e-13;
            c.krhf.scf.grad_conv = 1e-10;
            c.krhf.scf.max_iter = 300;
            if s.jk == Jk::Gdf {
                c.krhf.jk = KJkKind::RsGdf;
                c.krhf.rsgdf = gdf_cfg(s.exxdiv);
            }
            c.grid = s.grid.clone();
            ScfOut::R(solve_krks(cell, &prep, aux.as_ref(), &mesh, &c).expect("k-RKS"))
        }
        Route::Uks(f) => {
            let mut c = KUksConfig::new(cell, s.exxdiv, f);
            c.kuhf.hcore = hcore_cfg();
            c.kuhf.scf.energy_conv = 1e-13;
            c.kuhf.scf.grad_conv = 1e-10;
            c.kuhf.scf.max_iter = 600;
            if s.jk == Jk::Gdf {
                c.kuhf.jk = KJkKind::RsGdf;
                c.kuhf.rsgdf = gdf_cfg(s.exxdiv);
            }
            c.grid = s.grid.clone();
            ScfOut::U(solve_kuks(cell, &prep, aux.as_ref(), &mesh, &c).expect("k-UKS"))
        }
        Route::Roks(f) => {
            let mut c = KRoksConfig::new(cell, s.exxdiv, f);
            c.rohf = rohf_cfg(cell, s);
            c.grid = s.grid.clone();
            ScfOut::Ro(solve_kroks(cell, &prep, aux.as_ref(), &mesh, &c).expect("k-ROKS"))
        }
        Route::Rohf => ScfOut::Ro(
            solve_krohf(cell, &prep, aux.as_ref(), &mesh, &rohf_cfg(cell, s)).expect("k-ROHF"),
        ),
    };
    assert!(out.converged(), "{:?} SCF not converged", s.route);
    out
}

pub fn energy(pos: &[[f64; 3]], s: &Spec) -> f64 {
    run_scf(&ecp_cell(pos), s).energy()
}

pub fn energy_strained(cell: &Cell, eps: &Mat3, s: &Spec) -> f64 {
    run_scf(&cell.strained(eps).expect("strained"), s).energy()
}

pub fn fixture_cell(cell: Cell, s: &Spec) -> Fx {
    let bs = hi_basis();
    let prep = prep_for(&cell, &bs);
    let aux = prep_for(&cell, &hi_aux());
    let mesh = KPointMesh::gamma_centred(&cell, s.mesh).unwrap();
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcore_cfg()).unwrap();
    assert!(hk.n_ecp_triples > 0, "no ECP triples: the cell is vacuous");
    let scf = run_scf(&cell, s);
    Fx {
        cell,
        prep,
        aux,
        mesh,
        hk,
        spec: s.clone(),
        scf,
    }
}

pub fn fixture(pos: &[[f64; 3]], s: &Spec) -> Fx {
    fixture_cell(ecp_cell(pos), s)
}

/// Production-wired defects for the anchors: the HF-part (ECP) mutants of
/// the force and the stress (`cfg.grad.mutation` / `cfg.stress.mutation`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Mutants {
    pub force: Option<KGradMutation>,
    pub stress: Option<KStressMutation>,
}

fn rks_gcfg(s: &Spec, m: Option<KGradMutation>) -> KRksGradConfig {
    KRksGradConfig {
        grad: KGradConfig {
            budget_bytes: Some(AMPLE),
            mutation: m,
            ..Default::default()
        },
        grid: KXcGrid::Atom(s.grid.clone()),
        exxdiv: Some(s.exxdiv),
        ..Default::default()
    }
}

fn rks_scfg(s: &Spec, m: Option<KStressMutation>) -> KRksStressConfig {
    KRksStressConfig {
        stress: KStressConfig {
            budget_bytes: Some(AMPLE),
            mutation: m,
            ..Default::default()
        },
        grid: KXcGrid::Atom(s.grid.clone()),
        exxdiv: Some(s.exxdiv),
        ..Default::default()
    }
}

/// The J/K source the SCF used, handed to `f` as both the force and the
/// stress enum (built once).
fn with_jk<R>(fx: &Fx, f: impl FnOnce(KGradJk<'_>, KStressJk<'_>) -> R) -> R {
    match fx.spec.jk {
        Jk::Dense => {
            let eri = KDenseAftEri::build(
                &fx.cell,
                &fx.prep,
                &fx.mesh,
                &fx.hk.s,
                fx.spec.exxdiv,
                &Default::default(),
            )
            .unwrap();
            f(KGradJk::Dense(&eri), KStressJk::Dense(&eri))
        }
        Jk::Gdf => {
            let kc = gdf_cfg(fx.spec.exxdiv);
            let gdf = KRsGdf::build(&fx.cell, &fx.prep, &fx.aux, &fx.mesh, &fx.hk.s, &kc)
                .unwrap()
                .with_exxdiv(fx.spec.exxdiv);
            let src = || KRsGdfGradSource {
                gdf: &gdf,
                cfg: &kc,
                aux: &fx.aux,
            };
            f(KGradJk::RsGdf(src()), KStressJk::RsGdf(src()))
        }
    }
}

/// Analytic force (`natoms × 3`, dE/dR) and its ECP part.
pub fn force(fx: &Fx, m: Option<KGradMutation>) -> (Array2<f64>, Array2<f64>) {
    let s = &fx.spec;
    let h = hcore_cfg();
    let g: KRksGradient = with_jk(fx, |gj, _| {
        let c = rks_gcfg(s, m);
        let u = KUksGradConfig {
            base: c.clone(),
            spin_mutation: None,
        };
        match (&fx.scf, s.route) {
            (ScfOut::R(r), Route::Rks(f)) => {
                kpoint_rks_gradient(&fx.cell, &fx.prep, &fx.mesh, &h, &fx.hk, gj, &r.scf, f, &c)
            }
            (ScfOut::U(r), Route::Uks(f)) => {
                kpoint_uks_gradient(&fx.cell, &fx.prep, &fx.mesh, &h, &fx.hk, gj, &r.scf, f, &u)
            }
            (ScfOut::Ro(r), Route::Roks(f)) => {
                kpoint_roks_gradient(&fx.cell, &fx.prep, &fx.mesh, &h, &fx.hk, gj, &r.scf, f, &u)
            }
            (ScfOut::Ro(r), Route::Rohf) => kpoint_rohf_gradient(
                &fx.cell, &fx.prep, &fx.mesh, &h, &fx.hk, gj, &r.scf, s.exxdiv, &u,
            ),
            _ => unreachable!(),
        }
        .unwrap_or_else(|e| panic!("{:?} force: {e}", s.route))
    });
    (g.grad, g.hf.parts.ecp)
}

/// Analytic `dE/dε` and its ECP part.
pub fn stress(fx: &Fx, m: Option<KStressMutation>) -> (Mat3, Mat3) {
    let s = &fx.spec;
    let h = hcore_cfg();
    let st: KRksStress = with_jk(fx, |_, sj| {
        let c = rks_scfg(s, m);
        let u = KUksStressConfig {
            base: c.clone(),
            spin_mutation: None,
        };
        match (&fx.scf, s.route) {
            (ScfOut::R(r), Route::Rks(f)) => {
                kpoint_rks_stress(&fx.cell, &fx.prep, &fx.mesh, &h, &fx.hk, sj, &r.scf, f, &c)
            }
            (ScfOut::U(r), Route::Uks(f)) => {
                kpoint_uks_stress(&fx.cell, &fx.prep, &fx.mesh, &h, &fx.hk, sj, &r.scf, f, &u)
            }
            (ScfOut::Ro(r), Route::Roks(f)) => {
                kpoint_roks_stress(&fx.cell, &fx.prep, &fx.mesh, &h, &fx.hk, sj, &r.scf, f, &u)
            }
            (ScfOut::Ro(r), Route::Rohf) => kpoint_rohf_stress(
                &fx.cell, &fx.prep, &fx.mesh, &h, &fx.hk, sj, &r.scf, s.exxdiv, &u,
            ),
            _ => unreachable!(),
        }
        .unwrap_or_else(|e| panic!("{:?} stress: {e}", s.route))
    });
    (st.de_deps, st.hf.parts.ecp)
}

// ------------------------------------------------------------ FD

/// Central FD of the SCF energy along atom `a`, axis `x`.
pub fn fd_force(pos: &[[f64; 3]], s: &Spec, a: usize, x: usize, h: f64) -> f64 {
    (energy(&moved(pos, a, x, h), s) - energy(&moved(pos, a, x, -h), s)) / (2.0 * h)
}

/// Central FD of the SCF energy under strain `ε_ij`.
pub fn fd_strain(cell: &Cell, s: &Spec, i: usize, j: usize, h: f64) -> f64 {
    (energy_strained(cell, &eps_at(i, j, h), s) - energy_strained(cell, &eps_at(i, j, -h), s))
        / (2.0 * h)
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

pub fn amax(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}
