//! Shared setup of `pbc_kuks_mgga_rsgdf.rs` (per commit) and
//! `validation_pbc_kuks_mgga_rsgdf.rs` (weekly): the open-shell k-point
//! meta-GGA on the RS-GDF J/K (issue #492). Energies on the exact pair-midpoint
//! aux, force / stress fixtures on an atom-centred aux, and the SCF-energy
//! finite differences.
#![allow(dead_code)]

use crate::common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::site_basis::SiteBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::{PeriodicGrid, PeriodicGridConfig};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::KDenseAftEri;
use ferric_pbc::kdft::{solve_krks, solve_krks_on_grid, KRksConfig, KRksResult};
use ferric_pbc::kgrad::{KGradConfig, KGradJk, KRsGdfGradSource};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::KJkKind;
use ferric_pbc::kstress::{KStressConfig, KStressJk};
use ferric_pbc::kuks::{solve_kuks, solve_kuks_on_grid, KUksConfig, KUksMggaMutation, KUksResult};
use ferric_pbc::kuxcgrad::{kpoint_uks_gradient, KUSpinMutation, KUksGradConfig, KUksGradient};
use ferric_pbc::kuxcstress::{kpoint_uks_stress, KUStressMutation, KUksStress, KUksStressConfig};
use ferric_pbc::kxcgrad::{
    kpoint_rks_gradient, KRksGradConfig, KRksGradient, KXcGrid, KXcMutation,
};
use ferric_pbc::kxcstress::{kpoint_rks_stress, KRksStress, KRksStressConfig, KXcStressMutation};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use ferric_pbc::stress::Mat3;
use ndarray::Array2;
use std::collections::HashMap;

pub const OMEGA_HCORE: f64 = 0.8;
pub const ANCHOR_ALPHA: f64 = 0.5;
pub const AMPLE: usize = 1 << 31;
/// The uniform grid of the exact-aux energy pins (PySCF `UniformGrids` 32^3).
pub const PIN_GRID: usize = 32;
/// Explicit neighbour cutoff of the atom grid (the default depends on the
/// cell and puts one-step discontinuities into the energy that FD would see).
pub const CUTOFF: f64 = 14.0;
/// H2 singlet (the closed-shell limit).
pub const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
/// Three H in the a = 4 cube, doublet (N_alpha = 2, N_beta = 1): the polarized
/// cell of the dense-AFT k-UKS meta-GGA pins.
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

/// Which J/K a fixture runs on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Jk {
    Dense,
    Gdf,
}

/// `pos` in the a = 4 cube with multiplicity `mult`.
pub fn cube_cell(pos: &[[f64; 3]], mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(pos);
    mol.multiplicity = mult;
    Cell::new(mol, cubic(4.0)).expect("cell")
}

/// Singlet for an even number of H, doublet for an odd one.
pub fn h_cell(pos: &[[f64; 3]]) -> Cell {
    cube_cell(pos, 1 + pos.len() % 2)
}

/// Even-tempered s + p aux on H (the RS-GDF force / stress aux of
/// `pbc_kxcgrad_mgga.rs` / `pbc_kxcstress_mgga.rs`).
pub fn et_sp_aux() -> BasisSet {
    let shell = |l, a| Shell {
        l,
        pure: false,
        exponents: vec![a],
        coefficients: vec![1.0],
    };
    let mut shells: Vec<Shell> = [4.7, 1.9, 0.75, 0.3].map(|a| shell(0, a)).to_vec();
    shells.extend([1.25, 0.5].map(|a| shell(1, a)));
    let mut m = HashMap::new();
    m.insert(1, shells);
    BasisSet {
        name: "et-sp-aux-H".into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

/// Aux s functions (exponent `2 alpha`) at the midpoints of every atom pair
/// `(i <= j)` shifted by the half-lattice `classes` (all 8 = every pair
/// density of a one-s-per-atom basis, so the fit is exact; the construction
/// of `pbc_krsgdf.rs` / `pbc_krks_mgga_rsgdf.rs` extended to any atom count).
pub fn pair_midpoint_sites(cell: &Cell, classes: &[usize]) -> Vec<[f64; 4]> {
    let r = cell.positions();
    let a = cell.lattice();
    let mut out = Vec::new();
    for i in 0..r.len() {
        for j in i..r.len() {
            for k in classes {
                let h = [(k >> 2) & 1, (k >> 1) & 1, k & 1];
                let mut c = [0.0; 3];
                for (d, cd) in c.iter_mut().enumerate() {
                    let ha: f64 = (0..3).map(|x| h[x] as f64 * a[x][d]).sum();
                    *cd = 0.5 * (r[i][d] + r[j][d] + ha);
                }
                out.push([c[0], c[1], c[2], 2.0 * ANCHOR_ALPHA]);
            }
        }
    }
    out
}

/// The exact aux of [`pair_midpoint_sites`] (all 8 classes), or the on-site
/// class 0 alone (`exact = false`: an incomplete fit, the negative control).
pub fn midpoint_aux(cell: &Cell, exact: bool) -> PreparedBasis {
    let classes: Vec<usize> = if exact { (0..8).collect() } else { vec![0] };
    SiteBasis::new(&pair_midpoint_sites(cell, &classes), 0)
        .unwrap()
        .prep
}

pub fn rsgdf_cfg() -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 0.8,
            exxdiv: ExxDiv::Ewald,
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        mutation: None,
    }
}

pub fn kuks_cfg(cell: &Cell, f: &str, jk: Jk, grid: &KXcGrid) -> KUksConfig {
    let mut cfg = KUksConfig::new(cell, ExxDiv::Ewald, f);
    cfg.kuhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);
    cfg.kuhf.scf.energy_conv = 1e-13;
    cfg.kuhf.scf.grad_conv = 1e-10;
    cfg.kuhf.scf.max_iter = 300;
    if jk == Jk::Gdf {
        cfg.kuhf.jk = KJkKind::RsGdf;
        cfg.kuhf.rsgdf = rsgdf_cfg();
    }
    if let KXcGrid::Atom(g) = grid {
        cfg.grid = g.clone();
    }
    cfg
}

pub fn krks_cfg(cell: &Cell, f: &str, jk: Jk, grid: &KXcGrid) -> KRksConfig {
    let mut cfg = KRksConfig::new(cell, ExxDiv::Ewald, f);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);
    cfg.krhf.scf.energy_conv = 1e-13;
    cfg.krhf.scf.grad_conv = 1e-10;
    cfg.krhf.scf.max_iter = 300;
    if jk == Jk::Gdf {
        cfg.krhf.jk = KJkKind::RsGdf;
        cfg.krhf.rsgdf = rsgdf_cfg();
    }
    if let KXcGrid::Atom(g) = grid {
        cfg.grid = g.clone();
    }
    cfg
}

pub fn atom_grid() -> KXcGrid {
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(CUTOFF);
    KXcGrid::Atom(g)
}

/// Inputs of one k-UKS SCF.
pub struct Run<'a> {
    pub bs: &'a BasisSet,
    pub aux: Option<&'a PreparedBasis>,
    pub n: [usize; 3],
    pub f: &'a str,
    pub grid: &'a KXcGrid,
    pub mutation: Option<KUksMggaMutation>,
}

/// Converged k-UKS energy of `run` on `cell` (`aux = None`: dense-AFT J/K).
pub fn kuks(cell: &Cell, run: &Run<'_>) -> KUksResult {
    let prep = prep_for(cell, run.bs);
    let mesh = KPointMesh::gamma_centred(cell, run.n).unwrap();
    let jk = if run.aux.is_some() {
        Jk::Gdf
    } else {
        Jk::Dense
    };
    let mut cfg = kuks_cfg(cell, run.f, jk, run.grid);
    cfg.mgga_mutation = run.mutation;
    let r = match run.grid {
        KXcGrid::Atom(_) => solve_kuks(cell, &prep, run.aux, &mesh, &cfg),
        KXcGrid::Uniform(u) => {
            let g = PeriodicGrid::uniform(cell, *u).unwrap();
            solve_kuks_on_grid(cell, &prep, run.aux, &mesh, &g, &cfg)
        }
    }
    .unwrap_or_else(|e| panic!("solve_kuks {} {:?}: {e}", run.f, run.n));
    assert!(r.scf.converged, "{} {:?} not converged", run.f, run.n);
    r
}

fn krks(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: Option<&PreparedBasis>,
    mesh: &KPointMesh,
    f: &str,
    grid: &KXcGrid,
) -> KRksResult {
    let jk = if aux.is_some() { Jk::Gdf } else { Jk::Dense };
    let cfg = krks_cfg(cell, f, jk, grid);
    match grid {
        KXcGrid::Atom(_) => solve_krks(cell, prep, aux, mesh, &cfg),
        KXcGrid::Uniform(u) => {
            let g = PeriodicGrid::uniform(cell, *u).unwrap();
            solve_krks_on_grid(cell, prep, aux, mesh, &g, &cfg)
        }
    }
    .unwrap_or_else(|e| panic!("solve_krks {f}: {e}"))
}

/// A converged k-UKS state (STO-3G H, et-sp aux) with everything its force and
/// stress need, on the dense-AFT or the RS-GDF J/K.
pub struct Fx {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub aux: PreparedBasis,
    pub mesh: KPointMesh,
    pub n: [usize; 3],
    pub hk: PeriodicHcoreK,
    pub res: KUksResult,
    pub grid: KXcGrid,
    pub f: String,
    pub jk: Jk,
}

pub fn fixture(cell: Cell, n: [usize; 3], f: &str, jk: Jk, grid: KXcGrid) -> Fx {
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let aux = prep_for(&cell, &et_sp_aux());
    let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
    let res = kuks(
        &cell,
        &Run {
            bs: &bs,
            aux: (jk == Jk::Gdf).then_some(&aux),
            n,
            f,
            grid: &grid,
            mutation: None,
        },
    );
    let hk = periodic_hcore_kpts(
        &cell,
        &prep,
        &mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA_HCORE),
    )
    .unwrap();
    Fx {
        cell,
        prep,
        aux,
        mesh,
        n,
        hk,
        res,
        grid,
        f: f.into(),
        jk,
    }
}

/// The SCF energy of the fixture's setup on a displaced / strained `cell`.
pub fn energy_like(fx: &Fx, cell: &Cell) -> f64 {
    let aux = prep_for(cell, &et_sp_aux());
    kuks(
        cell,
        &Run {
            bs: &pyscf_sto3g_h(),
            aux: (fx.jk == Jk::Gdf).then_some(&aux),
            n: fx.n,
            f: &fx.f,
            grid: &fx.grid,
            mutation: None,
        },
    )
    .scf
    .energy
}

/// Run `body` with the fixture's J/K as both a gradient and a stress source.
fn with_jk<T>(fx: &Fx, body: impl FnOnce(KGradJk<'_>, KStressJk<'_>) -> T) -> T {
    match fx.jk {
        Jk::Dense => {
            let eri = KDenseAftEri::build(
                &fx.cell,
                &fx.prep,
                &fx.mesh,
                &fx.hk.s,
                ExxDiv::Ewald,
                &Default::default(),
            )
            .unwrap();
            body(KGradJk::Dense(&eri), KStressJk::Dense(&eri))
        }
        Jk::Gdf => {
            let kc = rsgdf_cfg();
            let gdf = KRsGdf::build(&fx.cell, &fx.prep, &fx.aux, &fx.mesh, &fx.hk.s, &kc)
                .unwrap()
                .with_exxdiv(ExxDiv::Ewald);
            let src = || KRsGdfGradSource {
                gdf: &gdf,
                cfg: &kc,
                aux: &fx.aux,
            };
            body(KGradJk::RsGdf(src()), KStressJk::RsGdf(src()))
        }
    }
}

pub fn try_force(
    fx: &Fx,
    xm: Option<KXcMutation>,
    sm: Option<KUSpinMutation>,
) -> Result<KUksGradient, FerricError> {
    let cfg = KUksGradConfig {
        base: KRksGradConfig {
            grad: KGradConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
            grid: fx.grid.clone(),
            mutation: xm,
            ..Default::default()
        },
        spin_mutation: sm,
    };
    let h = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);
    with_jk(fx, |g, _| {
        kpoint_uks_gradient(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &h,
            &fx.hk,
            g,
            &fx.res.scf,
            &fx.f,
            &cfg,
        )
    })
}

pub fn force(fx: &Fx, xm: Option<KXcMutation>, sm: Option<KUSpinMutation>) -> KUksGradient {
    try_force(fx, xm, sm).unwrap_or_else(|e| panic!("kpoint_uks_gradient {xm:?} {sm:?}: {e}"))
}

pub fn try_stress(
    fx: &Fx,
    xm: Option<KXcStressMutation>,
    sm: Option<KUStressMutation>,
) -> Result<KUksStress, FerricError> {
    let cfg = KUksStressConfig {
        base: KRksStressConfig {
            stress: KStressConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
            grid: fx.grid.clone(),
            mutation: xm,
            ..Default::default()
        },
        spin_mutation: sm,
    };
    let h = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);
    with_jk(fx, |_, s| {
        kpoint_uks_stress(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &h,
            &fx.hk,
            s,
            &fx.res.scf,
            &fx.f,
            &cfg,
        )
    })
}

pub fn stress(fx: &Fx, xm: Option<KXcStressMutation>, sm: Option<KUStressMutation>) -> KUksStress {
    try_stress(fx, xm, sm).unwrap_or_else(|e| panic!("kpoint_uks_stress {xm:?} {sm:?}: {e}"))
}

/// The closed-shell twin of a singlet fixture: energy, force and stress of an
/// independent k-RKS SCF on the same cell, mesh, grid and J/K.
pub fn rks_twin(fx: &Fx) -> (f64, KRksGradient, KRksStress) {
    let aux = (fx.jk == Jk::Gdf).then_some(&fx.aux);
    let r = krks(&fx.cell, &fx.prep, aux, &fx.mesh, &fx.f, &fx.grid);
    let h = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);
    let gcfg = KRksGradConfig {
        grad: KGradConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        grid: fx.grid.clone(),
        ..Default::default()
    };
    let scfg = KRksStressConfig {
        stress: KStressConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        grid: fx.grid.clone(),
        ..Default::default()
    };
    let (g, s) = with_jk(fx, |gj, sj| {
        (
            kpoint_rks_gradient(
                &fx.cell, &fx.prep, &fx.mesh, &h, &fx.hk, gj, &r.scf, &fx.f, &gcfg,
            ),
            kpoint_rks_stress(
                &fx.cell, &fx.prep, &fx.mesh, &h, &fx.hk, sj, &r.scf, &fx.f, &scfg,
            ),
        )
    });
    (r.scf.energy, g.unwrap(), s.unwrap())
}

// ------------------------------------------------------------ small helpers

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

pub fn msub(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = *a;
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] -= b[i][j];
        }
    }
    o
}

pub fn mmax(a: &Mat3) -> f64 {
    a.iter().flatten().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// Central FD of the converged SCF energy (the fixture's setup) under atom
/// displacement of `pos` (cell `mk(pos)`), per listed `(atom, axis)`.
pub fn fd_scf_force(
    fx: &Fx,
    mk: &dyn Fn(&[[f64; 3]]) -> Cell,
    pos: &[[f64; 3]],
    h: f64,
    comps: &[(usize, usize)],
) -> Vec<f64> {
    comps
        .iter()
        .map(|&(a, x)| {
            let ep = energy_like(fx, &mk(&moved(pos, a, x, h)));
            let em = energy_like(fx, &mk(&moved(pos, a, x, -h)));
            (ep - em) / (2.0 * h)
        })
        .collect()
}

/// Central FD of the converged SCF energy under cell strain (`dE/dε`).
pub fn fd_scf_stress(fx: &Fx, h: f64, comps: &[(usize, usize)]) -> Vec<f64> {
    comps
        .iter()
        .map(|&(i, j)| {
            let ep = energy_like(fx, &strained(&fx.cell, i, j, h));
            let em = energy_like(fx, &strained(&fx.cell, i, j, -h));
            (ep - em) / (2.0 * h)
        })
        .collect()
}

/// Largest `|g[c] - fd[c]|` over the listed force components.
pub fn force_miss(g: &Array2<f64>, fd: &[f64], comps: &[(usize, usize)]) -> f64 {
    comps
        .iter()
        .zip(fd)
        .fold(0.0_f64, |m, (&(a, x), f)| m.max((g[(a, x)] - f).abs()))
}

/// Largest `|s[c] - fd[c]|` over the listed strain components.
pub fn stress_miss(s: &Mat3, fd: &[f64], comps: &[(usize, usize)]) -> f64 {
    comps
        .iter()
        .zip(fd)
        .fold(0.0_f64, |m, (&(i, j), f)| m.max((s[i][j] - f).abs()))
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

/// The XC- and Pulay-side spin / tau force defects that can only show on a
/// polarized state (names for the ledger).
pub fn force_mutants() -> Vec<(String, Option<KXcMutation>, Option<KUSpinMutation>)> {
    use KUSpinMutation as U;
    let xm = [KXcMutation::TauNoAo, KXcMutation::TauDoubled];
    let sm = [
        U::TauSpinSwap,
        U::TauDropBeta,
        U::TauTotalBothChannels,
        U::TauNoVxcBeta,
        U::PulayJOneSpin,
    ];
    xm.iter()
        .map(|m| (format!("{m:?}"), Some(*m), None))
        .chain(sm.iter().map(|m| (format!("{m:?}"), None, Some(*m))))
        .collect()
}

/// The stress counterparts of [`force_mutants`].
pub fn stress_mutants() -> Vec<(String, Option<KXcStressMutation>, Option<KUStressMutation>)> {
    use KUStressMutation as U;
    let xm = [
        KXcStressMutation::NoTauStrain,
        KXcStressMutation::TauDoubled,
    ];
    let sm = [
        U::TauSpinSwap,
        U::TauDropBeta,
        U::TauTotalBothChannels,
        U::TauNoVxcBeta,
        U::PulayJOneSpin,
    ];
    xm.iter()
        .map(|m| (format!("{m:?}"), Some(*m), None))
        .chain(sm.iter().map(|m| (format!("{m:?}"), None, Some(*m))))
        .collect()
}

/// The PySCF 2.13.1 `KUKS` energy pinned for `key` ("SCAN 1x1x3", ...) by
/// `scripts/gen_pyscf_kuks_mgga_rsgdf_refs.py`.
pub fn pyscf_pin(key: &str) -> f64 {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/reference/pbc_h3_s05_kuks_mgga_rsgdf.json"
    );
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("reference json")).unwrap();
    assert_eq!(doc["pyscf_version"], "2.13.1");
    doc["e_tot"][key].as_f64().expect("pinned energy")
}
