//! Shared setup of `pbc_kroks_mgga.rs` (per commit) and
//! `validation_pbc_kroks_mgga.rs` (weekly): the k-point ROKS SCAN / r2SCAN
//! energy (issue #492), its anchors, the PySCF `KROKS` pins and the defects
//! they must catch.
#![allow(dead_code)]

use crate::common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::PeriodicGrid;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdft::{solve_krks_on_grid, KRksConfig, KRksResult};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kroks::{solve_kroks_on_grid, KRohfMutation, KRoksConfig, KRoksResult};
use ferric_pbc::kuks::{solve_kuks_on_grid, KUksConfig, KUksMggaMutation, KUksResult};
use ferric_pbc::lattice::Cell;
use std::collections::HashMap;

pub const OMEGA_HCORE: f64 = 0.8;
/// The uniform XC grid of the PySCF pins (`UniformGrids` 32^3).
pub const PIN_GRID: usize = 32;
/// Three H in the a = 4 cube, doublet (N_alpha = 2, N_beta = 1): the cell of
/// `scripts/gen_pyscf_kroks_mgga_refs.py` (and of the k-UKS RS-GDF pins).
pub const H3: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [0.3, 1.9, 0.4]];
/// The first two atoms of [`H3`].
pub const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
pub const FUNCTIONALS: [&str; 2] = ["SCAN", "R2SCAN"];

/// `pos` in the a = 4 cube with multiplicity `mult`.
pub fn cube_cell(pos: &[[f64; 3]], mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(pos);
    mol.multiplicity = mult;
    Cell::new(mol, cubic(4.0)).expect("cell")
}

/// The pins' basis: one s per H, exponent 0.5.
pub fn s05() -> BasisSet {
    single_s_h(0.5)
}

/// Two uncontracted s per H (exponents 0.8, 0.3): a virtual for the fully
/// polarized anchor, with no tight primitive to enlarge the dense-AFT mesh.
pub fn two_s() -> BasisSet {
    let s = |a: f64| Shell {
        l: 0,
        pure: false,
        exponents: vec![a],
        coefficients: vec![1.0],
    };
    let mut m = HashMap::new();
    m.insert(1, vec![s(0.8), s(0.3)]);
    BasisSet {
        name: "two-s-H".into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

pub fn uniform_grid(cell: &Cell) -> PeriodicGrid {
    PeriodicGrid::uniform(cell, [PIN_GRID; 3]).expect("uniform grid")
}

pub fn kroks_cfg(cell: &Cell, f: &str) -> KRoksConfig {
    let mut cfg = KRoksConfig::new(cell, ExxDiv::Ewald, f);
    cfg.rohf.hcore = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);
    cfg.rohf.scf.energy_conv = 1e-13;
    cfg.rohf.scf.grad_conv = 1e-10;
    cfg.rohf.scf.max_iter = 300;
    cfg
}

pub fn kuks_cfg(cell: &Cell, f: &str) -> KUksConfig {
    let mut cfg = KUksConfig::new(cell, ExxDiv::Ewald, f);
    cfg.kuhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);
    cfg.kuhf.scf.energy_conv = 1e-13;
    cfg.kuhf.scf.grad_conv = 1e-10;
    cfg.kuhf.scf.max_iter = 300;
    cfg
}

pub fn krks_cfg(cell: &Cell, f: &str) -> KRksConfig {
    let mut cfg = KRksConfig::new(cell, ExxDiv::Ewald, f);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);
    cfg.krhf.scf.energy_conv = 1e-13;
    cfg.krhf.scf.grad_conv = 1e-10;
    cfg.krhf.scf.max_iter = 300;
    cfg
}

/// A deliberate defect of the k-ROKS meta-GGA (the shared tau evaluator, or
/// the ROKS coupling of the spin Focks).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Defect {
    Tau(KUksMggaMutation),
    Ro(KRohfMutation),
}

/// The defects of the ledger: `tau_beta` from `D_alpha`, tau dropped from the
/// kernel input, and two ROKS coupling defects (`V^xc_alpha` and `V^xc_beta`
/// swapped between the spin Focks; the bare alpha Fock diagonalised instead of
/// the Roothaan one).
pub const DEFECTS: [Defect; 4] = [
    Defect::Tau(KUksMggaMutation::TauBetaFromAlpha),
    Defect::Tau(KUksMggaMutation::DropTau),
    Defect::Ro(KRohfMutation::XcSpinSwap),
    Defect::Ro(KRohfMutation::BareAlphaFock),
];

/// k-ROKS on the dense-AFT J/K and `grid` (a non-converged SCF is an error).
pub fn try_kroks(
    cell: &Cell,
    bs: &BasisSet,
    n: [usize; 3],
    f: &str,
    grid: &PeriodicGrid,
    defect: Option<Defect>,
) -> Result<KRoksResult, FerricError> {
    let prep = prep_for(cell, bs);
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let mut cfg = kroks_cfg(cell, f);
    match defect {
        Some(Defect::Tau(m)) => cfg.mgga_mutation = Some(m),
        Some(Defect::Ro(m)) => cfg.rohf.mutation = Some(m),
        None => {}
    }
    solve_kroks_on_grid(cell, &prep, None, &mesh, grid, &cfg)
}

pub fn kroks(
    cell: &Cell,
    bs: &BasisSet,
    n: [usize; 3],
    f: &str,
    grid: &PeriodicGrid,
) -> KRoksResult {
    let r = try_kroks(cell, bs, n, f, grid, None)
        .unwrap_or_else(|e| panic!("solve_kroks {f} {n:?}: {e}"));
    assert!(r.scf.converged, "{f} {n:?}");
    r
}

pub fn kuks(cell: &Cell, bs: &BasisSet, n: [usize; 3], f: &str, grid: &PeriodicGrid) -> KUksResult {
    let prep = prep_for(cell, bs);
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    solve_kuks_on_grid(cell, &prep, None, &mesh, grid, &kuks_cfg(cell, f))
        .unwrap_or_else(|e| panic!("solve_kuks {f} {n:?}: {e}"))
}

pub fn krks(cell: &Cell, bs: &BasisSet, n: [usize; 3], f: &str, grid: &PeriodicGrid) -> KRksResult {
    let prep = prep_for(cell, bs);
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    solve_krks_on_grid(cell, &prep, None, &mesh, grid, &krks_cfg(cell, f))
        .unwrap_or_else(|e| panic!("solve_krks {f} {n:?}: {e}"))
}

/// `|E(defect) - pin|` of the H3 pin cell, or `INFINITY` when the defective
/// SCF does not converge (the run cannot reproduce the pin either way).
pub fn defect_miss(f: &str, n: [usize; 3], d: Defect, pin: f64) -> f64 {
    let cell = cube_cell(&H3, 2);
    match try_kroks(&cell, &s05(), n, f, &uniform_grid(&cell), Some(d)) {
        Ok(r) => (r.scf.energy - pin).abs(),
        Err(e) => {
            eprintln!("{f} {n:?} {d:?}: SCF failed ({e})");
            f64::INFINITY
        }
    }
}

/// Pin key, e.g. `"SCAN 1x1x3"`.
pub fn key(f: &str, n: [usize; 3]) -> String {
    format!("{f} {}x{}x{}", n[0], n[1], n[2])
}

/// The PySCF 2.13.1 `KROKS` energy pinned for `key` by
/// `scripts/gen_pyscf_kroks_mgga_refs.py` (`table = "e_tot"`), or the `KUKS`
/// sanity row of the same setup (`table = "e_tot_kuks"`).
pub fn pyscf_pin(table: &str, key: &str) -> f64 {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/reference/pbc_h3_s05_kroks_mgga.json"
    );
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("reference json")).unwrap();
    assert_eq!(doc["pyscf_version"], "2.13.1");
    doc[table][key].as_f64().expect("pinned energy")
}

// ------------------------------------------------- force and stress (#492, item 3)

use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::kdense_aft::KDenseAftEri;
use ferric_pbc::kgrad::{KGradConfig, KGradJk};
use ferric_pbc::kroks::KRoScfResult;
use ferric_pbc::kscf::KScfResult;
use ferric_pbc::kstress::{KStressConfig, KStressJk};
use ferric_pbc::kuscf::KUScfResult;
use ferric_pbc::kuxcgrad::{
    kpoint_roks_gradient, kpoint_uks_gradient, KUSpinMutation, KUksGradConfig, KUksGradient,
};
use ferric_pbc::kuxcstress::{
    kpoint_roks_stress, kpoint_uks_stress, KUStressMutation, KUksStress, KUksStressConfig,
};
use ferric_pbc::kxcgrad::{kpoint_rks_gradient, KRksGradConfig, KXcGrid, KXcMutation};
use ferric_pbc::kxcstress::{kpoint_rks_stress, KRksStressConfig, KXcStressMutation};
use ferric_pbc::stress::Mat3;
use ndarray::Array2;

/// Ample derivative budget (bytes).
pub const AMPLE: usize = 1 << 31;

/// The integrals a derivative needs on one cell and mesh: the hcore and the
/// dense-AFT J/K the SCF of [`try_kroks`] builds, and the pins' uniform grid.
pub struct DerivFx {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub mesh: KPointMesh,
    pub hcfg: PeriodicHcoreConfig,
    pub hk: PeriodicHcoreK,
    pub eri: KDenseAftEri,
    pub grid: KXcGrid,
}

pub fn deriv_fx(cell: &Cell, bs: &BasisSet, n: [usize; 3]) -> DerivFx {
    let prep = prep_for(cell, bs);
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);
    let hk = periodic_hcore_kpts(cell, &prep, &mesh, &hcfg).unwrap();
    let eri = KDenseAftEri::build(
        cell,
        &prep,
        &mesh,
        &hk.s,
        ExxDiv::Ewald,
        &Default::default(),
    )
    .unwrap();
    DerivFx {
        cell: cell.clone(),
        prep,
        mesh,
        hcfg,
        hk,
        eri,
        grid: KXcGrid::Uniform([PIN_GRID; 3]),
    }
}

pub fn gcfg(fx: &DerivFx, xm: Option<KXcMutation>, sm: Option<KUSpinMutation>) -> KUksGradConfig {
    KUksGradConfig {
        base: rks_gcfg(fx, xm),
        spin_mutation: sm,
    }
}

pub fn rks_gcfg(fx: &DerivFx, xm: Option<KXcMutation>) -> KRksGradConfig {
    KRksGradConfig {
        grad: KGradConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        grid: fx.grid.clone(),
        mutation: xm,
        ..Default::default()
    }
}

pub fn scfg(
    fx: &DerivFx,
    xm: Option<KXcStressMutation>,
    sm: Option<KUStressMutation>,
) -> KUksStressConfig {
    KUksStressConfig {
        base: rks_scfg(fx, xm),
        spin_mutation: sm,
    }
}

pub fn rks_scfg(fx: &DerivFx, xm: Option<KXcStressMutation>) -> KRksStressConfig {
    KRksStressConfig {
        stress: KStressConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        grid: fx.grid.clone(),
        mutation: xm,
        ..Default::default()
    }
}

pub fn roks_force(
    fx: &DerivFx,
    scf: &KRoScfResult,
    f: &str,
    cfg: &KUksGradConfig,
) -> Result<KUksGradient, FerricError> {
    kpoint_roks_gradient(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &fx.hcfg,
        &fx.hk,
        KGradJk::Dense(&fx.eri),
        scf,
        f,
        cfg,
    )
}

pub fn roks_stress(
    fx: &DerivFx,
    scf: &KRoScfResult,
    f: &str,
    cfg: &KUksStressConfig,
) -> Result<KUksStress, FerricError> {
    kpoint_roks_stress(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &fx.hcfg,
        &fx.hk,
        KStressJk::Dense(&fx.eri),
        scf,
        f,
        cfg,
    )
}

/// The k-UKS force and stress of an independent k-UKS SCF (the fully
/// polarized anchor's oracle).
pub fn uks_derivs(fx: &DerivFx, scf: &KUScfResult, f: &str) -> (KUksGradient, KUksStress) {
    let g = kpoint_uks_gradient(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &fx.hcfg,
        &fx.hk,
        KGradJk::Dense(&fx.eri),
        scf,
        f,
        &gcfg(fx, None, None),
    )
    .unwrap_or_else(|e| panic!("kpoint_uks_gradient {f}: {e}"));
    let s = kpoint_uks_stress(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &fx.hcfg,
        &fx.hk,
        KStressJk::Dense(&fx.eri),
        scf,
        f,
        &scfg(fx, None, None),
    )
    .unwrap_or_else(|e| panic!("kpoint_uks_stress {f}: {e}"));
    (g, s)
}

/// The k-RKS force and stress of an independent closed-shell SCF (the
/// singlet anchor's oracle).
pub fn rks_derivs(fx: &DerivFx, scf: &KScfResult, f: &str) -> (KUksGradient, KUksStress) {
    let g = kpoint_rks_gradient(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &fx.hcfg,
        &fx.hk,
        KGradJk::Dense(&fx.eri),
        scf,
        f,
        &rks_gcfg(fx, None),
    )
    .unwrap_or_else(|e| panic!("kpoint_rks_gradient {f}: {e}"));
    let s = kpoint_rks_stress(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &fx.hcfg,
        &fx.hk,
        KStressJk::Dense(&fx.eri),
        scf,
        f,
        &rks_scfg(fx, None),
    )
    .unwrap_or_else(|e| panic!("kpoint_rks_stress {f}: {e}"));
    (g, s)
}

/// Per-part `(label, max |a - b|)` rows of two force/stress pairs.
pub fn part_rows(
    a: &(KUksGradient, KUksStress),
    b: &(KUksGradient, KUksStress),
) -> Vec<(&'static str, f64)> {
    vec![
        ("force xc_ao", diff(&a.0.xc_ao, &b.0.xc_ao)),
        ("force xc_point", diff(&a.0.xc_point, &b.0.xc_point)),
        ("force xc_weight", diff(&a.0.xc_weight, &b.0.xc_weight)),
        ("force hf", diff(&a.0.hf.grad, &b.0.hf.grad)),
        ("force total", diff(&a.0.grad, &b.0.grad)),
        ("stress xc_ao", mdiff(&a.1.xc_ao, &b.1.xc_ao)),
        ("stress xc_weight", mdiff(&a.1.xc_weight, &b.1.xc_weight)),
        ("stress de_deps", mdiff(&a.1.de_deps, &b.1.de_deps)),
    ]
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

pub fn mmax(a: &Mat3) -> f64 {
    a.iter().flatten().fold(0.0_f64, |m, x| m.max(x.abs()))
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

/// Central FD of the converged k-ROKS energy (H3 doublet, s05, the pins'
/// uniform grid rebuilt per geometry) under atom displacement.
pub fn fd_roks_force(
    pos: &[[f64; 3]],
    n: [usize; 3],
    f: &str,
    h: f64,
    comps: &[(usize, usize)],
) -> Vec<f64> {
    let e = |p: &[[f64; 3]]| {
        let c = cube_cell(p, 2);
        kroks(&c, &s05(), n, f, &uniform_grid(&c)).scf.energy
    };
    comps
        .iter()
        .map(|&(a, x)| (e(&moved(pos, a, x, h)) - e(&moved(pos, a, x, -h))) / (2.0 * h))
        .collect()
}

/// Central FD of the converged k-ROKS energy under cell strain (`dE/dε`).
pub fn fd_roks_stress(
    cell: &Cell,
    n: [usize; 3],
    f: &str,
    h: f64,
    comps: &[(usize, usize)],
) -> Vec<f64> {
    let e = |c: &Cell| kroks(c, &s05(), n, f, &uniform_grid(c)).scf.energy;
    comps
        .iter()
        .map(|&(i, j)| (e(&strained(cell, i, j, h)) - e(&strained(cell, i, j, -h))) / (2.0 * h))
        .collect()
}

/// `max |F[a, x] - fd|` over `comps`.
pub fn worst_force(g: &Array2<f64>, fd: &[f64], comps: &[(usize, usize)]) -> f64 {
    comps
        .iter()
        .zip(fd)
        .fold(0.0_f64, |m, (&(a, x), d)| m.max((g[(a, x)] - d).abs()))
}

/// `max |dE/dε[i][j] - fd|` over `comps`.
pub fn worst_stress(s: &Mat3, fd: &[f64], comps: &[(usize, usize)]) -> f64 {
    comps
        .iter()
        .zip(fd)
        .fold(0.0_f64, |m, (&(i, j), d)| m.max((s[i][j] - d).abs()))
}

/// `Σ_k Σ |D_α − D_β|`: proof that a state is genuinely polarized.
pub fn spin_asymmetry(scf: &KRoScfResult) -> f64 {
    scf.density_alpha
        .iter()
        .zip(&scf.density_beta)
        .map(|(a, b)| (a - b).iter().map(|z| z.norm()).sum::<f64>())
        .sum()
}
