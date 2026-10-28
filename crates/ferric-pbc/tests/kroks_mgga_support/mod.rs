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
