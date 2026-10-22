//! k-point RKS on an ALL-ELECTRON cell with a 1s core (LiH, PySCF STO-3G, no
//! ECP), Gamma-centred and shifted meshes, vs PySCF `KRKS` (issue #356
//! validation breadth). Every earlier k-KS pin is H-only or uses an ECP.
//!
//! Pins: `testdata/reference/pbc_kks_core_sto3g.json`
//! (generator `scripts/gen_pyscf_kks_core_refs.py`); PySCF and ferric use the
//! SAME uniform grid, so a small N is a valid pin.
//!
//! # Physics / artifact hypothesis (written before measuring)
//!
//! Physics: the Li 1s primitive (alpha = 16.1) is ~5x sharper than any H
//! function in the existing pins, so the dense-AFT J/K and the one-electron
//! pieces see a harder core; if the machinery is right ferric still matches
//! PySCF at SCF/grid noise (~1e-8 Ha). Artifact: a core-specific defect
//! (Li nuclear attraction / Ewald with Z = 3, the G-sphere truncating the 1s
//! pair density) would show as a gap that grows with the core sharpness
//! and is identical on every mesh; a mesh-handling defect (dropped shift,
//! Madelung) would instead differ between Gamma-centred and shifted rows.
//! Both would break the pin (only Gamma PBE is pinned: PySCF at the 80^3 J mesh
//! it needs costs ~32 min per row, so no shifted or PBE0 LiH row exists); the sensitivity checks below (`shift_is_not_dropped`,
//! `wrong_madelung_is_detected`) show these tests can fail.
//!
//! LIMIT: no all-electron cell heavier than Li is pinned (Mg STO-3G has
//! alpha ~ 300; its PySCF/ferric G-set comparison was not attempted), the
//! cell has 4 electrons, and the shifted-mesh Madelung convention is
//! validated only against PySCF's.

mod common;

use common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_core::mol::{Atom, Molecule};
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::PeriodicGrid;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdft::{solve_krks_on_grid, KRksConfig};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::lattice::Cell;
use std::collections::HashMap;

const OMEGA: f64 = 0.8;
const GRID_N: usize = 35;
const REF_JSON: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../testdata/reference/pbc_kks_core_sto3g.json"
);
const A: [[f64; 3]; 3] = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.4, 0.3, 6.0]];
const LI: [f64; 3] = [0.3, 0.2, 0.4];
const H: [f64; 3] = [0.3, 0.2, 3.4];
/// Measured max |ferric - PySCF| over the pins is recorded in the test
/// comment below; the bar is >= 10x that and far below the mutant scale.
const PIN_BAR: f64 = 1e-7; // measured 6.5e-9 (15x margin)
const MUTANT_MIN: f64 = 1e-3;

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

/// PySCF `sto-3g` for Li and H, digit for digit.
fn basis() -> BasisSet {
    let mut m = HashMap::new();
    m.insert(
        1,
        vec![shell(
            0,
            &[3.42525091, 0.62391373, 0.1688554],
            &[0.15432897, 0.53532814, 0.44463454],
        )],
    );
    let sp = [0.6362897, 0.1478601, 0.0480887];
    m.insert(
        3,
        vec![
            shell(
                0,
                &[16.119575, 2.9362007, 0.7946505],
                &[0.15432897, 0.53532814, 0.44463454],
            ),
            shell(0, &sp, &[-0.09996723, 0.39951283, 0.70011547]),
            shell(1, &sp, &[0.15591627, 0.60768372, 0.39195739]),
        ],
    );
    BasisSet {
        name: "pyscf-sto-3g-LiH".into(),
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

fn lih_cell() -> Cell {
    let mol = Molecule {
        atoms: vec![atom("Li", 3, LI), atom("H", 1, H)],
        charge: 0,
        multiplicity: 1,
    };
    Cell::new(mol, A).expect("LiH cell")
}

fn energy(cell: &Cell, mesh: &KPointMesh, f: &str, exxdiv: ExxDiv) -> f64 {
    let bs = basis();
    let prep = prep_for(cell, &bs);
    let grid = PeriodicGrid::uniform(cell, [GRID_N; 3]).unwrap();
    let mut cfg = KRksConfig::new(cell, exxdiv, f);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.krhf.scf.energy_conv = 1e-12;
    cfg.krhf.scf.grad_conv = 1e-9;
    let r = solve_krks_on_grid(cell, &prep, None, mesh, &grid, &cfg)
        .unwrap_or_else(|e| panic!("solve_krks {f}: {e}"));
    assert!(r.scf.converged, "{f} not converged");
    r.scf.energy
}

fn mesh_of(cell: &Cell, n: [usize; 3], gamma: bool) -> KPointMesh {
    if gamma {
        KPointMesh::gamma_centred(cell, n).unwrap()
    } else {
        KPointMesh::monkhorst_pack(cell, n).unwrap()
    }
}

fn rows() -> Vec<serde_json::Value> {
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(REF_JSON).unwrap()).unwrap();
    let cell = lih_cell();
    assert_eq!(
        j["cells"]["lih_core"]["mesh"].as_u64().unwrap() as usize,
        GRID_N
    );
    for (i, a) in cell.lattice().iter().enumerate() {
        for (d, x) in a.iter().enumerate() {
            assert!((j["cells"]["lih_core"]["a"][i][d].as_f64().unwrap() - x).abs() < 1e-12);
        }
    }
    j["rows"].as_array().unwrap().clone()
}

fn row_mesh(row: &serde_json::Value) -> [usize; 3] {
    let n: Vec<usize> = row["mesh"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    [n[0], n[1], n[2]]
}

/// One PySCF pin (Gamma, PBE). PySCF's AFT J mesh is 80^3 (its default 35^3
/// is 4.8e-3 Ha off, see the generator); the XC grid is the shared 35^3.
/// Measured ferric - PySCF: see `PIN_BAR`.
#[test]
#[ignore = "validation: PBC k-point RKS on an all-electron LiH cell vs PySCF KRKS; ~95 s release"]
fn core_electron_cell_krks_matches_pinned_pyscf() {
    let cell = lih_cell();
    let rows = rows();
    assert_eq!(rows.len(), 1, "reference rows");
    let row = &rows[0];
    let n = row_mesh(row);
    let e = energy(&cell, &mesh_of(&cell, n, true), "PBE", ExxDiv::Ewald);
    let e_ref = row["energy"].as_f64().unwrap();
    eprintln!(
        "LiH {n:?} PBE: {e:.12} PySCF {e_ref:.12} dE {:.2e}",
        e - e_ref
    );
    assert!((e - e_ref).abs() < PIN_BAR, "LiH PBE dE {:e}", e - e_ref);
}

/// Sensitivity check (no LiH pin covers a shifted mesh; compares ferric with
/// ferric): the shifted 1x1x2 mesh must differ from the Gamma-centred
/// one by far more than the pin bar (measured 3.3e-3 PBE).
#[test]
#[ignore = "validation: LiH shifted-vs-Gamma-centred mesh sensitivity check; ~95 s release"]
fn shift_is_not_dropped() {
    let cell = lih_cell();
    let g = energy(
        &cell,
        &mesh_of(&cell, [1, 1, 2], true),
        "PBE",
        ExxDiv::Ewald,
    );
    let s = energy(
        &cell,
        &mesh_of(&cell, [1, 1, 2], false),
        "PBE",
        ExxDiv::Ewald,
    );
    eprintln!("shifted - gamma-centred {:.3e}", s - g);
    assert!((s - g).abs() > MUTANT_MIN);
}

/// Sensitivity check (no LiH PBE0 pin; compares ferric with ferric): PBE0 with the exxdiv/Madelung term dropped moves the energy
/// by 2.6e-1 Ha, >> `MUTANT_MIN`, so the PBE0-capable path would not hide it.
#[test]
#[ignore = "validation: LiH dropped-exxdiv sensitivity check; ~95 s release"]
fn wrong_madelung_is_detected() {
    let cell = lih_cell();
    let mesh = mesh_of(&cell, [1, 1, 1], true);
    let a = energy(&cell, &mesh, "PBE0", ExxDiv::Ewald);
    let b = energy(&cell, &mesh, "PBE0", ExxDiv::None);
    eprintln!("ExxDiv::None - Ewald {:.3e}", b - a);
    assert!((b - a).abs() > MUTANT_MIN);
}
