//! QM-MM Lennard-Jones across a covalent QM/MM cut follows the force field's
//! own nonbonded rules: 1-2 and 1-3 pairs excluded, 1-4 pairs scaled by the
//! topology's `scale_lj_14` (issue #319).
//!
//! Test order: (1) exactness anchor — with no bond path between the regions
//! `qmmm_mm_terms` is BIT-IDENTICAL to the unscaled every-pair sum; (2) one
//! test per pair class on a chain with realistic carbon LJ, each pair
//! isolated by giving epsilon to only that QM/MM pair; (3) the independent
//! reference: OpenMM's own `createExceptionsFromBonds` on ethanol cut at C-C,
//! QM-MM LJ isolated by per-region epsilon switching
//! (`scripts/gen_openmm_qmmm_lj_refs.py`); (4) the MM-part gradient vs a
//! central finite difference across the cut with realistic LJ.

use ferric_mm::{qm_mm_lj_energy_gradient, Bond, LjParams, MmTopology};
use ferric_scf::qmmm::{qmmm_mm_terms, QmSelection, QmmmAtom, QmmmSystem, DEFAULT_LINK_SCALE};
use ndarray::Array2;
use serde::Deserialize;
use std::path::PathBuf;

const ANG2BOHR: f64 = 1.0 / 0.529_177_210_92;
const KCAL: f64 = 1.0 / 627.509_474;

// Bars: >= 10x the worst MEASURED value (beside it; the OpenMM agreement is at
// the last bits of the quantity, so its bars sit at ~1e-11 relative) and far
// below the smallest effect a mutation produces (see the trailing ledger).
/// |E_lj(ferric) - E_lj(OpenMM)| (Ha).
const OPENMM_E_BAR: f64 = 1e-14; // measured 8.7e-19 (the quantity is ~8e-4 Ha)
/// max |dE/dR(ferric) + F(OpenMM)| (Ha/Bohr).
const OPENMM_G_BAR: f64 = 1e-14; // measured 2.0e-18
/// Analytic vs central FD (h = 1e-5 Bohr) of the MM terms (Ha/Bohr).
const FD_BAR: f64 = 1.5e-10; // measured 1.47e-11

fn measuring() -> bool {
    std::env::var_os("FERRIC_MEASURE_BARS").is_some()
}

fn coords_of(atoms: &[QmmmAtom]) -> Array2<f64> {
    let mut c = Array2::<f64>::zeros((atoms.len(), 3));
    for (i, a) in atoms.iter().enumerate() {
        c[(i, 0)] = a.x;
        c[(i, 1)] = a.y;
        c[(i, 2)] = a.z_pos;
    }
    c
}

fn dist(a: &QmmmAtom, b: &QmmmAtom) -> f64 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z_pos - b.z_pos).powi(2)).sqrt()
}

fn lj_pair(a: LjParams, b: LjParams, r: f64) -> f64 {
    let sigma = 0.5 * (a.sigma + b.sigma);
    let eps = (a.epsilon * b.epsilon).sqrt();
    let sr6 = (sigma / r).powi(6);
    4.0 * eps * (sr6 * sr6 - sr6)
}

const C_LJ: LjParams = LjParams {
    sigma: 3.4 * ANG2BOHR,
    epsilon: 0.109 * KCAL,
};
const OFF: LjParams = LjParams {
    sigma: 0.0,
    epsilon: 0.0,
};

// ---------------------------------------------------------------------------
// (1) Exactness anchor
// ---------------------------------------------------------------------------

/// The pre-fix construction, rebuilt from public pieces: `qmmm_mm_terms` with
/// QM epsilon switched off (bonded + MM-MM only; its QM-MM pass adds +0.0)
/// plus the UNSCALED every-pair `qm_mm_lj_energy_gradient`, folded in the
/// same order `qmmm_mm_terms` folds it.
fn every_pair_lj(sys: &QmmmSystem, top: &MmTopology, coords: &Array2<f64>) -> (f64, Array2<f64>) {
    let mut top_qm_off = top.clone();
    for &i in &sys.qm_indices {
        top_qm_off.lj[i] = OFF;
    }
    let (e0, mut g) = qmmm_mm_terms(sys, &top_qm_off, coords).unwrap();
    let pick = |idx: &[usize]| {
        let mut c = Array2::<f64>::zeros((idx.len(), 3));
        for (row, &i) in idx.iter().enumerate() {
            for k in 0..3 {
                c[(row, k)] = coords[(i, k)];
            }
        }
        c
    };
    let qm_lj: Vec<_> = sys.qm_indices.iter().map(|&i| top.lj[i]).collect();
    let mm_lj: Vec<_> = sys.mm_indices.iter().map(|&i| top.lj[i]).collect();
    let (e_x, g_qm, g_mm) = qm_mm_lj_energy_gradient(
        &qm_lj,
        &pick(&sys.qm_indices),
        &mm_lj,
        &pick(&sys.mm_indices),
    );
    for (row, &i) in sys.qm_indices.iter().enumerate() {
        for k in 0..3 {
            g[(i, k)] += g_qm[(row, k)];
        }
    }
    for (row, &i) in sys.mm_indices.iter().enumerate() {
        for k in 0..3 {
            g[(i, k)] += g_mm[(row, k)];
        }
    }
    (e0.lj + e_x, g)
}

/// QM water + MM water, realistic TIP3P-ish LJ, bonds inside each molecule
/// only: no QM-MM pair is 1-2/1-3/1-4, so the result must be bit-identical
/// to the every-pair sum.
#[test]
fn no_bond_path_across_the_cut_is_bit_identical_to_every_pair_sum() {
    let roh = 0.9572 * ANG2BOHR;
    let mut atoms = vec![];
    for oz in [0.0, 3.0 * ANG2BOHR] {
        atoms.push(QmmmAtom::new("O", 8, 0.1, 0.0, oz, -0.834));
        atoms.push(QmmmAtom::new("H", 1, roh, 0.2, oz + 0.3, 0.417));
        atoms.push(QmmmAtom::new("H", 1, -0.3, roh, oz - 0.2, 0.417));
    }
    let lj: Vec<LjParams> = atoms
        .iter()
        .map(|a| match a.z {
            8 => LjParams {
                sigma: 3.1507 * ANG2BOHR,
                epsilon: 0.1521 * KCAL,
            },
            _ => LjParams {
                sigma: 1.0 * ANG2BOHR,
                epsilon: 0.046 * KCAL,
            },
        })
        .collect();
    let bonds = [(0, 1), (0, 2), (3, 4), (3, 5)]
        .map(|(i, j)| Bond {
            i,
            j,
            k: 0.4,
            r0: roh,
        })
        .to_vec();
    let charges = atoms.iter().map(|a| a.charge).collect();
    let top = MmTopology::new(charges, lj, bonds, vec![], vec![]).unwrap();
    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();
    let coords = coords_of(&atoms);

    let (e, g) = qmmm_mm_terms(&sys, &top, &coords).unwrap();
    let (e_ref, g_ref) = every_pair_lj(&sys, &top, &coords);
    assert!(e.lj != 0.0, "anchor is vacuous: zero LJ");
    assert_eq!(e.lj.to_bits(), e_ref.to_bits(), "{} vs {}", e.lj, e_ref);
    for (a, b) in g.iter().zip(g_ref.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "gradient {a} vs {b}");
    }
}

/// The same reconstruction is NOT equal once a bond crosses the cut — the
/// anchor above can see the difference (it is not a self-comparison).
#[test]
fn every_pair_sum_differs_once_a_bond_crosses_the_cut() {
    let (atoms, top) = chain(&[C_LJ; 5], 0.5);
    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0]), 0, 1).unwrap();
    let coords = coords_of(&atoms);
    let (e, _) = qmmm_mm_terms(&sys, &top, &coords).unwrap();
    let (e_ref, _) = every_pair_lj(&sys, &top, &coords);
    assert!((e.lj - e_ref).abs() > 1.0, "{} vs {}", e.lj, e_ref);
}

// ---------------------------------------------------------------------------
// (2) One test per pair class
// ---------------------------------------------------------------------------

/// Zig-zag carbon chain 0-1-2-3-4 (C-C 1.53 A, 109.5 deg, trans), LJ from
/// `lj`, zero charges, the given LJ 1-4 scale (Coulomb 1-4 left at AMBER's
/// 1/1.2 so the two factors differ). QM = {0}: (0,1) is 1-2, (0,2) 1-3,
/// (0,3) 1-4, (0,4) 1-5.
fn chain(lj: &[LjParams], scale_lj_14: f64) -> (Vec<QmmmAtom>, MmTopology) {
    let cc = 1.53 * ANG2BOHR;
    let half = (109.5_f64 / 2.0).to_radians();
    let (dx, dy) = (cc * half.sin(), cc * half.cos());
    let atoms: Vec<QmmmAtom> = (0..5)
        .map(|i| {
            let y = if i % 2 == 0 { 0.0 } else { dy };
            QmmmAtom::new("C", 6, dx * i as f64, y, 0.0, 0.0)
        })
        .collect();
    let bonds = (0..4)
        .map(|i| Bond {
            i,
            j: i + 1,
            k: 0.35,
            r0: cc,
        })
        .collect();
    let top = MmTopology::new(vec![0.0; 5], lj.to_vec(), bonds, vec![], vec![])
        .unwrap()
        .with_scales(scale_lj_14, 1.0 / 1.2);
    (atoms, top)
}

/// QM-MM LJ of the single pair (0, k): only atoms 0 and k carry epsilon, so
/// every MM-MM pair and every other QM-MM pair is zero.
fn isolated_pair_lj(k: usize, scale_lj_14: f64) -> (f64, f64) {
    let mut lj = [OFF; 5];
    lj[0] = C_LJ;
    lj[k] = C_LJ;
    let (atoms, top) = chain(&lj, scale_lj_14);
    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0]), 0, 1)
        .unwrap()
        .with_link_atoms(&[(0, 1)], DEFAULT_LINK_SCALE)
        .unwrap();
    let (e, _) = qmmm_mm_terms(&sys, &top, &coords_of(&atoms)).unwrap();
    let r = dist(&atoms[0], &atoms[k]);
    (e.lj, lj_pair(C_LJ, C_LJ, r))
}

#[test]
fn bonded_1_2_pair_across_the_cut_is_excluded() {
    let (e, unexcluded) = isolated_pair_lj(1, 0.5);
    // Realistic carbon sigma at a C-C bond length: thousands of kcal/mol.
    assert!(unexcluded > 1000.0 * KCAL, "{}", unexcluded / KCAL);
    assert_eq!(e, 0.0);
}

#[test]
fn bonded_1_3_pair_across_the_cut_is_excluded() {
    let (e, unexcluded) = isolated_pair_lj(2, 0.5);
    assert!(unexcluded > 1.0 * KCAL, "{}", unexcluded / KCAL);
    assert_eq!(e, 0.0);
}

#[test]
fn bonded_1_4_pair_across_the_cut_is_scaled_by_the_topology_factor() {
    for s in [0.5, 0.75] {
        let (e, full) = isolated_pair_lj(3, s);
        assert!(full.abs() > 1e-4, "{full}");
        assert!(
            (e - s * full).abs() <= 1e-15 * full.abs(),
            "scale {s}: {e} vs {}",
            s * full
        );
    }
}

#[test]
fn pair_1_5_across_the_cut_gets_the_full_term() {
    let (e, full) = isolated_pair_lj(4, 0.5);
    assert!(full.abs() > 1e-5, "{full}");
    assert!((e - full).abs() <= 1e-15 * full.abs(), "{e} vs {full}");
}

// ---------------------------------------------------------------------------
// (3) OpenMM reference
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Case {
    name: String,
    n_atoms: usize,
    qm_indices: Vec<usize>,
    coordinates_angstrom: Vec<[f64; 3]>,
    lj_sigma_angstrom: Vec<f64>,
    lj_epsilon_kcal: Vec<f64>,
    bonds: Vec<[usize; 2]>,
    scale_lj_14: f64,
    scale_coul_14: f64,
    qm_mm_lj_kcal: f64,
    mm_mm_lj_kcal: f64,
    qm_mm_lj_no_exclusions_kcal: f64,
    qm_mm_lj_forces_kcal_per_angstrom: Vec<[f64; 3]>,
    mm_mm_lj_forces_kcal_per_angstrom: Vec<[f64; 3]>,
}

#[derive(Deserialize)]
struct RefFile {
    cases: Vec<Case>,
}

fn load_cases() -> Vec<Case> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/reference/qmmm_lj_ethanol_openmm.json");
    let txt = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    serde_json::from_str::<RefFile>(&txt).unwrap().cases
}

/// Ethanol order: 0 C0, 1 C1, 2 O, 3 H(O), 4-6 H on C0, 7-8 H on C1.
const ETHANOL_Z: [(&str, i32); 9] = [
    ("C", 6),
    ("C", 6),
    ("O", 8),
    ("H", 1),
    ("H", 1),
    ("H", 1),
    ("H", 1),
    ("H", 1),
    ("H", 1),
];

fn ethanol_case(c: &Case, bond_k: f64) -> (Vec<QmmmAtom>, MmTopology, QmmmSystem) {
    assert_eq!(c.n_atoms, ETHANOL_Z.len());
    let atoms: Vec<QmmmAtom> = c
        .coordinates_angstrom
        .iter()
        .zip(ETHANOL_Z)
        .map(|(p, (sym, z))| {
            QmmmAtom::new(
                sym,
                z,
                p[0] * ANG2BOHR,
                p[1] * ANG2BOHR,
                p[2] * ANG2BOHR,
                0.0,
            )
        })
        .collect();
    let lj = c
        .lj_sigma_angstrom
        .iter()
        .zip(&c.lj_epsilon_kcal)
        .map(|(&s, &e)| LjParams {
            sigma: s * ANG2BOHR,
            epsilon: e * KCAL,
        })
        .collect();
    let bonds = c
        .bonds
        .iter()
        .map(|&[i, j]| Bond {
            i,
            j,
            k: bond_k,
            r0: dist(&atoms[i], &atoms[j]),
        })
        .collect();
    let top = MmTopology::new(vec![0.0; c.n_atoms], lj, bonds, vec![], vec![])
        .unwrap()
        .with_scales(c.scale_lj_14, c.scale_coul_14);
    let cut: Vec<(usize, usize)> = c.bonds.iter().map(|&[i, j]| (i, j)).collect();
    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(c.qm_indices.clone()), 0, 1)
        .unwrap()
        .with_link_atoms(&cut, DEFAULT_LINK_SCALE)
        .unwrap();
    (atoms, top, sys)
}

#[test]
fn qm_mm_lj_across_the_cut_matches_openmm_exceptions() {
    let cases = load_cases();
    assert_eq!(cases.len(), 3);
    let (mut worst_e, mut worst_g) = (0.0_f64, 0.0_f64);
    for c in &cases {
        // k = 0: the gradient is then the LJ gradient alone.
        let (atoms, top, sys) = ethanol_case(c, 0.0);
        assert_eq!(sys.link_atoms.len(), 1, "{}: one C-C cut", c.name);
        let coords = coords_of(&atoms);
        let (e, g) = qmmm_mm_terms(&sys, &top, &coords).unwrap();

        let e_ref = (c.qm_mm_lj_kcal + c.mm_mm_lj_kcal) * KCAL;
        let de = (e.lj - e_ref).abs();
        worst_e = worst_e.max(de);
        assert!(
            de < OPENMM_E_BAR,
            "{}: E_lj {:.15e} vs OpenMM {:.15e} Ha (|d| {de:.2e})",
            c.name,
            e.lj,
            e_ref
        );
        // F in kcal/mol/A -> dE/dR in Ha/Bohr.
        let f2g = -KCAL / ANG2BOHR;
        for i in 0..c.n_atoms {
            for k in 0..3 {
                let g_ref = f2g
                    * (c.qm_mm_lj_forces_kcal_per_angstrom[i][k]
                        + c.mm_mm_lj_forces_kcal_per_angstrom[i][k]);
                let dg = (g[(i, k)] - g_ref).abs();
                worst_g = worst_g.max(dg);
                assert!(
                    dg < OPENMM_G_BAR,
                    "{}: atom {i} axis {k}: {:.6e} vs {:.6e}",
                    c.name,
                    g[(i, k)],
                    g_ref
                );
            }
        }

        // Negative control: the every-pair (pre-fix) construction against
        // OpenMM with NO exceptions — agrees with that, so the reference
        // distinguishes the two constructions by many orders of magnitude.
        let (e_every, _) = every_pair_lj(&sys, &top, &coords);
        let e_every_ref = (c.qm_mm_lj_no_exclusions_kcal + c.mm_mm_lj_kcal) * KCAL;
        assert!(
            (e_every - e_every_ref).abs() < 1e-9 * e_every_ref.abs(),
            "{}: every-pair {e_every} vs OpenMM-no-exceptions {e_every_ref}",
            c.name
        );
        assert!((e_every - e_ref).abs() > 1e9 * OPENMM_E_BAR, "{}", c.name);
    }
    if measuring() {
        eprintln!("MEASURE openmm: worst |dE| {worst_e:.2e} Ha, worst |dg| {worst_g:.2e} Ha/Bohr");
    }
}

// ---------------------------------------------------------------------------
// (4) FD across the cut with realistic LJ
// ---------------------------------------------------------------------------

#[test]
fn mm_terms_gradient_matches_fd_across_the_cut_with_realistic_lj() {
    let mut worst = 0.0_f64;
    for c in &load_cases() {
        let (atoms, top, sys) = ethanol_case(c, 0.35);
        // Off the bonded minimum so the bond terms contribute too.
        let mut coords = coords_of(&atoms);
        coords[(1, 2)] += 0.05;
        coords[(4, 0)] -= 0.04;
        let (_, g) = qmmm_mm_terms(&sys, &top, &coords).unwrap();
        let h = 1e-5;
        for i in 0..c.n_atoms {
            for k in 0..3 {
                let mut p = coords.clone();
                p[(i, k)] += h;
                let mut m = coords.clone();
                m[(i, k)] -= h;
                let fd = (qmmm_mm_terms(&sys, &top, &p).unwrap().0.total
                    - qmmm_mm_terms(&sys, &top, &m).unwrap().0.total)
                    / (2.0 * h);
                let d = (fd - g[(i, k)]).abs();
                worst = worst.max(d);
                assert!(d < FD_BAR, "{}: atom {i} axis {k}: |d| {d:.2e}", c.name);
            }
        }
    }
    if measuring() {
        eprintln!("MEASURE fd: worst {worst:.2e} Ha/Bohr");
    }
}
