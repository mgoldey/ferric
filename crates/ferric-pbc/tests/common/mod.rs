//! Shared fixtures for the Stage-1 PBC integration tests: the prototype's
//! cells (`reference/pbc/test_prototype.py`) and bases with PySCF's exact
//! numbers, and a Gamma-point RHF driver over `solve_rhf_injected`.
//!
//! Basis convention: the prototype ran PySCF with `cart=True`. Every basis
//! here is s or Cartesian p (`pure: false`; ferric never builds pure s/p, and
//! for l <= 1 Cartesian and spherical functions span the same space with the
//! same normalisation — only libint2's pure-p ORDER (y,z,x) would differ,
//! which `pair_ft` refuses anyway), so cart vs sph does not matter here.
#![allow(dead_code)]

use ferric_core::basis::{BasisSet, Shell};
use ferric_core::mol::{Atom, Molecule};
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_pbc::dense_aft::DenseAftEri;
use ferric_pbc::hcore::PeriodicHcore;
use ferric_pbc::lattice::Cell;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf_injected, PeriodicInjection, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::collections::HashMap;

/// `pbc_gamma.py` H2 geometry (Bohr).
pub const H2_ATOMS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.3, 0.2, 1.5]];
/// `test_prototype.py` TRI_A / TRI_ATOMS (Bohr).
pub const TRI_A: [[f64; 3]; 3] = [[4.6, 0.0, 0.0], [0.9, 4.3, 0.0], [0.5, 0.7, 4.8]];
pub const TRI_ATOMS: [[f64; 3]; 4] = [
    [0.1, 0.2, 0.3],
    [0.1, 0.2, 1.7],
    [2.4, 2.5, 2.2],
    [3.6, 2.9, 2.6],
];

pub fn h_atom(r: [f64; 3]) -> Atom {
    Atom {
        symbol: "H".into(),
        z: 1,
        x: r[0],
        y: r[1],
        zpos: r[2],
        ghost: false,
        n_core_ecp: 0,
    }
}

pub fn hydrogens(pos: &[[f64; 3]]) -> Molecule {
    Molecule {
        atoms: pos.iter().map(|r| h_atom(*r)).collect(),
        charge: 0,
        multiplicity: 1,
    }
}

/// Same rule as `ferric_core::basis::renormalize_contraction` (private):
/// coefficients refer to unit-normalised primitives and the contraction is
/// scaled to unit self-overlap — what `pair_ft` assumes of every `BasisSet`.
fn renormalized(l: i32, exps: &[f64], coefs: &[f64]) -> Shell {
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

fn h_basis(name: &str, shells: Vec<Shell>) -> BasisSet {
    let mut m = HashMap::new();
    m.insert(1, shells);
    BasisSet {
        name: name.into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

const STO3G_H_EXPS: [f64; 3] = [3.42525091, 0.62391373, 0.1688554];
const STO3G_H_COEFS: [f64; 3] = [0.15432897, 0.53532814, 0.44463454];

/// PySCF's `sto-3g` for H, digit for digit (ferric's bundled BSE copy
/// carries more digits; the pinned PySCF energies need PySCF's).
pub fn pyscf_sto3g_h() -> BasisSet {
    h_basis(
        "pyscf-sto-3g-H",
        vec![renormalized(0, &STO3G_H_EXPS, &STO3G_H_COEFS)],
    )
}

/// PySCF's `6-31G` for H, digit for digit (`pople-basis/6-31G.dat`; ferric's
/// bundled BSE copy carries more digits) — the Iteration 5 LMP2 prototype's
/// orbital basis (s only on H, so cart == sph).
pub fn pyscf_631g_h() -> BasisSet {
    h_basis(
        "pyscf-6-31g-H",
        vec![
            renormalized(
                0,
                &[18.7311370, 2.8253937, 0.6401217],
                &[0.03349460, 0.23472695, 0.81375733],
            ),
            renormalized(0, &[0.1612778], &[1.0]),
        ],
    )
}

/// `test_prototype.py` SP_BASIS: STO-3G s plus one Cartesian p (0.8).
pub fn sp_basis_h() -> BasisSet {
    h_basis(
        "pbc-proto-sp-H",
        vec![
            renormalized(0, &STO3G_H_EXPS, &STO3G_H_COEFS),
            renormalized(1, &[0.8], &[1.0]),
        ],
    )
}

/// One unit-normalised primitive s of exponent `alpha` on H (the RS-GDF
/// exactness anchor's orbital basis, `test_prototype.py` `ANCHOR_ALPHA`).
pub fn single_s_h(alpha: f64) -> BasisSet {
    h_basis("pbc-anchor-s-H", vec![renormalized(0, &[alpha], &[1.0])])
}

pub fn cubic(a: f64) -> [[f64; 3]; 3] {
    [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
}

pub fn h2_cell(a: f64) -> Cell {
    Cell::new(hydrogens(&H2_ATOMS), cubic(a)).expect("h2 cell")
}

pub fn triclinic_cell() -> Cell {
    Cell::new(hydrogens(&TRI_ATOMS), TRI_A).expect("triclinic cell")
}

pub fn prep_for(cell: &Cell, bs: &BasisSet) -> PreparedBasis {
    PreparedBasis::new(cell.mol(), bs).expect("prep")
}

pub fn max_abs_diff(a: &ndarray::Array2<f64>, b: &ndarray::Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter().zip(b.iter()).fold(0.0_f64, |m, (x, y)| {
        let d = (x - y).abs();
        assert!(d.is_finite(), "non-finite matrix difference: {x} vs {y}");
        m.max(d)
    })
}

pub fn array2(rows: &[&[f64]]) -> ndarray::Array2<f64> {
    let n = rows.len();
    let m = rows[0].len();
    ndarray::Array2::from_shape_fn((n, m), |(i, j)| rows[i][j])
}

/// Tight closed-shell RHF config the injected path accepts.
pub fn gamma_config() -> RhfConfig {
    RhfConfig {
        use_sad_guess: false,
        density_conv: 1e-10,
        max_iter: 200,
        ..Default::default()
    }
}

/// Gamma-point RHF on the lattice `(S, h, E_nn)` with arbitrary injected
/// J/K builders (e.g. the RS-GDF ones).
pub fn gamma_rhf_jk<'a>(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    j: Box<dyn ferric_scf::fock::JBuilder + 'a>,
    k: Box<dyn ferric_scf::fock::KBuilder + 'a>,
) -> ScfResult {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    // Never read on the injected path; required by the signature.
    let bounds = SchwarzBounds::compute(op, prep).expect("schwarz");
    let inj = PeriodicInjection {
        s: hc.s.clone(),
        h: hc.h.clone(),
        vnn: hc.enn,
        j,
        k,
    };
    let r = solve_rhf_injected(&ctx, cell.mol(), prep, op, &bounds, &gamma_config(), inj)
        .expect("gamma-point RHF");
    assert!(r.converged, "gamma-point RHF did not converge");
    r
}

/// Gamma-point RHF: `solve_rhf_injected` on the lattice `(S, h, E_nn)` and
/// the dense pure-AFT J/K (with whatever Madelung term `eri` carries).
pub fn gamma_rhf(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
) -> ScfResult {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    // Never read on the injected path; required by the signature.
    let bounds = SchwarzBounds::compute(op, prep).expect("schwarz");
    let inj = PeriodicInjection {
        s: hc.s.clone(),
        h: hc.h.clone(),
        vnn: hc.enn,
        j: Box::new(eri.j_builder()),
        k: Box::new(eri.k_builder()),
    };
    let r = solve_rhf_injected(&ctx, cell.mol(), prep, op, &bounds, &gamma_config(), inj)
        .expect("gamma-point RHF");
    assert!(r.converged, "gamma-point RHF did not converge");
    r
}
