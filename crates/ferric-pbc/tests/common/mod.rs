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

/// `m` with its `μ ≤ ν` half copied onto `μ > ν`: what the Gamma s2 walks
/// (module docs "Orbital-pair symmetry" of `hcore` / `rsgdf`) produce from
/// the ordered loop's elements.
pub fn mirror_upper(m: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    assert_eq!(m.nrows(), m.ncols());
    ndarray::Array2::from_shape_fn(m.dim(), |(i, j)| m[(i.min(j), i.max(j))])
}

/// [`mirror_upper`] for a pair-row tensor `(n², naux)` (row `μ n + ν`).
pub fn mirror_upper_pair_rows(j3: &ndarray::Array2<f64>, n: usize) -> ndarray::Array2<f64> {
    assert_eq!(j3.nrows(), n * n);
    ndarray::Array2::from_shape_fn(j3.dim(), |(r, p)| {
        let (m, k) = (r / n, r % n);
        j3[(m.min(k) * n + m.max(k), p)]
    })
}

/// Row `μ n + ν` ↔ `ν n + μ` of a pair-row tensor `(n², naux)`.
pub fn transpose_pair_rows(j3: &ndarray::Array2<f64>, n: usize) -> ndarray::Array2<f64> {
    assert_eq!(j3.nrows(), n * n);
    ndarray::Array2::from_shape_fn(j3.dim(), |(r, p)| j3[((r % n) * n + r / n, p)])
}

/// `m` with its `μ ≤ ν` half kept and `μ > ν` set to `conj m[(ν, μ)]`: what
/// the k-point s2 one-electron walks (`hcore::kpoint` module doc
/// "Orbital-pair symmetry") produce from the ordered loop's elements.
pub fn mirror_upper_herm(
    m: &ndarray::Array2<num_complex::Complex64>,
) -> ndarray::Array2<num_complex::Complex64> {
    assert_eq!(m.nrows(), m.ncols());
    ndarray::Array2::from_shape_fn(
        m.dim(),
        |(i, j)| {
            if i <= j {
                m[(i, j)]
            } else {
                m[(j, i)].conj()
            }
        },
    )
}

/// The shell index of every basis function of `prep`.
pub fn shell_of(prep: &PreparedBasis) -> Vec<usize> {
    let mut out = Vec::with_capacity(prep.nbasis());
    for (sh, &d) in prep.shell_dims().iter().enumerate() {
        assert_eq!(out.len(), prep.shell_offsets()[sh], "shell offsets");
        out.extend(std::iter::repeat_n(sh, d));
    }
    out
}

/// Residue index of integer coordinates `c` modulo `m` (row-major, the
/// `pair_ft::residues` convention), written out here so the tests derive
/// the k-point s2 bin map independently of the library.
fn test_residue(c: [i64; 3], m: [usize; 3]) -> usize {
    let r: Vec<usize> = (0..3)
        .map(|i| c[i].rem_euclid(m[i] as i64) as usize)
        .collect();
    (r[0] * m[1] + r[1]) * m[2] + r[2]
}

fn test_coords(r: usize, m: [usize; 3]) -> [i64; 3] {
    [
        (r / (m[1] * m[2])) as i64,
        ((r / m[2]) % m[1]) as i64,
        (r % m[2]) as i64,
    ]
}

/// The k-point s2 bin map `M(r_L, r_T) = (−r_L mod mod_l, (r_T − r_L) mod
/// mod_t)` on flat bins `r_L R_T + r_T` (`rsgdf::kpoint` module doc
/// "Orbital-pair symmetry at k"): bin `b` of row `μν` holds the sum bin
/// `M(b)` of row `νμ` holds, from `(μ_0 ν_L | P_T) = (ν_0 μ_{−L} | P_{T−L})`.
pub fn kpair_mirror(mod_l: [usize; 3], mod_t: [usize; 3], bin: usize) -> usize {
    let rt: usize = mod_t.iter().product();
    let (r, t) = (bin / rt, bin % rt);
    let c = test_coords(r, mod_l);
    let d = test_coords(t, mod_t);
    test_residue([-c[0], -c[1], -c[2]], mod_l) * rt
        + test_residue([d[0] - c[0], d[1] - c[1], d[2] - c[2]], mod_t)
}

/// The unsplit k-point s2 bins predicted from the FROZEN ordered bins `s1`
/// (`rsgdf::kpoint` module doc "Orbital-pair symmetry at k"): element `e =
/// (b, μν)` with mirror `M(e) = (M(b), νμ)` is the ordered `s1[e]` when `μ`'s
/// shell precedes `ν`'s, or the shells are equal and `e ≤ M(e)`
/// lexicographically in `(r_L, r_T, μ, ν)`; else `s1[M(e)]`.
pub fn kpair_s2_expected(
    s1: &[ndarray::Array2<f64>],
    n: usize,
    shell_of: &[usize],
    mod_l: [usize; 3],
    mod_t: [usize; 3],
) -> Vec<ndarray::Array2<f64>> {
    let rt: usize = mod_t.iter().product();
    (0..s1.len())
        .map(|b| {
            let mb = kpair_mirror(mod_l, mod_t, b);
            ndarray::Array2::from_shape_fn(s1[b].dim(), |(row, p)| {
                let (mu, nu) = (row / n, row % n);
                let e = (b / rt, b % rt, mu, nu);
                let me = (mb / rt, mb % rt, nu, mu);
                let own = shell_of[mu] < shell_of[nu] || (shell_of[mu] == shell_of[nu] && e <= me);
                if own {
                    s1[b][(row, p)]
                } else {
                    s1[mb][(nu * n + mu, p)]
                }
            })
        })
        .collect()
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
        xc: None,
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
        xc: None,
    };
    let r = solve_rhf_injected(&ctx, cell.mol(), prep, op, &bounds, &gamma_config(), inj)
        .expect("gamma-point RHF");
    assert!(r.converged, "gamma-point RHF did not converge");
    r
}
