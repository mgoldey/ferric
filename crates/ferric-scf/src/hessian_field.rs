//! Uniform external electric field in the analytic Hessian.
//!
//! The field couples through `h_field = E·r` (dipole integrals about the
//! origin, [`ferric_integrals::oneelectron::field_hcore_term`]) and the
//! classical field–nuclear energy `−E·Σ Z_A R_A`, which is LINEAR in the
//! nuclear coordinates, so its Hessian is exactly zero. The electronic part
//! needs, at fixed AO density `D`,
//!
//! * the skeleton term `Σ_μν D_μν ∂²M_μν/∂x∂y` (added to Term 2), and
//! * the first-derivative AO matrices `∂M/∂x` (added to `h1`, hence to `F^x`
//!   in the CPHF right-hand side),
//!
//! with `M = E·r` differentiated with respect to the basis centres.
//!
//! # How the derivatives are obtained
//!
//! libint2 in this build has `DISABLE_ONEBODY_PROPERTY_DERIVS`, so there is no
//! `emultipole1` derivative engine. The dipole integral is instead
//! differentiated numerically, but NOT by finite-differencing the Hessian or a
//! gradient: the integrals themselves are re-evaluated with displaced centres
//! and a fourth-order central stencil (step [`STEP`]). This is accurate to
//! ~1e-9 because the integrals are smooth analytic functions of the centres.
//! (The field-density term of the GRADIENT is also a finite difference of
//! these same integrals, so an FD-of-gradient check does not independently
//! test the integral derivatives; `tests/field_hessian.rs` therefore also
//! anchors against second differences of the SCF energy.)
//!
//! `M_μν` depends only on the centres of the two atoms carrying μ and ν, and
//! the dipole operator carries no other-atom dependence, so every displaced
//! evaluation is done on a SUB-MOLECULE of one or two atoms: O(N²) tiny
//! integral builds, not O(N²) full-molecule builds. The same-atom block
//! `M_AA = R_A S_AA + ⟨μ|r−R_A|ν⟩` is exactly linear in `R_A`, so it has no
//! second derivative (skipped) and the first derivative `E_x S_AA` comes from
//! the same stencil.

use ferric_core::basis::BasisSet;
use ferric_core::external_potential::ExternalPotential;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::field_hcore_term;
use ndarray::{s, Array2};

/// Displacement step (Bohr) of the fourth-order stencils.
const STEP: f64 = 2e-3;
/// `(offset in steps, weight)` of the first-derivative stencil (÷ `12·STEP`).
const D1: [(f64, f64); 4] = [(-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0)];
/// Second-derivative stencil (÷ `12·STEP²`).
const D2: [(f64, f64); 5] = [
    (-2.0, -1.0),
    (-1.0, 16.0),
    (0.0, -30.0),
    (1.0, 16.0),
    (2.0, -1.0),
];

/// `(local atom, axis, step offset)` displacement of a sub-molecule.
type Disp = (usize, usize, f64);

/// The nonzero uniform field of an external potential, if any.
pub(crate) fn nonzero_field(ext: Option<&ExternalPotential>) -> Option<[f64; 3]> {
    ext.and_then(|e| e.field).filter(|f| *f != [0.0; 3])
}

/// AO indices carried by each atom, in shell order.
fn ao_by_atom(prep: &PreparedBasis, natoms: usize) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); natoms];
    let (dims, offs) = (prep.shell_dims(), prep.shell_offsets());
    for (sh, &atom) in prep.shell_to_atom().iter().enumerate() {
        out[atom].extend(offs[sh]..offs[sh] + dims[sh]);
    }
    out
}

fn sub_molecule(mol: &Molecule, atoms: &[usize]) -> Molecule {
    let mut m = mol.clone();
    m.atoms = atoms.iter().map(|&a| mol.atoms[a].clone()).collect();
    m
}

/// `E·r` matrix of the sub-molecule with the displacements applied.
fn shifted_matrix(
    sub: &Molecule,
    bs: &BasisSet,
    field: [f64; 3],
    disps: &[Disp],
) -> Result<Array2<f64>, FerricError> {
    let mut m = sub.clone();
    for &(atom, axis, off) in disps {
        let a = &mut m.atoms[atom];
        let v = off * STEP;
        match axis {
            0 => a.x += v,
            1 => a.y += v,
            _ => a.zpos += v,
        }
    }
    field_hcore_term(&PreparedBasis::new(&m, bs)?, field)
}

/// First derivative of the sub-molecule's `E·r` matrix w.r.t. `(atom, axis)`.
fn deriv1_matrix(
    sub: &Molecule,
    bs: &BasisSet,
    field: [f64; 3],
    p: (usize, usize),
) -> Result<Array2<f64>, FerricError> {
    let mut acc: Option<Array2<f64>> = None;
    for (off, w) in D1 {
        let m = shifted_matrix(sub, bs, field, &[(p.0, p.1, off)])? * (w / (12.0 * STEP));
        acc = Some(match acc {
            Some(a) => a + m,
            None => m,
        });
    }
    Ok(acc.expect("stencil is non-empty"))
}

/// First-derivative AO matrices `∂(E·r)/∂R_{A,x}` for all `3·natoms`
/// coordinates (index `3A + x`): basis-centre dependence only (the dipole
/// operator has none on the nuclei).
pub(crate) fn field_first_derivative_matrices(
    prep: &PreparedBasis,
    mol: &Molecule,
    field: [f64; 3],
) -> Result<Vec<Array2<f64>>, FerricError> {
    let natoms = mol.atoms.len();
    let nbf = prep.nbasis();
    let bs = prep.basis_set();
    let aos = ao_by_atom(prep, natoms);
    let mut mats = vec![Array2::<f64>::zeros((nbf, nbf)); 3 * natoms];
    for a in 0..natoms {
        for axis in 0..3 {
            let mat = &mut mats[3 * a + axis];
            let m = deriv1_matrix(&sub_molecule(mol, &[a]), bs, field, (0, axis))?;
            for (i, &mu) in aos[a].iter().enumerate() {
                for (j, &nu) in aos[a].iter().enumerate() {
                    mat[(mu, nu)] += m[(i, j)];
                }
            }
            let na = aos[a].len();
            for c in (0..natoms).filter(|&c| c != a) {
                let m = deriv1_matrix(&sub_molecule(mol, &[a, c]), bs, field, (0, axis))?;
                for (i, &mu) in aos[a].iter().enumerate() {
                    for (j, &nu) in aos[c].iter().enumerate() {
                        mat[(mu, nu)] += m[(i, na + j)];
                        mat[(nu, mu)] += m[(i, na + j)];
                    }
                }
            }
        }
    }
    Ok(mats)
}

/// One atom pair's field energy at fixed density,
/// `g(R_a, R_c) = 2 Σ_{μ∈a,ν∈c} D_μν M_μν` (both off-diagonal blocks).
struct PairEnergy<'a> {
    sub: Molecule,
    bs: &'a BasisSet,
    field: [f64; 3],
    w: Array2<f64>,
}

impl PairEnergy<'_> {
    fn value(&self, disps: &[Disp]) -> Result<f64, FerricError> {
        let m = shifted_matrix(&self.sub, self.bs, self.field, disps)?;
        let (na, nc) = self.w.dim();
        let blk = m.slice(s![..na, na..na + nc]);
        Ok(2.0 * (&self.w * &blk).sum())
    }

    /// `∂²g/∂p∂q` by fourth-order stencils (`p == q` uses the pure
    /// second-derivative stencil).
    fn second(&self, p: (usize, usize), q: (usize, usize)) -> Result<f64, FerricError> {
        let mut acc = 0.0;
        if p == q {
            for (off, w) in D2 {
                acc += w * self.value(&[(p.0, p.1, off)])?;
            }
            return Ok(acc / (12.0 * STEP * STEP));
        }
        for (oi, wi) in D1 {
            for (oj, wj) in D1 {
                acc += wi * wj * self.value(&[(p.0, p.1, oi), (q.0, q.1, oj)])?;
            }
        }
        Ok(acc / (144.0 * STEP * STEP))
    }

    /// Add this pair's second derivatives to the Hessian blocks of atoms
    /// `(a, c)`: the mixed `(a,c)` block and each atom's own `3×3` block.
    fn accumulate(&self, h: &mut Array2<f64>, a: usize, c: usize) -> Result<(), FerricError> {
        for x in 0..3 {
            for y in 0..3 {
                let v = self.second((0, x), (1, y))?;
                h[(3 * a + x, 3 * c + y)] += v;
                h[(3 * c + y, 3 * a + x)] += v;
            }
        }
        for (local, atom) in [(0usize, a), (1usize, c)] {
            for x in 0..3 {
                for y in x..3 {
                    let v = self.second((local, x), (local, y))?;
                    h[(3 * atom + x, 3 * atom + y)] += v;
                    if x != y {
                        h[(3 * atom + y, 3 * atom + x)] += v;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Skeleton field term `Σ_μν D_μν ∂²(E·r)_μν/∂x∂y` (3N × 3N, Ha/Bohr²).
pub(crate) fn field_skeleton_hessian(
    prep: &PreparedBasis,
    mol: &Molecule,
    field: [f64; 3],
    d: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let natoms = mol.atoms.len();
    let bs = prep.basis_set();
    let aos = ao_by_atom(prep, natoms);
    let mut h = Array2::<f64>::zeros((3 * natoms, 3 * natoms));
    for a in 0..natoms {
        for c in a + 1..natoms {
            let w = Array2::from_shape_fn((aos[a].len(), aos[c].len()), |(i, j)| {
                d[(aos[a][i], aos[c][j])]
            });
            let pe = PairEnergy {
                sub: sub_molecule(mol, &[a, c]),
                bs,
                field,
                w,
            };
            pe.accumulate(&mut h, a, c)?;
        }
    }
    Ok(h)
}
