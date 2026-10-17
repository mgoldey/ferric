//! AO representation of the point-group operations and irrep-adapted bases.
//!
//! In the symmetry frame an operation `g = diag(sx, sy, sz)` carries the AO
//! `phi_{A,k}` on atom `A` to `s_k(g) * phi_{gA,k}`: the same function on the
//! image atom times the parity sign of its angular part. Real solid harmonics
//! and Cartesian monomials both have definite parity in each coordinate, so
//! the sign is `sx^px * sy^py * sz^pz` with `(px, py, pz)` from
//! [`ao_function_parity`]. The representation is therefore a signed
//! permutation matrix, independent of radial parts and normalization.
//!
//! AO order follows libint2 (the layout `PreparedBasis` uses): atoms in input
//! order, shells in basis order, and within a shell
//! - pure: `m = -l ..= l` (so `p` is y, z, x);
//! - Cartesian: `lx` descending, then `ly` descending (xx, xy, xz, yy, yz, zz).

use super::{irreps, Irrep, SymmetryDetection, OP_SIGNS};
use crate::basis::{num_functions, BasisSet};
use crate::mol::Molecule;
use crate::FerricError;
use ndarray::Array2;

/// Parities `(px, py, pz)` (0 even, 1 odd) of function `k` of a shell with
/// angular momentum `l` under x -> -x, y -> -y, z -> -z.
///
/// Pure real solid harmonics, `m = k - l`: `m = 0` has `(0, 0, l mod 2)`;
/// `m > 0` (cos-like, `Re (x+iy)^m`) has `(m mod 2, 0, (l-m) mod 2)`; `m < 0`
/// (sin-like, `Im (x+iy)^|m|`) has `((|m|-1) mod 2, 1, (l-|m|) mod 2)`.
/// Cartesian: the parities of the exponents `(lx, ly, lz)`.
pub fn ao_function_parity(l: i32, k: usize, pure: bool) -> [u8; 3] {
    if pure {
        let m = k as i32 - l;
        let a = m.abs();
        return match m.signum() {
            0 => [0, 0, (l % 2) as u8],
            1 => [(a % 2) as u8, 0, ((l - a) % 2) as u8],
            _ => [((a - 1) % 2) as u8, 1, ((l - a) % 2) as u8],
        };
    }
    let mut idx = 0;
    for lx in (0..=l).rev() {
        for ly in (0..=(l - lx)).rev() {
            if idx == k {
                return [(lx % 2) as u8, (ly % 2) as u8, ((l - lx - ly) % 2) as u8];
            }
            idx += 1;
        }
    }
    panic!("function index {k} out of range for l={l}");
}

/// The signed-permutation AO representation of each operation of a detection.
#[derive(Debug, Clone)]
pub struct AoRepresentation {
    nao: usize,
    ops: Vec<usize>,
    /// `target[k][mu]`: the AO that `mu` is carried to by the k-th operation.
    target: Vec<Vec<usize>>,
    /// `sign[k][mu]`: the factor on that image.
    sign: Vec<Vec<f64>>,
}

/// One AO's angular type: (l, index within shell, pure).
type AoKind = (i32, usize, bool);

/// `det.molecule` is already the symmetrized geometry, so a correct caller
/// passes it unchanged (distance 0). Anything farther than roundoff would make
/// the atom permutations inexact symmetries of the integrals.
const SAME_GEOMETRY_TOL: f64 = 1e-12;

fn check_same_geometry(det: &SymmetryDetection, mol: &Molecule) -> Result<(), FerricError> {
    let reference = &det.molecule.atoms;
    let ok = mol.atoms.len() == reference.len()
        && mol.atoms.iter().zip(reference).all(|(a, b)| {
            let d = ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.zpos - b.zpos).powi(2)).sqrt();
            a.z == b.z
                && a.ghost == b.ghost
                && a.n_core_ecp == b.n_core_ecp
                && d <= SAME_GEOMETRY_TOL
        });
    if ok {
        Ok(())
    } else {
        Err(FerricError::General(
            "AoRepresentation: the molecule is not the reoriented, symmetrized \
             SymmetryDetection::molecule (atoms or coordinates differ by more than 1e-12 Bohr; pass it \
             unchanged); the AO representation is only valid in the symmetry frame"
                .into(),
        ))
    }
}

/// AO kinds of one atom, in order.
fn atom_ao_kinds(bs: &BasisSet, z: i32) -> Result<Vec<AoKind>, FerricError> {
    let shells = bs
        .for_element(z)
        .ok_or_else(|| FerricError::Basis(format!("no basis functions for Z={z}")))?;
    let mut out = Vec::new();
    for sh in shells {
        for k in 0..num_functions(sh.l, sh.pure) {
            out.push((sh.l, k, sh.pure));
        }
    }
    Ok(out)
}

impl AoRepresentation {
    /// Build the representation for `det.molecule` in basis `bs`.
    ///
    /// `mol` is the molecule the integrals will be built from. It must be
    /// `det.molecule` (same atoms, same coordinates to 1e-12 Bohr, i.e. pass `det.molecule` unchanged): the
    /// representation is only valid in the symmetry frame, so a molecule in
    /// the original input frame is rejected.
    pub fn new(
        det: &SymmetryDetection,
        mol: &Molecule,
        bs: &BasisSet,
    ) -> Result<Self, FerricError> {
        check_same_geometry(det, mol)?;
        let kinds: Vec<Vec<AoKind>> = det
            .molecule
            .atoms
            .iter()
            .map(|a| atom_ao_kinds(bs, a.z))
            .collect::<Result<_, _>>()?;
        let mut offset = vec![0usize; kinds.len() + 1];
        for (i, k) in kinds.iter().enumerate() {
            offset[i + 1] = offset[i] + k.len();
        }
        let nao = offset[kinds.len()];
        let mut target = Vec::new();
        let mut sign = Vec::new();
        for (&g, perm) in det.ops.iter().zip(&det.atom_perms) {
            let mut t = vec![0usize; nao];
            let mut s = vec![0.0; nao];
            for (a, ks) in kinds.iter().enumerate() {
                let b = perm[a];
                if kinds[b].len() != ks.len() {
                    return Err(FerricError::Basis(format!(
                        "atoms {a} and {b} are symmetry images but carry different basis sizes"
                    )));
                }
                for (j, &(l, k, pure)) in ks.iter().enumerate() {
                    let par = ao_function_parity(l, k, pure);
                    let sg: i32 = (0..3)
                        .map(|ax| (OP_SIGNS[g][ax] as i32).pow(par[ax] as u32))
                        .product();
                    t[offset[a] + j] = offset[b] + j;
                    s[offset[a] + j] = sg as f64;
                }
            }
            target.push(t);
            sign.push(s);
        }
        Ok(Self {
            nao,
            ops: det.ops.clone(),
            target,
            sign,
        })
    }

    /// Number of AOs.
    pub fn nao(&self) -> usize {
        self.nao
    }

    /// Dense `nao x nao` matrix R of the k-th operation: `R[target[mu], mu] = sign[mu]`.
    pub fn dense(&self, k: usize) -> Array2<f64> {
        let mut r = Array2::zeros((self.nao, self.nao));
        for mu in 0..self.nao {
            r[[self.target[k][mu], mu]] = self.sign[k][mu];
        }
        r
    }

    /// Dense projector `P = (1/|G|) sum_g chi(g) R(g)` onto `irrep`.
    ///
    /// `irrep` must be one of [`irreps`](crate::symmetry::irreps) of this
    /// representation's operations (compared by label and characters); a
    /// foreign irrep, such as one of another group or of a different order,
    /// is an error. Taking the irrep by value from the representation's own
    /// list would not stop a caller building a lookalike, so it is verified.
    pub fn projector(&self, irrep: &Irrep) -> Result<Array2<f64>, FerricError> {
        self.check_irrep(irrep)?;
        let mut p = Array2::zeros((self.nao, self.nao));
        for (k, &chi) in irrep.characters.iter().enumerate() {
            p.scaled_add(chi as f64, &self.dense(k));
        }
        Ok(p / self.ops.len() as f64)
    }

    fn check_irrep(&self, irrep: &Irrep) -> Result<(), FerricError> {
        if irreps(&self.ops)?.contains(irrep) {
            Ok(())
        } else {
            Err(FerricError::General(format!(
                "irrep {} {:?} does not belong to this representation's group (operations {:?})",
                irrep.label, irrep.characters, self.ops
            )))
        }
    }

    /// Orthonormal (Euclidean) irrep-adapted AO combinations: one column per
    /// AO orbit whose projection onto `irrep` is nonzero, shape `(nao, dim)`.
    ///
    /// Errors for an `irrep` that is not one of this representation's own
    /// (see [`projector`](Self::projector)).
    pub fn so_basis(&self, irrep: &Irrep) -> Result<Array2<f64>, FerricError> {
        self.check_irrep(irrep)?;
        Ok(self.so_basis_unchecked(irrep))
    }

    fn so_basis_unchecked(&self, irrep: &Irrep) -> Array2<f64> {
        let mut visited = vec![false; self.nao];
        let mut cols: Vec<Vec<f64>> = Vec::new();
        for seed in 0..self.nao {
            if visited[seed] {
                continue;
            }
            let mut v = vec![0.0; self.nao];
            for (k, &chi) in irrep.characters.iter().enumerate() {
                v[self.target[k][seed]] += chi as f64 * self.sign[k][seed];
                visited[self.target[k][seed]] = true;
            }
            let n2: f64 = v.iter().map(|x| x * x).sum();
            if n2 > 0.5 {
                let inv = 1.0 / n2.sqrt();
                cols.push(v.into_iter().map(|x| x * inv).collect());
            }
        }
        let mut out = Array2::zeros((self.nao, cols.len()));
        for (c, col) in cols.iter().enumerate() {
            for (r, x) in col.iter().enumerate() {
                out[[r, c]] = *x;
            }
        }
        out
    }

    /// `(irrep, dimension)` for every irrep of the group (dimension may be 0).
    pub fn irrep_dimensions(&self) -> Result<Vec<(Irrep, usize)>, FerricError> {
        Ok(irreps(&self.ops)?
            .into_iter()
            .map(|ir| {
                let d = self.so_basis_unchecked(&ir).ncols();
                (ir, d)
            })
            .collect())
    }
}
