//! Abelian point-group symmetry: detection, atom permutations, the AO
//! representation of each operation, and irrep-adapted (SO) bases.
//!
//! Scope is D2h and its seven subgroups (C1, Ci, Cs, C2, C2v, C2h, D2, D2h),
//! the groups SCF programs use for orbital blocking. A molecule of higher
//! symmetry (Td, D6h, D-infinity-h, ...) is handled by its largest Abelian
//! subgroup.
//!
//! # Frame convention
//!
//! [`detect`](crate::symmetry::detect) **reorients**: it returns a copy of the molecule translated to
//! its centre of nuclear charge, rotated into the symmetry frame, and
//! symmetrized (each atom replaced by the group average of its images). In
//! that frame every operation is a diagonal sign matrix `diag(sx, sy, sz)`,
//! which is what makes the AO representation a signed permutation. Callers
//! must build their integrals from [`SymmetryDetection::molecule`](crate::symmetry::SymmetryDetection::molecule), not from
//! the input geometry. The frame is canonical: the C2 axis (C2, C2v, C2h) is z,
//! the mirror plane of Cs is xy, the mirrors of C2v are xz and yz, and the
//! mirror of C2h is xy.
//!
//! # Operations
//!
//! The eight D2h operations are indexed `0..8` in the order of [`OP_NAMES`](crate::symmetry::OP_NAMES):
//! `E, C2(z), C2(y), C2(x), i, sigma(xy), sigma(xz), sigma(yz)`. A detected
//! group is the ascending list of the indices that map the molecule onto
//! itself.
//!
//! # Tolerance
//!
//! `tol` is a distance in Bohr: an operation is accepted when every atom lands
//! within `tol` of a same-species atom ([`DEFAULT_SYMMETRY_TOL`](crate::symmetry::DEFAULT_SYMMETRY_TOL) = 1e-4 Bohr).
//! A distortion larger than `tol` lowers the detected group; a distortion
//! smaller than `tol` is removed by the symmetrization (the shift is reported
//! in [`SymmetryDetection::max_symmetrization_shift`](crate::symmetry::SymmetryDetection::max_symmetrization_shift)).
//!
//! # Species
//!
//! Two atoms are interchangeable only if they share `z`, ghost flag and
//! `n_core_ecp`. The nuclear charge centroid is weighted by the physical `z`.

mod ao_rep;
mod detect;

pub use ao_rep::{ao_function_parity, AoRepresentation};
pub use detect::{detect, SymmetryDetection};

use crate::FerricError;

/// Default symmetry-detection tolerance in Bohr.
pub const DEFAULT_SYMMETRY_TOL: f64 = 1e-4;

/// Sign pattern `(sx, sy, sz)` of each D2h operation in the symmetry frame.
pub const OP_SIGNS: [[i8; 3]; 8] = [
    [1, 1, 1],
    [-1, -1, 1],
    [-1, 1, -1],
    [1, -1, -1],
    [-1, -1, -1],
    [1, 1, -1],
    [1, -1, 1],
    [-1, 1, 1],
];

/// Names of the eight D2h operations, indexed like [`OP_SIGNS`].
pub const OP_NAMES: [&str; 8] = [
    "E",
    "C2(z)",
    "C2(y)",
    "C2(x)",
    "i",
    "sigma(xy)",
    "sigma(xz)",
    "sigma(yz)",
];

const C2_OPS: [usize; 3] = [1, 2, 3];
const INVERSION: usize = 4;
const MIRROR_OPS: [usize; 3] = [5, 6, 7];

/// Index of the product of two D2h operations (sign patterns multiply).
pub fn compose(a: usize, b: usize) -> usize {
    let s = [
        OP_SIGNS[a][0] * OP_SIGNS[b][0],
        OP_SIGNS[a][1] * OP_SIGNS[b][1],
        OP_SIGNS[a][2] * OP_SIGNS[b][2],
    ];
    OP_SIGNS
        .iter()
        .position(|t| *t == s)
        .expect("D2h is closed")
}

/// The Abelian point groups: D2h and its subgroups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbelianGroup {
    C1,
    Ci,
    Cs,
    C2,
    C2v,
    C2h,
    D2,
    D2h,
}

impl AbelianGroup {
    /// Identify the group formed by the D2h operation indices `ops`
    /// (ascending, containing the identity).
    pub fn from_ops(ops: &[usize]) -> Result<Self, FerricError> {
        let has = |i: usize| ops.contains(&i);
        let n_c2 = C2_OPS.iter().filter(|&&i| has(i)).count();
        let n_mirror = MIRROR_OPS.iter().filter(|&&i| has(i)).count();
        let inv = has(INVERSION);
        let g = match (ops.len(), n_c2, n_mirror, inv) {
            (1, 0, 0, false) => Self::C1,
            (2, 0, 0, true) => Self::Ci,
            (2, 0, 1, false) => Self::Cs,
            (2, 1, 0, false) => Self::C2,
            (4, 1, 2, false) => Self::C2v,
            (4, 1, 1, true) => Self::C2h,
            (4, 3, 0, false) => Self::D2,
            (8, 3, 3, true) => Self::D2h,
            _ => {
                return Err(FerricError::General(format!(
                    "operation set {ops:?} is not an Abelian point group"
                )))
            }
        };
        if !has(0) {
            return Err(FerricError::General("operation set lacks identity".into()));
        }
        Ok(g)
    }

    /// Schoenflies symbol.
    pub fn name(self) -> &'static str {
        match self {
            Self::C1 => "C1",
            Self::Ci => "Ci",
            Self::Cs => "Cs",
            Self::C2 => "C2",
            Self::C2v => "C2v",
            Self::C2h => "C2h",
            Self::D2 => "D2",
            Self::D2h => "D2h",
        }
    }
}

/// A one-dimensional irreducible representation of an Abelian group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Irrep {
    /// Mulliken label.
    pub label: &'static str,
    /// Character (+1 or -1) of each operation, aligned with the group's
    /// ascending operation list.
    pub characters: Vec<i8>,
}

/// Exponents `(a, b, c)` of x, y, z whose monomial `x^a y^b z^c` carries each
/// D2h irrep, in the order Ag, B1g, B2g, B3g, Au, B1u, B2u, B3u.
const D2H_IRREP_MONOMIALS: [[u8; 3]; 8] = [
    [0, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
];

/// The irreps of the Abelian group formed by `ops`, totally symmetric first.
///
/// Every irrep of a D2h subgroup is the restriction of a D2h irrep; distinct
/// restrictions are kept once, in D2h order (Ag, B1g, B2g, B3g, Au, B1u, B2u,
/// B3u). The count equals `ops.len()`.
pub fn irreps(ops: &[usize]) -> Result<Vec<Irrep>, FerricError> {
    let group = AbelianGroup::from_ops(ops)?;
    let mut out: Vec<Irrep> = Vec::new();
    for e in D2H_IRREP_MONOMIALS {
        let characters: Vec<i8> = ops
            .iter()
            .map(|&g| {
                (0..3)
                    .map(|ax| OP_SIGNS[g][ax].pow(e[ax] as u32))
                    .product::<i8>()
            })
            .collect();
        if out.iter().all(|r| r.characters != characters) {
            let label = irrep_label(group, ops, &characters);
            out.push(Irrep { label, characters });
        }
    }
    Ok(out)
}

fn irrep_label(group: AbelianGroup, ops: &[usize], chars: &[i8]) -> &'static str {
    let chi = |idx: usize| chars[ops.iter().position(|&o| o == idx).expect("op in group")];
    let first = |set: &[usize]| {
        *set.iter()
            .find(|i| ops.contains(i))
            .expect("op class present")
    };
    let second_mirror = || {
        *MIRROR_OPS
            .iter()
            .filter(|i| ops.contains(i))
            .nth(1)
            .expect("2 mirrors")
    };
    match group {
        AbelianGroup::C1 => "A",
        AbelianGroup::Ci => pick(chi(INVERSION), "Ag", "Au"),
        AbelianGroup::Cs => pick(chi(first(&MIRROR_OPS)), "A'", "A''"),
        AbelianGroup::C2 => pick(chi(first(&C2_OPS)), "A", "B"),
        AbelianGroup::C2v => {
            match (
                chi(first(&C2_OPS)),
                chi(first(&MIRROR_OPS)),
                chi(second_mirror()),
            ) {
                (1, 1, _) => "A1",
                (1, -1, _) => "A2",
                (-1, 1, _) => "B1",
                _ => "B2",
            }
        }
        AbelianGroup::C2h => match (chi(first(&C2_OPS)), chi(INVERSION)) {
            (1, 1) => "Ag",
            (-1, 1) => "Bg",
            (1, -1) => "Au",
            _ => "Bu",
        },
        AbelianGroup::D2 => match (chi(1), chi(2), chi(3)) {
            (1, 1, 1) => "A",
            (1, -1, -1) => "B1",
            (-1, 1, -1) => "B2",
            _ => "B3",
        },
        AbelianGroup::D2h => {
            let k = D2H_IRREP_MONOMIALS
                .iter()
                .position(|e| {
                    ops.iter().zip(chars).all(|(&g, &c)| {
                        let t: i8 = (0..3).map(|ax| OP_SIGNS[g][ax].pow(e[ax] as u32)).product();
                        t == c
                    })
                })
                .expect("D2h irrep");
            ["Ag", "B1g", "B2g", "B3g", "Au", "B1u", "B2u", "B3u"][k]
        }
    }
}

fn pick(c: i8, plus: &'static str, minus: &'static str) -> &'static str {
    if c > 0 {
        plus
    } else {
        minus
    }
}

#[cfg(test)]
mod tests;
