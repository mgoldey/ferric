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
//! * C2v follows the Mulliken convention: the plane holding more atoms is yz
//!   (a planar C2v molecule lies in yz), so B1 is symmetric under sigma(xz)
//!   and B2 under sigma(yz) whatever the input orientation. If both planes
//!   hold the same number of atoms the geometry does not decide B1 vs B2.
//! * Among equal-order subgroups the one with more C2 axes wins, so Td and D2d
//!   molecules resolve to D2 (not C2v) in every orientation.
//! * D2 and D2h take their axes from the order in which candidate directions
//!   are proposed (inertia axes first, then atom vectors in input order,
//!   then pair-derived directions; the first order-8 frame wins). For
//!   benzene (D6h) that selects the D2h subgroup whose z axis is the first
//!   such candidate; the other choices are conjugate under the full group.
//!   Labels B1g/B2g/B3g and B1u/B2u/B3u of D2/D2h therefore depend on the
//!   input; per-irrep dimension multisets do not.
//! * Cs (A', A''), C2 (A, B), Ci and C2h labels do not depend on the frame.
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
//! `n_core_ecp`. Isotopes and masses are ignored (HDO is C2v: the electronic
//! problem has that symmetry). The reference centre is the centroid weighted
//! by the element's `z`; ghosts count with their element's `z`, which only
//! fixes the centre (equivalent atoms always carry equal weight).
//!
//! # What detection does not see
//!
//! Only nuclei, species and the basis. External point charges, smeared
//! charges, uniform fields, PCM/COSMO cavities, polarizable sites and QM/MM
//! environments can break a symmetry detection reports. Code that wires
//! symmetry into an SCF must refuse those inputs; use
//! [`require_no_external_potential`](crate::symmetry::require_no_external_potential)
//! and its counterparts for the other model terms.
//!
//! # Tolerance limits
//!
//! Atom matching uses a tolerance ball. Same-species atoms closer than
//! `2 tol` make the permutation ambiguous, so [`detect`](crate::symmetry::detect) returns
//! an error for them instead of guessing.

mod ao_rep;
mod detect;

pub use ao_rep::{ao_function_parity, AoRepresentation};
pub use detect::{detect, SymmetryDetection};

use crate::external_potential::ExternalPotential;
use crate::FerricError;

/// Refuse a non-empty external potential: symmetry detection ignores it, so
/// point-group blocking would be wrong for a field or charge set that breaks
/// the symmetry of the nuclear frame.
pub fn require_no_external_potential(ext: Option<&ExternalPotential>) -> Result<(), FerricError> {
    match ext {
        Some(e) if !e.is_empty() => Err(FerricError::General(
            "point-group symmetry cannot be combined with an external potential \
             (point charges, smeared charges or a field): detection ignores it"
                .into(),
        )),
        _ => Ok(()),
    }
}

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
