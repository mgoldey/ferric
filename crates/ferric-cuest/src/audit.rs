//! cuEST compatibility audit for ferric basis sets.
//!
//! cuEST imposes three structural rules on a Gaussian basis:
//!
//! 1. **Angular-momentum contiguity**: on every centre (here, every element)
//!    the angular momenta present must be contiguous from `l = 0`: a centre
//!    with `p` needs `s`, one with `d` needs `s` and `p`.
//! 2. **Uniform representation**: every shell in the basis must use the same
//!    representation (all spherical `pure = true`, or all Cartesian
//!    `pure = false`). Mixed spherical/Cartesian within one molecule is not
//!    supported.
//! 3. **Spherical aux basis**: a density-fitting auxiliary basis must be
//!    entirely spherical (`pure = true`).
//!
//! These checks are pure Rust and need no device.

use ferric_core::basis::BasisSet;

/// A single cuEST-compatibility violation found while auditing a basis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasisViolation {
    /// Which rule was broken.
    pub kind: ViolationKind,
    /// Human-readable detail (element / l values / offending shells).
    pub detail: String,
}

/// The kind of cuEST-compatibility violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationKind {
    /// An element's angular momenta have a gap below their maximum.
    LContiguityHole,
    /// Both spherical (`pure=true`) and Cartesian (`pure=false`) shells occur.
    MixedRepresentation,
    /// An auxiliary (DF) basis contains a Cartesian (`pure=false`) shell.
    AuxNotSpherical,
}

/// Result of auditing one or more basis sets for cuEST compatibility.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BasisAudit {
    /// Violations found, in detection order.
    pub violations: Vec<BasisViolation>,
}

impl BasisAudit {
    /// `true` iff no violations were found.
    pub fn is_ok(&self) -> bool {
        self.violations.is_empty()
    }
}

/// Audit a single (primary) basis set for cuEST compatibility.
///
/// Checks rules (1) and (2). Rule (3) is checked by
/// [`audit_basis_with_aux`] when an auxiliary basis is supplied.
pub fn audit_basis(bs: &BasisSet) -> BasisAudit {
    let mut violations = Vec::new();

    // Rule 2: uniform representation. Only angular momenta l>=2 distinguish
    // spherical from Cartesian (l=0,1 have identical function counts), so a
    // difference in the `pure` flag among s/p shells is not a real mixed
    // representation. Flag a violation only when two shells with l>=2 disagree.
    let mut high_pure: Option<bool> = None;
    let mut mixed = false;
    for shells in bs.shells.values() {
        for sh in shells {
            if sh.l >= 2 {
                match high_pure {
                    None => high_pure = Some(sh.pure),
                    Some(p) if p != sh.pure => mixed = true,
                    _ => {}
                }
            }
        }
    }
    if mixed {
        violations.push(BasisViolation {
            kind: ViolationKind::MixedRepresentation,
            detail: format!(
                "basis {:?} mixes spherical (pure=true) and Cartesian (pure=false) shells for l>=2",
                bs.name
            ),
        });
    }

    // Rule 1: angular-momentum contiguity from 0, per element. cuEST rejects a
    // centre with a p shell but no s shell (and so on), so the union over
    // elements is not enough: {s} on one element and {p} on another is a hole.
    let mut elements: Vec<_> = bs.shells.iter().collect();
    elements.sort_by_key(|(z, _)| **z);
    for (z, shells) in elements {
        let present: std::collections::BTreeSet<i32> = shells.iter().map(|sh| sh.l).collect();
        let Some(&max_l) = present.iter().next_back() else {
            continue;
        };
        let missing: Vec<i32> = (0..=max_l).filter(|l| !present.contains(l)).collect();
        if !missing.is_empty() {
            violations.push(BasisViolation {
                kind: ViolationKind::LContiguityHole,
                detail: format!(
                    "basis {:?}, element Z={z}: missing angular momentum l={missing:?} (present: {:?})",
                    bs.name,
                    present.iter().copied().collect::<Vec<_>>()
                ),
            });
        }
    }

    BasisAudit { violations }
}

/// Audit a primary basis together with an optional density-fitting auxiliary
/// basis.
///
/// Applies [`audit_basis`] to the primary basis, and additionally checks that
/// the auxiliary basis (if any) is entirely spherical (rule 3).
pub fn audit_basis_with_aux(primary: &BasisSet, aux: Option<&BasisSet>) -> BasisAudit {
    let mut audit = audit_basis(primary);
    if let Some(aux_bs) = aux {
        for shells in aux_bs.shells.values() {
            for sh in shells {
                if !sh.pure {
                    audit.violations.push(BasisViolation {
                        kind: ViolationKind::AuxNotSpherical,
                        detail: format!(
                            "aux basis {:?}: Cartesian (pure=false) shell with l={}",
                            aux_bs.name, sh.l
                        ),
                    });
                    break;
                }
            }
        }
    }
    audit
}
