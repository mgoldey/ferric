//! cuEST compatibility audit for ferric basis sets.
//!
//! cuEST imposes three structural rules on a Gaussian basis (see the plan,
//! §1.1 and §5.2/§5.3):
//!
//! 1. **Angular-momentum contiguity**: the set of angular momenta `l` present
//!    in the basis must be contiguous starting at `l = 0` (no holes such as a
//!    basis that has `s` and `d` but no `p`).
//! 2. **Uniform representation**: every shell in the basis must use the same
//!    representation (all spherical `pure = true`, or all Cartesian
//!    `pure = false`). Mixed spherical/Cartesian within one molecule is not
//!    supported.
//! 3. **Spherical aux basis**: a density-fitting auxiliary basis must be
//!    entirely spherical (`pure = true`).
//!
//! These checks are pure Rust and require no device, so they can run on the
//! dev box. They are the CPU-side half of Phase 0.4.

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
    /// The set of angular momenta present has a gap below its maximum.
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

    // Rule 1: angular-momentum contiguity from 0.
    // Collect the set of l values present anywhere in the basis.
    let mut present: std::collections::BTreeSet<i32> = std::collections::BTreeSet::new();
    for shells in bs.shells.values() {
        for sh in shells {
            present.insert(sh.l);
        }
    }
    if let Some(&max_l) = present.iter().next_back() {
        for l in 0..=max_l {
            if !present.contains(&l) {
                // Report which elements actually carry the surrounding momenta
                // for diagnostics.
                violations.push(BasisViolation {
                    kind: ViolationKind::LContiguityHole,
                    detail: format!(
                        "basis {:?}: missing angular momentum l={} (present: {:?})",
                        bs.name,
                        l,
                        present.iter().copied().collect::<Vec<_>>()
                    ),
                });
            }
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
