//! Two-electron operator definitions for integral evaluation.

/// The mathematical form of the two-electron operator kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OperatorKind {
    /// Standard 1/r12 Coulomb operator.
    Coulomb,
    /// erf(omega * r12) / r12 -- long-range attenuated Coulomb.
    ErfCoulomb,
    /// erfc(omega * r12) / r12 -- short-range attenuated Coulomb.
    ErfcCoulomb,
    /// exp(-omega * r12) / r12 -- Yukawa / screened Coulomb.
    Yukawa,
}

/// A two-electron operator with its parameters.
#[derive(Debug, Clone, Copy)]
pub struct Operator {
    pub kind: OperatorKind,
    pub omega: f64,
    pub distance: f64,
}

impl Operator {
    /// Standard Coulomb operator (1/r12).
    pub fn coulomb() -> Self {
        Self { kind: OperatorKind::Coulomb, omega: 0.0, distance: 0.0 }
    }

    /// Long-range attenuated Coulomb: erf(omega * r12) / r12.
    pub fn erf(omega: f64) -> Self {
        Self { kind: OperatorKind::ErfCoulomb, omega, distance: 0.0 }
    }
}
