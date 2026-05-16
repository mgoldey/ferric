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

pub const MAX_COMPONENTS: usize = 8;

/// A two-electron operator with its parameters.
#[derive(Debug, Clone, Copy)]
pub struct Operator {
    pub kind: OperatorKind,
    pub omega: f64,
    pub distance: f64,
    // Composite fields
    pub is_composite: bool,
    pub num_components: usize,
    pub c_coeffs: [f64; MAX_COMPONENTS],
    pub c_omegas: [f64; MAX_COMPONENTS],
    pub c_kinds: [OperatorKind; MAX_COMPONENTS],
}

impl Operator {
    /// Create a primitive operator.
    pub fn primitive(kind: OperatorKind, omega: f64, distance: f64) -> Self {
        Self {
            kind,
            omega,
            distance,
            is_composite: false,
            num_components: 0,
            c_coeffs: [0.0; MAX_COMPONENTS],
            c_omegas: [0.0; MAX_COMPONENTS],
            c_kinds: [OperatorKind::Coulomb; MAX_COMPONENTS],
        }
    }

    /// Create a composite operator as a linear combination of primitive operators.
    pub fn composite(components: &[(f64, OperatorKind, f64)]) -> Self {
        let mut op = Self::primitive(OperatorKind::Coulomb, 0.0, 0.0);
        op.is_composite = true;
        op.num_components = components.len().min(MAX_COMPONENTS);
        for i in 0..op.num_components {
            let (coeff, kind, omega) = components[i];
            op.c_coeffs[i] = coeff;
            op.c_kinds[i] = kind;
            op.c_omegas[i] = omega;
        }
        op
    }

    /// Standard Coulomb operator (1/r12).
    pub fn coulomb() -> Self {
        Self::primitive(OperatorKind::Coulomb, 0.0, 0.0)
    }

    /// Long-range attenuated Coulomb: erf(omega * r12) / r12.
    pub fn erf(omega: f64) -> Self {
        Self::primitive(OperatorKind::ErfCoulomb, omega, 0.0)
    }

    /// Short-range attenuated Coulomb: erfc(omega * r12) / r12.
    pub fn erfc(omega: f64) -> Self {
        Self::primitive(OperatorKind::ErfcCoulomb, omega, 0.0)
    }

    /// Spike implementation: Approximate terfc(r, r0) as a sum of 3 erfc operators.
    /// These are dummy coefficients to test the composite engine architecture.
    pub fn terfc_fit(r0: f64) -> Self {
        let base_omega = 1.0 / (r0 * std::f64::consts::SQRT_2);
        let mut op = Self::composite(&[
            ( 1.5, OperatorKind::ErfcCoulomb, base_omega * 0.8),
            (-0.6, OperatorKind::ErfcCoulomb, base_omega * 1.2),
            ( 0.1, OperatorKind::ErfcCoulomb, base_omega * 2.0),
        ]);
        op.distance = r0;
        op
    }
}
