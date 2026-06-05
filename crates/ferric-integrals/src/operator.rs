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
    /// Contracted Gaussian geminal f12 ≈ Slater geminal -exp(-gamma r12)/gamma.
    Cgtg,
    /// f12 / r12 (geminal times Coulomb).
    CgtgCoulomb,
    /// [Ti, f12] kinetic commutator integrand |∇f12|^2 (delcgtg2).
    Delcgtg2,
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
        // i indexes three parallel fixed-size arrays; clearer than a zip chain.
        #[allow(clippy::needless_range_loop)]
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

    /// Slater-type geminal f12 = -exp(-gamma·r12)/gamma, represented as the
    /// standard 6-term fit by a sum of Gaussians, exp(-gamma r) ≈ Σ cᵢ exp(-αᵢ r²).
    ///
    /// Coefficients/exponents are the Tew–Klopper (2005) least-squares fit of
    /// exp(-r) on [0,10] a.u.; scaling r→gamma·r maps the exponents by gamma².
    /// The −1/gamma prefactor of the Slater geminal is folded into the
    /// coefficients so the carried geminal IS f12 (not exp(-gamma r)).
    ///
    /// The fit is carried in the composite arrays: `c_omegas[i]` = Gaussian
    /// exponent αᵢ, `c_coeffs[i]` = coefficient cᵢ. `kind` selects which geminal
    /// operator (Cgtg / CgtgCoulomb / Delcgtg2) the same fit feeds.
    pub fn stg(gamma: f64, kind: OperatorKind) -> Self {
        // Tew–Klopper 6-Gaussian fit of exp(-r): (exponent αᵢ, coeff cᵢ).
        const FIT: [(f64, f64); 6] = [
            (0.241_393, 0.301_846),
            (0.844_001, 0.255_338),
            (3.044_055, 0.197_575),
            (13.499_604, 0.139_390),
            (76.617_811, 0.082_572),
            (765.962_887, 0.034_801),
        ];
        let mut op = Self::primitive(kind, gamma, 0.0);
        op.is_composite = true;
        op.num_components = FIT.len();
        let g2 = gamma * gamma;
        let pref = -1.0 / gamma; // f12 = -(1/gamma) exp(-gamma r12)
        for (i, (alpha, c)) in FIT.iter().enumerate() {
            op.c_omegas[i] = alpha * g2; // r→gamma·r scales the Gaussian exponent
            op.c_coeffs[i] = pref * c;
            op.c_kinds[i] = kind;
        }
        op
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
