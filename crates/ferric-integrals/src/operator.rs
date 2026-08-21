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
    /// terfc(r12, r0) / r12 -- exact "tempered" short-range Coulomb via 2D
    /// interpolation tables (Dutoi/Goldey). `distance` carries r0, `omega` the
    /// curvature-constrained 1/(r0*sqrt2). Handled by the standalone table engine,
    /// not libint2.
    Terfc,
    /// terf(r12, r0) / r12 -- exact "tempered" LONG-range complement of Terfc:
    /// terf + terfc = Coulomb (identity, exact to machine precision -- both
    /// share the identical table lookup / OS recurrence / cart->pure
    /// transform in the shim; only the final combine step differs). Same
    /// `distance`/`omega` (r0, curvature-constrained 1/(r0*sqrt2)) convention
    /// as Terfc. Handled by the standalone table engine, not libint2.
    Terf,
    /// exp(-omega * r12) / r12 -- Yukawa / screened Coulomb. `omega` carries the
    /// decay parameter ζ (zeta). Evaluated natively by libint2's
    /// `Operator::stg_x_coulomb` (aliased `yukawa` in libint2, TennoGmEval core),
    /// NOT via a Gaussian fit -- this is the exact screened-Coulomb kernel.
    Yukawa,
    /// exp(-zeta * r12) -- EXACT Slater-type geminal (STG), evaluated natively by
    /// libint2's `Operator::stg` (TennoGmEval core). `omega` carries the decay
    /// parameter ζ. This is the UNFITTED counterpart of the 6-Gaussian
    /// [`OperatorKind::Cgtg`] fit: `Cgtg` approximates exp(-γr) by a sum of
    /// Gaussians (Tew–Klopper), `SlaterGeminal` is the exact kernel. Note the
    /// sign/prefactor convention: `SlaterGeminal` carries the bare +exp(-ζr)
    /// (matching libint2's `stg`), whereas `Cgtg` folds the −1/γ prefactor of the
    /// F12 geminal into its fit coefficients.
    SlaterGeminal,
    /// Contracted Gaussian geminal f12 ≈ Slater geminal -exp(-gamma r12)/gamma.
    Cgtg,
    /// f12 / r12 (geminal times Coulomb).
    CgtgCoulomb,
    /// [Ti, f12] kinetic commutator integrand |∇f12|^2 (delcgtg2).
    Delcgtg2,
}

/// Maximum number of integral components per shell quartet (rank-2 multipoles: 1+3+6 = 10, but only 8 used for derivatives of up to second order).
pub const MAX_COMPONENTS: usize = 8;

/// A two-electron operator with its parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Operator {
    pub kind: OperatorKind,
    pub omega: f64,
    pub distance: f64,
    // Composite fields — crate-internal; use `components()` for external access.
    pub(crate) is_composite: bool,
    pub(crate) num_components: usize,
    pub(crate) c_coeffs: [f64; MAX_COMPONENTS],
    pub(crate) c_omegas: [f64; MAX_COMPONENTS],
    pub(crate) c_kinds: [OperatorKind; MAX_COMPONENTS],
}

impl Operator {
    /// Whether this operator is a linear combination of primitives.
    pub fn is_composite(&self) -> bool {
        self.is_composite
    }

    /// Iterate over the (coefficient, kind, omega) triples of a composite operator.
    /// Returns an empty iterator for primitive operators.
    pub fn components(&self) -> impl Iterator<Item = (f64, OperatorKind, f64)> + '_ {
        let n = if self.is_composite { self.num_components } else { 0 };
        (0..n).map(move |i| (self.c_coeffs[i], self.c_kinds[i], self.c_omegas[i]))
    }

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

    /// Yukawa / screened Coulomb: exp(-zeta * r12) / r12.
    ///
    /// Evaluated by libint2's native `Operator::stg_x_coulomb` kernel (aliased
    /// `yukawa` in libint2, TennoGmEval core) -- the EXACT screened Coulomb, not
    /// a Gaussian fit. `zeta` is the decay constant (Bohr⁻¹), analogous to
    /// erf/erfc's `omega`.
    ///
    /// Limit: zeta -> 0 gives exp(0)/r = 1/r = Coulomb (approached from below).
    ///
    /// SUPPORTED RANGE: `zeta > 0`. libint2's TennoGmEval interpolates only for
    /// `U = zeta²/(4ρ) ∈ [1e-7, 1e3]` (ρ = reduced bra/ket exponent); outside it
    /// falls back to an upward recursion that is undefined (and, in a
    /// debug/`assert`-enabled libint2 build, ABORTS the process) on the
    /// same-center `T = 0` shell blocks that RI/3-center paths always contain.
    /// In practice a physically meaningful `zeta` (≈ 0.1 – a few Bohr⁻¹) stays in
    /// domain for ordinary Gaussian exponents. Do NOT pass a near-zero `zeta`
    /// (e.g. 1e-6) expecting a clean Coulomb limit on RI paths -- that regime is
    /// out of the kernel's table domain.
    pub fn yukawa(zeta: f64) -> Self {
        debug_assert!(zeta > 0.0, "Yukawa decay parameter zeta must be > 0 (got {zeta})");
        Self::primitive(OperatorKind::Yukawa, zeta, 0.0)
    }

    /// Exact (unfitted) Slater-type geminal: exp(-zeta * r12).
    ///
    /// Evaluated by libint2's native `Operator::stg` kernel (TennoGmEval core) --
    /// NOT to be confused with ferric's own [`Operator::stg`] function, which
    /// instead builds the 6-Gaussian Tew–Klopper FIT of exp(-γr) for the
    /// `Cgtg`/`CgtgCoulomb`/`Delcgtg2` geminal-composite engines. This function
    /// ([`Operator::slater_geminal`]) is the UNFITTED counterpart of that fit.
    /// Use this when the fit error of the 6-term expansion is not acceptable.
    ///
    /// Sign convention: carries the bare +exp(-ζr) (as libint2's native `stg`
    /// kernel returns), NOT the −(1/γ)exp(-γr) F12-geminal form that ferric's
    /// [`Operator::stg`] folds into its fit coefficients.
    ///
    /// SUPPORTED RANGE: `zeta > 0` (strictly). libint2's `eval_slater` requires
    /// `U = zeta²/(4ρ) > 0` (the integral factorizes into two overlaps at U = 0);
    /// the same TennoGmEval table-domain caveat as [`Operator::yukawa`] applies.
    pub fn slater_geminal(zeta: f64) -> Self {
        debug_assert!(zeta > 0.0, "Slater geminal zeta must be > 0 (got {zeta})");
        Self::primitive(OperatorKind::SlaterGeminal, zeta, 0.0)
    }

    /// Exact tempered short-range Coulomb terfc(r12, r0) / r12, evaluated via the
    /// Dutoi/Goldey 2D interpolation tables. `omega` is fixed by the curvature
    /// constraint r0 * omega = 1/sqrt(2); `distance` carries r0 (Bohr).
    pub fn terfc(r0: f64) -> Self {
        let omega = 1.0 / (r0 * std::f64::consts::SQRT_2);
        Self::primitive(OperatorKind::Terfc, omega, r0)
    }

    /// Exact tempered LONG-range Coulomb complement terf(r12, r0) / r12 = Coulomb
    /// − terfc(r12, r0) / r12, evaluated via the same Dutoi/Goldey 2D
    /// interpolation tables as [`Operator::terfc`]. Same curvature constraint
    /// r0 * omega = 1/sqrt(2); `distance` carries r0 (Bohr).
    ///
    /// Limits (matching erf/erfc exactly): r0 -> inf (omega -> 0): terf -> 0.
    /// r0 -> 0 (omega -> inf): terf -> Coulomb.
    pub fn terf(r0: f64) -> Self {
        let omega = 1.0 / (r0 * std::f64::consts::SQRT_2);
        Self::primitive(OperatorKind::Terf, omega, r0)
    }

    /// terfc(r12; r0, omega)/r12 with INDEPENDENT sharpness — the curvature
    /// constraint is NOT applied. Basis for decoupling: with the complement
    /// computed by a LR method, Dutoi's constraint does not bind (measured,
    /// scripts/ne2_seam_test.py); the table engine is already general in the
    /// reduced variables (S, s) — omega enters only via
    /// phi = (1/p+1/q+1/omega^2)^{-1/2}, s = (phi*r0)^2 <= (omega*r0)^2, and
    /// the shipped tables cover s <= 80 (r0*omega <= ~8.9). terf + terfc =
    /// Coulomb holds exactly for EVERY (r0, omega) — anchor-tested.
    pub fn terfc_with_omega(r0: f64, omega: f64) -> Self {
        Self::primitive(OperatorKind::Terfc, omega, r0)
    }

    /// Long-range complement of [`Operator::terfc_with_omega`]; same free
    /// (r0, omega) parameterization, same tables.
    pub fn terf_with_omega(r0: f64, omega: f64) -> Self {
        Self::primitive(OperatorKind::Terf, omega, r0)
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
