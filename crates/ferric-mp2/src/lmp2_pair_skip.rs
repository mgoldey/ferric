//! Provable-empty pair skip for the integral-direct amplitude-threshold LMP2
//! (`crate::lmp2_direct`, stage 5). Book: chapter 13, "Proving a pair
//! empty" (`book/src/13-local-mp2/02-integral-direct.md`); sympy proof
//! `scripts/lmp2_attenuated_pair_skip_sympy.py`.
//!
//! Stage 5 fits pair (i,j) on `D = D_i ∪ D_j` as `J_ab = aᵀ V⁻¹ b`, with
//! `a = B_i[D, a]`, `b = B_j[D, b]` the COMPUTED strip columns (AO tails and
//! Schwarz-skipped triples included: nothing below treats a strip as the
//! integral of a function) and `V = V_DD` the gathered same-kernel metric.
//! Split `D = A ∪ B`, `A = D_i`, `B = D_j \ D_i`. With `γ = ‖V_AA^{-1/2}
//! V_AB V_BB^{-1/2}‖ < 1`, the block-Neumann split of `(I + X)⁻¹` gives
//! [`fitted_bound`]:
//!
//! ```text
//! |J_ab| ≤ uA vA + uB vB + γ (uA vB + uB vA) + γ²/(1−γ) ‖u‖ ‖v‖
//! ```
//!
//! where every factor is an upper bound computed from data stage 5 already
//! holds, without the pair's own Cholesky:
//!
//! * `uA = max_a ‖V_{D_i}^{-1/2} a_{D_i}‖`, i's own-domain fitted norm;
//! * `vB = max_b ‖V_{D_j}^{-1/2} b_{D_j}‖ ≥ ‖V_BB^{-1/2} b_B‖` (Schur
//!   monotonicity, `B ⊆ D_j`);
//! * `uB = ‖a_B‖ / √λ_j`, `vA = ‖b_A‖ / √λ_i`, from the raw strip rows of
//!   the OTHER occupied's domain;
//! * `γ ≤ ‖V_AB‖_F / √(λ_i λ_j)`;
//! * `λ_k ≤ λ_min(V_{D_k D_k})` ([`eigen_floor`], [`verified_floor`]), which by Cauchy
//!   interlacing also floors every principal sub-block (`A = D_i`, `B ⊆ D_j`).
//!
//! The bound needs a positive-definite kernel (the fit is then a projection
//! and every metric block is SPD). It holds for Coulomb too, but there the
//! metric coupling `V_AB` decays as a power of the distance and `γ` never
//! drops below 1 at chemical sizes; it only reaches pairs for an attenuated
//! kernel, whose metric tail is Gaussian. [`PairSkip::Auto`] therefore turns
//! it on for `erfc` and `terfc` only.
//!
//! The retention test reads `max(|g|, |g̃|/κ) > ε` with `g = scale·J` and the
//! optional Fock-dressed `g̃` ([`crate::ragged::fock_dressed_gate`]); a pair
//! is skipped iff [`proved_empty`] shows every element of both is `≤ ε`.
//! The skip never changes an energy: a skipped pair is one whose fitted
//! blocks would all be masked away, and the groups' whitening is unchanged.

use ferric_integrals::operator::{Operator, OperatorKind};
use ndarray::Array2;

use crate::ragged::{fock_bound_keeps, fock_bound_weights};

/// The pair-skip switch of the integral-direct path
/// ([`crate::lmp2_direct::DirectConfig::pair_skip`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PairSkip {
    /// Fit every pair (the anchor limit).
    #[default]
    Off,
    /// [`PairSkip::On`] for an attenuated positive-definite kernel (`erfc`,
    /// `terfc`), [`PairSkip::Off`] otherwise. The production default.
    Auto,
    /// Skip every pair the bound proves empty, whatever the kernel (it must be
    /// positive definite: Coulomb, erf, erfc or terfc).
    On,
    /// Evaluate the bound on every pair but fit them all, and record the
    /// largest ratio of the fitted retention quantities to their bounds
    /// (`DirectStats::pair_skip_audit_*`). Never changes a result.
    Audit,
}

impl PairSkip {
    /// Strict parser for the user-facing spelling: `"auto"`, `"on"`, `"off"`
    /// or `"audit"`.
    pub fn parse_config_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "on" => Ok(Self::On),
            "off" => Ok(Self::Off),
            "audit" => Ok(Self::Audit),
            other => Err(format!(
                "unknown pair_skip {other:?}: expected \"auto\", \"on\", \"off\" or \"audit\""
            )),
        }
    }

    /// The user-facing spelling ([`Self::parse_config_str`] round-trips it).
    pub fn as_config_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::On => "on",
            Self::Audit => "audit",
        }
    }

    /// The mode a run with operator `op` uses: `Auto` resolves to `On` for
    /// [`is_attenuated_pd`] kernels and `Off` otherwise. `On` and `Audit`
    /// with a kernel that is not positive definite (terf, a composite, a
    /// geminal) are an error: the bound does not hold there.
    pub fn resolve(self, op: &Operator) -> Result<Self, String> {
        match self {
            Self::Off => Ok(Self::Off),
            Self::Auto if is_attenuated_pd(op) => Ok(Self::On),
            Self::Auto => Ok(Self::Off),
            Self::On | Self::Audit if is_pd_kernel(op) => Ok(self),
            Self::On | Self::Audit => Err(format!(
                "pair_skip = \"{}\" needs a positive-definite kernel (Coulomb, erf, erfc, \
                 terfc); operator {:?} is not one",
                self.as_config_str(),
                op.kind
            )),
        }
    }

    /// Whether the bound is evaluated (`On` or `Audit`, after [`Self::resolve`]).
    pub fn evaluates(self) -> bool {
        matches!(self, Self::On | Self::Audit)
    }
}

/// A short-range attenuated kernel the skip is on for by default: `erfc` or
/// `terfc`, not composite.
pub fn is_attenuated_pd(op: &Operator) -> bool {
    !op.is_composite() && matches!(op.kind, OperatorKind::ErfcCoulomb | OperatorKind::Terfc)
}

/// A positive-definite kernel the bound holds for (its Fourier transform is
/// positive): Coulomb, erf, erfc, terfc, not composite.
pub fn is_pd_kernel(op: &Operator) -> bool {
    !op.is_composite()
        && matches!(
            op.kind,
            OperatorKind::Coulomb
                | OperatorKind::ErfCoulomb
                | OperatorKind::ErfcCoulomb
                | OperatorKind::Terfc
        )
}

/// `γ_n = n·u/(1 − n·u)` (Higham, *Accuracy and Stability of Numerical
/// Algorithms*, 2nd ed., §3.1), `u = 2⁻⁵³` the unit roundoff; `∞` when
/// `n·u ≥ 1`. Every rounding allowance of the pair skip is built from it,
/// worst case and to all orders: Cholesky `L̂L̂ᵀ = V + ΔV`, `|ΔV| ≤
/// γ_{d+1}|L̂||L̂ᵀ|` (Thm 10.3); each triangular-solve column `(L̂ + δL)x̂ =
/// a`, `|δL| ≤ γ_d|L̂|` (Thm 8.5); each dot product of length `n`,
/// `|fl(xᵀy) − xᵀy| ≤ γ_n|x|ᵀ|y|` (§3.1). The level-3 BLAS the fit calls
/// are taken at `γ_{2d}` (twice the classical constant, for blocked
/// kernels; Strassen-like products are not used by OpenBLAS's dgemm/dtrsm).
pub fn gamma_n(n: usize) -> f64 {
    let nu = n as f64 * f64::EPSILON / 2.0;
    if nu < 1.0 {
        nu / (1.0 - nu)
    } else {
        f64::INFINITY
    }
}

/// `η ≥ ‖ΔV‖₂` for the computed Cholesky factor of a `d × d` SPD block whose
/// trace is at most `trace`: `‖ΔV‖₂ ≤ γ_{d+1}‖|L̂||L̂ᵀ|‖₂ ≤ γ_{d+1}‖L̂‖_F²`
/// and `‖L̂‖_F² = tr(V + ΔV) ≤ trace + γ_{d+1}‖L̂‖_F²`, so `η =
/// γ_{d+1}/(1 − γ_{d+1})·trace`.
pub fn chol_eta(d: usize, trace: f64) -> f64 {
    let g = gamma_n(d + 1);
    g / (1.0 - g) * trace
}

/// `θ ≥ ‖L̂⁻¹ δL‖₂` for any triangular-solve column of a `d × d` factor,
/// given `‖L̂⁻¹‖₂² ≤ inv2`: `‖δL‖₂ ≤ γ_{2d}‖L̂‖_F ≤ γ_{2d}√(trace/(1 −
/// γ_{d+1}))`. Then a computed column `x̂` and the exact `L̂⁻¹a` obey
/// `‖x̂ − L̂⁻¹a‖ ≤ ρ‖L̂⁻¹a‖`, `ρ = θ/(1 − θ)`. `None` unless `θ ≤ 1/4`.
pub fn solve_theta(d: usize, trace: f64, inv2: f64) -> Option<f64> {
    let g = gamma_n(d + 1);
    let theta = gamma_n(2 * d) * (trace / (1.0 - g) * inv2).sqrt();
    (theta.is_finite() && theta <= 0.25).then_some(theta)
}

/// `ρ = θ/(1 − θ)`.
fn rho_of(theta: f64) -> f64 {
    theta / (1.0 - theta)
}

/// The five factors of [`fitted_bound`] (unscaled, see the module doc).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairNorms {
    /// `uA ≥ max_a ‖α_A‖`: i's own-domain fitted norm.
    pub u_a: f64,
    /// `uB ≥ max_a ‖α_B‖`: i's strip on `B`, over `√λ_j`.
    pub u_b: f64,
    /// `vA ≥ max_b ‖β_A‖`: j's strip on `A`, over `√λ_i`.
    pub v_a: f64,
    /// `vB ≥ max_b ‖β_B‖`: j's own-domain fitted norm.
    pub v_b: f64,
    /// `γ ≥ ‖V_AA^{-1/2} V_AB V_BB^{-1/2}‖₂`.
    pub gamma: f64,
}

/// Upper bound on `max_ab |aᵀ V_DD⁻¹ b|` (claim 4 of the sympy proof), or
/// `None` when `γ ≥ 1` (or any factor is not finite): the split then has no
/// convergent remainder and nothing is proved.
pub fn fitted_bound(n: &PairNorms) -> Option<f64> {
    let PairNorms {
        u_a,
        u_b,
        v_a,
        v_b,
        gamma,
    } = *n;
    let finite = [u_a, u_b, v_a, v_b, gamma].iter().all(|x| x.is_finite());
    if !finite || !(gamma < 1.0) {
        return None;
    }
    let rem = gamma * gamma / (1.0 - gamma) * u_a.hypot(u_b) * v_a.hypot(v_b);
    Some(u_a * v_a + u_b * v_b + gamma * (u_a * v_b + u_b * v_a) + rem)
}

/// The floating-point allowance on the COMPUTED `J` of a `d`-function union
/// domain, worst case and to all orders. `trace` bounds `tr V_DD` from
/// above and `lam_min` floors both own-domain blocks, so `λ_D = (1 − γ)·
/// lam_min ≤ λ_min(V_DD)`. With `N_a = ‖V^{-1/2}a‖ ≤ ‖u‖/√(1−γ)` (likewise
/// `N_b`), `δ = η/λ_D` ([`chol_eta`]), `θ` from [`solve_theta`] with
/// `‖L̂⁻¹‖₂² ≤ 1/(λ_D − η)` and `ρ = θ/(1 − θ)`:
///
/// ```text
/// |Ĵ − aᵀV⁻¹b| ≤ N_a N_b [δ + (1 + ρ)²(1 + γ_{2d}) − 1] / (1 − δ)
/// ```
///
/// (Cholesky: `|aᵀ(V+ΔV)⁻¹b − aᵀV⁻¹b| ≤ δ/(1−δ)·N_aN_b`, `‖L̂⁻¹a‖ ≤
/// N_a/√(1−δ)`; the solves move each column by `ρ`; the product adds
/// `γ_{2d}‖x̂‖‖ŷ‖`). `∞` when `δ ≥ 1/2` or `θ > 1/4`: nothing is proved.
pub fn roundoff_allowance(d: usize, trace: f64, lam_min: f64, n: &PairNorms) -> f64 {
    let lam_d = (1.0 - n.gamma) * lam_min;
    let eta = chol_eta(d, trace);
    let delta = eta / lam_d;
    if !(delta.is_finite() && delta < 0.5) {
        return f64::INFINITY;
    }
    let Some(theta) = solve_theta(d, trace, 1.0 / (lam_d - eta)) else {
        return f64::INFINITY;
    };
    let r = rho_of(theta);
    let na_nb = n.u_a.hypot(n.u_b) * n.v_a.hypot(n.v_b) / (1.0 - n.gamma);
    na_nb * (delta + (1.0 + r) * (1.0 + r) * (1.0 + gamma_n(2 * d)) - 1.0) / (1.0 - delta)
}

/// A rigorous floor `λ ≤ λ_min(V)` of a `d × d` SPD block (trace at most
/// `trace`) from the computed inverse factor `Ẑ ≈ L̂⁻¹` with `‖Ẑ‖_F² =
/// linv_fro2` (as computed). Each column of `Ẑ` is a triangular solve, so
/// `‖L̂⁻¹e_k‖ ≤ ‖ẑ_k‖/(1 − ρ)`, and `tr((V + ΔV)⁻¹) = ‖L̂⁻¹‖_F² ≥ 1/λ_min(V
/// + ΔV) ≥ 1/(λ_min(V) + η)`; hence `λ_min(V) ≥ (1 − ρ)²/((1 + γ_{d²})
/// linv_fro2) − η` (the factor `1 + γ_{d²}` covers the rounded sum of
/// squares). `θ` uses `‖L̂⁻¹‖₂ ≤ ‖L̂⁻¹‖_F ≤ ‖Ẑ‖_F/(1 − ρ)`, which is
/// self-consistent while `θ ≤ 1/4` (a continuity argument from `δL = 0`).
/// `None` when that fails or the floor is not positive.
pub fn eigen_floor(linv_fro2: f64, trace: f64, d: usize) -> Option<f64> {
    if !(linv_fro2.is_finite() && linv_fro2 > 0.0 && trace.is_finite()) {
        return None;
    }
    let f2 = linv_fro2 * (1.0 + gamma_n(d * d + 1));
    // ‖L̂⁻¹‖_F² ≤ f2/(1−ρ)² and 1/(1−ρ) ≤ 3/2 at θ ≤ 1/4: bound inv2 by
    // (9/4)·f2, then θ from it (≤ 1/4 keeps that a priori bound valid)
    let theta = solve_theta(d, trace, 2.25 * f2)?;
    let r = rho_of(theta);
    let lam = (1.0 - r) * (1.0 - r) / f2 - chol_eta(d, trace);
    (lam.is_finite() && lam > 0.0).then_some(lam)
}

/// The diagonal shift `c` of Rump's positive-definiteness test (S. M. Rump,
/// "Verification of positive definiteness", BIT 46 (2006) 433, Thm 2.3)
/// for a `d × d` symmetric matrix of trace `trace` and largest diagonal
/// `diag_max`: if the floating-point Cholesky of `fl(M − cI)` runs to
/// completion, `M` is positive definite. `c = γ_{d+1}/(1 − γ_{d+1})·trace`
/// plus the rounding `u·diag_max` of forming the shifted diagonal, the whole
/// doubled for room (the underflow term of the theorem is far below either).
pub fn rump_shift(d: usize, trace: f64, diag_max: f64) -> f64 {
    let u = f64::EPSILON / 2.0;
    2.0 * (chol_eta(d, trace) + 4.0 * u * diag_max)
}

/// A floor `σ ≤ λ_min(V)` of an SPD block, PROVED by a shifted Cholesky:
/// estimate `λ_min` from above by power iteration on `V⁻¹ = L⁻ᵀL⁻¹`
/// (`linv = L⁻¹`, so each step is two matrix-vector products), try
/// `σ = λ_est/2`, then `λ_est/10`, and accept the first `σ` for which the
/// Cholesky of `V − (σ + c)I` ([`rump_shift`]) succeeds: then `V − σI` is
/// positive definite. `None` when neither shift verifies. Within a factor 2
/// to 10 of `λ_min` when it succeeds, against the factor of up to `d` of the
/// trace floor ([`eigen_floor`]).
pub fn verified_floor(v: &Array2<f64>, linv: &Array2<f64>) -> Option<f64> {
    use ndarray_linalg::{Cholesky, UPLO};
    let d = v.nrows();
    if d == 0 {
        return None;
    }
    // power iteration on V^-1 from a deterministic start
    let mut x = ndarray::Array1::from_shape_fn(d, |k| 1.0 + (k % 7) as f64 * 0.1);
    let mut mu = 0.0f64;
    for _ in 0..40 {
        let y = linv.t().dot(&linv.dot(&x));
        let ny = y.dot(&y).sqrt();
        if !(ny.is_finite() && ny > 0.0) {
            return None;
        }
        mu = x.dot(&y) / x.dot(&x);
        x = y / ny;
    }
    let lam_est = 1.0 / mu; // Rayleigh quotient of V^-1 <= its top eigenvalue
    if !(lam_est.is_finite() && lam_est > 0.0) {
        return None;
    }
    let trace: f64 = v.diag().sum();
    let diag_max = v.diag().fold(0.0f64, |m, &x| m.max(x));
    let c = rump_shift(d, trace, diag_max);
    [0.5, 0.1].into_iter().map(|f| f * lam_est).find(|&sigma| {
        let mut m = v.clone();
        m.diag_mut().mapv_inplace(|x| x - (sigma + c));
        m.cholesky(UPLO::Lower).is_ok()
    })
}

/// The factor that turns a COMPUTED own-domain fitted norm `fl(‖x̂‖)`, `x̂`
/// a triangular-solve column of the computed factor, into an upper bound
/// on the exact `‖V^{-1/2}a‖`, and a computed column-block norm
/// `fl(‖Ẑ[:, P]‖_F)` into one on `‖L̂⁻¹[:, P]‖_F` times the same
/// `√(1+δ)` (what the shell-wise bound needs, since `‖V^{-1/2}x‖ ≤
/// √(1+δ)‖L̂⁻¹x‖`). `V ⪰ λI` and `ΔV ⪯ ηI` give `V + ΔV ⪯ (1 + δ)V`,
/// `δ = η/λ`, so `aᵀV⁻¹a ≤ (1 + δ)‖L̂⁻¹a‖²`; the solve gives `‖L̂⁻¹a‖ ≤
/// ‖x̂‖/(1 − ρ)`; `1 + γ_{d²+2}` covers the rounded norms. `lam` must be a
/// proved floor of `λ_min(V)` and `linv_fro2` the computed `‖Ẑ‖_F²`.
/// `None` when [`solve_theta`] refuses.
pub fn norm_inflation(linv_fro2: f64, trace: f64, lam: f64, d: usize) -> Option<f64> {
    let f2 = linv_fro2 * (1.0 + gamma_n(d * d + 1));
    let r = rho_of(solve_theta(d, trace, 2.25 * f2)?);
    let delta = chol_eta(d, trace) / lam;
    let q = (1.0 + delta).sqrt() * (1.0 + gamma_n(d * d + 2)) / (1.0 - r);
    q.is_finite().then_some(q)
}

/// The Fock-dressed gate's worst-case weight `w_max = max_a Σ_{c≠a}
/// max(|F_ac|, |F_ca|)/D_c` over the FULL virtual space: the
/// [`fock_bound_weights`] of a unit Schwarz factor. `None` when some
/// `D_c ≤ 0` (no dressed bound).
pub fn unit_fock_weight(f_vv: &Array2<f64>, fo: &[f64], fv: &[f64]) -> Option<f64> {
    let ones = vec![1.0; fv.len()];
    fock_bound_weights(&ones, 1, f_vv, fo, fv).map(|(_, wmax)| wmax[0])
}

/// Whether a pair whose every fitted element obeys `|g_ab| ≤ g_max` (scaled)
/// is proved empty at threshold `eps`. With the Fock screen on
/// (`fock = Some((κ, w_max))`) the dressed gate is bounded by
/// [`fock_bound_keeps`] on a constant Schwarz factor `s = √g_max`
/// (`q_ia q_jb = g_max ≥ |g_ab|`, weights `s·w_max`): that bound is
/// `g_max (1 + 2 w_max)`, the same for every virtual, so one call with one
/// virtual decides the whole pair. The mask keeps `> eps`, so a proof is
/// `keeps == false`.
pub fn proved_empty(g_max: f64, eps: f64, fock: Option<(f64, f64)>) -> bool {
    if !(g_max.is_finite() && eps > 0.0) {
        return false;
    }
    let s = g_max.sqrt();
    let (kappa, wmax) = fock.unwrap_or((f64::INFINITY, 0.0));
    let q = [s, s];
    let w = [s * wmax, s * wmax];
    // fock_bound_keeps tests `>= eps`; the mask keeps `> eps`, so `false`
    // here is a proof (and a stricter one)
    !fock_bound_keeps(&q, &q, &w, &w, 1, 0, 1, 0, eps, kappa)
}

/// The worst retention quantity the dressed gate can see given the bound
/// `g_max` (what [`proved_empty`] compares with `eps`): `g_max` when the
/// Fock screen is off, `g_max · max(1, (1 + 2 w_max)/κ)` when it is on.
pub fn gate_bound(g_max: f64, fock: Option<(f64, f64)>) -> f64 {
    match fock {
        None => g_max,
        Some((kappa, wmax)) => g_max * (1.0f64).max((1.0 + 2.0 * wmax) / kappa),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{s, Array1};
    use ndarray_linalg::{Cholesky, Inverse, Norm, SVD, UPLO};

    /// A deterministic SPD matrix of condition about `cond`, `n × n`, with
    /// the A-B coupling (first `na` vs rest) scaled by `coupling`.
    fn spd(n: usize, na: usize, cond: f64, coupling: f64, seed: u64) -> Array2<f64> {
        let mut x = seed;
        let mut rnd = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let g = Array2::from_shape_fn((n, n), |_| rnd());
        let (q, _) = ndarray_linalg::QR::qr(&g).unwrap();
        let ev = Array1::from_shape_fn(n, |k| cond.powf(-(k as f64) / (n - 1) as f64));
        let mut v = q.dot(&Array2::from_diag(&ev)).dot(&q.t());
        for r in 0..n {
            for c in 0..n {
                if (r < na) != (c < na) {
                    v[(r, c)] *= coupling;
                }
            }
        }
        v
    }

    fn floor_of(v: &Array2<f64>) -> f64 {
        let l = v.cholesky(UPLO::Lower).unwrap();
        let li = l.inv().unwrap();
        eigen_floor(li.iter().map(|x| x * x).sum(), v.diag().sum(), v.nrows()).unwrap()
    }

    fn own_norm(v: &Array2<f64>, x: &Array1<f64>) -> f64 {
        x.dot(&v.inv().unwrap().dot(x)).sqrt()
    }

    /// The bound against a dense solve on the adversarial pair: no
    /// cross-domain strip rows (`a_B = b_A = 0`), α_A and β_B along the top
    /// singular pair of Y, so the γ terms carry all of J.
    #[test]
    fn fitted_bound_covers_the_adversarial_pair() {
        let (n, na) = (7, 4);
        for (seed, coupling) in [(1u64, 1e-3), (2, 3e-2), (3, 0.2), (4, 0.5)] {
            let v = spd(n, na, 1e3, coupling, seed);
            let (vaa, vbb, vab) = (
                v.slice(s![..na, ..na]).to_owned(),
                v.slice(s![na.., na..]).to_owned(),
                v.slice(s![..na, na..]).to_owned(),
            );
            let wa = vaa.cholesky(UPLO::Lower).unwrap().inv().unwrap();
            let wb = vbb.cholesky(UPLO::Lower).unwrap().inv().unwrap();
            let y = wa.dot(&vab).dot(&wb.t());
            let (u, sv, vt) = y.svd(true, true).unwrap();
            let (u, vt) = (u.unwrap(), vt.unwrap());
            let mut a = Array1::<f64>::zeros(n);
            let mut b = Array1::<f64>::zeros(n);
            a.slice_mut(s![..na])
                .assign(&vaa.cholesky(UPLO::Lower).unwrap().dot(&u.column(0)));
            b.slice_mut(s![na..])
                .assign(&(-&vbb.cholesky(UPLO::Lower).unwrap().dot(&vt.row(0))));
            let j = a.dot(&v.inv().unwrap().dot(&b));
            let (li, lj) = (floor_of(&vaa), floor_of(&vbb));
            let norms = PairNorms {
                u_a: own_norm(&vaa, &a.slice(s![..na]).to_owned()),
                u_b: 0.0,
                v_a: 0.0,
                v_b: own_norm(&vbb, &b.slice(s![na..]).to_owned()),
                gamma: sv[0],
            };
            let gb = fitted_bound(&norms).expect("gamma < 1");
            assert!(
                j.abs() <= gb * (1.0 + 1e-10),
                "seed {seed}: |J| {j} > bound {gb}"
            );
            // the code's computable gamma is never below the exact one
            let g_code = vab.norm_l2() / (li * lj).sqrt();
            assert!(g_code >= sv[0], "seed {seed}: Frobenius gamma below ||Y||");
            // mutant: dropping the remainder fails once gamma is not small
            let no_rem = norms.u_a * norms.v_b * norms.gamma;
            if sv[0] > 0.1 {
                assert!(j.abs() > no_rem, "seed {seed}: remainder mutant not caught");
            }
        }
    }

    #[test]
    fn fitted_bound_refuses_gamma_at_or_above_one() {
        let n = |gamma| PairNorms {
            u_a: 1.0,
            u_b: 0.0,
            v_a: 0.0,
            v_b: 1.0,
            gamma,
        };
        assert!(fitted_bound(&n(1.0)).is_none());
        assert!(fitted_bound(&n(f64::NAN)).is_none());
        assert!(fitted_bound(&n(0.5)).is_some());
    }

    #[test]
    fn eigen_floor_is_below_lambda_min_and_refuses_ill_conditioning() {
        for (cond, seed) in [(1e2, 7u64), (1e6, 8), (1e10, 9)] {
            let v = spd(9, 4, cond, 1.0, seed);
            let lam = ndarray_linalg::EigValsh::eigvalsh(&v, UPLO::Lower).unwrap()[0];
            assert!(floor_of(&v) <= lam, "cond {cond}: floor above lambda_min");
        }
        // kappa so large the computed inverse is not trusted: no floor
        assert!(eigen_floor(1e20, 1.0, 1000).is_none());
        // a floor below the Cholesky error eta: none
        assert!(eigen_floor(1e15, 1.0, 1000).is_none());
    }

    /// The shift-verified floor sits below λ_min, within a factor 10 of it,
    /// and far above the trace floor when many eigenvalues are small.
    #[test]
    fn verified_floor_is_below_lambda_min_and_tighter_than_the_trace() {
        for (cond, seed) in [(1e2, 17u64), (1e6, 18), (1e10, 19)] {
            let v = spd(40, 30, cond, 1.0, seed);
            let lam = ndarray_linalg::EigValsh::eigvalsh(&v, UPLO::Lower).unwrap()[0];
            let l = v.cholesky(UPLO::Lower).unwrap();
            let linv = l.inv().unwrap();
            let s = verified_floor(&v, &linv).expect("a shift verifies");
            assert!(
                s <= lam,
                "cond {cond}: verified floor {s} above lambda_min {lam}"
            );
            assert!(s >= 0.1 * lam * 0.99, "cond {cond}: floor {s} vs {lam}");
            assert!(
                s > floor_of(&v),
                "cond {cond}: no gain over the trace floor"
            );
        }
        // mutant: accepting an unverified shift above lambda_min is caught
        let v = spd(40, 30, 1e6, 1.0, 18);
        let lam = ndarray_linalg::EigValsh::eigvalsh(&v, UPLO::Lower).unwrap()[0];
        let mut m = v.clone();
        m.diag_mut().mapv_inplace(|x| x - 2.0 * lam);
        assert!(
            m.cholesky(UPLO::Lower).is_err(),
            "a shift past lambda_min must fail Cholesky"
        );
    }

    #[test]
    fn proved_empty_matches_the_dressed_bound() {
        let (eps, w, kappa) = (1e-4, 0.8, 1.0);
        let gb = |g| gate_bound(g, Some((kappa, w)));
        // just under and just over eps after dressing
        let g_under = 0.99 * eps / (1.0 + 2.0 * w);
        let g_over = 1.01 * eps / (1.0 + 2.0 * w);
        assert!(gb(g_under) < eps && proved_empty(g_under, eps, Some((kappa, w))));
        assert!(gb(g_over) > eps && !proved_empty(g_over, eps, Some((kappa, w))));
        // screen off: the bare bound
        assert!(proved_empty(0.99 * eps, eps, None));
        assert!(!proved_empty(1.01 * eps, eps, None));
        assert!(!proved_empty(f64::NAN, eps, None));
        assert!(!proved_empty(0.0, 0.0, None), "eps = 0 keeps everything");
    }

    #[test]
    fn pair_skip_parses_strictly_and_resolves_per_kernel() {
        for m in [PairSkip::Off, PairSkip::Auto, PairSkip::On, PairSkip::Audit] {
            assert_eq!(PairSkip::parse_config_str(m.as_config_str()), Ok(m));
        }
        assert!(PairSkip::parse_config_str("On").is_err());
        assert!(PairSkip::parse_config_str("").is_err());
        let auto = PairSkip::Auto;
        assert_eq!(auto.resolve(&Operator::erfc(0.2223)), Ok(PairSkip::On));
        assert_eq!(auto.resolve(&Operator::terfc(1.5)), Ok(PairSkip::On));
        assert_eq!(auto.resolve(&Operator::coulomb()), Ok(PairSkip::Off));
        assert_eq!(auto.resolve(&Operator::terf(1.5)), Ok(PairSkip::Off));
        assert_eq!(PairSkip::On.resolve(&Operator::coulomb()), Ok(PairSkip::On));
        assert!(PairSkip::On.resolve(&Operator::terf(1.5)).is_err());
        assert!(PairSkip::Audit.resolve(&Operator::terfc_fit(1.5)).is_err());
    }
}
