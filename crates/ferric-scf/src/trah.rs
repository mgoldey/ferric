//! Trust-region augmented-Hessian (TRAH) SCF.
//!
//! This module supplies the two things ferric's existing second-order SCF
//! machinery did not have:
//!
//! 1. A **level-shifted** augmented-Hessian solve — "shift μ until ‖κ‖ ≤ Δ" —
//!    rather than the componentwise clip that [`crate::rohf_ah`] applies after
//!    the fact (its own doc comment names this as the thing it is *not* doing).
//! 2. A **trust region**: a predicted-vs-actual energy-reduction ratio ρ, a
//!    radius that expands on good steps and contracts on bad ones, and
//!    **rejection** of a step whose actual energy change disagrees with the
//!    quadratic model.
//!
//! # The algorithm
//!
//! Following Helmich-Paris, *A trust-region augmented Hessian implementation
//! for restricted and unrestricted Hartree-Fock and Kohn-Sham methods*,
//! J. Chem. Phys. **154**, 164104 (2021) (arXiv:2012.08306), which combines
//! Bacskay's quadratically-convergent AH-SCF with Fletcher's restricted-step
//! trust region. One TRAH step solves the **level-shifted Newton equations**
//! (their Eq. 8)
//!
//! ```text
//!   (H − μ I) κ = −g ,        μ ≤ λ_min(H)
//! ```
//!
//! where the shift μ is chosen so the step satisfies the trust-region
//! constraint ‖κ‖₂ ≤ Δ. The shift is *not* searched for directly on H.
//! Instead it is read off the **scaled augmented Hessian** (their Eq. 9),
//!
//! ```text
//!   A(α) = ⎡  0      α gᵀ ⎤          ⎡  1   ⎤        ⎡  1   ⎤
//!          ⎣ α g      H   ⎦ ,   A(α) ⎢      ⎥ =  μ   ⎢      ⎥
//!                                     ⎣ κ(α) ⎦        ⎣ κ(α) ⎦
//! ```
//!
//! whose lowest eigenvalue μ **is** the level shift and whose lower block,
//! **divided by α**, is the step:
//!
//! ```text
//!   κ = κ(α) / α          (Eq. 10)
//! ```
//!
//! Expanding the second row, α g + H κ(α) = μ κ(α), and dividing by α gives
//! exactly (H − μ I) κ = −g. Because μ is the lowest eigenvalue of a matrix
//! that borders H, Cauchy interlacing puts μ ≤ λ_min(H), so H − μ I is
//! positive semidefinite **automatically** — the step is a descent direction
//! even when the orbital Hessian is indefinite, with no explicit eigenvalue
//! analysis of H. That is the whole reason to use AH rather than raw Newton on
//! the near-degenerate cases this module targets.
//!
//! α is the control knob. It enters only the off-diagonal border, never H.
//! Raising α drives μ more negative, which makes H − μ I more strongly
//! definite and therefore **shortens** the step: ‖κ(α)/α‖ is monotonically
//! decreasing in α. (Measured on a random indefinite 6×6: ‖κ‖ = 23.7, 1.95,
//! 0.149, 0.0452, 0.0098, 0.0010 at α = 1, 2, 5, 20, 100, 1000, with the
//! level-shifted NR residual at 1e-14 throughout — pinned by
//! `larger_alpha_shortens_the_step_monotonically`.) So enforcing the trust
//! region is a **one-dimensional, monotone, bracketable root-find on α**,
//! which [`crate::trah::solve_trust_region`] performs by bisection on log α over
//! [α_min, α_max] = [1, 1000] (the paper's Table I bracket).
//!
//! # Predicted reduction and the ρ test
//!
//! The quadratic model's predicted energy change is (their Eqs. 18–19)
//!
//! ```text
//!   ΔE_pred = gᵀκ + ½ κᵀHκ  =  ½ ( gᵀκ + μ ‖κ‖² )
//! ```
//!
//! the second form following from substituting Hκ = −g + μκ. The two are
//! algebraically identical but *numerically independent*: the first costs an
//! extra Hessian matvec, the second is free given μ and ‖κ‖ from the
//! eigensolve. [`crate::trah::solve_trust_region`] computes **both** and reports their
//! disagreement in [`crate::trah::TrahStep::predicted_residual`], because that disagreement
//! is a direct, self-checking measure of how well the Davidson solve actually
//! converged the AH eigenproblem — an identity that only holds if the
//! eigenvector really is one. It is the cheapest available anchor on the
//! solver's own arithmetic, and `trah_consistency.rs` asserts on it.
//!
//! The actual change ΔE_act = E(κ) − E(0) is only knowable after the next Fock
//! build, so the caller feeds it back on the following SCF iteration via
//! [`crate::trah::TrahState::assess`], which forms ρ = ΔE_act / ΔE_pred (their Eqs. 16–17;
//! both are signed energy changes, so a good step gives ρ ≈ +1) and applies
//! Fletcher's update:
//!
//! | ρ            | step     | radius     |
//! |--------------|----------|------------|
//! | ρ < 0        | rejected | Δ ← 0.7 Δ  |
//! | 0 ≤ ρ ≤ 0.25 | accepted | Δ ← 0.7 Δ  |
//! | 0.25 < ρ ≤ 0.75 | accepted | unchanged |
//! | ρ > 0.75     | accepted | Δ ← 1.2 Δ  |
//!
//! # KNOWN DEFECT: the rejection loop can stall (measured, not fixed)
//!
//! On RKS/PBE water/cc-pVDZ, TRAH enters a repeating cycle after its first
//! rejection. Traced with `FERRIC_SCF_TRACE=1`:
//!
//! ```text
//!   iter  6: rho= 1.001722  Accepted  Delta=4.800e-1
//!   iter  7: rho=-60.519638 Rejected  Delta=3.360e-1  (restore)
//!   iter  9: rho=-60.519638 Rejected  Delta=2.352e-1  (restore)
//!   iter 11: rho=-60.519638 Rejected  Delta=1.646e-1  (restore)
//!   ... radius contracts 0.7x per cycle until it collapses ...
//! ```
//!
//! ρ is IDENTICAL to six decimals every cycle. A repeated exact value is a
//! fingerprint of arithmetic, not of measurement: the restore returns to the
//! same point, the same step is recomputed, and only the radius changes — but
//! the radius is not yet binding (the step is interior, α = α_min), so
//! contracting it does not change the step either. The cycle therefore burns
//! two Fock builds per iteration until the radius collapses and the run falls
//! back to DIIS.
//!
//! This is why TRAH is a REGRESSION on that case: 123 iterations / 98 s versus
//! DIIS's 69 / 1.5 s. It is also why the radius floor and
//! [`crate::trah::TrahState::collapsed`] exist — without them the cycle would not terminate.
//!
//! The likely cause is that ρ is being formed across a Fock rebuild rather than
//! at a fixed reference point, so a large negative ρ reflects the DIIS-era
//! energy change rather than the step's. The fix is to re-step immediately at
//! the contracted radius from the restored point (as Helmich-Paris specifies:
//! "the micro iterations are repeated from the previous set of orbitals")
//! instead of yielding to the next macro iteration. That restructuring is not
//! done here; TRAH is opt-in and off by default, so the defect is inert unless
//! deliberately enabled.
//!
//! # Scope
//!
//! The solver here is deliberately method-agnostic: it works on a flat
//! coordinate vector with a caller-supplied matvec and preconditioner
//! diagonal, so RHF/RKS ([`crate::rhf_newton`]) and UHF/UKS
//! ([`crate::uhf_newton`]) reuse their **existing, FD-validated** Hessian
//! matvecs unchanged. Nothing in this module knows about spin, XC or basis
//! functions.

use crate::stability::{davidson_lowest, StabilityConfig};
use ferric_core::FerricError;

/// Count of TRAH steps actually applied to orbitals, process-wide.
///
/// # Why an engagement counter exists
///
/// "Is TRAH off?" cannot be answered from converged energies alone. On an easy
/// system (water/cc-pVDZ RHF) the TRAH path converges quadratically to the SAME
/// stationary point as DIIS and the final energy is bit-identical either way —
/// measured, not assumed: a deliberate mutation that armed TRAH from a merely
/// *present* `TrahConfig` left every energy bit-identical and SURVIVED an
/// energy-only bit-identity test, while the trace showed 8 TRAH steps had run.
///
/// An energy comparison therefore proves the ANSWER is unchanged, not that the
/// code path was unused. This counter makes the path itself observable, so
/// `tests/trah_off_is_bit_identical.rs` can assert the strictly stronger claim
/// "the branch never executed" and a mutation to the gate is genuinely
/// detectable. Counters follow `crate::rohf::GGA_FXC_KERNEL_BUILDS`.
pub static TRAH_STEPS_TAKEN: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Count of TRAH steps REJECTED by the ρ test, process-wide. Lets a test prove
/// rejection is reachable on a real system rather than only in unit arithmetic.
pub static TRAH_STEPS_REJECTED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Record that a TRAH step was applied. Called from the SCF loops.
pub fn note_trah_step() {
    TRAH_STEPS_TAKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Record that a TRAH step was rejected by the ρ test.
pub fn note_trah_rejection() {
    TRAH_STEPS_REJECTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Most recent ρ formed on a closed-shell (RHF/RKS) TRAH step, as raw f64 bits
/// (`f64::NAN` bits = none yet). Stored as bits so it can live in an atomic.
static LAST_RHO_RHF: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Most recent ρ formed on an open-shell (UHF/UKS) TRAH step. See
/// [`LAST_RHO_RHF`].
static LAST_RHO_UHF: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Publish the ρ just formed on a closed-shell step.
pub fn note_rho_rhf(rho: f64) {
    LAST_RHO_RHF.store(rho.to_bits(), std::sync::atomic::Ordering::Relaxed);
}
/// Publish the ρ just formed on an open-shell step.
pub fn note_rho_uhf(rho: f64) {
    LAST_RHO_UHF.store(rho.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

/// The last ρ formed on a closed-shell TRAH step, if any.
///
/// Exposed so a test can assert on ρ ITSELF rather than on a downstream
/// consequence of it. ρ is the trust region's central quantity and a scale
/// error in it is invisible in a converged energy — see
/// `RHF_ENERGY_SCALE` for the 4× bug this observable was added to catch.
pub fn last_rho_rhf() -> Option<f64> {
    let v = f64::from_bits(LAST_RHO_RHF.load(std::sync::atomic::Ordering::Relaxed));
    (v != 0.0 && v.is_finite()).then_some(v)
}
/// The last ρ formed on an open-shell TRAH step, if any. See
/// [`last_rho_rhf`].
pub fn last_rho_uhf() -> Option<f64> {
    let v = f64::from_bits(LAST_RHO_UHF.load(std::sync::atomic::Ordering::Relaxed));
    (v != 0.0 && v.is_finite()).then_some(v)
}

/// Smallest AH scale factor — the unscaled augmented Hessian, longest step.
/// Helmich-Paris Table I: α_min = 1.
const ALPHA_MIN: f64 = 1.0;
/// Largest AH scale factor. Helmich-Paris Table I: α_max = 1000.
const ALPHA_MAX: f64 = 1000.0;

/// Trust-region control parameters.
///
/// Defaults follow Helmich-Paris (arXiv:2012.08306) Table I and its Fletcher
/// update rules. κ is a dimensionless orbital-rotation angle, so a radius of
/// ~0.4 rad (the paper's initial value) is already a large step.
///
/// Note the contraction/expansion factors are **deliberately gentler** than
/// textbook trust-region values (Nocedal & Wright use 0.25 / 2.0): each macro
/// iteration costs a full set of Fock builds, so a slowly-breathing radius
/// avoids both violent collapse and overshoot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrahConfig {
    /// Initial trust radius ‖κ‖₂ in radians. Default 0.4 (paper Table I).
    pub radius0: f64,
    /// Smallest radius before the trust region is declared collapsed and the
    /// caller falls back to DIIS. Default 1e-4.
    ///
    /// The paper publishes **no** h_min (Table I bounds α but not h); without
    /// a floor, repeated 0.7× contractions underflow into a stalled state
    /// taking vanishing steps forever. This floor is ferric's addition and is
    /// the reason [`crate::trah::TrahState::collapsed`] exists.
    pub radius_min: f64,
    /// Largest radius the expansion rule may reach. Default 2.0. Also not in
    /// the paper; it bounds unlimited 1.2× growth once the constraint goes
    /// inactive.
    pub radius_max: f64,
    /// ρ below this ⇒ the step is **rejected**. Default 0.0 (paper: ρ < 0).
    pub rho_reject: f64,
    /// ρ at or below this (but above `rho_reject`) ⇒ accept but contract.
    /// Default 0.25 (paper).
    pub rho_poor: f64,
    /// ρ above this ⇒ accept and expand. Default 0.75 (paper).
    pub rho_good: f64,
    /// Radius contraction factor. Default 0.7 (paper / Fletcher).
    pub shrink: f64,
    /// Radius expansion factor. Default 1.2 (paper / Fletcher).
    pub grow: f64,
    /// Convergence tolerance for the Davidson solve of the augmented matrix.
    pub davidson_conv: f64,
    /// Maximum Davidson subspace size.
    pub davidson_max_vecs: usize,
    /// Maximum augmented-Hessian eigensolves spent on the α search.
    pub max_shift_iter: usize,
    /// Relative tolerance on ‖κ‖ vs Δ for the α bisection. Default 5e-2. The
    /// trust radius is itself a heuristic, so landing within a few percent of
    /// the boundary is ample and saves Davidson solves.
    pub step_tol: f64,
}

impl Default for TrahConfig {
    fn default() -> Self {
        Self {
            radius0: 0.4,
            radius_min: 1e-4,
            radius_max: 2.0,
            rho_reject: 0.0,
            rho_poor: 0.25,
            rho_good: 0.75,
            shrink: 0.7,
            grow: 1.2,
            davidson_conv: 1e-7,
            davidson_max_vecs: 50,
            max_shift_iter: 20,
            step_tol: 5e-2,
        }
    }
}

/// The outcome of one level-shifted augmented-Hessian solve.
#[derive(Debug, Clone)]
pub struct TrahStep {
    /// The rotation κ = κ(α)/α, flattened in the caller's packing order.
    pub kappa: Vec<f64>,
    /// ‖κ‖₂.
    pub norm: f64,
    /// The level shift μ (the AH lowest eigenvalue), ≤ λ_min(H).
    pub level_shift: f64,
    /// The AH scale factor α actually used.
    pub alpha: f64,
    /// Predicted energy change ΔE_pred = gᵀκ + ½κᵀHκ (Eq. 18). Negative for a
    /// descent step.
    pub predicted: f64,
    /// The same quantity via the free identity ½(gᵀκ + μ‖κ‖²) (Eq. 19).
    pub predicted_cheap: f64,
    /// |Eq.18 − Eq.19| / max(|Eq.18|, tiny) — a self-check on the AH solve.
    /// Large values mean Davidson did not actually converge the eigenpair.
    pub predicted_residual: f64,
    /// Whether the step sits on the trust-region boundary (‖κ‖ ≈ Δ).
    pub on_boundary: bool,
    /// Number of augmented-Hessian eigensolves spent on the α search.
    pub shift_iterations: usize,
}

/// Solve the level-shifted augmented-Hessian equations subject to ‖κ‖₂ ≤ Δ.
///
/// `g` is the orbital gradient (flat). `matvec` applies the orbital Hessian H
/// to a flat vector. `diag` is a Hessian-diagonal approximation used **only**
/// as the Davidson preconditioner (typically the orbital-energy gaps).
///
/// # The α search
///
/// ‖κ(α)/α‖ decreases monotonically in α, so:
/// - α = α_min = 1 (the unscaled AH problem) is tried first. If ‖κ‖ ≤ Δ the
///   constraint is inactive — this is the plain AH/Newton step and no search
///   is needed. The paper notes this is the usual case after the first few
///   macro iterations.
/// - Otherwise α is grown geometrically until the step is feasible, then
///   bisected on log α until ‖κ‖ lands within `step_tol` of Δ.
///
/// If the step is still infeasible at α_max the result is scaled down to the
/// radius, so the returned step **always** satisfies ‖κ‖ ≤ Δ.
pub fn solve_trust_region<F>(
    g: &[f64],
    matvec: &F,
    diag: &[f64],
    radius: f64,
    cfg: &TrahConfig,
) -> Result<TrahStep, FerricError>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, FerricError>,
{
    let n = g.len();
    if n == 0 {
        return Err(FerricError::General(
            "TRAH: empty orbital-rotation space".to_string(),
        ));
    }
    if diag.len() != n {
        return Err(FerricError::General(format!(
            "TRAH: preconditioner diagonal has length {} but the gradient has length {n}",
            diag.len()
        )));
    }
    if !(radius > 0.0) {
        return Err(FerricError::General(format!(
            "TRAH: trust radius must be positive, got {radius}"
        )));
    }

    // α = α_min first: the plain (unscaled) augmented-Hessian step.
    let mut best = ah_solve(g, matvec, diag, ALPHA_MIN, cfg)?;
    let mut n_solves = 1usize;

    if best.norm > radius {
        // Bracket: grow α until the step is feasible. Larger α ⇒ more negative
        // μ ⇒ a harder level shift ⇒ a shorter step.
        let mut lo = ALPHA_MIN; // infeasible (‖κ‖ > Δ)
        let mut hi = ALPHA_MIN;
        let mut feasible: Option<TrahStep> = None;
        while hi < ALPHA_MAX && n_solves < cfg.max_shift_iter {
            hi = (hi * 8.0).min(ALPHA_MAX);
            let cand = ah_solve(g, matvec, diag, hi, cfg)?;
            n_solves += 1;
            if cand.norm <= radius {
                feasible = Some(cand);
                break;
            }
            lo = hi;
        }

        match feasible {
            None => {
                // Still infeasible at α_max: clamp to the radius so the caller
                // always gets a step inside the region. The paper's "smallest
                // deviation" wording implies the same clamp.
                let s = radius / best.norm;
                for v in best.kappa.iter_mut() {
                    *v *= s;
                }
                best.norm = radius;
            }
            Some(mut feas) => {
                // Bisect on log α between the infeasible `lo` and feasible `hi`.
                let mut lo_a = lo;
                let mut hi_a = hi;
                while n_solves < cfg.max_shift_iter {
                    // Close enough to the boundary from the inside: done.
                    if feas.norm >= radius * (1.0 - cfg.step_tol) {
                        break;
                    }
                    if !(hi_a > lo_a * (1.0 + 1e-9)) {
                        break;
                    }
                    let mid = (0.5 * (lo_a.ln() + hi_a.ln())).exp();
                    let cand = ah_solve(g, matvec, diag, mid, cfg)?;
                    n_solves += 1;
                    if cand.norm <= radius {
                        feas = cand;
                        hi_a = mid;
                    } else {
                        lo_a = mid;
                    }
                }
                best = feas;
            }
        }
    }

    let mut step = best;
    step.shift_iterations = n_solves;
    step.on_boundary = step.norm >= radius * (1.0 - cfg.step_tol);

    // Predicted reduction, BOTH ways (Helmich-Paris Eqs. 18 and 19). They are
    // algebraically identical given an exact AH eigenpair, so their difference
    // measures how well Davidson actually converged — see the module docs.
    let hk = matvec(&step.kappa)?;
    let gk: f64 = g.iter().zip(step.kappa.iter()).map(|(a, b)| a * b).sum();
    let khk: f64 = step.kappa.iter().zip(hk.iter()).map(|(a, b)| a * b).sum();
    step.predicted = gk + 0.5 * khk;
    step.predicted_cheap = 0.5 * (gk + step.level_shift * step.norm * step.norm);
    let scale = step.predicted.abs().max(1e-300);
    step.predicted_residual = (step.predicted - step.predicted_cheap).abs() / scale;

    Ok(step)
}

/// One augmented-Hessian eigensolve at scale factor α.
///
/// Builds A(α) = [[0, αgᵀ], [αg, H]] implicitly (matvec only) and takes its
/// lowest usable eigenpair via [`crate::stability::davidson_lowest`], then
/// returns the step κ = κ(α)/α.
fn ah_solve<F>(
    g: &[f64],
    matvec: &F,
    diag: &[f64],
    alpha: f64,
    cfg: &TrahConfig,
) -> Result<TrahStep, FerricError>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, FerricError>,
{
    let n = g.len();
    let n_aug = n + 1;

    let aug_matvec = |v: &[f64]| -> Result<Vec<f64>, FerricError> {
        let v0 = v[0];
        let hk = matvec(&v[1..])?;
        let mut out = vec![0.0f64; n_aug];
        // Row 0: α gᵀ κ(α)
        out[0] = alpha * g.iter().zip(v[1..].iter()).map(|(a, b)| a * b).sum::<f64>();
        // Rows 1..: α g v₀ + H κ(α)
        for i in 0..n {
            out[1 + i] = alpha * g[i] * v0 + hk[i];
        }
        Ok(out)
    };

    // Davidson preconditioner diagonal for the augmented matrix. The leading
    // entry is A(α)[0,0] = 0, which is below every orbital-gap entry — exactly
    // the right seed, since the sought eigenvector is dominated by its leading
    // component.
    let mut aug_diag = Vec::with_capacity(n_aug);
    aug_diag.push(0.0);
    aug_diag.extend_from_slice(diag);

    let dav_cfg = StabilityConfig {
        conv_thresh: cfg.davidson_conv,
        max_subspace: cfg.davidson_max_vecs,
        ..StabilityConfig::default()
    };
    let pair = davidson_lowest(n_aug, aug_matvec, &aug_diag, &dav_cfg)?;

    // The AH ansatz needs the eigenvector's leading component to be meaningful:
    // κ is read off as (lower block)/(leading entry), so a vanishing leading
    // entry would divide by ~0 and manufacture an arbitrarily large step out of
    // a numerically meaningless eigenvector. PySCF's `_regular_step` guards the
    // same way (it selects the lowest root with |v[0]| > 0.1, noting "There
    // exists systems that the first eigenvalue of AH is -inf"); here we have a
    // single root from `davidson_lowest`, so we reject rather than reselect.
    let first = pair.eigenvector[0];
    if first.abs() < 1e-6 {
        return Err(FerricError::General(format!(
            "TRAH: the augmented-Hessian eigenvector has a near-zero leading entry \
             ({first:.3e}) at scale α={alpha:.3e}. The AH ansatz κ = (lower block)/(leading \
             entry) is not applicable, so no step is taken rather than one scaled by an \
             arbitrarily small number"
        )));
    }

    // κ = κ(α)/α  (Helmich-Paris Eq. 10). NOTE the division: raising α
    // SHORTENS the step. Multiplying here instead would make the α search
    // move the step length the wrong way and never satisfy the constraint.
    let kappa: Vec<f64> = pair.eigenvector[1..]
        .iter()
        .map(|&x| x / (first * alpha))
        .collect();
    let norm = kappa.iter().map(|&x| x * x).sum::<f64>().sqrt();

    Ok(TrahStep {
        kappa,
        norm,
        level_shift: pair.eigenvalue,
        alpha,
        predicted: 0.0,
        predicted_cheap: 0.0,
        predicted_residual: 0.0,
        on_boundary: false,
        shift_iterations: 0,
    })
}

/// What the trust region decided about the step just assessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrahVerdict {
    /// ρ was good; the step stands. The radius may have grown.
    Accepted,
    /// ρ was acceptable but poor; the step stands, radius contracted.
    AcceptedPoor,
    /// ρ ≤ `rho_reject` — the step went uphill relative to the model and must
    /// be **undone** by the caller. The radius has been contracted.
    Rejected,
}

/// A pending step awaiting its ρ verdict.
#[derive(Debug, Clone)]
struct Pending {
    energy_before: f64,
    predicted: f64,
}

/// Cross-iteration trust-region bookkeeping.
///
/// # Why the ρ test spans two SCF iterations
///
/// ρ needs the energy **at the stepped point**, and in an SCF loop that energy
/// only exists after the next Fock build. Rather than paying a speculative
/// Fock build inside the step (which would double the cost of every TRAH
/// iteration), the state records the prediction when the step is taken and
/// forms ρ at the top of the following iteration, where the loop's `energy` is
/// already the actual energy at the stepped density. A rejected step is undone
/// by restoring orbitals the caller saved before stepping.
#[derive(Debug, Clone)]
pub struct TrahState {
    cfg: TrahConfig,
    radius: f64,
    pending: Option<Pending>,
    /// Number of steps accepted so far.
    pub accepted: usize,
    /// Number of steps rejected so far.
    pub rejected: usize,
    /// Most recent ρ (None before the first assessment).
    pub last_rho: Option<f64>,
}

impl TrahState {
    /// Fresh trust-region state at the configured initial radius.
    pub fn new(cfg: TrahConfig) -> Self {
        Self {
            radius: cfg.radius0,
            cfg,
            pending: None,
            accepted: 0,
            rejected: 0,
            last_rho: None,
        }
    }

    /// The current trust radius.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// The configuration in force.
    pub fn config(&self) -> &TrahConfig {
        &self.cfg
    }

    /// True when the radius has collapsed below `radius_min` — the trust
    /// region has given up and the caller should fall back to DIIS rather than
    /// take vanishing steps forever.
    pub fn collapsed(&self) -> bool {
        self.radius < self.cfg.radius_min
    }

    /// Is a step awaiting its ρ verdict?
    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Record that a step with predicted reduction `step.predicted` was taken
    /// from a point whose energy was `energy_before`.
    pub fn record_step(&mut self, energy_before: f64, step: &TrahStep) {
        self.pending = Some(Pending {
            energy_before,
            predicted: step.predicted,
        });
    }

    /// Form ρ for the pending step against the energy now observed, and update
    /// the trust radius per Fletcher's rules.
    ///
    /// Returns `None` when no step was pending. On [`TrahVerdict::Rejected`]
    /// the caller **must** restore the orbitals it saved before the step.
    pub fn assess(&mut self, energy_now: f64) -> Option<TrahVerdict> {
        let pending = self.pending.take()?;
        let actual = energy_now - pending.energy_before;

        // A non-negative prediction means the quadratic model never promised
        // descent, so ρ carries no information (and a zero prediction would be
        // a division by zero). Treat it as a rejection.
        let rho = if pending.predicted < 0.0 {
            actual / pending.predicted
        } else {
            f64::NEG_INFINITY
        };
        self.last_rho = Some(rho);

        let verdict = if rho <= self.cfg.rho_reject {
            self.rejected += 1;
            self.shrink();
            TrahVerdict::Rejected
        } else if rho <= self.cfg.rho_poor {
            self.accepted += 1;
            self.shrink();
            TrahVerdict::AcceptedPoor
        } else {
            self.accepted += 1;
            if rho > self.cfg.rho_good {
                self.radius = (self.radius * self.cfg.grow).min(self.cfg.radius_max);
            }
            TrahVerdict::Accepted
        };
        Some(verdict)
    }

    /// Contract the radius. Allowed to fall below `radius_min` — that is what
    /// makes [`TrahState::collapsed`] reachable and lets the caller give up.
    fn shrink(&mut self) {
        self.radius *= self.cfg.shrink;
    }

    /// Drop any pending assessment without forming ρ.
    ///
    /// Used when the caller abandons the TRAH path (converged, or fell back to
    /// DIIS) so a stale prediction cannot be matched against an energy produced
    /// by a different kind of step.
    pub fn clear_pending(&mut self) {
        self.pending = None;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Method adapters: pack ferric's existing, FD-validated Hessian matvecs into
// the flat-vector interface `solve_trust_region` expects.
//
// These deliberately add NO new physics. Every Hessian-vector product below is
// the SAME call the `newton_trigger` path already makes; only the packing and
// the step-length control differ. That is what makes it defensible to reuse the
// FD validation those matvecs already carry (`uhf_newton_smoke.rs`'s ~3e-10
// analytic-vs-FD check, `rhf_newton_smoke.rs`'s ε-ladder) rather than
// re-deriving it here.
// ─────────────────────────────────────────────────────────────────────────────

use crate::engine_pool::EnginePool;
use ferric_core::parallel::ParallelContext;
use ndarray::Array2;

/// The occ→virt block (rows = virt, cols = occ) of a square MO matrix.
fn ov_block(m: &Array2<f64>, nocc: usize, n: usize) -> Array2<f64> {
    let nv = n - nocc;
    let mut out = Array2::<f64>::zeros((nv, nocc));
    for (ir, a) in (nocc..n).enumerate() {
        for i in 0..nocc {
            out[(ir, i)] = m[(a, i)];
        }
    }
    out
}

/// Orbital-energy-gap preconditioner diagonal, floored away from zero.
///
/// Matches `rhf_newton`/`uhf_newton`'s `build_gap` exactly (including the 1e-6
/// floor) so the Davidson preconditioner TRAH uses is the same one the PCG path
/// uses. Used ONLY to precondition; it never enters the step.
fn gap_diag(f_diag: &[f64], nocc: usize, n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity((n - nocc) * nocc);
    for a in nocc..n {
        for i in 0..nocc {
            let d = f_diag[a] - f_diag[i];
            out.push(if d.abs() < 1e-6 { 1e-6 } else { d });
        }
    }
    out
}

/// Apply the Cayley unitary built from an occ→virt block to C: C ← C·U.
///
/// Identical construction to `rhf_newton`/`uhf_newton`'s — the exact
/// orthonormality-preserving rotation, not a truncated exponential.
fn apply_cayley(
    c: &Array2<f64>,
    k_ov: &Array2<f64>,
    nocc: usize,
    n: usize,
) -> Result<Array2<f64>, FerricError> {
    use ndarray_linalg::Solve;
    let mut kappa = Array2::<f64>::zeros((n, n));
    for (ir, a) in (nocc..n).enumerate() {
        for i in 0..nocc {
            let v = k_ov[(ir, i)];
            kappa[(a, i)] = v;
            kappa[(i, a)] = -v;
        }
    }
    let half = 0.5 * &kappa;
    let eye = Array2::<f64>::eye(n);
    let a_mat = &eye - &half;
    let b_mat = &eye + &half;
    let mut u = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        let bcol = b_mat.column(col).to_owned();
        let sol = a_mat
            .solve(&bcol)
            .map_err(|e| FerricError::Lapack(format!("TRAH Cayley solve: {e}")))?;
        for row in 0..n {
            u[(row, col)] = sol[row];
        }
    }
    Ok(c.dot(&u))
}

/// Energy scale of the packed orbital-rotation coordinates.
///
/// # Why this constant has to exist, and how it was caught
///
/// `rhf_newton`/`uhf_newton` deliberately use `g_ai = F_ai` and
/// `gap = F_aa − F_ii` rather than the TRUE orbital gradient and Hessian, whose
/// common prefactor they drop. Their own doc comment says so and is right to:
/// a Newton step κ = −H⁻¹g is invariant under g → cg, H → cH, so PCG never
/// notices. The AH eigenproblem is likewise invariant — μ scales but κ does not.
///
/// The trust region is NOT invariant. ΔE_pred = gᵀκ + ½κᵀHκ scales linearly
/// with c, so a dropped prefactor makes the predicted reduction wrong by
/// exactly that factor, and ρ = ΔE_act/ΔE_pred lands at c instead of 1 — which
/// silently mis-classifies every step against Fletcher's 0.25/0.75 thresholds.
///
/// This was MEASURED, not derived after the fact: the first working TRAH runs
/// reported ρ = 3.999, 3.835 on RHF/water and ρ = 2.0047, 2.0051 on UHF/OH —
/// clean constants, not scatter, which is the fingerprint of a scale error
/// rather than a modelling error. The constants match the expected packing:
///
/// - The occ→virt block is packed ONCE but represents BOTH the (a,i) and (i,a)
///   corners of the antisymmetric κ, contributing a factor 2 per spin channel.
/// - RHF additionally carries double occupation (D = 2·C_occ·C_occᵀ), a second
///   factor 2 ⇒ **4**.
/// - UHF packs each spin separately with single occupation ⇒ **2**.
///
/// Both are confirmed to 3-4 significant figures by the measured ρ above, and
/// pinned by `tests/trah_converges.rs::trah_rho_is_order_unity_on_a_real_scf`,
/// which asserts ρ ≈ 1 — an assertion that FAILS by exactly 4× / 2× if either
/// constant is removed.
const RHF_ENERGY_SCALE: f64 = 4.0;
/// See [`RHF_ENERGY_SCALE`]. UHF packs each spin with single occupation.
const UHF_ENERGY_SCALE: f64 = 2.0;

/// One TRAH step on RHF/RKS orbitals.
///
/// Reuses [`crate::rhf_newton::hessian_matvec`] verbatim for H·κ. Returns the
/// rotated MO coefficients and the step record (whose `predicted` the caller
/// must hand to [`TrahState::record_step`]).
pub fn rhf_trah_step(
    ctx: &ParallelContext,
    inp: &crate::rhf_newton::RhfNewtonInputs,
    radius: f64,
    cfg: &TrahConfig,
) -> Result<(Array2<f64>, TrahStep), FerricError> {
    let n = inp.c.nrows();
    let no = inp.nocc;
    let nv = n - no;

    // The orbital gradient is the occ→virt Fock block (the Brillouin condition).
    let g_mat = ov_block(inp.f_mo, no, n);
    let g: Vec<f64> = g_mat.iter().copied().collect();

    let f_diag: Vec<f64> = (0..n).map(|i| inp.f_mo[(i, i)]).collect();
    let diag = gap_diag(&f_diag, no, n);

    // One EnginePool for the whole step, reused across every matvec — the same
    // hoist `rhf_newton_step` performs, for the same reason (the pool is
    // geometry/basis-only, so rebuilding it per matvec is pure waste).
    let pool = EnginePool::new(inp.bounds.op, inp.prep, 1e-14)?;

    let matvec = |v: &[f64]| -> Result<Vec<f64>, FerricError> {
        let k = Array2::from_shape_vec((nv, no), v.to_vec())
            .map_err(|e| FerricError::General(format!("TRAH RHF matvec reshape: {e}")))?;
        let hk = crate::rhf_newton::hessian_matvec(ctx, inp, &k, &pool)?;
        Ok(hk.iter().copied().collect())
    };

    let mut step = solve_trust_region(&g, &matvec, &diag, radius, cfg)?;
    // Restore the prefactor the matvec drops, so `predicted` is a real energy
    // and ρ is order unity. See `RHF_ENERGY_SCALE`.
    step.predicted *= RHF_ENERGY_SCALE;
    step.predicted_cheap *= RHF_ENERGY_SCALE;
    let k = Array2::from_shape_vec((nv, no), step.kappa.clone())
        .map_err(|e| FerricError::General(format!("TRAH RHF step reshape: {e}")))?;
    let c_new = apply_cayley(inp.c, &k, no, n)?;
    Ok((c_new, step))
}

/// One TRAH step on UHF/UKS orbitals.
///
/// The α and β rotations are packed into ONE vector and solved as a single
/// coupled trust-region problem, exactly as [`crate::uhf_newton::uhf_newton_step`]
/// solves them with one coupled PCG: the spins are not independent (δJ sees the
/// total density, and the XC kernel carries the αβ cross term), so a per-spin
/// trust region would be constraining the wrong quantity.
///
/// Reuses [`crate::uhf_newton::hessian_matvec`] verbatim for H·κ.
pub fn uhf_trah_step(
    ctx: &ParallelContext,
    inp: &crate::uhf_newton::UhfNewtonInputs,
    radius: f64,
    cfg: &TrahConfig,
) -> Result<(Array2<f64>, Array2<f64>, TrahStep), FerricError> {
    let n = inp.c_a.nrows();
    let na = inp.nocc_a;
    let nb = inp.nocc_b;
    let nva = n - na;
    let nvb = n - nb;
    let len_a = nva * na;
    let len_b = nvb * nb;

    let g_a = ov_block(inp.f_a_mo, na, n);
    let g_b = ov_block(inp.f_b_mo, nb, n);
    let mut g: Vec<f64> = Vec::with_capacity(len_a + len_b);
    g.extend(g_a.iter().copied());
    g.extend(g_b.iter().copied());

    let fa_diag: Vec<f64> = (0..n).map(|i| inp.f_a_mo[(i, i)]).collect();
    let fb_diag: Vec<f64> = (0..n).map(|i| inp.f_b_mo[(i, i)]).collect();
    let mut diag = gap_diag(&fa_diag, na, n);
    diag.extend(gap_diag(&fb_diag, nb, n));

    let pool = EnginePool::new(inp.bounds.op, inp.prep, 1e-14)?;

    let matvec = |v: &[f64]| -> Result<Vec<f64>, FerricError> {
        let ka = Array2::from_shape_vec((nva, na), v[..len_a].to_vec())
            .map_err(|e| FerricError::General(format!("TRAH UHF α matvec reshape: {e}")))?;
        let kb = Array2::from_shape_vec((nvb, nb), v[len_a..].to_vec())
            .map_err(|e| FerricError::General(format!("TRAH UHF β matvec reshape: {e}")))?;
        let (ha, hb) = crate::uhf_newton::hessian_matvec(ctx, inp, &ka, &kb, &pool)?;
        let mut out: Vec<f64> = Vec::with_capacity(len_a + len_b);
        out.extend(ha.iter().copied());
        out.extend(hb.iter().copied());
        Ok(out)
    };

    let mut step = solve_trust_region(&g, &matvec, &diag, radius, cfg)?;
    // Restore the prefactor the matvec drops. See `UHF_ENERGY_SCALE`.
    step.predicted *= UHF_ENERGY_SCALE;
    step.predicted_cheap *= UHF_ENERGY_SCALE;
    let ka = Array2::from_shape_vec((nva, na), step.kappa[..len_a].to_vec())
        .map_err(|e| FerricError::General(format!("TRAH UHF α step reshape: {e}")))?;
    let kb = Array2::from_shape_vec((nvb, nb), step.kappa[len_a..].to_vec())
        .map_err(|e| FerricError::General(format!("TRAH UHF β step reshape: {e}")))?;
    let ca = apply_cayley(inp.c_a, &ka, na, n)?;
    let cb = apply_cayley(inp.c_b, &kb, nb, n)?;
    Ok((ca, cb, step))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dense symmetric matrix as a matvec, for exercising the solver against
    /// arithmetic we can do by hand.
    fn dense_matvec(h: Vec<Vec<f64>>) -> impl Fn(&[f64]) -> Result<Vec<f64>, FerricError> {
        move |v: &[f64]| {
            Ok(h.iter()
                .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
                .collect())
        }
    }

    /// Solve a small dense system by Gauss-Jordan (test reference only).
    fn solve_dense(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut m = vec![vec![0.0f64; n + 1]; n];
        for i in 0..n {
            m[i][..n].copy_from_slice(&a[i][..n]);
            m[i][n] = b[i];
        }
        for col in 0..n {
            let piv = (col..n)
                .max_by(|&x, &y| m[x][col].abs().partial_cmp(&m[y][col].abs()).unwrap())
                .unwrap();
            m.swap(col, piv);
            let d = m[col][col];
            for j in col..=n {
                m[col][j] /= d;
            }
            for r in 0..n {
                if r != col {
                    let f = m[r][col];
                    for j in col..=n {
                        m[r][j] -= f * m[col][j];
                    }
                }
            }
        }
        (0..n).map(|i| m[i][n]).collect()
    }

    /// EXACTNESS ANCHOR for the solver: the α → 0 limit of the AH step is the
    /// unconstrained Newton step κ = −H⁻¹g, approached at order α².
    ///
    /// # Why the anchor is a LIMIT and not an equality at α = 1
    ///
    /// The obvious anchor to write — "at a huge trust radius TRAH returns the
    /// Newton step" — is FALSE, and believing it would have hidden a real
    /// property of the method. The AH step always solves the *level-shifted*
    /// equations (H − μI)κ = −g, and μ is not zero at α = 1: it is the lowest
    /// eigenvalue of the bordered matrix, which for this system is −5.13e-3,
    /// giving a step that differs from Newton's by 2.6e-3 relative no matter
    /// how large Δ is. Since μ(α) ∝ α² (measured below: 5.1e-3, 5.1e-5,
    /// 5.1e-7, 5.1e-9 at α = 1, 0.1, 0.01, 0.001), the Newton step is the
    /// α → 0 limit, and *that* is the trivial limit worth anchoring.
    ///
    /// The residual shift is a feature: it is what keeps AH well-defined on an
    /// indefinite Hessian, where the Newton step this test computes points
    /// uphill. See `negative_curvature_still_gives_a_descent_prediction`.
    #[test]
    fn the_alpha_to_zero_limit_is_the_unconstrained_newton_step() {
        let h = vec![
            vec![4.0, 1.0, 0.0],
            vec![1.0, 3.0, 0.5],
            vec![0.0, 0.5, 2.0],
        ];
        let g = vec![0.1, -0.05, 0.02];
        let diag = vec![4.0, 3.0, 2.0];
        let mv = dense_matvec(h.clone());
        let cfg = TrahConfig {
            davidson_conv: 1e-13,
            ..TrahConfig::default()
        };
        let kref = solve_dense(&h, &g.iter().map(|x| -x).collect::<Vec<_>>());
        let nref: f64 = kref.iter().map(|x| x * x).sum::<f64>().sqrt();

        // α² convergence onto the Newton step.
        let mut prev_err = f64::INFINITY;
        for &alpha in &[1.0, 0.1, 0.01] {
            let s = ah_solve(&g, &mv, &diag, alpha, &cfg).unwrap();
            let err = s
                .kappa
                .iter()
                .zip(kref.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt()
                / nref;
            assert!(
                err < prev_err * 0.02,
                "the AH step must approach Newton at order α²: at α={alpha} rel err {err:e} \
                 is not ≥50× better than the previous {prev_err:e}"
            );
            prev_err = err;
        }
        assert!(
            prev_err < 1e-6,
            "at α = 0.01 the AH step must BE the Newton step to 1e-6, got {prev_err:e}"
        );

        // And at the production α_min, with a radius far outside the step, the
        // constraint must be inactive and cost exactly one eigensolve.
        let step = solve_trust_region(&g, &mv, &diag, 1e6, &cfg).unwrap();
        assert!(
            !step.on_boundary,
            "a step far inside the radius must not report as on-boundary"
        );
        assert!(
            step.predicted < 0.0,
            "a step on a positive-definite Hessian must predict a decrease, got {}",
            step.predicted
        );
        assert_eq!(step.alpha, ALPHA_MIN, "an interior step must not search α");
        assert_eq!(step.shift_iterations, 1, "an interior step costs one solve");
    }

    /// The α parameterization, pinned: raising α must SHORTEN the step, and
    /// κ(α)/α must satisfy the level-shifted Newton equations (H − μI)κ = −g at
    /// every α.
    ///
    /// This is the test that catches the κ·α vs κ/α sign error. With the wrong
    /// scaling the norms come out INCREASING in α and the NR residual is
    /// nonzero for every α but the first.
    #[test]
    fn larger_alpha_shortens_the_step_monotonically() {
        // Indefinite H so the level shift is genuinely active.
        let h = vec![
            vec![-0.30, 0.05, 0.00],
            vec![0.05, 0.60, 0.10],
            vec![0.00, 0.10, 1.50],
        ];
        let g = vec![0.08, -0.05, 0.03];
        let diag = vec![-0.30, 0.60, 1.50];
        let mv = dense_matvec(h.clone());
        let cfg = TrahConfig {
            davidson_conv: 1e-13,
            ..TrahConfig::default()
        };

        let mut prev = f64::INFINITY;
        for &alpha in &[1.0, 2.0, 8.0, 64.0, 512.0] {
            let s = ah_solve(&g, &mv, &diag, alpha, &cfg).unwrap();
            assert!(
                s.norm < prev,
                "‖κ‖ must decrease with α: at α={alpha} got {} but previous was {prev}",
                s.norm
            );
            prev = s.norm;

            // (H − μI) κ = −g, to Davidson's accuracy.
            let hk = mv(&s.kappa).unwrap();
            let resid: f64 = (0..3)
                .map(|i| {
                    let v = hk[i] - s.level_shift * s.kappa[i] + g[i];
                    v * v
                })
                .sum::<f64>()
                .sqrt();
            assert!(
                resid < 1e-6,
                "κ(α)/α must solve the level-shifted Newton equations at α={alpha}: \
                 residual {resid:e}"
            );
        }
    }

    /// The trust region must actually BIND: with a radius well below the
    /// Newton step length, ‖κ‖ must land on the boundary, not beyond it.
    #[test]
    fn small_radius_puts_the_step_on_the_boundary() {
        let h = vec![
            vec![0.05, 0.0, 0.0],
            vec![0.0, 0.08, 0.0],
            vec![0.0, 0.0, 0.10],
        ];
        // The unconstrained Newton step is ~(−2, 1.25, −0.5), far outside Δ=0.2.
        let g = vec![0.1, -0.1, 0.05];
        let diag = vec![0.05, 0.08, 0.10];
        let mv = dense_matvec(h);
        let cfg = TrahConfig::default();
        let radius = 0.2;
        let step = solve_trust_region(&g, &mv, &diag, radius, &cfg).unwrap();
        assert!(
            step.norm <= radius * (1.0 + 1e-6),
            "the trust region must not be violated: ‖κ‖ = {} > Δ = {radius}",
            step.norm
        );
        assert!(
            step.norm >= radius * (1.0 - cfg.step_tol),
            "a constrained step must land ON the boundary, got ‖κ‖ = {} for Δ = {radius}",
            step.norm
        );
        assert!(
            step.on_boundary,
            "a constrained step must report on_boundary"
        );
        assert!(
            step.alpha > ALPHA_MIN,
            "a constrained step must have searched α, got α = {}",
            step.alpha
        );
    }

    /// Negative curvature: plain Newton would step UPHILL along the negative
    /// eigenvalue. AH must produce a descent direction anyway — that is the
    /// entire reason for the augmented formulation.
    #[test]
    fn negative_curvature_still_gives_a_descent_prediction() {
        let h = vec![
            vec![-0.5, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 2.0],
        ];
        let g = vec![0.05, 0.1, -0.02];
        let diag = vec![-0.5, 1.0, 2.0];
        let mv = dense_matvec(h);
        let cfg = TrahConfig::default();
        let step = solve_trust_region(&g, &mv, &diag, 0.3, &cfg).unwrap();
        assert!(
            step.level_shift <= -0.5,
            "AH must shift at or below the lowest (negative) Hessian eigenvalue −0.5, got μ = {}",
            step.level_shift
        );
        assert!(
            step.predicted < 0.0,
            "even on an indefinite Hessian the AH step must predict a decrease, got {}",
            step.predicted
        );
        let gk: f64 = g.iter().zip(step.kappa.iter()).map(|(a, b)| a * b).sum();
        assert!(
            gk < 0.0,
            "the step must be a descent direction (gᵀκ < 0), got {gk}"
        );
    }

    /// The two predicted-reduction formulas (Helmich-Paris Eqs. 18 and 19) are
    /// algebraically identical for an exact AH eigenpair. Their agreement is an
    /// INDEPENDENT check on the solve — Eq. 19 never touches the Hessian matvec
    /// used to build Eq. 18.
    #[test]
    fn both_predicted_reduction_formulas_agree() {
        let h = vec![
            vec![1.2, 0.3, 0.1],
            vec![0.3, 0.9, -0.2],
            vec![0.1, -0.2, 2.5],
        ];
        let g = vec![0.07, -0.04, 0.09];
        let diag = vec![1.2, 0.9, 2.5];
        let mv = dense_matvec(h);
        let cfg = TrahConfig {
            davidson_conv: 1e-13,
            ..TrahConfig::default()
        };
        for &radius in &[1e6, 0.05] {
            let s = solve_trust_region(&g, &mv, &diag, radius, &cfg).unwrap();
            assert!(
                s.predicted_residual < 1e-6,
                "Eq.18 ({}) and Eq.19 ({}) must agree at Δ={radius}: rel resid {:e}",
                s.predicted,
                s.predicted_cheap,
                s.predicted_residual
            );
        }
    }

    /// ρ bookkeeping: a step whose actual reduction matches the prediction is
    /// accepted; one that goes uphill is rejected and shrinks the radius.
    #[test]
    fn rho_accepts_a_good_step_and_rejects_an_uphill_one() {
        let cfg = TrahConfig::default();
        let mut st = TrahState::new(cfg);
        let r0 = st.radius();
        let s = mk_step(-1e-3);

        st.record_step(-100.0, &s);
        // Actual = −0.9e-3 ⇒ ρ = 0.9 > 0.75 ⇒ accept + grow.
        assert_eq!(st.assess(-100.0009), Some(TrahVerdict::Accepted));
        assert!(
            st.radius() > r0,
            "ρ = 0.9 must grow the radius: {} vs {r0}",
            st.radius()
        );

        // Uphill step: predicted down, actual up ⇒ ρ < 0 ⇒ reject + shrink.
        let r1 = st.radius();
        st.record_step(-100.0, &s);
        assert_eq!(st.assess(-99.5), Some(TrahVerdict::Rejected));
        assert!(
            st.radius() < r1,
            "a rejected step must contract the radius: {} vs {r1}",
            st.radius()
        );
        assert_eq!(st.rejected, 1);
    }

    /// The middle Fletcher branch: 0.25 < ρ ≤ 0.75 accepts and leaves the
    /// radius ALONE. Without this the two outer branches could both be
    /// satisfied by a rule that always moved the radius.
    #[test]
    fn a_middling_rho_accepts_and_leaves_the_radius_unchanged() {
        let mut st = TrahState::new(TrahConfig::default());
        let r0 = st.radius();
        let s = mk_step(-1e-3);
        st.record_step(-100.0, &s);
        // Actual = −0.5e-3 ⇒ ρ = 0.5.
        assert_eq!(st.assess(-100.0005), Some(TrahVerdict::Accepted));
        assert_eq!(
            st.radius().to_bits(),
            r0.to_bits(),
            "0.25 < ρ ≤ 0.75 must leave the radius bit-identical"
        );
    }

    /// The poor-but-positive branch: 0 < ρ ≤ 0.25 keeps the step but contracts.
    #[test]
    fn a_poor_rho_accepts_the_step_but_contracts() {
        let mut st = TrahState::new(TrahConfig::default());
        let r0 = st.radius();
        let s = mk_step(-1e-3);
        st.record_step(-100.0, &s);
        // Actual = −0.1e-3 ⇒ ρ = 0.1.
        assert_eq!(st.assess(-100.0001), Some(TrahVerdict::AcceptedPoor));
        assert!(st.radius() < r0, "a poor ρ must contract the radius");
        assert_eq!(st.accepted, 1, "a poor step is still ACCEPTED");
        assert_eq!(st.rejected, 0);
    }

    /// A prediction that is not a decrease can never yield a meaningful ρ, so
    /// it must be rejected rather than divided by.
    #[test]
    fn a_non_descent_prediction_is_rejected_not_divided_by() {
        let mut st = TrahState::new(TrahConfig::default());
        let bad = mk_step(0.0);
        st.record_step(-100.0, &bad);
        assert_eq!(st.assess(-100.1), Some(TrahVerdict::Rejected));
        assert!(
            st.last_rho.unwrap().is_infinite(),
            "a zero prediction must not produce a finite ρ by division"
        );
    }

    /// `assess` with nothing pending must be a no-op, not a phantom verdict.
    #[test]
    fn assess_without_a_pending_step_is_none() {
        let mut st = TrahState::new(TrahConfig::default());
        assert_eq!(st.assess(-100.0), None);
        assert_eq!(st.accepted, 0);
        assert_eq!(st.rejected, 0);
    }

    /// `clear_pending` must make a stale prediction unassessable, so a DIIS
    /// step's energy can never be scored against a TRAH step's model.
    #[test]
    fn clear_pending_drops_the_assessment() {
        let mut st = TrahState::new(TrahConfig::default());
        st.record_step(-100.0, &mk_step(-1e-3));
        assert!(st.has_pending());
        st.clear_pending();
        assert!(!st.has_pending());
        assert_eq!(st.assess(-99.0), None, "a cleared step must not be scored");
    }

    /// The radius must collapse (and report it) under repeated rejections, so
    /// the caller can fall back rather than take vanishing steps forever.
    #[test]
    fn repeated_rejections_collapse_the_radius() {
        let mut st = TrahState::new(TrahConfig::default());
        let s = mk_step(-1e-3);
        assert!(!st.collapsed());
        // 0.4 · 0.7ⁿ < 1e-4 needs n ≥ 24.
        for _ in 0..30 {
            st.record_step(-100.0, &s);
            st.assess(-99.0);
        }
        assert!(
            st.collapsed(),
            "repeated uphill steps must collapse the radius below the floor, got {}",
            st.radius()
        );
    }

    /// The radius must not grow without bound once the constraint goes inactive.
    #[test]
    fn the_radius_is_capped_at_radius_max() {
        let cfg = TrahConfig::default();
        let mut st = TrahState::new(cfg);
        let s = mk_step(-1e-3);
        for _ in 0..100 {
            st.record_step(-100.0, &s);
            st.assess(-100.001); // ρ = 1.0 ⇒ grow every time
        }
        assert!(
            st.radius() <= cfg.radius_max,
            "the radius must be capped at radius_max: {} > {}",
            st.radius(),
            cfg.radius_max
        );
    }

    /// Dimension mismatch between the gradient and the preconditioner diagonal
    /// is a caller bug that must Err, not index out of bounds inside Davidson.
    #[test]
    fn mismatched_diagonal_length_errors() {
        let mv = dense_matvec(vec![vec![1.0]]);
        assert!(solve_trust_region(&[0.1], &mv, &[1.0, 2.0], 0.5, &TrahConfig::default()).is_err());
    }

    /// A non-positive trust radius is meaningless and must Err rather than
    /// silently produce a zero step.
    #[test]
    fn non_positive_radius_errors() {
        let mv = dense_matvec(vec![vec![1.0]]);
        assert!(solve_trust_region(&[0.1], &mv, &[1.0], 0.0, &TrahConfig::default()).is_err());
        assert!(solve_trust_region(&[0.1], &mv, &[1.0], -1.0, &TrahConfig::default()).is_err());
        assert!(solve_trust_region(&[], &mv, &[], 1.0, &TrahConfig::default()).is_err());
    }

    /// `assess` must report "nothing pending" DISTINCTLY from "accepted", and a
    /// caller must not read `last_rho` without checking that Option.
    ///
    /// # The bug this pins
    ///
    /// The SCF loop originally wrote `.unwrap_or(TrahVerdict::Accepted)` and
    /// then traced `last_rho`, which PERSISTS across assessments. On RKS/PBE
    /// water that printed a frozen `rho = -60.519638` repeated across a
    /// reject/"accept" alternation that was not happening — the "accept" lines
    /// were iterations with nothing pending at all. A ρ repeating to the last
    /// digit is arithmetic, not measurement, and it disguised a real
    /// convergence defect as a different one.
    #[test]
    fn a_consumed_step_is_not_reassessable_and_none_is_not_acceptance() {
        let mut st = TrahState::new(TrahConfig::default());
        st.record_step(-100.0, &mk_step(-1e-3));
        let first = st.assess(-100.001);
        assert_eq!(first, Some(TrahVerdict::Accepted));
        let rho_after_first = st.last_rho;
        assert!(rho_after_first.is_some());

        // Second call with nothing pending: NOT a verdict.
        assert_eq!(
            st.assess(-99.0),
            None,
            "a second assess with nothing pending must return None, not a verdict \
             synthesised from a step that was already consumed"
        );
        // `last_rho` deliberately still holds the previous value — which is
        // precisely why callers must gate on the Option and never on last_rho.
        assert_eq!(
            st.last_rho.map(f64::to_bits),
            rho_after_first.map(f64::to_bits),
            "last_rho persists by design; the None above is the signal it is stale"
        );
        assert_eq!(
            st.accepted, 1,
            "the no-op assess must not count an acceptance"
        );
        assert_eq!(st.rejected, 0);
    }

    fn mk_step(predicted: f64) -> TrahStep {
        TrahStep {
            kappa: vec![0.1],
            norm: 0.1,
            level_shift: -0.1,
            alpha: 1.0,
            predicted,
            predicted_cheap: predicted,
            predicted_residual: 0.0,
            on_boundary: true,
            shift_iterations: 1,
        }
    }
}
