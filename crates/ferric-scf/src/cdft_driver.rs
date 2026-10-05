//! cDFT outer driver: nested optimization. For fixed λ the inner UHF/UKS solve
//! adds Σ_C λ_C W^C to the Fock; the outer loop drives the residual
//! c_C(λ) = N_C\[ρ_λ\] − target_C to zero.
//!
//! One constraint (k = 1, the tested case) is solved by `solve_scalar`:
//! Newton with a finite-difference Jacobian, a sign-change bracket, and guards
//! for the two ways c(λ) stops being a smooth function in practice — an inner
//! SCF that does not converge, and an inner SCF that lands in a different
//! basin at λ + h than at λ. Several constraints use a plain k × k
//! finite-difference Newton step.

use crate::result::ScfResult;
use crate::rhf::RhfConfig;
use crate::screening::SchwarzBounds;
use crate::uhf::solve_uhf_fockmod;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_dft::ao_grid::eval_basis_on_points;
use ferric_dft::cdft::{build_weight_matrix, population, SpinChannel};
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::Array2;

/// Is the λ-Newton per-iteration trace switched on?
///
/// Diagnosing an outer loop that fails to converge needs the ITERATE, not just
/// the final error: "monotone but slow", "limit cycle" and "the residual
/// function is discontinuous because the inner SCF changed basin" all present
/// as the same `Convergence` error and have different fixes. The trace is
/// behind an env var rather than a config field because it is a debugging aid,
/// not a physics knob — adding it to `RhfConfig` would put it in the CLI's
/// `deny_unknown_fields` TOML surface and in every struct literal.
fn trace_enabled() -> bool {
    std::env::var("FERRIC_CDFT_TRACE").is_ok_and(|v| v != "0" && !v.is_empty())
}

/// A sign-change bracket on the single-constraint residual `c(λ)`, and the
/// safeguard that keeps the λ-Newton loop inside it.
///
/// # The measured failure this exists to stop
///
/// `tests/cdft_outer_loop.rs` traces HeNe⁺/def2-SVP from a MINAO start at
/// targets 2.000 / 1.995 / 1.990 / 1.980 / 1.954484. The first, fourth and
/// fifth converge; 1.995 and 1.990 do not, and the failure is an ISLAND rather
/// than a difficulty gradient — the target FURTHEST from the natural population
/// converges and two nearer ones do not. All five runs are identical through
/// outer 4 and all five then overshoot into a region near λ ≈ −3.0 where the
/// inner SCF crosses into an over-filled-He state. They differ only in whether
/// the next step lands back on the physical branch.
///
/// Three things then go wrong for the two that do not, in causal order:
///
/// 1. **`c(λ)` is discontinuous, so the FD Jacobian is meaningless.** Every
///    inner solve restarts from the same fixed guess (it is never warm-started
///    from the previous λ), so it may pick a different basin at each λ. The
///    `fd = 1e-3` probe then straddles the jump and divides a state change by
///    1e-3. Measured Jacobians across one run: `−0.137, −0.018, −0.009, −0.031,
///    +0.204, −148.3, +240.5, −247.5` — three sign flips and four orders of
///    magnitude, from two solves 0.001 apart in λ differing by 0.826 Ha.
/// 2. **The clamp converts that into an exact limit cycle.** With both FD
///    points inside the flipped state the Jacobian goes small (−0.0243) against
///    a residual of 0.896, so the raw Newton step is −36.90; `clamp(−1, 1)`
///    truncates it to −1.0, and the return step (+1.053) truncates to +1.0. λ
///    then alternates between exactly −3.3266069180979265 and
///    −2.3266069180979265 — a gap of exactly 1.0, the clamp width — with
///    bit-identical residuals and energies for twenty iterations.
/// 3. Nothing detects the stall, so the loop spends its whole budget on two
///    repeating states before reporting a bare "did not converge".
///
/// # Why bracketing, and not a better Jacobian or a line search
///
/// Both of those treat the derivative as recoverable. It is not: `c(λ)` genuinely
/// jumps, so no step size and no FD width makes the local linear model valid
/// across the jump. What survives a discontinuity is the INTERMEDIATE VALUE
/// property — `c` is negative below the root and positive above it, and the trace
/// exhibits exactly such a pair (`c = −0.014` at λ = −2.43, `c = +0.064` at
/// λ = −3.08). Bisection needs no derivative, so defect (1) cannot affect it,
/// and it halves the interval every step, so defect (2) cannot occur.
///
/// This is the standard safeguarded-Newton (`rtsafe`) structure: keep Newton's
/// fast local convergence, fall back on bisection whenever Newton proposes a
/// point outside the bracket.
///
/// # When it fires — and the prediction about this that was REFUTED
///
/// The safeguard fires ONLY when a two-sided bracket exists AND the clamped
/// Newton iterate falls outside it. That condition is never met while Newton is
/// behaving, which is pinned by
/// `bracket_safeguard_is_inert_while_newton_stays_inside`.
///
/// It was PREDICTED from that, before measuring, that the safeguard would
/// therefore be a bit-identical no-op on every path that already converged —
/// and in particular on the hcore-started path the downstream suites are
/// baselined against. **That prediction was wrong, and the exactness anchor is
/// what caught it.** On the baselined hcore integer-target run the safeguard
/// fires twice, because that run was ALSO leaving the bracket: pre-fix it
/// wandered into the flipped state at outer 5 and passed through SIX
/// CONSECUTIVE inner solves that hit their 400-iteration cap without converging
/// before stumbling back out at outer 12. It reached the right answer by luck,
/// not by the loop working.
///
/// So bit-identity on that path would have meant preserving a trajectory
/// through six unconverged solves — i.e. preserving the defect. Post-fix it
/// converges in 8 outer iterations instead of 16, every inner solve converged,
/// to the same constrained solution within 7.7e-8 Ha (2.1e-6 eV) and 2.8e-8 in
/// N_C, with `|N_C − 2.0|` slightly SMALLER than before. That is 130x inside
/// the 1e-5 bar the downstream suites assert on this energy and 8500x inside
/// their 1e-3 bar on λ.
///
/// The measurement, the deltas and those tolerances are recorded in
/// `tests/cdft_outer_loop.rs::hcore_started_path_reaches_the_same_constrained_
/// solution`, which was re-scoped from bit-identity to "same solution, inside
/// the downstream bars" for exactly this reason rather than deleted.
///
/// Only `k == 1` is safeguarded. A bracket is a scalar notion; with several
/// constraints there is no ordering to bisect, and the multi-constraint path is
/// left exactly as it was rather than given a pretend-bracket.
#[derive(Debug, Default, Clone, Copy)]
struct Bracket {
    /// The best (root-nearest) observed point with `c < 0`, as `(λ, c)`.
    neg: Option<(f64, f64)>,
    /// The best (root-nearest) observed point with `c > 0`.
    pos: Option<(f64, f64)>,
}

impl Bracket {
    /// Record an observed `(λ, c)` pair, tightening the bracket if it can.
    ///
    /// # Why "tighter" cannot be decided by comparing λ alone
    ///
    /// A first version of this stored the most-positive λ among `c < 0` and the
    /// most-negative λ among `c > 0`. That silently assumed λ INCREASES toward
    /// the root from the negative-residual side — and on this problem it does
    /// the opposite: the constraint well deepens as λ goes NEGATIVE, so the
    /// `c < 0` points run 0.0, −1.0, −1.5, −2.5 toward the root while `c > 0`
    /// sits at −3.0. The comparison kept λ = 0.0 as the bound forever, the
    /// bracket never tightened below [0, −3], and bisection returned −1.5 every
    /// time. That converted the baselined hcore run — which converged in 16
    /// iterations before — into its OWN period-2 cycle (−1.5, −2.5, −1.5, …).
    ///
    /// The exactness anchor caught it on the first run. It is recorded here
    /// rather than silently corrected because the docstring on that version
    /// already CLAIMED to be direction-agnostic while the code was not: the
    /// comment was the thing that was wrong, and reading it instead of the loop
    /// is what let the bug in.
    ///
    /// The fix is to define "tighter" against the OTHER side, which needs no
    /// assumption about direction: among points of one sign, the best bound is
    /// the one CLOSEST to the opposite bound. Until an opposite bound exists
    /// there is no bracket to bisect anyway, so the newest point is simply kept.
    ///
    /// Only points that tighten the interval are kept, so the bracket is
    /// monotonically non-widening and bisection cannot be sent backwards by a
    /// stale outlier.
    ///
    /// # Why `converged` is a parameter and not a caller-side `if`
    ///
    /// An unconverged inner SCF's population is NOT a value of `c(λ)` — it is
    /// wherever the density happened to be when the iteration cap hit — so
    /// bracketing on one would pin the root against a number that is not on the
    /// curve, and the bisection midpoint would be meaningless. This is not
    /// hypothetical: at target 1.990 the inner SCF hits its 400-iteration cap at
    /// four different λ in a single outer loop, and the pre-fix hcore run passes
    /// through six consecutive capped solves.
    ///
    /// The filter started life as `if k == 1 && scf.converged` at the call site
    /// and was moved in here because MUTATION TESTING FOUND IT UNREACHABLE:
    /// deleting it left the entire suite — unit and integration, including the
    /// 1.990 sweep it was written for — completely green. That is the same
    /// failure mode `constraint_offset` and `accepts_candidate` in this file
    /// were already extracted for. Owning the decision here makes it directly
    /// testable; see `bracket_ignores_an_unconverged_inner_solve`.
    ///
    /// Being currently unreachable does NOT make it wrong — the safeguard's
    /// value is in the runs that fail, and on those the capped solves are real.
    /// It makes it an ASSUMPTION until tested, which is what the extraction and
    /// that test fix.
    fn observe(&mut self, lam: f64, c: f64, converged: bool) {
        if !converged || !c.is_finite() || !lam.is_finite() || c == 0.0 {
            return;
        }
        let (this, other) = if c < 0.0 {
            (&mut self.neg, self.pos)
        } else {
            (&mut self.pos, self.neg)
        };
        match (*this, other) {
            // No bound of this sign yet: take it.
            (None, _) => *this = Some((lam, c)),
            // No opposite bound yet, so "tighter" is undefined; keep the most
            // recent, which is the iterate the loop is actually working from.
            (Some(_), None) => *this = Some((lam, c)),
            // Both exist: keep whichever of this sign is nearer the opposite
            // bound. Direction-free by construction.
            (Some((cur, _)), Some((opp, _))) => {
                if (lam - opp).abs() < (cur - opp).abs() {
                    *this = Some((lam, c));
                }
            }
        }
    }

    /// Given the λ the Newton step just proposed, return `Some(midpoint)` if it
    /// must be overridden, or `None` to keep it.
    ///
    /// `None` whenever there is no two-sided bracket yet, or the proposal lies
    /// strictly inside it. The comparison is strict at both ends: a proposal
    /// exactly ON a bound is already a point whose residual is known and
    /// nonzero, so re-evaluating it would repeat an iteration — precisely the
    /// cycle being fixed.
    fn safeguard(&self, newton_lam: f64) -> Option<f64> {
        let (neg, _) = self.neg?;
        let (pos, _) = self.pos?;
        let (a, b) = if neg <= pos { (neg, pos) } else { (pos, neg) };
        if newton_lam.is_finite() && newton_lam > a && newton_lam < b {
            return None;
        }
        Some(0.5 * (a + b))
    }
}

/// Finite-difference step in λ for the outer-loop Jacobian.
const FD_STEP: f64 = 1e-3;

/// Largest outer-loop step in λ (Ha per electron), per component.
const MAX_STEP: f64 = 1.0;
/// Smallest backtrack distance before [`ScalarStepper::backtrack`] reports a
/// stall: `FD_STEP × 1e-6`. He₂⁺/def2-SVP at 3.50 Å reached 5e-10 with the
/// inner SCF still unconverged.
const BACKTRACK_FLOOR: f64 = FD_STEP * 1e-6;

/// Below this |dc/dλ| the finite-difference Jacobian is treated as singular.
const SINGULAR_JAC: f64 = 1e-14;

/// Largest accepted Hellmann–Feynman mismatch between a main point and its
/// finite-difference probe, in Ha. See [`hf_mismatch`].
///
/// Placed between two measured populations (HeNe⁺/def2-SVP guess catalogue
/// at ERI precision 1e-14 and 1e-20, He₂⁺/PBE def2-SVP 2.50–3.50 Å, and the
/// `cdft_uhf`/`cdft_outer_loop` suites; every pair whose probe converged):
///
/// * same basin: 359 pairs, 1.8e-14 … 7.0e-9 Ha (inner-SCF noise in V plus
///   the trapezoid error);
/// * different basin: 15 pairs, 5.5e-5 … 0.35 Ha (the V gap between two SCF
///   solutions);
/// * one pair at 2.7e-7 Ha, on a steep stretch of c(λ) (dc/dλ = −6.6), that
///   neither population claims.
///
/// 1e-6 is 140× above the largest same-basin value and 55× below the smallest
/// cross-basin one.
const HF_MISMATCH_TOL: f64 = 1e-6;

/// One evaluation of the single-constraint residual at some λ.
#[derive(Debug, Clone, Copy)]
struct Sample {
    /// c(λ) = N_C[ρ_λ] − target.
    c: f64,
    /// The λ-augmented value V(λ) = E[ρ_λ] + λ·c(λ). On one SCF branch
    /// dV/dλ = c (Hellmann–Feynman), which is what [`hf_mismatch`] checks.
    v: f64,
    /// Did the inner SCF converge? An unconverged density is not a point of
    /// c(λ).
    converged: bool,
}

/// How far a finite-difference pair departs from dV/dλ = c on ONE branch.
///
/// For `a` at λ and `b` at λ + h on the same smooth SCF branch,
/// V(λ+h) − V(λ) = ∫c dλ = h·(c_a + c_b)/2 − h³c''/12, so the mismatch below is
/// the trapezoid error plus inner-SCF noise. If the inner SCF at λ + h fell
/// into a different basin, V jumps by the energy gap between the two
/// solutions and the mismatch is that gap. This detects a basin change
/// directly, whatever the size or sign of the resulting "Jacobian". A bound on
/// dc/dλ could not: the Jacobians of one failing HeNe⁺ run (see [`Bracket`])
/// were −0.137, −0.018, −0.009, −0.031, +0.204, −148.3, +240.5, −247.5, and
/// small negative values like the first four are also what a basin change on a
/// flat plateau produces.
fn hf_mismatch(a: &Sample, b: &Sample, h: f64) -> f64 {
    ((b.v - a.v) - 0.5 * h * (a.c + b.c)).abs()
}

/// The converged outcome of [`solve_scalar`].
struct ScalarRoot<T> {
    lambda: f64,
    outer_iters: usize,
    payload: T,
}

/// What [`ScalarStepper::step`] decided, for the trace.
#[derive(Debug, Clone, Copy, PartialEq)]
enum StepKind {
    /// Newton with the finite-difference Jacobian of this iteration.
    Newton,
    /// The probe was untrusted; Newton with the last trusted Jacobian.
    StaleJacobian,
    /// The probe was untrusted and no trusted Jacobian exists yet; a step of
    /// the trust radius in the direction that lowers |c|.
    SignStep,
}

/// The decision state of the single-constraint outer loop, kept apart from the
/// inner solves so every rule can be tested on synthetic residuals.
///
/// # The step, when nothing is wrong
///
/// Given c at λ and at λ + `FD_STEP`, take J = Δc/`FD_STEP`, step
/// λ ← λ − clamp(c/J, ±radius), and replace the step by the bracket midpoint if
/// it leaves a two-sided sign-change bracket ([`Bracket`]). `radius` is
/// `MAX_STEP` unless guard 1 shrank it. When neither guard fires this is the
/// same arithmetic, in the same order, as the loop without guards (`guards =
/// false`), so such a run is bit-identical to it; pinned by
/// `guards_are_inert_on_a_smooth_residual`.
///
/// # Guard 1: an unconverged inner solve means the step went too far
///
/// If the inner SCF at λ does not converge and an earlier λ did, no Newton step
/// is taken from that density: its c is not a value of c(λ), and a finite
/// difference across it measures nothing. λ backtracks halfway to the last
/// converged λ, and the trust radius shrinks to half the remaining distance so
/// the next Newton step cannot land back on the λ that failed. The radius
/// doubles back toward `MAX_STEP` after every trusted Newton step. With no
/// earlier converged λ there is nothing to backtrack to, and the plain step is
/// taken.
///
/// # Guard 2: a probe in another SCF basin is not a derivative
///
/// Every inner solve starts from the same guess, so the solve at λ + h can land
/// in a different basin from the one at λ. The pair is trusted only if the
/// probe converged, J is finite and non-singular, and [`hf_mismatch`] is at
/// most `HF_MISMATCH_TOL`. An untrusted pair is replaced by the last trusted J
/// (a quasi-Newton step) or, before any J has been trusted, by a step of
/// `radius` in the direction that lowers |c|: along the lowest solution c(λ) is
/// non-increasing, because min over ρ of E + λ·c is concave in λ with
/// derivative c, so c > 0 means λ must increase. The bracket safeguard still
/// applies afterwards.
///
/// # What it does not handle (measured on HeNe⁺/def2-SVP, target N_He = 2)
///
/// * No converged point yet: guard 1 has nothing to back off to. From the
///   `state A (N_He = 1)` orbitals at ERI precision 1e-20 the first 23 inner
///   solves hit their 400-iteration cap, so the plain step is taken from each.
/// * A converged branch on which dc/dλ > 0, with no sign change seen: the same
///   run then alternates between λ = −0.2324 (c = −0.961, J = +0.199) and
///   +0.7676 (c = −0.992, J = −0.011), one clamp width apart. Both probes are
///   on their main point's branch, both residuals are negative, so neither
///   guard nor the bracket fires. The same start converges at precision
///   1e-14.
/// * A start far from the root, under an outer cap too small to reach it. The
///   step is clamped to `MAX_STEP` per iteration, so arriving from λ⁰ takes at
///   least |λ_root − λ⁰| / `MAX_STEP` iterations whatever the residual does,
///   and the error on running out is a bare "did not converge" that cannot be
///   told from a stall. Measured on the six He₂⁺/PBE diabats
///   (`tests/validation_cdft_et.rs::he2_plus_diabats_converge_from_lambda_zero`):
///   from λ = 0, with roots at λ ≈ 1.55-2.46, every point is on a monotone
///   trajectory with a two-sided bracket and |c| already 4.1e-5 … 6.9e-4 at
///   outer 8, and all six converge to the same root as a start AT the root
///   given 11-21 iterations. Nothing here is wrong in that case — the budget
///   is — so no guard fires and none should.
#[derive(Debug, Clone)]
struct ScalarStepper {
    guards: bool,
    bracket: Bracket,
    last_good: Option<f64>,
    last_jac: Option<f64>,
    radius: f64,
}

impl ScalarStepper {
    fn new(guards: bool) -> Self {
        Self {
            guards,
            bracket: Bracket::default(),
            last_good: None,
            last_jac: None,
            radius: MAX_STEP,
        }
    }

    /// Guard 1. Called with an inner solve at `lam` that was not within
    /// tolerance. Returns `Some(next λ)` to backtrack without a probe, or
    /// `None` to go on to the probe and [`Self::step`].
    ///
    /// Errors once the backtrack distance falls below [`BACKTRACK_FLOOR`]: the
    /// inner SCF fails arbitrarily close to the last converged λ, so further
    /// halving would only spend the remaining `cdft_max_outer` iterations,
    /// each a full inner SCF, without progress.
    fn backtrack(&mut self, lam: f64, s: &Sample) -> Result<Option<f64>, FerricError> {
        if !self.guards || s.converged {
            return Ok(None);
        }
        let Some(good) = self.last_good else {
            return Ok(None);
        };
        let back = good + 0.5 * (lam - good);
        if (back - good).abs() < BACKTRACK_FLOOR {
            return Err(FerricError::Convergence(format!(
                "cDFT outer loop stalled: the inner SCF does not converge within \
                 {BACKTRACK_FLOOR:e} of the last converged lambda {good:+.12} \
                 (unconverged at {lam:+.12})"
            )));
        }
        self.radius = self.radius.min(0.5 * (back - good).abs());
        Ok(Some(back))
    }

    /// Record the main point `s` at `lam` (before the probe is run).
    fn record(&mut self, lam: f64, s: &Sample) {
        self.bracket.observe(lam, s.c, s.converged);
        if s.converged {
            self.last_good = Some(lam);
        }
    }

    /// Is the pair (`s` at λ, `p` at λ + `FD_STEP`) a usable derivative?
    fn trusts(&self, s: &Sample, p: &Sample) -> bool {
        let jac = (p.c - s.c) / FD_STEP;
        // An unconverged MAIN point reaches the probe only before any λ has
        // converged; then there is no better model than this one.
        !self.guards
            || !s.converged
            || (p.converged
                && jac.is_finite()
                && jac.abs() >= SINGULAR_JAC
                && hf_mismatch(s, p, FD_STEP) <= HF_MISMATCH_TOL)
    }

    /// Guard 2 plus the bracket safeguard: the next λ from `lam`, its main
    /// point `s` and its probe `p`. Errors only when a TRUSTED Jacobian is
    /// singular, which with guards on requires an unconverged main point.
    fn step(&mut self, lam: f64, s: &Sample, p: &Sample) -> Result<(f64, StepKind), FerricError> {
        let jac = (p.c - s.c) / FD_STEP;
        let trusted = self.trusts(s, p);
        let (slope, kind) = if trusted {
            if jac.abs() < SINGULAR_JAC {
                return Err(FerricError::Lapack("cDFT Jacobian singular".into()));
            }
            if self.guards && s.converged {
                self.last_jac = Some(jac);
            }
            (Some(jac), StepKind::Newton)
        } else if self.last_jac.is_some() {
            (self.last_jac, StepKind::StaleJacobian)
        } else {
            (None, StepKind::SignStep)
        };
        let mut next = lam;
        match slope {
            Some(j) => next -= (s.c / j).clamp(-self.radius, self.radius),
            None => next += if s.c > 0.0 { self.radius } else { -self.radius },
        }
        if self.guards && trusted {
            self.radius = (2.0 * self.radius).min(MAX_STEP);
        }
        if let Some(safe) = self.bracket.safeguard(next) {
            next = safe;
        }
        Ok((next, kind))
    }
}

/// The single-constraint (k = 1) outer loop: a root of c(λ) = 0 to `tol`,
/// driven by [`ScalarStepper`] (read that for the rules).
///
/// `eval(λ, label)` runs one inner solve and returns its [`Sample`] plus a
/// payload (the SCF result) that is handed back for the accepted λ.
fn solve_scalar<T>(
    eval: &mut dyn FnMut(f64, &str) -> Result<(Sample, T), FerricError>,
    lam0: f64,
    tol: f64,
    max_outer: usize,
    guards: bool,
    trace: bool,
) -> Result<ScalarRoot<T>, FerricError> {
    let mut lam = lam0;
    let mut st = ScalarStepper::new(guards);

    for outer in 1..=max_outer {
        if trace {
            eprintln!(
                "[cdft-trace] outer={outer:2}  lam={lam:+.12}  radius={:.3e}",
                st.radius
            );
        }
        let (s, payload) = eval(lam, "main")?;
        if s.c.abs() < tol {
            return Ok(ScalarRoot {
                lambda: lam,
                outer_iters: outer,
                payload,
            });
        }
        if let Some(back) = st.backtrack(lam, &s)? {
            if trace {
                eprintln!(
                    "[cdft-trace]   BACKTRACK: inner SCF unconverged; halving toward \
                     the last converged lam={:+.12} -> {back:+.12}, radius {:.3e}",
                    st.last_good.unwrap_or(f64::NAN),
                    st.radius
                );
            }
            lam = back;
            continue;
        }
        st.record(lam, &s);
        let (p, _) = eval(lam + FD_STEP, "probe")?;
        let (next, kind) = st.step(lam, &s, &p)?;
        if trace {
            eprintln!(
                "[cdft-trace]   jac={:+.6e}  hf_mismatch={:.3e}  probe_conv={}  \
                 step={kind:?}  bracket=[{:.6}, {:.6}]  -> lam={next:+.12}",
                (p.c - s.c) / FD_STEP,
                hf_mismatch(&s, &p, FD_STEP),
                p.converged,
                st.bracket.neg.map_or(f64::NAN, |(l, _)| l),
                st.bracket.pos.map_or(f64::NAN, |(l, _)| l),
            );
        }
        lam = next;
    }

    Err(FerricError::Convergence(format!(
        "cDFT outer loop did not converge in {max_outer} iters"
    )))
}

/// Result of a constrained SCF.
#[derive(Debug, Clone)]
#[must_use]
pub struct CdftResult {
    /// Inner SCF result at the converged λ (energy is the ordinary KS energy at
    /// the constrained density — the constraint term is already excluded).
    pub scf: ScfResult,
    /// Converged Lagrange multipliers, one per constraint.
    pub lambdas: Vec<f64>,
    /// Final fragment populations `N_C[ρ_λ]`.
    pub populations: Vec<f64>,
    /// Outer-loop iterations taken.
    pub outer_iters: usize,
    /// The constraint weight operators `W^C` (AO basis, one per constraint, in
    /// `config.constraints` order) that the solve actually used.
    ///
    /// Carried so a caller that needs them afterwards — the Wu–Van Voorhis
    /// coupling ([`crate::cdft_coupling::DiabaticState::w`]) — gets the SAME
    /// matrices the constraint was solved with, rather than re-running the grid
    /// build and hoping its grid config matches. They are density-independent
    /// and nbf² each, so holding them costs nothing next to the MO sets.
    pub weight_matrices: Vec<Array2<f64>>,
}

/// Solve constrained UHF/UKS. Reads `config.constraints` and
/// `config.cdft_lambda_tol`. Requires at least one constraint.
///
/// Starts every inner solve from the configured default guess and the
/// λ-Newton loop from λ = 0; [`solve_cdft_uhf_seeded`] is the same solve from
/// a caller-supplied starting point.
pub fn solve_cdft_uhf(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &BasisSet,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
) -> Result<CdftResult, FerricError> {
    solve_cdft_uhf_seeded(ctx, mol, prep, bs, bounds, config, CdftSeed::default())
}

/// A caller-supplied starting point for [`solve_cdft_uhf_seeded`].
///
/// The constrained problem can have several solutions at one target (HeNe⁺ at
/// the integer Becke target is the measured case), and WHICH one a solve
/// returns is decided by where it starts. This lets a caller choose: e.g. start
/// from another code's converged orbitals to test whether ferric stays on that
/// code's state.
///
/// `Default` (both `None`) reproduces [`solve_cdft_uhf`] exactly.
#[derive(Debug, Clone, Copy, Default)]
pub struct CdftSeed<'a> {
    /// α and β MO coefficients, each `(nbf, nbf)` with the occupied columns
    /// first (the `solve_uhf_fockmod` guess contract). Used as the guess at
    /// EVERY inner solve of the first λ-Newton loop, not only the first one —
    /// the same re-seeding the stability descent relies on to stay in the
    /// basin it aimed at.
    pub mos: Option<(&'a Array2<f64>, &'a Array2<f64>)>,
    /// Starting multipliers, one per constraint in `config.constraints` order.
    /// `None` falls back to `config.cdft_lambda_init`, and λ = 0 is used only
    /// when both are absent. Only the first λ-Newton loop uses the start; a
    /// stability-descent restart starts at λ = 0 as before.
    pub lambdas: Option<&'a [f64]>,
}

/// [`solve_cdft_uhf`] from a caller-supplied starting point (see [`CdftSeed`]).
///
/// The stability descent (`config.cdft_stability_descent`) still runs on the
/// seeded solution; a caller who wants to know only whether the seed's state
/// is a fixed point should switch it off.
pub fn solve_cdft_uhf_seeded(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &BasisSet,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    seed: CdftSeed<'_>,
) -> Result<CdftResult, FerricError> {
    let cons = &config.constraints;
    if cons.is_empty() {
        return Err(FerricError::General(
            "solve_cdft_uhf: no constraints".into(),
        ));
    }
    let k = cons.len();
    let zero_lambdas = vec![0.0_f64; k];
    // Starting multipliers: an explicit seed wins, then
    // `RhfConfig::cdft_lambda_init`, then 0.
    let lam_start: &[f64] = match seed.lambdas.or(config.cdft_lambda_init.as_deref()) {
        None => &zero_lambdas,
        Some(l) if l.len() == k && l.iter().all(|v| v.is_finite()) => l,
        Some(l) => {
            return Err(FerricError::General(format!(
                "solve_cdft_uhf_seeded: {} starting multipliers for {k} constraints \
                 (or a non-finite value): {l:?}",
                l.len()
            )))
        }
    };

    // Build the DFT grid + AO values once, then W^C per constraint once.
    // The weight quadrature must be converged tighter than cdft_lambda_tol or
    // the constraint can never be satisfied; the default (75,110) grid plateaus
    // at ~1e-4 on a population, so use the 302-pt angular grid (the Lebedev
    // table max), which recovers populations to ~1e-8. Honor an explicit
    // config.dft_grid if the caller set one.
    let grid_cfg = config.dft_grid.clone().unwrap_or(AtomicGridConfig {
        n_radial: 99,
        n_angular: 302,
        ..Default::default()
    });
    let grid = build_atomic_grid(mol, &grid_cfg);
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    // Pre-flight the AO-grid buffer against the memory budget.
    //
    // This path is the largest ungated grid allocation in ferric-scf: the cDFT
    // default grid is 99x302 = 29,898 points/atom, roughly 3.6x the 75x110
    // production default, so `chi` alone is 12.2 GB at danuglipron/def2-SVP
    // scale (73 atoms, nbf ~ 700) and 27.9 GB at def2-TZVP —
    // and `build_weight_matrix` clones it, doubling that. Without this check the
    // job walked straight into the allocation and was OOM-killed by the kernel;
    // now it fails fast with a message naming the knobs.
    //
    // `ValueOnly`: only chi is built here (no gradients). The clone inside
    // `build_weight_matrix` is a second plane, so this under-charges by 1x --
    // stated rather than silently absorbed, because charging it here would
    // reject grids that the pre-clone code path handles fine. The honest fix is
    // to remove the clone; see the note there.
    ferric_dft::ao_grid::check_ao_grid_budget(
        ferric_dft::ao_grid::AoGridKind::ValueOnly,
        prep.nbasis(),
        pts.len(),
    )
    .map_err(|e| FerricError::General(format!("cDFT AO grid: {e}")))?;

    let w_mats: Vec<Array2<f64>> = {
        let chi = eval_basis_on_points(mol, bs, &pts)
            .map_err(|e| FerricError::General(format!("cDFT AO grid eval: {e:?}")))?;
        let w = cons
            .iter()
            .map(|c| build_weight_matrix(mol, &grid, &chi, &c.fragment))
            .collect();
        // Scope chi so it is FREED here rather than at function end. It is dead
        // after this point (nothing below reads it), but Rust would otherwise
        // keep the full (nbf, npts) buffer resident through the entire
        // lambda-Newton loop below -- which runs a full inner UHF per iteration
        // and is exactly where the peak matters.
        w
    };

    // The λ-Newton loop, as a closure over an OPTIONAL starting orbital guess.
    //
    // Factored out (it used to be inline) so the stability-descent block below
    // can re-run the WHOLE outer loop from a rotated guess. The guess is
    // applied at EVERY inner solve, not only the first: re-seeding each λ is
    // what keeps a descent inside the basin it was aimed at, whereas seeding
    // only λ⁰ lets the subsequent inner solves drift back into the saddle.
    let lambda_newton_from = |guess: Option<(&Array2<f64>, &Array2<f64>)>,
                              lam_start: &[f64]|
     -> Result<CdftResult, FerricError> {
        let run_inner = |lam: &[f64]| -> Result<(ScfResult, Vec<f64>, Vec<f64>), FerricError> {
            let fm = |f_a: &mut Array2<f64>, f_b: &mut Array2<f64>| {
                for (ci, c) in cons.iter().enumerate() {
                    let l = lam[ci];
                    match c.spin {
                        SpinChannel::Total => {
                            // same potential to both spins
                            let lw = l * &w_mats[ci];
                            *f_a += &lw;
                            *f_b += &lw;
                        }
                        SpinChannel::SpinDiff => {
                            let lw = l * &w_mats[ci];
                            *f_a += &lw;
                            *f_b -= &lw;
                        }
                    }
                }
            };
            let scf = solve_uhf_fockmod(ctx, mol, prep, bounds, config, guess, Some(&fm))?;
            let d_a = &scf.density_alpha;
            let d_b = scf.density_beta.as_ref().unwrap_or(d_a);
            let mut pops = vec![0.0; k];
            let mut resid = vec![0.0; k];
            for (ci, c) in cons.iter().enumerate() {
                let n_c = population(&w_mats[ci], d_a, d_b, &c.spin);
                pops[ci] = n_c;
                resid[ci] = n_c - c.target;
            }
            Ok((scf, resid, pops))
        };

        let trace = trace_enabled();

        // k = 1: the safeguarded scalar root finder (see `solve_scalar`). The
        // inner solve is wrapped as an oracle returning the residual, the
        // λ-augmented value V(λ) = E + λ·c and the inner convergence flag.
        if k == 1 {
            let mut eval =
                |l: f64, what: &str| -> Result<(Sample, (ScfResult, Vec<f64>)), FerricError> {
                    let (scf, resid, pops) = run_inner(&[l])?;
                    let s = Sample {
                        c: resid[0],
                        v: scf.energy + l * resid[0],
                        converged: scf.converged,
                    };
                    if trace {
                        eprintln!(
                            "[cdft-trace]   {what:<5} lam={l:+.12}  N_C={:.12}  resid={:+.6e}  \
                         E={:.10}  V={:.10}  inner_conv={}  inner_iters={}",
                            pops[0], resid[0], scf.energy, s.v, scf.converged, scf.iterations
                        );
                    }
                    Ok((s, (scf, pops)))
                };
            let root = solve_scalar(
                &mut eval,
                lam_start[0],
                config.cdft_lambda_tol,
                config.cdft_max_outer,
                true,
                trace,
            )?;
            let (scf, pops) = root.payload;
            return Ok(CdftResult {
                scf,
                lambdas: vec![root.lambda],
                populations: pops,
                outer_iters: root.outer_iters,
                weight_matrices: w_mats.clone(),
            });
        }

        // k > 1: plain k × k Newton with a finite-difference Jacobian and a
        // ±1 clamp per component. There is no scalar ordering to bracket or
        // bisect on, so none of the k = 1 safeguards apply here.
        let mut lam = lam_start.to_vec();
        let max_outer = config.cdft_max_outer;
        let fd = FD_STEP;

        for outer in 1..=max_outer {
            let (scf, resid, pops) = run_inner(&lam)?;
            let max_resid = resid.iter().fold(0.0_f64, |m, &r| m.max(r.abs()));
            if trace {
                eprintln!(
                    "[cdft-trace] outer={outer:2}  lam={lam:?}  N_C={pops:?}  \
                         resid={resid:?}  max|r|={max_resid:.6e}  \
                         E={:.10}  inner_conv={}  inner_iters={}",
                    scf.energy, scf.converged, scf.iterations
                );
            }
            if max_resid < config.cdft_lambda_tol {
                return Ok(CdftResult {
                    scf,
                    lambdas: lam,
                    populations: pops,
                    outer_iters: outer,
                    weight_matrices: w_mats.clone(),
                });
            }

            // Finite-difference Jacobian J_{ij} = ∂c_i/∂λ_j.
            let mut jac = Array2::<f64>::zeros((k, k));
            for j in 0..k {
                let mut lam_p = lam.clone();
                lam_p[j] += fd;
                let (scf_p, resid_p, _) = run_inner(&lam_p)?;
                if trace {
                    eprintln!(
                        "[cdft-trace]   fd j={j} lam+={:.6}  resid+={resid_p:?}  \
                             E+={:.10}  dE={:.3e}  inner_conv={}",
                        lam_p[j],
                        scf_p.energy,
                        scf_p.energy - scf.energy,
                        scf_p.converged
                    );
                }
                for i in 0..k {
                    jac[(i, j)] = (resid_p[i] - resid[i]) / fd;
                }
            }

            // Solve J · Δλ = c, then λ ← λ − Δλ, each component clamped to ±1.
            let mut delta = solve_linear(&jac, &resid)?;
            if trace {
                eprintln!("[cdft-trace]   jac={jac:?}  raw_step={delta:?}");
            }
            for d in delta.iter_mut() {
                *d = d.clamp(-MAX_STEP, MAX_STEP);
            }
            for j in 0..k {
                lam[j] -= delta[j];
            }
        }

        Err(FerricError::Convergence(format!(
            "cDFT outer loop did not converge in {max_outer} iters"
        )))
    };

    // The stability descent restarts from λ = 0, exactly as before seeding
    // existed; only the first loop takes the caller's starting point.
    let lambda_newton =
        |guess: Option<(&Array2<f64>, &Array2<f64>)>| lambda_newton_from(guess, &zero_lambdas);

    let first = lambda_newton_from(seed.mos, lam_start)?;
    if !config.cdft_stability_descent {
        return Ok(first);
    }
    Ok(stability_descent(
        ctx,
        mol,
        prep,
        bounds,
        config,
        &w_mats,
        first,
        &lambda_newton,
    ))
}

/// Step sizes tried when following a downhill eigenvector, in radians.
///
/// NOT a guess: measured on `test/cdft-constrained-stability`, where steps
/// ε ≤ 0.5 rad fall straight back into the saddle's own DIIS basin and only
/// ~1 rad escapes it. The list keeps the smaller ones so a system where a
/// gentler step suffices is not over-rotated past its minimum, and takes the
/// LOWEST constraint-satisfying result over the whole sweep rather than the
/// first success.
const DESCENT_STEPS: [f64; 3] = [0.4, 0.8, 1.2];

/// Maximum descent rounds. Each round is one eigensolve plus up to
/// `DESCENT_STEPS.len()` full λ-Newton solves, so this bounds the worst-case
/// cost at a small multiple of the unfixed solve.
const MAX_DESCENT_ROUNDS: usize = 3;

/// **cDFT state selection.** Given a converged constrained solution, check
/// whether it is a SADDLE of its own λ-augmented functional and, if so, follow
/// the downhill direction and re-converge the whole λ-Newton loop from there.
///
/// # Why the plain UHF Hessian is the right operator here
///
/// The cDFT Lagrangian `W[ρ,λ] = E[ρ] + λ(N_C[ρ] − N_target)` is a saddle in
/// the combined (ρ, λ) space BY CONSTRUCTION — minimized over ρ, maximized over
/// λ — so an ordinary `E[ρ]` stability analysis at a constrained solution would
/// report spurious instabilities on perfectly good diabats. The correct
/// question is internal stability of the λ-AUGMENTED problem AT FIXED λ.
///
/// That happens to need no new derivation. The constraint enters as `λW` with
/// `W` built ONCE from geometry and the Becke grid, outside the λ loop and
/// never rebuilt from the density (see `build_weight_matrix` above). A
/// density-independent one-electron term contributes to the Fock but NOT to the
/// Fock response `δF/δD`, so the λ-augmented orbital Hessian EQUALS the
/// ordinary UHF orbital Hessian evaluated at the constrained orbitals with the
/// CONSTRAINED (λ-augmented) orbital energies — which is exactly what
/// `uhf_newton::hessian_matvec` computes when handed this solution's `C` and
/// its λ-augmented MO Fock.
///
/// This is not assumed: it is finite-differenced against the analytic matvec by
/// `augmented_hessian_equals_plain_hessian_at_fixed_lambda` in
/// `tests/cdft_constrained_stability.rs`, and the λ-dependence of the gradient
/// it rests on is separately FD-checked at λ = −2.75 (with a doubled-λW
/// mutation) by `tests/cdft_state_selection.rs`.
///
/// `solve_uhf_fockmod` applies the Fock modifier AFTER forming the energy, so
/// `ScfResult::energy` is the BARE energy at the constrained density while
/// `fock_alpha`/`fock_beta` ARE λ-augmented — the combination this needs.
///
/// # Failure policy
///
/// Every failure mode returns the INPUT solution unchanged, after printing why.
/// A descent that cannot be computed, does not re-converge, drifts off the
/// constraint, or lands HIGHER must never make the answer worse than not
/// having tried — this function can only ever improve on `first` or leave it
/// alone.
#[allow(clippy::too_many_arguments)]
fn stability_descent(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    w_mats: &[Array2<f64>],
    first: CdftResult,
    lambda_newton: &dyn Fn(Option<(&Array2<f64>, &Array2<f64>)>) -> Result<CdftResult, FerricError>,
) -> CdftResult {
    // A KS reference needs the f_xc response kernel or the analysis is of the
    // WRONG OPERATOR (the HF Hessian at a KS density), which is exactly the
    // trap `uhf::stability_uhf` documents. Rather than reproduce that kernel
    // plumbing here, the descent SKIPS on any XC reference and says so. Pure
    // UHF — which is what the cDFT lane is validated on — is unaffected.
    if config.xc.is_some() {
        eprintln!(
            "cDFT stability descent: SKIPPED on a KS reference (xc = {:?}). The \
             λ-augmented Hessian would need the f_xc response kernel, and analysing \
             the HF Hessian at a KS density instead would be a wrong-operator \
             verdict. The constrained solution is returned as converged, which does \
             NOT mean it is the lowest state at this constraint.",
            config.xc
        );
        return first;
    }

    let mut best = first;
    for round in 1..=MAX_DESCENT_ROUNDS {
        let Some((va, vb, lmin, verdict)) =
            augmented_instability(ctx, mol, prep, bounds, config, w_mats, &best)
        else {
            return best;
        };
        if verdict != crate::stability::StabilityVerdict::Unstable {
            if round == 1 {
                eprintln!(
                    "cDFT stability descent: the constrained solution is {} \
                     (λ_min = {lmin:+.4e}); no descent taken.",
                    verdict_label(verdict)
                );
            }
            return best;
        }
        eprintln!(
            "cDFT stability descent (round {round}): the constrained solution at \
             E = {:.8} is a SADDLE of its λ-augmented functional (λ_min = \
             {lmin:+.4e}); following the downhill eigenvector.",
            best.scf.energy
        );

        // Sweep the step sizes and keep the LOWEST solution that still
        // satisfies the constraint. A restart that fails to converge or that
        // drifts off the target is skipped, not fatal.
        let mut improved: Option<CdftResult> = None;
        for &step in &DESCENT_STEPS {
            let Some(cb) = best.scf.mos_beta.as_ref() else {
                return best;
            };
            let nocc_a = occupied_alpha(mol);
            let nocc_b = occupied_beta(mol);
            let g_a = rotate_mos(&best.scf.mos_alpha, &va, nocc_a, step);
            let g_b = rotate_mos(cb, &vb, nocc_b, step);
            let cand = match lambda_newton(Some((&g_a, &g_b))) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("cDFT stability descent: step {step} did not re-converge ({e:?})");
                    continue;
                }
            };
            // The candidate must satisfy the SAME constraint, else it is a
            // different problem's answer and not a better B.
            let off = constraint_offset(&cand.populations, &config.constraints);
            if off >= config.cdft_lambda_tol {
                eprintln!(
                    "cDFT stability descent: step {step} re-converged at E = {:.8} but \
                     OFF the constraint by {off:.2e} (tol {:.1e}); discarded.",
                    cand.scf.energy, config.cdft_lambda_tol
                );
                continue;
            }
            if accepts_candidate(
                cand.scf.energy,
                best.scf.energy,
                improved.as_ref().map(|b| b.scf.energy),
            ) {
                improved = Some(cand);
            }
        }

        match improved {
            Some(c) => {
                eprintln!(
                    "cDFT stability descent (round {round}): reached a LOWER constrained \
                     solution, E = {:.8} (was {:.8}, ΔE = {:.8} Ha = {:.4} eV), \
                     λ = {:?}, N_C = {:?}",
                    c.scf.energy,
                    best.scf.energy,
                    best.scf.energy - c.scf.energy,
                    (best.scf.energy - c.scf.energy) * 27.211_386_245_988,
                    c.lambdas,
                    c.populations
                );
                best = c;
            }
            None => {
                eprintln!(
                    "cDFT stability descent (round {round}): the solution is a saddle \
                     (λ_min = {lmin:+.4e}) but NO step reached a lower \
                     constraint-satisfying solution. Returning the saddle, which is \
                     therefore NOT established as the lowest state at this constraint."
                );
                return best;
            }
        }
    }
    eprintln!(
        "cDFT stability descent: still descending after {MAX_DESCENT_ROUNDS} rounds; \
         returning the lowest found (E = {:.8}). It is NOT established as the bottom.",
        best.scf.energy
    );
    best
}

/// Should a descended candidate replace the best-so-far?
///
/// Only if it is BELOW both the incumbent (`best_e`) and any better candidate
/// already found this round (`improved_e`). Strict `<` throughout, so an exactly
/// equal energy does not churn the answer.
///
/// Extracted for the same reason as [`constraint_offset`]: inline it was
/// UNREACHABLE. Replacing `cand < best` with `true` — i.e. accepting a strictly
/// HIGHER state — left the whole cDFT suite GREEN, because on HeNe⁺ every
/// descended candidate happens to be lower. That guard is the entire reason the
/// descent cannot make an answer worse than not having tried, so it must be
/// tested rather than assumed. See `descent_never_accepts_a_higher_state`.
fn accepts_candidate(cand_e: f64, best_e: f64, improved_e: Option<f64>) -> bool {
    cand_e < best_e && improved_e.is_none_or(|b| cand_e < b)
}

/// How far a candidate solution sits from its constraint targets, as the max
/// over constraints of `|N_C − target_C|`.
///
/// Extracted from the descent loop so it can be tested DIRECTLY. Inline, it was
/// unreachable: on HeNe⁺ every descended candidate satisfies the constraint, so
/// deleting the check entirely (`if false && off >= tol`) left the whole cDFT
/// suite GREEN. A guard no test has ever been seen to exercise is an
/// assumption, and this one decides whether a lower-energy state is accepted —
/// exactly the decision the fix exists to make. See
/// `constraint_offset_rejects_a_candidate_that_missed_the_target`.
///
/// A candidate with FEWER populations than constraints (which cannot happen —
/// the λ-Newton loop fills one per constraint — but is representable) would
/// have its missing constraints silently ignored by a plain `zip`, so the
/// length mismatch is treated as infinitely far off rather than as agreement.
fn constraint_offset(pops: &[f64], cons: &[ferric_dft::cdft::Constraint]) -> f64 {
    if pops.len() != cons.len() {
        return f64::INFINITY;
    }
    pops.iter()
        .zip(cons.iter())
        .fold(0.0_f64, |m, (&n, c)| m.max((n - c.target).abs()))
}

fn verdict_label(v: crate::stability::StabilityVerdict) -> &'static str {
    use crate::stability::StabilityVerdict as V;
    match v {
        V::Stable => "STABLE",
        V::Unstable => "UNSTABLE",
        V::Marginal => "MARGINAL (|λ_min| at the noise floor — neither proven)",
        V::Indeterminate => "INDETERMINATE (the eigensolve did not converge)",
    }
}

fn occupied_alpha(mol: &Molecule) -> usize {
    let nelec = mol.nelec() as usize;
    (nelec + (mol.multiplicity - 1)) / 2
}
fn occupied_beta(mol: &Molecule) -> usize {
    let nelec = mol.nelec() as usize;
    (nelec - (mol.multiplicity - 1)) / 2
}

/// λ_min and its eigenvector for the λ-augmented orbital Hessian at a converged
/// constrained solution. `None` on any analysis failure, always after saying so.
#[allow(clippy::type_complexity)]
fn augmented_instability(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    w_mats: &[Array2<f64>],
    sol: &CdftResult,
) -> Option<(
    Array2<f64>,
    Array2<f64>,
    f64,
    crate::stability::StabilityVerdict,
)> {
    let c_a = &sol.scf.mos_alpha;
    let c_b = sol.scf.mos_beta.as_ref()?;
    // `fock_alpha`/`fock_beta` are ALREADY λ-augmented (the Fock modifier is
    // applied inside the inner SCF), so no λW is re-added here. Re-adding it
    // would double-count the constraint — the epcdft-class defect the λ = 0
    // anchor cannot see. `w_mats` is taken only to assert that.
    debug_assert_eq!(w_mats.len(), config.constraints.len());
    let nocc_a = occupied_alpha(mol);
    let nocc_b = occupied_beta(mol);
    let f_a = &sol.scf.fock_alpha;
    let f_b = sol.scf.fock_beta.as_ref()?;
    let f_a_mo = c_a.t().dot(f_a).dot(c_a);
    let f_b_mo = c_b.t().dot(f_b).dot(c_b);
    let inputs = crate::uhf_newton::UhfNewtonInputs {
        prep,
        bounds,
        c_a,
        c_b,
        f_a_mo: &f_a_mo,
        f_b_mo: &f_b_mo,
        nocc_a,
        nocc_b,
        k_mix_sr: 1.0,
        fxc: None,
        thresh: config.integral_thresh,
        ooc_budget: 0,
    };
    match crate::stability::uhf_internal_stability(
        ctx,
        &inputs,
        &crate::stability::StabilityConfig::default(),
    ) {
        Ok(res) => {
            let vb = res.eigenvector_beta.clone()?;
            let verdict = res.verdict();
            Some((
                res.eigenvector_alpha.clone(),
                vb,
                res.lowest_eigenvalue,
                verdict,
            ))
        }
        Err(e) => {
            eprintln!(
                "cDFT stability descent: the λ-augmented stability analysis FAILED ({e}). \
                 The constrained solution is returned unchanged, and is NOT established \
                 as the lowest state at this constraint."
            );
            None
        }
    }
}

/// Cayley rotation of `c` by `eps · κ_ov`, exactly orthonormality-preserving.
///
/// `(I − κ/2)⁻¹(I + κ/2)` is orthogonal for antisymmetric κ to machine
/// precision, unlike a truncated `exp(κ)` — which matters because the rotated
/// orbitals are fed straight back in as an SCF guess.
fn rotate_mos(c: &Array2<f64>, k_ov: &Array2<f64>, nocc: usize, eps: f64) -> Array2<f64> {
    use ndarray_linalg::Solve;
    let n = c.nrows();
    let mut kappa = Array2::<f64>::zeros((n, n));
    for (ir, a) in (nocc..n).enumerate() {
        for i in 0..nocc {
            if ir >= k_ov.nrows() || i >= k_ov.ncols() {
                continue;
            }
            let v = eps * k_ov[(ir, i)];
            kappa[(a, i)] = v;
            kappa[(i, a)] = -v;
        }
    }
    let half = 0.5 * &kappa;
    let eye = Array2::<f64>::eye(n);
    let am = &eye - &half;
    let bm = &eye + &half;
    let mut u = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        match am.solve(&bm.column(col).to_owned()) {
            Ok(sol) => {
                for row in 0..n {
                    u[(row, col)] = sol[row];
                }
            }
            // A singular (I − κ/2) cannot happen for antisymmetric κ (its
            // eigenvalues are 1 ± i·imag), but the solve is fallible, so fall
            // back to the identity column rather than panicking inside a
            // best-effort descent.
            Err(_) => u[(col, col)] = 1.0,
        }
    }
    c.dot(&u)
}

/// Solve J x = b for small k via Gaussian elimination with partial pivoting.
/// Avoids a hard ndarray-linalg dependency for the k=1/k=2 case.
fn solve_linear(j: &Array2<f64>, b: &[f64]) -> Result<Vec<f64>, FerricError> {
    let n = b.len();
    let mut a = j.clone();
    let mut x = b.to_vec();
    for col in 0..n {
        // pivot
        let mut piv = col;
        for r in (col + 1)..n {
            if a[(r, col)].abs() > a[(piv, col)].abs() {
                piv = r;
            }
        }
        if a[(piv, col)].abs() < 1e-14 {
            return Err(FerricError::Lapack("cDFT Jacobian singular".into()));
        }
        if piv != col {
            for c in 0..n {
                let t = a[(col, c)];
                a[(col, c)] = a[(piv, c)];
                a[(piv, c)] = t;
            }
            x.swap(col, piv);
        }
        for r in (col + 1)..n {
            let f = a[(r, col)] / a[(col, col)];
            for c in col..n {
                a[(r, c)] -= f * a[(col, c)];
            }
            x[r] -= f * x[col];
        }
    }
    // back-substitute
    for col in (0..n).rev() {
        let mut s = x[col];
        for c in (col + 1)..n {
            s -= a[(col, c)] * x[c];
        }
        x[col] = s / a[(col, col)];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_dft::cdft::{Constraint, SpinChannel};

    fn con(target: f64) -> Constraint {
        Constraint {
            fragment: vec![0],
            target,
            spin: SpinChannel::Total,
        }
    }

    /// The descent's constraint-satisfaction guard, tested directly because it
    /// is UNREACHABLE through the driver on the systems in the suite.
    ///
    /// MEASURED 2026-09-16: deleting the guard inline (`if false && off >=
    /// tol`) left every cDFT test GREEN, because on HeNe⁺ every descended
    /// candidate happens to land on the target. That makes the inline check an
    /// assumption; this makes it a test. The guard matters because without it
    /// the descent would accept a LOWER-energy state that solves a DIFFERENT
    /// constraint — which is not a better diabat, just a different one.
    #[test]
    fn constraint_offset_rejects_a_candidate_that_missed_the_target() {
        // On target: offset is the residual, below any sane tolerance.
        assert!(constraint_offset(&[2.000_000_02], &[con(2.0)]) < 1e-5);
        // Off target by 0.05 e: far above the 1e-5 default tolerance, so the
        // descent discards it however low its energy is.
        let off = constraint_offset(&[1.95], &[con(2.0)]);
        assert!(
            off > 1e-5,
            "a candidate 0.05 e off the target must be rejected; offset = {off:.3e}"
        );
        assert!((off - 0.05).abs() < 1e-12, "offset = {off}");
        // MAX over constraints, not sum or first: one satisfied constraint must
        // not mask a violated one.
        let both = constraint_offset(&[2.0, 1.0], &[con(2.0), con(1.5)]);
        assert!(
            (both - 0.5).abs() < 1e-12,
            "the offset must be the MAX over constraints, so a satisfied one \
             cannot hide a violated one; got {both}"
        );
        // A length mismatch is infinitely far off, not silently truncated.
        assert!(constraint_offset(&[2.0], &[con(2.0), con(1.0)]).is_infinite());
        assert!(constraint_offset(&[], &[con(2.0)]).is_infinite());
    }

    /// The descent's "never make it worse" guarantee, tested directly because
    /// it is UNREACHABLE through the driver on the systems in the suite.
    ///
    /// MEASURED 2026-09-16: replacing the `cand < best` test with `true` — so
    /// the descent would accept a strictly HIGHER state — left every cDFT test
    /// GREEN, because on HeNe⁺ every descended candidate is lower anyway. This
    /// is the guard that makes the fix safe to default ON: without it, a
    /// descent that overshoots into a worse basin would silently replace a good
    /// answer with a bad one.
    #[test]
    fn descent_never_accepts_a_higher_state() {
        // Lower than the incumbent, nothing better found yet: ACCEPT.
        assert!(accepts_candidate(-130.42, -130.40, None));
        // HIGHER than the incumbent: REJECT, however tempting.
        assert!(
            !accepts_candidate(-130.38, -130.40, None),
            "a descended state ABOVE the incumbent must never be accepted — that \
             would make the fix capable of making an answer worse"
        );
        // Equal to the incumbent: REJECT (strict <), so an equal-energy result
        // does not churn which solution is returned.
        assert!(!accepts_candidate(-130.40, -130.40, None));
        // Lower than the incumbent but ABOVE a better candidate already found
        // this round: REJECT, so the sweep keeps the LOWEST step, not the last.
        assert!(!accepts_candidate(-130.41, -130.40, Some(-130.43)));
        // Lower than both: ACCEPT.
        assert!(accepts_candidate(-130.45, -130.40, Some(-130.43)));
    }

    /// **The safeguard must be INERT while Newton behaves.**
    ///
    /// This is the unit-level half of the bit-identity anchor. The integration
    /// anchor (`hcore_started_path_is_bit_identical`) proves the baselined solve
    /// does not move; this proves WHY, at the one decision point that could move
    /// it — so a future edit that makes the safeguard fire more eagerly fails
    /// here, in milliseconds, instead of silently re-baselining a suite that
    /// takes minutes to run.
    #[test]
    fn bracket_safeguard_is_inert_while_newton_stays_inside() {
        let mut b = Bracket::default();
        // No bracket at all: nothing to safeguard against.
        assert_eq!(b.safeguard(-3.0), None);
        b.observe(-2.4, -0.014, true);
        // One-sided only — still no override, because without a sign change no
        // root has been straddled and a "midpoint" would be meaningless.
        assert_eq!(
            b.safeguard(-99.0),
            None,
            "a ONE-SIDED bound must not trigger bisection"
        );
        b.observe(-3.08, 0.064, true);
        // Two-sided, Newton proposes a point INSIDE: keep Newton's step
        // verbatim. This is the case every converging run takes, and it is what
        // makes the safeguard bit-identical on the baselined paths.
        assert_eq!(b.safeguard(-2.7), None);
        assert_eq!(b.safeguard(-3.0), None);
    }

    /// **The safeguard must FIRE on the measured limit cycle.**
    ///
    /// The literals are the real iterates traced on HeNe⁺/def2-SVP at target
    /// 1.995 (see `tests/cdft_outer_loop.rs`): the clamp pinned λ to exactly
    /// −3.3266069180979265 and −2.3266069180979265, one clamp-width apart,
    /// forever. Both lie outside the bracket the healthy early iterations had
    /// already established, so both are overridden.
    #[test]
    fn bracket_safeguard_fires_on_the_measured_limit_cycle() {
        let mut b = Bracket::default();
        b.observe(-2.469_514_680_205_723, -0.017_918_277_589_467_29, true);
        b.observe(-3.039_317_923_694_346_4, 0.060_103_146_697_081_83, true);
        let mid = 0.5 * (-3.039_317_923_694_346_4 + -2.469_514_680_205_723);
        assert_eq!(
            b.safeguard(-3.326_606_918_097_926_5),
            Some(mid),
            "the clamp's lower cycle point is OUTSIDE the bracket and must be \
             replaced by the midpoint; leaving it is exactly the 20-iteration \
             period-2 cycle this fix exists to break"
        );
        assert_eq!(b.safeguard(-2.326_606_918_097_926_5), Some(mid));
        // And the midpoint itself is inside, so the very next step is free to be
        // a Newton step again — the safeguard must not latch on, or the method
        // degrades to pure bisection and loses its quadratic tail.
        assert_eq!(b.safeguard(mid), None, "the safeguard must not LATCH");
    }

    /// **A Newton proposal landing exactly ON a bound must be overridden.**
    ///
    /// ADDED AFTER A SURVIVING MUTATION, not alongside the others. Relaxing
    /// `safeguard`'s strict `>`/`<` to `>=`/`<=` left ALL SIX of the other tests
    /// in this module green, because every λ they probe is strictly outside or
    /// strictly inside the bracket and none sits on the boundary. That is the
    /// repo's "a test you have never seen fail is an assumption" in its other
    /// form: an untested branch condition.
    ///
    /// The behaviour matters rather than being pedantry about `<` vs `<=`. A
    /// bound is a λ whose residual is already known and NONZERO — it was stored
    /// precisely because `c` there had a definite sign. Accepting a Newton step
    /// back onto it means re-running an inner SCF that can only reproduce a
    /// residual the loop has already seen and rejected, which is a repeated
    /// iterate: the exact shape of the period-2 cycle being fixed. Bisecting
    /// instead always produces a λ the loop has not visited.
    #[test]
    fn bracket_safeguard_overrides_a_proposal_exactly_on_a_bound() {
        let mut b = Bracket::default();
        b.observe(-2.0, -0.1, true);
        b.observe(-3.0, 0.1, true);
        assert_eq!(
            b.safeguard(-2.0),
            Some(-2.5),
            "a Newton step landing exactly on the negative-residual bound must be \
             bisected away: that lambda's residual is already known and nonzero, so \
             re-evaluating it is a repeated iterate, not progress"
        );
        assert_eq!(
            b.safeguard(-3.0),
            Some(-2.5),
            "likewise for the positive-residual bound"
        );
        // Strictly inside is still kept, so this does not over-fire.
        assert_eq!(b.safeguard(-2.5), None);
        assert_eq!(b.safeguard(-2.001), None);
        assert_eq!(b.safeguard(-2.999), None);
    }

    /// **An unconverged inner solve must never set a bracket bound.**
    ///
    /// ADDED AFTER A SURVIVING MUTATION, like the boundary test above. Deleting
    /// the `converged` filter entirely left the WHOLE suite green — unit tests
    /// and integration alike, including the 1.990 sweep whose trace is the
    /// reason the filter was written (four inner solves hit the 400-iteration
    /// cap in a single outer loop there). An untested guard is an assumption,
    /// which is precisely why `constraint_offset` and `accepts_candidate` in
    /// this same file were extracted and pinned.
    ///
    /// The guard is currently UNREACHABLE on the paths that converge, and that
    /// is recorded rather than hidden: it earns its place on the runs that fail,
    /// where an unconverged population is not a value of `c(λ)` at all and
    /// bracketing against it would aim bisection at a point that is not on the
    /// curve.
    #[test]
    fn bracket_ignores_an_unconverged_inner_solve() {
        let mut b = Bracket::default();
        // Two unconverged observations that WOULD form a perfectly good-looking
        // bracket if the flag were ignored.
        b.observe(-2.0, -0.1, false);
        b.observe(-3.0, 0.1, false);
        assert_eq!(
            b.safeguard(-9.0),
            None,
            "an unconverged inner solve must not establish a bracket: its \
             population is wherever the density happened to be when the iteration \
             cap hit, not a value of c(lambda)"
        );
        // A converged pair does establish one...
        b.observe(-2.0, -0.1, true);
        b.observe(-3.0, 0.1, true);
        assert_eq!(b.safeguard(-9.0), Some(-2.5));
        // ...and a later UNCONVERGED point, however tight it looks, must not
        // move it.
        b.observe(-2.49, -0.001, false);
        assert_eq!(
            b.safeguard(-9.0),
            Some(-2.5),
            "an unconverged point must not TIGHTEN an established bracket either"
        );
    }

    /// **"Tighter" must be measured against the OPPOSITE bound, not by
    /// comparing λ against a hardcoded direction.**
    ///
    /// This pins the regression that the exactness anchor caught. The λ values
    /// here are the real hcore-started iterates: the `c < 0` side walks
    /// 0.0 → −1.0 → −2.0 → −2.5 toward the root while the `c > 0` side sits at
    /// −3.0. A rule that keeps the most-POSITIVE λ among `c < 0` would freeze
    /// the bound at 0.0, never tighten, and bisect to −1.5 forever — which is
    /// precisely what it did, turning a 16-iteration success into a period-2
    /// cycle.
    #[test]
    fn bracket_tightens_when_lambda_runs_negative_toward_the_root() {
        let mut b = Bracket::default();
        b.observe(0.0, -0.0465, true);
        b.observe(-3.0, 0.0714, true);
        assert_eq!(b.safeguard(-9.0), Some(-1.5));
        // Each later negative-residual point is NEARER −3.0 and must replace
        // the bound.
        b.observe(-1.0, -0.0411, true);
        assert_eq!(b.safeguard(-9.0), Some(-2.0));
        b.observe(-2.0, -0.0315, true);
        assert_eq!(b.safeguard(-9.0), Some(-2.5));
        b.observe(-2.5, -0.0219, true);
        assert_eq!(
            b.safeguard(-9.0),
            Some(-2.75),
            "the bracket must TIGHTEN as the iterate walks toward the root; a \
             bound frozen at its first value makes bisection return the same \
             lambda forever, which is a cycle, not a safeguard"
        );
        // A point FURTHER from the opposite bound must be ignored.
        b.observe(-0.5, -0.044, true);
        assert_eq!(b.safeguard(-9.0), Some(-2.75), "the bracket must not widen");
    }

    /// **Non-finite and exactly-zero observations must not corrupt the bracket.**
    ///
    /// `c == 0` is a ROOT, not a bound: recording it on either side would drag
    /// the midpoint away from a λ that already satisfies the constraint. In the
    /// loop an exact zero returns before `observe` is reached, so this is pinned
    /// here rather than left to chance.
    #[test]
    fn bracket_rejects_nonfinite_and_zero_observations() {
        let mut b = Bracket::default();
        b.observe(-2.0, -0.1, true);
        b.observe(-3.0, 0.1, true);
        let before = b.safeguard(-9.0);
        assert_eq!(before, Some(-2.5));
        b.observe(f64::NAN, -1.0, true);
        b.observe(-2.5, f64::NAN, true);
        b.observe(f64::INFINITY, 1.0, true);
        b.observe(-2.5, 0.0, true);
        assert_eq!(
            b.safeguard(-9.0),
            before,
            "a non-finite or exactly-zero observation must be dropped, not stored"
        );
        // A NaN Newton proposal is outside any bracket and must be replaced
        // rather than written into lambda.
        assert!(
            b.safeguard(f64::NAN).is_some(),
            "a NaN Newton step must never be propagated into lambda"
        );
    }

    // ---------------------------------------------------------------------
    // The k = 1 outer loop on synthetic residuals.
    //
    // Each oracle returns (Sample, ()) and records the λ of every MAIN
    // evaluation, so trajectories can be compared bit for bit.
    // ---------------------------------------------------------------------

    fn sample(c: f64, v: f64, converged: bool) -> Sample {
        Sample { c, v, converged }
    }

    /// Run `solve_scalar` on `oracle` from `lam0` at tolerance 1e-5, returning
    /// the result and the λ of every main evaluation.
    fn run_oracle(
        oracle: &dyn Fn(f64) -> Sample,
        lam0: f64,
        max_outer: usize,
        guards: bool,
    ) -> (Result<ScalarRoot<()>, FerricError>, Vec<f64>) {
        let mut mains = Vec::new();
        let mut eval = |l: f64, what: &str| -> Result<(Sample, ()), FerricError> {
            if what == "main" {
                mains.push(l);
            }
            Ok((oracle(l), ()))
        };
        let r = solve_scalar(&mut eval, lam0, 1e-5, max_outer, guards, false);
        (r, mains)
    }

    /// A smooth, strictly decreasing c(λ) with its exact V(λ) = ∫c dλ; root at
    /// λ = −2.75. Newton from 0 hits the ±1 clamp first, so the run exercises
    /// clamped and unclamped steps.
    fn smooth(l: f64) -> Sample {
        let x = l + 2.75;
        sample(
            -0.05 * x - 0.02 * x.powi(3),
            -0.025 * x * x - 0.005 * x.powi(4),
            true,
        )
    }

    /// Plateau, root, cliff: c = 0.5 − 0.5·(λ/2.457)^8 rises to its root at
    /// λ = 2.457 through a flat start, and past λ = 2.47 the inner SCF "does not
    /// converge" and returns a λ-dependent garbage population. The shape of the
    /// He₂⁺/PBE failure (flat N(λ), root at the edge of an over-localization
    /// cliff whose inner solves do not converge).
    fn plateau_cliff(l: f64) -> Sample {
        const ROOT: f64 = 2.457;
        if l > 2.47 {
            return sample(-0.95 + 0.3 * (997.0 * l).sin(), 0.0, false);
        }
        if l <= 0.0 {
            return sample(0.5, 0.5 * l, true);
        }
        sample(
            0.5 - 0.5 * (l / ROOT).powi(8),
            0.5 * l - 0.5 * ROOT / 9.0 * (l / ROOT).powi(9),
            true,
        )
    }

    /// A bistable window: for |λ| < 0.5 the inner SCF lands in a delocalized
    /// basin when λ falls in an even 1e-3 cell and a localized one otherwise, so
    /// EVERY finite-difference pair straddles the two basins (the He₂⁺ λ = 0
    /// start, where a 1e-3 step flips the SCF and the "Jacobian" is O(100)).
    /// The basins have different V, by 0.02 Ha at λ = 0. The localized branch
    /// is linear with its root at λ = 2.
    fn bistable(l: f64) -> Sample {
        if l.abs() < 0.5 && (l / 1e-3).floor().rem_euclid(2.0) == 0.0 {
            return sample(0.5 - 0.01 * l, 0.5 * l - 0.005 * l * l + 0.02, true);
        }
        sample(0.3 - 0.15 * l, 0.3 * l - 0.075 * l * l, true)
    }

    /// **Exactness anchor: with nothing wrong, the guards change nothing.**
    ///
    /// On a smooth residual every inner solve converges and every probe is on
    /// the main point's branch, so neither guard may fire and the trajectory
    /// must be BIT-identical to the unguarded loop's. This is what lets the
    /// guards land without moving any run that already converged without them.
    #[test]
    fn guards_are_inert_on_a_smooth_residual() {
        let (on, path_on) = run_oracle(&smooth, 0.0, 40, true);
        let (off, path_off) = run_oracle(&smooth, 0.0, 40, false);
        let (on, off) = (on.unwrap(), off.unwrap());
        assert_eq!(
            path_on.iter().map(|l| l.to_bits()).collect::<Vec<_>>(),
            path_off.iter().map(|l| l.to_bits()).collect::<Vec<_>>(),
            "the guarded trajectory left the unguarded one on a residual where no \
             guard has a reason to fire: {path_on:?} vs {path_off:?}"
        );
        assert_eq!(on.lambda.to_bits(), off.lambda.to_bits());
        assert_eq!(on.outer_iters, off.outer_iters);
        // Non-vacuous: the run took clamped AND unclamped Newton steps.
        assert!(on.outer_iters >= 4, "outer = {}", on.outer_iters);
        assert_eq!(path_on[1], -1.0, "the first step must hit the ±1 clamp");
        assert!((on.lambda + 2.75).abs() < 1e-3, "root {}", on.lambda);
    }

    /// **Plateau plus an unconverged cliff: the unguarded loop wanders on the
    /// cliff; guard 1 backs off it and finds the root at its edge.**
    ///
    /// Unguarded, the first clamped step lands at λ = 3 on the cliff, every
    /// later Newton step is taken from an unconverged population, and the run
    /// exhausts its cap between 3.0 and 3.2 (the He₂⁺ def2-SVP 3.50 Å limit
    /// cycle has this shape). Asserting that failure keeps the oracle honest:
    /// if the unguarded loop ever passes here, the test no longer exercises
    /// guard 1.
    #[test]
    fn backtracking_finds_a_root_at_the_edge_of_an_unconverged_cliff() {
        let (off, _) = run_oracle(&plateau_cliff, 1.0, 40, false);
        assert!(
            off.is_err(),
            "the unguarded loop converged on the cliff oracle, so it no longer \
             tests guard 1"
        );
        let (on, path) = run_oracle(&plateau_cliff, 1.0, 40, true);
        let on = on.unwrap_or_else(|e| panic!("guarded loop failed: {e:?}; path {path:?}"));
        assert!(
            (on.lambda - 2.457).abs() < 1e-4,
            "root {} (path {path:?})",
            on.lambda
        );
        // Every accepted point is a converged one: the root is on the plateau.
        assert!(plateau_cliff(on.lambda).converged);
        // It got there through at least one backtrack (λ = 3 is on the cliff).
        assert!(path.iter().any(|&l| l > 2.47), "path {path:?}");
        assert!(
            on.outer_iters <= 12,
            "outer = {} (path {path:?})",
            on.outer_iters
        );
    }

    /// **A probe in another basin: the unguarded loop crawls; guard 2 rejects
    /// the pair and steps by the sign of c.**
    ///
    /// Unguarded, J ≈ (0.3 − 0.5)/1e-3 = −200 and Newton crawls in ~2.5e-3
    /// steps through the bistable window for the whole cap. The Hellmann–Feynman
    /// mismatch (0.02 Ha, the V gap between the basins) rejects the pair; with
    /// no trusted J yet the step is +radius (c > 0), which leaves the window,
    /// and Newton on the localized branch finishes.
    #[test]
    fn a_probe_in_another_basin_is_not_used_as_a_derivative() {
        let (off, _) = run_oracle(&bistable, 0.0, 40, false);
        assert!(
            matches!(off, Err(FerricError::Convergence(_))),
            "the unguarded loop escaped the bistable window, so this oracle no \
             longer tests guard 2"
        );
        let (on, path) = run_oracle(&bistable, 0.0, 40, true);
        let on = on.unwrap_or_else(|e| panic!("guarded loop failed: {e:?}; path {path:?}"));
        assert!((on.lambda - 2.0).abs() < 1e-4, "root {}", on.lambda);
        // λ = 0 (rejected pair, sign step) → 1 (Newton on the linear branch) → 2.
        assert_eq!(path.len(), 3, "path {path:?}");
        assert_eq!(
            path[1], 1.0,
            "the rejected pair must give a +radius sign step"
        );
    }

    /// Guard 1 stops instead of halving forever: an inner SCF that fails
    /// arbitrarily close to the last converged λ is reported as a stall.
    #[test]
    fn backtrack_reports_a_stall_below_the_floor() {
        let mut st = ScalarStepper::new(true);
        st.record(2.0, &sample(0.1, 0.0, true));
        let mut lam = 3.0;
        let mut halvings = 0;
        let err = loop {
            match st.backtrack(lam, &sample(-0.9, 0.0, false)) {
                Ok(Some(back)) => {
                    lam = back;
                    halvings += 1;
                    assert!(halvings < 64, "backtrack never reported the stall");
                }
                Ok(None) => panic!("an unconverged point with an anchor must backtrack"),
                Err(e) => break e,
            }
        };
        assert!(
            matches!(err, FerricError::Convergence(ref m) if m.contains("stalled")),
            "{err:?}"
        );
        // |back − good| halves from 0.5; it falls below 1e-9 after 29 halvings.
        assert_eq!(halvings, 29);
    }

    /// Guard 1 in isolation: the backtrack point and the radius it leaves.
    #[test]
    fn backtrack_halves_toward_the_last_converged_lambda_and_shrinks_the_radius() {
        let mut st = ScalarStepper::new(true);
        // Nothing converged yet: nothing to back off to.
        assert_eq!(st.backtrack(3.0, &sample(-0.9, 0.0, false)).unwrap(), None);
        st.record(2.0, &sample(0.1, 0.0, true));
        // A converged point never backtracks.
        assert_eq!(st.backtrack(3.0, &sample(-0.9, 0.0, true)).unwrap(), None);
        // An unconverged one goes halfway back, and the radius becomes half
        // the remaining distance.
        assert_eq!(
            st.backtrack(3.0, &sample(-0.9, 0.0, false)).unwrap(),
            Some(2.5)
        );
        assert_eq!(st.radius, 0.25);
        // An unconverged point never becomes the anchor.
        st.record(2.5, &sample(-0.9, 0.0, false));
        assert_eq!(st.last_good, Some(2.0));
        // From a converged point at 2.5 Newton wants +100; the shrunken radius
        // keeps it short of the λ = 3.0 that failed.
        let s = sample(0.1, 0.0, true);
        let p = sample(0.1 - 1e-6, 0.1 * FD_STEP - 0.5e-6 * FD_STEP, true);
        st.record(2.5, &s);
        let (next, kind) = st.step(2.5, &s, &p).unwrap();
        assert_eq!(kind, StepKind::Newton);
        assert_eq!(
            next, 2.75,
            "the step must be clamped to the shrunken radius"
        );
        // A trusted step doubles the radius back toward MAX_STEP.
        assert_eq!(st.radius, 0.5);
        // Off, the guard never fires.
        let mut off = ScalarStepper::new(false);
        off.record(2.0, &sample(0.1, 0.0, true));
        assert_eq!(off.backtrack(3.0, &sample(-0.9, 0.0, false)).unwrap(), None);
    }

    /// Guard 2 in isolation: which pairs are trusted.
    #[test]
    fn a_pair_is_trusted_only_on_one_converged_branch() {
        let st = ScalarStepper::new(true);
        let s = sample(0.2, 1.0, true);
        // Same branch: c drops by 1e-4 over h, V rises by h·(mean c).
        let good = sample(0.2 - 1e-4, 1.0 + FD_STEP * (0.2 - 0.5e-4), true);
        assert!(st.trusts(&s, &good));
        // Unconverged probe.
        assert!(!st.trusts(
            &s,
            &Sample {
                converged: false,
                ..good
            }
        ));
        // Another basin: V off by 1e-3 Ha.
        assert!(!st.trusts(
            &s,
            &Sample {
                v: good.v + 1e-3,
                ..good
            }
        ));
        // Just inside and just outside the mismatch bar.
        assert!(st.trusts(
            &s,
            &Sample {
                v: good.v + 0.5 * HF_MISMATCH_TOL,
                ..good
            }
        ));
        assert!(!st.trusts(
            &s,
            &Sample {
                v: good.v + 2.0 * HF_MISMATCH_TOL,
                ..good
            }
        ));
        // Singular and non-finite Jacobians.
        assert!(!st.trusts(
            &s,
            &Sample {
                c: s.c,
                v: s.v + FD_STEP * s.c,
                ..good
            }
        ));
        assert!(!st.trusts(
            &s,
            &Sample {
                c: f64::NAN,
                ..good
            }
        ));
        // An unconverged MAIN point is trusted (no better model exists yet).
        assert!(st.trusts(
            &Sample {
                converged: false,
                ..s
            },
            &Sample { v: 99.0, ..good }
        ));
        // Guards off: everything is trusted.
        assert!(ScalarStepper::new(false).trusts(&s, &Sample { v: 99.0, ..good }));
    }

    /// Guard 2's two fallbacks: the last trusted Jacobian, else a sign step.
    #[test]
    fn an_untrusted_pair_falls_back_to_the_last_jacobian_then_the_sign_of_c() {
        let bad = |c: f64| sample(c, 0.0, false); // unconverged probe
                                                  // No history: step by +radius when c > 0 ...
        let mut st = ScalarStepper::new(true);
        let (next, kind) = st.step(0.0, &sample(0.4, 0.0, true), &bad(0.0)).unwrap();
        assert_eq!((next, kind), (1.0, StepKind::SignStep));
        // ... and by −radius when c < 0.
        let mut st = ScalarStepper::new(true);
        let (next, kind) = st.step(0.0, &sample(-0.4, 0.0, true), &bad(0.0)).unwrap();
        assert_eq!((next, kind), (-1.0, StepKind::SignStep));
        // An untrusted step does not grow the radius.
        assert_eq!(st.radius, MAX_STEP);
        // With history: a trusted J = −0.1 is remembered and reused.
        let mut st = ScalarStepper::new(true);
        let s = sample(0.05, 0.0, true);
        let p = sample(0.05 - 1e-4, FD_STEP * (0.05 - 0.5e-4), true);
        let (next, _) = st.step(0.0, &s, &p).unwrap();
        assert!((next - 0.5).abs() < 1e-12, "next = {next}");
        let s2 = sample(0.02, 0.0, true);
        let (next2, kind2) = st.step(0.5, &s2, &bad(0.0)).unwrap();
        assert_eq!(kind2, StepKind::StaleJacobian);
        assert!((next2 - 0.7).abs() < 1e-9, "next = {next2}");
    }

    /// The stepper applies the bracket safeguard to EVERY kind of step: a
    /// Newton or fallback proposal outside a two-sided bracket is replaced by
    /// its midpoint, and one inside is kept.
    #[test]
    fn the_stepper_keeps_every_step_inside_the_bracket() {
        let mut st = ScalarStepper::new(true);
        st.record(-2.0, &sample(-0.1, 0.0, true));
        st.record(-3.0, &sample(0.1, 0.0, true));
        // From λ = −2 with c = −0.1, J = −0.01 proposes λ = −12, clamped to −3:
        // ON the c > 0 bound, which the safeguard treats as outside.
        let s = sample(-0.1, 0.0, true);
        let p = sample(-0.1 - 1e-5, FD_STEP * (-0.1 - 0.5e-5), true);
        let (next, kind) = st.step(-2.0, &s, &p).unwrap();
        assert_eq!(kind, StepKind::Newton);
        assert_eq!(next, -2.5, "an out-of-bracket Newton step must be bisected");
        // A sign step (no trusted J yet) that leaves the bracket is bisected too.
        let mut st = ScalarStepper::new(true);
        st.record(-2.0, &sample(-0.1, 0.0, true));
        st.record(-2.2, &sample(0.1, 0.0, true));
        let unconverged_probe = sample(0.0, 0.0, false);
        let (next, kind) = st.step(-2.0, &s, &unconverged_probe).unwrap();
        assert_eq!(kind, StepKind::SignStep);
        assert!((next + 2.1).abs() < 1e-15, "next = {next}");
    }

    /// The Hellmann–Feynman check is the trapezoid rule on dV/dλ = c: on one
    /// branch it returns the trapezoid error (asserted from both sides, so a
    /// mismatch that ignored V or c would fail), and across two branches the
    /// V gap.
    #[test]
    fn hf_mismatch_is_the_trapezoid_error_on_one_branch_and_the_gap_across_two() {
        let a = smooth(-1.0);
        let b = smooth(-1.0 + FD_STEP);
        let m = hf_mismatch(&a, &b, FD_STEP);
        // Trapezoid error h³|c''|/12 with |c''| = 0.12·|x| = 0.21 at x = 1.75: 1.75e-11.
        assert!(m < 1e-10 && m > 1e-12, "same-branch mismatch {m:.3e}");
        let shifted = Sample { v: b.v + 0.02, ..b };
        let gap = hf_mismatch(&a, &shifted, FD_STEP);
        assert!((gap - 0.02).abs() < 1e-10, "gap {gap}");
    }
}
