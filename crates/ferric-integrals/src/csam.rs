//! CSAM integral screening: a NON-RIGOROUS (underestimating) tight *estimate*
//! of the two-electron integral magnitude, ported from Psi4's
//! `shell_significant_csam()` (`psi4/src/psi4/libmints/twobody.cc`,
//! lines 206-224 and 363-398 at the commit this was read from). Psi4's
//! `twobody.cc` calls it CSAM without expanding the acronym in the code this
//! module was ported from.
//!
//! # THIS IS NOT AN UPPER BOUND — read before using it
//!
//! CSAM **can and does underestimate** `|(MN|RS)|`, **for every kernel,
//! Coulomb included**. It therefore deliberately does NOT implement
//! `ferric_scf::screening::Bound`, whose contract (see `crate::schwarz`'s long
//! comment on why an underestimating bound is fatal) is that a bound never
//! falls below the true integral. It is exposed only as an explicitly-named
//! *estimate* whose error is controlled by, and shrinks with, the screening
//! threshold.
//!
//! ## Primary-source basis for that claim
//!
//! Thompson & Ochsenfeld, "Distance-including rigorous upper bounds and tight
//! estimates for two-electron integrals over long- and short-range operators",
//! J. Chem. Phys. 147, 144101 (2017), doi:10.1063/1.4994190. Read locally from
//! the verbatim reprint as Publication I of Thompson's LMU dissertation,
//! `wiki/papers/thompson-2020-lmu-dissertation-integral-bounds.pdf`, PDF pages
//! 63-72. The published abstract separates the two families explicitly:
//!
//!   - a **rigorous** bound (CSB, Eq. (8)), demonstrated for operators with
//!     *exponential distance decay* (`e^-r12` and `erfc(0.11 r12)/r12`);
//!   - a **non-rigorous estimate** (CSA1/CSA2/CSAM, Eqs. (9)/(11)/(12)), which
//!     "gives results very close to an exact screening for these operators
//!     *and for the long-range 1/r12 operator*, with errors that are
//!     completely controllable through the integral screening threshold."
//!
//! Psi4's `shell_significant_csam` carries no citation, only bare
//! "Eq. 1/9/11/12" comments, and applies the estimate with no clamp against
//! Schwarz and no fallback — i.e. Psi4 ships the non-rigorous estimate as its
//! default `SCREENING=CSAM`.
//!
//! ## What "non-rigorous" costs in ENERGY — the number that matters
//!
//! Non-rigor means the estimate can fall below the true integral. It does NOT
//! mean the resulting energy error is large, and the two must not be conflated
//! (this module's history did conflate them; see "Correction" below). The
//! paper measures the energy error directly, in HF SCF calculations, across
//! its whole test suite:
//!
//!   - **Table V (PDF p. 70)**, aug-cc-pVDZ at the tight thresholds
//!     `theta_j = theta_k = 1e-12`, reference = QQ at `1e-14`. CSAM converged-
//!     energy errors over six systems (Amylose8, Angiotensin, DNA2, (H2O)68,
//!     (LiF)32, Triphenylmethyl): **-0.20, +1.80, -0.80, -0.80, +1.00, -0.10
//!     NANOhartree** — versus QQ's own -1.50..+0.70 nH and QQR's -6.80..+1.40
//!     nH on the same rows. At this threshold CSAM is not worse than the
//!     rigorous QQ bound; on four of six systems it is better.
//!   - **Table III (PDF p. 69)**, cc-pVDZ at the looser `1e-10`: CSAM errors
//!     0.05-2.32 MICROhartree on most systems, worst 9.35 uH on CNT(6,3)8 —
//!     against QQ 0.02-4.18 uH on the same rows. Roughly 2x the rigorous
//!     bound's error, at ~1.3x the speed.
//!
//! So the honest characterization is: **non-rigorous, with a threshold-
//! controlled error that is nanohartree-scale at tight thresholds and stays
//! within ~2x the rigorous QQ bound's own error at loose ones.**
//!
//! What non-rigor DOES cost is the absence of a guarantee. The paper is
//! explicit that the error is not threshold-free: "the errors are roughly
//! proportional to the system size for each estimate" (Sec. III B 2, PDF
//! p. 70) and Fig. 2 shows the screening error growing linearly with alkane
//! chain length at fixed threshold. A rigorous bound has no such term. That is
//! the real reason to prefer CSB where CSB is available — not a large measured
//! error, because none was ever measured.
//!
//! ## Correction (2026-09-14): the "~1e5x accuracy loss" framing was WRONG
//!
//! An earlier revision of this header, and commit `ab525443`'s message,
//! justified refusing attenuated operators with a table of "underestimate
//! ratios" up to `1.0e8` and a `218493x` figure on benzene/cc-pVDZ, and
//! described the consequence as a "silent ~1e5x accuracy loss". **That
//! framing is retracted.** Three independent defects, each verified:
//!
//! 1. **A ratio is not an energy error, and none was ever computed.**
//!    `scripts/csam_bound_validity_proto.py` maximizes `true/estimate` over
//!    ALL quartets with no magnitude filter (the only guard is
//!    `csam > 1e-300`). A ratio grows without bound; an energy error does not.
//!    The paper's own Table I/II captions restrict their statistics to
//!    "shell-quartets with exact norms above `1e-12`", and report the energy
//!    consequence separately in Tables III and V. Commit `ab525443` computed
//!    no erfc energy error at all, so "~1e5x accuracy loss" had no measurement
//!    behind it in either direction.
//!
//!    Scope note, so this correction is not itself overstated: adding the
//!    paper's `1e-12` filter to the harness (done, `--min-magnitude`) leaves
//!    the chain-study numbers UNCHANGED; only at a `1e-8` cut does the
//!    6-center row fall back. The missing filter is a real methodological gap
//!    in the harness, but point 2 below — not point 1 — is what actually
//!    generates those figures.
//!
//! 2. **The large rows are float64 roundoff, not screening failure.** The
//!    harness computes erfc as `Coulomb - erf` in double precision
//!    (`eri_erfc`), which is precisely the catastrophic cancellation the paper
//!    warns about on the same page: "Due to the way in which the integrals over
//!    the `erfc(0.11 r12)/r12` operator are calculated as the difference
//!    between the integrals over the `1/r12` and `erf(0.11 r12)/r12`
//!    operators, numerical instability occurs for large distances between
//!    charge distributions. Such integrals are calculated as the small
//!    difference between large values." (PDF p. 66.) Re-evaluating that same
//!    `eri_erfc` at 200-bit precision with mpmath, on the harness's own
//!    two-s-Gaussian geometry at `omega = 1.0`:
//!
//!    ```text
//!      separation(bohr)    float64        mpmath(200b)    rel. error
//!             0.00       1.477463e+01     1.477463e+01     6.4e-16
//!             2.64       3.105075e+00     3.105075e+00     2.1e-15
//!             5.28       3.057790e-02     3.057790e-02     6.9e-14
//!             7.92       1.646151e-05     1.646151e-05     1.2e-10
//!            10.56       5.642242e-10     5.642224e-10     3.2e-06
//!            13.20       7.105427e-15     1.299934e-15     4.5e+00   <-- 447%
//!    ```
//!
//!    The 6-center row of the old table (13.2 bohr extent, ratio `1.0e8`) sits
//!    exactly where the harness's own `true` value is wrong by 447%. The
//!    5-center row (10.56 bohr) is contaminated at the 1e-6 level. Those rows
//!    measure IEEE754, not CSAM.
//!
//! 3. **`Operator::erf` was refused without ever being measured.** The harness
//!    has no `erf` path at all — only Coulomb and erfc-by-subtraction. And erf
//!    is the one attenuated kernel Psi4 actually screens with CSAM in
//!    production: `Libint2ErfERI` (`psi4/src/psi4/libmints/eri.cc:179,182`)
//!    builds BOTH its compute and its sieve engine on
//!    `libint2::Operator::erf_coulomb`, and Psi4's RSH exchange is assembled
//!    as `alpha*K_full + beta*K_erf` — erfc is never formed in any Psi4 SCF
//!    path (`grep erf_complement_eri` over `libfock`/`libscf_solver`/`scfgrad`:
//!    zero hits). Refusing erf was refusing the best-evidenced case.
//!
//! What SURVIVES the correction: CSAM is genuinely non-rigorous (the paper says
//! so in its own voice, and the Coulomb rows of the harness — which do NOT
//! involve cancellation — show real violations of 1.39-2.53x); its error grows
//! with system size at fixed threshold (Fig. 2); and CSB is the rigorous
//! alternative for short-range kernels. What does NOT survive: any claim about
//! the MAGNITUDE of a CSAM erfc energy error, in either direction.
//!
//! # Operator routing: Coulomb estimates here, short-range kernels to CSB
//!
//! `csam_x_table` accepts `Coulomb` and `ErfCoulomb`; `ErfcCoulomb` is
//! directed to [`crate::csb`] instead. This is the paper's own recommendation,
//! not an invention — its Conclusion (PDF p. 71) says:
//!
//! > In fact, one could already consider CSB as the new default estimate for
//! > short-range operators due to its rigorous nature, ease of implementation,
//! > low cost, and vastly superior performance compared to the QQ bound.
//!
//! and Sec. III A backs it with measurement: under `erfc` the *rigorous* CSB
//! already reaches a 2.5x speedup at `theta = 1e-14` "with virtually no error
//! increase", such that "there can be no justification for the use of QQ
//! instead of CSB for this operator" (PDF p. 67). When a rigorous bound is
//! that good on a kernel, spending non-rigor there buys nothing.
//!
//! **`ErfCoulomb` is ACCEPTED, deliberately.** Three reasons, in order of
//! weight:
//!
//!   1. It is what the production reference does. Psi4 screens `erf_coulomb`
//!      with CSAM by default (`eri.cc:179,182`, above), and `beta*K_erf` is
//!      the only attenuated exchange any Psi4 SCF path builds.
//!   2. `erf(omega r)/r` is LONG-range — it decays as `1/r`, like Coulomb, and
//!      its Fourier transform `e^{-k^2/(4 omega^2)}/(2 pi^2 k^2)` (paper Table
//!      VI) has no oscillation. The mechanism the old header blamed for the
//!      erfc blow-up (exponential suppression of the `(pp|qq)` numerator of
//!      `X_PQ` at large `p,q` separation, collapsing `X` while the true
//!      integral stays large) does not operate on `erf`: `erf` suppresses
//!      NEAR-field, not far-field. Expect `erf` to behave like Coulomb here,
//!      not like erfc.
//!   3. Refusing it was never measured. Accepting it restores the one case
//!      with direct production precedent, and the user still has `"schwarz"`
//!      and `"csb"` available as rigorous alternatives.
//!
//! This is a PREDICTION recorded before measurement, per CLAUDE.md's
//! experimental protocol. If `erf` is later measured to behave like `erfc`
//! rather than like Coulomb, reason 2 is the claim that failed and the routing
//! should move `ErfCoulomb` to CSB alongside `ErfcCoulomb` — a one-line change
//! at the `match` below.
//!
//! # The ESTIMATE, restated in ferric's (UNSQUARED) Schwarz convention
//!
//! (The square-root bookkeeping below is correct and was verified against
//! Psi4's source — `shell_pair_values_[P,Q] = max |(pq|pq)|` really is the
//! SQUARED Schwarz quantity, so `sqrt(csam_2)` really is the right factor for
//! ferric's unsquared convention. Getting it right is necessary but NOT
//! sufficient for validity: the underlying CSAM inequality is non-rigorous for
//! Coulomb regardless, per the section above.)
//!
//! Psi4 works throughout in SQUARED Schwarz quantities:
//!
//! ```text
//!   mn_mn = shell_pair_values_[N,M]        = max_{mu in M, nu in N} |(mu nu|mu nu)|
//!   rs_rs = shell_pair_values_[S,R]        = max_{rho in R,sig in S} |(rho sig|rho sig)|
//!   X_PQ  = shell_pair_exchange_values_[Q,P]
//!         = max_{p in P, q in Q} |(pp|qq)| / ( sqrt(|(pp|pp)|) * sqrt(|(qq|qq)|) )
//!   csam_2 = max(X_MR * X_NS, X_MS * X_NR)
//!   mnrs_2 = mn_mn * rs_rs * csam_2                  <-- SQUARED bound on |(mn|rs)|
//!   accept iff |mnrs_2| >= screening_threshold_squared_
//! ```
//!
//! `X_PQ` is a Cauchy-Schwarz ratio (numerator and denominator are both
//! Coulomb self-repulsions of the same operator), so it lies in `[0, 1]`
//! REGARDLESS of whether the surrounding convention is squared or not — it is
//! not itself a "squared" quantity that needs re-rooting.
//!
//! ferric's Schwarz table (`crate::schwarz::schwarz`) stores the UNSQUARED
//! quantity `Q(i,j) = sqrt(max|(ij|ij)|)`, and `SchwarzBounds::estimate` is
//! already `Q(i,j) * Q(k,l)` — the unsquared bound on `|(ij|kl)|` directly
//! (verified by reading `crates/ferric-scf/src/screening.rs`: `estimate`
//! returns `self.q[(sh1,sh2)] * self.q[(sh3,sh4)]` with no further
//! squaring/rooting anywhere in the call chain). So:
//!
//! ```text
//!   ferric  Q(i,j) * Q(k,l)               = sqrt(mn_mn * rs_rs)   [Psi4 terms]
//!   ferric CSAM bound on |(ij|kl)|         = sqrt(mnrs_2)
//!                                          = sqrt(mn_mn * rs_rs * csam_2)
//!                                          = Q(i,j) * Q(k,l) * sqrt(csam_2)
//! ```
//!
//! i.e. the multiplicative refinement factor to compose on top of ferric's
//! existing (unsquared) Schwarz estimate is **`sqrt(csam_2)`, NOT `csam_2`**.
//! Using the bare (squared) `csam_2` here would silently square the effective
//! decay factor — e.g. a true per-pair ratio of 0.01 would enter as 0.0001
//! instead — making the estimate underestimate far MORE severely than CSAM
//! already does. That hazard was real and is resolved; it was simply not the
//! cause of the non-rigor documented at the top of this module, which
//! survives a perfectly correct square-root convention (the independent
//! Python reproduction works entirely in Psi4's own squared convention and
//! still violates).
//!
//! # Per-operator tables
//!
//! `X` must be built from the SAME operator the caller will screen with —
//! never reuse a Coulomb-built `X` table under `erfc`/`erf`. Attenuation
//! already lives inside the `(pp|qq)`/`(pp|pp)` integrals that build `X`, so
//! (unlike `qqr.rs`'s now-removed `exp(-omega^2 R^2)` term) there is no
//! separate operator-dependent geometric factor to get wrong here — but the
//! table itself is operator-specific and this module's public entry point
//! takes an `Operator` for exactly that reason.
//!
//! `csam_x_table`'s accepted set versus the `{Coulomb, ErfCoulomb,
//! ErfcCoulomb}` that `schwarz()` supports:
//!
//! ```text
//!   Coulomb       accepted  — non-rigorous, nH-scale error (Table V), the
//!                             regime CSAM was designed and measured for
//!   ErfCoulomb    accepted  — long-range like Coulomb; Psi4's own production
//!                             CSAM kernel (eri.cc:179,182)
//!   ErfcCoulomb   routed    — typed error naming `screening = "csb"`; the
//!                             paper's own recommendation for short-range
//!   everything else rejected — Yukawa / the geminal kernels, because
//!                             `Engine::new_2e` does not accept them
//! ```
//!
//! # Precision discipline (mirrors `schwarz.rs::SCHWARZ_TABLE_PRECISION`)
//!
//! Exactly the same hazard applies here as to the Schwarz table itself: if
//! the diagonal `(pp|pp)` pass or the `(PP|QQ)` quartet pass is built at a
//! libint2 engine precision no tighter than the threshold CSAM will enforce,
//! libint2 may decline to compute a genuinely-small-but-nonzero block and
//! `Engine::compute_quartet`/hand back `None`, which this module would
//! otherwise record as `X = 0.0`. A stored `X = 0.0` makes `csam_2 = 0` for
//! that shell-pair combination, i.e. the CSAM bound at that quartet collapses
//! to exactly zero regardless of the true integral — a silent, unconditional
//! UNDERESTIMATE, which is fatal for a screening bound (see the long comment
//! on `SCHWARZ_TABLE_PRECISION` for the measured alkane_8 fingerprint of this
//! exact failure mode on the Schwarz table). So `CSAM_TABLE_PRECISION`
//! reuses [`crate::schwarz`]'s policy: build at precision `0.0` (libint2
//! internal prescreening disabled for the table build only — the screening
//! THRESHOLD applied downstream by the caller is unaffected), and floor the
//! per-shell self-repulsion normalizers so no divide-by-zero or spurious
//! `X = 0/0` can occur.
//!
//! # Not parallelized
//!
//! Unlike `schwarz()` / `schwarz3_aux()`, this builder is serial-only. It is
//! O(nshells^2) quartet evaluations (one `(PP|QQ)` per unordered shell pair),
//! same asymptotic cost as the Schwarz table's O(nshells^2) `(ij|ij)` pass,
//! so it is not expected to dominate setup — but that has NOT been measured
//! here (no benchmark job was run to build this module; see the task's
//! no-compute constraint). If profiling later shows this table's build time
//! matters at large nshells, mirror `schwarz.rs`'s row-blocked rayon split
//! rather than parallelizing per-pair (that file's own comments explain why
//! per-pair parallelization of a libint2-engine loop is actively harmful).
//! Flagged here as an explicit gap rather than guessed at.

use crate::basis_bridge::PreparedBasis;
use crate::engine::Engine;
use crate::operator::{Operator, OperatorKind};
use crate::schwarz::SCHWARZ_Q_FLOOR;
use ferric_core::FerricError;
use ndarray::Array2;

/// Engine precision used to build the CSAM `X` table. See the module-level
/// "Precision discipline" section: this must be at least as tight as
/// `schwarz::SCHWARZ_TABLE_PRECISION`, and for the same reason. Sharing the
/// literal `0.0` (rather than importing the Schwarz constant, which is
/// crate-private) keeps this module's contract self-contained and equally
/// legible if the two are ever tuned independently in the future.
const CSAM_TABLE_PRECISION: f64 = 0.0;

/// Floor applied to the per-shell self-repulsion normalizer
/// `sqrt(|(pp|pp)|)` before it is used as an `X` denominator, so a
/// genuinely-zero (or underflowed) diagonal cannot produce a `0/0` or an
/// artificially inflated `X` from dividing by a near-zero number. Same value
/// and rationale as `schwarz::SCHWARZ_Q_FLOOR` (re-exported from that module
/// rather than redefined, so the two floors can never drift apart silently).
const CSAM_NORM_FLOOR: f64 = SCHWARZ_Q_FLOOR;

/// Per-shell-function self-repulsion norms `sqrt(|(pp|pp)|)`, one row per
/// shell holding one entry per contracted function `p` in that shell — the
/// same `(PP|PP)` diagonal pass `schwarz_pair` uses, but keeping every
/// function's own value rather than collapsing to the shell max, because
/// `X_PQ` in Psi4's definition maximizes over `(p, q)` pairs of INDIVIDUAL
/// functions within the two shells, not over shells as a whole.
fn shell_function_norms(eng: &mut Engine, prep: &PreparedBasis, nsh: usize) -> Vec<Vec<f64>> {
    let dims = prep.shell_dims();
    (0..nsh)
        .map(|p| {
            let np = dims[p];
            match eng.compute_quartet(prep, p, p, p, p) {
                Some(block) => (0..np)
                    .map(|a| {
                        // (aa|aa) sits on the fully-diagonal element of the
                        // np^4 block for the (p,p|p,p) quartet: index
                        // ((a*np+a)*np+a)*np+a.
                        let idx = ((a * np + a) * np + a) * np + a;
                        block[idx].abs().sqrt().max(CSAM_NORM_FLOOR)
                    })
                    .collect(),
                None => vec![CSAM_NORM_FLOOR; np],
            }
        })
        .collect()
}

/// Build the CSAM exchange-type table `X[P*nsh+Q] = X[Q*nsh+P] =
/// max_{p in P, q in Q} |(pp|qq)| / (norm[p] * norm[q])`, symmetric by
/// construction (the maximization and the quartet `(PP|QQ)` are both
/// symmetric under `P <-> Q`).
///
/// `X_PP` (diagonal) is always exactly `1.0` by Cauchy-Schwarz equality
/// (`(pp|pp)` divided by its own norms), which is the trivial-limit anchor
/// this module's tests check directly.
///
/// Accepts `Coulomb` and `ErfCoulomb`. `ErfcCoulomb` returns `Err` naming
/// `screening = "csb"` — the rigorous [`crate::csb`] bound is the paper's own
/// recommended default for short-range operators (Conclusion, PDF p. 71), and
/// under `erfc` it already reaches a 2.5x speedup at `theta = 1e-14` with
/// virtually no error increase, so non-rigor buys nothing there. Also returns
/// `Err`, as before, for any operator `Engine::new_2e` cannot build at all
/// (Yukawa, the geminal kernels).
///
/// This is a ROUTING decision, not a safety verdict: CSAM is non-rigorous for
/// `erfc` and for `Coulomb` alike, and no `erfc` energy error was ever
/// measured (see the "Correction" section of this module's header for why the
/// figures that previously justified a blanket refusal do not support one).
pub fn csam_x_table(op: Operator, prep: &PreparedBasis) -> Result<Array2<f64>, FerricError> {
    // Routing, per the module header's "Operator routing" section:
    //   - Coulomb:    CSAM's designed and measured regime (paper Tables III/V).
    //   - ErfCoulomb: long-range like Coulomb, and Psi4's own production CSAM
    //                 kernel (eri.cc:179,182 build BOTH compute and sieve
    //                 engines on libint2::Operator::erf_coulomb).
    //   - ErfcCoulomb: send to CSB. Not because CSAM is measured-dangerous
    //                 there — it was not measured — but because the rigorous
    //                 bound is both available and, per the paper, excellent on
    //                 exactly this kernel. Choosing rigor when rigor is free
    //                 needs no further justification.
    match op.kind {
        OperatorKind::Coulomb | OperatorKind::ErfCoulomb => {}
        OperatorKind::ErfcCoulomb => {
            return Err(FerricError::General(format!(
                "CSAM screening is not offered for the short-range operator {:?}; use \
                 screening = \"csb\" instead. CSB (Thompson & Ochsenfeld, JCP 147, 144101 \
                 (2017), Eq. (8)) is a RIGOROUS upper bound, and the paper's own conclusion \
                 recommends it as the default estimate for short-range operators: under erfc \
                 it reaches a 2.5x speedup at threshold 1e-14 with virtually no error \
                 increase, so the non-rigorous CSAM estimate has nothing to add here. \
                 screening = \"schwarz\" also remains available and rigorous.",
                op.kind
            )))
        }
        _ => {
            return Err(FerricError::Libint(format!(
                "operator {:?} not implemented for CSAM screening",
                op.kind
            )))
        }
    }
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let mut eng = Engine::new_2e(op, prep, CSAM_TABLE_PRECISION)?;

    // Diagonal pass: per-function self-repulsion norms, shell by shell.
    let norms = shell_function_norms(&mut eng, prep, nsh);

    // Off-diagonal pass: one (PP|QQ) quartet per unordered shell pair.
    let mut x = Array2::<f64>::zeros((nsh, nsh));
    for p in 0..nsh {
        x[(p, p)] = 1.0; // exact Cauchy-Schwarz equality, set directly (no
                          // engine roundoff risk on the value that anchors
                          // the trivial limit).
        let np = dims[p];
        for q in 0..p {
            let nq = dims[q];
            let maxv = match eng.compute_quartet(prep, p, p, q, q) {
                Some(block) => {
                    // (PP|QQ) block is laid out as shell-dims (p1,p2,p3,p4)
                    // = (np,np,nq,nq); (aa|bb) is at index
                    // ((a*np+a)*nq+b)*nq+b.
                    let mut m = 0.0f64;
                    for a in 0..np {
                        for b in 0..nq {
                            let v = block[((a * np + a) * nq + b) * nq + b].abs();
                            if v > m {
                                m = v;
                            }
                        }
                    }
                    m
                }
                None => 0.0,
            };
            // CORRECTION (2026-09-14): an earlier revision of this comment
            // claimed "Psi4 does not recompute a per-(a,b) numerator either".
            // That was FACTUALLY WRONG. Psi4 (`twobody.cc`, the `max_val`
            // loop at lines ~387-394) computes
            //
            //     X_PQ = max_{a,b} ( |(aa|bb)| / (sqrt_a * sqrt_b) )
            //
            // i.e. the max of the per-function RATIOS: each (a,b) pair keeps
            // its OWN numerator `|(aa|bb)|` paired with its OWN denominator.
            // ferric instead hoists a single constant
            // `maxv = max_{a,b} |(aa|bb)|` and maximizes `maxv / denom`, which
            // selects the SMALLEST denominator -- i.e.
            //
            //     X_PQ(ferric) = max_{a,b} |(aa|bb)| / min_{a,b} (n_a * n_b)
            //
            // Since max(f)/min(g) >= max(f/g), ferric's X is always >= Psi4's,
            // never smaller (verified numerically on multi-function shells:
            // every table entry came out ferric-looser or equal, zero entries
            // tighter). A LOOSER X means a LARGER estimate, so this
            // discrepancy makes ferric underestimate LESS than Psi4 -- it is
            // not the cause of the bound violation documented in this module's
            // header, and "fixing" it to match Psi4 exactly would make the
            // underestimate WORSE, not better. It is left as-is deliberately;
            // this comment exists so the difference is recorded rather than
            // rediscovered. Note also that for multi-function shells ferric's
            // raw ratio can exceed 1 (the diagonal reached 2.34 in the same
            // check) and is clamped below, where Psi4's is exactly 1.
            let mut ratio_max = 0.0f64;
            for a in 0..np {
                for b in 0..nq {
                    let denom = norms[p][a] * norms[q][b];
                    let ratio = maxv / denom.max(CSAM_NORM_FLOOR * CSAM_NORM_FLOOR);
                    if ratio > ratio_max {
                        ratio_max = ratio;
                    }
                }
            }
            // X is a Cauchy-Schwarz ratio and must lie in [0, 1]; clamp
            // defensively against roundoff pushing a near-1 ratio slightly
            // over (would otherwise make sqrt(csam_2) able to exceed 1 and
            // loosen -- never tighten -- past plain Schwarz, which is safe,
            // but clamping keeps the invariant exact rather than "safe by
            // accident").
            let xv = ratio_max.min(1.0);
            x[(p, q)] = xv;
            x[(q, p)] = xv;
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis_bridge::PreparedBasis;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    #[test]
    fn test_csam_x_diagonal_is_one() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let x = csam_x_table(Operator::coulomb(), &prep).unwrap();
        let nsh = prep.nshells();
        for p in 0..nsh {
            assert!(
                (x[(p, p)] - 1.0).abs() < 1e-12,
                "X[{p},{p}] = {} != 1.0 (Cauchy-Schwarz equality on the diagonal)",
                x[(p, p)]
            );
        }
    }

    #[test]
    fn test_csam_x_symmetric_and_bounded() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let x = csam_x_table(Operator::coulomb(), &prep).unwrap();
        let nsh = prep.nshells();
        for p in 0..nsh {
            for q in 0..nsh {
                assert!(
                    (x[(p, q)] - x[(q, p)]).abs() < 1e-12,
                    "X not symmetric at ({p},{q})"
                );
                assert!(
                    x[(p, q)] >= 0.0 && x[(p, q)] <= 1.0 + 1e-9,
                    "X[{p},{q}] = {} out of [0,1]",
                    x[(p, q)]
                );
            }
        }
    }

    #[test]
    fn test_csam_x_rejects_unsupported_operator() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        assert!(csam_x_table(Operator::yukawa(1.0), &prep).is_err());
    }

    /// `ErfcCoulomb` is ROUTED to CSB, not silently estimated: the short-range
    /// kernel is where the paper's own conclusion recommends the rigorous CSB
    /// bound as the default (PDF p. 71), and where CSB is measured to be
    /// excellent (2.5x at theta=1e-14 with virtually no error increase).
    ///
    /// The error MESSAGE is part of the contract: a user who hits this must be
    /// told which knob to turn, not merely that something was refused. This
    /// test asserts the message names `csb`, so a future edit that drops the
    /// pointer fails here.
    #[test]
    fn test_csam_x_routes_erfc_to_csb() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();

        let err = csam_x_table(Operator::erfc(1.0), &prep)
            .expect_err("CSAM must route the short-range erfc kernel to CSB");
        let msg = format!("{err}");
        assert!(
            msg.contains("csb"),
            "the error must name the rigorous alternative the user should switch to; got: {msg}"
        );
    }

    /// `ErfCoulomb` is ACCEPTED. This is the arm that changed on 2026-09-14,
    /// and it is deliberately a SEPARATE test from the Coulomb ones so that a
    /// regression to blanket refusal fails loudly here rather than quietly
    /// widening an existing assertion.
    ///
    /// Justification (module header, "Operator routing"): Psi4 screens
    /// `erf_coulomb` with CSAM in production — `Libint2ErfERI` builds both its
    /// compute and its sieve engine on `libint2::Operator::erf_coulomb`
    /// (`psi4/src/psi4/libmints/eri.cc:179,182`) — and `erf(omega r)/r` is
    /// LONG-range, decaying as `1/r` like Coulomb, so the far-field `X_PQ`
    /// collapse that motivates routing `erfc` away does not operate on it.
    /// The previous refusal of `erf` was never measured.
    ///
    /// This test checks the table is BUILDABLE and well-formed, which is all a
    /// no-compute change can honestly claim. It does NOT establish that CSAM's
    /// `erf` error is Coulomb-like — that is the open prediction recorded in
    /// the module header, and it needs a measurement this test does not make.
    #[test]
    fn test_csam_x_accepts_erf() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();

        let x = csam_x_table(Operator::erf(1.0), &prep)
            .expect("erf is long-range and is Psi4's own production CSAM kernel");
        let nsh = prep.nshells();
        for p in 0..nsh {
            assert!(
                (x[(p, p)] - 1.0).abs() < 1e-12,
                "erf X[{p},{p}] = {} != 1.0 (Cauchy-Schwarz equality on the diagonal)",
                x[(p, p)]
            );
            for q in 0..nsh {
                assert!(
                    x[(p, q)] >= 0.0 && x[(p, q)] <= 1.0 + 1e-9,
                    "erf X[{p},{q}] = {} out of [0,1]",
                    x[(p, q)]
                );
            }
        }

        // Control: Coulomb must still succeed too, so this test cannot pass by
        // CSAM having become permissive about everything.
        assert!(
            csam_x_table(Operator::coulomb(), &prep).is_ok(),
            "Coulomb must remain supported — a blanket success would make the routing test vacuous"
        );
    }
}
