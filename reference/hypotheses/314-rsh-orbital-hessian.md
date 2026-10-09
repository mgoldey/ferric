# Issue #314 — pre-registered hypotheses

Dated 2026-10-04, branch `fix/rsh-orbital-hessian`, base `origin/main` @ af833f6c.
Committed BEFORE any ω ≠ 0 measurement, per the repo's Experimental Protocol.

## The defect, as verified in the code (not from the issue text)

* `uhf_newton.rs:53` — `UhfNewtonInputs` carries `pub k_mix_sr: f64` and nothing
  else about the kernel. `omega` appears nowhere in the file.
* `uhf_newton.rs:208` — `let c_k = inp.k_mix_sr; df_σ = δJ − c_k·δK_σ`, with
  `δK_σ` from `build_jk_with_pool` at the ambient (Coulomb) operator.
* `rhf_newton.rs:66,183` — the same shape for the singlet channel
  (`δF = δJ − ½·c_k·δK`), and `rhf_newton.rs:304` for the triplet channel
  (`δF = −½·c_k·δK`).
* `rohf_newton.rs:45,368` — same scalar; its doc comment says "ignored for RSH",
  and `rohf.rs:966,1016` pass `k_mix_sr: if k_mix.omega > 0.0 { 0.0 } else { c_k }`,
  i.e. the ROHF Newton path currently drops exchange response ENTIRELY for RSH.
* `cdft_driver.rs:1116` hardcodes `k_mix_sr: 1.0` (pure-HF cDFT; no XC, no ω).
* `stability.rs:1101` `ks_reference_is_analysable` returns
  `Err(StabilitySkip::RangeSeparated)` on `omega != 0.0`.

## The fact that shapes the fix (and that the issue does not state)

`driver.rs:125` is explicit:

> "RSH exchange is density-fitted ONLY: there is no conventional four-centre
> erf/erfc exchange path."

The converged RSH Fock is assembled by `fock_assembly::subtract_rsh_exchange`
from two `DfK` fitters (`Operator::erfc(ω)` and `Operator::erf(ω)`), built once
in `driver::prepare`. The Newton matvec, by contrast, uses the direct
four-centre `build_jk_with_pool`.

Therefore the Hessian's RSH exchange response MUST come from the SAME DF-K
fitters. Using direct four-centre erf/erfc integrals instead (which libint2 can
compute, and for which `schwarz.rs` admits screening) would give the second
derivative of a DIFFERENT energy functional from the one the SCF minimized, and
the analytic-vs-FD check would then disagree at the DF fitting error (~1e-5–1e-4),
not at 3e-10.

## H_physics

The ω ≠ 0 orbital Hessian, with exchange response
`δK_total,σ = c_SR·δK[erfc(ω)](δD_σ) + c_LR·δK[erf(ω)](δD_σ)` built from the same
DF-K fitters the Fock uses, is the true second derivative of the RSH energy.

**If real I expect:**

1. **ω = 0 bit-identity.** `k_mix.omega == 0.0` must leave the existing code path
   untouched, so every existing matvec output is bit-identical (exact `f64`
   equality), and every existing Newton/stability test keeps its number.
2. **Analytic-vs-FD at ω ≠ 0 to ~3e-10**, matching the bar the ω = 0 test holds,
   PROVIDED the FD reference Fock is built from the same DF-K fitters. The
   agreement must be at machine-ish precision, not at DF fitting error.
3. **N2⁺ / def2-SVP: λ_min > 0 at ω = 0.53 and λ_min < 0 at ω = 0.56**, sign and
   rough magnitude matching PySCF's +1.129e-4 and −2.712e-3. The straddle is the
   premise, and I will assert the premise as well as the conclusion.
4. **The downhill eigenvector descends.** Seeding a UHF/UKS re-converge along the
   λ_min < 0 eigenvector at ω = 0.56 / 0.60 / 0.70 must reach the lower state
   #313 found by hand-rotating (ΔE −8.9e-5, −4.6e-4, −2.1e-3). The Hessian's
   direction and #313's hand-chosen 45° β HOMO/LUMO rotation must land on the
   same state.

## H_artifact — distinguishable observables

The ways this fix can be wrong, each with an observable that differs from
H_physics, so the experiment can tell them apart:

* **A1: erf/erfc swapped.** Then I have built `c_SR·K_LR + c_LR·K_SR`. Observable:
  analytic-vs-FD *fails* (the FD reference uses the Fock's correct assignment),
  and λ_min at 0.53/0.56 would be some other pair of numbers. Distinguished from
  H_physics by (2), which H_physics passes and A1 fails. Mutation (a) tests this.
* **A2: c_LR dropped (only the SR term built).** Observable: at the wB97X-ish
  `c_LR = 1.0` used here, FD disagreement of order the LR exchange response
  itself — large, not subtle. Mutation (b).
* **A3: `k_mix.sr` used for both coefficients.** Observable: wrong by
  `(c_LR − c_SR)·δK_LR`; FD fails. Mutation (c).
* **A4: the DF-vs-direct inconsistency I am avoiding.** If I had built direct
  four-centre erf/erfc K instead, FD would agree only to the DF fitting error.
  Observable: an FD residual stuck near 1e-5–1e-4 that does NOT shrink when the
  FD step is refined — i.e. a *floor*, not a step-size-limited error. This is the
  diagnostic that distinguishes "wrong operator" from "FD step too coarse", and
  it is why I record the FD residual at two step sizes.
* **A5: the sign flip is reproduced for the wrong reason.** A λ_min that goes
  negative because the eigensolve is iteration-starved, or because the state
  converged is not the symmetric branch PySCF probed, would look like success.
  Observable: the eigensolver's own `converged` flag and residual must be clean,
  AND J(ω) must reproduce #313's +3.928e-3 / +2.676e-4 (so I know I am on the
  same branch). Both are asserted, not assumed.
* **A6: the test cannot fail.** If λ_min at 0.53 and 0.56 do not actually straddle
  zero, a test asserting "signs differ" is arithmetic, not measurement. I assert
  the premise (λ_min(0.53) > 0 AND λ_min(0.56) < 0, each with its own margin)
  rather than only `signum() != signum()`.

## Where H_physics could be REFUTED, and that would be the finding

The issue's diagnosis is that the Coulomb kernel is what blocks the sign flip.
It could instead be that:

* the sign flip needs the **UKS f_xc kernel** at ω ≠ 0 as well, and the RSH
  functional in question is a meta-GGA or VV10-carrying one whose kernel ferric
  does not have (wB97X-V carries VV10 — `lr_kernel.rs:102` refuses it) — in which
  case ω ≠ 0 becomes analysable only for a VV10-free RSH, and the N2⁺ reproduction
  needs whichever functional PySCF's probe used;
* ferric's DF-K fitting error at the N2⁺/def2-SVP scale (~1e-4-ish in energy)
  could be comparable to λ_min = +1.129e-4 at ω = 0.53, in which case the sign at
  0.53 is **not resolvable** by this construction and the honest answer is a
  resolution limit, not a reproduction. I will report the DF-K error scale next to
  λ_min rather than quoting the agreement alone.

Either outcome is reported with the data.

## ROHF scope — decided in advance, to be justified from measurement

`StabilitySkip::Rohf` is a SEPARATE refusal: the Roothaan open-shell Hessian is a
third operator, and fixing the exchange kernel there does not make ROKS
analysable. Preliminary decision: **fix the kernel in `rohf_newton.rs` too**
(because `rohf.rs:966` currently passes `0.0`, i.e. silently drops ALL exchange
response for RSH ROKS Newton steps, which is a real defect in the Newton
accelerator independent of stability), but do NOT claim ROKS stability analysis
is unblocked. Final decision recorded in the report with its justification.
