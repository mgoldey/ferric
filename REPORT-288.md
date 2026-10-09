# REPORT — issue #288: ω tuning has no control over the cation state

## Part 1: hypotheses, written and committed BEFORE any measurement


Date: 2026-10-03. Branch `fix/omega-tuning-cation-state` off `origin/main@0b4fa7dc`.
Nothing in this file has been measured yet. It exists so the conclusion cannot
be fitted to the data after the fact.

## What is being changed

`omega_tuning::eval_j` solves the doublet cation from the default guess (MINAO
projection, `uhf_guess_mos`) at every ω, and checks only `converged`.
`tune_omega`'s golden-section search visits ω out of order. So the cation state
is whatever the symmetric guess happens to reach at each ω independently, and
two J points in one curve can sit on two different branches with no warning.

Planned deliverable: continuation seeding (seed each SCF from the converged
orbitals of the NEAREST already-evaluated ω) plus a branch-consistency check
built from quantities available without a stability operator.

## Scope: item 3 is out

Issue task 3 (`uhf_internal_stability` on each cation) depends on the separate
open issue "Range-separated hybrids: orbital-Hessian matvec ...":
`stability::ks_reference_is_analysable` refuses ω ≠ 0 with
`StabilitySkip::RangeSeparated`. No ω≠0 orbital Hessian is written here.

## Physics hypotheses (P)

* **P1 — the branch exists and continuation holds it.** For N2/def2-SVP
  ωB97X-V the symmetric D∞h ²Σg⁺ N2⁺ stops being a minimum between ω = 0.53
  and 0.56 (PySCF `symmetric_cation_probe`: λ_min +1.13e-4 at 0.53, −2.71e-3
  at 0.56). If a lower, symmetry-broken (hole-localized) state exists past the
  onset and ferric's UKS can reach it, then seeding ω = 0.60 from a converged
  ω = 0.56-or-above broken cation gives E(0.60) **below** the symmetric
  E(0.60), with a visibly non-zero per-atom spin-population asymmetry
  |p_N1 − p_N2| on the two symmetry-equivalent nitrogens.
* **P2 — J gains a kink at the onset.** If branches are mixed, J(ω) sampled
  densely across 0.50–0.60 shows a slope change or a non-monotone step at the
  onset; continuation on one branch removes it from that branch's curve.
* **P3 — ⟨S²⟩ is a WEAK discriminator here.** PySCF's own probe gives
  ⟨S²⟩ = 0.7559 (ω 0.53) and 0.7563 (ω 0.56) — a 3e-4 change across the onset,
  which is the same size as the smooth ω-drift of ⟨S²⟩ over 0.2→0.5
  (0.7532→0.7556). So a ⟨S²⟩ threshold alone cannot separate "hole localized"
  from "ω moved": the spatial spin-population asymmetry is the discriminator
  that can, because the symmetric state has p_N1 = p_N2 by symmetry and the
  broken one does not. Prediction: the asymmetry separates the branches by
  orders of magnitude while ⟨S²⟩ separates them by <1e-3.
* **P4 — the default guess is symmetric, so today's code stays symmetric.**
  The MINAO projection guess for N2 respects D∞h; the β HOMO/LUMO mixing in
  `solve_uhf_fockmod` only applies when nocc_a == nocc_b, which a doublet is
  not. So today's `eval_j` at ω = 0.60 should land on the SYMMETRIC saddle,
  p_N1 ≈ p_N2 to grid noise, and its J should lie on the symmetric branch
  (continuing the 0.2…0.5 trend to J ≈ 0 near 0.56), NOT on the broken branch.

## Artifact hypotheses (A) — what a broken implementation would show

* **A1 — seeding does nothing (silent no-op).** If `solve_uhf_with_guess` is
  handed MOs that are then discarded (wrong shape silently ignored, wrong
  argument position, the seed overwritten by `init_guess_density`), then
  continuation-on vs continuation-off give **bit-identical** energies at every
  ω, including past the onset. P1 predicts a difference past the onset;
  A1 predicts zero difference everywhere. **These are distinguishable**, so
  the experiment is valid — but only if the no-difference case is reported as
  a no-op suspicion, not as "the branch does not exist".
  *Discriminator:* the seed must also be shown to MATTER somewhere, i.e. a
  deliberately wrong seed must change the result. If no seed ever changes
  anything, the plumbing is dead, not the physics.
* **A2 — continuation only ever seeds from the previous call.** Golden section
  visits ω out of order, so "previous eval" ≠ "nearest ω". If I accidentally
  implement previous-eval seeding, the N2 ω-ordered sweep (monotone ω) cannot
  tell the two apart — both are the same there. **The monotone sweep CANNOT
  distinguish A2 from the intended design and must not be used as its
  evidence.** The nearest-ω selector therefore needs a direct unit test on a
  synthetic out-of-order ω list, not an SCF test.
* **A3 — the asymmetry metric is an artifact of the AO partition.** A
  Mulliken/Löwdin spin population on symmetry-equivalent atoms could be
  non-zero for a symmetric density if the per-atom AO assignment were wrong
  (shell→atom mapping off by one). *Discriminator:* a symmetric state must
  give |p_N1 − p_N2| at the 1e-10 level, i.e. the metric's own trivial limit.
  If the symmetric state shows a "large" asymmetry, the metric is broken, not
  the state.
* **A4 — the flag fires on everything (vacuous alarm).** A branch threshold set
  below the ω-drift of the metric flags every adjacent pair, which looks like
  "the check works". *Discriminator:* H2O and NH3 across 0.2–0.6 must produce
  ZERO flags; the threshold must be derived from the measured both-sides
  separation, never guessed.

## Exactness anchors (written before the sweep, must pass before it runs)

1. `continuation disabled` ⇒ every J, ε_HOMO, IP is **bit-identical** to the
   pre-change code (default struct field is off ⇒ the old code path).
2. The asymmetry metric on a converged SYMMETRIC N2⁺ is ≤ 1e-8 (trivial limit
   of "hole localization": no localization).
3. The nearest-ω selector returns the only candidate when exactly one exists,
   and `None` on the first evaluation (no neighbour) ⇒ default guess ⇒ anchor 1
   covers the first eval for free.

## Pass-condition reachability

The N2 test's GO condition is "the tool flags the 0.50–0.60 window". It is
reachable only if ferric's branch metric actually separates adjacent ω by more
than its ω-drift. That separation is measured FIRST (the table in the report);
the threshold is then placed between the two measured sides. If the measured
separation is not there, the test is NOT written to pass anyway — the negative
is reported, with the dependency on the ω≠0 Hessian named as the reason a
stability verdict would have been the right instrument.

---

# Part 2: measurement and outcome (written after Part 1, which is unmodified)

All numbers below are from this worktree, release build, via
`scripts/validation/run_slot.sh`, `OPENBLAS_NUM_THREADS=1`.

## 2.1 Verdict against the hypotheses

| hypothesis | outcome |
|---|---|
| **P1** broken branch exists and continuation holds it | **SPLIT.** The lower branch EXISTS (−8.9e-5 Ha at ω=0.56, −4.6e-4 at 0.60, −2.1e-3 at 0.70 relative to the symmetric state) but ferric reaches it ONLY when deliberately seeded toward it. Continuation from a neighbouring ω does NOT find it. |
| **P2** J gains a kink at the onset | **REFUTED.** J on the symmetric branch is smooth and strictly monotone across 0.50→0.60, crossing zero between 0.56 and 0.57. Nothing to find. |
| **P3** ⟨S²⟩ is a weak discriminator | **CONFIRMED, and stronger than predicted.** ⟨S²⟩'s smooth ω-drift (1.06e-4 per 0.01 Bohr⁻¹) is ~3 orders of magnitude LARGER than the asymmetry signal separating the branches (2.7e-7). ⟨S²⟩ cannot carry the check at all; it is reported, not gated on. |
| **P4** default guess stays symmetric | **CONFIRMED.** Both arms land on the symmetric branch at every ω in 0.50–0.60; \|ΔE_cation\| ≤ 2.8e-13 Ha. |
| **A1** seeding is a silent no-op | **RULED OUT**, but only after adding a test for it — see the mutation ledger, M3/M6. The discriminator A1 demanded ("a wrong seed must change the result") was initially absent for the `eval_j_seeded` path and a seed-discard mutation SURVIVED. |
| **A2** nearest-ω confused with previous-eval | **RULED OUT** by a unit test on a synthetic out-of-order ω list, exactly as Part 1 required (the monotone SCF sweep cannot distinguish them). |
| **A3** asymmetry is an AO-partition artifact | **RULED OUT.** Trivial limit holds: D_α = D_β gives asymmetry < 1e-14; a lopsided density gives > 0.5; a molecule with no equivalent atoms gives exactly 0. |
| **A4** flag fires on everything | **RULED OUT.** H2O over a full 10-eval tuning run: zero flags, asymmetry at 1e-15. |

The headline is that **the issue's premise was right about the defect and wrong
about the observable**. `eval_j` really does have no state control. But in the
N2 window the defect does not manifest as a mixed-branch J curve: the default
guess is symmetric, it stays symmetric at every ω, and the curve is
self-consistent. What has changed past ω = 0.56 is a CURVATURE of the energy
surface, and no energy, ⟨S²⟩ or population observable sees a curvature. That is
precisely the orbital-Hessian dependency, now measured rather than assumed.

## 2.2 The N2 window, with and without continuation

N2/def2-SVP, ωB97X-V, exact J both states, DF-K def2-universal-jkfit,
(75,110) grid, conv 1e-10 / 1e-7. `omega_tuning_cation_branch.rs::n2_cation_branch_sweep_measurement`.

ARM A = independent solves at each ω (today's behaviour, `continuation: false`).
ARM B = continuation, monotone ω order, seed = nearest already-evaluated.

| ω | E_cation (A) | J (A) | ⟨S²⟩ (A) | asym (A) | asym (B) | seed (B) | \|ΔE_cat\| A−B | \|ΔJ\| A−B |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0.50 | −108.792405628 | +7.900e-3 | 0.755622 | 2.9e-10 | 2.9e-10 | default | 0.000e0 | 0.000e0 |
| 0.51 | −108.790957016 | +6.539e-3 | 0.755728 | 2.3e-10 | 2.78e-8 | from 0.50 | 2.8e-13 | 5.9e-9 |
| 0.52 | −108.789510606 | +5.216e-3 | 0.755835 | 1.3e-10 | 2.95e-8 | from 0.51 | 1.4e-13 | 6.8e-9 |
| 0.53 | −108.788066681 | +3.928e-3 | 0.755942 | 6.7e-10 | 3.12e-8 | from 0.52 | 1.1e-13 | 7.1e-9 |
| 0.54 | −108.786625573 | +2.674e-3 | 0.756049 | 2.8e-10 | 3.22e-8 | from 0.53 | 2.6e-13 | 7.4e-9 |
| 0.55 | −108.785187660 | +1.455e-3 | 0.756155 | 3.8e-11 | 3.34e-8 | from 0.54 | 1.4e-13 | 7.7e-9 |
| 0.56 | −108.783753355 | +2.676e-4 | 0.756262 | 9.4e-11 | 3.39e-8 | from 0.55 | 1.1e-13 | 8.0e-9 |
| 0.57 | −108.782323104 | −8.879e-4 | 0.756369 | 1.2e-10 | 3.47e-8 | from 0.56 | 2.6e-13 | 7.4e-9 |
| 0.58 | −108.780897381 | −2.013e-3 | 0.756475 | 2.0e-10 | 3.47e-8 | from 0.57 | 1.7e-13 | 6.3e-10 |
| 0.59 | −108.779476681 | −3.108e-3 | 0.756581 | 2.2e-10 | 3.51e-8 | from 0.58 | 1.4e-13 | 3.7e-8 |
| 0.60 | −108.778061516 | −4.175e-3 | 0.756686 | 2.4e-10 | 3.49e-8 | from 0.59 | 1.1e-13 | 6.3e-9 |

**Seed used per evaluation** is the "seed (B)" column: ARM B's ω = 0.50 is the
first evaluation and has no neighbour, so it uses the solver's own guess; every
later ω continues from the nearest already-evaluated one, which in a monotone
sweep is the previous ω. (Golden section's out-of-order case is covered by the
`nearest_evaluated` unit test, not by this sweep — see A2 above.)

Reading: continuation changes nothing here. Both arms are on one branch, the
energies agree to 1e-13 Ha, and J moves only by the SCF's own stopping noise
(≤3.7e-8 Ha). ARM B's asymmetry sits a couple of decades higher (3e-8 vs
2e-10) purely because a warm start stops at a slightly different point inside
the same basin.

### Cross-check against the generator's `symmetric_cation_probe`

PySCF's probe is the independent construction (different code, different
guesses, orbital-Hessian λ_min available):

| ω | J ferric | J PySCF | \|Δ\| | ⟨S²⟩ ferric | ⟨S²⟩ PySCF | \|Δ\| | PySCF λ_min |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.53 | +3.928000e-3 | +3.928074e-3 | 7.4e-8 | 0.755942 | 0.755942 | 3.0e-7 | **+1.129e-4** |
| 0.56 | +2.676000e-4 | +2.680675e-4 | 4.7e-7 | 0.756262 | 0.756262 | 1.6e-7 | **−2.712e-3** |

ferric is on the identical branch PySCF's probe is on. λ_min flips sign between
these two ω while J and ⟨S²⟩ stay perfectly smooth — the quantitative statement
that **the onset is invisible to every observable available without the orbital
Hessian**. This is the single most load-bearing measurement in this report.

### The lower branch is reachable, but only when seeded toward it

`n2_broken_branch_probe_measurement`: seed the β HOMO/LUMO at 45° (an
occupied→VIRTUAL rotation) and re-solve.

| ω | E symmetric | E seeded | ΔE | asym symmetric | asym seeded | iters |
|---:|---:|---:|---:|---:|---:|---:|
| 0.56 | −108.783753355 | −108.783842824 | **−8.947e-5** | 9.4e-11 | 3.43e-7 | 20 |
| 0.60 | −108.778061516 | −108.778519860 | **−4.583e-4** | 2.4e-10 | 2.70e-7 | 18 |
| 0.70 | −108.764333830 | −108.766388024 | **−2.054e-3** | 3.2e-12 | 3.67e-8 | 15 |

So past the onset the smooth J curve is NOT the ground-state cation curve, and
`tune_omega` does not say so. The drop grows with ω, consistent with an
instability that deepens past its onset.

**A near-miss worth recording.** The first version of this probe rotated the β
HOMO with the orbital BELOW it — a rotation WITHIN the occupied block. That
leaves D_β = C_occ C_occᵀ invariant, so every ω converged in 2 iterations to
the state it started from with ΔE ≤ 5.7e-14 Ha. It looked exactly like "ferric
cannot reach a broken branch", which would have been a false negative closing
the lane. The fix (occ→virt) plus an added in-test guard asserting the seed's
β density actually differs by > 1e-3 is what distinguishes the two. The guard
is now permanent, because that failure mode is silent.

## 2.3 Before/after for every existing ω-tuning test

`omega_star_before_and_after_continuation`. Both arms are verified to visit the
SAME golden-section ω (asserted per evaluation), so the deltas are like-for-like.

| test file | system / basis | ω* OFF | ω* ON | \|Δω*\| | J OFF | J ON | \|ΔJ\| | evals | flag |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `omega_tuning.rs` (`tune_omega_h2_converges_and_improves_j`) | H2 / 6-31G | 0.780486515 | 0.780486515 | **0.000e0** | +6.962690e-5 | +6.962690e-5 | 3.62e-12 | 11 | none |
| `validation_rsh_omega.rs` (`omega_star_h2o_wb97xv_vs_pyscf`) | H2O / def2-SVP | 0.504911565 | 0.504911565 | **0.000e0** | +3.354672e-7 | +2.334593e-7 | **1.020e-7** | 25 | none |
| `validation_rsh_omega.rs` (`omega_star_nh3_wb97xv_vs_pyscf`) | NH3 / def2-SVP | 0.443178728 | 0.443178728 | **0.000e0** | −1.672940e-7 | −1.488471e-7 | 1.845e-8 | 25 | none |

`validation_rsh_omega.rs`'s `j_of_omega_*` tests call `eval_j`, which never
continues, so they are unaffected by construction (not by measurement).

**Decision: `continuation` defaults to FALSE** (`DEFAULT_CONTINUATION`).

ω* is bit-identical in all three cases. J is NOT: it moves 1.02e-7 Ha on H2O.
That is ~200× inside the validation row's J bar (2e-5 Ha) and would pass every
existing test — but "inside the bar" is not "unchanged", and only unchanged
justifies silently changing every existing caller's numbers. So the fix ships
off by default and is opted into per system. The measured reason is in a comment
on the constant, not only here.

(Mechanism of the 1.02e-7: each SCF stops at a slightly different point inside
its 1e-10 energy / 1e-7 density thresholds depending on where it started. It is
stopping noise, not a different state — the per-ω |ΔE_cation| is ≤ 1e-13.)

## 2.4 `branch_tol`: derived, not chosen

My Part-1 design gated on BOTH ⟨S²⟩ and the asymmetry with a shared 0.05
tolerance. **The data refuted that before it shipped**, in two ways:

```
quantity                        symmetric branch, ω 0.50→0.60   broken branch (ω 0.60)
spin asymmetry                  2.4e-10 … 3.5e-8                2.7e-7
⟨S²⟩                            0.755622 → 0.756686             —
⟨S²⟩ drift per 0.01 Bohr⁻¹      1.06e-4                         —
E_cation − E_symmetric          0                               −4.6e-4 Ha
```

1. **0.05 was far too loose.** The real branch switch moves the asymmetry by
   2.7e-7, five orders of magnitude below 0.05, so the check as designed would
   never have fired on the thing it was built for. A tolerance chosen before
   seeing the data was wrong by five decades.
2. **⟨S²⟩ cannot be gated on at all.** Its smooth ω-drift (1.06e-4 per 0.01
   Bohr⁻¹) is ~3 decades LARGER than the asymmetry signal. Any ⟨S²⟩ bar tight
   enough to see a switch fires on every ω step. ⟨S²⟩ is therefore reported on
   every `OmegaEval` and gated on by nothing.

`DEFAULT_BRANCH_TOL = 1e-7`, placed between the two measured sides: above the
symmetric branch's entire range over a 0.10 Bohr⁻¹ sweep (≤3.5e-8), below the
broken branch's value (2.7e-7). Both measured values and this reasoning are in
the constant's doc comment.

**Honest margin statement.** That is ≈1.5 decades of separation from ONE system,
not the many decades a fully localized hole would give. The lower N2⁺ state is
not strongly hole-localized in the Mulliken-spin sense — which is itself a
finding, and the reason this check is a prompt to look, not a proof. Its
absence is explicitly not a certificate.

## 2.5 First evaluation with no neighbour

`nearest_evaluated(&[], ω)` returns `None`, the seed is `OmegaSeed::Default`,
and the SCF uses the solver's own guess (MINAO projection via `uhf_guess_mos`,
or hcore if that fails). Tested three ways:

* `nearest_evaluated_is_not_the_previous_evaluation` (unit, in `omega_tuning.rs`)
  asserts `None` on the empty list, the sole candidate when exactly one exists,
  deterministic tie-breaking to the earlier entry, and — the point of the test —
  that on a golden-section ω order the NEAREST is not the PREVIOUS evaluation
  (`done = [0.4292, 0.5708]`, next ω 0.3292 → index 0, not the just-evaluated
  0.5708).
* `the_first_evaluation_has_no_seed` asserts `evals[0].seed == Default` and
  `evals[1].seed == Continued{..}` on a real tuning run.
* The Python smoke test asserts `seed_from_omega is None` on the first eval.

Because the first evaluation falls back to the default guess, exactness anchor 1
covers it for free: with continuation off, every evaluation is that path.

## 2.6 Task items: done, and left

| item | status |
|---|---|
| **1. Continuation from the nearest already-evaluated ω** | **DONE.** `OmegaSeedState` carries the neutral total density (→ `RhfConfig::init_guess_density`) and the cation α/β MOs (→ `uhf::solve_uhf_with_guess`); `nearest_evaluated` selects by \|Δω\| with deterministic tie-breaking; `OmegaEval::seed` records `Default` or `Continued{from_omega}`. The cation seed is MOs and not a density on purpose: a UKS density guess is spin-summed by `uhf_guess_mos` and would discard the α/β difference that IS the branch. |
| **2. Branch-consistency check without a stability operator** | **DONE, with its reach measured and stated.** `spin_population_asymmetry` over geometrically-equivalent atoms, compared against the same neighbour continuation uses; sets `OmegaEval::branch_changed` and `OmegaTuneResult::branch_warning`. It CANNOT see the N2 onset, because there is no discontinuity there to see — measured in 2.2, not assumed. |
| **3. `uhf_internal_stability` on each cation + `check_cation_stability`** | **NOT DONE — deliberately out of scope.** Depends on the open issue "Range-separated hybrids: orbital-Hessian matvec ...": `stability::ks_reference_is_analysable` refuses ω ≠ 0 with `StabilitySkip::RangeSeparated`, because the matvec builds its exchange response from the plain Coulomb kernel and does not reproduce an RSH Fock's SR/LR split. No ω≠0 Hessian was written. The limitation is documented in the module doc, the Python docstring and `validation.md`. |
| **4. Expose in Python + return per-eval flags** | **DONE.** `tune_omega(..., continuation=, branch_tol=)`; returns `branch_warning` plus per-eval dicts with `omega, eps_homo, ip, j, e_neutral, e_cation, cation_iterations, neutral_iterations, cation_s_squared, cation_spin_asymmetry, seed, seed_from_omega, branch_changed`. `branch_tol <= 0` is the documented off switch. |
| **5. N2 test at ω = 0.50/0.53/0.56/0.60** | **DONE, asserting what was measured** (`n2_onset_is_not_visible_without_an_orbital_hessian`), which is NOT "the tool flags the onset". It pins: the two arms agree to ≤1e-11 Ha; both stay below 1e-6 asymmetry; J is strictly monotone; ⟨S²⟩ drifts smoothly in one direction; a seeded cation at ω = 0.60 IS 4.6e-4 Ha lower; and the plain ω = 0.60 solve is the higher state. If a future change makes the plain tuner reach the lower branch, the monotonicity and asymmetry asserts fail and the doc is revisited. |

### What the ω≠0 orbital-Hessian issue would need, from this work

The instrument this issue wanted is λ_min of the UKS orbital Hessian at ω ≠ 0.
PySCF's values bracket the onset cleanly (+1.1e-4 → −2.7e-3 between ω = 0.53
and 0.56) on a branch ferric reproduces to 7e-8 Ha in J, so ferric has the
right state to analyse and needs only the operator. Concretely the matvec needs
its exchange response split the way the converged Fock is — c_SR·K[erfc(ω)] +
c_LR·K[erf(ω)] — rather than one plain-Coulomb K. Once that exists,
`check_cation_stability` is a small addition here: the cation solve already
returns `ScfResult`, and the per-eval flag plumbing (`branch_changed`,
`branch_warning`) is in place to carry a saddle verdict with no new API.

## 2.7 Mutation ledger

Every mutation was applied to the source, run, and the observed line read off.
Counts are quoted verbatim, including `ignored`, because a green line with zero
tests executed proves nothing.

| # | mutation | test | observed |
|---|---|---|---|
| M1 | `nearest_evaluated`: `min_by` → `max_by` (seed from the FARTHEST evaluated ω) | `nearest_evaluated_is_not_the_previous_evaluation` | **FAILED** — `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 319 filtered out`, panic at `omega_tuning.rs:704` |
| M2 | `if cfg.continuation` → `if false` (continuation silently disabled) | `the_first_evaluation_has_no_seed` | **FAILED** — `0 passed; 1 failed; 0 ignored`, panic at test line 207 |
| M3 | `cat_seed` always `None` (cation seed accepted then discarded) | `the_continuation_seed_reaches_the_cation_through_eval_j_seeded` | **FAILED** — `0 passed; 1 failed; 0 ignored`, panic at line 333; printed `cation 7 iters unseeded vs 7 seeded` (vs 7 → 6 unmutated) |
| M4 | branch gate `if d_asym > tol` → `if false && ...` | `an_impossibly_tight_branch_tolerance_fires` **and** `branch_tol_none_disables_the_check_entirely` | **BOTH FAILED** — `0 passed; 2 failed; 0 ignored`, panics at lines 349 and 401 |
| M5 | `spin_population_asymmetry` returns `0.0 * worst` (metric always zero) | `spin_asymmetry_sees_a_lopsided_spin_density` | **FAILED** — `2 passed; 1 failed; 0 ignored`, panic at line 149 (and note the two trivial-limit tests still PASS, which is why a metric that always returns 0 needed its own non-zero test) |
| M6 | neutral `init_guess_density` assignment removed | `the_continuation_seed_reaches_the_cation_through_eval_j_seeded` | **FAILED** — `0 passed; 1 failed; 0 ignored`, panic at line 340; printed `neutral 6 vs 6` (vs 6 → 5 unmutated) |

**M3 initially SURVIVED, and that is the most useful thing in this ledger.**
Run against `n2_onset_is_not_visible_without_an_orbital_hessian` it gave
`test result: ok. 1 passed`. The reason is structural: that test asserts the
seeded and unseeded arms AGREE, so deleting the seed makes it pass trivially.
The existing `a_different_cation_seed_changes_the_solver_path` exercises
`solve_uhf_with_guess` directly and never reaches the `eval_j_seeded` wrapper.
There was no test that the seed traverses the wrapper at all.

The fix needed a new observable: seeded and unseeded converge to the SAME
energy (that is the point of a correct seed), so energy cannot distinguish
them. `OmegaEval` therefore carries `cation_iterations` and `neutral_iterations`,
and the new test asserts the seeded counts are strictly lower. Margins are thin
but real and reproducible: cation 7 → 6, neutral 6 → 5 on N2⁺/STO-3G at
ω = 0.41 seeded from 0.40. M3 and M6 each now fail at the assertion naming
their own path.

After every mutation the source was restored and verified: `git diff` on
`omega_tuning.rs` shows only the intended `cation_iterations` /
`neutral_iterations` addition, and a grep for all five mutation strings returns 0.

## 2.8 Exactness anchors

| anchor | test | result |
|---|---|---|
| Continuation off ⇒ bit-identical to an independent `eval_j` | `continuation_off_is_bit_identical_to_independent_eval_j` | PASS, and asserts it is testing the DEFAULT path |
| Asymmetry metric on D_α = D_β is 0 | `spin_asymmetry_is_zero_for_a_spin_unpolarized_density` | PASS (< 1e-14) |
| ...and is not identically 0 | `spin_asymmetry_sees_a_lopsided_spin_density` | PASS (> 0.5); M5 proves it can fail |
| ...and is a no-op, not a false alarm, with no equivalent atoms | `spin_asymmetry_is_zero_when_no_atoms_are_equivalent` | PASS (exactly 0 on HF) |
| Nearest-ω selector: empty, singleton, out-of-order, ties | `nearest_evaluated_is_not_the_previous_evaluation` | PASS; M1 proves it can fail |
| Zero flags on a single-branch system over a full run | `h2o_tuning_raises_no_branch_flag` | PASS, asymmetry 1e-15 at all 10 ω |
| The flag is reachable | `an_impossibly_tight_branch_tolerance_fires` | PASS; M4 proves it can fail |

### Reachability near-miss

`an_impossibly_tight_branch_tolerance_fires` and
`branch_tol_none_disables_the_check_entirely` were first written on H2/6-31G
and `an_impossibly_tight...` FAILED on the real code: H2⁺ has ONE electron, so
nocc_β = 0, D_β = 0, and the asymmetry between its two equivalent H is
IDENTICALLY zero at every ω. No tolerance, however tight, can fire there. The
GO condition was unreachable by construction — the Part-1 "check the pass
condition is REACHABLE" rule catching a real case. Both tests moved to
N2/STO-3G, which has a β occupied set, and each now carries an explicit
assertion that the metric is non-zero on its system before testing whether the
check fires.

## 2.9 Concerns

1. **`branch_tol`'s margin is ≈1.5 decades from one system.** 3.5e-8
   (symmetric, over a 0.10 Bohr⁻¹ sweep) vs 2.7e-7 (broken, ω = 0.60). The
   broken N2⁺ state is not strongly hole-localized in the Mulliken-spin sense,
   so the discriminator is much weaker than the physics picture suggests. A
   second system with equivalent centres would either widen the evidence or
   show 1e-7 is wrong. Until then the flag is a prompt, which the docs say.
2. **The branch check cannot see the case that motivated the issue.** Measured,
   not assumed (2.2): at the N2 onset there is no discontinuity in any
   available observable. Anyone reading "branch check added" as "mixed-branch
   ω* can no longer happen" would be wrong, which is why the module doc, the
   Python docstring and `validation.md` all state the limitation.
3. **The M3/M6 margin is one SCF iteration.** 7→6 and 6→5. It is deterministic
   and reproducible at fixed thresholds, but a change to DIIS or the default
   guess could collapse it and the test would then fail for an unrelated
   reason. The failure would be loud and its message names the cause.
4. **`continuation=True` is untested at validation tier.** The default is off,
   so CI exercises the off path. The on path is covered by the ignored
   measurement tests and the before/after table here, not by a weekly job.
5. **`spin_population_asymmetry`'s equivalence test is geometric, not
   point-group.** Same Z and the same sorted distance multiset to 1e-6 Bohr.
   Sufficient for the symmetric centres this check targets; it would group
   accidentally-equidistant atoms, which costs a false comparison but not a
   wrong energy.

---

# Part 3: defects found in my own work after Part 2 was drafted

These were caught by running the checks, not by reading the code. Each is
recorded because the near-miss is the useful part.

## 3.1 `nearest_evaluated`'s tie case: the TEST was wrong, not the code

The unit test asserted `nearest_evaluated(&[0.3, 0.5], 0.4) == Some(0)` under
the comment "ties resolve to the earlier entry". It failed with `Some(1)`.

There is no tie. In f64, `|0.3 − 0.4| = 0.10000000000000003` and
`|0.5 − 0.4| = 0.09999999999999998`: **0.5 really is closer to 0.4 than 0.3 is**,
so `Some(1)` was correct and my assertion encoded a decimal intuition the
arithmetic does not share. The test now (a) asserts the exact tie it means to
test using binary fractions, with an `assert_eq!` on the two distances FIRST so
the case cannot silently stop being a tie, and (b) keeps the 0.3/0.5 triple
asserting `Some(1)`, documenting that it is ordinary selection by f64 distance.

The code was also changed, for a different and real reason: `Iterator::min_by`
returns the LAST of several equal minima, so on a genuine tie it picked the
later evaluation while the doc promised the earlier one. It is now an explicit
fold keeping the first strict minimum. Both behaviours are now pinned by
mutation (M1 tie-break, M1b nearest-vs-farthest), failing at distinct lines.

## 3.2 A test run that executed ZERO tests and reported `ok`

An intermediate command combined `--test <three binaries> --lib omega_tuning`.
Cargo applied the `omega_tuning` name filter to every binary, so the run
printed three green lines reading `test result: ok. 0 passed; 0 failed;
0 ignored; 0 measured; 14 filtered out`. Three "ok" lines, zero tests executed.
Re-run without the filter: 15 tests, all passing. The `filtered out` count is
the only field that distinguishes the two, which is why the counts in this
report are quoted in full rather than summarised as "passing".

## 3.3 The complexity gate's out-of-scope baseline rewrite

A bare `--update-baseline` wanted to rewrite **104 keys, 38 of them in files
this branch never touches** (`stability.rs` 13, `ferric-cli/config.rs` 7,
`cohsex.rs` 5, `rhf_newton.rs` 4, and others). Committing that would have
silently absorbed unrelated drift.

Handled by diffing the regenerated baseline against HEAD's and merging only
keys under `omega_tuning.rs`, `ferric-python/src/lib.rs` and
`omega_tuning_cation_branch.rs`, with an assertion that nothing outside those
three moved. Verified: 33 keys touched, **0 out of scope**.

Before that, the Python `tune_omega` CC regression (9 → 32) was reduced by a
real extraction rather than absorbed: the per-eval dict building moved to
`omega_eval_dict`, taking it to 16. The branch check similarly moved out of
`tune_omega`'s closure into the named `branch_warning_for`. The remaining
growth is genuine — these functions do more than they did — and is now visible
in the committed baseline diff rather than hidden.

## 3.4 Final check status

| check | result |
|---|---|
| `cargo fmt --all --check` | PASS |
| `cargo clippy -p ferric-scf -p ferric-python --all-targets -- -D warnings` | PASS, no output |
| `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` | PASS |
| `scripts/complexity_gate.py` | PASS — 9015 functions, no regressions vs baseline |
| `omega_tuning.rs` (integration) | `4 passed; 0 failed; 0 ignored` |
| `omega_tuning_cation_branch.rs` | `9 passed; 0 failed; 5 ignored` (the 5 are `#[ignore = "validation: RSH omega tuning"]`, each run explicitly and passing — see below) |
| `omega_tuning_consistent_coulomb.rs` | `2 passed; 0 failed; 0 ignored` |
| `ferric-scf --lib -- omega_tuning` | `1 passed; 0 failed; 0 ignored; 319 filtered out` |
| pytest `-k tune_omega` | `3 passed, 19 deselected` |

The five ignored tests, run with `--ignored`:

| test | result |
|---|---|
| `n2_cation_branch_sweep_measurement` | `1 passed` — produced the 2.2 table |
| `n2_broken_branch_probe_measurement` | `1 passed` — produced the lower-branch table |
| `omega_star_before_and_after_continuation` | `1 passed` — produced the 2.3 table |
| `n2_onset_is_not_visible_without_an_orbital_hessian` | `1 passed` |
| `h2o_tuning_raises_no_branch_flag` | `1 passed` — 10 ω, asymmetry 1e-15, zero flags |

Total: **16 passing non-ignored + 5 passing validation-tier, 0 failures.**

---

# Part 4: acceptance criteria, item by item

| criterion | status |
|---|---|
| Exactness anchor: continuation disabled ⇒ J(ω) bit-identical to today's; `omega_tuning.rs` and `validation_rsh_omega.rs` pass unchanged | **MET.** `continuation_off_is_bit_identical_to_independent_eval_j` asserts every tuner evaluation equals a bare `eval_j` at the same ω, and asserts it is testing the DEFAULT path. Both existing test files pass; `validation_rsh_omega.rs`'s `j_of_omega_*` tests use `eval_j`, which never continues. |
| ...with seeding enabled, agree with today's to ≤1e-9 Ha where the default guess already reaches the stable state | **MET at a tight SCF setting; asserted** (Part 5). At 1e-10 / 1e-7 the max per-eval |ΔJ| is 1.7e-7 Ha; it is ε_HOMO stopping noise that falls about a decade per decade of `density_conv`, reaching 2.65e-10 Ha at 1e-12 / 1e-9, which `omega_star_before_and_after_continuation` now asserts (bar 1e-9). |
| Measured: the N2⁺ onset bracket as ferric sees it vs the generator's `symmetric_cation_probe`; bar set from the measurement | **MET.** 2.2: ferric's J on the symmetric branch matches the probe to 7.4e-8 Ha (ω = 0.53) and 4.7e-7 Ha (ω = 0.56), ⟨S²⟩ to 3.0e-7. ferric sees NO onset in any observable it can compute; PySCF's λ_min brackets it at exactly those two ω. The bar that was set from measurement is `DEFAULT_BRANCH_TOL` (2.4). |
| Negative control: today's `eval_j` at ω = 0.60 for N2 lands on the symmetric saddle / other branch — the new check must flag it, the old code must not | **HALF MET, and the unmet half is the finding.** Confirmed that `eval_j` at ω = 0.60 lands on the symmetric state and that a lower state exists there (4.6e-4 Ha below, reached when seeded). **The new check does NOT flag it, and cannot.** A branch check compares adjacent ω; at ω = 0.60 every adjacent ω is on the same branch, so there is no difference to detect. The quantity that distinguishes a saddle from a minimum at a single ω is the orbital-Hessian eigenvalue, which is item 3's dependency. `n2_onset_is_not_visible_without_an_orbital_hessian` asserts this state of affairs rather than papering over it. |
| Mutation-test: disable the branch check; seed from the wrong ω; ignore the stability verdict — each must fail a test | **MET for the two that exist** (M4 branch check disabled → 2 tests fail; M1/M1b wrong-ω seed → fail at distinct lines), plus M2, M3, M5, M6 beyond what was asked. **The stability-verdict mutation does not exist to run**, because no stability verdict is computed (item 3). |
| `#[ignore = "validation: RSH omega tuning"]` on the PySCF-comparison test | **MET.** All five validation-tier tests in the new file carry it verbatim; `validation_rsh_omega.rs`'s existing annotations are unchanged. |
| Update the "RSH ω tuning" row in `site/src/reference/validation.md`, current facts only | **MET.** Both the capability table and the validation matrix row updated with measured numbers; no "previously", no PR references. `site/src/using/python.md` updated too. |
| fmt / clippy -D warnings / `RUSTDOCFLAGS="-D warnings" cargo doc`; ruff if Python changes | **MET** (3.4). ruff passes via the pre-commit hook on the Python test change. |
| Item 3 (stability check) | **NOT DONE, out of scope by instruction**, dependency named: `stability::ks_reference_is_analysable` refuses ω ≠ 0 with `StabilitySkip::RangeSeparated`. |

## What a reviewer should push back on

1. The ≤1e-9 agreement criterion holds only at a tight SCF setting (1e-12 /
   1e-9: 2.65e-10 Ha); at the validation row's 1e-10 / 1e-7 the difference is
   1.7e-7 Ha. Part 5 shows it is stopping noise, not a different state.
2. The branch check cannot flag the case in the issue's own negative control.
   If that was the primary deliverable rather than continuation, this PR does
   not deliver it and the ω≠0 orbital-Hessian issue has to land first.
3. `DEFAULT_BRANCH_TOL` rests on one system with ≈1.5 decades of margin.

---

# Part 5: the H2O ΔJ question (review follow-up)

Question: is H2O's continuation-on vs -off |ΔJ| SCF convergence noise
(H-conv: shrinks with the threshold) or a different SCF solution (H-state:
plateaus)? One slot job, H2O/def2-SVP, full ω* tune (25 golden-section points)
per arm per setting, `h2o_continuation_dj_vs_scf_convergence_measurement`.

## A defect found first: the comparison had become OFF vs OFF

The first run of the sweep returned |ΔJ| = 0.000e0 at every setting with
IDENTICAL SCF iteration counts on both arms (250/250, 275/275, 325/325). Too
clean. Cause: the test's `cfg()` takes `..Default::default()`, and
`DEFAULT_CONTINUATION` is `false`, so the "on" arm was a second off arm. The
committed `omega_star_before_and_after_continuation` had the same defect from
the moment the default was flipped: the 1.020e-7 figure in 2.3 came from a
binary built while the default was still `true`, and the committed test could
no longer reproduce it.

Fixed: both on arms now set `continuation: true` explicitly and call
`assert_arm_continued`, which requires every evaluation after the first to
record `OmegaSeed::Continued`.

## Measurement (real on arm)

| energy_conv / density_conv | same ω sequence | max per-eval \|ΔJ\| | max \|Δε_HOMO\| | max \|ΔE_cation\| | max \|ΔE_neutral\| | final \|ΔJ\| | cation SCF iters off / on | wall off / on |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1e-9 / 1e-6 | no (ω* 0.504915142 vs 0.504920929) | 3.315e-6 | 3.347e-6 | 6.8e-7 | 7.1e-7 | 3.412e-7 | 225 / 101 | 72 s / 33 s |
| 1e-10 / 1e-7 | yes | 1.712e-7 | 1.713e-7 | 8.2e-13 | 5.8e-12 | **1.020e-7** | 250 / 129 | 82 s / 46 s |
| 1e-11 / 1e-8 | yes | 1.354e-8 | 1.354e-8 | 1.6e-13 | 5.4e-13 | 1.627e-9 | 275 / 160 | 89 s / 47 s |
| 1e-12 / 1e-9 | yes | **2.650e-10** | 2.648e-10 | 5.7e-14 | 1.4e-13 | 4.524e-11 | 325 / 193 | 97 s / 84 s |

`test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 5 filtered out`,
`SLOT_DONE rc=0`.

**H-conv holds.** |ΔJ| falls about a decade per decade of `density_conv` and
is entirely ε_HOMO; both total energies agree to ≤6e-12 Ha from 1e-10 / 1e-7
down. ε_HOMO is first-order in the density residual while the energies are
second-order, which is why the energies agree far better than J. The original
1.020e-7 final |ΔJ| reproduces exactly at 1e-10 / 1e-7. At 1e-9 / 1e-6 the
noise is large enough to flip a golden-section comparison, so the two arms
visit different ω and ω* moves by 5.8e-6 Bohr⁻¹.

## What changed

* `omega_star_before_and_after_continuation` runs H2O at
  `H2O_TIGHT_CONV = (1e-12, 1e-9)` and asserts max per-eval |ΔJ| ≤ 1e-9
  (measured 2.650e-10). H2 and NH3 are unchanged and log only.
* It also asserts the on arm's cation SCF iterations are < 0.8 × the off
  arm's (measured 193/325 = 0.594). The ΔJ bound alone cannot fail if the
  seed is discarded: a discarded seed makes both arms the same computation,
  |ΔJ| = 0, which passes any upper bound. Identical computations give a ratio
  of exactly 1.000 (measured, 325/325 in the defective run), so 0.8 sits
  between the two measured sides.
* Cost: the H2O pair now takes ~180 s against ~130 s at 1e-10 / 1e-7.

## Does this change why `DEFAULT_CONTINUATION` is off?

It changes the stated reason, not the decision. Continuation is not less
accurate: both arms are equally converged to within the threshold, and
continuation needs ~40 % fewer SCF iterations. The remaining reason to keep it
off is narrower: at the thresholds existing callers use, turning it on would
change their J values at the 1e-7 Ha level. The constant's doc now says that.
Whether that is worth keeping off is a separate decision; this PR does not flip it.

## Mutation check

| # | mutation | test | observed |
|---|---|---|---|
| M7 | continuation seed discarded: `cat_seed` → `None` and the neutral `init_guess_density` assignment removed | `omega_star_before_and_after_continuation` | **FAILED** — `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 5 filtered out`, panic at line 451 (the iteration assertion): "continuation took 325 cation SCF iterations vs 325 independent (ratio 1.000; measured 0.594, bar < 0.8)". The ΔJ bound passed under the mutation (max \|ΔJ\| 0.000e0), as argued above. |

Unmutated run of the same test in the same slot job:
`test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 5 filtered out`;
H2 \|ΔJ\| 1.333e-11 (iterations 45/55), H2O 2.650e-10 (193/325), NH3 at
1e-10 / 1e-7 1.405e-7 (131/273, logged only). Source restored after the
mutation; a grep for both mutation strings returns 0.

`cargo test --release -p ferric-cli` (whole suite, same slot job): every
binary `ok`, 0 failed; fmt and clippy `-D warnings` clean.
