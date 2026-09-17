# Pre-registered hypotheses — the cDFT λ-Newton outer loop from a MINAO start

**Registered BEFORE any number in `cdft_outer_loop.rs` was measured.** Written
because the repo's protocol requires the artifact hypothesis to sit next to the
physics hypothesis, and requires an exactness anchor to exist before a sweep runs.

## The observation being pursued

`1a1eeddd` recorded, and deliberately did not tune away:

> started from MINAO instead, the constrained lambda-Newton loop stops
> converging within its 30-iteration cap at the intermediate targets 1.990 and
> 1.995

The `max_outer = 30` cap in `cdft_driver.rs` was NOT widened, on the grounds
that widening it would hide the finding. This file is the cDFT lane picking it up.

### Scope correction (2026-09-17) — the integer target is NOT affected

A first framing of this lane also attributed the INTEGER target (N_He = 2.000)
to the same failure. That is FALSE and is recorded here rather than quietly
dropped, because it changes the experiment: from a MINAO start the integer
target converges in 10 of 30 outer iterations, to the same state hcore reaches
(agreeing to ~3e-7 Ha) and in FEWER iterations than hcore's 23.

That correction is what makes this lane tractable, because it supplies a
CONTROL. The three genuinely open cases —

  1. intermediate targets 1.990 / 1.995 from a MINAO start,
  2. the `state A (N_He = 1)` catalogue guess, at both the integer and the
     natural (1.954484) target,
  3. every `DESCENT_STEPS` re-convergence inside `stability_descent`, whose
     failures are swallowed as warnings —

share molecule, basis, solver and knobs with a run that DOES converge. Diffing a
converging trace against a non-converging one controls for everything except
what differs, which is a far stronger experiment than studying a failure alone.
Any diagnosis below that does not survive that diff is not established.

## Candidate failure modes (mutually exclusive; the trace decides)

* **F-SLOW** — the Newton iterate converges monotonically but needs > 30 steps.
  *Predicts:* `max|resid|` strictly decreasing at every outer step, λ settling,
  and the residual within an order of magnitude of `cdft_lambda_tol` by step 30.
* **F-CYCLE** — the iterate limit-cycles. *Predicts:* λ and `max|resid|` revisit
  the same values with period ≥ 2 and no downward trend.
* **F-OVERSHOOT** — the step is too large and the iterate diverges.
  *Predicts:* `|λ|` growing without bound, residual growing.
* **F-PLATEAU** — the residual stalls at a nonzero value with λ still moving.
  *Predicts:* `max|resid|` flat while `Δλ` stays O(clamp).
* **F-DISCONTINUOUS** — the inner SCF, restarted from the SAME fixed guess at
  every λ, falls into DIFFERENT basins at nearby λ. Then c(λ) has a jump and no
  Newton variant converges, because the function being rooted is not continuous.
  *Predicts:* a λ interval across which c(λ) jumps by ≫ tol for a λ change of
  O(fd), and/or the FD Jacobian changing sign or magnitude wildly between
  consecutive outer steps; and the inner SCF energy jumping discontinuously in λ.

## The artifact hypotheses (what a broken EXPERIMENT would look like)

Stated so a construction error is not mistaken for physics:

* **A-NOTCONV** — the *inner* SCF is not converging at all (hitting `max_iter`),
  so the residual is noise from an unconverged density rather than a property of
  c(λ). *Predicts:* `scf.converged == false` on the traced iterations.
  Distinguishable from every F above by one boolean, which the trace records.
* **A-GRID** — the Becke weight quadrature is too coarse to resolve
  `cdft_lambda_tol = 1e-5`, so the target is unreachable by construction.
  *Predicts:* the residual floors at the quadrature error regardless of λ, and
  the floor MOVES when the grid is refined. The lane already uses the 99×302
  grid for this reason; the trace records whether the floor sits at ~1e-4
  (the 75×110 plateau) or far below.
* **A-TESTRIG** — my instrumented copy of the loop differs from the driver's.
  Guarded by instrumenting the DRIVER ITSELF behind an env var rather than
  re-implementing the loop in the test.

**F-DISCONTINUOUS vs A-NOTCONV are the pair most at risk of being confused**, and
they have the same surface symptom (erratic residual). The discriminator is
`scf.converged` plus the inner iteration count: a converged inner SCF with a
jumping energy is F-DISCONTINUOUS; an unconverged one is A-NOTCONV. The trace
records both, so the experiment can tell them apart.

## Exactness anchor (written and passing BEFORE the algorithm is touched)

`hcore_started_path_is_bit_identical` — the hcore-started HeNe⁺ constrained solve
at the integer target must reproduce, to the last bit, the energy / λ / population
it produced before any change to the driver. A large existing suite is baselined
against that path; any fix that moves it is out of scope by construction.

Recorded baseline (captured from the unmodified driver on this branch, see the
test's own doc comment for the literals).

## A design constraint the anchor imposes, stated before the fix is chosen

The exactness anchor and the most natural fix are in direct tension, and that is
worth writing down BEFORE the trace arrives so the fix is not quietly chosen to
suit whatever the trace shows.

`run_inner` passes the SAME fixed `guess` to every inner SCF. It never warm-starts
from the previous λ's converged orbitals. If the trace shows F-DISCONTINUOUS, the
principled fix is continuation — warm-start each inner solve from the previous
λ's orbitals, so the loop FOLLOWS one branch of c(λ) instead of re-running a
basin lottery at every λ.

But continuation changes the iterate on EVERY path, including the hcore path the
anchor pins and the suite is baselined against. So a blanket warm-start is
excluded by the anchor, not by taste.

That leaves two admissible shapes, and the choice between them must be made on
the evidence, not on convenience:

* **Conditional** — keep the existing cold-start iterate as the primary, and
  engage continuation only as a FALLBACK once the primary has demonstrably
  failed (i.e. after the cap, or on a detected stall). Bit-identical on every
  path that already converges, by construction.
* **Blanket + re-baseline** — warm-start always, and re-record every baseline in
  the suite. Out of scope here: this lane does not own those files, and
  re-baselining a suite to accommodate a fix is exactly the silent
  experiment-swap `1a1eeddd` refused.

The conditional shape is therefore the only one available to this lane. Its
weakness must be stated too: a fallback that only runs after 30 wasted
iterations is slow, and a fallback is not a proof that the primary is sound. If
the trace shows the primary iterate is structurally wrong (not merely slow), the
honest report is that plain cold-start Newton is the defect and the fallback is a
mitigation, NOT a fix.

## Pass-condition reachability

Every assertion added here must be checked for reachability the way
`1a1eeddd` had to retrofit onto the H-GUESS discriminator: a gate over an empty
set returns arithmetic, not measurement. Each new test therefore asserts
`n > 0` on whatever it maximizes over, before reading the verdict.
