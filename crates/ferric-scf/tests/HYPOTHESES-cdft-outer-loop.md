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

## OUTCOME (2026-09-17) — measurement first, then the verdict

Kept short and explicitly separate from the tables above, per the repo's rule
that raw measurements survive re-interpretation while one-line verdicts calcify.

**Measured.** HeNe⁺/def2-SVP, R = 2.0 Å, fragment [He], MINAO start, 99×302
grid, `cdft_lambda_tol = 1e-5`. Targets 2.000 / 1.980 / 1.954484 converged in
10 / 8 / 6 outer iterations; 1.995 and 1.990 hit the 30-iteration cap. All five
ran an identical first four iterations and all five then overshot into a region
near λ ≈ −3.0 where the inner SCF crosses into an over-filled-He state.

Traced Jacobians on the 1.995 run: `−0.137, −0.018, −0.009, −0.031, +0.204,
−148.3, +240.5, −247.5, −0.024`. Two inner solves 0.001 apart in λ differed by
0.826 Ha. From outer 10 onward λ alternated between exactly −3.3266069180979265
and −2.3266069180979265 — a gap of exactly 1.0, the `clamp(−1, 1)` width — with
bit-identical residuals and energies for twenty iterations.

**Adjudication.** F-CYCLE and F-DISCONTINUOUS, jointly and causally: the
discontinuity destroys the Jacobian, and the clamp then converts the resulting
divergence into a period-2 cycle instead of letting it fail fast. F-SLOW is
refuted (the iterates repeat bit-for-bit; no cap helps). F-OVERSHOOT and
F-PLATEAU are refuted by the same repetition.

A-NOTCONV is **refuted for 1.995** (`inner_conv = true` on every traced
iteration, `inner_iters` 9–297) and **partly live for 1.990**, where outers
5/7/8/10 report `inner_conv = false` at the 400-iteration cap — at those λ the
residual fed to the Jacobian is unconverged noise. That is a SECOND, independent
defect; it is recorded here rather than folded into the first, and the fix only
addresses it to the extent of refusing to BRACKET on an unconverged point.
A-GRID is refuted by the control, which resolves N_C to 3.3e-7 on the same grid.

**Fix.** Safeguarded Newton (`rtsafe` structure): maintain a sign-change bracket
on c(λ) from converged inner solves only, and replace the clamped Newton step by
the bisection midpoint whenever it falls outside that bracket. Bisection needs
no derivative, so the discontinuity cannot affect it, and it halves the interval
every step, so the cycle cannot recur. `k = 1` only — a bracket is a scalar
notion and the multi-constraint path is left untouched rather than given a
pretend-bracket.

**A REGRESSION THE ANCHOR CAUGHT, RECORDED BECAUSE IT NEARLY SHIPPED.** The
first version of `Bracket::observe` kept the most-positive λ among `c < 0`, which
silently assumed λ increases toward the root from that side. On this problem it
decreases. The bound froze at λ = 0.0, the bracket never tightened below
[0, −3], and bisection returned −1.5 every time — converting the baselined
hcore run, which converged in 16 iterations, into its own period-2 cycle
(−1.5, −2.5, −1.5, …). The docstring on that version already CLAIMED to be
direction-agnostic while the code was not, which is the repo's "trust code over
doc comments" rule failing exactly as advertised. `hcore_started_path_is_bit_
identical` failed on the first run after the change and is the only reason this
was caught before the result was reported as a success.

**Not established.** That the safeguard is sufficient on any system other than
HeNe⁺, that the bracket always exists (a target outside the reachable range of
c(λ) would never produce a sign change, and the loop would then behave exactly as
before), or that the underlying discontinuity in c(λ) is gone. It is not — the
inner SCF still restarts cold at every λ. The safeguard makes the OUTER loop
robust to that discontinuity; it does not remove it. Continuation (warm-starting
each inner solve from the previous λ) is the fix for the discontinuity itself and
is deliberately NOT taken here, for the reason in the section above.

## Mutation ledger

Every new test was run against a deliberately broken driver. A test never seen
to fail is an assumption. Survivors are listed, not omitted.

| # | Mutation | Killed by | Survivors |
|---|----------|-----------|-----------|
| 1 | `Bracket::safeguard` returns `None` unconditionally — i.e. the whole fix disabled, reverting to the pre-fix algorithm | `..._fires_on_the_measured_limit_cycle`, `..._tightens_when_lambda_runs_negative...`, `..._rejects_nonfinite_and_zero...`, and `every_target_converges_from_a_minao_start` (which failed with exactly the pre-fix island, `[1.995, 1.99]`) | `..._is_inert_while_newton_stays_inside`; `hcore_started_path_reaches_the_same_constrained_solution`; `the_descent_reconverges_and_reaches_the_lower_state` |
| 2 | `observe` keeps the point FURTHEST from the opposite bound (bracket widens instead of tightening) | `bracket_tightens_when_lambda_runs_negative_toward_the_root`, alone | — (5 others green, correctly: only that test probes tightening) |
| 3 | the `converged` filter dropped, so unconverged inner solves may set a bound | **initially NOTHING — survived the entire suite, unit and integration, including the 1.990 sweep it was written for.** Now killed by `bracket_ignores_an_unconverged_inner_solve`, written in response | — |
| 4 | `safeguard` uses `>=`/`<=` instead of strict `>`/`<`, so a proposal exactly ON a bound is kept | **initially NOTHING — survived all 6 tests.** Now killed by `bracket_safeguard_overrides_a_proposal_exactly_on_a_bound`, which was WRITTEN IN RESPONSE to this survival | — |

### Survivors, and why each is not a coverage gap

**Mutation 1 / `..._is_inert_while_newton_stays_inside`.** The test asserts
`safeguard` returns `None` on the healthy cases; a mutation making it return
`None` ALWAYS therefore satisfies it. It is one half of a pair and is only
meaningful beside `..._fires_on_the_measured_limit_cycle`, which the same
mutation kills. Recorded so a future reader does not mistake the pair for
redundant coverage and delete the surviving half.

**Mutation 1 / `hcore_started_path_reaches_the_same_constrained_solution`.**
Correct, and informative. The pre-fix hcore path DID reach that solution — in 16
iterations, through six unconverged inner solves. The test asserts the SOLUTION,
not the route, so disabling the safeguard cannot break it. This is the anchor
behaving exactly as designed: it is a guard against the answer moving, and it
must not double as a guard against the fix being removed.

**Mutation 1 / `the_descent_reconverges_and_reaches_the_lower_state`.** The most
interesting survivor, and it MOVES A CLAIM. The stability descent reaches the
lower state (−130.4267, 0.667 eV below the saddle) even with the safeguard
disabled. So the descent's "step 0.4/0.8/1.2 did not re-converge" warnings, which
motivated including the descent in this lane, are NOT explained by the
outer-loop defect fixed here — at least not at this config. The honest statement
is that the descent works before and after, and that this test's value is
regression coverage against the swallowed-warning failure mode, NOT evidence
that the safeguard repaired it. Any claim that this fix repairs the descent is
unsupported and is not made.

### Two guards that were UNREACHABLE when written

Mutations 3 and 4 each survived the whole suite on the first pass. Both were
guards added on reasoning rather than on evidence, and both are exactly the
failure mode this file's own driver already documents twice (`constraint_offset`
and `accepts_candidate` were extracted from `stability_descent` for the same
reason: deleting them left the cDFT suite green).

The response was the repo's standard one — make the decision testable and pin
it — not to delete the guards. Being unreachable on the paths that converge does
not make a guard wrong; the `converged` filter earns its place on the runs that
FAIL, where four inner solves hit the iteration cap in a single outer loop. It
makes the guard an ASSUMPTION until tested. Both are now tested, and the fact
that they were not is recorded here rather than tidied away.

### Reachability of the new assertions

Each new test was also checked for a REACHABLE pass condition, not just a
failing mutation: `every_target_converges_from_a_minao_start` asserts
`converged + failed == targets.len()` before reading its verdict, and the
`Bracket` unit tests assert both the `Some` and the `None` branch of
`safeguard`, so neither is a gate over an empty set.

## Pass-condition reachability

Every assertion added here must be checked for reachability the way
`1a1eeddd` had to retrofit onto the H-GUESS discriminator: a gate over an empty
set returns arithmetic, not measurement. Each new test therefore asserts
`n > 0` on whatever it maximizes over, before reading the verdict.
