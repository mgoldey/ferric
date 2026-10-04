# Hypotheses — issue #282, the HeNe+ `state A (N_He=1)` start at ERI precision 1e-20

Written BEFORE any measurement on this branch. Date: 2026-10-04.
Branch `fix/cdft-hene-branch-guard`, rebased onto `origin/main` 0b4fa7dc.

## The failure, as the issue states it

`crates/ferric-scf/tests/cdft_guess_catalogue_precision.rs` runs eight starts
through `solve_cdft_uhf_seeded` on HeNe+/def2-SVP, R = 2.0 A, fragment [He],
`SpinChannel::Total`, target N_He = 2, `cdft_stability_descent: false`,
`max_iter: 400`, `cdft_max_outer: 40`, `cdft_lambda_tol: 1e-5`.

At ERI precision 1e-14 all eight converge; `state A (N_He=1)` reaches LOWER in
9 outer iterations. At 1e-20 that one start "does not converge in 40" and is
listed in `allowed_failures`.

The mechanism the driver's `ScalarStepper` docstring records:

1. the first 23 inner solves hit the 400-iteration cap before any converges, so
   guard 1 (backtrack to the last converged lambda) has no anchor;
2. then a clamp-width period-2 cycle on a *converged* branch with dc/dlambda > 0
   and no sign change: lambda alternates -0.2324 (c = -0.961, J = +0.199) and
   +0.7676 (c = -0.992, J = -0.011). Both probes are on their main point's
   branch and both residuals are negative, so neither guard 2 nor `Bracket`
   fires.

## What two sibling issues already changed about the starting point

These are NOT re-derived here; they are the reason the hypotheses below are
written the way they are.

**#281 (merged as PR #309).** The "N = 1.99936"-style numbers quoted in cDFT
issues were *transient DIIS iterates of unfinished inner solves*, not values of
c(lambda). At lambda differing in the 12th decimal, one inner solve converged in
77 iterations and another hit a 100-iteration cap at N = 1.000018 — same basin,
same energy to 1e-11. LESSON: before theorising about branches, read the inner
**iteration counts**, not just the `converged` flag.

**#283 (merged as 0b4fa7dc).** On the He2+ lambda = 0 failure the issue blamed
an FD-probe basin flip and a +/-490 Jacobian. Re-traced: guard 2 (`hf_mismatch`)
*already rejects* that Jacobian and the loop takes a `SignStep`. The real cause
was the **outer budget** — roots at lambda 1.55-2.46, Newton clamped to
`MAX_STEP = 1`, outer capped at 8. Verdict: "it was not lost, it was
interrupted." `run_inner`'s seeding is untouched by that work.

Both issues' stated diagnoses were wrong. #282's may be too.

## THE OPERATOR AND THE THEORY (stated first, so it cannot be retrofitted)

Along the **lowest** solution branch, c(lambda) is NON-INCREASING: V(lambda) =
min_rho (E[rho] + lambda*c[rho]) is a pointwise minimum of affine functions of
lambda, hence CONCAVE, and dV/dlambda = c (Hellmann-Feynman). A concave
function has a non-increasing derivative, so dc/dlambda <= 0 on the lowest
solution.

Therefore: **a CONVERGED branch on which dc/dlambda > 0 is not the lowest
solution at that lambda.** That is a signal about which state the inner solve
is sitting on, not merely a numerical stall. This is the issue's own strongest
lead and the thing H-PHYSICS below predicts.

Caveat that must be respected while using it: the sign of dc/dlambda is read
from a FINITE DIFFERENCE over `FD_STEP = 1e-3` of two separate inner SCF solves,
each re-seeded from the SAME fixed guess. If either solve is unconverged, the
"slope" is not a derivative of anything — which is exactly #281's lesson, and
exactly why guard 2's `hf_mismatch` exists.

## The competing hypotheses

**H-PHYSICS: the state-A orbitals seed an EXCITED/UPPER branch whose c(lambda)
has genuinely positive slope, and the cycle is the loop correctly failing to
find a root that does not exist on that branch.**

Observables it predicts:
* at both cycle points the inner SCF is GENUINELY CONVERGED — `scf.converged`
  true AND `scf.iterations` comfortably below the 400 cap (the #281 check);
* the positive slope is ROBUST: re-measured with a different `FD_STEP`, or by a
  one-sided difference the other way, it stays positive and O(0.2), not a
  noise-level number whose sign flips;
* `hf_mismatch` at the +J point is SMALL (same branch) — i.e. guard 2 has no
  grounds to reject, which is consistent with the issue's own report;
* the constrained solution at those lambdas is a SADDLE of its own
  lambda-augmented functional (lambda_min(H_aug) < 0 by a margin above the
  eigensolver noise floor), because an upper branch should be internally
  unstable. If so, the physical fix is a stability descent on the constrained
  problem, which `StabilitySkip::FockModified` currently refuses, and the
  honest outcome may be "this needs the constrained-Fock Hessian, here is the
  measurement" rather than a new guard.

**H-ARTIFACT-A: the cycle points are NOT converged (the #281 failure, again).**

Observables: at one or both cycle points `scf.iterations` is at or near the
400 cap, or `converged` is false and the loop reached `step` only because
`trusts` short-circuits on an unconverged MAIN point. Then "+J on a converged
branch" is a misreading of the trace, the residuals are transient DIIS
iterates, and the fix is about the inner solve's budget/seeding — NOT a new
outer guard. The #283 shape ("interrupted, not lost") would repeat.

**H-ARTIFACT-B: the clamp alone manufactures the cycle (the issue's own
artifact hypothesis).** With `MAX_STEP = 1` and |c| ~ 0.96-0.99, the raw Newton
step c/J is -4.8 at J = +0.199 and -90 at J = -0.011; BOTH clamp to exactly
one clamp width, in OPPOSITE directions because J changes sign. lambda then
alternates with period 2 and gap exactly 1.0 — the same arithmetic signature as
the pre-`Bracket` cycle this driver already fixed once (gap exactly the clamp
width, bit-identical residuals).

Observables that distinguish B from PHYSICS: the gap between the two cycle
lambdas is EXACTLY `MAX_STEP` to the bit, and the residuals/energies at each
point repeat BIT-IDENTICALLY across periods. Under PHYSICS alone there is no
reason for the gap to be exactly 1.0.

NOTE B and PHYSICS are NOT mutually exclusive, and saying so now is the point of
writing this down: a positive-slope upper branch can be the reason the loop is
stuck in a region with no root, while the clamp is the reason it is stuck
PERIODICALLY rather than wandering. If both hold, a guard that only breaks the
period (e.g. perturbing the step) would move the loop off the cycle without
ever reaching a root — which would look like progress and be none.

**H-ARTIFACT-C: the 23 capped inner solves are the whole story.** If guard 1
had an anchor it would backtrack and the cycle would never be entered. Under C,
the fix is "give guard 1 something to back off to when nothing has converged
yet" (the issue's task 2, second half) and the +J cycle never arises.
Observable: the loop's first converged lambda, and whether the cycle lambdas are
reachable at all once a near-lambda-0 anchor exists.

## Exactness anchor (MUST pass before any verdict is believed)

Non-negotiable, before any sweep:
* `guards_are_inert_on_a_smooth_residual` must still pass BIT-IDENTICALLY. Any
  new guard must be inert on a smooth, monotone residual — a guard that fires
  there is a regression, not a fix.
* every NEW synthetic oracle added to `cdft_driver.rs::tests` must have a
  trivial-limit companion: with the new guard disabled the oracle must exhibit
  the cycle (negative control), and with it enabled it must reach the root.
* the other seven catalogue starts must be unchanged at BOTH precisions — same
  state, same lambda, same E, same outer iteration count. The table is recorded
  before and after.

## Artifact-vs-physics predictions that would make the experiment VACUOUS

Stated so the experiment can be redesigned rather than believed:
* if the slope sign were read from a pair in which either solve is unconverged,
  H-PHYSICS and H-ARTIFACT-A predict the SAME trace line. They are distinguished
  ONLY by `scf.iterations`, so the trace MUST print it (it already does) and the
  measurement MUST read it.
* a new guard whose GO condition is "dc/dlambda > 0 and no sign change after k
  iterations" fires on the cycle under BOTH H-PHYSICS and H-ARTIFACT-B. So
  "the guard makes the start converge" does NOT by itself support H-PHYSICS,
  and must not be written up as if it did. What the start CONVERGES TO (UPPER
  vs LOWER) is the discriminating observable: under H-PHYSICS the seeded branch
  is an upper one and a correct reaction leaves the upper state or descends to
  LOWER; at 1e-14 this same start reaches LOWER in 9 outer iterations, which is
  the number the 1e-20 fix should be compared against.

## Mutation ledger (pre-registered — each MUST fail)

For whatever guard lands:
* the iteration threshold k off by one in BOTH directions;
* the slope-sign test flipped (`> 0` -> `< 0`);
* the "no sign change" condition deleted (so it fires even when a bracket
  exists);
* the guard's reaction replaced by the plain step (so the guard is detected but
  inert).

Each must turn at least one named test RED. The run must be checked for
`passed`/`failed`/**`ignored`** counts: two mutation attempts in a sibling task
this session were worthless in ways that looked like success — one did not
compile, one reported `0 passed; 5 ignored` because the tests are
`#[ignore]`-gated and the run lacked `-- --ignored`.

## What a good NEGATIVE outcome looks like

If the evidence says the loop needs a stability descent on the CONSTRAINED
problem — which `StabilitySkip::FockModified` currently refuses — the right
report is that measurement, with lambda_min(H_aug) at the cycle points and the
noise floor, and NOT a guard forced in to make the test green. A well-evidenced
"this needs the constrained-Fock Hessian" closes the issue honestly.

---

## MEASUREMENT 1 (2026-10-04, partial): H-ARTIFACT-A CONFIRMED at outer 1-11

Appended, NOT edited into the pre-registration above, per the protocol.

PROVENANCE CAVEAT, stated because it affects nothing but must not be hidden:
this trace comes from a run that was CANCELLED at outer 11 of 40 (I killed it
while reorganising the slot queue, mistakenly believing it had produced no
data). The lines below are the run's own stderr, unmodified, and the run was
mid-measurement when killed; the cut is a TRUNCATION, not a corruption. The
remaining iterations are being re-measured.

Release build, ERI precision 1e-20, HeNe+/def2-SVP R = 2.0 A, state-A seed,
target N_He = 2, `max_iter: 400`, `cdft_max_outer: 40`.

### The state-A REFERENCE solve (target 1.0) is healthy

| outer | lam | N_C | resid | inner_conv | inner_iters | jac | hf_mismatch |
|---|---|---|---|---|---|---|---|
| 1 | +0.000000 | 1.953538 | +9.535e-1 | true | 11 | -1.615e-2 | 9.7e-12 |
| 2 | +1.000000 | 1.005901 | +5.901e-3 | true | 22 | -8.915e-3 | 2.2e-12 |
| 3 | +1.661955 | 0.999635 | -3.653e-4 | true | 10 | -1.196e-2 | 1.1e-10 |
| 4 | +1.631409 | 0.999993 | -6.793e-6 | true | 10 | — | — |

Converged in 4 outer iterations to E = -130.3618594952, lam = +1.63140887 —
which matches the state-A diabat recorded in HYPOTHESES-constrained-stability.md
(lambda = +1.631409, E = -130.36185950). Every jac is NEGATIVE, consistent with
concavity. This is the seed, and it is sound.

### The state-A START at target 2.0: EVERY solve is iteration-capped

| outer | lam | N_C (main) | resid | inner_conv | inner_iters | jac | hf_mismatch | step |
|---|---|---|---|---|---|---|---|---|
| 1 | +0.000000 | 1.062626 | -9.374e-1 | **false** | **400** | +3.109e2 | 2.28e-2 | Newton |
| 2 | +0.003015 | 1.103962 | -8.960e-1 | **false** | **400** | +7.374e1 | 4.24e-3 | Newton |
| 3 | +0.015167 | 1.161356 | -8.386e-1 | **false** | **400** | -7.275e1 | 1.77e-2 | Newton |
| 4 | +0.003640 | 1.138120 | -8.619e-1 | **false** | **400** | +4.063e1 | 1.17e-2 | Newton |
| 5 | +0.024854 | 1.160649 | -8.394e-1 | **false** | **400** | +3.036e1 | 5.79e-3 | Newton |
| 6 | +0.052500 | 1.283740 | -7.163e-1 | **false** | **400** | -2.049e2 | 1.53e-2 | Newton |
| 7 | +0.049004 | 1.093312 | -9.067e-1 | **false** | **400** | +2.286e1 | 1.85e-3 | Newton |
| 8 | +0.088667 | 1.200925 | -7.991e-1 | **false** | **400** | -6.354e1 | 5.24e-4 | Newton |
| 9 | +0.076090 | 1.164410 | -8.356e-1 | **false** | **400** | +9.076e1 | 1.35e-2 | Newton |
| 10 | +0.085297 | 1.201393 | -7.986e-1 | **false** | **400** | +4.805e1 | 2.60e-3 | Newton |
| 11 | +0.101917 | 1.167771 | -8.322e-1 | **false** | **400** | — | — | — |

(Every PROBE is also `inner_conv=false inner_iters=400`.)

### What this refutes

1. **The issue's mechanism (2) is NOT what happens at these iterations.** It
   describes "a clamp-width period-2 cycle on a CONVERGED branch with
   dc/dlambda > 0". At outer 1-11 nothing is converged, there is no period-2
   cycle (lambda random-walks in [0.003, 0.102], never revisiting a point), and
   the Jacobians sign-flip across +310.9/-204.9. **H-ARTIFACT-A is confirmed**
   for this range; H-PHYSICS has no support here. The guard I drafted from the
   concavity argument ("reject jac > 0 on a CONVERGED pair") would have been
   INERT on every line above, because `s.converged` is false throughout.

2. **Guard 2 is being BYPASSED, and that is the live defect.** `hf_mismatch`
   runs 5.2e-4 … 2.3e-2, i.e. **100x to 23,000x above HF_MISMATCH_TOL = 1e-6**,
   and the probe never converges — so `trusts` should reject every pair. It
   accepts all of them (`step=Newton` on every line) because of its own
   short-circuit:

   ```rust
   !self.guards || !s.converged || (p.converged && … && hf_mismatch(..) <= HF_MISMATCH_TOL)
   ```

   `!s.converged` returns TRUSTED unconditionally. The code comment justifies
   this as "no better model than this one" — but a Jacobian of +310 formed from
   two 400-iteration-capped DIIS snapshots is not a model of anything, and the
   loop spends its entire budget on it. This is the same CLASS of finding as
   #283's: the guard that should fire already exists, and the failure is in
   what reaches it.

3. **Guard 1 cannot engage, for a derived reason.** `backtrack` requires
   `last_good`, which `record` sets only on a converged point. With no inner
   solve ever converging there is no anchor, exactly as the issue says — but the
   consequence is not "the plain step is taken", it is "an untrusted Jacobian is
   taken as TRUSTED", which is worse and is what (2) describes.

### What is still open

* outer 12-40: whether any solve converges later, and whether the documented
  lam = -0.2324 / +0.7676 cycle with jac = +0.199 appears AFTER one does. The
  issue's numbers must come from somewhere; this trace does not reach them.
* the direct c(lambda) scan (no outer loop): where the root actually is from
  this seed, and whether the FD slope sign at the documented cycle points is
  robust across h = 1e-4/1e-3/1e-2.
* WHY the inner solve cannot converge from the state-A orbitals at target 2.0
  while the same guess converges in 10-22 iterations at target 1.0. At 1e-14
  this start converges in 9 outer iterations, so the inner solve CAN converge
  from it; what the extra ERI precision changes is the open question.

### Provisional reading (dated, explicitly not a verdict)

The fix suggested by the data so far is NOT a new outer guard on the slope. It
is that an unconverged main point must not confer trust on its probe — and then
"no converged point yet" needs a defined behaviour (the issue's task 2, second
half), because with the short-circuit closed there is no Jacobian at all on
these iterations. Whether that is enough to reach a root, or whether the inner
solve's inability to converge at this target is the real blocker, is NOT yet
established and must not be asserted until the remaining measurements land.

---

## MEASUREMENT 2 (2026-10-04, COMPLETE): both mechanisms are real, in different phases

Full run, uncancelled: `rc=0`, 3 tests passed, 36.6 s, release, quiet box
(box held the slot; #283's suite and #288's job had finished).

**This SUPERSEDES measurement 1's provisional reading, which generalised from
outer 1-11 to the whole run. Measurement 1 is correct for the range it saw and
wrong as a verdict; it was committed as provisional for exactly this reason.**

### The run has TWO phases, and the issue describes both

Exactly **23 capped main solves, then 17 converged ones** — the issue's "first
23 inner solves hit their 400-iteration cap" is confirmed to the integer.

Phase 1 (outer 1-23): every main AND probe `inner_conv=false inner_iters=400`.
Jacobians sign-flip across +310.9/-204.9; `hf_mismatch` 5.2e-4 … 2.3e-2, i.e.
100x-23000x ABOVE tol. `trusts` accepts all of them via its `!s.converged`
short-circuit, so a Jacobian from two capped DIIS snapshots drives the step.
lambda random-walks in [-0.004, +0.102]. H-ARTIFACT-A CONFIRMED here.

Phase 2 (outer 24-40): a bit-identical period-2 cycle on CONVERGED solves.

| outer | lam | N_C | c | E | conv | iters | jac | hf_mismatch |
|---|---|---|---|---|---|---|---|---|
| 24 | -0.232435480779 | 1.039258195 | -9.607418e-1 | -130.3582888372 | **true** | **31** | **+1.990141e-1** | **2.096e-10** |
| 25 | +0.767564519221 | 1.008189106 | -9.918109e-1 | -130.3717295214 | **true** | **8** | -1.129936e-2 | 1.037e-12 |
| 26 | -0.232435480779 | 1.039258207 | -9.607418e-1 | -130.3582888344 | true | 31 | +1.990021e-1 | 2.156e-10 |
| 27 | +0.767564519221 | 1.008189106 | -9.918109e-1 | -130.3717295214 | true | 8 | -1.129936e-2 | 1.037e-12 |
| … repeats to outer 40 | | | | | | | | |

gap = 0.767564519221 - (-0.232435480779) = **exactly 1.000000000000 = MAX_STEP**.
Only THREE distinct converged states occur after outer 24 (two of them the same
lambda differing in E's 11th decimal, 2.8e-9 — inner-SCF exit noise).

**The cycle points are GENUINELY CONVERGED**: 31/26 and 8/8 iterations against a
400 cap, and `hf_mismatch` 2.1e-10 / 1.0e-12 is far BELOW `HF_MISMATCH_TOL`
= 1e-6, so guard 2 correctly has no grounds to fire. H-PHYSICS CONFIRMED here.
Per #281's lesson the iteration counts were read, and they do NOT excuse the
slope this time.

### The slope sign is ROBUST (my FD-noise hypothesis is REFUTED)

| lam | h = 1e-4 | h = 1e-3 | h = 1e-2 | probes converged |
|---|---|---|---|---|
| -0.2324 | **+0.200683** | **+0.199158** | **+0.209687** | 31/31, 31/26, 31/37 |
| +0.7676 | -0.011307 | -0.011299 | -0.011216 | 8/8, 8/8, 8/8 |

Stable to ~5% over two decades of h, every probe converged. I had argued from
`density_conv = 1e-6` that FD noise could reach ~1e-1 and flip the sign; that
prediction is REFUTED. The positive slope is a property of the branch.

### The decisive new fact: TWO BRANCHES, and the root is on the OTHER one

Direct c(lambda) scan, one inner solve per lambda from the state-A seed, NO
outer loop involved (independent construction, not a re-reading of the trace):

| lam | c | E | conv | iters | branch |
|---|---|---|---|---|---|
| -2.753700 | +0.860747625 | -128.127747908 | true | 24 | (collapsed) |
| **-2.439010** | **-0.000000122** | **-130.426707290** | true | 22 | **LOWER — THE ROOT** |
| -2.000000 | -0.020349050 | -130.472912532 | true | 34 | lower |
| -1.500000 | -0.028646434 | -130.487652444 | true | 11 | lower |
| -1.000000 | -0.034238265 | -130.494682023 | true | 11 | lower |
| -0.500000 | -0.039686606 | -130.498744411 | true | 11 | lower |
| **-0.232400** | **-0.960734932** | -130.358287240 | true | 31 | **state-A branch** |
| +0.000000 | -0.937374421 | -130.382251622 | **false** | **400** | (no solution) |
| +0.267600 | -0.975638511 | -130.378575579 | true | 11 | state-A |
| +0.500000 | -0.987559970 | -130.374352635 | true | 9 | state-A |
| **+0.767600** | **-0.991811295** | -130.371729213 | true | 8 | **state-A branch** |
| +1.000000 | -0.994098880 | -130.369717946 | true | 7 | state-A |

The state-A branch carries c = -0.96 … -0.99 across its whole extent: **it has
NO root**. The lower branch carries c = -0.020 … -0.040 and its root is at
lambda = -2.43901, c = -1.2e-7, E = **-130.426707290**.

### Which state, and why (acceptance criterion)

**LOWER: E = -130.4267065, lambda = -2.43901.** The scan hits it at
E = -130.426707290, lambda = -2.439010 — agreeing with the documented LOWER to
**8e-7 Ha** and **1e-8 in lambda**, inside the test's 1e-5 / 1e-3 bars. It is
also the state this start reaches at precision 1e-14 in 9 outer iterations, so a
fix is restoring the 1e-14 answer, not inventing one.

WHY the loop cannot get there unaided: the cycle sits on a branch with no root,
both probes are on that branch, both residuals are negative, so `Bracket` never
gets a two-sided bracket (the trace shows `bracket=[-0.232435, NaN]` — one-sided
forever) and guard 2 has nothing to object to. The ONLY local signal that the
branch is wrong is **dc/dlambda = +0.199 > 0**, which by concavity of
V(lambda) = min_rho(E + lambda*c) cannot happen on the lowest solution. The
issue's theoretical lead is therefore CORRECT and is the only available signal.

### Both of the issue's mechanisms are real; neither alone explains the failure

A guard on the slope alone would not have helped in phase 1 (nothing is
converged there, so it is inert — measurement 1 established that). Closing the
`!s.converged` short-circuit alone would not help in phase 2 (everything is
converged there and `hf_mismatch` is legitimately tiny). **The failure needs
both fixes**, and that is the finding that neither the issue nor measurement 1
states on its own.

### Note on StabilitySkip::FockModified

No stability descent is needed for this row. `cdft_driver::augmented_instability`
already forms the correct operator (the lambda-augmented Hessian, valid because
W is density-independent so it drops out of dF/dD), and the catalogue runs with
`cdft_stability_descent: false` anyway. `StabilitySkip::FockModified` guards the
BARE UHF stability call in `uhf.rs`, which is a different question. The root is
reachable by the outer loop once it stops stepping along a rootless branch, so
this does not need the constrained-Fock descent.
