# Issue #283 — the cDFT λ-Newton loop from λ = 0 on He₂⁺

**Verdict: no solver change was needed.** The λ = 0 start finds the correct
He₂⁺ diabat on all six points, to ≤ 3.1e-10 in λ and ≤ 1.1e-10 Ha in E against
the NWChem-started root. What blocked it was the validation row's own
`cdft_max_outer = 8` — a cost cap chosen for a start that is already AT the
root — not the inner-SCF basin flip the issue diagnosed.

`run_inner`'s seeding is **UNCHANGED**. Zero executable lines were modified in
`src/`; the diff there is doc comments only. This matters because the sibling
task on #282 is working the same file (`cdft_driver.rs`).

Commits: `678bfe43` (hypotheses, written before measuring) and `edafcf25` (fix).

---

## 1. Hypotheses, as committed before any measurement

Committed in `crates/ferric-scf/tests/HYPOTHESES-cdft-lambda0-he2.md` at
`678bfe43`, 36 min before the first sweep. Both were stated with a *shared*
discriminating quantity so the sweep could not be read as confirming whichever
one I preferred.

**H-PHYSICS** — the branches are disjoint, so re-seeding cannot help. λ = 0 is
the unconstrained delocalized state (N = 1.5) and the target N = 1.0 is on a
localized branch behind a barrier. Prediction: the λ = 0 probe is same-basin
(`hf_mismatch` ≤ 1e-6), the measured slope is the true plateau slope, and
candidate fix (c) leaves ≥ 4 of 6 points failing because the previous λ's
orbitals are also delocalized.

**H-ARTIFACT** — the ±490 Jacobian is manufactured by the shared seed. The
inner solve at λ = 0 sits near a basin boundary, so a 1e-3 λ change with the
same guess tips DIIS into the localized basin. Prediction: `hf_mismatch` at
λ = 0 is ≥ 5.5e-5 (the measured cross-basin band), guard 2 fires, the loop
takes a SignStep; fix (c) makes the pair same-basin but *may not fix the
failure*, because the plateau slope is the same small number either way.

The hypotheses file states explicitly: *"They are NOT mutually exclusive: the
artifact can be real AND insufficient. That is the outcome I consider most
likely a priori."* It also named the falsifier that decided the matter — an
N(λ) scan showing whether the branch reachable from λ = 0 contains the root.

**Outcome: H-PHYSICS is REFUTED. H-ARTIFACT is CONFIRMED but IRRELEVANT** — it
is real on 4 of 6 points, guard 2 (PR #248) already handles it, and it is not
what causes the failure. A third mechanism neither hypothesis named — the outer
iteration budget — is the whole cause. The hypotheses were useful by being
wrong in a way I could see.

---

## 2. The λ = 0 trace, all six points (`FERRIC_CDFT_TRACE=1`)

Release, 99×302 grid, level shift 0.3, inner cap 150, `cdft_lambda_tol` 1e-10,
`cdft_max_outer` 8, ERI precision at the default. Measured 2026-10-04.

### 2a. The outer-1 probe at λ = 0 — the hypotheses' discriminating quantity

| point | N at λ=0 | jac | `hf_mismatch` | probe conv | step taken |
|---|---:|---:|---:|---|---|
| def2-SVP 2.50 Å | 1.500000018 | −1.035 | 3.342e-13 | yes | Newton |
| def2-SVP 3.00 Å | 1.499999782 | +5.965 | 2.673e-5 | **no** | SignStep |
| def2-SVP 3.50 Å | 1.499999... | +53.65 | 1.818e-3 | **no** | SignStep |
| aug-cc-pVDZ 2.50 Å | 1.500000017 | −1.047 | 4.146e-13 | yes | Newton |
| aug-cc-pVDZ 3.00 Å | 1.499999784 | −163.97 | 1.632e-2 | **no** | SignStep |
| aug-cc-pVDZ 3.50 Å | 1.499999841 | −7.779 | 3.594e-5 | **no** | SignStep |

Reading: on 4 of 6 points the λ = 0 probe IS compromised, with `hf_mismatch` in
the measured cross-basin band (≥ 5.5e-5) and an unconverged probe. Guard 2
rejects the Jacobian on all four and the loop takes a `SignStep` of one trust
radius in the direction that lowers |c| — i.e. **the ±490-class Jacobian is
never used**. H-ARTIFACT's mechanism is real; its consequence was already
fixed by PR #248, which postdates the issue's diagnosis.

### 2b. Per-outer trajectory at the 8-iteration cap

def2-SVP 2.50 Å, the cleanest case (every probe trusted, plain Newton):

| outer | λ | N | c | jac | `hf_mismatch` | step |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.000000000 | 1.500000018 | +5.000e-1 | −1.035e0 | 3.3e-13 | Newton |
| 2 | 0.482936716 | 1.008216788 | +8.217e-3 | −2.154e-3 | 1.2e-12 | Newton |
| 3 | 1.482936716 | 1.007329734 | +7.330e-3 | −8.547e-4 | 4.3e-12 | Newton |
| 4 | 2.482936716 | 0.953105891 | −4.689e-2 | −5.901e-1 | 2.8e-10 | Newton |
| 5 | 2.403466433 | 0.987419555 | −1.258e-2 | −2.696e-1 | 3.2e-10 | Newton |
| 6 | 2.356802213 | 0.996422199 | −3.578e-3 | −1.329e-1 | 1.7e-10 | Newton |
| 7 | 2.329885818 | 0.999328837 | −6.712e-4 | −8.808e-2 | 1.1e-10 | Newton |
| 8 | 2.322266242 | 0.999958786 | −4.121e-5 | −7.870e-2 | 9.6e-11 | Newton |

At the cap it proposes λ = 2.321742540 and stops. **The converged root is
2.321736641.** It was not lost; it was interrupted five decimals from the
answer, with a two-sided bracket [2.322266, 1.482937] in hand, every inner
solve converged, every probe trusted, and |c| falling by ~16× per iteration.

The other five points at the cap, same picture:

| point | λ at outer 8 | \|c\| at outer 8 | next proposed λ | true root | notes |
|---|---:|---:|---:|---:|---|
| def2-SVP 2.50 | 2.322266 | 4.12e-5 | 2.321743 | 2.321737 | all Newton |
| def2-SVP 3.00 | 2.410762 | — | 2.405670 | 2.405048 | 1 SignStep, 1 backtrack |
| def2-SVP 3.50 | 2.625000 | — | 2.500000 | 2.457063 | 1 SignStep, 3 backtracks |
| aug-cc-pVDZ 2.50 | 1.569833 | 1.56e-4 | 1.569261 | 1.569217 | 1 backtrack |
| aug-cc-pVDZ 3.00 | 1.559978 | 6.13e-4 | 1.556934 | 1.556441 | 1 SignStep, 1 backtrack |
| aug-cc-pVDZ 3.50 | 1.560091 | 6.89e-4 | 1.555509 | 1.554270 | 1 SignStep, 1 backtrack |

The backtracks are guard 1 working correctly: a ±1-clamped step from the
plateau overshoots onto the over-localization region, the inner solve there does
not converge, and the loop halves back. Each costs an outer iteration, which is
why def2-SVP 3.50 Å — with three of them — is the point that needs 21.

---

## 3. The N(λ) scan: why all three proposed fixes are unnecessary

This is the falsifier the hypotheses named. Fixed-λ UKS/PBE solves at
λ = 0.0 … 4.0 in steps of 0.1, inner cap 300, in two modes: **FIXED** (every
solve from the driver's default guess, i.e. today's behaviour) and **CONT**
(each solve seeded from the previous λ's converged orbitals, i.e. candidate
fix (c) in its purest form).

def2-SVP 2.50 Å, FIXED mode, abridged:

| λ | N | E (Ha) | inner conv | iters |
|---:|---:|---:|---|---:|
| 0.0 | 1.500000018 | −4.979691 | yes | 9 |
| 0.1 | 1.999430880 | −4.737944 | **no** | 300 |
| 0.2 | 1.998779364 | −4.770762 | **no** | 300 |
| 0.3 | 1.010651504 | −4.874228 | **no** | 300 |
| **0.4** | **1.008480131** | **−4.873712** | **yes** | **11** |
| 1.0 | 1.007679862 | −4.873224 | yes | 12 |
| 1.5 | 1.007315007 | −4.872765 | yes | 11 |
| 2.0 | 1.006432407 | −4.871163 | yes | 67 |
| 2.2 | 1.004702445 | −4.867489 | yes | 14 |
| 2.3 | 1.001447816 | −4.860134 | yes | 14 |
| **2.32** | **= 1.0 (root)** | **−4.856788** | | |
| 2.4 | 0.988325056 | −4.829136 | yes | 16 |
| 2.5 | 0.942608463 | −4.716797 | yes | 18 |
| 3.0 | 0.520030763 | −3.550707 | yes | 132 |
| 3.4 | 0.039584002 | −1.996669 | yes | 12 |
| 4.0 | 0.034535913 | −1.978948 | yes | 10 |

**The decisive reading: the branch reachable from λ = 0 CONTAINS the root.** By
λ = 0.4 the fixed-guess solve is already on the localized branch (N = 1.00848,
11 iterations), and N falls **smoothly and monotonically** from 1.00848 through
exactly 1.0 at λ ≈ 2.32. There is no barrier to cross and no second branch to
find. H-PHYSICS predicted no root on this branch; there is one.

CONT mode is **numerically identical to FIXED** on this branch — same N to 10
decimals, same E, at every λ from 0.4 to 4.0 (e.g. λ = 2.3: both 1.0014478157,
both −4.8601343743). Only the iteration counts differ, and not consistently in
continuation's favour (λ = 2.9: FIXED 57 iters, CONT 184). So:

* **candidate (c), re-seed each inner solve including the FD probe — rejected.**
  It converges to the same point. It cannot help in λ ∈ [0.1, 0.3] either,
  because there the previous λ's orbitals ARE the delocalized λ = 0 state: CONT
  fails at 0.1/0.2/0.3 exactly as FIXED does. Measuring this is what showed the
  brief's hypothesis (and mine) to be beside the point.
* **candidate (a), symmetry-broken / localized start — unnecessary.** The
  localized branch is reached unaided by λ = 0.4, which the loop passes through
  on its first Newton step (it lands at λ = 0.483).
* **candidate (b), target continuation — unnecessary.** It exists to reach a
  branch that is already reachable.

The λ ∈ [0.1, 0.3] non-convergence is real but **the loop never visits it**: the
first step from λ = 0 is to λ ≈ 0.48 (Newton) or λ = 1.0 (SignStep), both past
it. The dN/dλ ≈ −7e-5 plateau is also real, and it is what makes the clamped
step run long — the cost, not a correctness obstacle.

---

## 4. The outer-cap sweep: the measurement that selected the fix

Each point solved from λ = 0 at `cdft_max_outer` ∈ {8, 12, 16, 20, 30}, and
separately from NWChem's λ at cap 8 for the anchor. Inner cap 150 throughout.

| point | cap 8 | 12 | 16 | 20 | 30 | outer needed | from NWChem's λ |
|---|---|---|---|---|---|---:|---:|
| def2-SVP 2.50 Å | FAIL | OK | OK | OK | OK | **11** | 3 |
| def2-SVP 3.00 Å | FAIL | FAIL | OK | OK | OK | **13** | 4 |
| def2-SVP 3.50 Å | FAIL | FAIL | FAIL | FAIL | OK | **21** | 3 |
| aug-cc-pVDZ 2.50 Å | FAIL | FAIL | OK | OK | OK | **13** | 4 |
| aug-cc-pVDZ 3.00 Å | FAIL | FAIL | OK | OK | OK | **14** | 4 |
| aug-cc-pVDZ 3.50 Å | FAIL | FAIL | OK | OK | OK | **14** | 4 |

Wherever two caps both converge they give bit-identical λ and E, which is what a
cap being an upper bound means. The fix is therefore `LAM0_MAX_OUTER = 30`, and
`MEASURED_DEEPEST_OUTER_FROM_ZERO = 21` is pinned by a compile-time assert so
lowering the cap below the measured depth cannot build.

30 is **also `RhfConfig`'s own default `cdft_max_outer`.** So a real user calling
`run_cdft` without a λ start already had this budget; only the validation row's
cap of 8 made the capability look absent. The issue's premise — "a user calling
`run_cdft` without a λ start gets a failure or a wrong-branch state" — is, on
this measurement, **not true at the library default**.

### Why 8 cannot work, as arithmetic rather than as a measurement

The Newton step is clamped to `MAX_STEP` = 1 per outer iteration, so arriving at
λ_root from λ⁰ needs at least ⌈|λ_root − λ⁰| / 1⌉ iterations however good the
Jacobian is. From λ = 0 to λ ≈ 2.46 that is 3 iterations of pure travel before
any refinement, plus one per guard-1 backtrack, plus the ~4 the final quadratic
approach takes (|c| falls 3.6e-3 → 6.7e-4 → 4.1e-5 → …). 11 is the floor, not a
contingency; 8 was never enough.

### Interaction with PR #309 (now main, `0b4fa7dc`)

#309 raised this row's **inner** cap 100 → 150 because the deepest inner solve
needs 106. My change touches the **outer** cap, for the λ = 0 start only, and
the two compose cleanly but are not independent:

* All six λ = 0 runs here were measured at inner cap 150, i.e. **on top of**
  #309. At the old inner cap of 100, guard 1 cannot tell cap-truncation from
  overshoot (#309's finding), so a λ = 0 failure would have been ambiguous
  between "needs more outer iterations" and "needs more inner iterations". The
  hypotheses file named this confounder in advance and the plan was to run both
  caps; #309 landing in main during the pause made that moot and the
  measurement is at 150 only. **This is a dependency, not a coincidence: #283's
  conclusion is only valid on top of #309.**
* The NWChem-started rows keep `E2E_MAX_OUTER = 8` and are untouched, so
  #309's own numbers (including its negative control
  `inner_cap_100_is_what_breaks_def2_svp_350`) still hold unchanged.

---

## 5. Before/after for every existing cDFT test

**Structurally unchanged: `git diff` on `crates/*/src/` contains zero
executable lines** — doc comments only (verified by filtering `///` lines out of
the diff; the result is empty). `run_inner`, `ScalarStepper`, `solve_scalar`,
`CdftSeed`, `solve_cdft_uhf_seeded` and every constant in `cdft_driver.rs` are
byte-identical to `0b4fa7dc`. The new test adds a config constructor and a test
function; it changes no existing one. `diabat_config` is untouched, so the
NWChem-started rows build the same `RhfConfig` as before.

Suites run with `--include-ignored` after the change (`cdft_uhf`,
`cdft_outer_loop`, `cdft_state_selection`, `cdft_guess_catalogue_precision`,
`validation_cdft`, `validation_cdft_state`): see §7. λ and E in these suites are
asserted against fixed reference values, not recorded as deltas, so "unchanged"
means every such assertion still passes at its existing bar.

---

## 6. Mutation ledger

Each mutation must make `he2_plus_diabats_converge_from_lambda_zero` fail. See
§7 for the observed counts — **a green line with zero tests executed proves
nothing**, so each entry records the `test result:` line, not just the exit code.

| # | mutation | what it breaks | expected |
|---|---|---|---|
| 1 | `LAM0_MAX_OUTER` 30 → 8 | the fix itself: the λ = 0 runs lose their budget | FAIL on def2-SVP 2.50, the first point |
| 2 | `LAM0_MAX_OUTER` 30 → 21 | the margin over the measured deepest (21) | FAIL at compile time (the `const _: () = assert!` is `>`, not `>=`) |
| 3 | negative control's `starved.cdft_max_outer = E2E_MAX_OUTER` → `LAM0_MAX_OUTER` | the control arm: the starved run is no longer starved | FAIL — "the negative control is gone" |
| 4 | `diabat_config_from_zero`'s `cdft_lambda_init: None` → `Some(vec![lam])` (wrong seed: hand it the root) | the test's whole premise — it would no longer start from λ = 0 | FAIL on `outer_iters > 1`, or the control arm converges |
| 5 | anchor bar `TOL_LAMBDA` → 1e-12 | that the anchor is a real comparison and not slack | FAIL (measured max \|dλ\| is 3.07e-10) |

---

## 7. Test results as observed

Counts read from the `test result:` line in every case.

### The new test, unmutated

```
He2+ R=2.50 def2-svp: lambda0 -> lam +2.321736641180 N 0.999999999976 E -4.856787855242 outer 11 | NWChem-start -> lam +2.321736640903 E -4.856787855292 outer 3
He2+ R=3.00 def2-svp: lambda0 -> lam +2.405048345270 N 1.000000000009 E -4.865106758352 outer 13 | NWChem-start -> lam +2.405048345413 E -4.865106758321 outer 4
He2+ R=3.50 def2-svp: lambda0 -> lam +2.457063435841 N 0.999999999950 E -4.869160282453 outer 21 | NWChem-start -> lam +2.457063436148 E -4.869160282346 outer 3
He2+ R=2.50 aug-cc-pvdz: lambda0 -> lam +1.569217321192 N 0.999999999959 E -4.863290969808 outer 13 | NWChem-start -> lam +1.569217320956 E -4.863290969900 outer 4
He2+ R=3.00 aug-cc-pvdz: lambda0 -> lam +1.556440654286 N 0.999999999996 E -4.869166397662 outer 14 | NWChem-start -> lam +1.556440654381 E -4.869166397639 outer 4
He2+ R=3.50 aug-cc-pvdz: lambda0 -> lam +1.554270462551 N 0.999999999989 E -4.872239730994 outer 14 | NWChem-start -> lam +1.554270462514 E -4.872239731000 outer 4
lambda0 vs NWChem-start over all 6 points: max |dlambda| 3.07e-10 (bar 3e-5), max |dE| 1.07e-10 Ha (bar 2e-6)
He2+ R=3.50 def2-svp: lambda = 0 at the row's 8-iteration cap fails as issue #283 reported: cDFT outer loop did not converge in 8 iters
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 5 filtered out; finished in 259.67s
```

The exactness anchor passes at 3.07e-10 against a 3e-5 bar — **five orders
inside it**, on the quantity that would differ by O(1) if the two starts found
different states.

### Negative control for the issue as filed

All six points from λ = 0 at the row's `cdft_max_outer = 8`, inner cap 150:

```
RESULT def2-svp    2.50 cap=150: FAIL Convergence("cDFT outer loop did not converge in 8 iters") 17.2s
RESULT def2-svp    3.00 cap=150: FAIL Convergence("cDFT outer loop did not converge in 8 iters") 25.9s
RESULT def2-svp    3.50 cap=150: FAIL Convergence("cDFT outer loop did not converge in 8 iters") 27.7s
RESULT aug-cc-pvdz 2.50 cap=150: FAIL Convergence("cDFT outer loop did not converge in 8 iters") 12.5s
RESULT aug-cc-pvdz 3.00 cap=150: FAIL Convergence("cDFT outer loop did not converge in 8 iters") 15.8s
RESULT aug-cc-pvdz 3.50 cap=150: FAIL Convergence("cDFT outer loop did not converge in 8 iters") 17.2s
```

6 of 6 fail, which is the issue's symptom reproduced. Disabling the fix (cap
back to 8) restores it — that is mutation 1, and it is also asserted inside the
new test as its own negative-control arm on def2-SVP 3.50 Å.

### Regression suites, after the change

`cargo test --release -p ferric-scf --include-ignored` on every cDFT suite the
acceptance criteria name:

| suite | result |
|---|---|
| `cdft_uhf` | `ok. 2 passed; 0 failed; 0 ignored` (57.1 s) |
| `cdft_outer_loop` | `ok. 3 passed; 0 failed; 0 ignored` (9.7 s) |
| `cdft_state_selection` | `ok. 9 passed; 0 failed; 0 ignored` (188.9 s) |
| `cdft_guess_catalogue_precision` | `ok. 6 passed; 0 failed; 0 ignored` (2.9 s) |
| `validation_cdft` | `ok. 3 passed; 0 failed; 0 ignored` (105.0 s) |
| `validation_cdft_state` | `ok. 7 passed; 0 failed; 0 ignored` (27.1 s) |

30 tests, 0 failed, 0 ignored. Non-zero `passed` counts in every row, so these
runs executed; `--include-ignored` is what makes that true for the two
`validation_*` suites, whose tests are `#[ignore]`-gated.

---

## 8. Observed mutation runs

Every entry below is a run whose `test result:` line was read. Mutations were
applied one at a time and the file restored from a backup after each.

**Mutation 2 — `LAM0_MAX_OUTER` 30 → 21.** Fails at COMPILE time, as designed:

```
error[E0080]: evaluation panicked: LAM0_MAX_OUTER is at or below the measured
deepest outer count from lambda = 0 (21, def2-SVP 3.50 A) -- see its doc
comment for the table
error: could not compile `ferric-scf` (test "validation_cdft_et")
```

A cap equal to the measured depth is rejected because the assert is `>`, not
`>=`: a cap with no margin over a basin-sensitive measurement is not a bound.

**Mutation 1 — the fix removed** (`diabat_config_from_zero`'s cap
`LAM0_MAX_OUTER` → `E2E_MAX_OUTER`):

```
panicked at validation_cdft_et.rs:1548:17:
He2+ R=2.50 def2-svp: lambda = 0 start did not converge in 30 outer
iterations: Convergence("cDFT outer loop did not converge in 8 iters")
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 5 filtered out
```

Fails on the FIRST point, in 17.7 s. This is the acceptance criterion "disable
the change → the failure returns". (The panic message says 30 while the config
carries 8 — a cosmetic inconsistency *created by the mutation*, since the
message interpolates the const and the mutation changed only the struct field.)

**Mutation 3 — the negative-control arm defanged** (`starved.cdft_max_outer`
`E2E_MAX_OUTER` → `LAM0_MAX_OUTER`):

```
panicked at validation_cdft_et.rs:1628:20:
He2+ R=3.50 def2-svp: lambda = 0 now converges in 8 outer iterations
(lam +2.457063435841 N 0.999999999950 outer 21). The negative control is gone...
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 5 filtered out
```

**Mutation 4 — wrong seed** (`cdft_lambda_init: None` → `Some(vec![2.4570638130])`,
i.e. the config named "from zero" is handed the root):

```
panicked at validation_cdft_et.rs:1628:20:
He2+ R=3.50 def2-svp: lambda = 0 now converges in 8 outer iterations
(lam +2.457063436148 N 0.999999999906 outer 3). The negative control is gone...
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 5 filtered out
```

**This one did not fail where I predicted, and the difference matters.** §6
predicted it would trip the `outer_iters > 1` assert. It did not: a run started
AT the root still takes 3 outer iterations, so `outer_iters > 1` is satisfied
and that assert is **blind to this mutation**. What caught it is the
negative-control arm, which noticed the starved 8-iteration run converging in
3. So the control arm is the load-bearing assertion in this test, and
`outer_iters > 1` is a weaker check than I credited it with — it guards only
against a λ start that is already within tolerance, which is the case it is
documented for. Recorded rather than quietly re-labelled, because a mutation
surviving the assertion you aimed at it is information about the assertion.

**Mutation 5 — the anchor bar tightened below the measured value**
(`TOL_LAMBDA` → 1e-12 on the λ-from-0 vs λ-from-NWChem comparison):

```
panicked at validation_cdft_et.rs:439:5:
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 5 filtered out
```

The anchor is therefore a live comparison at its own precision (max measured
|dλ| = 3.07e-10) and not slack that would pass on any pair of numbers.

---

## 9. Concerns and limits of this result

1. **`LAM0_MAX_OUTER = 30` has a margin of 9 over a basin-sensitive
   measurement.** The deepest count (21, def2-SVP 3.50 Å) comes from a
   trajectory with three guard-1 backtracks, and backtrack counts depend on
   where the clamped step happens to land. A different BLAS kernel could move
   it. 30 was chosen because it is `RhfConfig`'s default rather than because 21
   + 9 is a derived bound, and the compile-time assert pins only that the cap
   exceeds the measured depth, not that the depth is a ceiling.
2. **The conclusion depends on PR #309 being in main.** All six λ = 0 runs are
   at inner cap 150. At the old cap of 100, guard 1 reads a truncated inner
   solve as overshoot, which would have confounded "needs more outer
   iterations" with "needs more inner iterations". I did not re-measure at 100.
3. **The N(λ) scan is on a 0.1 grid and only def2-SVP 2.50/3.00 Å were read in
   full.** It establishes that the branch from λ = 0 reaches the root, which is
   what it was for; it does not rule out a finer feature between grid points,
   and the λ ∈ [0.1, 0.3] non-convergence window is bounded only to that
   resolution. The loop steps over that window rather than through it, so this
   is a limit on the scan, not on the fix.
4. **The issue's stated user impact may be overstated, and I did not test the
   Python path.** `RhfConfig`'s default `cdft_max_outer` is already 30 and its
   default `cdft_lambda_tol` is 1e-5 (far looser than the row's 1e-10), so a
   user calling `run_cdft` on this system probably never hit the reported
   failure. I verified the defaults by reading them, not by running `run_cdft`
   from Python on He₂⁺. If that matters for closing the issue, it is one run.
5. **One acceptance criterion is satisfied differently than the issue asked.**
   Item 4 says to implement the chosen strategy "as the DEFAULT when
   `cdft_lambda_init` is None ... otherwise behind a config field". Neither was
   needed: no strategy was implemented, because none was required. The
   corresponding change is a test-side cap plus doc corrections, so the
   "existing results unchanged" condition holds trivially (zero executable
   lines changed in `src/`).
6. **Gates, as observed on the final tree.** `cargo fmt --all --check` clean;
   `cargo clippy --release -p ferric-scf -p ferric-python --all-targets --
   -D warnings` exit 0; `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
   -p ferric-scf -p ferric-python` exit 0. The doc gate first appeared to fail
   (`DOC_RC=101`) and that was **my harness, not the tree**: I read
   `${PIPESTATUS[0]}` after an intervening `echo`, so the variable no longer
   referred to the cargo invocation. Re-run on its own it is 0. Recorded
   because a wrapper that reports a status for the wrong command is exactly the
   failure mode that makes a green gate meaningless, and it happened to be
   wrong in the safe direction only by luck.
