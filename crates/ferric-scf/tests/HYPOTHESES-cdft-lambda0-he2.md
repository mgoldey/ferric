# Issue #283: the cDFT λ-Newton loop from λ = 0 on He₂⁺

**Written BEFORE any measurement on this branch.** Dated 2026-10-03, against
`origin/main` @ `cc948b41`. Nothing below has been run yet; every number here
is a PREDICTION, and the point of writing them first is that a construction
artifact and a physics result are indistinguishable after the fact.

## The claim under test

`validation_cdft_et.rs` hands every He₂⁺ diabat NWChem's converged multiplier
via `cdft_lambda_init`. Its own doc (lines ~1000-1030) says why: from λ = 0 the
loop "failed or crawled", with a traced "Jacobian" of ±490 from a 1e-3 FD step,
and walked λ to −0.11 at def2-SVP 3.00 Å. A user calling `run_cdft` without a λ
start therefore gets a failure or a wrong-branch state for a textbook symmetric
dimer cation.

## H-PHYSICS: c(λ) genuinely has no usable derivative at λ = 0

λ = 0 is the unconstrained problem. For a symmetric dimer cation at these
separations the unconstrained UKS/PBE solution is the DELOCALIZED one
(N_He1 = 1.5 by symmetry), and the target N = 1.0 is on a different, localized
branch. The two branches are separated by a genuine barrier in the constrained
functional, and the delocalized branch does not deform continuously into the
localized one as λ grows: at some λ_c the localized solution becomes the lowest
and the SCF jumps. If this is the situation, then:

* dc/dλ measured ON the delocalized branch is small and nearly constant
  (the plateau dN/dλ ≈ −0.006 the doc reports), because moving λ a little does
  not localize a symmetric state;
* the FD probe at λ + 1e-3, seeded identically, lands in the SAME delocalized
  basin (not a different one), so `hf_mismatch` is SMALL and guard 2 does NOT
  fire;
* Newton with a tiny true slope takes a huge step, gets clamped to ±1, and
  walks along the plateau until it falls off the far cliff;
* **re-seeding each inner solve from the previous λ's orbitals CANNOT help**,
  because the previous λ's orbitals are ALSO delocalized. Continuation along a
  branch that never reaches the target is still stuck. Prediction: fix (c) alone
  leaves ≥ 4 of 6 points failing, and where it converges it converges to the
  SAME λ as today, not a new branch.

If H-PHYSICS holds, the fix must change WHICH BRANCH the first inner solve sits
on — i.e. candidate (a), a symmetry-broken / fragment-localized start, or
candidate (b), target continuation, which reaches the localized branch by moving
the TARGET rather than λ.

## H-ARTIFACT: the ±490 Jacobian is manufactured by the shared seed

`run_inner` is a closure over ONE immutable `guess`, called at the main point,
the probe, and (k>1) the FD columns. If the inner SCF at λ = 0 is on the
delocalized branch but SITS NEAR THE BASIN BOUNDARY, then a 1e-3 change in λ
with the same guess can tip the DIIS path into the localized basin — a
discontinuity in the MEASURED c that is not a discontinuity in c(λ) restricted
to one branch. The ±490 figure is 0.49 population units over 1e-3, i.e. exactly
the delocalized→localized population gap (1.5 → ~1.0) divided by the FD step.
If this is the situation:

* `hf_mismatch` at λ = 0 is LARGE (≥ 5.5e-5, the measured cross-basin band), so
  guard 2 fires, the Jacobian is rejected, and the loop takes a SignStep;
* re-seeding the probe from the main point's converged orbitals (fix c) makes
  the pair same-basin, `hf_mismatch` drops into 1e-14…1e-9, and the Jacobian
  becomes the plateau slope;
* but that plateau slope is the SAME small number, so **fixing the artifact may
  not fix the failure** — it makes the loop honest about being on the wrong
  branch rather than putting it on the right one.

## The discriminating measurement

These two predict DIFFERENT things about ONE quantity: `hf_mismatch` at λ = 0
and λ = 1e-3 from the common seed.

| | hf_mismatch at λ=0 | guard 2 at λ=0 | what fix (c) does |
|---|---|---|---|
| H-PHYSICS | ≤ 1e-6 (same basin) | inert | nothing; same λ path |
| H-ARTIFACT | ≥ 5.5e-5 | fires, SignStep | Jacobian becomes the plateau slope |

They are NOT mutually exclusive: the artifact can be real AND insufficient.
That is the outcome I consider most likely a priori, and it is why the
pass condition below is about the FINAL ROOT, never about the Jacobian.

## What would make me report "this needs more than re-seeding"

If N(λ) on the branch reachable from λ = 0 never passes through the target —
i.e. the measured N(λ) is a plateau at ~1.5 that drops discontinuously past
~1.0 to the over-localization cliff (0.01-0.06) with NO λ at which a CONVERGED
inner solve gives N ≈ 1.0 — then c(λ) = 0 has no root on that branch and no
root finder, however safeguarded, can find one. The honest fix is then (a) or
(b), and I will say so with the N(λ) scan rather than forcing (c).

**Falsifier for the whole enterprise**: a λ-scan on the localized branch
(seeded from NWChem's λ then held) showing N crossing 1.0 smoothly, while the
same scan seeded from λ = 0's orbitals shows no crossing, is direct evidence
that the branches are disjoint and the problem is branch SELECTION, not root
finding.

## Pass conditions, fixed in advance

1. **Exactness anchor (write first, run before any sweep)**: with the chosen
   fix DISABLED, every existing cDFT test's λ and E are bit-identical. A fix
   that is inert when not requested cannot regress a baselined row.
2. The λ-from-0 root equals the NWChem-started root to the row's bars
   (TOL_LAMBDA, TOL_DE). A different root is a BRANCH ERROR, not a success.
3. Negative control: today's loop from λ = 0 must FAIL or land elsewhere on
   ≥ 1 named point, and disabling the fix must restore that failure.
4. Mutation: a wrong seed and a skipped continuation step must each make the
   new test FAIL, with the passed/failed/ignored counts read.

## Known confounders, named before measuring

* ERI precision 1e-20 vs 1e-14 changes outer-loop outcomes on HeNe⁺. Measure
  at the default only.
* PR #309 (open, NOT in this base) raises this row's inner cap 100 → 150
  because the deepest solve needs 106. At cap 100 an inner solve can be
  CAP-TRUNCATED rather than in another basin, and guard 1 cannot tell those
  apart. A λ = 0 failure at cap 100 is therefore ambiguous; I will run the
  λ = 0 traces at BOTH 100 and 150 so a cap artifact cannot be read as a
  basin result.
* Basin-sensitive, not flaky: assert outcomes, never iteration counts.
