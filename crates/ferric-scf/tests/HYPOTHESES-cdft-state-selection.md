# Hypotheses — cDFT constrained state SELECTION (written BEFORE any measurement)

Date: 2026-09-16. Branch `fix/cdft-state-selection`, off
`fix/cdft-constrained-stability-solver` (4b8ef0db).

## The established defect (not under test here)

Three external codes (NWChem 7.2.2, ORCA 6.1.1, PySCF 2.13.0) agree that the
UNCONSTRAINED HeNe+/def2-SVP UHF ground state at R = 2.0 A is the sigma-hole
state at E = -130.5053405. ferric converges to the ^2Pi state at
E = -130.50034664, 0.136 eV higher. ORCA independently reproduced BOTH numbers
to ~1e-9 Ha and called ferric's ^2Pi state UNSTABLE (lambda_min = -0.00486901
Eh). So integrals, basis and thresholds are RULED OUT: ferric computes the right
energy for the state it finds; it finds the wrong state.

NWChem's CONSTRAINED solve at the same Becke coordinate and the same integer
target N_He = 2.000 is MULTI-VALUED across its own guesses:
  LOW  (default atomic guess)  E = -130.447402218  lambda = -2.2808  hole: Ne 2p pi
  HIGH (hcore guess + swap)    E = -130.423435597  lambda = -2.6662  hole: sigma/He
ferric reports E = -130.40219057, lambda = -2.753705, which its own
lambda-augmented stability analysis calls a SADDLE (lambda_min = -3.9992e-2),
with a state 0.6671 eV lower reachable by following the negative eigenvector.

The multiplicity appears ONLY at the over-constrained integer target. At
ferric's natural promolecule target (N_He = 1.954484) NWChem's two guesses agree
to < 1e-7 Ha and ferric shows no instability.

## What Part 1 measures

Re-converge state B (fragment = [He], SpinChannel::Total) from SEVERAL
INDEPENDENT ORBITAL GUESSES at the SAME Becke target, and independently SWEEP
the target from the natural value to the integer value. Record for EVERY run:
E_bare, N_final, lambda, outer_iters, and the lambda-augmented stability verdict
(dense-cross-checked).

Guesses: ferric's default (hcore, as the driver does today); explicit hcore;
SAD; the converged STATE A orbitals; the converged natural-target (1.954484)
orbitals; and the post-descent state from the stability analysis
(E = -130.42670715).

Targets: 1.954484 (natural), 1.98, 1.99, 1.995, 2.000.

## THE TWO COMPETING HYPOTHESES (pre-registered, and DISTINGUISHABLE)

**H-GUESS (physics: guess-dependence / multiple constrained solutions).**
  The constrained SCF at the integer target has MULTIPLE distinct solutions, and
  which one is reached is decided by the starting orbitals.
  OBSERVABLE: at target = 2.000, E is MULTI-VALUED across guesses -- the spread
  max(E) - min(E) is >> 1e-6 Ha, and runs cluster into 2+ well-separated energy
  levels. Runs at the SAME energy have the SAME lambda and the same N_final.
  Crucially, E correlates with GUESS IDENTITY and NOT with |N_final - N_target|:
  two guesses that both hit the target to 1e-8 can still land at energies
  differing by ~0.02-0.045 Ha. At the natural target E is SINGLE-VALUED
  (spread < 1e-6 Ha) across the same guesses.

**H-SATURATE (artifact: targeting / constraint saturation).**
  The Becke population of He saturates below 2.000, so the target is only
  asymptotically reachable; lambda runs away and where the lambda-Newton
  tolerance trips decides the answer. The QE `epcdft` Hirshfeld constraint
  behaves exactly this way (q_He = 1.000 is an asymptote; lambda -1.22 -> -2.00
  moves q by 0.017).
  OBSERVABLE: E varies SMOOTHLY AND MONOTONICALLY with the residual population
  error |N_final - N_target|, and the spread in E TRACKS that residual rather
  than the guess label. Concretely: runs that reach the SAME N_final to 1e-8
  have the SAME E to ~1e-6 Ha REGARDLESS of guess; and across the target sweep,
  dN/dlambda collapses toward zero as the target approaches 2.000 (the
  saturation signature), with |lambda| growing without bound.

These predict DIFFERENT OBSERVABLES and are therefore distinguishable:
  * H-GUESS  => E is a function of the GUESS at fixed (target, N_final).
  * H-SATURATE => E is a function of N_final alone, and N_final is a function of
                  where the tolerance tripped.
The discriminating statistic is stated in advance: **partition the runs at
target = 2.000 by N_final agreeing to 1e-7. If within such a group E still
differs by >> 1e-6 Ha, H-GUESS is supported and H-SATURATE is refuted for that
group. If E is constant within each N_final group and varies only across them,
H-SATURATE is supported.**

BOTH may hold simultaneously (guess-dependence AND a saturating coordinate).
That is a legitimate third outcome and will be reported as such, not forced into
one bucket.

A fourth outcome must not be tuned away: **E is SINGLE-VALUED across all
guesses at target = 2.000.** That REFUTES H-GUESS outright. The task then stops
before implementing any guess-based fix, because the premise would not have been
reproduced. That outcome is a result, not a failure.

## dN/dlambda: the saturation test proper

H-SATURATE makes a second, sharper prediction that H-GUESS does not: the
constraint Jacobian dc/dlambda (already computed by the driver's
finite-difference Newton) should COLLAPSE toward 0 as the target approaches
2.000. If instead |dc/dlambda| stays O(0.01-1) all the way to the integer
target, the coordinate is NOT saturating there and H-SATURATE is refuted
independently of the energy spread. This is recorded for every target in the
sweep.

## EXACTNESS ANCHOR (must pass before any verdict)

lambda = 0 (vacuous constraint) must reproduce the UNCONSTRAINED solve exactly
from the same guess. This anchors the harness, not the physics.

## THE ANCHOR'S KNOWN BLIND SPOT (stated so the validation cannot hide behind it)

ferric's cDFT exactness anchor is AT lambda = 0. It is therefore STRUCTURALLY
BLIND to any defect proportional to lambda -- exactly the class of bug found in
QE's epcdft, which double-adds a lambda-proportional term and still passes its
own lambda = 0 anchor bit-identically.

So this task's validation MUST include a lambda != 0 check that does NOT reduce
to the anchor. The one used here: at a converged constrained state with
lambda != 0, finite-difference the AUGMENTED energy E[rho] + lambda*N_C[rho]
along a random orbital rotation and compare against the ANALYTIC augmented
gradient (the occ-virt block of the lambda-augmented MO Fock). A term
double-added at strength lambda shifts the analytic gradient by exactly that
amount while the FD of the correctly-written functional does not follow, so the
two disagree at O(lambda) -- a defect the lambda = 0 anchor cannot see. This is
run at lambda = -2.75 (state B), the largest |lambda| in the lane.

## MUTATION PLAN (every new guard must be seen to FAIL)

Every guard added in Part 2/3 is mutation-tested IN THE FOREGROUND against a
deliberately broken version, and the failure output is recorded. A test never
seen to fail is an assumption.

## TOO CLEAN IS A STOP CONDITION

If the fix makes every guess agree to machine precision at the integer target,
that is a reason to AUDIT, not to report. Real multi-solution landscapes have
basins with edges.
