# Hypotheses — cDFT constrained-state stability (written BEFORE any measurement)

Date: 2026-09-16. Branch `test/cdft-constrained-stability`, off `origin/main` (286b48a1).

## The question

Are the HeNe+ / def2-SVP / R=2.0 A cDFT diabats A (hole on He, lambda=+1.631409,
E=-130.36185950) and B (hole on Ne, lambda=-2.753705, E=-130.40219057) genuine
MINIMA of their own constrained manifolds, or merely stationary points?

Today's state-identity audit (`test/cdft-state-identity`, d1d30fa5) proved the
UNCONSTRAINED HeNe+ UHF state ferric converges to is an internally unstable
saddle (matches PySCF's MOM-forced pi state to 1.26e-9 Ha; PySCF `stability()`
puts it 0.136 eV above the sigma minimum). That audit established symmetry and
spin for A and B, explicitly NOT their variational status.

## THE OPERATOR (stated before measuring, so it cannot be retrofitted)

A constrained SCF solution is NOT expected to be a minimum of the bare E[rho].
The cDFT Lagrangian W[rho,lambda] = E[rho] + lambda(N_C[rho] - N_target) is a
saddle in the combined (rho,lambda) space BY CONSTRUCTION: minimized over rho,
maximized over lambda. Running an ordinary E[rho] stability analysis at a
constrained solution would report spurious instabilities on perfectly good
diabats.

The correct check is internal stability of the LAMBDA-AUGMENTED problem AT FIXED
CONVERGED LAMBDA: is the state a minimum of E[rho] + lambda*N_C[rho] over
orbital rotations?

## Structural claim to be VERIFIED (not assumed)

In `cdft_driver.rs` the constraint enters as `lw = l * &w_mats[ci]`, a FIXED
one-electron operator added to the Fock. W is built once from geometry + grid
(`build_weight_matrix`), outside the lambda loop, and never rebuilt from the
density. A density-independent one-electron term contributes to the Fock but NOT
to the Fock RESPONSE (dF/dD).

CLAIM: therefore the lambda-augmented orbital Hessian EQUALS the ordinary UHF
orbital Hessian evaluated at the constrained density/orbitals with the
CONSTRAINED orbital energies. i.e. `uhf_newton::hessian_matvec` fed
(C_constrained, F_constrained_MO) already IS the lambda-augmented Hessian, with
no new derivation.

FALSIFICATION TEST (run before any verdict): finite-difference the augmented
functional E[rho] + lambda*N_C[rho] along a random orbital rotation at fixed
lambda, and compare to the analytic matvec. If W were rebuilt from the density
anywhere, or if the Becke grid weights carried a density dependence, the FD
would disagree and the claim would be FALSE -- that is a significant finding and
the task stops there.

## The two competing hypotheses

**H-PHYSICS: both A and B are minima of their lambda-augmented functionals.**
  Observable: lambda_min(H_aug) > 0 for BOTH states, comfortably above the
  eigensolver residual and the numerical noise floor. The 1.0975 eV A-B gap then
  sits between two genuine constrained minima -- the first real variational
  support that gap has ever had.

**H-ARTIFACT: one or both are saddles within their own constrained manifolds.**
  Observable: lambda_min(H_aug) < 0 for at least one state, by a margin larger
  than the residual. Plausible because (a) the UNCONSTRAINED state on this exact
  system is a saddle, and (b) state B is dragged to N_He = 2.000 when the
  natural promolecule Becke population at R=2.0 A is 1.954484 -- 0.046 e past
  its natural value (`test/cdft-atomic-ip-anchor`, 794a1551). If an instability
  appears ONLY at the over-constrained integer target and not at the natural
  target, that localizes the cause to over-constraint rather than to cDFT.

These predict DIFFERENT observables: the SIGN of lambda_min. They are
distinguishable. H-PHYSICS additionally predicts the sign is robust to
tightening the eigensolver; H-ARTIFACT predicts a negative eigenvalue whose
eigenvector, when followed and re-converged AT THE SAME CONSTRAINT, reaches a
strictly LOWER constrained energy (which would invalidate the reported diabat
energies and the 1.0975 eV gap).

A third outcome is possible and must not be tuned away: **lambda_min ~ 0 within
the noise floor** = marginal stability. That is a real result and will be
reported as such, with the noise floor stated.

## Secondary experiment (pre-registered)

State B at the NATURAL target N_He = 1.954484 vs the INTEGER target N_He =
2.000000. H-ARTIFACT-OVERCONSTRAINT predicts stability differs between them
(instability at integer, stability at natural). H-PHYSICS predicts both stable.

## Exactness anchor (MUST pass before any verdict is believed)

At lambda = 0 the constrained analysis must reduce EXACTLY to the unconstrained
one, and on HeNe+ it must reproduce the KNOWN answer: the unconstrained UHF
state is UNSTABLE (PySCF stability() rejects it). One check validates both the
reduction and the sign convention. Both directions must also be demonstrated:
the analysis must report STABLE on a system that is (water/STO-3G, H2), and
UNSTABLE on the known-unstable HeNe+ reference.

## POST-HOC CORRECTION (2026-09-16): the eigensolver was wrong on one anchor

Recorded here rather than silently edited above, because a hypotheses file is a
pre-registration and rewriting it would destroy what it is for.

The original run used a PRIVATE single-root Davidson written inside
`cdft_constrained_stability.rs`. An adversarial review found that solver returns
a CONVERGED WRONG eigenvalue on the water/STO-3G anchor, and the finding
reproduced:

```
  dense (ground truth)   lambda_min = +3.6243948468e-1
  private single-root    lambda_min = +3.6883351924e-1  resid 6.63e-11, 2 iters
  library block solver   lambda_min = +3.6243948468e-1  resid 3.72e-15, 7 iters
```

The orbital Hessian is block diagonal in the molecule's irreps. On water (C2v)
the unit-vector seed lands in a closed 2x2 block whose own lowest root is
+3.6883e-1; a single-root iteration converges there and never sees the true
lambda_min. The residual is small because the iteration DID converge - just on
the wrong root - so no residual gate could have caught it. The assertion could
not catch it either: the bar was `lambda_min > 1e-3`, which both the right value
and the wrong one clear by ~360x.

FIX. The private solver is deleted; `lowest_eigenvalue` now delegates to
`ferric_scf::stability::uhf_internal_stability` (block tracking `n_block >= 2`,
dense symmetry-breaking seed, block-wide convergence). Independently, EVERY
lambda_min in the file is now cross-checked against a DENSE Hessian built from
the same matvec - the only check that catches this class of bug, and the durable
half of the fix. Its reachability is pinned by
`dense_cross_check_rejects_the_symmetry_block_root`, which re-runs the old
solver, asserts it still returns the wrong value with a small residual, asserts
the OLD bar accepts it, and asserts the NEW guard rejects it.

WHAT CHANGED, and what did not. The defect is symmetry-specific: it fires on
water (C2v) and NOT on HeNe+ (C-inf-v). Re-measured with the library solver and
confirmed against dense:

| quantity                      | before        | after (= dense) | status    |
|-------------------------------|---------------|-----------------|-----------|
| water/STO-3G anchor lambda_min| +3.688e-1     | +3.6243948468e-1| CORRECTED |
| H2/STO-3G anchor lambda_min   | +4.034e-1     | +4.0344721450e-1| same      |
| HeNe+ lambda=0 lambda_min     | -4.8706726e-3 | -4.8706726180e-3| same      |
| STATE A lambda_min            | +1.31943167e0 | +1.3194316735e0 | same      |
| STATE B lambda_min            | -3.99918672e-2| -3.9991867160e-2| same      |
| B @ natural target lambda_min | -3.718e-3     | -3.7183550645e-3| same      |
| descent dE                    | 0.02451658 Ha | 0.02451658 Ha   | same      |
| descent dE in eV              | 0.6671 eV     | 0.6671 eV       | same      |
| aug-Hessian FD rel            | 5.70e-11      | 5.701e-11       | same      |
| descent endpoint lambda_min   | -5.91e-9      | -1.0102213317e-9| see below |

So the HEADLINE IS UNAFFECTED: state B is still a saddle, the descent still
reaches 0.6671 eV lower, and the 1.0975 eV A-B gap is still wrong. Only the
water anchor moved.

The descent ENDPOINT deserves its own line. Its lambda_min is a near-exact zero
mode, and the dense value (-1.0102213317e-9) differs from the old private
solver's (-5.91e-9) in digits that are below the noise floor of both. The
verdict is the same - MARGINAL - and it is now read from
`StabilityResult::verdict()`. Note the iterative solve at that point reports
INDETERMINATE (it stagnates at a lowest-root residual of ~1.2e-8 against a
1e-8 request), which is the honest report for an iterative method on a
degenerate zero mode; the verdict is therefore taken from the DENSE eigenvalue,
which is exact by construction. Loosening `conv_thresh` until the iterative
solver claimed convergence would have manufactured the verdict.
