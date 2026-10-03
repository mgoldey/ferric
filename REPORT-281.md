# Issue #281 — cDFT He₂⁺/def2-SVP at 3.50 Å, inner SCF fails within 1e-9 of the λ root

Worktree `~/qc/ferric-cdft350`, branch `fix/cdft-he2-350-convergence`, base `origin/main` @ `ef9d958c`.

## 1. Hypotheses, written BEFORE any measurement

Timestamped by this file's first commit; nothing below §1 was known when §1 was written.

The observable on main: the outer loop reaches λ = 2.4570634438 with |N − 1| = 9.6e-10
(tol 1e-10); the *converged* inner solves there take 77–99 of their 100 iterations; every
inner solve between there and 1e-9 above it hits the 100-cap unconverged, reporting
N = 1.000018, 1.00013, **1.99936**. The outer loop then backtracks to a 5e-10 step and
spends `cdft_max_outer = 8`.

### H-SLOW (physics: the inner SCF is merely slow near the root)
c(λ) has a near-vertical cliff at the root (the reported plateau dN/dλ ≈ −0.006 ending in an
over-localization cliff), so the inner SCF's effective condition number blows up there: the
constrained Fock's occupied/virtual gap closes as the hole localizes and DIIS needs more
iterations than 100 at level shift 0.3. Nothing is bistable; the fixed point is unique and
reachable.

**Predictions if H-SLOW is true:**
- P1. Raising the inner cap 100 → 200 → 400 monotonically *reduces* the number of
  unconverged λ points. At 400 every λ in the neighbourhood converges.
- P2. The converged points' N lands smoothly on the c(λ) curve: N → 1 continuously as
  λ → λ*, and in particular the N = 1.99936 point, once converged, reports N ≈ 1.0000x
  consistent with its neighbours — **not** 2.
- P3. `hf_mismatch` between a main point and its FD probe stays in the same-basin band
  (≤ 7e-9 Ha per the const's own measured population), so guard 2 never fires and the
  Jacobian is trusted.
- P4. The outer loop converges at a larger cap *without any other change*, and λ lands
  within TOL_LAMBDA (3e-5) of NWChem's 2.4570634438-ish reference.
- P5. Iteration counts for the *other four* points are essentially unchanged (they already
  converge well inside 100), so a cap raise is free for them.

### H-BASIN (artifact/physics: two nearby SCF solutions at the same λ)
At λ near the root the constrained problem has (at least) two solutions: the intended
hole-on-He1 state with N_He1 ≈ 1, and a second state with N_He1 ≈ 2 (the *mirror*, hole on
He2, which has N_He1 = N_elec − 1 = 2 exactly). N = 1.99936 is ~2, i.e. the mirror image;
λ·W pushes charge *off* He1, and past the cliff the lowest solution of E + λ·c is the state
that dumps the hole on the other atom. The 100-iteration "unconverged" density is then a
transient oscillating *between* the two basins, and the cap is not the binding constraint —
the fixed-point structure is.

**Predictions if H-BASIN is true:**
- Q1. Raising the cap does **not** monotonically clean up: some λ converge to N ≈ 2 (the
  wrong state) rather than to N ≈ 1. At 400 the point that reported 1.99936 converges
  *to ≈ 2*, cleanly, with few DIIS oscillations near the end.
- Q2. `hf_mismatch` across an FD pair that straddles the two states lands in the measured
  cross-basin band (5.5e-5 … 0.35 Ha), so guard 2 fires and marks the Jacobian untrusted.
- Q3. The energies of the two solutions at one λ differ by a finite gap (V jumps), visible
  directly in the trace's `V=` column between consecutive λ a few 1e-10 apart.
- Q4. A larger cap alone does **not** converge the outer loop; it converges the inner
  solves to the wrong answer and the outer loop either accepts a λ whose N ≈ 2 (a wrong
  state, caught by the population assert) or keeps backtracking.
- Q5. The fix must be state-selective: re-seed each inner solve from the previous
  *converged* λ's orbitals (`CdftSeed`/`solve_cdft_uhf_seeded` exists, but the outer loop
  currently applies ONE fixed guess to every λ — confirmed by reading
  `lambda_newton_from`'s `run_inner`, which closes over an immutable `guess`), so the
  continuation stays on the branch it is tracking.

### Discriminating observable
P2 and Q1 are *opposite* statements about the same number: what the N = 1.99936 point
converges to at cap 400. P3 and Q2 are opposite statements about `hf_mismatch` on the pairs
that straddle the cliff. Either experiment alone distinguishes them; both are read off the
same trace. **The experiment can therefore distinguish the hypotheses**, which is the
precondition for running it.

### Stated in advance: what a cap raise must NOT be allowed to do
If a larger cap makes the outer loop "converge" at a λ whose population is ≈ 2 rather than
≈ 1, that is H-BASIN passing a weaker test, not a fix. The population assert in
`solve_diabat` (|N − 1| < 1e-9) and the λ-vs-NWChem assert are the guards; neither may be
relaxed.
