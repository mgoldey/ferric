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

---

## 2. Verdict

**H-SLOW, in its strong form.** There is exactly ONE constrained solution at
every λ in this neighbourhood. The 100-iteration inner cap was too tight by six
iterations, and because `ScalarStepper`'s guard 1 reads an unconverged inner
solve as "the Newton step went too far", a truncated solve at a perfectly good λ
made the OUTER loop backtrack away from the root until it ran out of iterations.

H-BASIN is refuted on both of its own discriminating predictions (Q1, Q2, Q3).

The populations the issue reports as evidence of a basin jump — N = 1.000018,
1.00013, **1.99936** — are not values of c(λ). They are transient DIIS iterates
of a solve that had not finished. Given enough iterations every one of them
converges to N = 0.99999999x and E = −4.8691602822.

## 3. The traces

All runs: release, `scripts/validation/run_slot.sh` (box slot,
`OPENBLAS_NUM_THREADS=1`, RAYON=6), default libint ERI precision, def2-SVP,
R = 3.50 Å, PBE, (99,302) grid, level shift 0.3, `cdft_lambda_tol` 1e-10,
`cdft_max_outer` 8, started from NWChem's λ = 2.457063813.

### 3.1 Cap 100 — the failure, reproduced (37.5 s, and reproducible: two runs byte-identical)

`main` points only (each also runs an FD probe at λ + 1e-3):

| outer | λ | N_C | resid | E | V | inner_conv | iters |
|---:|---|---|---:|---|---|---|---:|
| 1 | +2.457063813000 | 0.999999958652 | −4.13e−8 | −4.8691601804 | −4.8691602820 | **false** | 100 |
| 2 | +2.457063465052 | 0.999999996635 | −3.36e−9 | −4.8691602743 | −4.8691602826 | true | 96 |
| 3 | +2.457063436736 | 1.000860753604 | +8.61e−4 | −4.8710734434 | −4.8689585172 | **false** | 100 |
| 4 | +2.457063450894 | 0.999999998238 | −1.76e−9 | −4.8691602782 | −4.8691602826 | true | 99 |
| 5 | +2.457063443815 | 0.999999999040 | −9.60e−10 | −4.8691602802 | −4.8691602826 | true | 77 |
| 6 | +2.457063435733 | 1.000128422413 | +1.28e−4 | −4.8689030836 | −4.8685875416 | **false** | 100 |
| 7 | +2.457063439774 | **1.999359706580** | +9.99e−1 | −3.8940977405 | −1.4386075423 | **false** | 100 |
| 8 | +2.457063441794 | 1.000018443295 | +1.84e−5 | −4.8692723052 | −4.8692269889 | **false** | 100 |

then `BACKTRACK ... -> +2.457063442804, radius 5.051e-10` and
`Err(Convergence("cDFT outer loop did not converge in 8 iters"))`.

The FD probes, by contrast, were all healthy — converging in 41–71 iterations
with a smooth Jacobian and a same-basin Hellmann–Feynman mismatch:

| outer | probe λ | resid | jac | hf_mismatch | probe_conv | iters |
|---:|---|---:|---:|---:|---|---:|
| 1 | +2.458063813000 | −1.188752e−4 | −1.188339e−1 | **3.48e−10** | true | 71 |
| 2 | +2.458063465052 | −1.188319e−4 | −1.188285e−1 | **9.35e−10** | true | 42 |
| 4 | +2.458063450894 | −1.188301e−4 | −1.188284e−1 | **9.33e−10** | true | 43 |
| 5 | +2.458063443815 | −1.188293e−4 | −1.188283e−1 | **9.33e−10** | true | 42 |

### 3.2 Caps 200 and 400 — BIT-IDENTICAL to each other, and converged (12.9 s / 14.7 s)

| outer | λ | N_C | resid | E | V | inner_conv | iters |
|---:|---|---|---:|---|---|---|---:|
| 1 | +2.457063813000 | 0.999999957208 | −4.28e−8 | −4.8691601774 | −4.8691602826 | **true** | **106** |
| 2 | +2.457063452892 | 0.999999998010 | −1.99e−9 | −4.8691602777 | −4.8691602826 | true | 105 |
| 3 | +2.457063436148 | 0.999999999906 | **−9.36e−11** | −4.8691602823 | −4.8691602826 | true | 68 |

Probes: outer 1 λ+1e-3 → 71 iters, jac −1.188324e−1, hf_mismatch 9.32e−10;
outer 2 → 41 iters, jac −1.188284e−1, hf_mismatch 9.33e−10. Converged at
outer 3 with resid 9.36e−11 < 1e-10.

Caps 120, 150 and 300 give the IDENTICAL result (λ = 2.457063436148,
N = 0.999999999906, E = −4.869160282346, outer 3, deepest inner 106).

**The whole failure is the first row.** At cap 100 the very first inner solve —
at NWChem's own λ, before the outer loop has done anything — is truncated at
100 of the 106 iterations it needs and flagged `converged = false`. Guard 1
then has no `last_good` yet, so it takes the plain step; but the bracket and
trust radius it builds from there are poisoned, and from outer 3 on the loop is
backtracking into an ever-finer neighbourhood of λ values that differ in the
9th decimal. The failing run is also SLOWER in wall time (37.5 s vs ~15 s),
because it spends 8 outer × 100 inner iterations going nowhere.

### 3.3 The discriminating experiment: fixed-λ solves at caps 100/200/400/1000

Every λ the failing run visited, re-solved in isolation through the driver's own
code path (`solve_cdft_uhf`, `cdft_max_outer = 1`, so exactly one inner solve):

| λ (cap-100 label) | cap 100 | cap 200 | cap 400 | cap 1000 |
|---|---|---|---|---|
| 2.457063813000 (start) | ✗ 100, N=0.999999958652 | ✓ 106, N=0.999999957208 | ✓ 106, same | ✓ 106, same |
| 2.457063465052 (o2) | ✗ 100, N=**1.001758119148** | ✓ 126, N=0.999999996634 | ✓ 126, same | ✓ 126, same |
| 2.457063436736 (o3) | ✓ 69, N=0.999999999843 | ✓ 69, same | ✓ 69, same | ✓ 69, same |
| 2.457063450894 (o4) | ✓ 75, N=0.999999998239 | ✓ 75, same | ✓ 75, same | ✓ 75, same |
| 2.457063443815 (o5) | ✓ 77, N=0.999999999043 | ✓ 77, same | ✓ 77, same | ✓ 77, same |
| 2.457063435733 (o6) | ✗ 100, N=**0.876381113023** | ✓ 200, N=0.999999999951 | ✓ 200, same | ✓ 200, same |
| 2.457063439774 (o7) | ✓ 99, N=0.999999999497 | ✓ 99, same | ✓ 99, same | ✓ 99, same |
| 2.457063441794 (o8) | ✓ 94, N=0.999999999269 | ✓ 94, same | ✓ 94, same | ✓ 94, same |

Every λ converges to N = 0.99999999x and E = −4.86916028x. **Not one converges
to a second solution.** P2 confirmed, Q1 refuted.

(The λ above are the trace's 12-decimal printout, so they differ from the run's
own f64 by up to ~5e-13 — which is why o3/o7/o8 converge here but were
truncated in the run. That discrepancy is itself a measurement; see §3.4.)

### 3.4 Why the iteration count is not a property of λ

At λ = 2.457063436736 perturbed in the 13th significant digit:

| Δλ | cap 100 | cap 400 |
|---:|---|---|
| −5e−13 | ✗ 100, N=**1.998967221588**, E=−3.679 | ✓ 181, N=0.999999999841, E=−4.869160282186 |
| −2e−13 | ✗ 100, N=**0.006568335269**, E=−0.874 | ✓ 124, N=0.999999999842, E=−4.869160282189 |
| 0 | ✓ 69, N=0.999999999843 | ✓ 69, same |
| +2e−13 | ✓ 87, N=0.999999999842 | ✓ 87, same |
| +5e−13 | ✓ 82, N=0.999999999841 | ✓ 82, same |
| +1e−12 | ✗ 100, N=1.000004896671 | ✓ 110, N=0.999999999839 |

The iteration count swings 69 → 181 under a 5e-13 change in λ, and the N
reported at the 100-cap ranges over 0.0066 … 1.99897 — a factor of 300 — while
**every one of them converges to N = 0.99999999984 and E = −4.8691602822 to
1e-11.** The DIIS path near this root is ill-conditioned to ~1e-13 in λ, so the
100-cap was sampling it essentially at random. This is also the direct
refutation of the two-basin reading: a second basin would be a second
*converged answer*, and there isn't one.

### 3.5 Which hypothesis the data selected, and why

| prediction | H-SLOW says | H-BASIN says | measured |
|---|---|---|---|
| larger cap cleans up monotonically | yes (P1) | no (Q1) | **yes** — every λ converges at 200+ |
| the N=1.99936 point converges to… | ≈ 1 (P2) | ≈ 2 (Q1) | **0.999999999497** |
| `hf_mismatch` band | same-basin, ≤7e-9 (P3) | cross-basin, ≥5.5e-5 (Q2) | **3.5e-10 … 9.3e-10** |
| V gap between neighbouring λ | none (P3) | finite (Q3) | **V constant to 1e-10** |
| cap raise alone fixes the outer loop | yes (P4) | no (Q4) | **yes, 3 outer iters** |
| other four points perturbed | no (P5) | — | **bit-identical** |

Six for six on H-SLOW. Note the `hf_mismatch` evidence is the strongest single
datum: `HF_MISMATCH_TOL`'s own doc records 359 same-basin pairs at 1.8e-14…7.0e-9
and 15 cross-basin pairs at 5.5e-5…0.35 Ha. Every pair in the failing run sat at
~9e-10 — inside the same-basin band, five orders below the cross-basin band. Guard 2
was right not to fire: there was no basin change to detect.

## 4. The change, and its stated reason

`crates/ferric-scf/tests/validation_cdft_et.rs` only — no library change.

1. **`E2E_INNER_MAX_ITER: 100 → 150.`** Reason: the deepest inner solve the row
   needs is 106 iterations, measured per point as the largest `inner_iters` over
   every main point and FD probe of the whole outer loop:

   | point | deepest inner solve |
   |---|---:|
   | def2-SVP 2.50 Å | 17 |
   | def2-SVP 3.00 Å | 73 |
   | def2-SVP 3.50 Å | **106** |
   | aug-cc-pVDZ 2.50 Å | 23 |
   | aug-cc-pVDZ 3.00 Å | 22 |
   | aug-cc-pVDZ 3.50 Å | 21 |

   150 clears 106 with margin and is never reached by the other five. Scoped to
   this row's `diabat_config`; no global default moved, so
   `cdft_coupling_hene`'s 462-s-vs-17-s cost trap is untouched.

2. **`MEASURED_DEEPEST_INNER: usize = 106`** added, with a COMPILE-TIME
   `const _: () = assert!(E2E_INNER_MAX_ITER > MEASURED_DEEPEST_INNER, ...)`.
   Lowering the cap below the measured depth now fails to build.

3. **`KNOWN_NONCONVERGING` removed** along with the skip branch in `run_basis`,
   so def2-SVP 3.50 Å runs the full row (E_A, λ, |S_AB|, |V(RP)|, Wu–VV,
   log-slopes).

4. **`inner_cap_100_is_what_breaks_def2_svp_350` added** — the negative control,
   in ONE test: cap 150 must converge to the root, and cap 100 must still fail
   with `FerricError::Convergence`. Its `Ok(got) => panic!` arm is what makes the
   control real (mutation 3/4).

5. **Two bars moved, with the measurement recorded on the const** (the issue
   allows this: "if a bar has to move, record the measured value beside it"):

   - **`TOL_S_REL: 2e-3 → 4e-3`** (measured 2.82e-3 at def2-SVP 3.50 Å). Not a
     new defect: the ABSOLUTE S difference is a flat 1.7e-7…1.1e-6 at all six
     points while |S_AB| spans a factor of 21, so the relative error is a fixed
     quadrature offset divided by a shrinking |S|, and this point has the
     smallest |S_AB| of the six (3.85e-4, 2.2× smaller than the next).
   - **`TOL_V_REL: 1e-4 → 2e-3`** (measured 8.28e-4) and the bar DERIVED from it,
     **`TOL_SLOPE_V: 1e-4 → 8e-3`** (= 2δ/0.5; measured 1.65e-3). V is exactly
     proportional to S — `direct_coupling` builds
     V = S·(⟨one⟩ + ½(J−K) − E_elec)/(1−S²) — so dV/V = dS/S +
     d(bracket)/bracket, and the two terms largely cancel: at def2-SVP 3.50 Å,
     +2.82e-3 against −3.64e-3 leaves −8.28e-4. V's relative error is therefore
     SMALLER than S's at every one of the six points. The kernel is untouched:
     on NWChem's own determinants |V(RP)| at this point still matches to
     4.04e-11 Ha.

   Both bars remain far below the defects they guard: the E-only coupling form is
   a factor of 3 (1500× the new V bar), and a |S|²/λ² scaling error moves the
   slope by ≈2.5 Å⁻¹ (300× the new slope bar). Mutation 5 confirms this by
   measurement rather than argument.

6. `site/src/reference/validation.md` cDFT-ET row updated to current facts; the
   `#[ignore = "validation: cDFT-ET coupling (Wu–Van Voorhis)"]` attributes are
   unchanged on all five tests. Stale `KNOWN_NONCONVERGING` mentions in
   `run_basis`'s doc removed; the dated λ=0 experiment note kept as a historical
   record.

What was NOT needed, and why it was not done: re-seeding from the previous λ
(`CdftSeed` exists, and `lambda_newton_from`'s `run_inner` does close over ONE
immutable `guess` applied at every λ — so the outer loop does NOT re-seed per-λ,
confirming the issue's open question in the negative); a smaller level shift;
TRAH/Newton. None was required once the cap cleared the measured depth, and each
would have changed the SCF path on all six points for no measured gain. The
answer to the issue's §2 ordering is therefore: the first item sufficed.

`cdft_lambda_tol` was NOT loosened. The accepted root's residual is 9.36e-11,
genuinely below the 1e-10 tolerance — the solve now reaches the root instead of
being stopped short of it.

## 5. Results: all six points, before and after

The four previously-passing points are **bit-identical** at caps 100 and 150 —
same λ, N, E, outer count and deepest inner count (a cap is an upper bound, and
a solve finishing in 17–73 iterations cannot see it):

| point | λ (cap 100) | λ (cap 150) | E_a (cap 100) | E_a (cap 150) | outer | deepest inner |
|---|---|---|---|---|---:|---:|
| def2-SVP 2.50 | +2.321736640903 | +2.321736640903 | −4.856787855292 | −4.856787855292 | 3 | 17 |
| def2-SVP 3.00 | +2.405048345413 | +2.405048345413 | −4.865106758321 | −4.865106758321 | 4 | 73 |
| aug-cc-pVDZ 2.50 | +1.569217320956 | +1.569217320956 | −4.863290969900 | −4.863290969900 | 4 | 23 |
| aug-cc-pVDZ 3.00 | +1.556440654381 | +1.556440654381 | −4.869166397639 | −4.869166397639 | 4 | 22 |
| aug-cc-pVDZ 3.50 | +1.554270462514 | +1.554270462514 | −4.872239731000 | −4.872239731000 | 4 | 21 |
| **def2-SVP 3.50** | **(outer loop fails)** | **+2.457063436148** | **—** | **−4.869160282346** | **3** | **106** |

def2-SVP 3.50 Å against NWChem, full row:

| quantity | ferric | reference | error | bar |
|---|---|---|---|---|
| E_nuc | +6.047739553371e−1 | +6.047739553370e−1 | 1.43e−13 | 1e−9 |
| E_unc (99×302) | −4.990128759836 | −4.990128769001 | 9.17e−9 Ha | 1e−7 |
| E_A − E_unc (300×974) | +1.209684774896e−1 | +1.209684984060e−1 | **2.09e−8 Ha** | 2e−6 |
| λ_A (300×974) | +2.457063436148 | +2.457063813000 | **3.77e−7** | 3e−5 |
| population N_He1 | 0.999999999906 | 1.0 (target) | 9.4e−11 | 1e−9 |
| ⟨S²⟩ | 0.750000 | 0.75 (doublet) | — | — |
| MIRROR N_He2[B] vs N_He1[A] | +9.999999999082e−1 | +9.999999999082e−1 | 3.33e−16 | 5e−6 |
| \|S_AB\| | +3.853018491140e−4 | +3.842175533749e−4 | 2.82e−3 rel | 4e−3 |
| \|V(RP)\| on ferric's dets | +4.447243061099e−4 | +4.450929000000e−4 | 8.28e−4 rel | 2e−3 |
| coupling_hab vs textbook Wu–VV | — | — | ≤1e−18 | 1e−12 |

E_A − E_unc at 2.09e-8 Ha is the BEST of all six points, and λ at 3.77e-7 is the
second best — the two quantities that say "this is the right state" are the two
that agree best. The def2-SVP log-slopes, which could not be computed before
because the 3.50 Å point was skipped, now run:

| step | NWChem V | ferric direct | \|Δ\| | bar |
|---|---|---|---|---|
| 2.50→3.00 Å | −2.507893 | −2.507896 | 2.95e−6 | 8e−3 |
| 3.00→3.50 Å | −2.803192 | −2.804840 | 1.65e−3 | 8e−3 |

reproducing the module doc's independently-derived −2.508 / −2.803 reference
slopes.

## 6. Mutation ledger — each entry is an OBSERVED failure, with its counts

Every line below was run and its passed/failed/ignored counts read. The tests
are `#[ignore]`-gated, so each run passed `-- --ignored`; a line reporting
`0 passed` would prove nothing and none does.

| # | mutation | expected | OBSERVED |
|---:|---|---|---|
| 1 | `E2E_INNER_MAX_ITER` back to 100 (at the time a runtime assert) | fail | `test result: FAILED. 0 passed; 1 failed; 0 ignored` — panicked "cap 100 must converge: Convergence(\"cDFT outer loop did not converge in 8 iters\")" |
| 2 | cap = 106, exactly at the measured depth | fail | `FAILED. 0 passed; 1 failed; 0 ignored` — panicked "E2E_INNER_MAX_ITER = 106 is at or below the measured deepest inner solve (106)". (The solve itself still converged — outer 3, inner 68 — because the outer loop never revisits the 106-iteration λ; the assert guards the MARGIN, which is the point.) |
| 3 | negative control's `Ok` arm made non-asserting, AND the control's cap raised to 150 so that arm is reached | pass **vacuously**, proving the arm carries the control | `ok. 1 passed; 0 failed` with "MUTANT: cap 100 converged; control silently accepted" printed |
| 4 | pristine arms, control's cap raised to 150 (the complement of 3 — mutate the branch the test REACHES) | fail | `FAILED. 0 passed; 1 failed; 0 ignored` — panicked "cap 100 now CONVERGES (lam +2.457063436148 N 0.999999999906 outer 3). The negative control is gone…" |
| 5 | halve the transition-density exchange in `direct_coupling` (the defect the moved V/S bars guard) | fail | `FAILED. 2 passed; 3 failed; 0 ignored` — kernel H2(RP) missed by 9.18e-3 vs its 2e-10 bar, and BOTH end-to-end tests failed. The moved bars still catch it by ~7 orders of magnitude. |
| 6 | cap below the measured depth, after converting to `const _: () = assert!(…)` | fail to COMPILE | `build rc=101`, `error[E0080]: evaluation panicked: E2E_INNER_MAX_ITER is at or below the measured deepest inner solve (106)…` |

Mutations 3 and 4 are a matched pair, and the reason there are two: a mutation
that survives can mean a weak assertion OR an unreached branch, and those need
opposite fixes. 4 proves the branch is reachable and the assertion fires; 3
proves that with the assertion removed the same reachable branch goes silent.

## 7. Verification

| check | command | result |
|---|---|---|
| full row, both bases | `cargo test --release -p ferric-scf --test validation_cdft_et -- --ignored --nocapture --test-threads=1` | **ok. 5 passed; 0 failed; 0 ignored** (64 s) |
| trivial-limit anchors | (in the above) `wu_vv_coupling_is_invariant_under_constraint_offset`, rotation anchor `⟨A\|H\|AQ⟩ = det(Q)·E_HF` | pass — offset invariance \|dH\| ≤ 4.8e-18 (bar 1e-12), rotation \|d\| ≤ 2.7e-14 (bar 1e-11) |
| driver lib tests (incl. the synthetic plateau/root/cliff guards) | `cargo test --release -p ferric-scf --lib cdft` | ok. 17 passed; 0 failed |
| cDFT unit suites | `cargo test --release -p ferric-scf --test cdft_outer_loop --test cdft_uhf` | ok. 1 passed / ok. 6 passed; 0 failed |
| fmt | `cargo fmt --check -p ferric-scf` | clean |
| clippy | `cargo clippy -p ferric-scf --all-targets -- -D warnings` | rc 0 |
| rustdoc | `RUSTDOCFLAGS="-D warnings" cargo doc -p ferric-scf --no-deps` | rc 0 |

All runs through `scripts/validation/run_slot.sh`, `OPENBLAS_NUM_THREADS=1`,
default libint ERI precision (the #226 trap: not re-run at 1e-14).

## 8. Concerns and limits

- **The margin is 44 iterations, and the quantity it bounds is noisy.** §3.4
  shows the iteration count moving 69→181 under a 5e-13 change in λ, so "106" is
  one sample of a stiff distribution, not a hard ceiling. A future change to the
  guess, DIIS, the grid or ERI precision could push the deepest solve past 150
  and the point would fail again — in the same confusing way, since guard 1 would
  again read truncation as overshoot. The compile-time assert pins the
  *recorded* depth, not the true worst case.
- **The real fragility is in the driver, and I did not fix it.** Guard 1 cannot
  distinguish "the step went too far" from "the inner solve ran out of
  iterations", and those call for opposite responses: backtrack vs. iterate
  longer. `Sample` already carries `converged`; it does not carry *why*. A
  driver-side fix — re-running an iteration-capped solve from its own
  best-effort density before declaring the step bad, or distinguishing
  cap-truncation from divergence — would make the row robust instead of
  calibrated. That is a library change with its own validation surface, out of
  scope for this issue, and worth its own ticket.
- **The two moved bars were not predicted in §1.** They are a real consequence
  of running a point that had never been run, and §4.5/§5 give the structural
  reason (V ∝ S; flat absolute S offset; smallest |S| of the six) plus the
  mutation that shows they still bite. But they were moved AFTER seeing the
  data, which is weaker evidence than a bar derived in advance. The honest
  summary: the |S_AB| story (flat absolute offset) is well supported by six
  points; the |V(RP)| story (near-cancellation of dS/S against
  d(bracket)/bracket) is supported by the algebra and by V-rel < S-rel holding
  at all six points, but rests on three worked points, not six.
- **An absolute bar would be the better instrument for |S_AB|** than a relative
  one, given that the error is a fixed offset. I left it relative to avoid
  inventing a per-basis floor from one run; noted on the const.
- **Narrow scope.** One geometry, one basis, one functional. The row's grade
  remains "Proven (narrow)". Nothing here speaks to whether ferric's outer loop
  finds this state unaided from λ = 0 — it still does not, and the row still
  starts from NWChem's λ by design.
