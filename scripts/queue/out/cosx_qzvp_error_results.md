# Results: COSX accuracy at def2-QZVP (butane)

Pre-registration: `scripts/queue/out/cosx_qzvp_error_prereg.md` (committed
2026-09-08, before any number below).
Branch `meas/cosx-qzvp-error`, worktree `.claude/worktrees/cosx`.

System: butane, `testdata/molecules/alkane_4.xyz`, 14 atoms.
def2-QZVP: **nbf = 528, nocc = 17, L_max = 4** (matches the whitepaper's QZVP cell).

Protocol: one thread throughout (`OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1`),
runs under `scripts/ferric-limited --max=6G --high=5G`,
`/proc/pressure/memory` `full avg10` checked before and after every cell,
cpu/wall asserted ~1.

## Reference builder

`K_exact` = the direct four-centre `ferric_scf::rhf::build_jk` at
`integral_thresh = 1e-12`. **Not LinK** — see the prereg's "Reference choice":
standalone LinK on this branch is ~10x further from exact than COSX is, so a
COSX-vs-LinK deviation measures LinK. A `COSX_FK_DIRECT` block was added to
`crates/ferric-scf/tests/cosx_full_k.rs` for this lane, together with a
`COSX_FK_GRID` override so the grid arms run on ONE density in ONE process
(prereg artifact A3).

## Finding 0 (before any QZVP number): the requested (75,194) grid does not exist

`ferric_scf::cosx_k::validate_grid` rejects angular order 194:

```
cosx grid: angular order 194 is not tabulated (supported: [6, 14, 26, 50, 110, 302])
```

COSX's Lebedev table carries **[6, 14, 26, 50, 110, 302]** only. The task asked
for (75,194); 194 is a standard Lebedev order elsewhere in ferric
(`ferric_quadrature::lebedev`) but is not in COSX's own supported set. The
fine-grid arm is therefore run at **(75,302)** — the next tabulated order up —
and every "fine grid" figure below means (75,302), not (75,194). This is a
substitution, not a silent default: the knob hard-errors, per the repo's
config-honesty convention.

## Instrument validation (water / cc-pVDZ, before trusting it at QZVP)

Both new knobs were mutation-tested on the cheap system, since a knob that
silently does nothing is prereg artifact A5, and a test never seen to fail is
an assumption.

`max|K_cosx - K_direct|`, `max|K_direct| = 9.767e0`, fit ON:

| grid | points | max abs dev | relative | ‖dK‖_F | rel ‖·‖_F |
|---|---|---|---|---|---|
| (25,50)  |  3 750 | 9.826e-4 | 1.006e-4 | 4.317e-3 | 2.962e-4 |
| (50,110) | 16 500 | 6.108e-5 | 6.254e-6 | 2.029e-4 | 1.392e-5 |
| (75,302) | 67 950 | 2.696e-7 | 2.760e-8 | 6.366e-7 | 4.368e-8 |

Monotone over four orders of magnitude with point count: the grid knob is live
and the metric responds. The (50,110) relative error 6.3e-6 is consistent in
magnitude with the +4.9e-6 Ha SCF error already on record for water/cc-pVDZ,
which is the independent cross-check that the instrument measures what it
claims.

Fit knob, water/cc-pVDZ at (50,110):

| fit | max abs dev | ‖dK‖_F |
|---|---|---|
| ON  | 6.108e-5 | 2.029e-4 |
| OFF | 4.939e-5 | 2.873e-4 |

The fit knob is live (the numbers move). Note already that its effect is
**metric-dependent even on water**: fit-ON is better in the Frobenius norm
(2.03e-4 vs 2.87e-4) but *worse* in the max element (6.11e-5 vs 4.94e-5). Any
single-number "the fit helps Nx" claim is therefore norm-dependent, which is
worth carrying into how the paper words the fit's benefit.

## Cost structure discovered while setting up (affects what is feasible)

With `k_builder = "cosx"`, `rhf.rs:698` builds J with `DirectJ` — the **exact
four-centre Coulomb** — and `rhf.rs:741` scopes the incremental (differential)
Fock optimization strictly to the `DirectJK` path. So the COSX arm pays a full
non-incremental exact J every iteration while the direct arm gets incremental
Fock. At butane/def2-QZVP this makes the COSX SCF's wall time dominated by J,
not by K.

Measured: the (25,50) COSX SCF ran ~11 min to first iteration and ~6 min/iter
steady state at nbf=528, i.e. hours per SCF arm. This does not change any
ENERGY reported here, only which arms fit in the available windows; it is
recorded because it makes the whitepaper's COSX-vs-direct SCF *timing*
comparison an unfair one in COSX's disfavour, independently of accuracy.

## def2-TZVP anchor (butane, same molecule, same instrument)

Run first, because the whitepaper already carries a converged TZVP SCF error
(-1.2e-4 Ha) for this exact system: it is the independent check that this
instrument reproduces a known regime before it is pointed at the unmeasured one.

Reference SCF (`build_jk`, exact four-centre, `density_conv` 1e-6):
**E = -157.35328543 Ha, converged, 10 iterations, 97.9 s** (nbf = 184).
All four cells below contract that ONE density (prereg artifact A3), one thread,
PSI `full avg10` = 0.00 before and after all four, cpu/wall = 1.00 on every cell.

`max|K_direct| = 7.827479e0`, `‖K_direct‖_F = 3.966501e1`.

| grid | points | fit | max abs dev | relative | ‖dK‖_F | rel ‖·‖_F |
|---|---|---|---|---|---|---|
| (25,50)  |  17 500 | ON  | 3.202e-3 | 4.091e-4 | 3.208e-2 | 8.087e-4 |
| (50,110) |  77 000 | ON  | 2.483e-4 | 3.172e-5 | 2.829e-3 | 7.132e-5 |
| (75,302) | 317 100 | ON  | 8.782e-6 | 1.122e-6 | 1.322e-4 | 3.334e-6 |
| (50,110) |  77 000 | OFF | 1.268e-3 | 1.620e-4 | 1.520e-2 | 3.832e-4 |

Two results at TZVP:

* **The grid ladder is monotone and steep**: each refinement step buys ~13x
  (3.20e-3 -> 2.48e-4 -> 8.78e-6 in max abs). (50,110) is NOT converged at TZVP
  — (75,302) is another 28x better.
* **The overlap fit helps decisively, and it helps on a hydrocarbon chain**:
  at (50,110), fit-OFF is 1.268e-3 against fit-ON 2.483e-4, i.e. the fit is
  worth **5.1x** in max abs and **5.4x** in Frobenius. This REFUTES the specific
  worry registered in prereg P4 (that the ethane-like 0.5-0.9x behaviour might
  make the fit neutral-or-harmful on butane). Butane is not ethane-like here.

Sanity cross-check against the record: the K error at the production setting
(50,110)+fit is 2.48e-4 absolute / 3.17e-5 relative, and the whitepaper's
converged TZVP SCF error for this system is -1.2e-4 Ha. Same order of
magnitude, as it should be — an energy error and a max-element K error are not
the same quantity, but a 3e-5 relative K error producing a ~1e-4 Ha energy
error is coherent. The instrument is not producing a number from nowhere.

## THE MEASUREMENT: def2-QZVP K error (butane, nbf = 528)

### How the converged QZVP density was obtained (and what did NOT work)

The exact four-centre direct SCF at QZVP is **not a viable route**: it reached
only iteration 2 in ~25 minutes of single-threaded wall time before being
killed, consistent with the (528/184)^4 ~ 68x per-iteration cost over TZVP on
the analytic quartet path.

**DF-JK converged the same system in 26.3 s / 11 iterations**
(`E = -157.36421074 Ha`, converged=true, def2-universal-jkfit). That density was
saved ONCE (`dens_qzvp_butane.bin`) and every cell below contracts it, so no
comparison is across differing densities (prereg artifact A3) and nothing was
converged twice.

The density's provenance is DF-JK rather than exact-direct. This is the right
tradeoff and it does not bias the K comparison: **both** builders in each row
contract the *same* D, so the reported deviation is a property of the COSX
quadrature, not of the density. A different D would shift both K matrices
together.

### Results — one thread, PSI `full avg10` = 0.00 before AND after every cell, cpu ~ wall

`max|K_direct| = 7.882133e0`, `‖K_direct‖_F = 6.480234e1`.

| grid | points | fit | max abs dev | **relative** | ‖dK‖_F | rel ‖·‖_F | COSX build | direct build |
|---|---|---|---|---|---|---|---|---|
| (25,50)  |  17 500 | ON  | 3.140e-3 | 3.983e-4 | 5.319e-2 | 8.208e-4 | — | 423.7 s |
| **(50,110)** | **77 000** | **ON** | **4.247e-4** | **5.389e-5** | **7.806e-3** | **1.205e-4** | **89.95 s** | **420.7 s** |
| (75,302) | 317 100 | ON  | 1.815e-5 | 2.302e-6 | 2.583e-4 | 3.986e-6 | 337.69 s | 508.2 s |
| (50,110) |  77 000 | OFF | 2.497e-3 | 3.168e-4 | 4.523e-2 | 6.980e-4 | — | 455.8 s |

**The production-configuration answer (the number the whitepaper was missing):**
at butane/def2-QZVP, COSX at (50,110) with the overlap fit deviates from the
exact four-centre K by **max 4.247e-4 absolute, 5.39e-5 relative**
(Frobenius 7.806e-3 absolute, 1.21e-4 relative).

### Basis progression, same molecule, same instrument, production setting

| basis | nbf | L_max | max abs dev | relative dev |
|---|---|---|---|---|
| def2-TZVP | 184 | 3 | 2.483e-4 | 3.172e-5 |
| def2-QZVP | 528 | 4 | 4.247e-4 | 5.389e-5 |

**Prereg P1 outcome: CORRECT but at the very bottom of the predicted range.**
P1 predicted 1e-4..3e-3 "in the upper half". The measured 4.25e-4 is in the
range but in the LOWER half. The error grows TZVP -> QZVP by only **1.71x in
absolute** and **1.70x in relative** terms — a mild degradation, not the
"fixed grid can't keep up with rising L" collapse the reasoning implied. The
prediction's direction was right; its magnitude was pessimistic, and the honest
reading is that the reasoning behind it was only weakly confirmed.

### Grid sensitivity at QZVP (prereg P3)

Refinement factor per step at QZVP: (25,50) -> (50,110) is **7.4x**,
(50,110) -> (75,302) is a further **23.4x**. The ladder is monotone and steep,
i.e. **(50,110) is NOT a converged grid at quadruple zeta** — there is 23x of
accuracy still on the table.

The same ladder at TZVP is 12.9x then 28.3x. So the grid's *marginal value*
is comparable at TZ and QZ; QZVP is not qualitatively harder to integrate.

### The overlap fit at QZVP (prereg P4)

| basis | fit ON | fit OFF | fit benefit |
|---|---|---|---|
| def2-TZVP | 2.483e-4 | 1.268e-3 | **5.1x** |
| def2-QZVP | 4.247e-4 | 2.497e-3 | **5.9x** |

**Prereg P4 outcome: the prediction was right and the worry was WRONG.** P4
predicted the fit still helps by ~2-10x, and flagged a real possibility that it
would HURT on a hydrocarbon chain at high L (from the known 0.5-0.9x ethane
behaviour). Measured: the fit helps by 5.9x at QZVP, slightly MORE than its
5.1x at TZVP. The fit is not water-specific in this regime, and the
`overlap_fit = true` default is **vindicated at quadruple zeta**.

Note the fit costs essentially nothing: `fit 0.024 s` out of an 89.95 s build.

### An incidental cost result (same density, same thread, exact same comparison)

The COSX K build at the production grid is **89.95 s** against the exact direct
K's **420.74 s** on the identical density and one thread — a **4.68x** speedup
at a 5.4e-5 relative K error. At (75,302) COSX is 337.69 s vs 508.2 s, only
1.50x, so the accuracy gained by refining the grid mostly spends the advantage.

## The SCF energy arm: NOT MEASURED at QZVP, and why

**Deliverable 2 (SCF dE and iteration counts vs direct at QZVP) was not
obtained.** Reporting that plainly, rather than substituting a number from a
configuration that does not mean what it appears to mean.

Two routes were tried and both are genuinely blocked:

**Route 1 — COSX SCF against a direct SCF (the exact comparison asked for).**
With `k_builder = "cosx"`, `rhf.rs:698` builds J with `DirectJ`, the exact
four-centre Coulomb, and `rhf.rs:741` scopes the incremental (differential) Fock
optimization strictly to the `DirectJK` path. So the COSX arm pays a full,
non-incremental, exact J every iteration. At nbf = 528 this dominates: the run
was killed by its own 1150 s bound (`EXIT=143`) without converging, and the
plain direct SCF reached only iteration 2 in ~25 minutes. PSI stayed 0.00
throughout both, so this is a real cost limit, not a contaminated measurement.
An SCF here is a multi-hour job per arm, times five arms.

**Route 2 — hold J fixed with DF-J so the arms differ only in K. REJECTED as
an invalid configuration; it was caught by the pre-registration's own artifact
check.** The TZVP validation of this route returned:

```
dfj_direct       E = -157.3534052742  converged  51 iters
dfj_cosx_50_110  E = -157.3534052742  converged  51 iters
```

Bit-identical energies AND identical iteration counts. Prereg artifacts A5/A6
say verbatim that identical energies across arms mean the knob did not reach the
builder — broken configuration, not a null result. `df_j_aux` +
`k_builder = "cosx"` runs a J-density-fitted SCF whose **exchange is the direct
4-centre builder, with no COSX at all**, so the "COSX" arm and the "direct" arm
were the same calculation.

**Correction to my own first reading — ferric is NOT at fault here.**
`fock_assembly::resolve_k_builder` (lines 192-201) explicitly warns on this
combination, and the warning does fire:

```
[ferric] warning: k_builder = "cosx" is IGNORED because density-fitted J/K is
active (df_j_aux/df_k_aux set, or auto-defaulted for a functional); exchange
comes from the direct 4-centre builder
```

I missed it because my command filtered stdout through `grep` and the warning
goes to **stderr**. The config layer behaved correctly and told me exactly what
it was doing; the measurement error was mine. (`rhf.rs`'s in-source comment does
say "silently ignored", which is now stale relative to the code — the resolver
below it warns. Worth a one-line comment fix, nothing more.)

The lesson is a measurement-hygiene one already in this repo's memory —
*grep-filtered output hides failures* — and it is why the cheap TZVP validation
ran before the expensive QZVP arm. Had this route been run only at QZVP without
that check, it would have produced a converged pair of energies differing by ~0
and the natural, completely wrong conclusion "COSX is exact at QZVP".

### What stands in place of the SCF dE

The K-matrix error IS the underlying quantity, measured at QZVP on a converged
density, and it is the more direct probe of the approximation: an SCF dE is that
same error propagated through a self-consistent loop. The relative K deviation
at the production setting is 5.39e-5, and the TZVP row of the same table
(3.17e-5) sits next to a whitepaper SCF error of -1.2e-4 Ha — so a QZVP SCF
error of *order* 1e-4 Ha is the reasonable inference. **That is an inference,
not a measurement, and it is not offered as a number for the paper.**

One genuine data point does exist. Both QZVP SCFs start from the same core
guess, so iteration 1 is a pure K comparison at fixed density:

| | iteration-1 E (Ha) |
|---|---|
| direct           | -114.3388621771 |
| COSX (25,50)+fit | -114.3385841358 |
| difference       | **+2.780e-4** |

Consistent in magnitude with the (25,50) K error of 3.14e-3 on a guess density
far from convergence. Recorded for completeness; it is not a converged SCF
error and must not be quoted as one.

## Recommendation on the default at quadruple zeta

**Keep `overlap_fit = true`. It is clearly right at QZVP** — worth 5.9x, at a
cost of 0.024 s in a 90 s build. The prereg's concern that it might be
water-specific and fail on hydrocarbon chains is refuted for butane at both TZ
and QZ.

**(50,110) is defensible as a default but it is NOT converged at QZVP**, and the
paper should say so rather than imply the default is a settled choice:

* At the production setting COSX's relative K error at QZVP is 5.4e-5, and the
  basis progression TZVP -> QZVP is only 1.71x. The default does not *degrade*
  much going to quadruple zeta — the "fixed grid falls apart at high L" worry
  is not what the data shows.
* But (75,302) is 23.4x more accurate, so users who need tight exchange at QZ
  have a real and effective knob, and (50,110) is a speed/accuracy compromise
  rather than a converged answer.
* The cost of that accuracy is most of the advantage: COSX is 4.68x faster than
  the exact K at (50,110), but only 1.50x at (75,302).

**The most important wording change for the paper**: §3.1's "1.51x" and §3.3's
"137.04 s" should not stand alone. The honest sentence is that at
butane/def2-QZVP, COSX at (50,110)+fit builds K in 89.95 s against the exact
direct K's 420.74 s on the same density and one thread — **4.68x — at a
relative K error of 5.4e-5 (max element 4.2e-4 against max|K| = 7.9)**. A cost
ratio without the co-reported error is not a claim a reader can evaluate, and
that error is now measured rather than assumed.

## Scope and honesty limits

* ONE molecule (butane), a saturated hydrocarbon: no lone pairs, no
  heteroatoms, no diffuse functions. Not a QZVP-wide verdict.
* The density is DF-JK-converged, not exact-direct-converged. This does not bias
  the K comparison (both builders contract the same D) but it is not the exact
  ground state.
* The requested (75,194) does not exist in COSX; (75,302) was used. The two are
  not interchangeable — 302 is a finer angular order, so the fine-grid arm is
  *more* refined than the one requested.
* The SCF dE at QZVP is **not measured**. Do not let the K-error table be read
  as if it were.
* The COSX-vs-direct K timings above are honest single-thread same-density
  comparisons, but they were not the object of this lane and were not repeated;
  treat them as one measurement each, not as a benchmark.
