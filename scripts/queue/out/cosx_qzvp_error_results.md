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

<!-- QZVP numbers appended below as cells complete -->
