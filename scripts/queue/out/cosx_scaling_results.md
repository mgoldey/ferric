# COSX A-build wall vs N with the density-driven screen — results

Date: 2026-09-07. Branch `feat/cosx-s3-swap-screen` (screen commit be567326,
pre-registration 743aafeb, committed before any number below).
Pre-registration: `cosx_scaling_prereg.md` (same directory).

## Protocol as run

* Harness `crates/ferric-scf/tests/cosx_full_k.rs`, release, ONE thread for
  every timed segment (`OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1`, and the
  harness additionally runs each timed segment inside an explicit 1-thread
  rayon pool); cpu_s == wall_s to <= 0.1% on every row below.
* `scripts/ferric-limited --max=4G --high=3600M`, `FERRIC_MEM_BUDGET_GB=2`,
  foreground under `timeout` <= 480 s, one process per timed cell. K timings
  run ALONE on the box (no concurrent job of mine).
* PSI `full avg10` printed before/after every segment. Every row below is
  0.00/0.00 except where flagged. The C20 COSX build was taken three times:
  167.95 s (before 0.05 / after 0.09), 167.09 s (0.81 / 0.00), 168.49 s
  (0.00 / 0.00, KEPT) — the nonzero readings were external transients on the
  shared box (this process is ~50 MB resident at C20; cpu == wall to 0.03% in
  all three); agreement across takes 0.8%.
* COSX: `CosxK`, md3c1e backend, (50,110) grid, overlap fit ON, DENSITY-DRIVEN
  screen at the default `t = 1e-7`. "A-build" = kernel time (screen setup +
  3c1e blocks), "GEMMs" = the dense block products `F = D X`, `Ktilde += X G^T`
  (+ `S_num = X X^T` on this first build). `kept_dd` = density-driven kept
  (pair, sub-batch) fraction; `kept_geom` = what the geometry-only bound alone
  would keep.
* LinK: `LinkK` Schwarz 1e-12; cold (pool + pairs + build) and warm
  (`update_density` + `build`); the WARM wall is the denominator.
* Densities: C4/C8/C12 from a DF-JK (def2-universal-jkfit) SCF at default
  convergence (density_conv 1e-6); C16/C20 from the plain direct J+K SCF
  chained across two foreground windows via `init_guess_density`
  (density_conv 1e-5; final dp_rms 3.5e-6 and 1.1e-6). The DF-JK SCF no
  longer fits the 3.6 GB cgroup at nbf >= 394 (two ~2.2 GB tensors + spill
  page cache -> reclaim storm, PSI 69), and the `k_builder = "link"` SCF path
  is BROKEN (see "Incidental findings"). The density's provenance cancels out
  of every ratio (both builders contract the same D); its convergence level
  affects the kept fraction only through F values at the 1e-5 level.

## def2-SVP series (one thread)

| system | natoms | nsh | nbf | npts | COSX total s | A-build s (%) | GEMMs s | kept_dd | kept_geom | LinK cold s | LinK warm s | COSX/LinK | A-build/LinK |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alkane_4  | 14 | 54  | 106 | 77000  | 7.653   | 6.786 (88.7)  | 0.162  | 0.7916 | 0.9464 | 1.006  | 1.012  | 7.57 | 6.71 |
| alkane_8  | 26 | 102 | 202 | 143000 | 28.373  | 24.343 (85.8) | 1.425  | 0.4862 | 0.7351 | 5.058  | 5.099  | 5.56 | 4.77 |
| alkane_12 | 38 | 150 | 298 | 209000 | 58.694  | 47.092 (80.2) | 5.926  | 0.3071 | 0.5563 | 10.175 | 10.059 | 5.84 | 4.68 |
| alkane_16 | 50 | 198 | 394 | 275000 | 107.664 | 75.479 (70.1) | 22.589 | 0.2059 | 0.4403 | 15.741 | 15.645 | 6.88 | 4.82 |
| alkane_20 | 62 | 246 | 490 | 341000 | 168.494 | 99.557 (59.1) | 54.622 | 0.1459 | 0.3620 | 21.347 | 21.384 | 7.88 | 4.66 |

Other split columns (s): ao_eval 0.39 / 1.36 / 2.97 / 5.32 / 8.31; fold
(`G += A F`) 0.29 / 1.14 / 2.48 / 3.84 / 5.28; fit finalize <= 0.04.
`max|K_cosx - K_link|` 2.8e-3 .. 3.3e-3 at every size — this is LinK's
deviation from the direct K, not COSX's (see "Incidental findings").

### Tail exponents (log-log least squares over the LAST THREE sizes, N = atoms)

| quantity | tail exponent (C12,C16,C20) | pairwise C4->C8->C12->C16->C20 |
|---|---|---|
| COSX A-build wall | **1.54** | 2.06, 1.74, 1.72, 1.29 |
| COSX total wall | **2.16** | 2.12, 1.92, 2.21, 2.08 |
| COSX block GEMMs | 4.55 | 3.51, 3.76, 4.88, 4.10 |
| LinK warm wall | **1.54** | 2.61, 1.79, 1.61, 1.45 |

Kept pairs per sub-batch (`kept_dd x nsh(nsh+1)/2`): 1176, 2554, 3478, 4056,
4433 — pairwise exponents 1.25, 0.81, 0.56, 0.41, i.e. bending toward the
saturation the O(N) argument predicts but NOT there yet at 62 atoms (~50 Bohr).
The kept FRACTION falls as N^-1.5 on the tail (0.307 -> 0.146).

## Reading against the pre-registration

* Physics hypothesis ("tail exponent 1.3-1.7, kept fraction falling
  monotonically and well below geometry-only"): MET. A-build tail 1.54, the
  last pairwise step 1.29; kept_dd falls monotonically 0.79 -> 0.15 and is
  0.40-0.58x the geometry-only fraction at every size. The density-driven
  screen is doing what it was built to do; it is pre-asymptotic at C20, as
  predicted (the C16->C20 step is below the tail fit).
* Artifact hypothesis ("~2.0 with a flat kept fraction"): NOT observed.
  Too-clean check: exponents are not integers and the pairwise series has the
  expected bend (2.06 -> 1.29), not a constant.
* Full-K hypothesis ("dense GEMMs make the total exceed the A-build, 1.6-2.2"):
  MET, at the top of the predicted range — total tail 2.16. The A-build
  fraction falls 88.7% -> 59.1%; at C20 the dense half-transform GEMMs are
  54.6 s of 168.5 s and their cost is growing faster than N^3 (the measured
  GEMM rate drops 19 -> 9 GFlop/s from nbf 298 to 490 at one thread on the
  (nbf x 1024) planes). THIS is the next bottleneck for a 320-atom target: a
  sparse / AO-local half transform (`F = D X` with D and X both local), not the
  screen.
* LinK hypothesis ("1.3-1.5"): slightly above — tail 1.54 with pairwise 1.61,
  1.45 (the earlier 1.36 was C8->C16 on a different basis of N; on THIS series
  C8->C16 in atoms gives 1.71).

## Crossover statement

At def2-SVP COSX is SLOWER than LinK at every size: 7.6x (C4), 5.6x (C8),
5.8x, 6.9x, 7.9x (C20). The ratio has a minimum near C8 and RISES on the tail
because the dense GEMMs (N^2.16 total vs LinK's N^1.54) dominate; the A-build
alone sits at a flat 4.7-4.8x of LinK from C8 on (its tail exponent equals
LinK's, 1.54 vs 1.54, with the last step 1.29 vs 1.45). No crossover in
COSX's favour exists or is approached at SVP — as pre-registered, SVP is the
low-L end where seminumerical exchange loses; the 0.536x win is a QZVP
result (butane). What the series DOES establish: with the density-driven
screen the A-build is sub-quadratic (1.54 -> 1.29 and bending), so the
"O(N^2) A-build inverts the QZVP advantage around C10" concern from the
integral-only screen no longer holds for the A-build; the inversion risk has
moved to the dense half transform.

## def2-TZVP (C4, C8; one thread; C12 not attempted)

Densities from the DF-JK SCF (default convergence; fits the cgroup at these
sizes). PSI 0.00/0.00 on every segment, cpu == wall.

| system | natoms | nbf | npts | COSX total s | A-build s (%) | GEMMs s | kept_dd | kept_geom | LinK warm s | COSX/LinK |
|---|---|---|---|---|---|---|---|---|---|---|
| alkane_4 | 14 | 184 | 77000  | 17.453 | 15.435 (88.4) | 0.469 | 0.7597 | 0.9233 | 6.724  | 2.60 |
| alkane_8 | 26 | 356 | 143000 | 71.725 | 61.600 (85.9) | 3.612 | 0.4689 | 0.7172 | 39.368 | 1.82 |

Two points only (pre-onset; no tail fit): C4->C8 pairwise exponents in atoms
are COSX total 2.28, A-build 2.24, LinK warm 2.85. The COSX/LinK ratio FALLS
with N at TZVP (2.60 -> 1.82) where it rises at SVP, consistent with the
L-axis picture (LinK's quartet cost grows faster with L than the 3c1e blocks;
COSX wins outright at QZVP). alkane_12/def2-TZVP (nbf 528) was not attempted:
its DF-JK tensors (~2.5 GB each) do not fit the 3.6 GB cgroup and a one-thread
direct SCF at that size is hours of chained windows; nothing is extrapolated
from the two TZVP points.

## Incidental findings (not fixed here; for the coordinator)

1. `solve_rhf` with `k_builder = "link"` is numerically WRONG on this branch:
   butane/def2-SVP goes to E = -162.76 Ha (direct -157.186), alkane_16 to
   -12082 Ha, non-variational and non-convergent; bit-identical at 1 and 2
   rayon threads and with the default vs 2 GB budget. Iteration 1
   (`LinkK::build(D)`) agrees with the COSX-K SCF to 1.4e-3 Ha; the runs
   diverge from iteration 2, where the SCF switches to `build_from_occ(C_occ)`
   and scales by 2 — hypothesis: `LinkK::build_from_occ` does not follow the
   `K(D) = 2 K(C_occ C_occ^T)` convention that `CosxK` (anchored) follows.
   The COSX-K SCF on the same path converges to -157.18597 (dE +1.7e-4 vs
   direct, the grid error), so DirectJ and the pluggable-K plumbing are fine.
2. Standalone `LinkK` (thresh 1e-12) differs from the direct K by 2.784e-3
   (max element) on butane/def2-SVP, at 1 and 2 threads
   (`cosx_l_axis` `COSX_L_DIRECT_JK=1`). COSX's own error vs the direct K on
   the same density is 2.9e-4. Every COSX-vs-LinK ratio above is therefore
   against a LinK that is itself 10x further from the exact K than COSX is.
3. The DF-JK SCF at nbf >= ~400 exceeds a 3.6 GB cgroup and its budget-bounded
   spill turns into a reclaim storm (PSI 69, the SCF did not finish in 400 s).
   The direct SCF chained across windows (`COSX_FK_SCF=direct`,
   `COSX_FK_SCF_MAXITER`, `COSX_FK_SCF_RESTART_IN`) is the working route.
