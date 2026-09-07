# COSX L-axis measurement — results (2026-09-07, branch feat/cosx-seminumerical-k)

Pre-registration: `scripts/queue/out/l_axis_prereg.md` (commit 70585d37, written
and committed before any number below was taken). Harness:
`crates/ferric-scf/tests/cosx_l_axis.rs` (ignored test, one process per cell).

## Protocol actually followed

* def2-SVP (L_max=2) / def2-TZVP (3) / def2-QZVP (4), all bundled. Systems:
  butane (`alkane_4.xyz`, 14 atoms) and `alkane_8` (26 atoms).
* A-build: `ferric_integrals::cosx_a::a_matrix_at_point_with`, ONE reused
  libint2 nuclear engine, 500 grid points drawn uniformly at random without
  replacement (fixed-seed LCG, seed 20260907) from the full (50,110) Becke grid
  (77000 pts butane, 143000 pts alkane_8). Unscreened, and with the Stage 2
  shell-pair screen at 1e-7. "Per K build" = measured s/point x known point
  count (NOT an nbf extrapolation).
* Analytic K: `LinkK` (Schwarz bound, thresh 1e-12 = production
  `integral_thresh`) on a converged RHF density (DF-JK with
  def2-universal-jkfit; the density's provenance cancels out of every ratio
  because every builder contracts the same D). Warm build: engine pool
  already exists, `update_density` + `build` inside the timed region. Cold
  and warm agree to <1% at every cell (setup is negligible).
* DF-K: `DfK` (def2-universal-jkfit), warm `build` (density path) and warm
  `build_from_occ` (the path the SCF uses once MOs exist). Tensor size =
  naux*nbf^2*8 B computed exactly. Skipped where > 2.5 GB.
* EVERY timed segment: fresh 1-thread rayon pool, OPENBLAS_NUM_THREADS=1,
  cpu-seconds == wall-seconds to <1% (printed), /proc/pressure/memory
  `full avg10` printed before and after. Every number in the table below has
  0.00 before AND after; segments with nonzero PSI were discarded and rerun
  (one butane/QZVP DF-K cell and the SVP/TZVP direct-JK scope checks were
  affected by residual pressure from a cargo link and are excluded).
* alkane_8 and every QZVP cell ran under
  `scripts/ferric-limited --max=4G --high=3600M` with FERRIC_MEM_BUDGET_GB=2.
* Repeat check: butane/QZVP A-build was measured in three separate PSI-clean
  processes: 4.989 / 5.060 / 5.090 ms/pt unscreened (2% spread).

## Table

nbf: SVP/TZVP/QZVP butane 106/184/528, alkane_8 202/356/996. jkfit naux: 480 / 924.

| system   | basis     | L | nbf | A/pt unscr (ms) | A/pt scr 1e-7 (ms) | kept frac | A per K build unscr / scr (s) | LinK K warm, 1 thr (s) | DF-K occ / dens / setup (s) | 3-index (GB) | A/LinK unscr / scr | A/DF-K(occ) unscr | A/DF-K(dens) unscr | K/nbf^2 (LinK, s) | A/pt/nbf^2 unscr (s) | PSI |
|----------|-----------|---|-----|-----------------|--------------------|-----------|-------------------------------|------------------------|------------------------------|--------------|--------------------|-------------------|--------------------|-------------------|----------------------|-----|
| butane   | def2-SVP  | 2 | 106 | 0.438 | 0.411 | 0.946 | 33.7 / 31.7   | 1.003 | 0.021 / 0.075 / 0.456 | 0.043 | **33.6x** / 31.6x | 1580x | 451x  | 8.93e-5 | 3.90e-8 | 0.00 |
| butane   | def2-TZVP | 3 | 184 | 1.026 | 0.938 | 0.916 | 79.0 / 72.3   | 6.482 | 0.050 / 0.334 / 1.126 | 0.130 | **12.2x** / 11.1x | 1572x | 237x  | 1.92e-4 | 3.03e-8 | 0.00 |
| butane   | def2-QZVP | 4 | 528 | 5.090 | 4.385 | 0.873 | 391.9 / 337.6 | 259.4 | 0.430 / 7.070 / 9.737 | 1.071 | **1.51x** / 1.30x | 912x  | 55.4x | 9.31e-4 | 1.83e-8 | 0.00 |
| alkane_8 | def2-SVP  | 2 | 202 | 1.500 | 1.238 | 0.836 | 214.5 / 177.0 | 5.069 | 0.162 / 0.923 / 4.267 | 0.302 | **42.3x** / 34.9x | 1323x | 232x  | 1.24e-4 | 3.68e-8 | 0.00 |
| alkane_8 | def2-TZVP | 3 | 356 | 3.803 | 2.742 | 0.795 | 543.8 / 392.0 | 37.87 | 0.526 / 4.240 / 14.23 | 0.937 | **14.4x** / 10.4x | 1034x | 128x  | 2.99e-4 | 3.00e-8 | 0.00 |
| alkane_8 | def2-QZVP | 4 | 996 | 17.95 | 12.27 | 0.709 | 2567 / 1755   | not measured (see below) | SKIPPED: tensor > cap | **7.333** | n/a | n/a | n/a | n/a | 1.81e-8 | 0.00 |

Denominator scope check (butane, 1 thread): ferric's DEFAULT builder, direct
`build_jk` (J AND K in one sweep, an upper bound on direct K alone), took
400.5 s at QZVP (PSI 0.00) vs LinK's 259.4 s. LinK is the FASTER analytic K
at every L measured (SVP 1.38 vs 1.01 s, TZVP 9.6 vs 6.5 s — those two carried
residual PSI and are qualitative only). Using LinK is therefore the
CONSERVATIVE choice; against direct K the QZVP ratio would be <= 0.98x.

Thread-inflation calibration (not used for any table number): butane/QZVP LinK
on 8 threads = 46.4 s wall, 345.5 cpu-s, vs 259.4 s on 1 thread -> 1.33x
CPU inflation from contention on this box. Recorded so a future multi-thread
timing can be interpreted.

## The trend, in one sentence

A-build / LinK-K falls 33.6x -> 12.2x -> 1.51x on butane (42.3x -> 14.4x on
alkane_8 for the two bases that fit), i.e. ~22x from SVP to QZVP, and the two
per-nbf^2 growth rates explain it exactly: analytic K per nbf^2 RISES 10.4x
(8.9e-5 -> 1.9e-4 -> 9.3e-4 s) while A-build per point per nbf^2 FALLS 2.1x
(3.9e-8 -> 3.0e-8 -> 1.8e-8 s), and the latter is molecule-independent
(alkane_8/QZVP 1.81e-8 vs butane/QZVP 1.83e-8).

## Pre-registered hypotheses vs data

* H_phys (ratio FALLS monotonically SVP->TZVP->QZVP by a large factor, with
  K/nbf^2 rising steeply while A/pt/nbf^2 stays ~flat): MATCHED on every
  count, on both systems, with the A/pt/nbf^2 actually falling rather than
  flat. The design study's own butane cc-pVXZ numbers implied ~5x TZ->QZ;
  measured here it is 8.1x TZVP->QZVP (12.2 -> 1.51).
* H_art (flat or rising): NOT matched.
* Decision rule fixed in the prereg: "if the QZVP ratio vs LinK is
  single-digit, the earlier closure was measured in the wrong regime and the
  engine question REOPENS". Measured 1.51x (1.30x screened): single-digit by
  a wide margin.

## Verdict (dated, provisional as always)

The L-axis REOPENS the engine question. Every prior deficit in §35 of
wiki/amplitude-threshold-lmp2.md was taken at DZ/TZ against an nbf^2.5
EXTRAPOLATED analytic-K denominator; with a MEASURED LinK denominator the
deficit is already 34-42x at SVP and 11-14x at TZVP (not 126-219x), and at
QZVP — libint2 as-is, no kernel work, no overlap fit, the honest (50,110)
grid, unscreened — the A-build costs 1.5x one LinK K build (1.3x with the
existing 1e-7 screen, 0.98x against ferric's default direct K). The
"do-not-declare-a-negative-below-the-onset" error applied: the onset was
basis-set L, not molecular size.

What this does NOT say:
* It is not a COSX K timing. A full COSX K also needs the AO-values-on-grid
  and the two contraction GEMMs (X D, then A^g-contractions); those are
  O(npts*nbf*nocc)-class BLAS3 and were not measured here. It also omits the
  unmeasured overlap-fit / grid-pruning factors production codes use, in
  BOTH directions (the retraction found the fit is not free).
* alkane_8/QZVP analytic K was NOT measured: producing a converged density
  needs either DF-JK (7.3 GB tensor; the spilled attempt under the 4G
  cgroup drove system PSI full avg10 to 51 and was killed) or ~10 direct-K
  iterations at ~4 min each on 8 threads, neither fitting the box rules.
  Its A/pt/nbf^2 equals butane's, and both alkane_8 ratios that were
  measured sit ~20% above butane's, so the QZVP ratio is expected near 1.8x
  — EXPECTED, not measured.
* vs DF-K, COSX loses on TIME at every L where the tensor fits (900-1600x
  slower than the occ path; 55-450x slower than the density path). The
  COSX-vs-RI-K case is MEMORY, and it is real: alkane_8/QZVP's dressed
  tensor is 7.33 GB, above this box's free RAM, and it made the DF-JK SCF
  unrunnable here; COSX's resident state is O(nbf^2) per point.
* Observation, already explained by existing tests: max|K_LinK - K_direct|
  = 2.8e-3 (butane/SVP, thresh 1e-12) and equals max|K_LinK - K_DF-K| to
  four digits, i.e. LinK — not DF-K — is the outlier on this density. This
  is the KNOWN LinK pair-list deviation measured in
  `tests/screening_exactness.rs` (max|K_LinK - K_dense| = 1.3e-3 at
  thresh 1e-12 on alkane_8/cc-pVDZ, after the Schwarz-table fix), not a
  new defect, and it cannot change this verdict: the exact direct K is
  SLOWER than LinK, which only lowers the ratio.

Recommended next step, if pursued: re-run the §35 FINAL-VERDICT budget with
this measured denominator at QZVP (and TZVP), then decide whether the MD
3c1e kernel (<= 3.8-7.1x estimated) is even needed at L=4 — at 1.3-1.5x it
may not be. The reopen condition is now met by the numbers, not by argument.
