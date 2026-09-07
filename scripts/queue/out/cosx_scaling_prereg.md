# Pre-registration: COSX A-build wall vs N with the density-driven screen

Date: 2026-09-07. Branch `feat/cosx-s3-swap-screen`. Written and committed
BEFORE any number in the series below was taken.

## What is measured

Alkane series `alkane_4 / 8 / 12 / 16 / 20` (C_nH_{2n+2}, testdata/molecules),
def2-SVP; if the window allows, `alkane_4 / 8 / 12` at def2-TZVP. Per size, on
ONE converged DF-JK (def2-universal-jkfit) density, one rayon thread,
`OPENBLAS_NUM_THREADS=1`, release, PSI `full avg10 = 0.00` before AND after each
timed segment (a segment with nonzero PSI is discarded and re-run):

* full COSX `K` build (`CosxK`, md3c1e backend, (50,110) grid, overlap fit on,
  DENSITY-DRIVEN screen at the default `t = 1e-7`): total wall, A-build seconds
  and fraction, block-GEMM seconds, kept (pair, batch) fraction (density-driven)
  and geometry-only kept fraction;
* LinK `K` build (`LinkK`, Schwarz 1e-12): cold (pool + pairs + build) and
  warm (`update_density` + `build`) walls; the warm number is the one an SCF
  iteration pays and is the denominator.

Harness: `crates/ferric-scf/tests/cosx_full_k.rs` (`COSX_FK_*` env), one
process per cell, `scripts/ferric-limited --max=4G --high=3600M` with
`FERRIC_MEM_BUDGET_GB=2` for alkane_8 and up, foreground under `timeout` <= 8
min. A cell that does not fit the window is reported as such; NOTHING is
extrapolated from it.

Fit: TAIL ONLY — log-log slope through the LAST THREE sizes (C12, C16, C20 at
SVP) for each of: COSX A-build seconds, COSX total seconds, LinK warm seconds.
The C4 -> C8 step is pre-onset (butane is ~7 Bohr long; the alkane density
matrix decays over ~30 Bohr) and is reported but not fitted.

## Physics hypothesis (what I expect if the density-driven screen works)

* Grid points scale as N (atoms). Shell pairs scale as N^2. With an
  integral-magnitude-only screen the kept pairs PER POINT scale ~N^1.14 -> N^1
  (measured on this branch: every significant pair survives at every point, so
  the A-build is O(N^2)). With `bound_A x max(fmax_s1, fmax_s2) >= t`, `F = D X`
  is local to the point once the molecule exceeds the density-matrix decay
  length, so the kept pairs per point should saturate to ~const and the
  A-build tend to O(N).
* At C12 -> C20 / SVP (molecule 30 -> 50 Bohr) I expect the A-build to be
  PRE-asymptotic but bending: tail exponent **1.3 - 1.7**, with the
  density-driven kept fraction falling MONOTONICALLY with N (C4: 0.79 measured
  in the threshold sweep; C8 expected ~0.6; C20 expected <= 0.4) and sitting
  well below the geometry-only kept fraction at every size (C4: 0.95 measured).
  Literature (Laqua, Kussmann, Ochsenfeld sn-LinK; Neese COSX S-junction):
  seminumerical FLOPs go near-linear at LARGE N; I do not expect to reach 1.0
  by C20.
* The bounding-SPHERE batch bound is looser than a per-point bound (radius
  of a 256-point sub-batch of a Becke grid can be several Bohr), so the kept
  fraction will saturate ABOVE the per-point ideal; that is a cost (kept
  fraction), not a correctness issue.

## Artifact hypothesis (what a broken density-driven step looks like)

* A-build tail exponent **~2.0** AND kept fraction flat in N or tracking the
  geometry-only fraction => the `fmax` factor is not biting (e.g. fmax computed
  over the wrong rows, or the bound saturating at `total` for every pair).
* A-build wall FALLING faster than the kept fraction => the screen is
  dropping real work; the K error anchors (max|dK| < 1e-6 vs unscreened on
  water and butane, checked at the default t) must go red first. If the walls
  look too good AND the anchors are green, audit before writing anything up.
* Exponent exactly 1.00 at this size range would be suspicious (too clean):
  the decay-length argument says C12 -> C20 is pre-asymptotic.

## Full COSX K (not just the A-build)

The two block GEMMs per grid block (`F = D X`, `Ktilde += X G^T`) are DENSE:
`2 x nbf^2 x npts` flops = O(N^3). At C20/SVP that is ~3e11 flops ~ 30 s on
one thread, comparable to the screened A-build. I therefore expect the FULL
COSX K tail exponent to EXCEED the A-build's (somewhere 1.6 - 2.2) and the
A-build fraction of the total to FALL with N. This is the known next
bottleneck (sparse/AO-local half transform), not a screen defect; the GEMM
seconds are timed separately (`blas_s`) so the two are separable in the table.

## LinK on the same series

Measured earlier on this branch at SVP C8 -> C16: ~N^1.36. I expect the
C12 -> C20 warm-build tail exponent in **1.3 - 1.5**.

## Crossover statement to be made

def2-SVP is the LOW-L end where seminumerical exchange loses to analytic
exchange (the L-axis measurement: COSX beats LinK only at QZVP, 0.536x at
butane/QZVP). I expect COSX/LinK > 1 at EVERY size in this SVP series; the
result of interest is the TREND of the ratio (falling = the A-build exponent
is below LinK's; rising = above). No crossover in COSX's favour is predicted
at SVP; if one appears, that is a surprise to be audited, not celebrated.

## What gets reported

Per size: nbf, npts, COSX total wall, A-build s and %, GEMM s, kept_dd,
kept_geom, LinK cold and warm wall, COSX/LinK(warm). Then the three
last-three-point exponents, the kept-fraction trend, and the crossover
statement — plus the PSI readings for every segment.
