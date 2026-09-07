# COSX density-driven screen: threshold sweep and default choice

Date: 2026-09-07. Branch `feat/cosx-s3-swap-screen`. Harness
`crates/ferric-scf/tests/cosx_screen_sweep.rs` (`COSX_SS_*` env), one rayon
thread, `OPENBLAS_NUM_THREADS=1`, release, (50,110) grid + overlap fit,
md3c1e backend. `max|dK|` is against the UNSCREENED COSX K on the same grid;
`grid err` is the unscreened COSX K against the analytic direct K (thresh
1e-14). `kept_dd` = density-driven kept (pair, sub-batch) fraction;
`kept_geom` = fraction the geometry-only sphere bound alone would keep.
alkane_8 ran under `scripts/ferric-limited --max=4G --high=3600M`,
`FERRIC_MEM_BUDGET_GB=2`; PSI full avg10 was 0.00 before and after each cell.

## The screen (what is being thresholded)

Per 256-point sub-batch with `F = D X` in hand, pair `(s1, s2)` is evaluated iff

    bound_A(s1, s2, sphere(batch)) * max(fmax[s1], fmax[s2]) >= t,
    fmax[s] = max_{mu in s, g in batch} |F_{mu,g}|

`bound_A` = `PairBounds::coarse_estimate_sphere` (Hölder primitive bound,
never underestimates `max|A^g|`). The `max` over both shells covers the block's
two contributions `G_{s1} += A F_{s2}` and the mirror `G_{s2} += A F_{s1}`.

## Sweep

| system | basis | nsh | nbf | grid err | unscreened wall (A-build %) |
|---|---|---|---|---|---|
| water | cc-pVDZ | 12 | 24 | 5.60e-5 | 0.326 s (87%) |
| butane (alkane_4) | def2-SVP | 54 | 106 | 2.91e-4 | 9.41 s (90%) |
| octane (alkane_8) | def2-SVP | 102 | 202 | 4.55e-4 | 52.90 s (90%) |

| system | t | max\|dK\| | dK / grid err | kept_dd | kept_geom | wall s | speedup |
|---|---|---|---|---|---|---|---|
| water | 1e-5 | 2.90e-6 | 5.2e-2 | 0.9098 | 1.0000 | 0.340 | 0.96x |
| water | 1e-6 | 3.05e-8 | 5.5e-4 | 0.9693 | 1.0000 | 0.320 | 1.02x |
| water | 1e-7 | 1.56e-10 | 2.8e-6 | 0.9927 | 1.0000 | 0.350 | 0.93x |
| water | 1e-8 | 2.87e-12 | 5.1e-8 | 0.9996 | 1.0000 | 0.332 | 0.98x |
| water | 1e-9 | 0 | 0 | 1.0000 | 1.0000 | 0.323 | 1.01x |
| butane | 1e-5 | 1.54e-4 | 5.3e-1 | 0.5442 | 0.9153 | 5.283 | 1.78x |
| butane | 1e-6 | 1.32e-5 | 4.5e-2 | 0.6889 | 0.9326 | 6.458 | 1.46x |
| butane | 1e-7 | 7.59e-7 | 2.6e-3 | 0.7916 | 0.9464 | 7.463 | 1.26x |
| butane | 1e-8 | 7.43e-8 | 2.6e-4 | 0.8630 | 0.9614 | 8.131 | 1.16x |
| butane | 1e-9 | 4.55e-9 | 1.6e-5 | 0.9057 | 0.9709 | 8.575 | 1.10x |
| butane | 1e-10 | 3.97e-10 | 1.4e-6 | 0.9314 | 0.9778 | 8.845 | 1.06x |
| octane | 1e-6 | 1.71e-5 | 3.8e-2 | 0.3784 | 0.6960 | 22.707 | 2.33x |
| octane | 1e-7 | 2.03e-6 | 4.5e-3 | 0.4862 | 0.7351 | 28.414 | 1.86x |
| octane | 1e-8 | 1.70e-7 | 3.7e-4 | 0.5719 | 0.7604 | 32.676 | 1.62x |

## Readings

* Water: the geometry-only bound keeps 100% at EVERY threshold (the
  structural fact: every significant pair survives at every point); only the
  `fmax` factor drops anything.
* Default `t = 1e-7`: the loosest value with `max|dK| < 1e-6` on water AND
  butane (the anchor bar); 1e-6 gives 1.3e-5 on butane. At octane the screen
  error at 1e-7 is 2.0e-6 — above the 1e-6 anchor bar (set for water/butane),
  but 4.5e-3 of the grid error. The screen error grows ~N (7.6e-7 -> 2.0e-6,
  C4 -> C8), the grid error grows too (2.9e-4 -> 4.5e-4); the ratio moved
  2.6e-3 -> 4.5e-3. Not measured beyond C8 (unscreened K at C20 is out of the
  window).
* Density-driven vs geometry-only at 1e-7: butane drops 21% vs 5%, octane 51%
  vs 27%. "Far more", but NOT the "large majority" hoped for at C8 — the
  batch's bounding sphere (256 consecutive Becke points span ~2 radial shells,
  several Bohr at the outer shells) makes the geometry factor saturate at
  `total` for every pair near the batch; the F-factor alone then has to do
  the dropping. Sharper batch geometry (sub-spheres) is the next lever; it is
  a kept-fraction cost, not a correctness issue.
* Speedup at 1e-7: 1.26x (C4), 1.86x (C8). The scaling series
  (`cosx_scaling_prereg.md` -> `cosx_scaling_results.md`) is where this is
  judged.
