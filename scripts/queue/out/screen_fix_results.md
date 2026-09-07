# COSX shell-pair screen fix — results (2026-09-07, branch feat/cosx-screen-fix)

Pre-registration: `screen_fix_prereg.md` (commit 1e0a3eb3, before any
screened number with the new bound existed). Code: d9d1c8fa (bound + anchors),
cdca74ef (coarse gate, early exit, sphere bound, harness). All numbers below
from `crates/ferric-integrals/tests/{cosx_screen_anchors,cosx_screen_sparsity}.rs`,
single thread, release, PSI `full avg10` = 0.00 before and after every cell.

## 1. The bound

Hölder on the primitive expansion (module doc of `cosx_screen.rs` has the
derivation). Per primitive pair `(a@A, b@B)`, `p=a+b`, `P` product centre,
`R=|P−r_g|`, `K_AB = exp(−ab/p|A−B|²)`:

    |A^g|_block ≤ Σ_ij |c_i c_j| K_ij s_a s_b [ (2π/p) q_0 F0⁺(pR²) + (4π/p) Σ_{k≥1} q_k M_k F0⁺(pR²/2) ]

`q_k` from `(ρ+d_A)^{l_a}(ρ+d_B)^{l_b}`, `M_k=(k/(pe))^{k/2}`, `F0⁺(T)=min(1,½√(π/T))`,
`s` = max |column sum| of cart→sph. Every step is a pointwise inequality.
Decays as `K_AB` in pair separation and `1/R` in grid distance.

Why not diagonal Cauchy-Schwarz `sqrt(A_μμ A_νν)`: valid, never vanishes, but
carries NO `K_AB` — for μ on A and ν on a distant B it is `~1/√(R_A R_B)`,
which exceeds 1e-7 at every point of any Becke grid. Valid and vacuous.
Rejected on paper; not implemented.

Evaluation structure (`PairBounds::exceeds`): O(1) early-out on the `R=0`
value → one-sqrt coarse gate `min(total, Σβg / R_c)`, `R_c = |M−r| − |AB|/2`
(valid: every product centre lies on segment AB, every term non-increasing
in R) → per-term sum heaviest-first, exiting at the threshold (terms ≥ 0).
Decisions identical to `estimate() >= t` (asserted on every anchor sample).
`coarse_estimate_sphere(centre, radius)` gives the same gate for a whole
spatially local batch in O(1) per pair.

## 2. Anchors (all in tests/cosx_screen_anchors.rs)

Run against the OLD bound first (before the fix): never_underestimates RED,
K anchor RED at 1.062, trivial-limit RED at t=1e-300 (21/1485 pairs dropped
because their signed-overlap magnitude is exactly 0).

### (a) never_underestimates — 0 violations

| system | nsh | probes | (pair,probe) samples | violations | true/bound p10 | p50 | p90 | max |
|---|---|---|---|---|---|---|---|---|
| water / cc-pVDZ | 12 | 204 | 15 897 | 0 | 1.6e-2 | 0.229 | 0.998 | 1.000 |
| butane / def2-SVP | 54 | 201 | 295 601 | 0 | 5.3e-2 | 0.435 | 1.000 | 1.000 |
| butane / def2-QZVP (g) | 168 | 201 | 2 655 983 | 0 | 3.1e-3 | 0.114 | 1.000 | 1.000 |

Probes: random in the box (+3 Bohr), ON every nucleus, 1e-4 Bohr off every
nucleus, 50/60/200 Bohr away; every (l_a,l_b) up to (4,4) visited. p90 = 1.0
because the k=0 (s–s) term is asymptotically exact in the far field
(F0 → ½√(π/T) as erf → 1). Same-centre mixed-l pairs (the failure mode),
tracked at the probe where the true value is largest: bounds are O(1), worst
true/bound 0.154 on all three systems; 15/15, 60/64, 663/770 of them reach
|A| > 1e-2 somewhere, so the mode is exercised.

### legacy tripwire (permanent mutation test)
The old formula, rebuilt in the test file and run through the same checker on
water/cc-pVDZ: 4043 violations, worst underestimate (true − bound) = 0.383,
15/15 same-centre mixed-l pairs with bound < 1e-6 while true > 1e-2.

### (b) trivial limit — t = 0, −1, 1e-300 bit-identical to unscreened, all pairs kept (butane/def2-SVP, 71 probes).

### (c) reachability — octane/def2-SVP, 500 grid points, t=1e-7: kept fraction 0.712 (per-point 3638..3780 of 5253). Not tuned.

### (d) K-level — water/cc-pVDZ, weighted (50,110) Becke grid (16 500 pts), hcore-guess D, ferric-integrals only
|                       | old bound | new bound |
|---|---|---|
| max\|K_scr − K_unscr\|, t=1e-7 | **1.062** | **0.0** (all 5070 batch-level pairs kept) |
| max\|K_scr − K_unscr\|, t=1e-8 | — | **0.0** |
| grid error max\|K_cosx − K_exact(ERI)\| | 2.415e-5 | 2.415e-5 |

(0.71 in the PR record was with the SCF density; 1.06 here with the hcore-guess
density — same bug, same harness before/after.)

### decisions on the measurement grid (dropped ⇒ truly negligible)
butane/def2-SVP 2000 grid pts: dropped 193 107 (pair,point) at 1e-7, largest
true among them 9.995e-8; kept-but-below-t 0.46% of all pairs.
octane/def2-SVP 600 pts: dropped 908 111, largest true 9.999e-8; kept-but-below-t 1.3%.

### live-code mutation
Zeroing the same-centre mixed-l terms in `PairBounds::build` (the old failure
mode, applied to the REAL code): never_underestimates RED, K anchor RED at
1.062, dropped-pairs RED, trivial-limit RED. Reverted (`git checkout`).

## 3. Sparsity with the VALID screen

Fixed-seed (20260907) sample of 2000 positions of the (50,110) grid; per-POINT
kept shell pairs via `exceeds`. `geom-only` = pairs with `max_estimate >= t`
(once per geometry, no per-point work). `batch256` = `any_exceeds` over
batches of 256 UNSORTED random points.

### def2-SVP, t = 1e-7

| system | natoms | nsh | pairs | kept mean | kept frac | min..max | coarse-only | batch256 | geom-only | prereg |
|---|---|---|---|---|---|---|---|---|---|---|
| C4  | 14 | 54  | 1 485  | 1 388.4  | 0.935 | 1377..1395   | 0.942 | 0.948 | 0.950 | ≥0.97 |
| C8  | 26 | 102 | 5 253  | 3 738.1  | 0.712 | 3611..3780   | 0.728 | 0.739 | 0.743 | 0.85–0.95 |
| C12 | 38 | 150 | 11 325 | 6 070.4  | 0.536 | 5802..6160   | 0.550 | 0.565 | 0.569 | 0.65–0.85 |
| C16 | 50 | 198 | 19 701 | 8 361.9  | 0.424 | 7988..8501   | 0.435 | 0.452 | 0.456 | 0.55–0.75 |
| C20 | 62 | 246 | 30 381 | 10 615.4 | 0.349 | 10164..10802 | 0.358 | 0.375 | 0.380 | 0.45–0.65 |

t = 1e-8 (def2-SVP): 0.952, 0.746, 0.569, 0.453, 0.374.

### def2-TZVP, t = 1e-7

| system | nsh | pairs | kept mean | kept frac | coarse-only | batch256 | geom-only |
|---|---|---|---|---|---|---|---|
| C4  | 84  | 3 570  | 3 256.3  | 0.912 | 0.920 | 0.924 | 0.924 |
| C8  | 160 | 12 880 | 8 929.0  | 0.693 | 0.711 | 0.723 | 0.724 |
| C12 | 236 | 27 966 | 14 643.9 | 0.524 | 0.538 | 0.555 | 0.557 |
| C16 | 312 | 48 828 | 20 279.8 | 0.415 | 0.426 | 0.445 | 0.447 |
| C20 | 388 | 75 466 | 25 846.7 | 0.343 | 0.350 | 0.371 | 0.372 |

t = 1e-8 (def2-TZVP): 0.927, 0.727, 0.559, 0.447, 0.370.

### Tail exponents of the kept COUNT vs natoms (log-log least squares)

| series | tail C12–C20 | last two (C16–C20) | global C4–C20 (for contrast only) |
|---|---|---|---|
| DZ 1e-7 | 1.143 | 1.109 | 1.366 |
| DZ 1e-8 | 1.164 | 1.129 | 1.400 |
| TZ 1e-7 | 1.162 | 1.128 | 1.391 |
| TZ 1e-8 | 1.188 | 1.151 | 1.433 |
| DZ geom-only 1e-7 | 1.189 | 1.158 | 1.411 |
| total pairs nsh(nsh+1)/2 | 2.016 | | |

The global C4–C20 fit averages in pre-onset points and reads 0.25 higher
than the tail — quote the tail.

### Verdict against the pre-registration

* Pre-registered "survives" = C20 kept ≤ 0.70 AND tail exponent < 1.8:
  **met with margin** (0.349, 1.14). "Does not survive" (C20 ≥ 0.85 or
  exponent ≥ 1.9): not met.
* The fractions came in BELOW the pre-registered ranges at every size (C4 0.935
  vs ≥0.97; C20 0.349 vs 0.45–0.65). The prereg estimate used only the most
  diffuse s primitive (K_AB radius ≈ 15 Bohr); contracted s/p shells with
  exponents ≥ 0.4 are separable at ≈ 8 Bohr and are the majority of pairs.
  The prereg's artifact clause for C4 < 0.95 (underestimate the anchor missed)
  was checked directly: every dropped pair on the measurement grid has
  true < 1e-7 (largest 9.995e-8) — the drops are real.
* Versus the INVALID Stage 2 numbers (0.96 → 0.57 DZ over the same series):
  the valid screen keeps 0.935 at C4 and 0.349 at C20. The old bound wrongly
  dropped O(nsh) same-centre blocks (a vanishing share of the nsh² pairs) but
  its pair-separation decay was sqrt(overlap) — SLOWER than the truth — so
  it kept far more distant pairs; the valid bound decays at the correct K_AB
  rate. The invalid record therefore UNDERSTATED the asymptotic sparsity
  while corrupting K by O(1). (On my 500-point octane sample the old bound
  kept 0.836 vs the valid 0.712.)
* Basis independence: DZ and TZ fractions agree within 0.02 at every size —
  the kept set is a shell-centre geometry property.
* ~92% of the drop is grid-INDEPENDENT (geom-only 0.380 vs per-point 0.349 at
  C20): a once-per-geometry significant-pair list, O(nsh²) total, no per-point
  work, captures nearly all of it. The remaining ~3 points come from the
  1/R factor and cost per-point work.
* Per-point spread is tight (C20 DZ 10164..10802, ±3%) — no sample bias.

Plain answer: **yes, O(N) sparsity survives a valid screen** — the kept-pair
count grows as N^1.14 (DZ) / N^1.16 (TZ) over C12–C20, already close to
linear, and the fraction of the nsh² pairs falls as ~N^−0.86.
Provisional: measured only to C20 (62 atoms, ≈48 Bohr) on linear alkanes at
two thresholds; the asymptote 1.0 is supported, not reached.

## 4. Screen cost per point

`exceeds` over all pairs per point, vs `Md3c1e::for_each_pair` per point at
B=256 unscreened, same points, same process:

| system / basis | pairs | prim-pair terms | A-build s/pt | screen s/pt | ratio |
|---|---|---|---|---|---|
| alkane_4 / def2-SVP | 1 485 | 4 886 | 1.18e-4 .. 1.63e-4 (3 runs) | 2.4e-5 | 0.15–0.21 |
| alkane_8 / def2-SVP | 5 253 | 16 924 | 3.84e-4 | 7.4e-5 | 0.19 |

≈16 ns per pair per point for the screen vs ≈75–90 ns per pair per point for
the batched kernel; pre-registered 1–5% was wrong (the kernel amortizes
primitive-pair setup over 256 points; a per-POINT screen cannot). Per-point
screening therefore pays for itself only where it drops > ~20% of pairs
(C8 and up). The production structure is (i) the geom-only list once per
geometry (free, ~92% of the benefit) and (ii) `coarse_estimate_sphere` once
per spatially local batch (O(1) per pair per batch) — both are in the API;
the batching policy is the K builder's.

## 5. For the integrator / ferric-scf owner

* `crates/ferric-scf/tests/cosx_k_anchors.rs::cosx_a_screen_is_unsound_tripwire`
  is designed to fail once the screen is fixed — it should now FLIP; the
  `CosxK` refusal of `screen_thresh > 0` can be lifted.
* `ferric_integrals::cosx_a::{CosxScreen, PairBounds}` paths are preserved
  (re-exports of `cosx_screen`); `PairBounds::build/estimate/nshells`
  signatures unchanged; new: `exceeds`, `any_exceeds`, `coarse_estimate`,
  `coarse_estimate_sphere`, `max_estimate`, `nterms`.
* ferric-scf tests that assert something about the OLD screen's kept counts
  (cosx_a_anchor.rs far-point drop at 1e-6, cosx_stage2_sweep, cosx_l_axis
  scr columns) will change numbers; they were built on an invalid bound.
