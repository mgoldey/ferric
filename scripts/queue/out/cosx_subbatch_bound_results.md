# Results: sub-batching the COSX bound test does not pay

Date: 2026-09-09. Branch `perf/cosx-subbatch-bound`. All numbers one thread
(`OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1`), **counts only, no timings** —
counts are deterministic and load-immune, and the box was under contention from
three other agents, so no wall-clock claim is made here.

---

## 0. VERDICT

**Sub-batching the screening decision does not pay, and the reason is
structural rather than tunable.** On butane/def2-SVP at the production
threshold `t = 1e-7`, shrinking the screening group from the whole 256-point
sub-batch down to 8 points reduces kept work by **0.12 percentage points**
(0.791577 -> 0.790411) for **8.2x the bound evaluations**.

**Recommended default group size: `0` (unsplit), i.e. do not enable this.**
The knob (`CosxConfig::screen_group`) is landed so the measurement stays
reproducible and so the verdict can be re-tested cheaply, not because a
non-zero value is recommended.

**The motivating premise was correct; the inferred fix was not.** The Python
reference study's measurement stands — the bound really does degenerate to its
distance-free value on most decisions. But that degeneracy is only ~1/3 caused
by the batch geometry, which is the only part sub-batching can address.

---

## 1. What was built

Three pieces, in commit order (anchors first, per the repo rule):

1. `PairBounds::coarse_estimate_box(s1, s2, lo, hi)` in
   `crates/ferric-integrals/src/cosx_screen.rs` — the Hölder pair bound over an
   axis-aligned box, using the EXACT point-to-box distance
   `R_c = max(0, d(M, box) - |AB|/2)`. Same validity argument as the existing
   sphere query (every product centre lies on segment `AB`; every term is
   non-increasing in `R`), same cost (three clamped differences, one `sqrt`).

2. `CosxConfig::screen_group` in `crates/ferric-scf/src/cosx_k.rs` — splits the
   SCREENING DECISION only into contiguous groups of that many points. The
   md3c1e kernel's 256-point `COSX_SUB_BATCH_POINTS` blocking is untouched. A
   pair kept by ANY group is evaluated for the whole sub-batch, so correctness
   holds by construction. `0` = one group = today's behaviour.

3. `Region` — each group carries BOTH a centroid ball and an AABB, and
   `Region::bound` returns their `min`. Both enclose the group's points, so
   both bounds are valid and so is the smaller.

Two new counters, `CosxTimings::bound_evals` and `::screen_degenerate`, make
the cost and the blindness measurable. They count DECISIONS (not scaled by
points), which is the honest accounting for a screening cost.

---

## 2. Design decisions, and the measurement behind each

### 2.1 How points are grouped: contiguous chunks of the existing order

Not a taste call — it follows from the grid. `build_atomic_grid_pruned` emits
points atom-major, radial-major, angular-minor, so 256 consecutive points are
~2.3 whole Lebedev(110) spheres of ONE atom, and a contiguous sub-chunk is an
ARC of one such sphere. Contiguous chunking therefore already gives spatially
coherent groups with no sort, no permutation, and no threat to the builder's
bit-identity guarantee (the partition stays a pure function of the point
count).

### 2.2 Enclosing volume: `min(centroid ball, AABB)`, not "a box because boxes are tighter"

The initial hypothesis was that an AABB must beat the sphere for a radially
ordered arc. **That is false, and the anchor caught it.** Measured over Lebedev
arcs (`cosx_screen_box_anchors.rs`):

| system | box tighter | sphere tighter |
|---|---|---|
| water/cc-pVDZ | 4.5% | 4.8% |
| butane/def2-SVP | 8.8% | **14.3%** |

Neither region contains the other — the box's corners stick out of the centroid
ball and the ball's caps stick out of the box — so neither bound dominates.
The provable statement is only that the box beats its OWN circumscribing
sphere (22.1% of queries on butane, best ratio 5.5e-2), and the anchor was
rewritten to assert that instead of the false claim. Since both are valid, the
implementation takes the `min`, which is strictly better than either.

This is the one place where writing the anchor first paid directly: the
original anchor asserted `box <= centroid-sphere` and FAILED on water pair
(6,6) (7.194e-1 vs 6.769e-1) before any sweep was run.

### 2.3 Group size: no value is recommended

See §3. The curve is flat, so there is no operating point to pick.

---

## 3. The deliverable tables

### 3.1 Degenerate fraction and kept work vs group size

butane/def2-SVP, (50,110)+overlap fit, `t = 1e-7`, converged RHF density.
`degenerate` = fraction of bound evaluations returning the distance-free
`R = 0` value. Counts are exact.

| group | degenerate | kept frac | kept pairs | bound evals | vs unsplit |
|---|---|---|---|---|---|
| 0 (unsplit) | 0.6515 | 0.791577 | 90 512 912 | 446 985 | 1.00x |
| 64 | 0.6419 | 0.790572 | 90 397 968 | 756 377 | 1.69x |
| 32 | 0.6441 | 0.790601 | 90 401 296 | 1 168 970 | 2.62x |
| 16 | 0.6465 | 0.790603 | 90 401 552 | 1 994 693 | 4.46x |
| 8 | 0.6423 | 0.790411 | 90 379 536 | 3 651 199 | **8.17x** |

water/cc-pVDZ, same settings (the screen barely bites at this size, as the
Python study predicted):

| group | degenerate | kept frac | bound evals |
|---|---|---|---|
| 0 | 0.8243 | 0.992681 | 5 070 |
| 64 | 0.7925 | 0.992283 | 5 436 |
| 32 | 0.7709 | 0.992283 | 5 972 |
| 16 | 0.7377 | 0.992283 | 6 959 |
| 8 | 0.6929 | 0.991652 | 9 063 |

**Read the two columns together.** On water the degenerate fraction falls a
lot (0.824 -> 0.693) and kept work still moves only 0.10 pp — the bound gets
less blind without the extra sight changing any decision. On butane the
degenerate fraction barely moves at all AND kept work barely moves. Neither
system supports the change.

Note the bound-evaluation growth is well below `G`, because `keep` stops at the
first group that keeps the pair: water (99% kept) grows only 1.8x at G=32,
butane (79% kept) grows 8.2x. That early-out is working; it is just not enough
to make a 0.12 pp gain worth 8x the screen cost.

### 3.2 Why: the cause split

`cosx_region_diagnostics.rs`, same grid and same point order, attributing every
degenerate query to one of two causes by re-evaluating with `|AB|/2` forced to
zero:

| system | group | degenerate | region reaches midpoint | `\|AB\|/2` eats it |
|---|---|---|---|---|
| water/cc-pVDZ | 256 | 78.21% | 68.16% | 10.04% |
| | 32 | 76.47% | 64.61% | 11.86% |
| | 8 | 74.83% | 61.92% | 12.91% |
| butane/def2-SVP | 256 | 63.99% | 33.64% | 30.35% |
| | 32 | 62.06% | 29.17% | 32.89% |
| | 8 | **59.55%** | **24.26%** | **35.28%** |

**This is the finding.** On butane only 33.6% of the degeneracy is the region
reaching the pair midpoint — the part a smaller group can fix, and it does move
that to 24.3%. The other 30.4% is `|AB|/2` eating the distance: a property of
the SHELL PAIR, not of the batch, which no amount of sub-batching can touch,
and whose *share* GROWS to 35.3% as the region shrinks.

So the screen is blind mostly because of diffuse shell pairs, not because of
batch geometry. Tightening the enclosing volume is the wrong lever. Anything
that would actually move this has to attack `|AB|/2` — i.e. the coarse bound's
"collapse the whole shell pair onto its midpoint" step — not the region.

---

## 4. Anchors and mutation proofs

### 4.1 Anchor values

| anchor | result |
|---|---|
| (a) trivial limit: `screen_group` 0 / 256 / 512 vs unsplit | **PASS, bitwise** K and identical counters, both systems |
| (a') grouped screen at `t = 0` vs unscreened | **PASS, bitwise**, `pairs_kept == pairs_total` |
| (b) `max\|dK\|` vs same unscreened K, water, G=64..8 | 1.56e-10 vs grid error 5.60e-5 (**PASS**, 0.1x bar) |
| (b) butane, G=64..8 | 7.48e-7..7.64e-7 vs grid error 2.91e-4 (**PASS**) |
| (c) degenerate fraction + kept work both fall | **the pre-registered assertion FAILED** — see §0; the test now pins the negative |
| box: never underestimates inside its box | PASS, 24.0M checks butane / 270k water |
| box: never looser than its circumscribing sphere | PASS, tighter on 22.1% (butane) |
| box: degenerates less often than the sphere | PASS (0.7785 vs 0.7789 butane; 0.9094 vs 0.9128 water) |
| box: degenerate box == point bound | PASS, **bitwise** |

### 4.2 Mutation proofs

| mutation | expected | observed |
|---|---|---|
| A: every group gets the WHOLE sub-batch's region | (c) RED, mechanism inert | **RED.** `kept` FREEZES at exactly 90 512 912 for every group size (real code reaches 90 379 536). This is what proves the negative result is physics, not a no-op build. |
| `drop_the_clamp` (`mid - lo` instead of the clamped overhang) | underestimation RED | **RED** |
| `box_is_the_sphere` (circumscribing-sphere form) | non-inertness RED, validity green | **RED on both non-inertness anchors, GREEN on underestimation** (it is still a valid bound) — exactly the intended split |
| B: misaligned groups (2 variants: `fmax` from group q+1; every group gets group 0's region) | (b) RED | **GREEN, 7.5-7.7e-7.** See below. |

### 4.3 Anchor (b) is WEAK, and that is reported rather than hidden

Mutation B did not turn anchor (b) red, in either variant. That is not a bad
mutation — it is a consequence of the result itself. At G=8 the union over 32
groups drops only 0.12 pp more work than the unsplit test, so misassigning
which group owns which bound perturbs a handful of marginal decisions and
cannot move K. **An anchor can only detect a defect that changes the answer,
and here almost nothing changes the answer.**

Anchor (b) is therefore honest evidence that K is CORRECT, and is NOT evidence
that the group/point alignment is right. This is recorded in its doc comment.
If grouping is ever made to pay, (b) must be re-mutated before being trusted as
an alignment guard.

---

## 5. What would reopen this

The verdict is provisional and dated, and the anchor is written to FAIL if it
is overturned (it asserts kept work does NOT fall more than 0.5 pp and the
degenerate fraction does NOT fall more than 0.05). Specifically:

* **A tighter treatment of `|AB|/2`.** That term is now the majority cause of
  degeneracy at every group size and grows in share as the region shrinks. A
  per-primitive-pair region test (rather than collapsing the shell pair onto its
  midpoint) attacks the part that actually dominates. Cost discipline is the
  obvious obstacle — that collapse is why the coarse bound is one `sqrt`.
* **Larger systems.** Everything here is water (3 atoms) and butane (4 heavy
  atoms). The repo rule "do not declare a negative below the onset" applies:
  locality effects have a size threshold, and butane at 10.5 Bohr is below the
  ~30 Bohr scale where alkane density-matrix locality turns on. The kept-work
  column does fall faster on butane than water, so the trend is not obviously
  flat with size. **This negative is stated for these two systems at these
  thresholds and is not licensed to be extrapolated to C20+.** Re-running
  §3.1 at alkane_8/12/16 is the cheap next check, and it needs only counts.

---

## 6. Reproduction

```
OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  cargo test -p ferric-integrals --release --test cosx_screen_box_anchors -- --nocapture
OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  cargo test -p ferric-scf --release --test cosx_group_screen_anchors -- --nocapture --test-threads=1
OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  cargo test -p ferric-scf --release --test cosx_region_diagnostics -- --nocapture --test-threads=1
```
