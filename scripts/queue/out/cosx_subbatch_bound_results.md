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

**On the non-monotonicity between group sizes** (butane G=64 keeps 90 397 968
but G=32 keeps 90 401 296, i.e. slightly MORE): this is genuine, not noise —
counts are deterministic. Kept work is only guaranteed monotone against the
UNSPLIT test, which is the relation the union property proves and which
`grouped_screen_never_drops_what_the_unsplit_screen_keeps` asserts; it is NOT
guaranteed between two non-trivial group sizes. Halving a group shrinks its
`fmax` (helps) but its centroid ball is not nested inside the longer arc's
ball, so the geometric factor can move either way. The anchor is written
against unsplit for exactly this reason.

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

## 4.5 THE SIZE AXIS: the negative is confirmed past the 30 Bohr onset

Run 2026-09-09 on coordinator clearance, because a negative measured only below
the onset is not licensed for C20+ (repo rule). Counts only.

### 4.5.1 The hypothesis under test

`|AB|/2` is a fixed property of a shell pair, but the batch region was expected
to GROW with the molecule — a Becke grid over a longer chain spreading its
256-point sub-batches further apart. If so, the region term would come to
dominate at C16, grouping would start to matter, and the butane-scale flat
curve would be an artifact of small molecules.

### 4.5.2 The cause split vs size (`cause_split_vs_molecular_size_across_the_locality_onset`)

def2-SVP, (50,110), box volume, ~25 sub-batches sampled across the whole grid
at each size:

| system | diam (Bohr) | nsh | group | degenerate | region reaches | `\|AB\|/2` eats it |
|---|---|---|---|---|---|---|
| alkane_4 | 10.5 | 54 | 256 | 67.56% | **37.88%** | 29.68% |
| | | | 32 | 65.42% | 34.75% | 30.67% |
| | | | 8 | 63.07% | 30.73% | 32.33% |
| alkane_8 | 19.9 | 102 | 256 | 53.76% | 17.12% | **36.64%** |
| | | | 32 | 50.83% | 12.67% | 38.16% |
| | | | 8 | 49.11% | 10.55% | 38.56% |
| alkane_12 | 29.3 | 150 | 256 | 52.97% | 21.56% | 31.41% |
| | | | 32 | 49.86% | 16.76% | 33.11% |
| | | | 8 | 48.09% | 14.16% | 33.93% |
| **alkane_16** | **38.7** | 198 | 256 | 44.79% | **15.92%** | **28.86%** |
| | | | 32 | 43.41% | 13.41% | 29.99% |
| | | | 8 | 42.28% | 10.99% | 31.28% |

**The hypothesis is refuted, and in the OPPOSITE direction from the concern.**
The region share does not grow with size — it COLLAPSES, 37.9% -> 15.9% from
C4 to C16 — while `|AB|/2` stays dominant at every size past C4 (1.8x the
region term at C16). Butane was the most FAVOURABLE case for grouping, not the
least. At C16 the fixable share of the degeneracy is smaller than it was at
butane in both absolute and relative terms.

### 4.5.3 The mechanism, measured directly (`subbatch_region_extent_is_size_independent`)

Rather than infer why, the sub-batch extent (AABB half-diagonal) was measured:

| system | diam | sub-batches | median extent | p90 | max |
|---|---|---|---|---|---|
| alkane_4 | 10.5 | 301 | **2.028** | 14.37 | 28.98 |
| alkane_8 | 19.9 | 559 | **2.028** | 16.07 | 29.40 |
| alkane_12 | 29.3 | 817 | **2.028** | 15.40 | 32.42 |
| alkane_16 | 38.7 | 1075 | **2.028** | 15.40 | 35.81 |

**The diameter grows 3.70x; the median sub-batch extent grows 1.00x.** A Becke
sub-batch is an ATOM-LOCAL object — 256 consecutive points are ~2.3 Lebedev
shells of one atom, and that does not care how long the chain is. Adding
carbons adds MORE sub-batches (301 -> 1075), not bigger ones. Meanwhile `nsh`
grows linearly (54 -> 198), so the O(nsh^2) pair denominator fills up with
distant pairs, and a distant pair that degenerates does so because of `|AB|/2`,
never because a 2 Bohr region reached it.

This is asserted, not just printed: the test FAILS if the median extent ever
grows more than 2x while the diameter grows 3.7x.

### 4.5.4 Verdict on the size axis

**The negative is confirmed past the onset and the lane closes properly.**
alkane_16 at 38.7 Bohr is past the ~30 Bohr scale, and grouping's addressable
share of the problem is smaller there than at butane. The default stays `0`.

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
* **~~Larger systems.~~ CLOSED by §4.5** (2026-09-09). The size axis was run to
  alkane_16 (38.7 Bohr, past the ~30 Bohr onset) and the negative is confirmed
  there: the region term COLLAPSES with size (37.9% -> 15.9%) rather than
  growing, because a Becke sub-batch is atom-local (median extent 2.028 Bohr at
  every size while the diameter grows 3.70x). Butane was grouping's best case.
  What is still NOT covered: other chemistries (only linear alkanes), other
  bases (only def2-SVP on the size axis), and 3D/globular systems where a
  sub-batch's neighbours are denser than along a chain.

### 5.1 A standing warning for whoever touches this next

**Anchor (b) is NOT an alignment guard.** Two misalignment mutations were
applied and BOTH left it green (§4.3). It verifies that K is correct; it does
not verify that each group's bound is applied to that group's points. The
reason is the result itself — at G=8 the union drops only 0.12 pp more work, so
a misassigned bound cannot move K.

Concretely: **if anyone makes grouping actually pay, anchor (b) must be
re-mutated before it is trusted**, and until it goes red under a misalignment
mutation it is not evidence about alignment. Do not read its green as coverage
it does not have. This is recorded here, in the anchor's own doc comment, and
in the implementing commit message, because a green test with unstated blind
spots is exactly how the repo's "a test you have never seen fail is an
assumption" failure mode recurs.

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

The size axis (§4.5) and the kept-work size sweep are heavier and explicit:

```
# cause split + region extent vs size (seconds; needs no SCF)
OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 scripts/ferric-limited --max=4G --high=3600M -- \
  cargo test -p ferric-scf --release --test cosx_region_diagnostics -- --nocapture --test-threads=1

# kept-work counts C4..C16 (minutes per system; converged RHF + 5 COSX builds each)
OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 scripts/ferric-limited --max=4G --high=3600M -- \
  cargo test -p ferric-scf --release --test cosx_group_screen_anchors -- --ignored --nocapture \
  --test-threads=1 kept_work_vs_group_size_across_the_locality_onset
```
