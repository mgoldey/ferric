# Results: a distance-dependent integral partition bound for 3c-1e / COSX

**2026-09-09.** Pre-registration: `scripts/queue/out/ipb_distance_prereg.md`
(committed before any number). Code: `scripts/ipb_distance_proto.py` (bounds,
anchors, K build, threshold sweep) and `scripts/ipb_distance_degeneracy.py`
(size-axis degeneracy/tightness, no SCF, no K build). Python/numpy/PySCF only,
one thread, deterministic counts.

Measurement is separated from interpretation below, per repo convention: §2–§5
are tables, §6 is the verdict and is explicitly provisional.

---

## 1. What was built

Three bounds on `max_g |A^g_{mu,nu}|` for a shell pair over a grid batch,
implemented on ONE core with ONE density weight (`max(fmax[s1], fmax[s2])`), so
the only thing varying between them is the integral estimate:

| name | what it is |
|---|---|
| `ferric` | ferric's Hölder primitive-pair sphere bound, ported from `cosx_screen.rs`. Carries `1/R` decay; subtracts `|AB|/2` from the distance. |
| `ipb-flat` | Thompson & Ochsenfeld's integral partition bound in the batch-independent form sn-LinK uses (Eq 15 of JCTC 16, 1456). Exact radial moment; **no distance decay**. |
| `ipb-dist` | **new.** `min(V_0, sum_ab [S^ab_0/R_ab + V^ab_{R_ab}])` — exact radial moment AND a valid distance factor. |
| `min(ferric,ipb-dist)` | the pointwise min of two valid bounds, i.e. what a production implementation would actually do. |

### 1.1 The formula, and the trap

The previous attempt's anchor rejected its distance form because **Eq (A10)'s
`R` is the partitioning-ball radius**, so `V_R` bounds only the charge OUTSIDE
the ball — a tail integral, not a bound at distance. The missing half is the
IPB paper's own Eq (A15), Newton's shell theorem: the inside charge is collapsed
to the centre and gets an exact `1/D`, the outside charge is exactly `V_D`.

**A second reference-point subtlety, which anchor A0 caught on the first run and
which is the same error class one level down.** Newton's theorem is a statement
about a function spherically symmetric *about a particular point*. Each
primitive pair's radial majorant is symmetric about its own `P_ab`, **not** about
the contracted pair centre `C_uv = P_min` (they differ by up to 0.69 Bohr on
water/cc-pVDZ). A first version used the (A10)-shifted radius `R_ab` in the tail
term but the unshifted distance `D` in the `S_0/D` inside term — mixing two
reference points. That produced **288 violations, all two-centre s-s, max ratio
1.014**: a 1.4% underestimate, small enough to pass for rounding and large enough
to drop significant integrals. The fix is to take the split per primitive with
`R_ab` in both terms.

Both closed forms (`S_R` ~ A14, `V_R` ~ A16) were verified against an independent
4000-node Gauss-Legendre radial quadrature of the same majorant, `<= 5e-11`
relative, `(la,lb)` up to `(3,4)`, `R` up to 3 Bohr, **before** the prototype was
written.

---

## 2. Anchors (water/cc-pVDZ, 197 probes x 66 pairs)

Probes: every nucleus exactly (T=0), 1e-4 and 1e-8 Bohr off each nucleus, a
cloud through the molecular volume, and shells at 20/50/100/200 Bohr.
Truth = PySCF `int1e_grids`. Same-centre pairs reported separately, because that
is where ferric's original signed-overlap bound failed catastrophically.

| anchor | result |
|---|---|
| A0 `ferric` never underestimates, two-centre | PASS — 0/5811; true/bound p50 1.93e-1, p90 9.58e-1, max 9.97e-1 |
| A0 `ferric` never underestimates, same-centre | PASS — 0/4011; p50 2.40e-1, p90 1.00, max 1.00 |
| A0 `ipb-flat` never underestimates, two-centre | PASS — 0/5811; p50 1.30e-2, p90 1.85e-1, max 9.63e-1 |
| A0 `ipb-flat` never underestimates, same-centre | PASS — 0/4011; p50 8.72e-3, p90 1.86e-1, max 1.00 |
| A0 `ipb-dist` never underestimates, two-centre | PASS — 0/5811; p50 2.96e-1, p90 9.99e-1, max 1.00 |
| A0 `ipb-dist` never underestimates, same-centre | PASS — 0/4011; p50 3.26e-1, p90 1.00, max 1.00 |
| A1a trivial limit `IPB-D(D=0) == IPB-flat` | PASS — max relative difference **exactly 0** |
| A1b `IPB-D` never exceeds `IPB-flat` | PASS — max `(IPB-D/IPB-flat - 1)` = 0 over D in {0.1 … 100} |
| A2 non-inert: the distance factor bites | **FAIL at water** — see §3 |
| A3 unscreened K reproduces analytic K | PASS — E_grid 6.31e-5 vs bar 9.76e-3 |
| A4 threshold-0 trivial limit, all four screens | PASS — max\|dK\| = 0, kept = 3036/3036 |

**Tightness reading (A0, at zero-radius probes).** `ipb-dist` is the tightest of
the three at the median: true/bound p50 **0.296** against `ferric`'s 0.193 and
`ipb-flat`'s 0.013 — i.e. `ipb-flat` is ~23x looser than `ipb-dist` at the median
probe, and `ferric` ~1.5x looser. The distance-dependent IPB is, as
pre-registered, the tightest bound of the three *as a bound*.

### 2.0 A0 at def2-SVP and def2-QZVP (g functions) — and the tightness ordering

§6.2a's lever 2 records that only STO-3G was ever measured, and that the IPB's
advantage is radial/angular so QZ is where it should show best. Run on
alkane_4 (10.5 Bohr) with the same 197-probe set:

| basis | nbf | nsh | lmax | bound | probes | violations | true/bound p50 | p90 |
|---|---|---|---|---|---|---|---|---|
| def2-SVP | 106 | 54 | 2 | `ferric` | 243 880 | **0** | 0.300 | 0.959 |
| | | | | `ipb-flat` | 243 880 | **0** | 0.0327 | 0.222 |
| | | | | `ipb-dist` | 243 880 | **0** | **0.542** | 1.00 |
| def2-QZVP | 528 | 168 | **4** | `ferric` | 2 332 330 | **0** | 0.110 | 0.774 |
| | | | | `ipb-flat` | 2 332 330 | **0** | 0.00776 | 0.101 |
| | | | | `ipb-dist` | 2 332 330 | **0** | **0.225** | 1.00 |

(two-centre pairs; same-centre pairs are in the raw output and also give 0
violations at both bases, 210 792 checks at QZVP.)

**Zero violations across 2.5 million QZVP pair-probe checks including g
functions.** A1a/A1b hold exactly at both bases.

The tightness ordering is `ipb-dist` > `ferric` > `ipb-flat` at every basis, and
the spread WIDENS with angular momentum:

| | SVP | QZVP |
|---|---|---|
| `ipb-dist` / `ferric` | 1.8x tighter | **2.0x tighter** |
| `ipb-dist` / `ipb-flat` | 16.6x tighter | **29.0x tighter** |

This is the pre-registered "expected tightness gain" landing, and it lands
hardest exactly where §6.2a predicted it would — the batch-independent IPB, the
form sn-LinK actually ships, is **29x looser than the distance-dependent one at
def2-QZVP**, which is the basis regime where COSX wins at all (§2 of the
whitepaper). Note also that `ipb-flat` degrades ~4x from SVP to QZVP (0.0327 ->
0.00776) while `ipb-dist` degrades only ~2.4x: the distance factor is doing
proportionally MORE work at high angular momentum, not less.

### 2.1 Mutation tests — every anchor proven able to fail

| mutation | what it breaks | caught by |
|---|---|---|
| `drop_inside_term` | omits `S_0/R_ab`, i.e. **exactly the previous attempt's error** | A0: 9511 violations, true/bound p50 **1.9e8**, max inf |
| `no_min_with_flat` | drops the `min(V_0, …)` | A1b: `IPB-D` exceeds `IPB-flat` by **17.9x** |
| `rab_unclamped` | lets `R_ab` go negative, silently shrinking the tail | A0: 6 violations (a small mutation still caught) |
| `drop_mirror` | drops the K accumulation's mirror term | A3: unscreened K wrong by **1.35** |

The `drop_inside_term` result is worth keeping: it reproduces the previous
attempt's failure mode exactly, which independently confirms the diagnosis that
its error was the dropped inside charge rather than anything about `R_ab`.

---

## 3. Degeneracy: does the distance factor bite? (`ipb_distance_degeneracy.py`)

"Degenerate" = the bound falls back to its distance-free value for that
(shell pair, grid batch). For `ferric` that is the `R=0` total; for `ipb-dist`
it is `V_0`. Both measured on the SAME decisions. STO-3G, Becke (25,50)
unpruned, 256-point batches.

| system | diameter | decisions | deg `ipb-dist` | deg `ferric` | `flat/dist` p50 | p90 | max |
|---|---|---|---|---|---|---|---|
| alkane_1 | 3.90 | 700 | 88.86% | 90.00% | 1.00 | 1.48 | 8.2 |
| alkane_2 | 5.78 | 3 120 | 81.51% | 87.88% | 1.00 | 3.03 | 19.8 |
| alkane_4 | 10.46 | 17 457 | 78.70% | 86.28% | 1.00 | 4.03 | 45.7 |
| alkane_8 | 19.87 | 114 681 | 64.20% | 78.82% | 1.00 | 9.67 | 98.5 |
| alkane_12 | 29.28 | 363 258 | **48.85%** | 69.79% | **1.26** | 18.8 | 151 |

`ipb-dist` degenerates **less** than `ferric` at every size, and the gap widens
with size (1.1 pp at C1 → 20.9 pp at C12). The tightening over `ipb-flat` rises
from a median of exactly 1.00 (C1–C8) to 1.26x at C12, with a p90 of 18.8x.

**The batch-size confound, measured rather than assumed.** Degeneracy is
strongly sensitive to the grid, because a coarser grid means fewer, larger
batches and a larger batch radius eats more distance. Control at (15,26):

| system | diameter | deg `ipb-dist` @ (25,50) | @ (15,26) | deg `ferric` @ (15,26) |
|---|---|---|---|---|
| alkane_4 | 10.46 | 78.70% | 97.50% | 98.33% |
| alkane_8 | 19.87 | 64.20% | 86.20% | 92.66% |
| alkane_12 | 29.28 | 48.85% | 64.64% | 80.76% |
| alkane_16 | 38.70 | — | **52.27%** | 73.57% |

A coarse-grid point may **not** be compared against a fine-grid one: at C12 the
grid change alone moves degeneracy by 15.8 pp, which is larger than most of the
size effect. Within each grid series the trend is monotone and the `ferric` gap
grows with size (C16 at the coarse grid: 52.3% vs 73.6%, a 21.3 pp gap).

**Against the pre-registered inert criterion.** The criterion was: if `ipb-dist`
lands in ferric's 45–80% degenerate band, the lane closes. It lands at
**48.9% at C12 and 52.3% at C16** — inside that band, though at its bottom edge
and consistently ~20 pp better than `ferric` measured on the same decisions.
That is a *qualified* trip of the inert criterion, and §6 treats it as such.

---

## 4. Kept work at matched K error — the deliverable question

Thresholds 1e-2 … 1e-8; kept work weighted by `ncart(s1) x ncart(s2) x |batch|`;
K error is `max|K_screened - K_unscreened|`. Reported at **matched K error**, not
matched threshold — §6.2a's recorded trap is that a threshold cuts on
`bound x weight`, so equal thresholds are not equal operating points.

### alkane_4 / STO-3G, grid (25,50), E_grid = 3.08e-03

kept-work % (weighted), log-interpolated over threshold to equal K error:

| K error | `ferric` | `ipb-flat` | `ipb-dist` | `min(ferric,ipb-dist)` |
|---|---|---|---|---|
| 1e-05 | 94.381 | 94.480 | **94.051** | 94.051 |
| 1e-06 | 96.606 | 96.675 | **96.413** | 96.413 |
| 1e-07 | 97.828 | 97.860 | **97.666** | 97.666 |
| 1e-08 | 98.600 | 98.603 | **98.480** | 98.480 |

`ipb-dist` wins on every row, by **0.12–0.33 pp** against `ferric` and
**0.12–0.43 pp** against `ipb-flat`. `min(ferric, ipb-dist)` is identical to
`ipb-dist` to 3 decimals, i.e. `ipb-dist` dominates ferric's bound essentially
everywhere it matters — but the margin is inside the ±0.5–0.83 pp band in which
§6.2a already found the two existing bounds indistinguishable.

### water / cc-pVDZ, grid (50,110)

Reported for completeness and as a scope limit, not as evidence: at 2.86 Bohr
the screen is vacuous below t=1e-4 (all four bounds keep 100.000% at every
threshold from 1e-5 down, K error exactly 0), so the matched-error interpolation
has no bracket and every entry is `--`. Water cannot discriminate these bounds.

---

## 5. The mechanism: why a 2-29x tighter bound buys 0.1-0.4 pp of work

This is the load-bearing part of the study, because the two headline numbers
point in opposite directions and the reconciliation is the transferable finding.

**The bound really is much tighter.** §2.0 measures `ipb-dist` at 2.0x `ferric`
and 29x `ipb-flat` at the median QZVP probe, with zero violations over 2.5M
checks. That is not a marginal improvement and it is not a measurement artifact
(three independent constructions, one exactness anchor, four mutations).

**The screen barely notices.** §4 measures 0.12-0.43 pp of kept work at matched
error. A 2x tighter bound moving the kept fraction by a third of a percentage
point means the screening decision is **not bound-limited**.

**Why — and the arithmetic is checkable from the table above.** The screen tests
`bound x weight >= t`. A bound that is uniformly `g`-times tighter is *exactly*
equivalent to keeping the old bound and raising the threshold by `g`, so its
maximum possible benefit is the slope of kept-work-versus-log-threshold, times
`log10(g)`. On C4/STO-3G that slope collapses as the threshold tightens:

| decade | d(kept-wt) | ceiling for a 2.0x tighter bound |
|---|---|---|
| 1e-2 -> 1e-3 | 29.00 pp | 8.73 pp |
| 1e-3 -> 1e-4 | 20.51 pp | 6.18 pp |
| 1e-4 -> 1e-5 | 9.50 pp | 2.86 pp |
| 1e-5 -> 1e-6 | 3.75 pp | 1.13 pp |
| 1e-6 -> 1e-7 | 2.02 pp | 0.61 pp |
| 1e-7 -> 1e-8 | 0.91 pp | **0.27 pp** |

At ferric's production threshold (1e-7) a 2x tighter bound **cannot** buy more
than 0.61 pp, and at 1e-8 not more than 0.27 pp — before any question of whether
the tightening lands on the decisions near the cut. The measured 0.12-0.43 pp is
therefore not a disappointing fraction of a large opportunity; it is a **large
fraction of a small one**.

That is the finding: **at these sizes and thresholds the kept-work-versus-
threshold curve is too flat for any bound improvement to pay.** The screening
product `bound x weight` spans many decades across the pair-batch population, so
the decisions sitting within a factor of 2 of the threshold — the only ones a 2x
tighter bound can flip — are a thin shell of that population, and the shell gets
thinner as the threshold tightens.

**The artifact hypothesis for this section, checked.** If the flatness were an
artifact of my harness rather than a property of the screen, the slope would
depend on which bound generated it. It does not, to within scatter:

| decade | `ferric` | `ipb-flat` | `ipb-dist` |
|---|---|---|---|
| 1e-5 -> 1e-6 | 3.75 pp | 3.26 pp | 4.20 pp |
| 1e-6 -> 1e-7 | 2.02 pp | 1.95 pp | 1.98 pp |

Three bounds, one curve shape (the 1e-5 decade scatters +-13% about 3.7 pp, the
1e-6 decade to +-2%). The flatness is a property of the screening product's
distribution, not of any one estimate of it.

**This is the same shape as the two negatives before it, arrived at from a third
direction.** §6.2a held the bound fixed and varied the STRUCTURE: ~0 pp. §6.2b
held both fixed and varied the batch GEOMETRY: 0.12 pp for 8.2x the evaluations.
This study holds structure and geometry fixed and varies the BOUND: 0.12-0.43 pp.
Three orthogonal levers, three results in the same sub-1-pp band. The common
cause is now visible and it is none of the three levers: it is the flatness of
kept-work-versus-threshold, i.e. the *distribution* of the screening product.

**What this does NOT say.** It does not say the bound is useless — see §6.

---

## 6. Verdict — provisional, dated 2026-09-09, scoped as stated

Separated from the measurements above on purpose: the tables survive
re-interpretation, this section is the part most likely to be wrong later.

### 6.1 The deliverable question, answered

> At matched K accuracy, does the distance-dependent IPB keep less work than
> BOTH (a) ferric's current bound and (b) the batch-independent IPB?

**Yes on both, by 0.12-0.43 pp — which is inside the noise band the last study
established, so the honest answer is "yes, and it does not matter at these
sizes and thresholds".** It wins on every row of every matched-error table
measured, with no row where it loses; the win is real, consistent, and too small
to justify a port on its own.

### 6.2 What is established

1. **The formulation is correct and is the one the source describes.** The
   distance-dependent IPB is `min(V_0, sum_ab [S^ab_0/R_ab + V^ab_{R_ab}])`,
   built from Eq (A15) with each primitive's split taken about its own `P_ab`.
   Zero violations over 2.5M QZVP pair-probe checks with g functions, exact
   reduction to `ipb-flat` at `D=0`, four mutations each caught by the intended
   anchor.
2. **It is the tightest of the three bounds, by a wide and basis-growing
   margin**: 2.0x `ferric` and 29x `ipb-flat` at the median def2-QZVP probe.
   This settles §6.2a's lever 1 in the affirmative *as a statement about the
   bound*: the source's "very tight" is accurate.
3. **It substantially cures the degeneracy disease.** 48.9% vs `ferric`'s 69.8%
   at C12, a gap that widens monotonically with size (1.1 pp at C1 -> 20.9 pp at
   C12 -> 21.3 pp at C16). §6.2b's `|AB|/2` diagnosis is confirmed: a bound that
   never subtracts `|AB|/2` degenerates ~20 pp less on the same decisions.
4. **And none of that converts into kept work**, for the reason quantified in
   §5: at ferric's production threshold the slope of kept-work-versus-log-
   threshold caps ANY 2x bound improvement at 0.61 pp, and at 1e-8 at 0.27 pp.

### 6.3 The pre-registered kill criteria, applied honestly

The pre-registration said the lane closes as INERT if the degenerate fraction
lands in ferric's 45-80% band, if the median tightening over `ipb-flat` is below
1.2x, or if the kept-work reduction is under 1 pp at every size.

* Degeneracy: **48.9% at C12, 52.3% at C16 — inside the band.** Tripped, though
  at the band's bottom edge and ~20 pp better than the bound it is compared to.
* Median tightening over `ipb-flat`: **1.26x at C12** (grid (25,50)) — just over
  the 1.2x bar; but 1.00x at C1-C8, so the criterion is met only at the largest
  size measured. The A0 probe-set median (29x at QZVP) is a *different*
  measurement — over probe points, not over real batch decisions — and the two
  must not be conflated. On real decisions the distance factor is mostly idle.
* Kept work: **0.12-0.43 pp, under 1 pp at every size measured.** Tripped.

Two of three inert criteria are tripped outright and the third only clears at
the largest system. **By its own pre-registered standard this lane closes.**

### 6.4 The counter-argument, stated because it is not weak

The pre-registration also listed a scope limit that this study did not remove
and that materially affects the verdict (§6.2a's lever 3): **the IPB is cheaper
to EVALUATE than a sphere query.** `ipb-flat` is a single precomputed number per
shell pair. `ipb-dist` is a per-primitive loop with an incomplete-gamma
evaluation per decision — so it is *more* expensive than either, and this study
measures counts, not wall time. A bound that keeps 0.3 pp less work at higher
per-decision cost is a straightforward loss, and that is the likely reading.

The one direction that could still pay is the **cheap composite**: use
`ipb-flat`'s precomputed scalar with a *single* `1/D` factor rather than the full
per-primitive treatment. That is not what was measured here and its validity
would need its own A0.

### 6.5 What would reopen this

Not a tighter bound — §5 shows the ceiling is set by the threshold-response
curve, not by bound quality. What would reopen it is a regime where that curve
is **steep**: larger systems at looser effective thresholds, where the 1e-3 ->
1e-4 decade (20.5 pp) rather than the 1e-7 -> 1e-8 decade (0.91 pp) is the
operating point. Whether COSX ever runs there is a question about the method's
accuracy target, not about screening.

### 6.6 Scope limits (all of these bound the verdict)

* **Counts only, never wall time.** The one axis on which `ipb-dist` is
  unambiguously worse (evaluation cost) is invisible here.
* **Kept-work sweeps are STO-3G, C4 and C8 only.** The QZVP measurement is A0
  (tightness) only — no K build, so no kept-work number at the basis where the
  bound looks best. That is the single largest gap and it inverts nothing
  measured, but it is not measured.
* **Linear alkanes only.** Globular systems, where a batch's neighbours are
  denser, are untested; §6.2b flags the same limit.
* **One density weight** (`max(fmax[s1], fmax[s2])`), held fixed by design so
  the bound is the only variable. Interaction with the sn-LinK two-branch
  structure is not measured.
* The degeneracy measurement is **strongly grid-dependent** (§3): 15.8 pp at C12
  from the grid alone. Cross-grid comparisons in that table are invalid.
