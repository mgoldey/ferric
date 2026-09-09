# Results: a distance-dependent integral partition bound for 3c-1e / COSX

**2026-09-09.** Pre-registration: `scripts/queue/out/ipb_distance_prereg.md`
(committed before any number). Code: `scripts/ipb_distance_proto.py` (bounds,
anchors, K build, threshold sweep) and `scripts/ipb_distance_degeneracy.py`
(size-axis degeneracy/tightness, no SCF, no K build). Python/numpy/PySCF only,
one thread, deterministic counts.

Measurement is separated from interpretation below, per repo convention: §2–§5
are tables, §6 is the verdict and is explicitly provisional.

**One-line status: the bound is CORRECT, is the TIGHTEST of the three by 2-29x,
and the lane is UNDECIDED — not closed.** The kept-work measurement that would
decide it was made only at alkane_4 (10.5 Bohr), which §5.3 shows is the size at
which the answer is structurally smallest. §6.5 names the single run that
settles it.

**§5 contains a withdrawn conclusion.** An earlier draft closed this lane on a
"the threshold-response curve is too flat for any bound to pay" argument; that
argument was measured and refuted (the slope triples from C4 to C8). The
withdrawn version is kept in §5.1-§5.2 rather than deleted, per the whitepaper's
own convention, because the reason it was wrong is the most transferable part.

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

### alkane_8 / STO-3G — NOT MEASURED, and why it is recorded rather than omitted

The C8 kept-work sweep was attempted and did not complete inside this session's
five-minute-per-script budget (the shared box was at 18-19 of 23 GB with another
agent's job resident, and `run()`'s A0 anchor at C8 — 903 pairs x 197 probes x 3
bounds — costs more than the sweep it precedes). What it did produce before
being killed: `nbf=58 nsh=42 batches=127 pairs=903 diameter=19.87 Bohr`,
`E_grid = 3.148e-03`. No kept-work rows.

`scripts/queue/ipb_c8_sweep.sh` was rewritten to skip the anchors (they add
nothing at C8 over the C4/def2-QZVP runs, which already cover l up to 4) so this
point is cheap to obtain later. **The size axis of the kept-work measurement is
therefore a single system, C4 at 10.5 Bohr, which is below the ~30 Bohr onset
this repo requires before declaring a locality negative.**

That gap turned out to be decisive rather than cosmetic: §5.3 measures the
threshold-response slope at C4/C8/C12/C16 and finds it **triples** past C4, so
C4 is the size at which any bound improvement has the smallest possible ceiling.
The §3 degeneracy axis and the §5.3 slope axis both run to C16; only the
kept-work axis stops at C4, and that is precisely the axis the deliverable
question asks about. See §6.5 for the single run that closes the gap.

### water / cc-pVDZ, grid (50,110)

Reported for completeness and as a scope limit, not as evidence: at 2.86 Bohr
the screen is vacuous below t=1e-4 (all four bounds keep 100.000% at every
threshold from 1e-5 down, K error exactly 0), so the matched-error interpolation
has no bracket and every entry is `--`. Water cannot discriminate these bounds.

---

## 5. The mechanism — and the size-dependence that REFUTES its first version

> **Read §5.3 before citing §5.1.** The first version of this section concluded
> that the threshold-response curve is too flat for any bound improvement to
> pay, and treated that as a general mechanism. It is **not** general: the slope
> it rests on **triples from C4 to C8** (2.02 -> 6.73 pp/decade). §5.1-§5.2 are
> kept verbatim as the C4 measurement they always were; §5.3 is the correction.
> This is exactly the failure mode the repo's protocol warns about — a
> construction-scale observation reproducing cleanly and being mistaken for a
> mechanism — and it was caught only because §6.4a wrote down the artifact
> hypothesis and then went and measured it.

### 5.1 The C4 observation (correct as measured, wrong as generalised)

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

### 5.2 The reading that followed from it, now WITHDRAWN

The conclusion drawn from §5.1 was: *"three orthogonal levers — structure
(§6.2a), batch geometry (§6.2b), and now the bound — all land in the same
sub-1-pp band, and the common cause is the flatness of kept-work-versus-
threshold."* That reading is **withdrawn**. The flatness is not a property of
COSX screening; it is a property of **alkane_4**.

### 5.3 The correction: the slope TRIPLES from C4 to C8 and keeps rising

`scripts/queue/ipb_slope.sh` measures d(kept-work)/d(log t) directly, with no K
build (the screen's decisions need only `F = D X`), which makes the size axis
cheap enough to reach where the kept-work sweep could not. ferric's bound,
STO-3G, grid (25,50), weighted kept work %:

| system | diameter | 1e-4 | 1e-5 | 1e-6 | 1e-7 | **slope over last decade** | ceiling for a 2.0x bound |
|---|---|---|---|---|---|---|---|
| alkane_4 | 10.46 | 82.502 | 92.004 | 95.758 | 97.781 | **2.02 pp/dec** | 0.61 pp |
| alkane_8 | 19.87 | 42.476 | 55.061 | 65.846 | 72.572 | **6.73 pp/dec** | 2.02 pp |
| alkane_12 | 29.28 | 23.075 | 32.456 | 42.190 | 49.631 | **7.44 pp/dec** | 2.24 pp |
| alkane_16 | 38.70 | 14.472 | 20.973 | 28.366 | 34.938 | **6.57 pp/dec** | 1.98 pp |

(The C4 row reproduces the full-K-build sweep of §4 to the last printed digit —
82.502 / 92.004 / 95.758 / 97.781 — which cross-validates the no-K-build path
against the one that assembles the matrix.)

**The slope rises 3.3x from C4 to C8, then PLATEAUS: 6.73 / 7.44 / 6.57 at
C8 / C12 / C16.** The C4 value is the outlier, not the trend; from C8 onward
(19.9 Bohr and up, i.e. spanning the ~30 Bohr onset in both directions) the
slope sits at 6.5-7.5 pp/decade with no drift, and the C12->C16 step is
*negative*. That non-monotonicity is what makes this a plateau rather than
growth, and it is worth more than a monotone series would be: a construction
artifact would not conveniently level off and then dip.

So: the asymptotic ceiling for a 2x tighter bound is **~2.0 pp**, roughly 3.3x
the C4 figure, established across three sizes rather than extrapolated from two.
Consequently:

* The §5.1 arithmetic is still an identity and still correct. What was wrong was
  the *number fed into it*. Past the onset a 2x tighter bound can buy up to
  **~2.0 pp**, not 0.61 pp — and that is outside the ±0.5-0.83 pp band in which
  the previous studies found their bounds indistinguishable. It is a ceiling,
  not a prediction; but the ceiling is the thing §5.1 claimed was closed, and it
  is not.
* alkane_4, at 10.5 Bohr, is **below the ~30 Bohr onset** this repo requires
  before a locality negative may be declared. The rule existed precisely for
  this, and the first version of §5 broke it.

**Why the slope steepens** (mechanism, offered as interpretation not
measurement): at C4 the screen keeps 97.8% at 1e-7 — nearly everything is above
the cut, so almost no population sits near it and the curve is against its
ceiling. At C12 it keeps 49.6%, i.e. the threshold is cutting through the *bulk*
of the screening-product distribution, which is exactly where the derivative is
largest. Flatness at C4 was a saturation artifact of a screen that had almost
nothing left to drop.

### 5.4 What the three negatives now look like together

The claim that three levers share one cause does **not** survive:

| study | lever varied | result | measured at |
|---|---|---|---|
| §6.2a | screening STRUCTURE | ~0 pp | C1-C16, STO-3G |
| §6.2b | batch GEOMETRY | 0.12 pp for 8.2x cost | C4-C16, def2-SVP |
| this | the integral BOUND | 0.12-0.43 pp | **C4 only**, STO-3G |

The first two carry a size axis; this one does not. Its sub-1-pp result sits at
the size where §5.3 shows the ceiling is *lowest*, so it cannot be pooled with
them as a third sighting of a common cause. **The correct statement is that the
bound lever is unmeasured where it matters**, and §5.3 gives a concrete reason
to expect a larger answer there — up to ~2.2 pp at C12, ~4x the C4 ceiling.

---

## 6. Verdict — provisional, dated 2026-09-09, scoped as stated

Separated from the measurements above on purpose: the tables survive
re-interpretation, this section is the part most likely to be wrong later.

### 6.1 The deliverable question, answered

> At matched K accuracy, does the distance-dependent IPB keep less work than
> BOTH (a) ferric's current bound and (b) the batch-independent IPB?

**Yes on both, by 0.12-0.43 pp — but ONLY alkane_4 / STO-3G was measured, and
§5.3 shows that is the size at which the answer is structurally smallest.
This lane is therefore NOT CLOSED. It is UNDECIDED, with the decisive
measurement identified and cheap.**

The reasoning, in the order it has to be read:

1. `ipb-dist` wins every matched-error row measured, never loses, and is a
   valid bound over 2.5M checks including g functions (§2, §2.0). None of that
   is in question.
2. Its measured margin, 0.12-0.43 pp, is inside the ±0.5-0.83 pp band in which
   §6.2a already found two other bounds indistinguishable — which is what
   originally read as a close.
3. **But the margin has a size-dependent ceiling, and C4 is where that ceiling
   is lowest.** The slope of kept-work-versus-log-threshold triples from C4 to
   C8 and then plateaus (2.02 -> 6.73 / 7.44 / 6.57 pp/decade at C4/C8/C12/C16,
   §5.3). A 2x tighter bound is capped at 0.61 pp at C4 and at **~2.0 pp** past
   the onset.
4. So the one number the deliverable question turns on was measured at the one
   size where it is guaranteed to look worst, and the honest reading is that
   **it is not yet known** whether `ipb-dist` clears the noise band at C8+.

An earlier draft of this file closed the lane on the C4 number plus a claim that
the flatness generalised. That claim was measured and **refuted** (§5.2, §5.3).
The refutation is recorded rather than the file being quietly rewritten, because
the near-miss is the most transferable thing here: a sub-onset measurement that
reproduced cleanly across three bounds looked exactly like a mechanism.

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
4. **At C4 none of that converts into kept work** — 0.12-0.43 pp — **and §5.3
   shows C4 is the size at which it structurally cannot.** The ceiling there is
   0.61 pp; past the onset it is ~2.0 pp. Whether the conversion happens at C8+
   is the open question.

### 6.3 The pre-registered kill criteria, applied honestly

The pre-registration said the lane closes as INERT if the degenerate fraction
lands in ferric's 45-80% band, if the median tightening over `ipb-flat` is below
1.2x, or if the kept-work reduction is under 1 pp at every size.

* Degeneracy: **48.9% at C12, 52.3% at C16 — inside the band.** Tripped, though
  at the band's bottom edge and ~20 pp better than the bound it is compared to
  on the same decisions.
* Median tightening over `ipb-flat`: **1.26x at C12** (grid (25,50)) — just over
  the 1.2x bar; but 1.00x at C1-C8, so the criterion clears only at the largest
  size measured. The A0 probe-set median (29x at QZVP) is a *different*
  measurement — over probe points, not over real batch decisions — and the two
  must not be conflated. On real decisions the distance factor is mostly idle.
* Kept work: 0.12-0.43 pp, under 1 pp — **but measured at one size, and the
  criterion says "at every size".** With only C4 in hand this criterion is
  **not evaluable**, and §5.3 shows C4 is not representative. Not tripped; not
  cleared; unmeasured.

**Two of three criteria are tripped; the third — the one the deliverable
question actually turns on — was not measured at a size where it could be.**
Under a strict reading of the pre-registration the lane closes on degeneracy
alone. Under an honest one it does not, because the degeneracy criterion was
written as a *proxy* for the kept-work question ("would mean the new bound
inherits the same disease"), and the proxy is now known to disagree with the
thing it proxies for: `ipb-dist` degenerates 20 pp LESS than `ferric` while
sitting in the same nominal band, and the band's boundaries were set from
`ferric`'s numbers before any of this was measured.

I am recording that as **UNDECIDED rather than CLOSED**, and flagging the
pre-registration's degeneracy criterion as poorly designed in hindsight: an
absolute band is the wrong test for a quantity whose comparator moved.

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

### 6.4a How the size-dependence question was asked, and answered

Kept as a record of method, because writing this section is what caught the
error. It was written while the verdict still read "closed", explicitly as the
load-bearing assumption of that close, with two disagreeing predictions
side by side:

* *Flatter at size* (the close survives): as a system grows, the pair-batch
  population fills with distant pairs whose screening product is many decades
  below the threshold — the whitepaper §6.2b "reading trap" paragraph observes
  exactly this filling. A population pushed away from the cut means a thinner
  shell near it, hence a flatter slope.
* *Steeper at size* (the close is wrong): ferric's kept fraction falls
  0.792 -> 0.486 -> 0.307 across C4/C8/C12 at fixed threshold, so the
  distribution is not merely translating — mass is crossing the cut, and a
  distribution actively crossing the threshold has a large derivative there.

**Measured: the second. 2.02 -> 6.73 / 7.44 / 6.57 pp/decade** (§5.3). The close
did not survive its own stated assumption, and the run that settled it took
under five minutes because it needs no K build.

The transferable part: the first reading was the one supported by an existing,
correct, cited observation from a previous study — the population *does* fill
with distant pairs. It is just that the filling and the crossing are different
things, and only the crossing sets the derivative. **A mechanism borrowed from a
neighbouring result is still an untested hypothesis in the new setting.**

### 6.5 What would settle this — one measurement, already scripted

Run the §4 matched-error sweep at **C8 and C12** (and, if affordable,
def2-SVP rather than STO-3G). `scripts/queue/ipb_c8_sweep.sh` does exactly this
and skips the redundant anchors; it did not complete here only because the box
was shared with another agent's job.

The prediction to test, stated now so it cannot be chosen afterwards: **if
`ipb-dist`'s margin over `ferric` scales with the ceiling, it should be
~0.4-1.4 pp at C8-C12** (the C4 margin of 0.12-0.43 pp times the 3.3x ceiling
increase). That straddles the ±0.5-0.83 pp indistinguishability band, so the
measurement genuinely discriminates:

* margin <= 0.5 pp at C12 -> the bound lever is confirmed dead, lane CLOSES,
  and the §5.1 reading is rehabilitated as a conclusion for a different reason;
* margin >= 1 pp at C12 -> `ipb-dist` clears the band that §6.2a's two bounds
  never did, and the lane becomes a wall-time question (§6.4) rather than a
  screening one.

Beyond that, the second unmeasured axis is **def2-QZVP kept work**. Two ceilings
there, at the plateau slope of ~6.6 pp/decade, using the §2.0 QZVP tightness
ratios:

| comparison | tightness `g` | `log10(g)` | ceiling at the plateau slope |
|---|---|---|---|
| `ipb-dist` vs `ferric` | 2.0x | 0.30 | **~2.0 pp** |
| `ipb-dist` vs `ipb-flat` | 29x | 1.46 | **~9.6 pp** |

The second row is the striking one, and it is the sn-LinK comparison
specifically — the batch-independent IPB is what that method ships, so a port
that took `ipb-flat` and added the distance factor has a ~9.6 pp ceiling at
QZVP against a ~3.0 pp ceiling at C4. Both are ceilings, not predictions, and
neither is likely achievable in full (the tightening is not uniform near the
cut). But the QZVP one is an order of magnitude above anything measured here,
at the basis where COSX wins at all (whitepaper §2), and leaving it unmeasured
is the main reason this lane cannot be closed today.

### 6.6 Scope limits (all of these bound the verdict)

* **Counts only, never wall time.** The one axis on which `ipb-dist` is
  unambiguously worse (evaluation cost) is invisible here.
* **The kept-work sweep is ONE system: alkane_4 / STO-3G, 10.5 Bohr.** C8 did
  not complete (see §4); C12/C16 were never attempted. 10.5 Bohr is **below the
  ~30 Bohr onset** at which this repo requires a locality negative to be
  measured, so by the repo's own standard the kept-work half of this verdict is
  a pre-onset measurement. §3's degeneracy axis and §5.3's slope axis both run
  to C16 and both say the C4 point is unrepresentative — the slope alone triples
  — so this is not a gap that can be argued around. **It is the reason the
  verdict is UNDECIDED rather than closed.**
* **No kept-work number at def2-QZVP**, the basis where the bound looks best by
  a factor of 29 and the only basis regime where COSX wins at all. The QZVP
  measurement here is A0 (tightness) only — no K build. Together with the
  previous bullet this is the study's largest gap: **the two axes on which
  `ipb-dist` looks strongest (large systems, high angular momentum) are exactly
  the two on which kept work was not measured.** A reader who wants to overturn
  §6.1 should start there rather than re-deriving the bound.
* **Linear alkanes only.** Globular systems, where a batch's neighbours are
  denser, are untested; §6.2b flags the same limit.
* **One density weight** (`max(fmax[s1], fmax[s2])`), held fixed by design so
  the bound is the only variable. Interaction with the sn-LinK two-branch
  structure is not measured.
* The degeneracy measurement is **strongly grid-dependent** (§3): 15.8 pp at C12
  from the grid alone. Cross-grid comparisons in that table are invalid.
