# Results: a distance-dependent integral partition bound for 3c-1e / COSX

**2026-09-09.** Pre-registration: `scripts/queue/out/ipb_distance_prereg.md`
(committed before any number). Code: `scripts/ipb_distance_proto.py` (bounds,
anchors, K build, threshold sweep) and `scripts/ipb_distance_degeneracy.py`
(size-axis degeneracy/tightness, no SCF, no K build). Python/numpy/PySCF only,
one thread, deterministic counts.

Measurement is separated from interpretation below, per repo convention: §2–§5
are tables, §6 is the verdict and is explicitly provisional.

**One-line status: the bound is CORRECT and is the TIGHTEST of the three by
2-29x. It beats the batch-independent IPB that sn-LinK ships by 1.6-2.9 pp of
kept work at C8, growing with size — and it does NOT beat ferric's existing
Hölder sphere bound (0.17-0.57 pp, flat in size). Nothing to port; something to
know about sn-LinK.**

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

### alkane_8 / STO-3G, grid (25,50), E_grid = 3.15e-03 — THE DECIDING TABLE

Raw output: `scripts/queue/out/ipb_c8_sweep_raw.txt`. 19.9 Bohr, i.e. past the
C4 regime where §5.3 shows the ceiling is artificially low. (The first attempt
at this point did not complete; `scripts/queue/ipb_c8_sweep.sh` was rewritten to
skip the redundant anchors, which is what made it fit.)

| bound | 1e-4 | 1e-5 | 1e-6 |
|---|---|---|---|
| `ferric` kept-wt % / K err | 42.476 / 6.49e-4 | 55.061 / 3.93e-5 | 65.846 / 5.01e-6 |
| `ipb-flat` | 45.314 / 5.97e-4 | 57.554 / 3.70e-5 | 67.214 / 4.62e-6 |
| `ipb-dist` | 40.534 / 6.86e-4 | 53.494 / 5.33e-5 | 64.652 / 5.95e-6 |
| `min(ferric,ipb-dist)` | 40.529 / 6.86e-4 | — | — |

kept-work % (weighted), log-interpolated to equal K error:

| K error | `ferric` | `ipb-flat` | `ipb-dist` | **D vs ferric** | **D vs flat** |
|---|---|---|---|---|---|
| 1e-04 | 50.871 | 53.179 | **50.300** | **−0.571 pp** | **−2.879 pp** |
| 5e-05 | 53.982 | 56.232 | **53.816** | **−0.167 pp** | **−2.416 pp** |
| 1e-05 | 62.225 | 63.629 | **62.009** | **−0.216 pp** | **−1.620 pp** |

**This splits the deliverable question in two, and the two halves answer
differently:**

* **vs `ipb-flat` (the form sn-LinK actually ships): −1.62 to −2.88 pp.** That
  is 2-5x outside the ±0.5-0.83 pp band in which §6.2a found its two bounds
  indistinguishable, and it *grew* from ~0.4 pp at C4. The distance factor is
  worth real work against the batch-independent IPB, and increasingly so with
  size.
* **vs `ferric` (what ferric has today): −0.17 to −0.57 pp.** Still inside the
  band, and essentially unchanged from C4 despite the 3.3x larger ceiling. The
  distance-dependent IPB does **not** beat ferric's Hölder sphere bound by an
  amount that matters, at either size measured.

`min(ferric, ipb-dist)` lands on `ipb-dist` to 3 decimals (40.529 vs 40.534), so
`ipb-dist` dominates ferric's bound essentially everywhere — it is simply not
enough tighter, on the decisions near the cut, to move the screen.

**The C4 -> C8 comparison is itself the control for §5.3's ceiling argument.**
Same three error targets, same interpolation, both sizes:

| K error | D vs `ferric` @C4 | @C8 | D vs `ipb-flat` @C4 | @C8 |
|---|---|---|---|---|
| 1e-4 | −0.703 pp | −0.571 pp | −0.807 pp | **−2.879 pp** |
| 5e-5 | −0.084 pp | −0.167 pp | −0.012 pp | **−2.416 pp** |
| 1e-5 | −0.330 pp | −0.216 pp | −0.428 pp | **−1.620 pp** |

The ceiling tripled from C4 to C8. **The `ipb-flat` margin grew 3.6x-200x with
it; the `ferric` margin did not move at all** (it is within its own scatter, and
at 5e-5 it even shrinks). Two bounds, the same ceiling increase, opposite
responses — which is what turns this from "the margin is small" into a
mechanism:

* the ceiling is **real and reachable** — `ipb-flat` reaches a large fraction of
  it, so nothing about the harness or the interpolation is suppressing margins;
* ferric's Hölder sphere bound is **already close to `ipb-dist` on the decisions
  near the cut**, notwithstanding being 2x looser at the median probe and
  degenerate 20 pp more often. Median tightness and degeneracy are the wrong
  statistics for predicting screening work; what matters is agreement in the
  thin shell around the threshold, and there the two agree.

That is a much more specific finding than "bounds don't matter", and it is also
a warning about §2.0 and §3: **a bound can be 2x tighter at the median and 20 pp
less degenerate and still be worth 0.2 pp.**

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

**A limit of the ceiling formula, found later and recorded here so §5.3 is not
read as stronger than it is (see §6.5a).** "A `g`-times tighter bound equals a
threshold raised by `g`" is exact only for a **uniform** tightening. For
`ipb-dist` vs `ipb-flat` the tightening is concentrated in the upper tail
(`p50 = 1.00x`, `p90 = 9.67x` at C8), and feeding the median gives a ceiling of
0.00 pp against a measured margin of 2.4 pp — the formula is simply violated
there. It remains serviceable for the `ferric` comparison, whose tightening is
closer to uniform. **Use the slope table below to compare sizes; do not use it
to predict a margin for a non-uniformly tighter bound.**

**Why the slope steepens** (mechanism, offered as interpretation not
measurement): at C4 the screen keeps 97.8% at 1e-7 — nearly everything is above
the cut, so almost no population sits near it and the curve is against its
ceiling. At C12 it keeps 49.6%, i.e. the threshold is cutting through the *bulk*
of the screening-product distribution, which is exactly where the derivative is
largest. Flatness at C4 was a saturation artifact of a screen that had almost
nothing left to drop.

### 5.4 What the three negatives now look like together

The claim that three levers share one cause does **not** survive, and the reason
is sharper than "this one wasn't measured at size":

| study | lever varied | result | measured at |
|---|---|---|---|
| §6.2a | screening STRUCTURE | ~0 pp | C1-C16, STO-3G |
| §6.2b | batch GEOMETRY | 0.12 pp for 8.2x cost | C4-C16, def2-SVP |
| this | the integral BOUND, vs `ferric` | 0.17-0.70 pp | C4 **and C8** |
| this | the integral BOUND, vs `ipb-flat` | **1.62-2.88 pp at C8** | C4 and C8 |

The bound lever produces **both** a sub-1-pp result and a clearly-outside-the-band
result, depending on which bound it is measured against. So "the lever doesn't
matter" is not a property of the lever at all — it is a property of the
*comparator*. ferric's existing bound happens to be good where it counts;
sn-LinK's happens not to be.

A single-cause story across three studies would have been tidy, and §4's C4->C8
control is what refused it: the same ceiling increase moved one margin 3.6-200x
and the other not at all. **Too clean is a stop condition, and this was the
stop.**

---

## 6. Verdict — provisional, dated 2026-09-09, scoped as stated

Separated from the measurements above on purpose: the tables survive
re-interpretation, this section is the part most likely to be wrong later.

### 6.1 The deliverable question, answered

> At matched K accuracy, does the distance-dependent IPB keep less work than
> BOTH (a) ferric's current bound and (b) the batch-independent IPB?

**The question was mis-posed as one question. It is two, and they answer
differently.**

| comparison | C4 (10.5 Bohr) | C8 (19.9 Bohr) | verdict |
|---|---|---|---|
| vs `ipb-flat` — the bound **sn-LinK ships** | 0.01-0.81 pp | **1.62-2.88 pp** | **YES, and growing with size.** Clears the ±0.5-0.83 pp band by 2-5x. |
| vs `ferric` — the bound **ferric has today** | 0.08-0.70 pp | 0.17-0.57 pp | **NO.** Inside the band at both sizes, and flat in size. |

So:

* **Against the literature bound, the distance factor is worth real work.**
  §6.2a's lever 1 asked whether an exact radial treatment plus a valid distance
  factor would dominate the batch-independent IPB. It does: 2.4 pp at C8, and
  the margin grew 3.6-200x when the ceiling tripled, so it is tracking the
  ceiling rather than sitting at noise.
* **Against ferric's existing Hölder sphere bound, it is not.** ferric's bound
  is 2x looser at the median probe (§2.0) and degenerate 20 pp more often (§3),
  and *none of that matters*, because the two agree on the thin shell of
  decisions near the threshold. The `ferric` margin did not respond at all to a
  3.3x larger ceiling, while `ipb-flat`'s did — that contrast is the evidence,
  not the small number by itself.

**Practical reading: there is nothing here worth porting into ferric, and there
is something here worth knowing about sn-LinK.** ferric's bound is not the weak
link its median-tightness and degeneracy statistics suggest.

An earlier draft closed this lane on the C4 number plus a claim that the
threshold-response flatness generalised. That claim was measured and **refuted**
(§5.2, §5.3), and the C8 measurement it motivated is what split the question in
two. Both the withdrawn claim and the refutation are kept.

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
4. **It converts into kept work against `ipb-flat` (1.6-2.9 pp at C8, growing
   with size) and not against `ferric` (0.17-0.57 pp, flat in size).** The
   conversion is comparator-dependent, not size-dependent, which the C4->C8
   control establishes by moving one margin and not the other under the same
   3.3x ceiling increase (§4).
5. **Median tightness and degeneracy are poor predictors of screening work.**
   `ipb-dist` is 2x tighter at the median probe than `ferric` and degenerate
   20 pp less often, and is worth 0.2 pp against it. Only agreement in the thin
   shell of decisions near the threshold matters. This is the most reusable
   thing in the study and it invalidates the shortcut of ranking bounds by
   median tightness — which is what §2.0 and §3 would otherwise invite.

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
* Kept work: **vs `ferric` 0.17-0.70 pp at C4 and C8 — under 1 pp at every size
  measured, TRIPPED. vs `ipb-flat` 1.62-2.88 pp at C8 — NOT tripped.** The
  criterion did not anticipate that the answer would depend on which of the two
  baselines it was measured against, and it has no rule for that case.

**Verdict on the criteria: they were written for a one-baseline question and the
result is two-baseline.** Applied literally to the ferric comparison, all three
trip and the lane closes. Applied to the `ipb-flat` comparison, kept work
clears comfortably and degeneracy is the only trip.

Two of the three criteria are also weaker than they looked when written:

* **The degeneracy band was a proxy, and it disagrees with what it proxies for.**
  It was written as "would mean the new bound inherits the same disease", i.e.
  as a stand-in for the kept-work question. `ipb-dist` sits inside the 45-80%
  band while degenerating 20 pp LESS than the bound whose numbers defined that
  band — and separately, §4 shows degeneracy does not predict kept work at all.
  An absolute band is the wrong test for a quantity whose comparator moved.
* **The 1.2x median-tightening bar is measured over the wrong population.** It
  averages over all (pair, batch) decisions, and §4 shows only the shell near
  the threshold matters.

Both are recorded as design faults of the pre-registration rather than being
quietly reinterpreted after the fact. The pre-registration itself is unedited.

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

### 6.5 The pre-registered C8 prediction, and how it scored

Recorded because it was written before the C8 run and is a fair scorecard.

> **Prediction (§6.5, pre-C8):** if `ipb-dist`'s margin over `ferric` scales with
> the ceiling it should be ~0.4-1.4 pp at C8-C12, straddling the band. Margin
> <= 0.5 pp -> the bound lever is dead. Margin >= 1 pp -> it clears the band.

**Measured at C8: 0.17-0.57 pp against `ferric`.** The prediction's low branch,
so by its own stated rule *the bound lever is dead against ferric's bound* — and
the margin did not scale with the ceiling at all, which the prediction had
assumed it would either do or fail to do for want of headroom. It had headroom
(§4's control shows `ipb-flat` used it) and still did not move.

**What the prediction missed entirely: the `ipb-flat` comparison.** It was framed
as one margin, against ferric, because the pre-registration framed the whole
study that way. The 1.6-2.9 pp result against the batch-independent IPB was not
predicted, is the larger effect, and is the part with external relevance.
*A pre-registration fixes the analysis but cannot fix the framing, and a
one-baseline frame hid the study's own main result until the data forced it.*

### 6.5a The remaining unmeasured axis: def2-QZVP kept work

Two ceilings
there, at the plateau slope of ~6.6 pp/decade, using the §2.0 QZVP tightness
ratios:

| comparison | tightness `g` | `log10(g)` | ceiling at the plateau slope |
|---|---|---|---|
| `ipb-dist` vs `ferric` | 2.0x | 0.30 | **~2.0 pp** |
| `ipb-dist` vs `ipb-flat` | 29x | 1.46 | **~9.6 pp** |

The second row is the one that matters — it is the sn-LinK comparison, since the
batch-independent IPB is what that method ships.

**A caution about these ceilings, found by checking them.** The ceiling formula
assumes a *uniform* `g`-times tightening, and `ipb-dist` vs `ipb-flat` is
strongly non-uniform: on real C8 (pair, batch) decisions the tightening is
`p50 = 1.00x` and `p90 = 9.67x` (§3). Feeding the median `g = 1.00` gives a
ceiling of **0.00 pp** against a measured margin of **2.4 pp** — the formula is
violated, because all of the tightening lives in the upper tail, exactly where
the threshold-crossing decisions are. Feeding `g = p90 = 9.67` gives 6.6 pp and
the margin is 36% of it; feeding the QZVP probe-median 29x gives 9.8 pp and 24%.

So the honest statement is: **the ceiling arithmetic is a valid bound only for a
uniform tightening, and is not a reliable predictor for this bound.** It was
adequate for the `ferric` comparison (where the tightening is closer to uniform
and the margin duly sat far below the ceiling) and it is not adequate here. The
QZVP number should be measured, not extrapolated — and the extrapolations above
are recorded with this caveat attached rather than deleted, because the failure
of the formula is itself informative about where a bound's benefit comes from.

**What to do next:** measure `ipb-flat` vs `ipb-dist` kept work at def2-SVP and
def2-QZVP. It bears on whether a distance factor is worth adding to sn-LinK —
not on ferric, where §6.1 already answers no.

### 6.6 Scope limits (all of these bound the verdict)

* **Counts only, never wall time.** The one axis on which `ipb-dist` is
  unambiguously worse (evaluation cost) is invisible here.
* **The kept-work sweep covers TWO systems, C4 and C8 (10.5 and 19.9 Bohr),
  STO-3G only.** C12/C16 were not run with a K build. 19.9 Bohr is still below
  the ~30 Bohr onset this repo cites, so the size axis of the kept-work
  measurement is short. What partly covers it: the C4->C8 step spans a 3.3x
  change in the threshold-response slope (§5.3) and moves the two margins in
  opposite ways, which is the discriminating comparison; and the slope has
  already plateaued by C8, so C12/C16 are not expected to differ in kind.
  Expected, not measured.
* **No kept-work number at def2-QZVP**, the basis where the bound looks best by
  a factor of 29 and the only basis regime where COSX wins at all. The QZVP
  measurement here is A0 (tightness) only — no K build. This is the study's
  largest remaining gap. It does not threaten §6.1's `ferric` answer (that
  margin is flat in both size and, per §2.0, roughly flat in basis: 2.0x at
  QZVP vs 1.8x at SVP). It could substantially change the `ipb-flat` number,
  which is the one with external relevance.
* **Linear alkanes only.** Globular systems, where a batch's neighbours are
  denser, are untested; §6.2b flags the same limit.
* **One density weight** (`max(fmax[s1], fmax[s2])`), held fixed by design so
  the bound is the only variable. Interaction with the sn-LinK two-branch
  structure is not measured.
* The degeneracy measurement is **strongly grid-dependent** (§3): 15.8 pp at C12
  from the grid alone. Cross-grid comparisons in that table are invalid.
