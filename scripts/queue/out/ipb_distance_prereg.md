# Pre-registration: a distance-dependent integral partition bound for 3c1e / COSX

**Written 2026-09-09, BEFORE any number was measured.** Companion script:
`scripts/ipb_distance_proto.py`. Results: `scripts/queue/out/ipb_distance_results.md`.

Repo rule being honoured: exactness anchor first, artifact hypothesis stated
next to the physics hypothesis, and a written statement of what result would
mean the bound is invalid or inert — all before the sweep runs.

---

## 1. The question

Whitepaper §6.2b identified the open lever: ferric's coarse sphere bound
"collapses the shell pair onto its midpoint" and subtracts `|AB|/2` from the
distance, which **eats the distance entirely on 29–37% of pair-batch
decisions**, a share that GROWS with system size. §6.2a found the complementary
weakness on the other side: the sn-LinK form of the integral partition bound
(IPB) is *deliberately batch-independent* and **carries no distance decay at
all**, though it treats the radial moment exactly via Newton's shell theorem.

Two bounds, loose in complementary places, measured to agree within ±0.5 pp.
The obvious third construction — **exact radial treatment PLUS a valid distance
factor** — is the subject of this study. Thompson & Ochsenfeld describe it but
do not implement it; Laqua's sn-LinK paper says it exists and calls the
resulting bounds "very tight but also necessarily batch-dependent"
(thompson.txt:11208–11209), then declines it for GPU-simplicity reasons.

**The deliverable question.** At matched K accuracy, does a distance-dependent
IPB keep less (shell-pair, grid-batch) work than BOTH (a) ferric's Hölder
sphere bound and (b) the batch-independent IPB?

---

## 2. The formula, stated before it is coded

### 2.1 What went wrong last time, and why this is a different object

The previous attempt's exactness anchor rejected its distance form with
3678–8758 violations. The diagnosis in the whitepaper is correct and I restate
it so I cannot repeat it: **Eq (A10)'s `R` is the PARTITIONING-BALL RADIUS.**

    R_ab = max(0, R − |P_ab − C_µν|)                      (A10)

`S_R` (A11/A14) and `V_R` (A16) are *tail* quantities: they integrate |Ω| only
over the complement of a ball of radius `R` about the pair center `C_µν`. `V_R`
bounds the potential *of the charge lying outside that ball*, maximised over ALL
probe points in R³. It is NOT "the potential at distance R". Substituting a
probe distance for `R` and using `V_R` alone therefore DROPS the charge inside
the ball, which is most of it — hence underestimation, hence the violations.

### 2.2 The correct construction: Newton's shell theorem, used as written

The IPB paper's own Eq (A15) is the missing half. For any spherically symmetric
`S` about a point `p`, and any probe `r'` at distance `D = |r' − p|` (writing the
integrals with origin at `p`):

    ∫ S(r)/|r − r'| dr  =  (1/D) ∫_{|r| ≤ D} S(r) dr  +  ∫_{|r| > D} S(r)/|r| dr    (A15)

Read the two terms. The first is the *inside* charge, which Newton's theorem
places entirely at the center, giving an exact `1/D` decay. The second is
*exactly* `V_D` — the tail potential of Eqs (3.2)/(A16), with its `R` correctly
being a partitioning-ball radius. Both terms are present. Nothing is dropped.

Bounding the inside charge by the TOTAL absolute charge `S_0 = S_{R=0}`
(Eq A11/A17 at `R = 0`, which is a valid over-estimate since `S ≥ 0`) gives the
bound this study implements:

    **IPB-D(pair, D)  =  min( V_0 ,  S_0 / D  +  V_D )**                          (★)

where
  * `S_0` = the R=0 absolute-overlap factor (Eq A11 for same-centre SHG /
    Eq A14 otherwise, at R = 0),
  * `V_D` = the tail potential, Eq (A16), with partitioning radius `R = D`,
    hence `R_ab = max(0, D − |P_ab − C_µν|)` — Eq (A10) used for the thing it
    actually is,
  * `V_0` = Eq (A16) at `R = 0`, which IS the batch-independent sn-LinK bound
    `Ξ_νλ` of that paper's Eq (15),
  * `D` = a rigorous LOWER bound on the distance from the pair center `C_µν` to
    any probe point in the query region (a grid batch's bounding sphere:
    `D = max(0, |C_µν − centre_b| − radius_b)`).

Note what (★) does NOT do: it does not subtract `|AB|/2`. The shell-pair extent
is handled *inside* `V_D` and `S_0`, exactly, through `R_ab` — every primitive
pair's own center `P_ab` gets its own shifted radius. That is precisely the
"treat the pair extent exactly instead of halving it" step §6.2b names as the
lever. `C_µν = P_min` (the smallest-combined-exponent primitive's center) per
the paper's own choice, motivated there by the outer region being dominated by
that primitive.

### 2.3 Two properties that must hold by construction, and will be asserted

* **Trivial limit.** `D = 0` ⇒ `S_0/D = ∞` ⇒ (★) returns `V_0` exactly, the
  batch-independent IPB. More generally `V_D ≤ V_0` and the `min` is present, so
  (★) can NEVER be worse than the batch-independent bound at any distance. The
  anchor asserts bit-equality at `D = 0` and `IPB-D ≤ IPB-0` everywhere.
* **Never underestimates.** Every step above is an equality (A15) or an
  over-estimate (`S_inside ≤ S_0`; the cGTO→pGTO absolute-value inequality A5;
  the triangle-inequality/`F_tu` expansion A12–A13; `M̃_lj` angular bound). Any
  violation is an implementation bug, not a formulation limit.

---

## 3. Hypotheses — physics and artifact, side by side (repo rule)

| | if the bound is REAL | if my implementation is BROKEN |
|---|---|---|
| A0 violations | 0 over every pair × probe, incl. same-centre, on-nucleus, 1e-4/1e-8 off-nucleus, 50–200 Bohr | non-zero, and concentrated at either D→0 (inside term wrong) or D→large (tail term wrong) — **the location of the violations distinguishes which half is broken** |
| trivial limit | bit-equal to IPB-0 at D=0 | differs (a `min` written as a sum, or `R_ab` clamping wrong) |
| tightness vs IPB-0 | ratio improves monotonically with D, → `S_0/D` asymptotically | ratio flat in D (distance factor inert) OR ratio > 1 (invalid) |
| tightness vs ferric | better at large D, **worse or equal at small D** — ferric's Hölder bound is a genuine bound too and its `sup` over ρ is not obviously looser at contact | IPB-D beats ferric EVERYWHERE by a constant factor ⇒ suspect a shared normalization error, not a win |

These are deliberately distinguishable. In particular the third row is the
non-inertness test and the fourth row is a "too clean is a stop condition"
tripwire.

## 3.1 Quantitative pre-registration (the numbers I will be held to)

* **Expected tightness gain.** Median `true/bound` ratio: I expect IPB-D to beat
  IPB-0 by **1–2 orders of magnitude at D ≳ 10 Bohr** (the `1/D` factor is the
  whole content) and by **< 5% at D < 2 Bohr**. Against ferric's bound I expect
  a **factor 1.5–5 improvement at moderate D (5–20 Bohr)**, because ferric loses
  `|AB|/2` off the distance and takes a `sup` over ρ where the IPB integrates.
* **Expected kept-work reduction.** At matched K error, **3–12 percentage points
  less kept (pair, batch) work than ferric's bound at C8–C16**, growing with
  size (the `|AB|/2` share grows with size per §6.2b's table). Against IPB-0 I
  expect **more**, because IPB-0 has no distance decay at all and must therefore
  keep every distant pair that IPB-D drops — I expect that gap to be LARGE
  (> 20 pp at C16), and if it is not, something is wrong with the harness's
  distance range rather than with the bound.
* **Degeneracy (the disease to check for inheritance).** Fraction of (pair,
  batch) decisions where `IPB-D == IPB-0` exactly, i.e. the `min` selects the
  distance-free branch. ferric's sphere bound degenerates on **45–80%**
  (§6.2b). I expect IPB-D's degenerate fraction to be **substantially lower**,
  because it never subtracts `|AB|/2`; **if it lands in the same 45–80% band,
  the lane closes** — the new bound has inherited the same disease from a
  different mechanism (the batch radius alone eating the distance).

---

## 4. What result would make me declare the bound INVALID or INERT

Stated now so the verdict cannot be chosen after seeing the data.

**INVALID (kill immediately, do not tune):**
* Any A0 violation `true > bound × (1 + 1e-12)` on any of the enumerated probe
  classes, on any of the three system/basis combinations. A bound that
  underestimates is the exact bug class that cost 1.06 Ha of K error in
  whitepaper §4 layer 1. There is no threshold at which "a few violations" is
  acceptable.
* Failure of the trivial limit at D = 0.

**INERT (report as a negative and close the lane):**
* Degenerate fraction (`IPB-D == IPB-0`) in the 45–80% band that afflicts
  ferric's sphere bound, at C8 or larger.
* Median tightness gain over IPB-0 below **1.2×** across the measured D range.
* Kept-work reduction vs BOTH baselines **< 1 pp at matched error** at every
  measured size — i.e. inside the ±0.5–0.83 pp noise band in which §6.2a already
  found the two existing bounds to be indistinguishable. A third bound landing
  in that same band is a third measurement of "the screen is not bound-limited",
  which is itself the useful finding.

**A well-argued negative is a complete deliverable.** The last two lanes
(sub-batching, sn-LinK structure) closed that way and the mechanism was the
transferable part.

---

## 5. Measurement plan

* **Systems / bases.** water/cc-pVDZ, butane/def2-SVP, butane/def2-QZVP (g
  functions — §6.2a lever 2 notes only STO-3G was ever measured, and the IPB's
  advantage is radial/angular so QZ is where it should show). Size axis
  C1…C16/def2-SVP for kept-work counts if cheap; STO-3G is permitted on the size
  axis for cost, stated explicitly wherever used.
* **Anchor A0 probe set.** ≥ 200 probes per system covering: every nucleus
  exactly (T = 0), 1e-4 and 1e-8 Bohr off a nucleus along a random direction,
  a random cloud through the molecular volume, and shells at 50, 100 and 200
  Bohr. Every shell pair, both orderings. Truth = PySCF `int1e_grids`.
* **Same-centre pairs reported separately.** That is where ferric's original
  signed-overlap bound failed catastrophically (whitepaper §4 layer 1).
* **Kept work = deterministic COUNTS** of (shell pair, grid batch) decisions,
  weighted by `ncart(s1) × ncart(s2)` where a cost weight is meaningful. No
  timings — load-immune, per repo convention.
* **Matched error, not matched threshold.** §6.2a's recorded trap: a threshold
  cuts on `bound × weight`, so equal thresholds are not equal operating points.
  Each bound gets its threshold swept and kept-work is read off at equal
  measured K error.
* **Mutation testing.** Every anchor gets a deliberately broken variant proving
  it CAN fail: (i) `V_D` without the `S_0/D` inside term — the previous
  attempt's error, which must reproduce violations; (ii) `min` replaced by the
  distance branch alone; (iii) `R_ab` unclamped (negative radius); (iv) the
  degeneracy counter fed a constant bound. Mutations run in a scratch copy or
  behind a `--mutate` flag, never left in the source (a `git commit -a` captured
  a mutant earlier today).

## 6. Scope limits accepted in advance

* Python/numpy + PySCF only, single-threaded, no cargo, no ferric binaries.
* Counts, not wall time. §6.2a lever 3 stands unaddressed: the IPB is cheaper to
  evaluate per decision than a sphere query, and this study cannot see that.
* Real spherical-harmonic (pure) shells are handled by the same max-column-sum
  factor the existing prototype uses; that factor is validated only through A0.
