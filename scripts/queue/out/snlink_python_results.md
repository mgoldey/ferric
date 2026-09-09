# Results: Python reference for sn-LinK-style screening of seminumerical K

Date: 2026-09-08/09. Branch `spec/snlink-python`. Pre-registration:
`snlink_python_prereg.md` (committed first, `e4b417d6`). Implementation:
`scripts/snlink_proto.py` (`c02da260`, `9d170c94`, `76277951`). All numbers one
thread, numpy/PySCF 2.13.0, **counts not timings** (deterministic and
load-immune). Systems reach alkane_16 (50 atoms, 38.7 Bohr, 3.66M pair-batches);
the count-only modes need no K accumulation and so run far past the anchor table.

---

## 0. VERDICT

**No. sn-LinK's screening structure does not keep less work than ferric's
current density-driven batch screen.** Across C1–C16 (3.9 → 38.7 Bohr) and
thresholds 1e-5 → 1e-8, the two agree to within **±0.83 percentage points** of
kept pair-batches, and sn-LinK's is *consistently the slightly worse one* at
every system from C6 up. The gap does not close with size; it drifts marginally
in ferric's favour.

**Recommendation: do NOT implement sn-LinK's screening structure in Rust.**
Not because it is bad, but because ferric's existing screen already captures
what it captures: anchor A4 shows sn-LinK reduces to ferric's screen exactly
when its density weight is collapsed, and the two differ by <1 pp everywhere
once it is not.

**The valuable finding is elsewhere, and it is a partial correction to the
whitepaper's §4.1.** At **pair-batch** granularity the density matrix is *not*
vacuous: it does 60–68% of the pruning, and its share **grows** with molecular
diameter (§4). That is the opposite of what LinK's density-pair list and COSX's
row mask do at the same sizes — and the reason is granularity, not chemistry.
§4 sets out the measurement and the reconciliation.

---

## 1. What was built

`scripts/snlink_proto.py` — both screens on the SAME seminumerical core and the
SAME integral bound, so the comparison isolates the screening structure:

```
X[mu,g] = sqrt(w_g) chi_mu(r_g);  F = D X;  G_g = A^g F_g;  K = ½(XG^T + (XG^T)^T)

ferric (cosx_k.rs:1096-1105):
    keep(s1,s2,b) iff bound(s1,s2,sphere_b) * max(fmax[s1], fmax[s2]) >= t
sn-LinK-style (reconstructed, see prereg §1.1):
    keep(s1,s2,b) iff bound(s1,s2,sphere_b) * max(xmax[s1], xmax[s2]) >= eps_E   (E)
                   OR bound(s1,s2,sphere_b) * max_l(dmax[l,s] * xmax[l,b]) >= eps_K   (K)
```

`bound` is ferric's Hölder primitive-pair bound, ported term-by-term from
`crates/ferric-integrals/src/cosx_screen.rs` and validated against true
`int1e_grids` blocks (anchor A0).

**Sourcing, restated because it bounds every claim here.** Every primary source
(Laqua/Kussmann/Ochsenfeld *JCTC* **14**, 3451 (2018) and *JCTC* **16**, 1456
(2020); Thompson/Ochsenfeld *JCP* **150**, 044101 (2019); Kussmann/Ochsenfeld
*JCP* **138**, 134114 (2013)) is paywalled and could not be read; the session's
web-search budget was exhausted. Verified from OA metadata/abstracts and the OA
review (*Pure Appl. Chem.* 2025, DOI 10.1515/pac-2025-0603, PMC12645584): that
sn-LinK is "a seminumerical counterpart to the LinK method"; that the 2018
method "combin[es] the preLinK method ... with explicit screening of integrals
for batches of grid points"; that preLinK is "a preselection method based on
Schwarz integral estimates"; that the 2020 paper adds Thompson & Ochsenfeld's
integral bounds. **Reconstructed, NOT to be cited as Laqua's:** the
two-threshold split and the exact density-weight form. **Deliberately
substituted:** the integral partition bound, for ferric's Hölder bound (holding
the bound fixed is what makes the structural comparison fair). So this measures
*a* sn-LinK-shaped screen, not sn-LinK.

---

## 2. Anchors (all pre-registered; every one mutation-tested)

| anchor | water | methane | ethane | **alkane_4 (STO-3G)** |
|---|---|---|---|---|
| A0 bound never underestimates | PASS 0/5478 | PASS 0/13005 | PASS 0/35728 | PASS 0/23782 |
| A1a unscreened K == analytic K | PASS 6.32e-05 | PASS 3.77e-04 | PASS 1.49e-04 | PASS 3.80e-04 |
| A1b trivial limit (thresholds → 0) | PASS **0.00e+00** | PASS **0.00e+00** | PASS **0.00e+00** | PASS **0.00e+00** |
| A2 screen error ≤ 0.1 × grid error | VACUOUS | VACUOUS | PASS 5.7e-05 | **PASS 1.65e-03** |
| A3 reachability | **FAIL** | **FAIL** | PASS 99.50% | **PASS 93.30%** |
| A4 positive control (SN → FE exactly) | PASS symdiff 0 | PASS symdiff 0 | PASS symdiff 0 | **PASS symdiff 0** |

A1b is exactly `0.00e+00`, not merely under the 1e-14 bar — the trivial limit is
bitwise.

A2 is reported **VACUOUS** rather than PASS on water/methane: with nothing
dropped, `E_scr = 0` by construction and the anchor measures nothing. A green
row that cannot fail is worse than one that has never failed, so it is labelled.
A3 fails there exactly as pre-registered (prereg §2 A3 predicted this *before*
any measurement). **alkane_4 is the row that matters**: every anchor passes
non-vacuously, at matched K error (ferric 6.24e-07, sn-LinK 6.27e-07, both ~600×
below the 3.80e-04 grid error) with 93.30% vs 93.26% kept.

### 2.1 Mutation proofs

Each anchor was broken by the defect it guards (`--mutate NAME`):

| mutation | what it breaks | caught by |
|---|---|---|
| `bound_underestimate` (bound ÷ 1e6) | the screen's one required property | **A0** (5466/5478 violations), A2 (ratio 2.5e4), A4 |
| `drop_mirror` (skip the s1↔s2 accumulation) | the K itself | **A1a** (E_grid 1.35 vs 9.8e-3 bar) |
| `counter_only` (skip every 4th batch, counter untouched) | K wrong, counts unchanged | **A1a** (E_grid 2.81) |
| `strict_gt` (`>` instead of `>=`) | the trivial limit's edge | **A2** (E_scr = 0.33 × E_grid) |

**Two anchors were rewritten because mutation testing showed they proved
nothing** — the main methodological return from this exercise:

1. **A1 as pre-registered (screened vs unscreened) missed `drop_mirror` and
   `counter_only` entirely** — an accumulation defect breaks both sides equally,
   so the difference stayed `0.00e+00` and A1 passed on a K wrong by 1.35.
   Split into A1a (unscreened vs analytic, catching the accumulation) and A1b
   (the trivial limit proper).
2. **A4's control used `eps_E = 0`, at which the E-branch fires
   unconditionally**, so it never reached the K-branch it was meant to isolate.
   It passed vacuously on water and was caught only by FAILING on ethane — the
   first system where the kept sets differ at all. Fixed to `eps_E = inf`.

---

## 3. The kept-work comparison (the deliverable table)

Kept (shell-pair, grid-batch) units. STO-3G, ferric's (50,110) grid unpruned,
converged RHF density, 256-point batches. **Positive `SN−FE` means sn-LinK keeps
MORE, i.e. is worse.**

| system | diam (Bohr) | nbf | pair-batches | thresh | FE kept % | SN kept % | **SN−FE (pp)** |
|---|---|---|---|---|---|---|---|
| alkane_1 | 3.9 | 9 | 3 024 | 1e-5 | 99.835 | 99.868 | +0.033 |
| | | | | 1e-7 | 100.000 | 100.000 | 0.000 |
| alkane_2 | 5.8 | 16 | 13 416 | 1e-5 | 96.228 | 96.117 | −0.112 |
| | | | | 1e-7 | 98.718 | 98.718 | 0.000 |
| alkane_4 | 10.5 | 30 | 76 153 | 1e-5 | 75.577 | 75.178 | −0.399 |
| | | | | 1e-6 | 87.219 | 87.586 | +0.366 |
| | | | | **1e-7** | **93.302** | **93.265** | **−0.037** |
| | | | | 1e-8 | 96.435 | 96.511 | +0.076 |
| alkane_6 | 15.2 | 44 | 227 040 | 1e-5 | 51.846 | 52.349 | +0.503 |
| | | | | **1e-7** | **76.949** | **77.014** | **+0.065** |
| alkane_8 | 19.9 | 58 | 504 777 | 1e-5 | 36.350 | 37.170 | +0.820 |
| | | | | **1e-7** | **60.734** | **61.218** | **+0.484** |
| alkane_10 | 24.6 | 72 | 948 064 | 1e-5 | 25.600 | 26.176 | +0.576 |
| | | | | **1e-7** | **47.121** | **47.783** | **+0.662** |
| alkane_12 | 29.3 | 86 | 1 595 601 | 1e-6 | 28.311 | 29.062 | +0.751 |
| | | | | **1e-7** | **37.581** | **38.357** | **+0.776** |
| alkane_16 | 38.7 | 114 | 3 658 225 | 1e-6 | 18.036 | 18.614 | +0.578 |
| | | | | **1e-7** | **24.843** | **25.566** | **+0.723** |

**The screen works** — kept falls 100% → 24.8% at 1e-7 across C1→C16 — **and the
two variants are indistinguishable.** The largest difference anywhere is
**+0.830 pp** (alkane_8, 1e-6) and it is in *ferric's* favour. From C6 upward the
sign is stably positive: sn-LinK keeps marginally more work at every size.

At matched accuracy on alkane_4 at 1e-7, the full three-way comparison
(with K actually accumulated, so the errors are measured not inferred):

| configuration | kept | kept % | weighted % | max\|dK\| | vs E_grid |
|---|---|---|---|---|---|
| unscreened | 76153/76153 | 100.00 | 100.00 | 0 | 0 |
| ferric, t=1e-7 | 71052/76153 | 93.30 | 96.05 | 6.24e-07 | 1.65e-03 |
| sn-LinK, eps=1e-7 | 71024/76153 | 93.26 | 95.99 | 6.27e-07 | 1.65e-03 |
| *sn-LinK E-branch only* | 61558/76153 | 80.83 | 86.32 | 8.15e-03 | **21.5** |
| *sn-LinK K-branch only* | 70978/76153 | 93.20 | 95.96 | 6.29e-07 | 1.66e-03 |

The two production rows differ by 28 pair-batches in 76 153 at K errors that
agree to 0.5%. The last two rows show why the OR structure is load-bearing: the
E-branch alone is 12.5 pp cheaper but its K error is **21× the grid error** —
inadmissible. The K-branch alone is admissible and marginally cheaper than the
union, but the union is what a robust screen must use (the E-branch exists to
catch pairs a transient density makes look small).

---

## 4. The density IS doing work at pair-batch granularity — a §4.1 refinement

This is the part worth carrying forward, and it corrects a reading I made and
then disproved within this study.

**The wrong first reading.** The size sweep appears to show the density branch
pruning hard: at alkane_10/1e-7 the geometric bound alone keeps 68.24% while the
density-weighted branch keeps 47.72%, dropping 194 577 pair-batches. Read
naively that says "the density does 20 pp of work".

**Why that reading is wrong.** *Both* screens' "density" factors secretly contain
the AO values on the grid:

```
ferric:  fmax[s]  = max |(D X)[s]|            <- contains X
sn-LinK: dweight  = max_l dmax[l,s] * xmax[l] <- contains X
```

A screen can therefore look density-driven while actually riding on the Gaussian
decay of `X` away from the batch, which is **pure geometry**. This is exactly the
confound the whitepaper's §4.1 warns about, in a new place.

**The decomposition (`--decompose`).** Replace `D` by a *constant* matrix of the
same magnitude, leaving `X` untouched. Then `geom → flatD` is what the AO
magnitude contributes through the same expression, and `flatD → +realD` is the
density's true contribution with the AO factor held fixed. STO-3G, t = 1e-7:

| system | diam | geom | +AO(X) | flatD | +realD | X does | **D does** | **D share** |
|---|---|---|---|---|---|---|---|---|
| alkane_1 | 3.9 | 100.00% | 98.61% | 100.00% | 100.00% | +0.00 | +0.00 | 0.0% |
| alkane_2 | 5.8 | 99.79% | 95.64% | 98.72% | 98.72% | −1.07 | +0.00 | 0.0% |
| alkane_4 | 10.5 | 98.39% | 80.83% | 96.37% | 93.20% | −2.01 | **−3.17** | **61.1%** |
| alkane_6 | 15.2 | 90.44% | 56.41% | 85.05% | 76.89% | −5.38 | **−8.16** | **60.2%** |
| alkane_8 | 19.9 | 78.78% | 38.91% | 71.60% | 61.13% | −7.18 | **−10.47** | **59.3%** |
| alkane_10 | 24.6 | 68.24% | 27.37% | 60.84% | 47.72% | −7.40 | **−13.12** | **63.9%** |
| alkane_12 | 29.3 | 59.67% | 20.51% | 52.72% | 38.30% | −6.95 | **−14.42** | **67.5%** |

So the density matrix genuinely accounts for **60–68% of the pruning** at
pair-batch granularity, its absolute contribution grows monotonically
(−3.2 → −14.4 pp), and its *share* grows too (61.1% → 67.5%). It is vacuous only
below ~10 Bohr, with a clean onset at alkane_4.

**Reconciliation with §4.1 — this is a granularity result, not a contradiction.**
The whitepaper's three sightings (COSX's row mask keeping 100% at C8; LinK's
density-pair list pruning 0.00% even at ~62 Bohr; the #52 threshold sweep) all
test the density at a coarser granularity: `max|D|` over a *shell row*, or a
*pair* with both maxima realized by core s-shell pairs so the product degenerates
to `Qmax²`. Here the density is tested as `max_l (dmax[l,s] · xmax[l,b])` — a
**(density, batch) combination**, resolved per grid batch, which is precisely the
"(bra pair, ket pair) combination" §4.1 identifies as where real locality lives.
The whitepaper's conclusion stands as written for the masks it measured; what
this adds is that the same density becomes non-vacuous once it is contracted
against a spatially local batch rather than maximized over a whole index.

**This is therefore NOT a fourth sighting of the vacuous-density effect.** It is
a measured boundary on it. The honest statement: density-derived masks are
vacuous *at row and pair granularity* through 62 Bohr (three independent
sightings, whitepaper §4.1); at *(pair, grid-batch)* granularity the density does
the majority of the pruning from 10 Bohr up (this study). Both can be true because
they are different contractions of the same matrix.

**The caveat that keeps this honest:** the density doing 60% of the work is
measured on the *screening product*, not on wall time, and it does not translate
into sn-LinK beating ferric — because ferric's `fmax` already contracts the
density against the batch (`F = D X` *is* that contraction). Both screens
exploit the same effect; §3 is what happens when you ask which exploits it better.

### 4.1 Supporting diagnostics (`--diagnostics`, cc-pVDZ)

| system | batch radius (median / max, Bohr) | R_c ≤ 0 fraction | min bound | min FE product |
|---|---|---|---|---|
| water | 1.85 / 16.47 | 79.9% | 1.39e-01 | 4.30e-05 (430× above 1e-7) |
| methane | 2.07 / 17.36 | 72.1% | 5.08e-02 | 3.22e-06 (32× above) |
| ethane | 1.82 / 21.10 | 68.2% | 2.64e-04 | 6.47e-09 (below 1e-7) |

For 68–80% of pair-batches the sphere query returns its distance-free `R = 0`
value, because a 256-point batch's bounding sphere swallows a small molecule
whole. That is why nothing is dropped at water/methane scale for either screen,
and it is the mechanism behind §6's recommendation. (Spatially sorting the grid
makes this *worse*: median radius 1.85 → 4.93 Bohr, degenerate fraction 79.9% →
100%. Measured, not assumed.)

The SN-K/FE product ratio has spread 15–25× (water 0.261/0.761/3.900 as
min/median/max; methane 0.146/0.534/3.671; ethane 0.207/0.710/4.082), with
sn-LinK's weight the tighter one on 75–90% of pair-batches. A spread of ~1.0
would have meant the comparison was a threshold rescaling in disguise — the
artifact channel pre-registered in prereg §3. It is not: the two genuinely order
pair-batches differently and sn-LinK's is usually the smaller number. It simply
does not change enough keep decisions to matter (§3).

**Do not read a size trend into the 75–90%.** Ordered by diameter (2.86, 3.35,
5.83 Bohr) it goes 74.7 → 90.3 → 80.6% — non-monotone, with the middle system the
extremum. Three non-monotone points are not a trend.

---

## 5. Honest limits

1. **The integral bound is ferric's, not Laqua's.** Holding it fixed is what
   makes the structural comparison fair, but it means this study says nothing
   about the lever the 2020 paper explicitly credits — Thompson & Ochsenfeld's
   integral partition bounds. Given §4.1's degeneracy finding, that is plausibly
   where a real win lives. **Not closed by this result.**
2. **STO-3G for the size axis, cc-pVDZ only through ethane.** The size sweep
   uses STO-3G to reach C16; the whitepaper's own L-axis result says the
   seminumerical method's case is at *high* angular momentum, which STO-3G
   cannot probe. Whether the SN/FE gap behaves differently at def2-QZVP is
   unmeasured. The gap is <1 pp at every point measured, so a reversal would
   have to be large and basis-driven.
3. **C16 (38.7 Bohr) is short of the ~62 Bohr onset** the whitepaper names for
   LinK-style pair lists. §4's density effect has a clean onset at 10 Bohr and is
   monotone through 29 Bohr, so it is established well within range; the SN-vs-FE
   *gap* is flat-to-slightly-negative over that whole range.
4. **The reconstruction may not be Laqua's screen.** If the real scheme differs
   structurally, this measures the wrong thing. Prereg §1.1 says what would
   settle it: read *JCTC* **14**, 3451 §2 and re-derive before writing Rust.
5. **Counts, not timings.** The weighted column
   (`ncart(s1)·ncart(s2)·|batch|`) is a cost proxy, not a measurement. No Rust
   was built or run.

---

## 6. Recommendation

**Do not implement sn-LinK's screening structure in Rust as a performance
change.** It is within ±0.83 pp of `cosx_k.rs` at every system and threshold
measured, and from C6 up it is consistently the marginally worse of the two.
Anchor A4 shows why: collapse its density weight and it becomes ferric's screen
exactly. The two are the same idea, and ferric already has it.

**Do pursue the near-field bound instead — §4.1 is the actionable finding.**
For 68–80% of pair-batch decisions ferric's bound degenerates to its
distance-free `R = 0` value, so the screen is running blind on most of its
decisions. Two things follow, neither needing the paywalled literature:

- **Sub-batch the sphere test** (cheapest first). The bound is evaluated once per
  256-point batch over a sphere enclosing all of it. Splitting the *test* (not
  the kernel) over tighter sub-spheres restores `R_c > 0` for many pairs at O(1)
  per pair per sub-sphere. Headroom is large: the bound's floor is 1.4e-1 while
  the products sit at 1e-5–1e-9.
- **Then, if someone obtains *JCP* 150, 044101 (2019),** swap `Bounds` for the
  integral partition bound and re-run this harness unchanged. `snlink_proto.py`
  isolates the bound in one injectable class specifically so that experiment is a
  one-class change with every anchor still in place.

**Worth recording in the whitepaper regardless of the above:** §4's granularity
boundary on the density-vacuity result. It refines §4.1 rather than contradicting
it, and it predicts that COSX's Λ row mask — noted vacuous in §4 layer 3 and
flagged in §6.3 as needing "the combination form" — should become non-vacuous
when restricted to the σ a given batch actually couples. §4 is direct evidence
for that §6.3 conjecture, obtained independently.

**Reproducing:**

```bash
# anchors, at a size where every one is non-vacuous (~2 min)
OMP_NUM_THREADS=1 python3 scripts/snlink_proto.py --systems alkane_4 --basis sto-3g --no-sweep

# kept-work vs diameter, counts only (C1-C10 ~35 s; adding C12,C16 ~2 min)
OMP_NUM_THREADS=1 python3 scripts/snlink_proto.py \
    --systems alkane_1,alkane_2,alkane_4,alkane_6,alkane_8,alkane_10 \
    --basis sto-3g --size-sweep

# density vs geometry decomposition (~1.5 min through C12)
OMP_NUM_THREADS=1 python3 scripts/snlink_proto.py \
    --systems alkane_1,alkane_2,alkane_4,alkane_6,alkane_8,alkane_10,alkane_12 \
    --basis sto-3g --decompose

# why nothing is dropped at water scale (~3 s)
OMP_NUM_THREADS=1 python3 scripts/snlink_proto.py --systems water,methane,ethane --diagnostics

# mutation proofs
OMP_NUM_THREADS=1 python3 scripts/snlink_proto.py --systems water --no-sweep --mutate drop_mirror
```
