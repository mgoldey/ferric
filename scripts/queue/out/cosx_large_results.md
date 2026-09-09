# Results: COSX at large system size — the ceiling map

Pre-registration: `scripts/queue/out/cosx_large_prereg.md`, committed 8f7a0bf0
on 2026-09-09 **before any number below**. Branch `meas/cosx-large-system`,
worktree `.claude/worktrees/cosx`, off `origin/main` 433db737.

**This file covers TWO sittings, both on 2026-09-09.** The first measured the two
def2-SVP rungs and concluded the lane was blocked by SCF cost. The second
unblocked it by converging the missing densities on all cores (a density is an
input, not a timing) and measured the two rungs the first could not reach —
alkane_20/def2-TZVP and alkane_48/def2-SVP — plus the direct exact-K reference.
Where the second sitting contradicts the first, the first sitting's text is kept
and marked rather than deleted, so the record shows what was believed when.

## Protocol as run

One thread throughout for **every timed K build**
(`OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1`, plus an
explicit 1-thread rayon pool around every timed segment), `FERRIC_MEM_BUDGET_GB=2`,
`scripts/ferric-limited --max=6G --high=5G` in the foreground under a bounded
`timeout`. `/proc/pressure/memory` `full avg10` printed before and after every
timed segment; every kept row is 0.00/0.00. cpu_s == wall_s to <= 0.05% on every
kept row. Peak RSS is `VmHWM` for the whole process (SCF included where the cell
ran one). COSX is production configuration at every rung and no knob is tuned
per rung: md3c1e backend, sparse half transforms, density-driven screen 1e-7,
grid (50,110), overlap fit ON.

**The one deliberate exception, and the reason the lane finished:** DENSITY
GENERATION is not a timed quantity, so it is not bound by the one-thread rule —
that rule exists to make timings comparable to each other. Densities converged in
the second sitting therefore used all 12 cores (`RAYON_NUM_THREADS=12`, OpenBLAS
still pinned to 1). Every such wall time is labelled as not-a-timing wherever it
appears, and no ratio in this file is computed from one. Every K build in every
table is single-threaded, without exception. One row (alkane_20/def2-TZVP
direct) is parenthesised because it printed a nonzero PSI reading; it is
excluded from all ratios and the reason is given at that table.

## Sizing (measured before the pre-registration was written; no build, no SCF)

`def2-universal-jkfit`'s `naux` is a function of the ATOMS only — 2256 / 3588 /
5364 for C20 / C32 / C48 — so DF-K's dressed 3-index tensor `naux * nbf^2 * 8`
grows as `nbf^2` at fixed system, not `nbf^3`.

| system | natoms | basis | nsh | nbf | L_max | naux | DF-K dressed tensor |
|---|---|---|---|---|---|---|---|
| alkane_20 | 62  | def2-SVP  | 246  | 490  | 2 | 2256 | 4.33 GB |
| alkane_20 | 62  | def2-TZVP | 388  | 872  | 3 | 2256 | 13.72 GB |
| alkane_20 | 62  | def2-QZVP | 760  | 2400 | 4 | 2256 | 103.96 GB |
| alkane_32 | 98  | def2-SVP  | 390  | 778  | 2 | 3588 | 17.37 GB |
| alkane_32 | 98  | def2-TZVP | 616  | 1388 | 3 | 3588 | 55.30 GB |
| alkane_32 | 98  | def2-QZVP | 1204 | 3804 | 4 | 3588 | 415.36 GB |
| alkane_48 | 146 | def2-SVP  | 582  | 1162 | 2 | 5364 | 57.94 GB |
| alkane_48 | 146 | def2-TZVP | 920  | 2076 | 3 | 5364 | 184.94 GB |
| alkane_48 | 146 | def2-QZVP | 1796 | 5676 | 4 | 5364 | 1382.50 GB |

The task brief's estimates are superseded: it guessed nbf 7510 for
alkane_48/QZVP (real: 5676) and 7890 GB for its tensor (real: 1382 GB), because
it assumed `naux ~ 2.5 * nbf` at every basis.

## Finding 0, taken before any large rung: the COSX/LinK ratios on record were
## measured against a LinK that was subsequently found to be WRONG

Rung 1 of this lane is alkane_20 / def2-SVP — deliberately the largest cell
already on record, run as an instrument check. It does NOT reproduce
`cosx_scaling_results.md`, and the reason is in the git history, not in the
measurement:

| quantity | `cosx_scaling_results.md` (2026-09-07 22:39) | this lane (2026-09-09) |
|---|---|---|
| COSX total wall | 168.494 s | 116.234 s |
| LinK warm wall  | 21.384 s   | 73.219 s |
| COSX / LinK     | 7.88       | **1.587** |

Both numbers moved, for two separately identifiable commits landed AFTER that
results file:

* `a272333c fix(link_k): ket loop covers all four density blocks; density-pair
  criterion is a valid bound` (2026-09-08 00:05) — a CORRECTNESS fix. The old
  LinK was skipping ket-side work it should have done, so it was fast for the
  wrong reason. LinK 21.4 s -> 73.2 s.
* `64264061 feat(cosx): shell-sparse block half transforms` (2026-09-08 18:07) —
  a genuine COSX speedup. COSX 168.5 s -> 116.2 s.

**Consequence for the whitepaper: every COSX/LinK ratio in
`cosx_scaling_results.md` (7.57 / 5.56 / 5.84 / 6.88 / 7.88 across C4..C20) is
against a buggy LinK and must not be quoted.** The measured ratio on a correct
LinK at the same cell is 1.587, i.e. COSX is ~1.6x slower than LinK at
alkane_20/def2-SVP — not ~8x slower. This is a large correction in COSX's
FAVOUR, and it was found by re-running an old cell rather than by any new
physics. It also means the "COSX/LinK worsens with N" trend that the
pre-registration predicted (P3: 8-11x at C32) is measured on a different
baseline and its earlier points are void.

The commit message of `a272333c` independently corroborates the mechanism
rather than relying on my re-run: it records that the old kernel lost 8% of K
at the SAD guess (`max|K_link - K_direct|` 0.55 -> 2.3e-7) and drove the
butane/def2-SVP SCF non-variational (-162.763, never converged, vs the correct
-157.1861392). The old LinK was cheap because it was skipping real work.

This also re-reads a number already in the record. `cosx_scaling_results.md`
reported `max|K_cosx - K_link|` of 2.8e-3..3.3e-3 at every size and attributed
it to LinK rather than COSX. On the fixed LinK the same cell gives **4.825e-4**
(vs `max|K| = 7.157`), i.e. ~6x smaller — consistent with that attribution
having been correct, and now measurable against a reference worth measuring
against.

### The other half of the same re-run: the block GEMMs stopped being the problem

The old lane's most load-bearing worry about COSX at size was that the dense
block GEMMs had tail exponent **4.55** and were taking over the total (54.6 s of
168.5 s, 32%, at alkane_20/def2-SVP). On the same cell with the same density
today:

| split (alkane_20 / def2-SVP) | 2026-09-07 | this lane |
|---|---|---|
| total | 168.494 s | 116.234 s |
| A-build | 99.557 s (59.1%) | 99.027 s (85.2%) |
| block GEMMs | 54.622 s (32.4%) | **3.029 s (2.6%)** |
| ao_eval | 8.31 s | 8.147 s |
| per-point GEMV | — | 5.208 s |

The A-build is unchanged to 0.5% (99.56 -> 99.03 s), which is the internal
control that this is the same measurement of the same thing; the entire
difference is the GEMM column, from `64264061`'s shell-sparse half transforms
(|A|/nbf = 0.2856 at this cell, so the block products run on ~29% of the row
space). COSX at this size is now **85% A-build**, i.e. back to being dominated by
the quantity that has the favourable N^1.54 tail, with the N^4.55 term reduced
to 2.6% of the total. Pre-registration P3 predicted 400-700 s at
alkane_32/def2-SVP from the OLD total-wall exponent of 2.16; that exponent was
inflated by the GEMM term and no longer describes this code.

## The rung table (measured)

Every row: one thread, one process, one shared converged density per rung, PSI
`full avg10` 0.00 before and after, cpu == wall. "COSX need" is the closed-form
`check_budget` block requirement; "peak RSS" is the measured `VmHWM` for the
whole cell (which also ran the other builders, so it over-states COSX alone).

| system | basis | nbf | builder | wall s | cpu s | peak RSS | \|A\|/nbf | kept_dd | outcome |
|---|---|---|---|---|---|---|---|---|---|
| alkane_20 | def2-SVP | 490 | COSX  | 116.234 | 116.23 | 645 MB (cell) | 0.2856 | 0.1458 | ran |
| alkane_20 | def2-SVP | 490 | LinK (warm) | 73.219 | 73.21 | " | — | — | ran |
| alkane_20 | def2-SVP | 490 | DF-K  | — | — | — | — | — | **refused: 4.33 GB tensor** |
| alkane_32 | def2-SVP | 778 | COSX  | 209.713 | 209.67 | 724 MB (cell) | 0.1877 | 0.0657 | ran |
| alkane_32 | def2-SVP | 778 | LinK (warm) | 192.725 | 192.69 | " | — | — | ran |
| alkane_32 | def2-SVP | 778 | DF-K  | — | — | — | — | — | **refused: 17.37 GB tensor** |
| alkane_48 | def2-SVP | 1162 | COSX  | 354.835 | 354.78 | 786 MB (cell) | 0.1285 | 0.0315 | ran |
| alkane_48 | def2-SVP | 1162 | LinK (warm) | 291.398 | 291.35 | " | — | — | ran |
| alkane_48 | def2-SVP | 1162 | direct (exact J+K) | 605.744 | 605.62 | 681 MB (cell) | — | — | ran |
| alkane_48 | def2-SVP | 1162 | DF-K  | — | — | — | — | — | **refused: 57.94 GB tensor** |
| alkane_20 | def2-TZVP | 872 | COSX  | 301.913 | 301.87 | 709 MB (cell) | 0.2850 | 0.1443 | ran |
| alkane_20 | def2-TZVP | 872 | LinK (warm) | 448.822 | 448.74 | " | — | — | ran |
| alkane_20 | def2-TZVP | 872 | direct (exact J+K) | (672.028) | (671.90) | 628 MB (cell) | — | — | **FLAGGED, see below** |
| alkane_20 | def2-TZVP | 872 | DF-K  | — | — | — | — | — | **refused: 13.72 GB tensor** |

The last two rungs are new in this pass (2026-09-09, second sitting) and are the
ones the first pass could not reach. What unblocked them is in
"How the two missing densities were finally produced" below; nothing about the
TIMING protocol changed — every row above is still one thread, cpu == wall, PSI
`full avg10` 0.00 before and after.

COSX splits, per rung: alkane_20/SVP total 116.234 = ao_eval 8.147 + A-build
99.027 (85.2%) + GEMV 5.208 + block GEMMs 3.029 + fit 0.024; alkane_32/SVP total
209.713 = ao_eval 20.025 + A-build 171.406 (81.7%) + GEMV 9.178 + block GEMMs
6.923 + fit 0.154; alkane_48/SVP total 354.835 = ao_eval 44.511 + A-build 277.212
(78.1%) + GEMV 13.797 + block GEMMs 13.875 + fit 0.679; alkane_20/TZVP total
301.913 = ao_eval 12.498 + A-build 256.934 (85.1%) + GEMV 17.951 + block GEMMs
10.879 + fit 0.123. All accounted to 98.7%+.

Note the `direct` rows: `build_jk` computes J AND K in one quartet sweep and only
K is used, so the direct column over-states a K-only build by whatever fraction
of that sweep J costs. It is reported unadjusted because that is what was
measured; it is an upper bound on exact-K alone, and it is used here only as the
ACCURACY reference and as a rough ceiling, never as the headline denominator
(LinK is).

**The one flagged row, recorded rather than quietly dropped or quietly kept.**
The `alkane_20/def2-TZVP` direct build printed `PSI full avg10 before=0.01`,
which this lane's own protocol says is grounds to DISCARD and retake. Its wall
time is therefore parenthesised above and **is not used in any ratio anywhere in
this file**. Two honest observations about it, in both directions:

* Its internal consistency is intact — cpu 671.90 against wall 672.028 is
  99.98%, i.e. the process was not actually starved of CPU during the segment,
  and 0.01 is a decaying tail from the SCF that had just finished, not
  contention during the build.
* The retake nevertheless could not be completed. It was launched on a verified
  clean box (`full avg10=0.00`, `avg60=0.00`, load 0.36) and a **different
  agent's workload started on the same machine mid-run** — load average went to
  18.4, CPU pressure `some avg10` to 50.6, memory PSI to 0.04, with several
  `ferric-cli` processes and two multi-core python drivers resident. That retake
  was killed rather than allowed to produce a contaminated number, and rather
  than allowed to compete with someone else's job for the box.

The ACCURACY numbers from that same cell are unaffected by timing contention and
are used: `max|K_cosx - K_direct| = 2.5886e-4` (relative 3.307e-5),
`||dK||_F/||K||_F = 8.746e-5`. A deviation between two matrices does not depend
on how fast either was computed. Only the wall time is withheld.

### How the two missing densities were finally produced

The first pass abandoned both of these rungs because a single-threaded SCF would
not fit a foreground window. The fix required no new physics and no new code:
**the one-thread rule binds the TIMINGS, because timings must be comparable; it
does not bind the DENSITY, which is an input.** Both densities were therefore
converged on all 12 cores (`RAYON_NUM_THREADS=12`, `OPENBLAS_NUM_THREADS=1` —
BLAS stays at one thread, that being the openblas-rayon hazard), chained across
bounded foreground windows with the harness's existing save/restart knobs, and
then every K build was timed at one thread on the saved result.

| system | basis | nbf | threads | window | iters | wall s | outcome |
|---|---|---|---|---|---|---|---|
| alkane_20 | def2-TZVP | 872 | 12 | w1 | 3 | 523.6 | unconverged, saved (dp_rms 1.733e-3) |
| alkane_20 | def2-TZVP | 872 | 12 | w2 (restart) | 5 | 901.6 | **converged, E = -782.10325185 Ha** |
| alkane_48 | def2-SVP | 1162 | 12 | w1 | 4 | 528.0 | unconverged, saved (dp_rms 8.0e-4) |
| alkane_48 | def2-SVP | 1162 | 12 | w2 (restart) | 5 | 717.6 | **converged, E = -1873.44027902 Ha** |

**These four wall times are NOT comparable timings and must never be quoted as
such** — they are 12-thread numbers recorded only to document what the densities
cost. SCF mode is LinK-K + direct J at `density_conv` 1e-5 in every window, the
same as the first pass used; the only change is the thread count.

The size of the effect is worth recording, because it is the whole reason this
lane could be finished: the first pass measured `alkane_20/def2-TZVP` failing to
complete a SINGLE SCF iteration in a 1740 s single-threaded window. At 12 threads
the same molecule converges in 8 iterations and 1425 s of wall. The first pass's
conclusion that "the ceiling is the SCF" was therefore correct about WHERE the
cost was and wrong about it being a ceiling: it was a self-imposed one-thread
constraint applied to a step that never needed it.

### COSX/LinK along the SIZE axis: it does NOT improve monotonically — the third
### rung reverses the trend the first two showed

**This section was written after two points and has been rewritten after the
third. The two-point reading below it was wrong, and it is left visible rather
than deleted, because it is the exact failure mode the "TOO CLEAN IS A STOP
CONDITION" and "DO NOT DECLARE A NEGATIVE BELOW THE ONSET" conventions warn
about — a two-point trend read as a trend.**

| system | atoms | nbf | COSX s | LinK warm s | COSX/LinK |
|---|---|---|---|---|---|
| alkane_20 | 62  | 490  | 116.234 | 73.219  | 1.587 |
| alkane_32 | 98  | 778  | 209.713 | 192.725 | 1.088 |
| alkane_48 | 146 | 1162 | 354.835 | 291.398 | **1.218** |

The ratio went 1.587 -> 1.088 -> **1.218**. It did not keep falling and it did
not cross parity. The pre-registered prediction for this rung — stated in this
file before it ran — was **COSX ~345 s, LinK ~447 s, ratio ~0.77**. COSX came in
at 354.8 s (a 2.8% hit on a number predicted from the mechanism), but LinK came
in at 291.4 s against 447 s predicted, so **the ratio prediction is a MISS and
the direction of the claim it supported is withdrawn.**

The pairwise tail exponents in atoms show exactly where the earlier reading went
wrong, and it was not COSX:

| segment | COSX total | COSX A-build | LinK warm |
|---|---|---|---|
| C20 -> C32 | 1.289 | 1.198 | 2.114 |
| C32 -> C48 | 1.319 | 1.206 | 1.037 |
| C20 -> C48 (both segments) | 1.303 | — | 1.613 |

**COSX is the stable one.** Its total exponent is 1.289 then 1.319, and its
A-build — 78-85% of the total at every rung — is 1.198 then 1.206, i.e.
reproducible to 1% across two independent size steps. **LinK is the erratic
one**: 2.114 then 1.037. A quantity cannot really scale as N^2.1 and then as
N^1.04 over adjacent segments of the same homologous series, so the honest
reading is that **the alkane_32 LinK point (192.7 s) is anomalously SLOW**, and
the "monotone improvement" the two-point series appeared to show was one outlier
in the denominator, not a trend in the numerator.

That reading has an independent check that does not depend on my exponent
arithmetic: at alkane_48 the exact direct build takes 605.7 s, i.e. LinK's
screening buys 2.08x over exact there — a plausible LinK. Whereas at alkane_32,
LinK at 192.7 s against COSX's 209.7 s would require the same screening to be
buying much less; the C32 cell was never run with the direct arm, so that is
inference, not measurement, and is flagged as such.

**What survives, stated at the strength the data supports:** across 62 -> 146
atoms at double zeta, COSX's cost grows as roughly **N^1.30** and LinK's as
roughly **N^1.61**, so COSX's cost profile is the better-behaved of the two on
the size axis — but the RATIO at any single size is not monotone, stays in the
band **1.09-1.59 in LinK's favour**, and **never crosses parity at double
zeta over the range measured**. Anyone quoting a crossover on the size axis at
double zeta would be quoting the C32 outlier.

Sparsity is behaving as the O(N) argument requires, and this is the one series
here that IS monotone across all three points: `kept_dd` 0.1458 -> 0.0657 ->
**0.0315** and `|A|/nbf` 0.2856 -> 0.1877 -> **0.1285** from C20 to C32 to C48,
with `kept_dd` staying 0.20-0.29x the geometry-only fraction (0.3619 -> 0.2330
-> 0.1565). The pre-registered values for alkane_48 were `kept_dd` ~0.035 and
`|A|/nbf` ~0.13; measured 0.0315 and 0.1285. **Both HIT.** Peak RSS was predicted
~0.9 GB and measured 786 MB — also a hit.

So the mechanism-level predictions (what COSX itself costs, how sparse it gets,
how much memory it holds) were accurate to a few percent at a rung never
previously reached, and the prediction that failed was the one about the
competitor. That is worth separating explicitly: **COSX is now predictable; LinK
is not yet.**

Density policy, applied uniformly: every builder in a rung contracts the SAME D,
so the density cancels from every ratio; what it must NOT be is structurally
unrepresentative (which is why SAD is excluded — see below). Rows whose density
is not fully converged carry the achieved `dp_rms` and are labelled, rather than
being presented as converged.

Densities: alkane_20/def2-SVP reuses `dens_alkane_20_svp.bin` from the earlier
scaling lane (direct J+K SCF, `density_conv` 1e-5). alkane_32/def2-SVP was
converged for this lane: direct J+K SCF, **E = -1249.34789834 Ha, converged,
9 iterations, 1677.4 s** at one thread (PSI 0.00 throughout) —
`dens_alkane_32_svp.bin`. That 28-minute SCF for a single 98-atom double-zeta
density is itself part of the ceiling: at one thread on this box, converging a
density costs more than every K build being compared on it.
alkane_48/def2-SVP and alkane_20/def2-TZVP were converged on 12 cores in the
second sitting — see the density table above; converged, `density_conv` 1e-5.

## THE HEADLINE: the L axis at size, the regime the campaign was authorised for

`alkane_20 / def2-TZVP` — 62 atoms, nbf 872, L_max 3 — is the cell this whole
campaign existed to measure and had never reached. It is the clean L-axis
experiment because it is the SAME 62-atom molecule and the SAME 341 000 grid
points as `alkane_20 / def2-SVP`: the system is held fixed and **only the
angular momentum moves.**

| builder | alkane_20/def2-SVP (L=2, nbf 490) | alkane_20/def2-TZVP (L=3, nbf 872) | SVP -> TZVP factor |
|---|---|---|---|
| COSX | 116.234 s | **301.913 s** | **2.60x** |
| LinK (warm) | 73.219 s | **448.822 s** | **6.13x** |
| COSX / LinK | 1.587 | **0.673** | ratio flips through parity |

**COSX wins here, and it is the first time it has won anything in this campaign.**
Pre-registration for this cell, stated in this file before it ran, was COSX
232-465 s, LinK 366-1098 s, ratio 0.3-0.7. Measured: **301.9 s, 448.8 s, 0.673.
All three inside their bands — a clean three-for-three HIT**, and the ratio is
below parity as predicted.

The mechanism is visible in the split rather than only in the total. Going
SVP -> TZVP at fixed molecule and fixed grid:

* COSX's A-build goes 99.027 -> 256.934 s (2.59x) while the POINT COUNT is
  unchanged at 341 000. The per-point A-build cost goes 2.90e-4 -> 7.53e-4 s/pt.
  COSX pays for L only through the shell dimensions of its 3c1e blocks, and that
  cost is roughly quadratic in nbf at fixed grid (nbf ratio 1.78, nbf^2 ratio
  3.17, measured 2.59 — sub-quadratic because the density screen tightens).
* LinK pays for L through the cost of an analytic four-centre quartet, which
  explodes with angular momentum. 6.13x for a 1.78x change in nbf is an
  effective exponent of **3.35 in nbf**, against COSX's **1.65**.

That is the literature's entire case for seminumerical exchange, reproduced here
as a measurement on this implementation for the first time.

Two controls that make this a measurement rather than an artifact:

* **`kept_dd` and `|A|/nbf` barely move**: 0.1458 -> 0.1443 and 0.2856 -> 0.2850
  from SVP to TZVP. Since the molecule and grid are identical and only the basis
  changed, the sparsity SHOULD be a geometric property and therefore nearly
  invariant — and it is, to 1%. Had COSX's win come from the screen discarding
  more work at TZVP, these would have dropped; they did not, so the speed
  advantage is not bought with sparsity.
* **The accuracy is BETTER at TZVP, not worse**: `max|K_cosx - K_direct|` is
  2.589e-4 (relative 3.31e-5) at TZVP against 4.827e-4 (relative 6.75e-5) at SVP
  on the same grid. A COSX win accompanied by degraded accuracy would not be a
  win — that was stated in advance as the artifact hypothesis for this lane —
  and the accuracy went the other way.

### Where the L-axis advantage does NOT extend, stated so it is not over-claimed

The clean, unflagged comparison that makes the boundary concrete does not need
the direct arm at all — it is LinK's own two numbers. LinK costs **448.8 s at
nbf 872 with L=3** and **291.4 s at nbf 1162 with L=2**: analytic exchange is
1.54x MORE expensive on 33% FEWER basis functions, purely because the angular
momentum went up by one. COSX on the same two cells goes 301.9 -> 354.8 s, i.e.
it gets CHEAPER at the high-L cell, tracking nbf and the grid (which is priced
by ATOMS, and C20 has 62 against C48's 146) rather than by L.

Analytic exchange is priced by angular momentum; COSX is priced by basis size
and atom count. That asymmetry is the whole finding, and it also says where COSX
will NOT help: on a big low-L system — exactly the alkane_48/def2-SVP rung where
COSX loses to LinK at 1.218.

(The flagged direct measurement points the same way — 672 s at C20/TZVP against
605.7 s at C48/SVP, more expensive on fewer basis functions — but it is a
flagged row, so the argument above is built on the LinK pair instead, which is
clean at both ends.)

**So the campaign's authorised regime resolves as: COSX wins on the L axis
(0.673 at 62 atoms, triple zeta) and loses on the size axis at low L
(1.088-1.587 at double zeta). The two axes point in opposite directions, and
only one cell in this lane has both at once — none does, since alkane_32/TZVP
was not reached.** See the next section.

### Prediction for the next rung, stated before it ran — RESOLVED: COSX HIT,
### LinK and the ratio MISSED (measured 354.8 s / 291.4 s / 1.218)

From the two-point tail exponents above (COSX 1.29, LinK 2.12 in atoms), at
alkane_48/def2-SVP (146 atoms, nbf 1162):

* COSX total **~345 s** (of which A-build ~280 s), LinK warm **~447 s**,
  so **COSX/LinK ~ 0.77** — COSX crossing BELOW parity at double zeta.
* `kept_dd` ~0.035, `|A|/nbf` ~0.13, peak RSS ~0.9 GB.

Artifact hypothesis stated alongside: if instead COSX comes in far cheaper than
345 s WITH a `kept_dd` that has collapsed by much more than the ~2x per rung
seen so far, that is a screen that has begun discarding real contributions
rather than a method that has begun to scale — and the tell would be
`max|K_cosx - K_link|` departing from the 4.83e-4 it has held flat across two
rungs. The accuracy column is therefore the control on the timing column, and a
COSX win with a degraded accuracy column is NOT a win.

## The memory wall, in closed form from the two builders' own preflights

This is the part of the ceiling map that does NOT need a timing, and it is the
strongest thing this lane has to say. The two builders differ not only in how
much memory they need but in whether they CHECK, and both facts are read from
the source, not extrapolated.

**COSX has a real preflight and errors cleanly.**
`ferric_scf::cosx_k::check_budget` (cosx_k.rs:1341) computes, per grid block,
`5 * COSX_BLOCK_POINTS * nbf * 8` (planes) + `7 * nbf^2 * 8` (squares:
Ktilde, S_num, S, L, L^T, D[Λ,A], D_occ) + `(threads+1) * 2 * nbf *
COSX_SUB_BATCH_POINTS * 8` (md3c1e per-thread), and returns a
`FerricError::General` naming the requirement and the budget if it does not fit.
Its dominant term is **O(nbf^2)** — it never forms anything of size
`naux * nbf^2`.

Verified live rather than read: with `FERRIC_MEM_BUDGET_GB=0.001`,
`CosxK::new` on water/cc-pVDZ returns

```
CosxK: one grid block needs 0.00 GB (nbf=24, 1024 pts x 5 planes + 2 per-thread
md3c1e buffers) but the memory budget is 0.00 GB — raise [memory] budget_gb /
FERRIC_MEM_BUDGET_GB or use fewer threads
```

i.e. a typed error naming the requirement, before any allocation.

**DF-K had NO preflight when this lane measured it — that has since been FIXED,
and this lane's measurements are what motivated the fix.**

As measured here, `ferric_integrals::three_index_source` was a bare
`if needed <= budget_bytes { in-core } else { spill }` (three_index_source.rs:190).
Above budget there was no error and no disk-space check: it called
`tempfile::tempfile()` and streamed the tensor to `/tmp`, which on this box is
the same 95%-full root partition with ~207 GB free.

**Update (2026-09-09, `5a7136d7 fix(integrals): disk-space preflight + one-shot
warning before 3-index spill`, landed on main after this lane's rung table was
taken).** The asymmetry described below — COSX refusing an overcommit with a
typed error while DF-K silently degraded — is now closed, and the commit cites
this lane's numbers (the ~55 GB C32/TZVP spill re-read per iteration, the 415 GB
C32/QZVP tensor against a 2.0x-smaller free disk) as its motivation. The spill
branches now `statvfs` the directory the writer will actually use (`TMPDIR`, not
an assumed `/tmp`), refuse with a typed error naming size, free space, path and
remedies when the tensor plus a `max(2 GB, 2% of free)` margin does not fit, and
emit a one-shot warning naming the per-iteration re-read when a spill does
proceed. The in-core path is bit-identical and stats nothing.

**What this changes in the reading below, and what it does not.** It changes the
MECHANISM of DF-K's failure from "silently fills a shared partition" to "refuses
with a diagnosis", which removes the safety hazard that made this lane refuse to
execute DF-K at all. It does NOT change any SIZE in the table below, nor any
conclusion drawn from those sizes: DF-K's tensor is still 4.33-1382 GB across
this rung grid against COSX's 645-786 MB measured peak RSS, so DF-K is still out
of memory at every rung from 62 atoms and double zeta upward. The rows below are
therefore left as measured, with this correction noted once at the top rather
than being edited in place — the sizes were and remain the finding.

| system | basis | nbf | naux | COSX block need | DF-K dressed tensor | DF-K / COSX |
|---|---|---|---|---|---|---|
| alkane_20 | def2-SVP  | 490  | 2256 | 0.038 GB | 4.33 GB | 116x |
| alkane_20 | def2-TZVP | 872  | 2256 | 0.085 GB | 13.72 GB | 161x |
| alkane_20 | def2-QZVP | 2400 | 2256 | 0.441 GB | 103.96 GB | 236x |
| alkane_32 | def2-SVP  | 778  | 3588 | 0.072 GB | 17.37 GB | 241x |
| alkane_32 | def2-TZVP | 1388 | 3588 | 0.176 GB | 55.30 GB | 314x |
| alkane_32 | def2-QZVP | 3804 | 3588 | 0.997 GB | 415.36 GB | 417x |
| alkane_48 | def2-SVP  | 1162 | 5364 | 0.133 GB | 57.94 GB | 437x |
| alkane_48 | def2-TZVP | 2076 | 5364 | 0.343 GB | 184.94 GB | 539x |
| alkane_48 | def2-QZVP | 5676 | 5364 | **2.083 GB** | **1382.49 GB** | **664x** |

(COSX column evaluated at one rayon thread, the configuration everything in this
lane runs at; more threads add `2*nbf*256*8` each, which is 23 MB per thread even
at nbf=5676, so the column is essentially thread-independent.)

Three things follow, and only the first two are about COSX being better:

1. DF-K's requirement exceeds this box's RAM at **every rung of this lane**,
   starting at the very first one (alkane_20/def2-SVP, 4.33 GB against ~5.2 GB
   free and a 2 GB configured budget). The failure is not at some far frontier;
   DF-K is already out at 62 atoms and a double-zeta basis.
2. COSX's requirement stays under 2.1 GB through alkane_48/def2-QZVP — 146 atoms
   at quadruple zeta, nbf 5676 — so on the memory axis COSX clears every rung
   this lane can name, and the gap WIDENS with both size (116x -> 437x at SVP)
   and angular momentum (116x -> 236x at C20), because DF-K's tensor carries the
   `naux` factor and COSX's block scratch does not.
3. **The COSX column is block scratch, not process RSS, and the two differ by
   an order of magnitude.** At alkane_20/def2-SVP the formula gives 0.038 GB
   while the measured `VmHWM` for the cell was **645 MB** — and that cell also
   ran LinK and held two K matrices, so it is an upper bound on COSX alone.
   The residue is the grid (341 000 points and their weights), the libint2
   engine pool, and the LinK pair lists. This is the repo's standing
   "budget is read everywhere, spent nowhere" defect, not a new one, and it
   means the COSX column below predicts the quantity `check_budget` GATES, not
   the RSS the cgroup enforces. Every COSX row in the timing table therefore
   carries a MEASURED peak RSS next to it, and the formula is used only for the
   rungs where a measurement was not affordable — flagged as such.
4. But note carefully what item 1 does NOT say. DF-K's over-budget behaviour is
   to SPILL, not to refuse, and a spilled DF-K may still produce a correct K,
   slowly, from disk. At 4.33 GB (C20/SVP) that spill is feasible on this
   partition; at 1382 GB it is not, and at 55-185 GB it would consume a quarter
   to most of the free disk. So "DF-K cannot run" is precise only where the
   tensor exceeds the DISK; between the RAM wall and the disk wall the honest
   statement is "DF-K becomes IO-bound", which is pre-registration R4. The
   deliverable therefore distinguishes the two, rather than collapsing them.

### Where the disk wall actually falls on this box

207 GB free on a Samsung 860 QVO (QLC SSD; sustained write ~80-160 MB/s once
the SLC cache is exhausted, which a multi-GB streaming spill does immediately).
`/tmp` is on that same partition. Classifying each rung by which wall it hits:

| rung | tensor | vs 5.2 GB RAM | vs 207 GB disk | verdict |
|---|---|---|---|---|
| C20/SVP  | 4.33 GB | over | fits (2%) | **spills; IO-bound but feasible** (R4) |
| C20/TZVP | 13.72 GB | over | fits (7%) | spills; IO-bound |
| C32/SVP  | 17.37 GB | over | fits (8%) | spills; IO-bound |
| C32/TZVP | 55.30 GB | over | fits (27%) | spills; ~10 min of pure write per setup |
| C48/SVP  | 57.94 GB | over | fits (28%) | spills; ~10 min of pure write per setup |
| C20/QZVP | 103.96 GB | over | fits (50%) | spills; half the free disk, ~20 min write |
| C48/TZVP | 184.94 GB | over | fits (89%) | at the edge; would leave 22 GB free |
| C32/QZVP | 415.36 GB | over | **EXCEEDS** (2.0x) | **impossible on this box** |
| C48/QZVP | 1382.49 GB | over | **EXCEEDS** (6.7x) | **impossible on this box** |

The spill is not a one-off setup cost. `DfK::build` (df_k.rs:268) drives
`ThreeIndexSource::for_each_block`, which on the spill backend re-reads the
tensor from disk on EVERY build — i.e. every SCF iteration, not once at setup.
So the IO column above is per-iteration: at alkane_32/def2-TZVP, 55.3 GB read
per iteration is ~6 minutes of pure IO per SCF step before a single FLOP, and a
15-iteration SCF pays it fifteen times. That is what "IO-bound" means here
concretely, and it is why the RAM wall, though not a hard impossibility below
the disk wall, is not a merely academic one either.

So the honest split is: DF-K is out of RAM at every rung from 62 atoms /
double-zeta upward, and out of DISK — genuinely unable to produce a K by any
route — from **alkane_32 / def2-QZVP** (98 atoms, nbf 3804) upward. Note also
that, AT THE TIME THIS WAS MEASURED, nothing in the library would TELL you this:
it would begin writing and fail on ENOSPC partway, having filled a 95%-full
shared partition. None of the disk rows above were executed; per the
pre-registration's binding safety rule, the sizes and the mechanism are the
finding and filling the disk to prove it is not.

**That last sentence is now out of date in the best possible way**: `5a7136d7`
(see the update above) makes the library refuse such a spill with a typed error
naming the size, the free space and the path, so a user hitting the C32/QZVP row
today gets a diagnosis instead of a filled partition. The size analysis is
unchanged; only the failure mode is.

### Prediction for alkane_20 / def2-TZVP, stated before that rung ran —
### RESOLVED: three for three HIT (measured 301.9 s / 448.8 s / 0.673)

This is the cell that tests COSX's actual claim — high angular momentum AT size
— rather than size alone. nbf 872, L_max 3, same 62-atom molecule and same
341 000 grid points as rung 1, so the grid is held fixed and only L moves.

* COSX **232-465 s** (2-4x its SVP time: the A-build's 3c1e blocks grow with the
  shell dimensions but not with the point count, which is unchanged).
* LinK **366-1098 s** (5-15x: an analytic quartet's cost explodes with L, which
  is the literature's whole case for seminumerical exchange).
* Therefore **COSX/LinK roughly 0.3-0.7, i.e. clearly BELOW parity.**

If instead COSX/LinK stays at or above 1 here, the L-axis case is not what the
literature claims on this implementation, and the whitepaper cannot lean on it.

## The rungs that were NOT reached in the FIRST sitting — SUPERSEDED, both since
## reached

**Status as of the second sitting (2026-09-09): both rungs described in this
section were subsequently MEASURED, and their pre-registered predictions are no
longer untested. The section is kept as written because its diagnosis was the
thing that unblocked the lane, and because its final claim — that the SCF is the
ceiling — needs correcting rather than deleting.**

The correction, stated up front: the SCF was the BOTTLENECK, not a CEILING. Both
densities were produced in about 20 minutes each once the SCF was allowed the
cores it wanted (see "How the two missing densities were finally produced"). What
follows was the state of knowledge before that.

Attempted and abandoned, recorded rather than omitted. **Both** further rungs
are well inside every K builder's MEMORY limit and both failed for the same
reason: their DENSITY could not be produced in the available foreground windows.

**alkane_48 / def2-SVP** (146 atoms, nbf 1162; COSX block requirement 0.133 GB):

* `k_builder = "link"` SCF, `max_iter` 12: killed at 1750 s without returning,
  so nothing was saved (the harness saves only after `solve_rhf` returns).
* Same with `max_iter` 2, to force a return inside one window: also did not
  complete in 1750 s, i.e. **fewer than two LinK-K + direct-J iterations fit a
  29-minute single-threaded window at nbf 1162**.

**alkane_20 / def2-TZVP** (62 atoms, nbf 872; COSX block requirement 0.085 GB) —
the L-axis-at-size cell, the one this lane most wanted:

* `k_builder = "link"` SCF, default `max_iter`: `Terminated` at the 1740 s
  timeout, no density written. PSI 0.00 before and after, so this is cost, not
  contention.

Both would need 5-8 chained windows (2.5-4 hours) to produce ONE density, before
any K builder is timed.

**This is the lane's actual ceiling, and it is not where the campaign expected
it.** The binding limit was never COSX's memory (peak RSS 645-724 MB against a
6 GB cap, closed-form block requirement 2.08 GB even at alkane_48/def2-QZVP) and
never DF-K's absence. It is that **a single-threaded SCF costs more than every K
build being compared on its output**: 1677 s for one 98-atom double-zeta
density, and >1740 s for a 62-atom TRIPLE-zeta one. The K builds themselves are
2-4 minutes.

Two consequences worth carrying forward:

1. The COSX-vs-LinK comparison at higher L and at 146 atoms is a **cheap**
   measurement (both extrapolate to 3-8 minutes per build) gated behind an
   **expensive** prerequisite. Anyone continuing this lane should converge the
   densities FIRST, in parallel across cores — the SCF is the only part of this
   that wants more than one thread, and the one-thread rule exists for the
   timings, not for the density generation.
2. The pre-registered predictions for both rungs (alkane_48/def2-SVP:
   COSX ~345 s, LinK ~447 s, ratio ~0.77; alkane_20/def2-TZVP: COSX 232-465 s,
   LinK 366-1098 s, ratio 0.3-0.7) stand **UNTESTED**. They are committed
   predictions, not results, and must not be quoted as findings.

**Both of those are now TESTED — see the tables above and the verdict table
below. Point 1 was right and acting on it is what finished the lane; point 2 no
longer holds.** For the record, since consequence 1 is the transferable lesson:
the first sitting concluded that alkane_20/def2-TZVP could not complete ONE SCF
iteration in a 1740 s window. That was a true measurement of a single-threaded
run and a false conclusion about the rung, because nothing required the SCF to
be single-threaded. The one-thread rule is a rule about COMPARABILITY of
timings, and it had been applied to a step that produces no timing at all. This
is worth carrying forward as a general trap: **a constraint adopted for one part
of an experiment silently propagating to the parts it does not govern, and being
mistaken for a property of the system.**

### The rung that is still not reached, and the honest reason

**alkane_32 / def2-TZVP** (98 atoms, nbf 1388, L_max 3) — the cell that would
have BOTH axes at once, size AND angular momentum, and the only one that could
say whether the L-axis win at 62 atoms survives to 98.

It was not attempted, and the reason is not cost:

* Cost was estimated as feasible. From the measured C20/TZVP SCF (8 iterations,
  1425 s at 12 threads) and an nbf ratio of 1388/872 = 1.59, the SCF projects to
  roughly 1-1.5 hours across two or three windows, and the K cells to roughly
  600 s (COSX) and 1100 s (LinK) each. That is comparable to what this sitting
  already spent.
* It was blocked by **machine availability, not by the method**. Partway through
  this sitting another agent's workload started on the same box: load average
  went from 0.4 to 22, CPU pressure `some avg10` to 51%, and available memory to
  2 GB. Under this lane's own binding rules — one thread on an otherwise idle
  box, PSI `full avg10` 0.00 before AND after every kept number — no comparable
  timing can be taken in that condition, and taking the machine for a further
  2-3 hours would have been taking it from someone else's job.

So the correct label for alkane_32/def2-TZVP is **NOT ATTEMPTED (resource
contention)**, which is a different and weaker statement than the first
sitting's "could not be produced". Nothing about it is believed to be
infeasible. Whoever picks it up needs one idle box for about three hours; the
recipe is `scripts/queue/cosx_large_scf_mt.sh` for the density followed by
`scripts/queue/cosx_large_build2.sh main` and `... direct` for the cells.

**Re-checked and declined a second time (13:12), after the lane was explicitly
asked to attempt it.** The box was measured immediately before deciding: load
average 18.6, `/proc/pressure/cpu` `some avg10` 43.5%, 3 GB available memory, and
several of another agent's `ferric-cli` processes plus a multi-core python driver
resident. Both halves of the cell are blocked by that, for different reasons:
the density wants all 12 cores and would be taking them from someone else's
running jobs, and the K builds afterwards need an idle box for a full hour to
produce comparable single-thread numbers. Starting the SCF anyway would have
produced a density (a density does not care about contention) but no timeable
window to use it in, and would have degraded another agent's measurements in the
process. Recorded here rather than left as silence, because "did not run" and
"could not run" are different claims and only the first one is true.

**A prediction for that cell, stated here before it runs so it cannot be fitted
afterwards.** At C20 the L=2 -> L=3 move multiplied COSX by 2.60 and LinK by
6.13, flipping the ratio from 1.587 to 0.673 (factor 0.424). At C48/SVP the
ratio is 1.218 and at C32/SVP it is 1.088. Applying the SAME L-axis factor to
the C32/SVP ratio gives **COSX/LinK ~ 0.46 at alkane_32/def2-TZVP** (band
0.40-0.75, the width allowing for the C32/SVP LinK point being the suspected
outlier — if it is, the SVP anchor is too low and the TZVP ratio lands nearer
0.6-0.75). In absolute terms, **COSX 500-750 s and LinK 900-1600 s.** Sparsity:
`kept_dd` ~0.065 and `|A|/nbf` ~0.19, i.e. essentially the C32/SVP values, since
the L-axis move at C20 left both invariant to 1%. Peak RSS under 1.1 GB.

Artifact hypothesis alongside it, as this lane requires: the L-axis advantage is
real if it is carried by the A-BUILD ratio at roughly unchanged `kept_dd`. If
instead a COSX win at C32/TZVP arrives WITH `kept_dd` well below 0.065, the
screen is discarding work that the bigger basis should have kept, and the tell
would be `max|K_cosx - K_direct|` rising above the ~2.6e-4 seen at C20/TZVP.
Accuracy is the control on the timing, and a faster COSX with a worse K is not a
result.

## Reading against the pre-registration

| pre-registered | outcome |
|---|---|
| P1: DF-K fails first, at the very first rung, by disk-spill with NO preflight refusal | **CONFIRMED AS MEASURED, and the defect it names has since been FIXED.** DF-K's tensor is 4.33 GB at alkane_20/def2-SVP against a 2 GB budget, and at measurement time `three_index_source.rs` had no refusal — it spilled. Never executed, per the safety rule. `5a7136d7` (landed on main after these rungs were taken, citing this lane's numbers) added the disk preflight, typed refusal and one-shot spill warning, so the prediction is confirmed about the code as it was and no longer describes the code as it is. |
| P1: COSX runs furthest, with its ceiling set by WALL TIME not RSS | **CONFIRMED.** Peak RSS 645 / 724 / 786 / 709 MB across all four measured rungs, against a 6 GB cap; the closed-form block requirement is 2.08 GB even at alkane_48/def2-QZVP. |
| P1: direct/LinK fail on TIME around nbf 1000-1400 | **REFUTED.** At nbf 1162 (alkane_48/def2-SVP) LinK builds K in 291.4 s and the exact direct J+K sweep in 605.7 s — both comfortably inside a 30-minute window, not out of it. Neither builder failed on time at any rung of this lane. |
| P1: LinK outlasts direct by 1.2-3x | **HIT on the margin, but no ceiling was reached to test it as a CEILING.** Measured LinK/direct at alkane_48/def2-SVP is 291.4/605.7 = **2.08x**, inside the predicted 1.2-3x band. The prediction was framed as "where each one fails"; since neither failed, this is confirmed as a cost ratio only. |
| P1: the ordering of ceilings is DF-K first, then direct, then LinK, then COSX | **UNTESTABLE AS STATED — only the first place was reached.** DF-K is out at every rung (memory, confirmed). The other three all ran at every rung attempted, so no ordering among them was observed. The lane produced a COST ordering instead, and it is not constant: at double zeta LinK < COSX < direct; at triple zeta COSX < LinK < direct. |
| P3: COSX \|A\|/nbf **< 0.25** at alkane_48/def2-SVP | **HIT — measured 0.1285**, comfortably inside. (This is the same measurement as the later, tighter ~0.13 prediction stated mid-lane; both are recorded because both were committed before the rung ran.) |
| P3: COSX/LinK 8-11x at alkane_32/def2-SVP | **REFUTED — measured 1.088.** See the miss analysis above; note that the follow-on reading of this miss ("the ratio improves with size") was itself refuted by the next rung. |
| P3: kept_dd 0.08-0.12 at alkane_32/def2-SVP | **Near miss, measured 0.0657** — below the band, same direction as the trend. |
| P3: COSX total wall 400-700 s at alkane_32/def2-SVP | **REFUTED — measured 209.7 s.** The prediction used the old total-wall exponent 2.16, which the sparse half transforms removed. |
| **alkane_48/def2-SVP: COSX ~345 s** (stated in this file before the rung ran) | **HIT — measured 354.8 s**, 2.8% high. |
| **alkane_48/def2-SVP: LinK ~447 s** | **MISS — measured 291.4 s**, 35% low. The prediction used LinK's C20->C32 tail exponent 2.12; LinK actually grew at 1.04 over C32->C48. |
| **alkane_48/def2-SVP: COSX/LinK ~0.77 (crossing below parity)** | **MISS — measured 1.218.** COSX did NOT cross parity on the size axis at double zeta, and the ratio rose rather than continuing to fall. The miss is entirely in the denominator: COSX landed where predicted, LinK did not. |
| **alkane_48/def2-SVP: kept_dd ~0.035, \|A\|/nbf ~0.13, peak RSS ~0.9 GB** | **HIT, HIT, HIT — measured 0.0315, 0.1285, 786 MB.** |
| **alkane_20/def2-TZVP: COSX 232-465 s** | **HIT — measured 301.9 s**, mid-band. |
| **alkane_20/def2-TZVP: LinK 366-1098 s** | **HIT — measured 448.8 s**, low in band. |
| **alkane_20/def2-TZVP: COSX/LinK 0.3-0.7, clearly BELOW parity** | **HIT — measured 0.673.** The L-axis claim is upheld: COSX beats LinK at 62 atoms and triple zeta. |
| R1 (COSX hits the same wall as DF-K — would refute the memory argument) | **NOT observed.** COSX ran at every rung attempted, at <1 GB peak RSS. |
| R2 (COSX/LinK worsens with N — would confine the claim to the L axis) | **PARTIALLY observed, and the confinement it implies is now the lane's conclusion.** The ratio does not worsen monotonically (1.587 -> 1.088 -> 1.218), but it never crosses parity on the size axis at double zeta either. It crosses only when L rises. So the defensible claim IS an L-axis claim, which is what R2 said the fallback wording would have to be. |
| R3 (COSX peak RSS grows super-linearly toward the cap) | **NOT observed.** 645 -> 724 -> 786 MB across a 2.4x nbf increase; sub-linear in nbf. |
| R4 (DF-K's spill works, so the wall is performance not feasibility) | **PARTIALLY UPHELD, and reported as such.** DF-K spills rather than refusing, so below the disk wall the honest claim is "IO-bound" (quantified: whole-tensor re-read per SCF iteration), not "cannot run". Only from alkane_32/def2-QZVP upward does the tensor exceed the free disk and DF-K become genuinely impossible here. |
| **alkane_32/def2-TZVP: COSX/LinK ~0.46 (band 0.40-0.75); COSX 500-750 s, LinK 900-1600 s; kept_dd ~0.065, \|A\|/nbf ~0.19; RSS < 1.1 GB** | **UNTESTED — NOT REACHED.** Not attempted for machine-contention reasons (another agent's workload took the box to load 18-22 and 52% CPU pressure), NOT for cost or feasibility reasons; the cost projects to ~3 hours total. This remains a committed prediction and must not be quoted as a finding. |
| **alkane_20/def2-TZVP direct exact-K wall time** | **UNTESTED — measured once at 672.0 s but with PSI `full avg10` 0.01, so DISCARDED per this lane's own protocol; the retake was killed by the same contention.** The accuracy figures from that cell ARE used (a matrix deviation does not depend on timing). |
| alkane_20/def2-QZVP, alkane_48/def2-TZVP, and every QZVP rung | **UNTESTED — never attempted.** Listed in the sizing table for their memory requirements only. No timing, no density, no claim. |

Tally over the whole lane: **six of my quantitative predictions were wrong and
are recorded as wrong; eleven were right; three remain explicitly UNTESTED and
are marked as such rather than being quietly dropped.** The two sittings failed
in opposite directions, which is itself the useful part.

* First sitting: I UNDER-estimated COSX (P3's 8-11x and 400-700 s), because both
  anchors were stale — the pre-fix LinK series and the pre-sparse-half-transform
  total-wall exponent. That is the "AN UNAPPLIED FIX INVALIDATES THE VERDICT"
  convention from the other side: the fixes had landed and my extrapolation had
  not been re-derived from them.
* Second sitting: I OVER-estimated COSX's relative position at alkane_48 (~0.77
  predicted, 1.218 measured) — and did so by extrapolating a two-point tail
  exponent for LinK, the very series whose first two points I had already
  flagged as resting on a corrected builder. Every COSX-side number in that same
  prediction was right to a few percent. **The pattern across both sittings is
  that I can predict COSX from its own mechanism and cannot predict LinK from a
  two-point fit** — and "fit the TAIL, not the whole series" does not rescue a
  fit when the tail is two points long.

The corollary worth carrying: the one prediction set that was fully correct
(alkane_20/def2-TZVP, three for three) was the one derived from a MECHANISM —
"the grid is unchanged so COSX pays for L only through shell dimensions, while
an analytic quartet's cost explodes with L" — rather than from an extrapolated
exponent.

## COSX's accuracy across the lane, and why its flatness is WEAKER evidence than
## it looks

The pre-registration asked whether COSX's accuracy stays flat as the system
grows. It does, and the numbers are almost too flat:

| system | basis | max\|K_cosx − K_link\| | max\|K_cosx − K_direct\| | \|\|K\|\|max | relative |
|---|---|---|---|---|---|
| alkane_20 | def2-SVP  | 4.825e-4 | — | 7.157e0 | 6.742e-5 |
| alkane_32 | def2-SVP  | 4.827e-4 | — | 7.157e0 | 6.744e-5 |
| alkane_48 | def2-SVP  | 4.827e-4 | **4.8273e-4** | 7.157e0 | 6.745e-5 |
| alkane_20 | def2-TZVP | 2.589e-4 | **2.5886e-4** | 7.828e0 | 3.307e-5 |

So the answer to "does COSX's accuracy stay flat?" is **yes — 4.825e-4 ->
4.827e-4 -> 4.827e-4 across C20 -> C32 -> C48, constant to 0.04% over a 2.4x
change in nbf and a 2.4x change in atom count.** That is a HIT on the
pre-registered expectation.

**But this lane's own conventions say "TOO CLEAN IS A STOP CONDITION", so it was
audited rather than written up.** Three digits of agreement across three system
sizes is exactly the fingerprint of a number that is not actually varying with
the thing it is plotted against. The audit:

* `||K||max` is **identically 7.157e0** at every SVP alkane in this repo, from
  butane (nbf 106) to alkane_48 (nbf 1162), and identically 7.827-7.828e0 at
  every TZVP alkane. That is not a bug: `max|K|` in an alkane chain is attained
  on a local C-H/C-C block, and a local quantity in a homologous chain saturates
  after the first few units. Lengthening the chain adds more copies of the same
  environment; it does not create a larger matrix element.
* The same locality explains the deviation. COSX's error is a grid-quadrature
  error on that same local block, so it too saturates. The constancy is
  therefore expected physics for THIS system class.
* **The control that shows the number is not simply stuck**: change the basis
  instead of the length, and it moves — 4.827e-4 -> 2.589e-4 and 7.157 -> 7.828
  at C20 going SVP -> TZVP. A frozen buffer would not do that.
* **The independent-construction check**: at both cells where the exact direct
  build was run, `max|K_cosx − K_direct|` reproduces `max|K_cosx − K_link|` to
  four digits (4.8273e-4 vs 4.827e-4; 2.5886e-4 vs 2.589e-4). Since LinK and the
  direct four-centre sweep are different constructions, this says the deviation
  is COSX's own quadrature error against exact — not an artifact of comparing
  COSX to a screened builder — and it retires the earlier concern in Finding 0
  about which builder owned the residual.

**The honest caveat, stated because the flatness will be quoted:** a linear
alkane is the system class MOST likely to show a saturating max-element, so this
evidence is weak for the general claim "COSX's accuracy is size-independent". It
is strong for "COSX's accuracy does not DEGRADE with size on chains", which is
what was measured. Demonstrating the general claim needs a system class whose
max|K| actually grows — a globular or conjugated system — and no such system was
run in this lane.

## Protocol decision: SAD guess densities are NOT usable in this lane

Recorded because it constrains everything below and cost the lane its cheap
route. The obvious way to climb rungs without paying for an SCF at each one is
to contract the SAD guess: every builder contracts the same D, so it cancels
from every ratio. It does not work here. `ferric_scf::guess::sad_guess` writes
ONLY atom-diagonal blocks and leaves every off-diagonal block exactly zero, so
COSX's density-driven shell-pair screen would discard nearly every off-atom
pair, and `kept_dd` — one of the quantities this lane is asked to report —
would measure the guess, not the molecule. At def2-QZVP it is worse still: the
`l >= 4` branch skips the free-atom solve entirely and leaves those blocks
zero. A COSX wall time taken on a SAD density would be an artifact that looks
exactly like the O(N) sparsity result the method predicts, which is the failure
mode `docs/ao-laplace-locality-saturation.md` §5 is the worked example of.
Densities in this lane are therefore SCF densities, and where one SCF does not
fit a foreground window that is itself reported as part of the ceiling.


## The plain statement asked for

**Largest (system, basis) where COSX produced a K:** alkane_48 / def2-SVP,
146 atoms, nbf 1162, in 354.8 s at one thread with 786 MB peak RSS. The
highest-angular-momentum cell it produced a K for is alkane_20 / def2-TZVP,
nbf 872, L_max 3, in 301.9 s.

**Could the alternatives?** Split honestly, because the three alternatives
behave completely differently:

* **DF-K: no, and it could not at any rung of this lane.** Its dressed 3-index
  tensor is 57.94 GB at alkane_48/def2-SVP and 13.72 GB at alkane_20/def2-TZVP,
  against ~5 GB of free RAM (and 4.33 GB already at the smallest rung). It was
  refused rather than run, because at the time of measurement the library's
  response to being over budget was not an error but a silent spill to `/tmp` on
  a 95%-full partition, re-read every SCF iteration. (That silence is since
  fixed by `5a7136d7`, which this lane's numbers motivated; the sizes and the
  conclusion are unaffected.) Above alkane_32/def2-QZVP (415 GB) the tensor
  exceeds the free disk outright and DF-K cannot produce a K by any route on
  this box.
* **LinK: yes, at every rung — and which of the two is faster depends entirely
  on the angular momentum.** At double zeta LinK is faster (COSX/LinK 1.587,
  1.088, 1.218 at C20, C32, C48). At triple zeta COSX is faster (0.673 at C20).
* **Direct exact four-centre J+K: yes, and it is never competitive with either**
  — 605.7 s at alkane_48/def2-SVP against LinK's 291.4 s and COSX's 354.8 s. It
  serves here as the accuracy reference, and it confirms both screened builders
  are doing real work rather than skipping it.

### Does the L-axis-at-size regime show COSX winning, tying, or losing?

**It shows COSX WINNING on the L axis, LOSING on the size axis at low L, and the
combined regime is still unmeasured.** Stated precisely, because the campaign
was authorised on this question:

* **L axis, at 62 atoms: COSX WINS, 0.673.** Holding the molecule and the grid
  fixed and moving only def2-SVP -> def2-TZVP costs COSX 2.60x and LinK 6.13x.
  This is the first COSX win recorded anywhere in this campaign, it was
  predicted in advance within a pre-registered band (0.3-0.7), and it came with
  BETTER accuracy (2.589e-4 vs 4.827e-4) and unchanged sparsity (`kept_dd`
  0.1443 vs 0.1458), so it is not bought by discarding work.
* **Size axis, at double zeta: COSX LOSES at every size measured**, 1.587 /
  1.088 / 1.218 at 62 / 98 / 146 atoms. It never crossed parity, and the
  apparent monotone improvement seen after two points was reversed by the third.
  COSX's own scaling is the better of the two (N^1.30 vs N^1.61 in atoms, with
  COSX's A-build exponent reproducible to 1% across both segments), but a better
  exponent has not yet turned into a win at any size that fits this box.
* **Both axes at once: NOT MEASURED.** alkane_32/def2-TZVP would answer it and
  was blocked by machine contention, not by cost or by any builder's limit. The
  prediction for it is on the record above (ratio ~0.46, band 0.40-0.75).

So the defensible claims from this lane are:

1. **DF-K is out of memory from 62 atoms and double zeta upward**, by a margin
   that grows from 116x to 664x across the rung grid, and this follows from the
   two builders' own sizing (`naux*nbf^2` vs `O(nbf^2)`) rather than from an
   extrapolation. COSX's peak RSS was 645-786 MB at all four measured rungs.
2. **COSX's advantage is an ANGULAR-MOMENTUM advantage, not a system-size
   advantage.** That is a narrower claim than the campaign set out to make and
   it is the one the data supports: measured 0.673 at triple zeta, measured
   1.088-1.587 (i.e. a loss) at double zeta across a 4.7x range in system size
   (62, 98 and 146 atoms; the 2.4x figure predates the C48 rung).
   The whitepaper should say "COSX wins at high L" and must NOT say "COSX wins
   at large N".
3. **COSX's accuracy does not degrade with size on these systems**
   (`max|K_cosx − K_direct|` 4.827e-4 at C48/SVP, matching the C20 and C32
   values to 0.04%) and improves with angular momentum (2.589e-4 at C20/TZVP).
   The flatness is partly a property of linear alkanes — see the audit section —
   so quote it as "does not degrade on chains", not as size-independence.
4. **COSX is now predictable from its own mechanism and LinK is not.** Every
   COSX-side pre-registered number at the two new rungs landed within a few
   percent (354.8 vs ~345 s; `kept_dd` 0.0315 vs ~0.035; `|A|/nbf` 0.1285 vs
   ~0.13; RSS 786 MB vs ~0.9 GB), while both LinK predictions derived from
   two-point tail exponents missed badly in both directions.

## What a follow-up should do first

1. **alkane_32 / def2-TZVP** — the one cell that puts both axes together, and
   the only thing standing between this lane and a complete answer. Needs one
   idle box for ~3 hours. Density via
   `scripts/queue/cosx_large_scf_mt.sh alkane_32 def2-tzvp <secs> w1` (12
   threads, chained with `RESTART=`), then
   `scripts/queue/cosx_large_build2.sh main alkane_32 def2-tzvp <secs>` and
   `... direct ...` at one thread. Prediction is pre-registered above.
2. **Re-take alkane_32 / def2-SVP LinK.** Its 192.7 s is the suspected outlier
   that produced the false "ratio improves with size" reading, and it is cheap
   to re-run on the existing saved density. Add the direct arm at that rung too,
   which this lane never ran there — LinK's screening buys 2.08x over exact at
   C48 and, if the C32 point is genuine, should buy noticeably less there.
3. **A non-chain system at high L**, to test whether the accuracy flatness and
   the L-axis win survive outside linear alkanes. Every system in this lane is a
   linear alkane, which is the most favourable possible geometry for a
   density-driven screen and the least informative one for a saturating max|K|.

## Lane verdict — one paragraph, for lifting into the whitepaper

Across four measured cells on one 12-core box — linear alkanes at 62, 98 and 146
atoms in def2-SVP (nbf 490/778/1162) and at 62 atoms in def2-TZVP (nbf 872), each
K build timed single-threaded on a converged SCF density with memory pressure
verified zero either side — **ferric's COSX is faster than its LinK when the
angular momentum is high, and slower when only the system is large.** Holding the
molecule and the 341 000-point grid fixed and moving def2-SVP to def2-TZVP costs
COSX 2.60x but LinK 6.13x, taking COSX/LinK from 1.587 to **0.673**; along the
size axis at double zeta the same ratio is 1.587, 1.088 and 1.218 at C20, C32 and
C48 and never crosses parity, even though COSX's cost grows more slowly with atom
count than LinK's (N^1.30 vs N^1.61, with COSX's dominant A-build term
reproducible to 1% across both size steps). The high-L win is not bought by
approximation: COSX's sparsity is unchanged between the two bases (kept_dd 0.1443
vs 0.1458) and its error against the exact four-centre K is smaller at triple zeta
than at double (2.589e-4 vs 4.827e-4, relative 3.3e-5 and 6.7e-5), the latter
confirmed by two independent references agreeing to four digits. Separately and on
sizing alone, DF-K is not a competitor anywhere in this range: its dressed
three-index tensor is 4.3 GB at the smallest cell and 57.9 GB at the largest,
against COSX's 645-786 MB measured peak RSS, a gap that widens with both size and
angular momentum because the tensor carries a naux factor that COSX's block
scratch does not. **What remains unmeasured is the combination** — no cell in this
lane has both high L and large N, so whether the 0.673 advantage holds, grows or
erodes as the system grows at fixed triple zeta is not known; alkane_32/def2-TZVP
is the single experiment that would settle it, it is projected at ~3 hours, and a
prediction for it (ratio ~0.46, band 0.40-0.75) is pre-registered above. The
defensible claim is therefore **"COSX wins at high angular momentum"**, not "COSX
wins at large N", and every system measured here is a linear alkane, which is the
most favourable geometry for a density-driven screen.
