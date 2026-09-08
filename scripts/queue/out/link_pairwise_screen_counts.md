# LinK pairwise density screen (FourPairK) — quartet COUNTS, before vs after

Date: 2026-09-08. Branch `fix/link-pairwise-screen` off origin/main `e2cd1898`.
Design + pre-registration: `link_pairwise_screen_design.md` (same directory,
commit 592c8adf, written before any code change).

Counts only — no wall-clock timings are reported or quoted. The harness prints
wall/cpu values; they are NOT timings here (the box was shared, and one
`direct_k` row shows wall 307.8 s vs cpu 50.0 s, i.e. it was descheduled).

## Protocol as run

* Harness `crates/ferric-scf/tests/link_fixed_audit.rs` (`#[ignore]`d),
  release, `LFA_BUILDERS=direct_k,direct_jk,link LFA_BUILDS=1`, thresh 1e-12
  (`RhfConfig::default().integral_thresh`), `OPENBLAS_NUM_THREADS=1`,
  `RAYON_NUM_THREADS=1`, `scripts/ferric-limited --max=4G --high=3600M`.
* Density: converged direct SCF, identical for all three builders in a cell
  (C8 in-process; C16 via `LFA_DENSITY_OUT`/`_IN`, E = -625.2555169626,
  10 iters — bit-identical to the pre-fix run's density).
* Counts are what each `build()` returns: unique canonical shell quartets that
  survived the builder's screen and for which libint2 returned a block.
  Deterministic and thread-invariant.
* BEFORE column = `link_fixed_counts.md` (post-#50, pre-this-branch), same
  systems, same threshold, same density provenance.

## The load-bearing result: LinK/build_jk

| system / def2-SVP | nsh | nbf | DirectK | build_jk | LinK BEFORE | LinK AFTER | LinK/build_jk BEFORE | **AFTER** |
|---|---|---|---|---|---|---|---|---|
| alkane_8  | 102 | 202 | 8,775,381  | 8,107,064  | 8,775,277  | **8,023,337**  | 1.082 | **0.9897** |
| alkane_16 | 198 | 394 | 48,644,165 | 42,916,434 | 48,642,325 | **40,939,242** | 1.133 | **0.9539** |
| alkane_20 | 246 | 490 | 80,535,973 | 70,171,415 | (not taken) | **65,105,182** | (n/a) | **0.9278** |

LinK now evaluates FEWER quartets than the default builder at all three sizes,
and the margin GROWS monotonically with system size (1.03% -> 4.61% -> 7.22%
fewer at C8/C16/C20). Growth with size is
the shape a locality-exploiting screen should have; a fixed fraction would have
been the fingerprint of arithmetic rather than physics.

vs DirectK (the like-for-like comparator, same Schwarz product, looser density
key): 0.99999 -> 0.91430 (C8), 0.99996 -> 0.84161 (C16). LinK remains a strict
subset, as #50 established.

## Correctness alongside the count drop

A count drop is only a fix if K does not move. `max|K - K_DirectK|`:

| system | build_jk vs DirectK (pre-existing baseline) | LinK vs DirectK BEFORE | LinK vs DirectK AFTER |
|---|---|---|---|
| alkane_8  | 9.176e-12 | 2.4e-14 | 9.923e-12 |
| alkane_16 | 9.399e-12 | 2.4e-14 | 1.288e-11 |
| alkane_20 | 9.506e-12 | (not taken) | 1.301e-11 |

The BEFORE value was small for a reason that stopped being true: LinK and
DirectK then shared an IDENTICAL per-quartet screen (both global max|D|), so
they walked the same quartets and the difference measured screen SAMENESS, not
accuracy. Now that LinK screens differently, it lands at the same
threshold-scale residual two differently-screened direct builders already show
between themselves (9.2-9.4e-12), at 1.1-1.4x that baseline. Independent
confirmations that this is a screening residual and not a broken bound:

* `link_scf_anchor`: C16 single-build `max|K_link - K_direct|` = 5.764e-12;
  C8/butane/water SCF energies agree to 1.1e-11 / 7.0e-12 / 0.0 with IDENTICAL
  iteration counts (10/10, 10/10, 11/11).
* the trivial-limit anchor (thresh -> 0) is unchanged — see
  `screening_exactness`.

## Kept pair fractions (same densities, thresh 1e-12)

| system | sp `Q·Qmax > t` | dp `\|D\|·qmax(j)·qmax(σ) > t` |
|---|---|---|
| alkane_8  | 8890/10404 = 0.8545  | 10404/10404 = **1.0000** |
| alkane_16 | 21960/39204 = 0.5601 | 39204/39204 = **1.0000** |
| alkane_20 | 28488/60516 = 0.4708 | 60516/60516 = **1.0000** |

Unchanged by this branch — neither list was modified. The GEOMETRIC list prunes
and its pruning strengthens with size (0.855 -> 0.560 -> 0.471); the DENSITY
list prunes nothing at ANY of the three sizes, including C20 at ~62 Bohr.

## The density-mask negative — now three independent sightings

The dp list keeping a full square is not a ferric-LinK quirk. Three independent
constructions, three different code paths, same verdict:

1. **This branch's derivation** (`pairs.rs`, commit e03fcad2):
   `|D|·qmax(j)·qmax(σ) > t` degenerates to `|D| > t/Qmax²` because `qmax(x)`
   maximizes over ALL partners and is a molecule-wide constant. MEASURED
   directly: `qmax` spans only **3.13x** across shells, and IDENTICALLY at C8
   and C16 (1.8785 / 0.60007 both) — so the product varies <10x while the mask
   must span ~50 Bohr. Test:
   `qmax_spread_is_small_enough_to_explain_the_vacuous_dp_list`.
2. **PR #52's threshold sweep** (open-shell k_builder, independent author):
   post-#50 `DensityPairs` prunes nothing at production thresholds —
   alkane_4/8 full square at 1e-12…1e-6; alkane_16 full square to 1e-8, first
   pruning only at 1e-6 (37696/39204 ≈ 3.8%). The onset threshold being
   size-independent is what `|D| > t/Qmax²` predicts.
3. **PR feat/cosx-sparse-gemm** (COSX shell-sparse half-transforms,
   independent author, different algorithm entirely): its density-derived row
   mask Λ (shells with max|D_ls| >= 1e-10) keeps |Λ|/nbf = **1.0000** at
   alkane_8, while its purely GEOMETRIC AO mask |A|/nbf falls 0.818 -> 0.286
   across C4->C20 and delivers that method's entire win.

The split is the same in all three: **a density-derived pair/row mask does not
bite at these sizes; a geometry-derived one does.** Agreement between three
independent constructions is the form of corroboration that can distinguish a
real effect from a construction bug (a bug is deterministic and would reproduce
across systems and bases, but not across three unrelated implementations).

Consequence for this ticket: the fix that mattered was NOT the pair list. It was
moving the density information to where it is actually per-pair — the
per-quartet screen — which is what FourPairK does.

## ONE OPEN FINDING: the QQR-vs-Schwarz agreement test regressed

`screening_exactness::link_k_qqr_matches_schwarz_at_production_thresh` FAILS on
this branch. 9 of 10 tests in that file pass; this is the one.

```text
alkane_8/cc-pVDZ, thresh=1e-8:  max|K_QQR - K_Schwarz|
  baseline when the test was written   3.0345e-9
  its "100x-too-aggressive envelope" mutant   5.3525e-7
  THIS BRANCH                          1.4889e-7      (bar 1e-7)
```

**What it is not.** It is not a bound-validity failure of `FourPairK`. Both
trivial-limit anchors PASS unchanged — `link_k_matches_dense_in_the_trivial_limit`
AND `link_k_qqr_matches_dense_in_the_trivial_limit` — and an invalid bound
cannot pass those at any threshold. Both alkane_8 threshold sweeps
(`..._schwarz` and `..._qqr`) also pass, as does
`screening_does_not_shift_the_energy_at_production_thresh` and both PySCF
cross-checks.

**What it probably is, stated as a hypothesis and NOT as a resolved verdict.**
That test compares two BOUNDS against each other (QQR vs Schwarz) inside LinK,
at a deliberately loose 1e-8 threshold chosen so the inner screen is
load-bearing. The density key is common to both arms, and tightening it from a
global scalar to the pairwise exchange max moves BOTH arms deeper into the
regime where QQR's distance envelope is the discriminating factor. So the
divergence between the two bounds is expected to grow, and the quantity the test
measures is not held fixed by this change. On that reading 1.49e-7 is QQR
truncation surfacing at a loose threshold, not new error in either arm.

**Why I am not closing it on that reading.** The measured value sits at 28% of
the mutant value the test's own author used to calibrate "too aggressive", which
is close enough that it deserves an independent check rather than an
explanation. The honest status is: the fix is not exonerated here, and the test
is not condemned either. What would settle it, in order of cost:

1. run this same test on UNMODIFIED origin/main at thresh 1e-8 to confirm the
   3.0345e-9 baseline still reproduces there (isolates branch vs environment);
2. sweep the comparison across thresholds — if this is QQR truncation the gap
   must SHRINK as the threshold tightens and vanish at thresh 0 (where it is
   already known to pass);
3. only if neither holds, treat it as a `FourPairK`/QQR interaction and audit
   `estimate()` under the new key.

Neither (1) nor (2) was run: each needs a fresh converged density plus repeated
dense builds on 202 basis functions, and the count table and mutation proofs
were the priority for this pass. Flagged for the coordinator as the one
outstanding item, not silently carried.

## What this does NOT say

No cost, exponent, crossover, or COSX/LinK ratio. Nothing here is a timing. The
quartet-count reduction (1.03% / 4.61% / 7.22% at C8/C16/C20 vs build_jk) is an upper
bound on any speedup, not a measurement of one: LinK also pays pair-list
bookkeeping and, in the SCF, a separate DirectJ sweep. Whether it is FASTER
than build_jk is unmeasured and is a separate timing pass.
