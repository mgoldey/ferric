# Fixed-LinK re-measurement — pre-registration

Date: 2026-09-07. Branch `fix/link-occ-path` (PR #50: three pair-list fixes in
`link_k.rs` / `pairs.rs`, HEAD 52f1f88b). Written and committed BEFORE any
number was taken.

## Why

Every LinK timing in this repo's COSX work (`l_axis_results.md`,
`cosx_scaling_results.md`, the "Choosing how exchange is built" section of
`site/src/methods/scf.md`, the COSX row of `site/src/reference/validation.md`)
was taken against the PRE-fix kernel, which silently SKIPPED quartets (its ket
loop covered one of four density blocks, its density-pair criterion used the
wrong Schwarz factor, and its significant-pair cut was at `Q > sqrt(thresh)`).
A kernel that skips work is too fast. Every LinK number, every COSX/LinK ratio
and the book's "LinK for anything beyond a few heavy atoms; ~N^1.5" guidance
therefore has to be re-measured against the fixed kernel — and the fix has to
be checked for OVER-inclusion (a pair list that keeps everything turns LinK
into a slower direct build).

Pre-fix numbers being replaced (one thread, warm build, Schwarz 1e-12):

| cell | pre-fix LinK K (s) | pre-fix direct J+K (s) | COSX K (s) |
|---|---|---|---|
| butane/def2-SVP  | 1.003 (l_axis) / 1.012 (scaling) | 1.38 (PSI-dirty, qualitative) | 7.653 |
| butane/def2-TZVP | 6.482 | 9.6 (PSI-dirty, qualitative) | (book: 358 s full SCF) |
| butane/def2-QZVP | 259.4 | 400.5 | 137 (book) |
| C4/C8/C12/C16/C20 def2-SVP | 1.012 / 5.099 / 10.059 / 15.645 / 21.384 | not measured | 7.653 / 28.373 / 58.694 / 107.664 / 168.494 |

Pre-fix LinK tail exponent (C12-C20, atoms): 1.54. Pre-fix pair fractions on
butane/def2-SVP at 1e-12: significant pairs 2698/2916 kept; density pairs
120/2916 at the SAD guess (from `tests/link_scf_anchor.rs`).

First post-fix datum (not a K-build timing): alkane_8/def2-SVP FULL SCF 196 s
LinK vs 99 s direct (default pool).

## Protocol (fixed before measuring)

* Harness: `crates/ferric-scf/tests/link_fixed_audit.rs` (new, `#[ignore]`d,
  env-driven, one process per timed cell). Release build.
* Density: converged from a DIRECT SCF (`RhfConfig::default()`, default pool
  — the SCF is not a timing), saved to a file and loaded by every timing
  process of that cell, so all builders contract the identical `D`.
* Builders, each a WARM build on that density (cold build first, warm =
  last, warm is the number reported), every timed segment inside an explicit
  1-thread rayon pool with `OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1`;
  cpu-s and wall-s both printed and must agree (cpu/wall within 2%, else
  discard):
  - `DirectK` (K only, Schwarz + global max|D| screen, thresh 1e-12) — the
    like-for-like comparator (same per-quartet screen as LinK);
  - `build_jk` (J+K in one sweep, Häser-Ahlrichs shell-pair density max) —
    ferric's DEFAULT builder, an upper bound on a direct K alone; reported
    because it is what a user gets without `k_builder`;
  - `LinkK` (Schwarz, thresh 1e-12), quartet count returned by `build`, plus
    the significant-pair and density-pair totals of the FIXED lists and of
    the PRE-fix criteria (recomputed on the same density: old sp
    `Q² > thresh`, old dp `max|D|·Q(j,σ) > thresh`);
  - `CosxK` (md3c1e, (50,110), overlap fit ON, density-driven screen at the
    default 1e-7) — the production config; K-build time from
    `last_timings().total_s` and the timed wall.
* `/proc/pressure/memory` `full avg10` before AND after every segment; any
  segment with a nonzero value is DISCARDED and rerun.
* All runs under `scripts/ferric-limited --max=4G --high=3600M --` in the
  foreground, `timeout 480` per process. If fixed LinK at QZVP does not fit
  one window, that is reported as the finding at that size and nothing is
  extrapolated.
* Quartet counts: `DirectK::build`, `build_jk` and `LinkK::build` all return
  the number of unique canonical shell quartets that survived their screen
  AND for which libint2 returned a block, so they are directly comparable.

## Hypotheses (stated before measuring)

### H1 — cost of the fixed kernel (butane)

The fixed ket loop visits the union `dp(i) ∪ dp(j)` for BOTH ket orderings and
keeps ~all pairs of a 14-atom molecule, so on butane the fixed LinK evaluates
nearly the same quartet set as `DirectK` and pays the pair-list bookkeeping on
top. Expected: fixed LinK K ≈ 0.9-1.3× `DirectK` at every basis (SVP/TZVP/
QZVP), i.e. roughly 1.5-2.5× the pre-fix LinK at SVP/TZVP and ≥1.3× at QZVP;
fixed LinK at QZVP expected 300-450 s (vs pre-fix 259 s, direct J+K 400 s).
COSX/LinK at QZVP expected to move from 0.54× to ~0.3-0.45×.

### H2 — over-inclusion check (alkane_8, alkane_16 / def2-SVP)

By construction LinK's evaluated set is a SUBSET of `DirectK`'s (identical
per-quartet screen, plus pair-list restriction), so `n_link ≤ n_directk` is
expected always; `n_link > n_directk` would mean a dedup/ownership bug.
The informative number is the RATIO `n_link / n_directk`:
* correct AND useful LinK: ratio clearly < 1 and FALLING with size (the pair
  lists prune long-range quartets that the global-max|D| screen keeps) —
  expected ~0.6-0.9 at C8, ~0.3-0.6 at C16;
* over-inclusive fix: ratio ≈ 1 (≥ 0.95) at C16 — the lists no longer prune
  anything and LinK is a direct build with overhead; report and STOP (do not
  touch link_k.rs / pairs.rs blind);
* vs the production `build_jk`: its Häser-Ahlrichs pairwise density screen is
  tighter than LinK's global max|D|, so `n_link` may EXCEED `n_jk` even when
  the lists are correct; that is a screening-quality finding, not a #50
  defect, and will be reported as such.
Kept fractions expected: significant pairs ~0.93 (C8) / ~0.75 (C16) of nsh²
(vs pre-fix ~0.92 / ~0.70 at the tighter Q² cut); density pairs on the
converged D expected ~0.5-0.7 (C8) / ~0.3-0.45 (C16) of nsh² (the pre-fix
`|D|·Q(j,σ)` cut was Gaussian-decaying and kept far fewer — that WAS the bug).

### H3 — scaling (alkane_4..20 / def2-SVP)

Expected fixed-LinK tail exponent (C12-C20, atoms): 1.7-2.1 (pre-fix 1.54; the
union ket loop roughly doubles the per-pair ket work and the density-pair
list now decays exponentially instead of like an overlap). Expected `DirectK`
tail exponent: 2.0-2.4 (Schwarz + global max|D| on a 1-D chain). Expected
crossover LinK-faster-than-direct: NOT inside C4-C20 at def2-SVP if the
exponent gap is ≤ 0.3 (LinK ~2× direct at C8 from the SCF datum); if the gap
is ≥ 0.5 the crossover lands around C20-C32. COSX/LinK ratios at SVP expected
to fall from 5.6-7.9× to ~3-5× (COSX numbers reused unchanged — COSX does not
call LinK).

### Artifact hypotheses

* A timing where cpu ≠ wall (contention) or PSI ≠ 0 — discarded, rerun.
* LinK count == DirectK count EXACTLY at every size — a fingerprint of the
  lists being vacuous (over-inclusion), not of chemistry.
* LinK K ≠ DirectK K beyond ~1e-9 (the anchor bar) on any cell — the fix
  regressed; stop and report.
* Warm LinK slower than cold — timing noise or pool storm; rerun.
