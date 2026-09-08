# Fixed-LinK over-inclusion check — quartet COUNTS (no wall-clock timings)

Date: 2026-09-08. Branch `fix/link-docs-and-counts` (origin/main after #50
merged: `a272333c` ket loop + density-pair bound, `d09d6a0d` significant-pair
criterion). Pre-registration: `link_fixed_prereg.md` (commit 7a793070 on this
branch, written before any number). The timing part of that prereg (H1, H3,
the three-basis butane table, the C4-C20 scaling column) was DEFERRED by the
coordinator because the box was contested; only the load-immune COUNTS
(H2) were taken. The harness's printed wall/cpu values are NOT reported as
timings and must not be quoted.

## Protocol as run

* Harness `crates/ferric-scf/tests/link_fixed_audit.rs` (new, `#[ignore]`d),
  release, `LFA_BUILDERS=direct_k,direct_jk,link LFA_BUILDS=1`, thresh 1e-12
  (`RhfConfig::default().integral_thresh`), `OPENBLAS_NUM_THREADS=1`,
  `FERRIC_MEM_BUDGET_GB=2`, `scripts/ferric-limited --max=4G --high=3600M`.
* Density: the plain DIRECT SCF (`RhfConfig::default()`, ambient 12-thread
  pool — deterministic builders, counts are thread-invariant), converged
  (alkane_8 E = -313.2093181436, 10 iters; alkane_16 E = -625.2555169626,
  10 iters). All three builders contract the identical converged `D`.
* Counts are what each `build` returns: unique canonical shell quartets that
  survived the builder's screen and for which libint2 returned a block.
  `DirectK` and `LinK` use the SAME per-quartet screen
  `Q(12)·Q(34)·max|D| ≥ thresh` (global max|D|); `build_jk` (the DEFAULT
  builder, J+K in one sweep) uses the Häser-Ahlrichs shell-pair density max
  `max(D12,D34,D13,D14,D23,D24)`.
* Each timed segment ran in an explicit 1-thread rayon pool; PSI `full
  avg10` was 0.00 before and after every segment (cpu == wall to <0.1%, for
  the record only).

## Counts

| system / def2-SVP | nsh | nbf | DirectK quartets (global max\|D\|) | fixed LinK quartets | LinK − DirectK | LinK / DirectK | `build_jk` quartets (pairwise D) | LinK / build_jk | max\|K_link − K_directK\| |
|---|---|---|---|---|---|---|---|---|---|
| alkane_8  | 102 | 202 | 8,775,381  | 8,775,277  | −104   | 0.99999 | 8,107,064  | 1.082 | 2.4e-14 |
| alkane_16 | 198 | 394 | 48,644,165 | 48,642,325 | −1,840 | 0.99996 | 42,916,434 | 1.133 | 2.4e-14 |

(`build_jk` K vs DirectK K: 9.2e-12 / 9.4e-12 — the two direct screens differ
at the threshold scale, as they should.)

## Kept pair fractions (same converged density, thresh 1e-12)

| system | significant pairs, FIXED `Q·Qmax > t` | pre-fix `Q² > t` | density pairs, FIXED `\|D\|·qmax(j)·qmax(σ) > t` | pre-fix `\|D\|·Q(j,σ) > t` |
|---|---|---|---|---|
| alkane_8  | 8890/10404 = **0.8545** | 7148/10404 = 0.6870 | 10404/10404 = **1.0000** | 8092/10404 = 0.7778 |
| alkane_16 | 21960/39204 = **0.5601** | 16076/39204 = 0.4101 | 39204/39204 = **1.0000** | 19152/39204 = 0.4885 |

The pre-fix values are the old criteria recomputed on this density in the
harness (`prefix_pair_totals`), not numbers copied from an earlier log; the
only earlier logged value, butane/def2-SVP sp 2698/2916 at 1e-12
(`tests/link_scf_anchor.rs`), is the same `Q² > t` cut.

## Reading against the pre-registration (H2)

* **Subset property: HOLDS.** At both sizes the fixed LinK evaluated FEWER
  quartets than `DirectK` under the identical per-quartet screen (−104,
  −1,840), and its K agrees with `DirectK` to 2.4e-14. There is no
  dedup/ownership double-visit; #50 is not over-inclusive in the sense that
  would make it wrong. `link_k.rs` / `pairs.rs` were NOT touched.
* **Pruning power: NIL at this threshold and size.** The pre-registered
  "over-inclusive" flag (`LinK/DirectK ≥ 0.95 at C16`) FIRES: 0.99996. The
  cause is visible in the pair table — the (valid) density-pair bound keeps
  100% of shell pairs on the converged density at 1e-12 at BOTH sizes (the
  far-end density blocks of a 50-Bohr alkane are still above
  1e-12/qmax², consistent with the ~30 Bohr decay length noted in memory),
  and the significant-pair list only removes pairs the direct screen removes
  anyway. So the fixed LinK does essentially the DirectK work plus list
  bookkeeping. This is the pre-registered "no onset yet" case, not a
  construction bug: the pre-fix lists pruned 22-51% of density pairs only
  because their criterion was WRONG (Gaussian-decaying `Q(j,σ)` on a pair
  that appears in no quartet; K error 1.7e-3).
* **vs the production builder: LinK evaluates MORE.** `build_jk`'s pairwise
  density screen removes 7.6% (C8) / 11.8% (C16) of the quartets that the
  global-max|D| screen keeps; LinK, screening with global max|D| per quartet
  and relying on its lists for the density dependence, evaluates 8% / 13%
  more quartets than the default builder — and the SCF path additionally
  pays a separate `DirectJ` sweep. This is consistent with the first
  post-fix SCF datum (alkane_8/def2-SVP: 196 s LinK vs 99 s direct) without
  any timing here. It is a screening-quality gap (per-quartet screen uses
  global max|D| instead of the four exchange density blocks), NOT a #50
  defect, and is reported for the coordinator rather than fixed blind.
* Artifact check: the LinK−DirectK deficit is not an exact repeat (−104 vs
  −1,840; ratios 0.99999 vs 0.99996), so the equality is not arithmetic.

## What this does NOT say

No cost, exponent, crossover, or COSX/LinK ratio for the fixed kernel — those
are the deferred timing pass (see the prereg for the protocol). Nothing here
is a K-build timing.
