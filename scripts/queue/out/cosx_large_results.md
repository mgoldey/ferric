# Results: COSX at large system size — the ceiling map

Pre-registration: `scripts/queue/out/cosx_large_prereg.md`, committed 8f7a0bf0
on 2026-09-09 **before any number below**. Branch `meas/cosx-large-system`,
worktree `.claude/worktrees/cosx`, off `origin/main` 433db737.

## Protocol as run

One thread throughout (`OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1`, plus an
explicit 1-thread rayon pool around every timed segment), `FERRIC_MEM_BUDGET_GB=2`,
`scripts/ferric-limited --max=6G --high=5G` in the foreground under a bounded
`timeout`. `/proc/pressure/memory` `full avg10` printed before and after every
timed segment; every kept row is 0.00/0.00. cpu_s == wall_s to <= 0.05% on every
kept row. Peak RSS is `VmHWM` for the whole process (SCF included where the cell
ran one). COSX is production configuration at every rung and no knob is tuned
per rung: md3c1e backend, sparse half transforms, density-driven screen 1e-7,
grid (50,110), overlap fit ON.

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

**DF-K has NO preflight.** `ferric_integrals::three_index_source` is a bare
`if needed <= budget_bytes { in-core } else { spill }` (three_index_source.rs:190).
Above budget there is no error and no disk-space check: it calls
`tempfile::tempfile()` and streams the tensor to `/tmp`, which on this box is
the same 95%-full root partition with 207 GB free.

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
that nothing in the library would TELL you this: it would begin writing and
fail on ENOSPC partway, having filled a 95%-full shared partition. None of the
disk rows above were executed; per the pre-registration's binding safety rule,
the sizes and the mechanism are the finding and filling the disk to prove it is
not.

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

