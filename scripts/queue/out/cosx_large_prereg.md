# Pre-registration: COSX at large system size — the ceiling map

Written and committed 2026-09-09 on branch `meas/cosx-large-system`
(worktree `.claude/worktrees/cosx`) **before any timing number was taken**.
Everything below is a prediction, not a result.

## What this lane is for

Every COSX number in this repo so far is <= 20 heavy atoms (`cosx_scaling_results.md`
tops out at alkane_20 / def2-SVP, nbf 490; `cosx_qzvp_error_results.md` is butane
at def2-QZVP, nbf 528). The regime that was used to justify building COSX at all —
large systems at high angular momentum, where DF-K's dressed 3-index tensor does
not fit in memory — has never been exercised. The deliverable is a CEILING MAP:
climb rungs until each builder fails, and record where and why.

## Sizing measured before pre-registration (not a timing, so not gated)

Cheap probe: `cosx_l_axis` with `COSX_L_SKIP_A=1 COSX_L_SKIP_SCF=1`, which only
constructs `PreparedBasis` + the jkfit aux basis and prints sizes. Real numbers,
replacing the estimates the task brief carried:

| system | natoms | basis | nsh | nbf | L_max | jkfit naux | dressed 3-index naux*nbf^2*8 |
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

Note the brief's estimates were HIGH for QZVP nbf (it guessed 7510 for
alkane_48/QZVP; the real number is 5676) and HIGH for the tensor at QZVP (it assumed naux ~ 2.5x nbf; def2-universal-jkfit's naux does NOT
grow with the orbital basis, only with the atoms — naux is 2256/3588/5364 for
C20/C32/C48 at EVERY orbital basis). The tensor therefore grows as nbf^2 at fixed
system, not as nbf^3. This correction is recorded here because it changes which
rungs are even worth attempting.

## Machine state at pre-registration (a constraint, not a result)

`free -h`: 23 GB total, ~5.2 GB available. A standing `ollama llama-server`
holds 6.25 GB resident (idle, 0.1% CPU) and a `chroma-mcp` python holds 1.6 GB.
Those are other people's processes and will not be killed. Disk: 207 GB free of
3.6 TB (95% used), and `/tmp` — where `tempfile::tempfile()` puts DF spill files —
is on that SAME partition.

## Predictions

### P1 — which rung fails first, per builder

* **DF-K fails first, and it fails at the very first rung of this lane.**
  alkane_20/def2-SVP already needs a 4.33 GB dressed tensor against a resolved
  budget of ~4.1-4.7 GB and ~5.2 GB of actual free RAM. Predicted mode:
  `ThreeIndexSource` takes its **disk-spill** branch (there is no refusal — I
  read `three_index_source.rs`; the only budget response is to spill), writes a
  multi-GB temp file into `/tmp` on the 95%-full root partition, and the run
  either thrashes (PSI full > 0) or is killed by the cgroup. I predict NO clean
  preflight error from DF-K at any rung of this lane.
* **Direct K (`build_jk`, exact four-centre) fails on TIME, not memory.** It
  holds no large tensor. Extrapolating the butane/def2-QZVP point (372 s at
  nbf 528, one thread) at roughly N^2.5-3 in nbf, I predict direct K exceeds the
  30-minute per-build stop rule somewhere around nbf 1000-1400, i.e. at
  alkane_32/def2-SVP (nbf 778) it is marginal and at alkane_20/def2-TZVP
  (nbf 872) or alkane_48/def2-SVP (nbf 1162) it is out.
* **LinK fails on TIME too, but ~1.2-3x later than direct** (its measured
  advantage over direct is 1.21x at butane/QZVP and larger at SVP where the
  scaling series in `cosx_scaling_results.md` gives a tail exponent 1.54).
  Predicted ceiling: alkane_48/def2-SVP or alkane_32/def2-TZVP.
* **COSX runs furthest.** Its resident cost is dominated by the block half
  transforms and the (nbf,nbf) K itself, not by any N^2*naux tensor: at
  nbf 2076 (alkane_48/TZVP) K alone is 34 MB, and the sparse half transforms
  hold |A|/nbf * nbf per block. I predict COSX's ceiling on this box is set by
  WALL TIME (the A-build tail exponent 1.54, times a growing grid — npts grows
  linearly in atoms) rather than by RSS, and that it clears at least
  alkane_32/def2-SVP and alkane_20/def2-TZVP.

Single most likely first failure overall: **DF-K at alkane_20/def2-SVP, by disk
spill into a 95%-full partition.** I will NOT let that spill run — see the
safety rule below.

### P2 — what would mean the memory-wall argument for COSX is WRONG

The COSX case rests on "COSX still builds a K where DF-K cannot". Any of these
refutes or badly weakens it, and I commit in advance to reporting them as such:

* **R1 (the direct refutation).** COSX hits its own hard wall at the SAME rung
  DF-K does, or earlier. If COSX cannot produce a K at alkane_20/def2-SVP or
  alkane_32/def2-SVP either — for whatever mechanism, RSS or 30-minute wall —
  then "DF-K runs out of memory here" is not a COSX advantage, it is a statement
  about the box, and the whitepaper may not use it.
* **R2 (the ratio inversion).** COSX's wall/K grows FASTER in system size than
  LinK's, so that the COSX/LinK ratio — 0.301x at butane/QZVP, but 5.6x-7.9x
  across the SVP series in `cosx_scaling_results.md` — moves further AGAINST
  COSX as N grows at fixed basis. If COSX/LinK at alkane_32/def2-SVP is worse
  than the 7.88x already on record at alkane_20/def2-SVP, then the large-system
  claim is about L only, and must be worded as such.
* **R3 (the peak-RSS surprise).** COSX's own peak RSS grows super-linearly and
  approaches the 6 GB cgroup at a rung where DF-K's tensor is still only
  moderately over budget. That would mean COSX buys perhaps one rung, not a
  regime, and the "does not fit in memory" framing overstates it.
* **R4 (the honest-tie).** DF-K's spill path actually WORKS at 4-17 GB tensors
  (streaming from disk at acceptable wall time) rather than thrashing. Then the
  wall is a performance wall, not a feasibility wall, and "DF-K cannot run" is
  false — the correct claim would be "DF-K becomes IO-bound".

### P3 — quantitative predictions (so a match is not post-hoc)

* COSX/LinK at alkane_32/def2-SVP: I predict **8-11x** (i.e. worse than the
  7.88x at alkane_20, continuing the monotone rise 7.57 -> 5.56 -> 5.84 -> 6.88
  -> 7.88; the block GEMMs have tail exponent 4.55 and are taking over).
* COSX kept-pair fraction (`kept_dd`) at alkane_32/def2-SVP: **0.08-0.12**
  (continuing 0.307 -> 0.206 -> 0.146 at C12/C16/C20).
* COSX |A|/nbf at alkane_48/def2-SVP: **< 0.25** (the sparsity that makes the
  half transforms affordable must keep falling, or R3 fires).
* COSX total wall at alkane_32/def2-SVP: **400-700 s** (168.5 s at C20/SVP,
  total-wall tail exponent 2.16, (98/62)^2.16 = 2.7).

## Protocol

* One thread everywhere: `OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1`, and each
  timed segment additionally runs in an explicit 1-thread rayon pool
  (`cosx_full_k.rs::timed`). cpu_s ~ wall_s asserted on every kept row.
* `/proc/pressure/memory` `full avg10` printed before AND after every timed
  segment. Any row with a nonzero reading is DISCARDED and retaken, or flagged.
* `scripts/ferric-limited --max=6G --high=5G --` in the FOREGROUND under a
  bounded `timeout` (<= 30 min per window). Long chains are split across
  windows via the harness's density save/load, never by raising the timeout.
* Peak RSS per cell read from `/proc/self/status` `VmHWM` (harness addition,
  see below), cross-checked against `/usr/bin/time -v` where available.
* COSX configuration fixed as production: md3c1e backend, sparse half
  transforms, density-driven screen at the default 1e-7, grid (50,110),
  overlap fit ON. No knob is tuned per rung.
* Densities: whatever is cheapest that is HONEST. Where a converged SCF does not
  fit a window, the density's provenance and convergence level are printed with
  the row and it is labelled. All builders in a rung contract the SAME density,
  so provenance cancels out of every ratio.

## Safety rules (self-imposed, binding)

* **DF-K is never run when its dressed tensor exceeds 2 GB.** At every rung of
  this lane the tensor is >= 4.33 GB, so the honest expectation is that DF-K is
  never actually built here and its column is the PREDICTED size plus the
  mechanism, read from the source. Proving the disk failure by filling a
  95%-full partition is not a deliverable; the mechanism is.
* `df -h /home/matt` before and after every rung. **ABORT the whole lane if free
  space drops below 100 GB.** Currently 207 GB.
* Stop climbing when a single K build exceeds ~30 min or peak RSS approaches
  6 GB. An honest ceiling is a complete deliverable; extrapolation past what ran
  is not permitted in the report.

## Harness changes needed (made before measuring, listed so they are not silent)

1. `VmHWM` peak-RSS print at the end of each `cosx_full_k.rs` cell.
2. A DF-K arm in `cosx_full_k.rs` that computes the dressed tensor size and,
   above `COSX_FK_DFK_MAX_GB` (default 2.0), REFUSES to build and prints the
   size — the l_axis harness already has exactly this guard; this copies it so
   both builders can be exercised on one density in one process.
3. A sizing-only mode so nbf/naux/tensor-GB can be read without any build.

Any further change is a deviation and will be reported as one.
