# ferric vs PySCF matched timing (issue #228 core slice)

> Unattended quiet-window run of the remaining #228 plan: see `OVERNIGHT.md` (`overnight_228.py`, `analyze_228.py`).

Files: `pyscf_bench.py` (PySCF side, one fresh process per call), `scatter_driver.py` (F P F P
interleaved driver, load gate), `toml_ri/` (ferric inputs, see "Deviation"), `bench_results/*.jsonl`
(every run, including invalid and discarded ones).  Ferric side reuses `omega_sweep.py`'s box probes.

## Written BEFORE the timing session (the pass condition and hypotheses)

Disclosure: the energy anchors below were run first and printed wall times as a side effect (ferric
diamond_prim ~40 s and dryice ~92 s at 1 thread; PySCF 1e-12 builds of ~1-2.5 min).  The committed
PySCF 1e-11 anchors (`bench_results/anchor_pyscf.jsonl`, commit 4b7a028f: totals 46.4 s / 138.9 s,
dryice hcore 75.6 s) predate the hypotheses, so the N=1 total ratios were KNOWN IN ADVANCE; only the
N=6 rows were unseen.  The hypotheses below are therefore NOT blind for N=1.

### Matching (what is held equal)
cc-pVDZ (spherical, ferric's bundled BSE JSON), aux cc-pvdz-ri (ferric's bundled JSON, same file to
PySCF), exxdiv=ewald, hcore guess, eigendecomposed aux metric with absolute cut 1e-10 (PySCF
`j2c_eig_always=True`, `linear_dep_threshold=1e-10`), orbital lindep 1e-6, same cell (xyz + lattice
files), 1 process per run, cderi on a private temp dir deleted at exit.  Precision: PySCF
`cell.precision` = the loosest value in {1e-8,1e-10,1e-11,1e-12} whose energy is within the gate of
PySCF's own 1e-12 value; ferric has no precision knob (RS-GDF 1e-13, hcore 1e-14 are fixed).

### Energy anchor (untimed, before any timing)
For each cell: ferric once, PySCF at precision 1e-12 once, same aux, same dropped-aux count.
Gate derived: the fitting error is identical in both codes (same aux), so the residual is the
PySCF lattice-sum precision plus SCF convergence (1e-10 density rms on both); the prior campaign pins
RS-GDF vs PySCF GDF at 6e-13..2.4e-12 (Gamma) and FINDINGS expects <~1e-8: bar **|dE| <= 1e-8 Ha/cell**.
Mutation check of the gate: PySCF at precision 1e-8 must FAIL it (else the gate cannot tell
precisions apart).  Timed PySCF precision = loosest value passing the gate against its own 1e-12.
Drop counts must be equal or the energies are not comparable (STOP).

### Pass condition for a reportable comparison cell
1. Anchor gate passed with equal drop counts, mutation check failed as it should.
2. >= 3 (target 5) valid runs per (cell, code, thread count); valid = load gate (1-min loadavg < 1.5
   before and after for N=1; foreign-CPU fraction < 2% for N>1); invalid runs are kept in the jsonl but
   excluded and re-run.
3. Every timed run's energy within the gate of the anchor, and converged.
4. Report median, min-max and IQR; a cell whose IQR/median > 10% is labelled noisy.
5. Only ratios of the same cell, same N, same stage definition; a list of every remaining mismatch
   goes next to every ratio.  Setup (hcore + DF build) and seconds/iteration are reported separately
   from the total, because iteration counts differ (different stopping rules).

### Hypotheses (physics H vs artifact A), stated before the timed numbers
- H1 (physics, from FINDINGS prediction): at N=1 ferric is slower than PySCF in the SR 3-centre build
  (libint recomputes general contractions per column, libcint once).  Counter-evidence already seen
  in the anchors (see disclosure), so H1 is at risk.
- A1: "ferric is faster than PySCF at 1 thread" would be SUSPICIOUS (FINDINGS surprise 6).  If seen,
  audit before believing it, in this order: PySCF threads really 1 (cpu_df_build/wall ~ 1), PySCF
  precision not needlessly tight (gate above), no cached cderi (fresh temp dir, file size recorded),
  PySCF stage that ferric does not do (cderi HDF5 write/read, python import, FFT-mesh hcore), ferric
  stage that PySCF does differently (SR column rotation is ON by default in ferric: the
  `_unrot` toml is the old baseline and is ~3.6x slower on diamond_prim), iteration counts.
  If a gap is mostly "PySCF hcore is slow", that is an algorithm difference (PySCF nuclear attraction
  on the FFT mesh at precision 1e-11), not "ferric's JK is faster": split by stage.  [See "Corrections to
  the first session" at the end: PySCF's hcore is range-separated, not FFT-mesh, and runs twice.]
- H2 (FINDINGS surprise 1, stated as a prediction that may be stale): ferric RAYON_NUM_THREADS=6 with
  OPENBLAS=1 should help only the parallel stages (pair FT chunk kernel now has rayon; SR walk,
  hcore and GEMMs with OPENBLAS=1 are serial).  Artifact signature if wrong: wall unchanged
  although the pair-FT stage shrank, or a speedup bigger than the parallel stages' share allows.
- A2: if ferric iteration count differs >2x from PySCF (FINDINGS surprise 5) compare s/iter only.
- A3 (FINDINGS surprise 3): peak RSS below 8*naux*nao^2 would be a measurement artifact.

## Deviation from the issue's named inputs: aux basis
`toml/*.toml` use `def2-universal-jkfit`.  On diamond_prim that aux gives DIFFERENT drop counts:
PySCF drops 21 of 150 (metric eigenvalues 1e-17..3e-11 below the 1e-10 cut, 3.06e-10 just above),
ferric drops 9 (naux kept 141).  Per the rule, energies are not comparable (ferric/STO-3G jkfit
-74.0024787783 vs PySCF -74.0023846489 quoted from FINDINGS (not re-measured here) = 9.4e-5, the size FINDINGS says a drop-count mismatch produces).  So the
timed cells use `cc-pvdz-ri` (`toml_ri/`, same file as `toml/` except `auxbasis`), where both codes
drop 0.  jkfit rows are therefore NOT done (open item: why ferric's jkfit metric spectrum differs
in the 1e-15..1e-10 band).  Raw PySCF jkfit anchor: `bench_results/anchor_pyscf_jkfit_diamond_prim.jsonl`.

# RESULTS (written after the timing session; head SHA of the code timed: a50e9964 + this directory)

Box: 12 logical / 6 physical cores, ambient foreign CPU 4-6% (~0.6 core) the whole session.  Binary:
`target/release/ferric` built from a50e9964 (`cargo build --release -p ferric-cli`).  PySCF 2.13.1.
All runs in `bench_results/*.jsonl` (`valid` field; invalid ones are kept, not used).

## Energy anchors (untimed; Ha per cell; cc-pVDZ / cc-pvdz-ri, 0 aux dropped in BOTH codes)
| cell | ferric | PySCF p=1e-12 | dE (bar 1e-8) | PySCF 1e-8 vs its 1e-12 (must FAIL) | 1e-10 | 1e-11 (timed) |
|---|---|---|---|---|---|---|
| diamond_prim | -74.9757090071 | -74.97570900706793 | ~3e-11 (ferric prints 10 dp) PASS | 1.8e-6 FAIL (gate discriminates) | 1.6e-8 FAIL | 3.5e-10 PASS |
| dryice | -750.8211184382 | -750.821118438013 | 1.9e-10 PASS (true value in ~[1.4e-10,2.4e-10]: ferric prints 10 dp) | 2.2e-6 FAIL | 7.3e-8 FAIL | 1.3e-10 PASS |
Matched PySCF precision = 1e-11 for both cells (the loosest tested value inside the gate; 1e-9 was
not tested).  Also pre-checked at STO-3G/cc-pvdz-ri: diamond_prim
dE 3.3e-10, dryice 1.1e-9.  jkfit: drop counts DIFFER (see Deviation) so no jkfit rows exist.
Every timed run's energy is within the gate of the anchor (ferric rep-to-rep spread 0; PySCF 1e-13).

## Timing, seconds, valid runs only (median [min-max] IQR; load = 1-min loadavg range incl. own load)
Total = harness wall around the whole process (PySCF includes ~1.5 s python/pyscf import).
Setup = hcore + DF build. SCF = rest of the iterations. Ferric: ferric defaults (SR column rotation
ON, gdf_omega 1.0), `OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=N`.  PySCF: precision 1e-11,
`OMP/OPENBLAS=N`.
| cell | N | code | n | total | setup | of which df-build / hcore | SCF | iters | s/iter |
|---|---|---|---|---|---|---|---|---|---|
| diamond_prim | 1 | ferric | 5 | 39.26 [39.14-39.98] IQR 0.37 | 39.20 | 35.2 / 4.0 | 0.018 | 10 | 0.002 |
| diamond_prim | 1 | PySCF | 4 | 46.98 [46.70-47.27] IQR 0.27 | 46.18 | 39.5 / 6.7 | 0.279 | 7 | 0.018 |
| dryice | 1 | ferric | 5 | 92.47 [92.26-92.75] IQR 0.30 | 85.62 | 77.0 / 8.6 | 6.85 | 20 | 0.343 |
| dryice | 1 | PySCF | 4 | 138.38 [136.99-140.42] IQR 0.90 | 118.20 | 45.1 / 73.2 | 19.61 | 13 | 0.744 |
| diamond_prim | 6 | ferric | 3 | 7.60 [7.47-7.82] IQR 0.17 | 7.56 | 6.7 / 0.84 | 0.018 | 10 | 0.002 |
| diamond_prim | 6 | PySCF | 3 | 11.43 [11.24-11.48] IQR 0.12 | 10.67 | 9.1 / 1.6 | 0.207 | 7 | 0.012 |
| dryice | 6 | ferric | 3 | 19.89 [19.80-20.01] IQR 0.10 | 17.97 | 16.2 / 1.8 | 1.89 | 20 | 0.094 |
| dryice | 6 | PySCF | 3 | 31.89 [31.64-36.23] IQR 2.30 | 26.19 | 10.7 / 15.4 | 5.13 | 13 | 0.195 |
Extra ferric-only rows (diamond_prim, 3 reps): `_unrot` baseline (sr_column_rotation=false, the
setting in toml/) N=1 140.36 [140.33-141.32] s, N=6 26.82 [26.45-27.09] s.  `gdf_omega = "auto"`
(ferric's own cost-model choice; NOT a sweep for the best omega) N=1: diamond_prim picks 1.0
(40.31 [40.23-40.44] s, identical to default); dryice picks 0.5: 53.34 [53.20-53.42] s (n=3, 1 extra
invalid) vs 92.47 s at the default 1.0, same energy.  PySCF auto omegas: 0.417 (diamond_prim), 0.330 (dryice).
Peak RSS: PySCF 0.17 GB / 2.95 GB (N=1; cderi file 0.37 MB / 76 MB); ferric dryice 0.41 GB came from one
interactive `/usr/bin/time -f %M` run whose raw output was NOT committed (unrecorded; treat as
anecdotal); ferric RSS was not recorded by the driver for any row.
Discarded runs (13 of 31 at N=1): the loadavg gate has only 0.5 slack above our own ~1.0 load.
12 of the 13 have walls within 1.3% of the valid median (their loadavg was inflated by the previous
run's tail); one (ferric diamond_prim rep 2) was +8% with 2.2 cores of foreign CPU (a real
perturbation); one (PySCF dryice rep 4) saw a foreign load spike to 5.9 yet its wall moved -0.8%.
PySCF dryice rep 2 AND PySCF diamond_prim rep 1 (N=1) each exhausted 3 attempts with no valid run -> n=4 not 5.  The diamond_prim rep-1 attempts had ~0.75 cores of foreign CPU against ~0.45 ambient, so that discard is partly a real perturbation, not only loadavg tail.  The dryice N=6 PySCF max 36.2 s is
one run 14% above the others (ambient, not flagged by the gate); median unaffected.

## Statements the data supports (same cell, same N, same stage definition)
- Total wall: ferric/PySCF = 0.84 (diamond_prim N=1), 0.67 (dryice N=1), 0.66 (diamond_prim N=6), 0.62
  (dryice N=6): ferric is FASTER at these settings.  Do not read this as "ferric's JK is faster": the
  stage split says otherwise for dryice.
- DF build only (the like-for-like stage): diamond_prim ferric 35.2 s vs PySCF 39.5 s (N=1), 6.7 vs 9.1
  (N=6); dryice ferric 77.0 vs PySCF 45.1 (N=1), 16.2 vs 10.7 (N=6): PySCF is 1.7x / 1.5x FASTER at
  the DF build on dryice.  The dryice total advantage is the hcore stage (ferric 8.6 s vs PySCF 73.2 s
  N=1, different algorithms: PySCF nuclear attraction on the FFT mesh at precision 1e-11).  [CORRECTED
  at the end of this file: two evaluations of 36.6 s, range-separated in both codes.]
- s/iteration: ferric faster, but iteration counts differ (stopping rules, below).  [The two s/iter
  columns are defined differently; see "Corrections to the first session".]
- Parallel scaling N=1 -> 6: ferric 5.2x (diamond_prim), 4.6x (dryice); PySCF 4.1x, 4.3x.
- ferric default vs its own unrotated baseline: 3.6x on diamond_prim at N=1 (3.5x at N=6: 26.82 s vs 7.60 s; the 5.2x is the unrotated build's own N=1->6 scaling).  Without SR
  column rotation ferric diamond_prim (140 s) is 3.0x SLOWER than PySCF (47 s).

## Mismatches that remain (read before quoting any ratio)
1. Precision knob: PySCF cell.precision 1e-11 (gated); ferric fixed (RS-GDF 1e-13, hcore 1e-14), tighter.
2. Stopping rules: ferric 10/20 iterations, PySCF 7/13 (same hcore guess, DIIS defaults of each
   code); PySCF conv_tol 1e-10 + grad 1e-6 vs ferric density rms 1e-10.  Iteration counts within 2x.
3. omega: ferric default fixed 1.0 Bohr^-1; PySCF auto 0.417 / 0.330 (ferric `auto`: 1.0 / 0.5).
4. hcore [CORRECTED at the end of this file]: different algorithms (see above); the hcore stage is 62% of PySCF's dryice setup (73.2 of 118.20 s at N=1; 59% at N=6).
5. Aux: cc-pvdz-ri instead of the toml's jkfit (deviation above).  Same JSON, 0 dropped both.
6. Exchange: ferric K is the RS-GDF dense-D loop; PySCF uses the occupied-orbital path with the cderi
   read from disk (HDF5); PySCF DF build also writes that file (negligible: SR rest 0.003 s on
   diamond_prim, see `pyscf_sr3_stages_diamond_prim_1e-11.json`).
7. Process overhead: harness wall includes python/pyscf import (~1.5 s) for PySCF only.
8. Load gate: N=1 uses 1-min loadavg < 1.5 (tight because our own run contributes ~1); N=6 uses
   foreign-CPU fraction < 12% (ambient was 4-6%).  SEE THE DEVIATION SECTION BELOW: this N=6 gate is
   not the pre-registered one.
9a. Run order is fixed ferric-then-PySCF within each rep, so PySCF's load gate absorbs ferric's
    loadavg tail: PySCF was discarded 9 times at N=1, ferric 4.  Discarded PySCF runs were slightly
    slower than kept ones, so excluding them lowers the PySCF median by ~0.5-1% (conservative for the
    "ferric faster" statement).
9b. Ferric convergence is not checked by the driver (the CLI only warns, exit 0); "converged" is
    implied only by the energy matching the anchor.  PySCF `converged` is recorded.
9c. Binary provenance: `target/release/ferric --version` reports commit a50e9964502fa604c3b79b8bed88ce1e1d38fdd0,
    dirty false (checked after the session, same binary, not rebuilt); it was not recorded per run.
9d. PySCF `max_memory` 8000 MB.
9. Basis conversion: ferric's BSE JSON (general contractions kept as one PySCF block) vs ferric's
   column-per-shell; same functions.  cart=False.
10. PySCF drop count is read from the returned metric rank; ferric's from its own printout.

## Deviation: the N>1 validity gate was relaxed after pre-registration
Pre-registered (commit 4b7a028f): N>1 valid iff foreign CPU < 2% of the box.  Changed to < 12%
(scatter_driver.py, between 4b7a028f and f9f48ba6) because the ambient foreign load is 4-6% (measured
3.2-5.8% on all 15 N=6 runs), so NO N=6 run passes the pre-registered gate.  The first N=6 launch under
the 2% gate never started a run (it sat in the quiet-wait for ~13 min and I killed it; its empty output
file was deleted), so no N=6 run was taken under 0.02 and then dropped.  The N=6 rows are therefore
reported under the relaxed gate and should be read as "ambient-load" numbers.

## Verdicts on the pre-registered hypotheses
- H1 (ferric slower than PySCF in the SR 3-centre build at N=1): REFUTED on diamond_prim, ferric SR
  3-centre 34.0 s vs libcint 37.3 s, but only with SR column rotation on; unrotated it is 125.8 s, so
  H1 holds for the old path.  Untested on dryice (PySCF not stage-split there).  DF-only diamond_prim:
  35.2 vs 39.5 s = 0.89 (N=1).
- A1 (ferric faster at 1 thread is suspicious): triggered and audited, see item 6 below.
- H2 (only pair FT parallel): REFUTED.  A2 (iteration counts >2x apart): not triggered.  A3 (RSS below
  8*naux*nao^2): not testable from recorded data.

## Audit of the FINDINGS "surprises" list
1. RHF wall falls with RAYON_NUM_THREADS: HAPPENED (5.2x / 4.6x with OPENBLAS=1).  Not an artifact:
   process CPU/wall = 5.5 / 5.2 at N=6 and 1.00 at N=1; stage times shrink SR 3-centre 34.0 -> 6.4 s
   (diamond_prim), pair FT 37.0 -> 6.9, LR GEMM 24.5 -> 5.6, K 6.1 -> 1.2 (dryice).  So the FINDINGS
   "whole RHF build is SERIAL" reading is out of date for this commit; H2 (only pair FT parallel) is REFUTED.
2. Build time vs SR3 count 1:4:8: NOT TESTED (diamond_conv/prim222/dryice_112 not run).  Observed only
   that dryice is not SR3-dominated (SR 3-centre 14.2 s, LR pair FT 37.0 s, LR GEMM 24.5 s at N=1).
3. RSS: ferric dryice 0.41 GB (unrecorded one-off, see above) is consistent with 16*naux*nao^2 = 0.30 GB plus a ~0.11 GB baseline; the baseline is a post-hoc FINDINGS figure, so this is weak evidence only.
4. |dE| > 1e-7: no (3e-11, 1.9e-10).  E(prim222)/8 vs k222: NOT RUN.
5. Iteration counts differ >2x: no (1.4x, 1.5x).
6. ferric FASTER than PySCF at 1 thread: HAPPENED on the totals.  Audited: PySCF cpu/wall = 1.00 (one
   thread); precision gated and looser than ferric's; fresh cderi temp dir each run, deleted at exit (file size recorded only in the anchor runs, not the timed rows; df
   build is 95% libcint SR int3c on diamond_prim: 37.3 of 39.3 s); the gap on dryice is the hcore
   algorithm, the df-build is faster in PySCF there; and the ferric result relies on SR column
   rotation (default on).  Not found to be a PySCF mis-setup; it remains a statement about these
   settings only.

## Open items of #228 (NOT done here)
k-point meshes beyond diamond 2x2x2 (diamond 2x2x2 itself not run either: toml exists, no anchor/timing),
dryice_112, diamond_prim222, diamond_conv, PBE/KS rows (grids differ, anchor protocol in FINDINGS not
run), jkfit rows (blocked by the aux-drop mismatch, itself a lead: why ferric drops 9 of 150 and PySCF
21), the omega comparison default vs BEST ferric omega (only default and `auto` were timed), 5 reps
for the N=6 rows (3 done) and for PySCF N=1 (4 done), ferric RSS in the driver.

# ADDENDUM: second session (k-mesh, larger cells, PBE, omega, N=6 repeats), PRE-REGISTERED

Written and committed BEFORE any anchor or timing run of this session (no number below this heading
existed when it was written).  Same matching rules, anchors, validity gate and reporting rules as the
first session above unless stated here.  Harness changes (this commit): `pyscf_bench.py` gains
`--kmesh a b c` (KRHF) and puts PBE on pbc `BeckeGrids` (atom grids on image atoms, original Becke
partition, unpruned, `atom_grid` = `--grid`); before this the PBE path set `atom_grid` on the
default pbc UniformGrids, where it has no effect, so no PBE number from the old script is usable.  It
also records `hcore_calls` (PySCF's RHF kernel with `init_guess='hcore'` may evaluate `get_hcore`
more than once; `t_hcore_s` is the sum).  `scatter_driver.py` now holds BOTH heavy slots
(`/tmp/ferric-compile.lock`, `/tmp/ferric-compile.2.lock`) from before the quiet-wait until the run
exits, alternates the code order per rep (F P on even reps, P F on odd: the first session's fixed
F-then-P order let PySCF's load gate absorb ferric's loadavg tail, mismatch 9a), records each run's
own peak RSS (wait4 `ru_maxrss`), `ferric --version` and ferric `converged` (an unconverged run is
invalid).  New ferric inputs in `toml_ri/` (ferric defaults, SR column rotation ON, cc-pvdz-ri);
new cells `diamond_prim222` / `dryice_112` (`.xyz` + `.lattice`) copied from the generator output of
`make_geometries.py` (FINDINGS "Real-size benchmark plan"; not re-generated here).

## Rows, in priority order (stop when the ~3 h timed-wall budget or a busy box ends the session)
1. `diamond_prim_k222` RHF (2x2x2 Gamma-centred mesh): N=6 then N=1, 5 valid reps per (code, N).
2. `diamond_prim`, `dryice` Gamma RHF at N=6: 5 fresh valid reps per code (the first session had 3).
3. `dryice_112`, `diamond_prim222`, `diamond_conv` Gamma RHF at N=6, 5 reps; N=1 (3 reps) only for a
   cell whose projected wall (anchor walls x reps) still fits the remaining budget.
4. PBE (`dryice_pbe`, `diamond_prim_pbe`, Gamma, 75x302 unpruned in both codes) at N=6, 5 reps, only if
   the PBE energy gate below passes.
5. omega: ferric-only sweep at N=6, `gdf_omega` in {0.3, 0.5, 0.7, 1.0 (default)} on diamond_prim and
   dryice, 3 reps, omega order rotated inside each rep.  "Best omega" = lowest median total among
   values whose energy is within 1e-8 Ha/cell of the default-omega anchor.  Reported next to the
   default and next to PySCF's automatic omega rows (row 2), labelled as such.
jkfit rows: NOT run (aux-drop mismatch, first-session Deviation section; unchanged).

## Settings
- ferric: `OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=N`, release binary of this worktree's base commit,
  `ferric --version` recorded per run.  k-mesh stopping rule (ferric defaults): dE < 1e-12 Ha/cell and
  max orbital-gradient element < 1e-9.  Gamma: density rms 1e-10 (as before).
- PySCF 2.13.1: Gamma conv_tol 1e-10 / conv_tol_grad 1e-6 (as before); k-mesh KRHF conv_tol 1e-12 /
  conv_tol_grad 1e-9 (the closest PySCF knobs to ferric's k rule; the gradient norms are defined
  differently, a mismatch to list).  `OMP/OPENBLAS=N`, `max_memory` 8000 MB, fresh cderi temp dir.
- Precision rule unchanged: timed PySCF `cell.precision` = loosest of {1e-8, 1e-10, 1e-11, 1e-12} whose
  energy is within the gate of PySCF's own 1e-12 energy for THAT cell (k-mesh included).  For a cell
  whose 1e-12 anchor alone exceeds 15 min, the ladder is {1e-11, 1e-12} and the result is labelled.
- Anchors (untimed) run at N=6, one per (code, cell, precision); they print walls, which are disclosed
  in the results and are what the row-3 N=1 projection uses.

## Energy gates (derived, not tuned)
- RHF, every cell incl. k222: **|E_ferric - E_PySCF(1e-12)| <= 1e-8 Ha/cell**, same aux, equal drop count
  (Gamma: dropped; k-mesh: max over q classes in ferric vs max over q in PySCF's `aux_metric` list).
  Justification as before: same aux => same fitting error; FINDINGS pins ferric RS-GDF vs PySCF GDF at
  <= 7e-12 for k-meshes; the 1e-8 bar is PySCF lattice-sum precision + SCF convergence.  Mutation
  check: PySCF at 1e-8 must FAIL the gate on k222 and diamond_conv (cheap cells); on the two largest
  cells it is skipped if the 1e-12 anchor exceeds 15 min (stated as not done).
- Supercell identity anchor: ferric E(diamond_prim222)/8 - E(diamond_prim_k222) within **1e-9 Ha** (exact
  identity of the two Hamiltonians, FINDINGS Iteration 11(c)); PySCF the same within 1e-8 (its
  precision).  A miss beyond these is a STOP for the prim222 and k222 rows.
- PBE: the two codes integrate on DIFFERENT 75x302 point sets (ferric TA-M4 x Lebedev with SSF over
  image atoms; PySCF BeckeGrids), so 1e-8 is unreachable by construction.  Gate: with
  eps_c = |E_c(75x302) - E_c(150x974)| for each code c on the same cell (untimed, at the matched
  PySCF precision of that cell's RHF anchor), a PBE row is reportable iff
  (a) |E_f(75x302) - E_P(75x302)| <= eps_f + eps_P + 1e-8, AND
  (b) |E_f(150x974) - E_P(150x974)| < |E_f(75x302) - E_P(75x302)| (the difference shrinks on refinement).
  Artifact hypothesis: if the 75x302 difference is a non-grid defect (e.g. a functional or density
  mismatch), it does NOT shrink on refinement and (b) fails -> no PBE timing row, investigate.  If
  the 150x974 anchor of a cell is too expensive for the budget, that cell's PBE row is not run.

## Hypotheses (stated before any number of this session)
- H3 (k222, no strong prior): ferric/PySCF total at k222 lies in [0.5, 2] at both N.  Artifact
  signatures: ferric k222 total below its own Gamma diamond_prim total (a k-mesh build that costs
  nothing), or PySCF KRHF with a different drop count (STOP, not a timing row).
- H4 (FINDINGS surprise 2, size scaling): ferric DF-build time across diamond_prim : diamond_conv :
  diamond_prim222 follows the SR3 triplet count 1 : 4 : 8 within +-30%.  My expectation is that it does
  NOT (the LR GEMM grows as nao^2 naux nG, ~N^4 at fixed density), so H4 is predicted to FAIL with the
  larger cells above 8x / 4x.  Same for dryice : dryice_112 (SR part 1 : 2).
- H5: the dryice PySCF DF-build advantage (1.7x at N=1, 1.5x at N=6, first session) does not shrink on
  dryice_112 (ferric's LR part grows faster).  Refuted if ferric/PySCF DF-build on dryice_112 at N=6
  is below 1.5.
- H6 (N=6 repeats): the first session's N=6 medians move by < 5% with 5 fresh reps under alternated
  order.  A larger move means the 3-rep rows were order- or load-biased.
- H7 (omega): ferric's best admissible omega on dryice is below 1.0 (its `auto` choice 0.5 was 1.7x
  faster at N=1); on diamond_prim the best is 1.0 within noise.  Any omega whose energy moves by
  > 1e-8 from the default is inadmissible (the gate, not the clock, decides).
- A4 (artifact, PySCF hcore): if `hcore_calls` > 1, the first session's PySCF hcore column was a
  SUM over calls; totals are unaffected (PySCF really does that work by default) but any per-call
  "algorithm" statement must divide by the call count.

# RESULTS of the second session (written after the runs)

Box: 12 logical / 6 physical cores; ambient foreign CPU 4-9% of the box on every run (0.5-1.1 cores;
other agents and desktop).  Binary: `target/release/ferric` built once from 654087b4 (= this
branch's base; the session commits touch only this directory), `--version` recorded on every
ferric run: commit 654087b422f4, dirty false, libint 2.13.1.  PySCF 2.13.1.  Every timed run held
both heavy slots.  All runs (valid and invalid) are in `bench_results/anchor2*.jsonl` and
`bench_results/timing2_*.jsonl`.  Session wall: 02:58-06:10 (anchors ~40 min, timing ~2.5 h).

## Deviations from the pre-registration (read first)
1. **N=1 end-of-run gate.**  Under the pre-registered N=1 gate (loadavg < 1.5 before AND after), the
   k222 N=1 row produced 1 valid ferric run (1 attempt) and 0 valid PySCF runs in 5 attempts
   (`timing2_k222_n1.jsonl`, kept): our own run (1.0) plus the ambient 0.6-0.8 cores puts the
   end-of-run loadavg at 1.5-2.7 for any run longer than ~60 s, while the PySCF walls were 89.1-89.6 s
   (0.6% spread) and the foreign CPU 5-7%.  I stopped that row and re-ran it with `N1_GATE=foreign`
   (start gate loadavg < 1.5 kept; end gate = foreign-CPU fraction < 0.12, the N>1 rule)
   (`timing2_k222_n1_fgate.jsonl`, 5 + 5 valid).  Only 1 of those 10 runs would also pass the old end
   gate.  The gate change does not move the numbers: the 5 PySCF runs rejected by the old gate had walls
   89.14-89.58 s (median 89.37) vs 89.44 under the new gate; ferric's one old-gate run 54.28 s vs 54.36.  The N=1 row is therefore an "ambient-load" number like the N=6 rows.
2. **PBE rows NOT timed.**  Gate part (b) could not be evaluated: ferric's periodic grid implements
   Lebedev orders up to 302 only (`invalid n_angular: 974`), so there is no ferric 150x974 run; and the
   PySCF PBE energies were IDENTICAL at 75x302 and 150x974 (diamond_prim -75.04944411759554 vs
   ...534; dryice -753.6935121493115 vs ...111).  That non-movement is the artifact signature the gate
   was built to catch: in PySCF 2.13.1 `mf.density_fit()` (df_jk.density_fit) calls `mf.copy().reset()`,
   and pbc/dft/rks.py `_patch_df_beckegrids` builds a fresh DEFAULT BeckeGrids (`atom_grid` {},
   NWChem pruning).  So every PySCF PBE number from `pyscf_bench.py` up to this commit, including the
   first session's script, ran on default pruned BeckeGrids (not UniformGrids, as the pre-registration
   above wrongly says), not on 75x302 unpruned.  The `grid` field in `anchor2_pbe.jsonl` is the
   REQUESTED grid, not the one used (records labelled `grid_note`) (diagnosed on diamond_prim/STO-3G: 26425 points for
   both requested grids; after the fix 43141 at 75x302 and 287667 at 150x974).  The fix (grid set again
   after `density_fit`, plus an assert) is in this commit; the PBE anchors were NOT re-run with it.
   Recorded only (untimed, NOT comparable): ferric 75x302 vs PySCF default grid differs by 1.5e-4
   (diamond_prim) and 1.1e-5 Ha/cell (dryice).
3. Order: the omega sweep ran before the (cancelled) PBE timing slot, not after it.
4. Harness edits after the pre-registration commit, all mechanical: `--pyscf-grid` passthrough,
   `N1_GATE`, k-point SCF stage accounting (`k SCF` counted as SCF; the k-mesh jsonl records carry
   STALE `t_scf_s` = 0, `t_setup_s` = total and `s_per_iter` = 0 from before the fix, and
   `summarize.py` recomputes all three from the stored stages), the grid fix of item 2, and
   `n_grid_points` copied into each driver record.  The omega records all carry `rep` = 0 (each
   was a 1-rep driver call); the rep is the line order: lines 1-8 rep 0, 9-16 rep 1, 17-24 rep 2.

## Energy anchors (untimed, N=6; walls disclosed: ferric k222 13 s, conv 34 s, prim222 94 s,
## dryice_112 315 s; PySCF 1e-12 runs 32 / 58 / 216 / 174 s)
cc-pVDZ / cc-pvdz-ri, 0 aux dropped in both codes for every cell (k222: every q class).  Ha per cell.
| cell | ferric | PySCF 1e-12 | dE (bar 1e-8) | PySCF 1e-11 vs own 1e-12 | 1e-10 | 1e-8 (must FAIL) |
|---|---|---|---|---|---|---|
| diamond_prim_k222 | -75.6961792415 | -75.69617924177923 | 2.8e-10 PASS | 7.5e-10 PASS | 3.7e-8 FAIL | 2.6e-6 FAIL |
| diamond_conv | -302.8758854144 | -302.87588540931 | 5.1e-9 PASS | 1.3e-9 PASS | not run | 2.6e-5 FAIL |
| diamond_prim222 | -605.5694339323 | -605.5694339369 | 4.6e-9 PASS | 9.2e-9 PASS (near the bar) | not run | 2.5e-5 FAIL |
| dryice_112 | -1501.4642014457 | -1501.4642014448 | 9.3e-10 PASS | 1.2e-9 PASS | not run | 7.1e-5 FAIL |
Timed PySCF precision = 1e-11 for every cell (1e-10 tested only on k222, where it fails; for the
Gamma cells 1e-10 was not tested, so 1e-11 is the loosest TESTED passing value, not proven loosest).
Supercell identity: ferric E(prim222)/8 - E(k222) = -75.69617924154 - (-75.6961792415) = |.| < 1e-10
(10 printed decimals) PASS (bar 1e-9); PySCF at 1e-12: E(prim222)/8 - E(k222) = 3.3e-10 PASS (bar 1e-8).
Every timed run: converged, energy spread 0 (ferric) / <= 2.7e-12 (PySCF), within the gate of its anchor.

## Timing, seconds, valid runs only (median [min-max] IQR); n = 5 per row, 0 discarded unless noted
Total = harness wall around the whole process.  Setup: ferric = every stage except SCF (k: `k hcore` +
`k J/K build`); PySCF = df build + hcore (sum over its 2 hcore calls, see A4).  RSS = the run's own peak.
| cell | N | code | total | setup | of which df-build / hcore | SCF | iters | s/iter (*) | RSS GiB |
|---|---|---|---|---|---|---|---|---|---|
| diamond_prim_k222 | 1 | ferric | 54.36 [54.23-54.71] IQR 0.42 | 53.16 | 47.35 / 5.85 | 1.13 | 10 | 0.113 | 0.22 |
| diamond_prim_k222 | 1 | PySCF | 89.44 [88.81-89.71] IQR 0.53 | 85.06 | 78.04 / 7.06 | 3.89 | 10 | 0.313 | 0.26 |
| diamond_prim_k222 | 6 | ferric | 13.03 [12.20-13.18] IQR 0.60 | 12.57 | 10.81 / 1.80 | 0.47 | 10 | 0.046 | 0.24 |
| diamond_prim_k222 | 6 | PySCF | 23.58 [23.40-23.89] IQR 0.28 | 19.63 | 17.84 / 1.76 | 3.41 | 10 | 0.280 | 0.38 |
| diamond_prim | 6 | ferric | 7.48 [7.41-7.82] IQR 0.20 | 7.44 | 6.59 / 0.81 | 0.020 | 10 | 0.002 | 0.10 |
| diamond_prim | 6 | PySCF | 11.65 [11.35-11.94] IQR 0.27 | 10.89 | 9.23 / 1.67 | 0.22 | 7 | 0.014 | 0.21 |
| dryice | 6 | ferric | 20.41 [20.25-21.19] IQR 0.18 | 18.44 | 16.54 / 1.89 | 1.95 | 20 | 0.097 | 0.39 |
| dryice | 6 | PySCF | 35.35 [34.82-36.01] IQR 0.52 | 29.63 | 8.31 / 21.32 | 5.04 | 13 | 0.196 | 2.79 |
| diamond_conv | 6 | ferric | 33.45 [32.66-34.20] IQR 0.70 | 33.05 | 28.29 / 4.40 | 0.39 | 11 | 0.036 | 0.16 |
| diamond_conv | 6 | PySCF | 42.98 [42.55-43.72] IQR 0.37 | 41.04 | 32.09 / 8.90 | 1.33 | 7 | 0.100 | 0.54 |
| diamond_prim222 | 6 | ferric | 92.69 [91.67-93.40] IQR 1.20 | 89.65 | 78.73 / 10.98 | 2.96 | 11 | 0.269 | 0.87 |
| diamond_prim222 | 6 | PySCF | 171.44 [169.70-171.78] IQR 1.57 | 157.57 | 91.80 / 65.25 | 12.22 | 7 | 0.528 | 5.17 |
| dryice_112 | 6 | ferric | 314.12 [314.00-315.32] IQR 1.27 | 290.98 | 285.50 / 5.52 | 23.14 | 21 | 1.102 | 2.49 |
| dryice_112 | 6 | PySCF | 142.94 [141.77-147.20] IQR 2.82 | 123.43 | 37.99 / 85.94 | 18.79 | 15 | 1.087 | 6.04 |
(*) DIFFERENT DEFINITIONS: ferric = SCF stage / iterations; PySCF = median gap between iteration
callbacks (excludes the pre-loop work inside the kernel).  Matched (SCF / iterations) PySCF values:
k222 0.389 (N=1) / 0.341 (N=6), diamond_prim 0.031, dryice 0.388, diamond_conv 0.189, prim222 1.746,
dryice_112 1.253.  On the matched definition ferric is faster per iteration on every row.
No row is noisy (every IQR/median < 3%).  ferric CPU/wall at N=6: 4.65 (k222), 5.2-5.8 (Gamma).
ferric stage split at N=6 (SR 3-centre / LR pair FT + aux FT/GEMM): diamond_prim 6.30 / 0.27,
diamond_conv 24.25 / 3.83, prim222 49.43 / 26.04, dryice 2.76 / 12.56, dryice_112 6.23 / 266.17.
N=1 rows for diamond_conv / prim222 / dryice_112: NOT run (budget; projected ~4 h with the N=1 gate).

### omega (ferric only, N=6, 3 reps each, `timing2_omega_n6.jsonl`, 0 discarded)
| cell | 0.3 | 0.5 | 0.7 | 1.0 (default) | best admissible | max |dE| vs default |
|---|---|---|---|---|---|---|
| diamond_prim | 27.24 | 12.78 | 9.45 | 7.91 | 1.0 | 1.5e-9 (0.3) |
| dryice | 16.44 | 11.69 | 13.10 | 20.30 | 0.5 (1.74x faster than default) | < 1e-10 |
PySCF's automatic omega (rows above, labelled): 0.417 (diamond_prim) / 0.330 (dryice).  At N=6 on
dryice: ferric best omega 11.69 s (this sweep) vs ferric default 20.41 s vs PySCF auto-omega 35.35 s
(different sub-sessions; same box, same gate).

## Statements the data supports (same cell, same N, same stage definition; mismatches below)
- Total wall ferric/PySCF: k222 0.61 (N=1), 0.55 (N=6); diamond_prim 0.64, dryice 0.58, diamond_conv
  0.78, prim222 0.54 (N=6): ferric faster.  **dryice_112: 2.20, ferric SLOWER** (N=6) [STALE, do not
  quote: timed on 654087b4, before #508 (fc0ccc041) resized the RS-GDF LR chunk (#507); dryice_112 has
  not been re-timed since; see "Session 4 pre-registration"], with ferric at its
  default omega 1.0 vs PySCF's automatic 0.285; ferric's best omega on dryice was 1.74x faster than its
  default, and the best omega on dryice_112 was NOT measured.  PySCF automatic omega per cell: k222
  0.316, diamond_prim 0.417, dryice 0.330, conv 0.370, prim222 0.316, dryice_112 0.285.
- DF build only: ferric/PySCF = 0.61 (k222 N=1), 0.61 (k222 N=6), 0.71 (diamond_prim), 0.88 (conv),
  0.86 (prim222), but 2.0 (dryice) and **7.5 (dryice_112)** [STALE, same reason as the 2.20 above].
  ferric's LR part (pair FT + GEMM) is the
  cause: 12.6 s -> 266 s from dryice to dryice_112 (21x for 2x the cell), while its SR 3-centre went
  2.8 -> 6.2 s.  Inside the LR part (medians): `LR sink: J3 + J2 GEMMs` 4.99 -> 216.9 s (43x; the
  `aux FT + GEMM` stage 5.68 -> 221.5 s, 39x) vs pair FT 6.95 -> 44.9 s (6.5x).  39-43x exceeds the
  ~16x that nao^2 naux nG scaling predicts for a doubled cell: a performance lead.  ferric's totals on
  the diamond cells and dryice are carried by the hcore stage (PySCF evaluates its range-separated
  hcore twice per run, A4) and the cheaper DF build on dense cells.
- SCF per iteration (matched definition, SCF / iterations): ferric faster on every cell, including
  dryice_112 (1.10 vs 1.25 s).
- Peak RSS: ferric 0.10-2.49 GiB, PySCF 0.21-6.04 GiB (interpretation, not measured: PySCF's in-memory
  hcore work; its cderi is on disk).  A3 (RSS below 8 naux nao^2 = measurement artifact): testable this
  session, NOT triggered (ferric RSS exceeds 8 naux nao^2 on every cell, e.g. dryice_112 2.49 vs 1.13 GiB).

## Verdicts on the pre-registered hypotheses
- H3 (k222 ratio in [0.5, 2]): HOLDS (0.61 / 0.55).  No artifact signature: ferric k222 (54.4 s) is 1.4x
  its own Gamma diamond_prim (39.3 s, first session, N=1); drop counts equal.
- H4 (DF build follows SR3 1:4:8 within +-30%): diamond_prim : conv : prim222 = 1 : 4.3 : 11.9, so it
  holds for conv and FAILS for prim222 (my prediction "fails" was right for prim222, wrong for conv).
  Post hoc: the SR 3-centre stage alone is 1 : 3.85 : 7.85, i.e. the FINDINGS triplet model holds for
  the stage it models; the LR part breaks proportionality.  dryice : dryice_112 SR part = 1 : 2.26 (holds).
- H5 (dryice PySCF DF-build advantage does not shrink on dryice_112): HOLDS, it grows (2.0x -> 7.5x).
- H6 (N=6 medians move < 5% vs the 3-rep first session): HOLDS for ferric (diamond_prim -1.6%, dryice
  +2.6%) and PySCF diamond_prim (+1.9%); FAILS for PySCF dryice (31.89 -> 35.35 s, +10.8%, mostly its
  hcore stage 15.4 -> 21.3 s).  Not explained here; ambient load was higher this session (1-min
  loadavg 4-8 incl. own runs vs first session) and PySCF's hcore is the most memory-bound stage.
- H7 (omega): HOLDS: dryice best 0.5 (1.74x vs default), diamond_prim best 1.0; all tested omegas are
  admissible (energy within 1.5e-9 of the default).
- A4 (PySCF hcore calls): TRIGGERED: `hcore_calls` = 2 on all 40 timed PySCF runs (init_guess='hcore'
  plus the kernel), recorded on every timed run of THIS session.  For the first session it is
  INFERRED (same script and settings, no `hcore_calls` field then): its PySCF hcore column was then a sum
  over 2 calls, ~36.6 s per call on dryice N=1, not 73.2 s.  Totals stand (PySCF does this work by default).

## Mismatches that remain (in addition to the first session's list, items 1-10, which all still apply)
11. N=1 gate deviation (Deviation 1); N=6 gate is the relaxed foreign-CPU < 12% (first session).
12. k-mesh stopping rules: ferric dE < 1e-12 + max orbital-gradient element < 1e-9; PySCF conv_tol 1e-12
    + conv_tol_grad 1e-9 (a norm, not a max element).  Same iteration count (10) on k222.
13. ferric's k-point stage table has no df/hcore sub-split beyond `k hcore` / `k J/K build`; PySCF's k
    "df" is GDF.build only.
14. Gamma 1e-10 was not tested on the new Gamma cells (Energy anchors note).
15. dryice_112 ferric runs saw 8.8-9.3% foreign CPU (higher than the other rows' 4-7%, still < 12%).
16. s/iter definitions differ (table footnote): ferric SCF / iterations, PySCF median callback gap.  The
    same mismatch is in the first session's table, unlisted there; it works AGAINST ferric (the callback
    gap is smaller than SCF / iterations), so #481's "ferric faster per iteration" statement stands.
17. omega: ferric default 1.0 in every two-code row vs PySCF automatic 0.285-0.417 (per-cell list above).

## Open items of #228 after this session
PBE rows (needs either a ferric Lebedev order > 302 or a grid-refinement gate redesigned around 302,
and a re-run of the PySCF PBE anchors with the grid fix); N=1 rows for diamond_conv / prim222 /
dryice_112; jkfit rows (aux-drop mismatch, unchanged); k-meshes beyond 2x2x2; why PySCF's dryice hcore
moved +38% between sessions; ferric's LR build scaling at dryice_112 (a performance lead, not a
benchmark item).

# Corrections to the first session (#481)
The first session's text above is left as written; these corrections apply to it.
- Results line "ferric 8.6 s vs PySCF 73.2 s N=1, different algorithms: PySCF nuclear attraction on the
  FFT mesh" (and mismatch 4): PySCF's GDF hcore nuclear attraction is range-separated (`_RSNucBuilder`),
  as ferric's is, not an FFT-mesh algorithm; and 73.2 s is ~2 x 36.6 s (two evaluations per run,
  INFERRED from this session's `hcore_calls` = 2 under the same script).  Per evaluation PySCF is ~4.3x
  slower than ferric's 8.6 s, not 8.5x.  The second evaluation is ~80% of the dryice N=1 total gap
  (36.6 of 45.9 s) and ~64% at N=6 (7.7 of 12.0 s).  With one evaluation ferric would still be faster on
  dryice N=1, by ~9 s instead of 46 s.
- Pre-registration lines 53-54 ("algorithm difference ... FFT mesh") are left as the pre-registered text;
  the premise is wrong as stated in the item above.
- The s/iter column compares different definitions (mismatch 16 above); the per-iteration conclusion
  stands because the mismatch favours PySCF.
- The PBE path of that session's `pyscf_bench.py` ran on default pruned BeckeGrids (Deviation 2 of the
  second session), not on UniformGrids as the addendum says; no PBE number was reported in either session.

# Addendum 2026-10-22: PBE refinement grid is 150x590
The pre-registered PBE gate above (section "Energy gates", parts (a) and (b)) names 150x974 as the
refinement grid.  974 is not a Lebedev order in ferric's table (6/14/26/50/110/194/302/434/590,
`ferric_dft::prune::SUPPORTED_LEBEDEV_ORDERS`), and the periodic grid refused everything above 302
until this commit, which is why the second session recorded `invalid n_angular: 974`.  The periodic
grid now validates against that shared list, so the refinement grid of the gate is **150x590** for
both codes (PySCF accepts `--grid 150 590`; 590 is in its Lebedev ladder).  The gate is otherwise
unchanged: eps_c = |E_c(75x302) - E_c(150x590)|, and parts (a) and (b) read with 150x590 in place of
150x974.  Inputs: `toml_ri/dryice_pbe_150x590.toml` and `toml_ri/diamond_prim_pbe_150x590.toml`
(the timed cc-pvdz-ri inputs), and `toml/dryice_pbe_150x590.toml` (def2-universal-jkfit).  The
`*_g150.toml` files are left at 974 as the inputs of the recorded failed runs.  No 150x590 PBE anchor
has been run yet; the PySCF PBE anchors also still need the re-run with the grid fix (Deviation 2).
Evidence that ferric's 590 grid is wired correctly (H-only toys, not the bench cells):
`crates/ferric-pbc/tests/pbc_rks.rs` (huge-box SSF anchor exact at 434/590; H2 a = 4 LDA error vs the
uniform reference -3.79e-4 at 75x302, -5.19e-5 at 100x590, -5.18e-5 at 150x590).

# ADDENDUM 2026-10-23: third session, Gamma PBE rows, PRE-REGISTERED

Written and committed BEFORE any anchor or timed run of this session (branch val/pbc-bench-pbe-228,
base origin/main d5436758, which contains the 150x590 periodic grid and the dryice long-range GEMM
fix).  Same matching rules (cc-pVDZ / cc-pvdz-ri from ferric's bundled JSON, exxdiv=ewald, hcore guess,
aux metric cut 1e-10, orbital lindep 1e-6, fresh process and fresh cderi dir per run), same reporting
rules (median, min-max, IQR; IQR/median > 10% labelled noisy; ratios only for the same cell, same N,
same stage definition, with the mismatch list next to them) as the sessions above unless stated here.
Disclosure (not blind): the second session's ANCHOR runs printed PBE walls at N=6 on an older binary
(654087b4) with PySCF on its default PRUNED grid: ferric diamond_prim 12.5 s, dryice 59.5 s; PySCF
18.7 s / 133.8 s (`anchor2_pbe.jsonl`).  Those PySCF numbers are on a different grid and are not reused.

## Rows
Cells `diamond_prim` and `dryice`, Gamma, PBE, ferric inputs `toml_ri/<cell>_pbe.toml` (75x302) and
`toml_ri/<cell>_pbe_150x590.toml` (refinement, anchors only).  Timed grid: 75x302 (the production
default of ferric's periodic grid) in BOTH codes; PySCF `--method pbe --grid 75 302` (pbc
BeckeGrids, unpruned, with the post-`density_fit` grid fix).
1. Anchors (untimed, N=6, one run each): ferric 75x302 and 150x590; PySCF 75x302 and 150x590 at
   precision 1e-11, plus PySCF 75x302 at 1e-12 (precision check below).
2. Timed N=6: 5 valid reps per (cell, code).
3. Timed N=1: 3 valid reps per (cell, code).
Order within each rep alternates F P (even reps) / P F (odd reps), `scatter_driver.py` default.

## Settings
- ferric: release binary built from this branch's base, `OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=N`,
  `ferric --version` recorded per run; default `gdf_omega` 1.0 Bohr^-1 (NOT tuned); SCF stop rule
  density rms 1e-10 (Gamma default).  PySCF 2.13.1: `OMP/OPENBLAS=N`, conv_tol 1e-10 /
  conv_tol_grad 1e-6, `max_memory` 8000 MB, GDF omega = PySCF's AUTOMATIC choice (labelled as such
  in the table; expected 0.417 diamond_prim / 0.330 dryice from the RHF sessions).
- PySCF precision: 1e-11 (the gated RHF value for these two cells).  PBE check: |E_P(75x302, 1e-11) -
  E_P(75x302, 1e-12)| <= 1e-8 Ha/cell, else the timed rows run at 1e-12 (labelled).
- Gates: every timed run holds BOTH heavy slots from before the quiet-wait until it exits.  N=6 valid
  iff foreign-CPU fraction < 0.12 (the relaxed gate of the first session, used again in the second).
  N=1: `N1_GATE=foreign` from the start (start loadavg < 1.5 AND foreign-CPU < 0.12; the second
  session's Deviation 1 adopted up front, not post hoc).  Invalid runs are kept in the jsonl and
  re-run (<= 3 attempts).  If the box is not quiet, the driver waits; no run is recorded noisy-by-choice.
- Every timed run must be converged and within 1e-8 Ha/cell of its own code's 75x302 anchor.

## Energy gate (unchanged from "Energy gates" + the 2026-10-22 addendum, 150x590 refinement)
eps_c = |E_c(75x302) - E_c(150x590)| per code c on the same cell.  A cell's PBE timing rows are
reportable iff (a) |E_f(75x302) - E_P(75x302)| <= eps_f + eps_P + 1e-8 AND (b) |E_f(150x590) -
E_P(150x590)| < |E_f(75x302) - E_P(75x302)|.  Additional artifact STOP: PySCF `n_grid_points` must
differ between its 75x302 and 150x590 anchors (the second session's grid-not-applied defect), and
ferric's 150x590 anchor must differ from its 75x302 one.
Artifact vs physics: if the 75x302 difference is a grid (partition/quadrature) effect it shrinks on
refinement and (b) passes; if it comes from a non-grid defect (functional, density, hcore, DF) it does
not shrink and (b) fails -> no timing row for that cell, investigate instead.  If (a) or (b) fails, the
timed rows are NOT run for that cell (or, if already run, not reported as a comparison).

## Hypotheses (stated before any number of this session)
- H8: both gate parts pass on both cells, and the 150x590 difference is at least 3x smaller than the
  75x302 one (the dominant 75x302 error is the angular partition error, per the ferric module doc).
- H9: ferric/PySCF total wall for PBE lies in [0.3, 1.0] at N=6 on both cells (the RHF Gamma ratios
  were 0.58-0.64 and the extra XC work is grid-bound in both).  Artifact signatures: a ferric PBE total
  BELOW its own RHF total on the same cell and N (RHF has exchange; PBE should cost J + XC, so this is
  possible, not a signature by itself) is checked against the stage table: a ferric XC stage that does
  not grow with grid size between its two anchors means the grid was not applied.
- H10: N=1 -> N=6 speedup of ferric PBE >= 4x on both cells (RHF: 4.6-5.2x); a smaller value points at
  a serial XC stage.

## Mismatches known before running (listed next to every ratio)
1. Grid construction: ferric atom-centred TA-M4 x Lebedev on the CELL atoms with the full (uncut)
   atomic grids, SSF partition with Bragg-Slater size adjustment over image atoms within a covering
   cutoff; PySCF pbc BeckeGrids: TA (same `treutler` radial rule, `radi.treutler_ahlrichs`) x Lebedev
   on EVERY image atom, points cut to the unit cell (boundary weights halved), ORIGINAL Becke
   partition with NO size adjustment (null radii table in `get_becke_grids`) over the image atoms
   that keep points.  Same (n_rad, n_ang) labels, different point sets and point counts.
2. Pruning: none in either code (ferric's periodic grid refuses pruning; PySCF `prune = None`).
3. omega: ferric fixed 1.0 vs PySCF automatic.
4. Precision/stopping rules/aux/process overhead: the first session's items 1, 2, 5, 7, 9d, plus 16
   (s/iter definitions).

# RESULTS of the third session (Gamma PBE; written after the runs)

Binary: `target/release/ferric` built once from d5436758 (this branch's base; the session commits
touch only this directory); `--version` on every ferric run: commit d54367588b31, dirty false,
libint 2.13.1.  PySCF 2.13.1.  Every run (anchors included) held both heavy slots.  Ambient foreign CPU
3.5-8% of the box on every timed run.  Records: `bench_results/anchor3_pbe.jsonl` (gate anchors),
`anchor3_pbe_diag434.jsonl` (post-gate diagnostic, below), `timing3_pbe_n6.jsonl`,
`timing3_pbe_n1.jsonl`.  0 runs discarded anywhere.

## Energy gate (untimed anchors, N=6; Ha per cell; 0 aux dropped in both codes, naux 112 / 672)
| cell | grid | ferric | PySCF (1e-11) | ferric - PySCF | PySCF points |
|---|---|---|---|---|---|
| diamond_prim | 75x302 | -75.0492921362 | -75.04944546185712 | +1.53e-4 | 43141 |
| diamond_prim | 150x590 | -75.0486131096 | -75.04866780563407 | +5.47e-5 | 173754 |
| dryice | 75x302 | -753.6935234956 | -753.6935159344612 | -7.56e-6 | 271064 |
| dryice | 150x590 | -753.6935212603 | -753.6935179689733 | -3.29e-6 | 1061091 |
- diamond_prim: eps_f = 6.79e-4, eps_P = 7.78e-4.  (a) 1.53e-4 <= 1.46e-3 PASS; (b) 5.47e-5 < 1.53e-4
  PASS.  **Reportable.**
- dryice: eps_f = 2.24e-6, eps_P = 2.03e-6.  (a) 7.56e-6 <= 4.28e-6 **FAIL**; (b) 3.29e-6 < 7.56e-6 PASS.
  **Not reportable: no dryice PBE timing row was run** (pre-registered rule).
- Artifact STOPs not triggered: PySCF point counts differ between grids (the grid fix is applied);
  ferric's energy and XC stage both move with the grid (XC stages 4.8 -> 17.8 s diamond_prim, 41.1 ->
  158.5 s dryice).  Every anchor converged.
- PySCF precision check (75x302, 1e-11 vs 1e-12): 4.2e-10 (diamond_prim), 1.4e-10 (dryice), PASS, so the
  timed rows use 1e-11.  (PySCF's automatic omega moves with precision: 0.417 -> 0.405, 0.330 -> 0.321.)
- Anchor walls (disclosed, N=6, one run, not timing data): ferric 12.5 / 25.4 s (diamond_prim 302 / 590),
  60.6 / 180.3 s (dryice); PySCF 22.5 / 71.6 s, 124.7 / 511.5 s.

### dryice diagnostic after the gate failure (post hoc, NOT a gate input)
One extra untimed anchor per code at 150x434 (`toml_ri/dryice_pbe_150x434.toml`, `--grid 150 434`):

| grid | ferric | PySCF | ferric - PySCF |
|---|---|---|---|
| 75x302 | -753.6935234956 | -753.6935159345 | -7.56e-6 |
| 150x434 | -753.6934987201 | -753.6935231043 | +2.44e-5 |
| 150x590 | -753.6935212603 | -753.6935179690 | -3.29e-6 |

Both codes are NON-MONOTONE along 75x302 -> 150x434 -> 150x590 on dryice (ferric moves +2.48e-5 then
-2.26e-5; PySCF -7.2e-6 then +5.1e-6).  The first step changes the radial AND the angular grid, so this
does not show non-monotonicity in the angular order alone.  Without a converged reference, a single
302 -> 590 step is therefore not a reliable estimate of either code's grid error (in either
direction), and gate part (a), which takes eps_c from that one step as the error bar, is not a valid
bound on this cell.  The cross-code difference changes sign with the grid,
and its largest magnitude (2.44e-5, at 150x434) stays below the largest single-code grid move
(2.48e-5, ferric 75x302 -> 150x434).  That is consistent with a grid-quadrature origin; a non-grid
defect smaller than this ~1e-5 Ha/cell grid noise is NOT excluded (interpretation; the gate is still
recorded as failed and the dryice timing rows stay unrun).  A
redesigned gate would need a converged reference per code (finer radial and angular, or a
uniform-grid reference), not a one-step eps.

## Timing, diamond_prim Gamma PBE, 75x302, seconds, valid runs only (median [min-max] IQR)
ferric at default gdf_omega 1.0; PySCF at its AUTOMATIC omega 0.417 (labelled), precision 1e-11.
Stage groups: ferric DF = all `rsgdf` stages; ferric XC = `xc grid build` + `xc AO cache` (once, before
the SCF) + `scf XC eval`; PySCF DF = GDF.build; PySCF "kernel - hcore" = the SCF kernel minus its
two hcore calls, which contains PySCF's XC work (grid build and per-iteration AO evaluation) and J.
| N | code | n | total | DF build | hcore | XC (ferric) / kernel - hcore (PySCF) | iters | RSS GiB |
|---|---|---|---|---|---|---|---|---|
| 6 | ferric | 5 | 12.51 [12.30-12.60] IQR 0.18 | 6.83 | 0.81 | 4.76 | 10 | 0.10 |
| 6 | PySCF | 5 | 21.94 [21.77-22.19] IQR 0.22 | 9.08 | 1.63 (2 calls) | 10.68 | 7 | 0.28 |
| 1 | ferric | 3 | 63.83 [63.29-64.26] IQR 0.48 | 36.12 | 4.04 | 23.56 | 10 | 0.07 |
| 1 | PySCF | 3 | 102.11 [102.01-102.23] IQR 0.11 | 39.79 | 6.79 (2 calls) | 54.89 | 7 | 0.24 |
No row is noisy (IQR/median <= 1.5%).  Every timed run converged; energy spread 0 (ferric) / 5e-13
(PySCF), identical to each code's own 75x302 anchor.  CPU/wall: ferric 5.45 (N=6) / 1.00 (N=1);
PySCF 5.36 / 1.00.  Foreign CPU 3.5-5.9% (N=6), 4.2-8.0% (N=1).  PySCF used 43141 grid points.

## Statements the data supports (diamond_prim only, same N, mismatch list below)
- Total wall ferric/PySCF: **0.57 (N=6), 0.63 (N=1)**, at ferric default omega 1.0 vs PySCF automatic
  omega 0.417, on DIFFERENT 75x302 point sets (mismatch 1 of the pre-registration).  These are
  "same nominal grid label" ratios, not same-quadrature ratios; the two codes' 75x302 energies differ
  by 1.5e-4 Ha/cell, within each code's own 302 -> 590 grid change (~7e-4).
- DF build (like-for-like stage, same as RHF): ferric/PySCF 0.75 (N=6), 0.91 (N=1); the RHF DF ratios on
  this cell were 0.71 at N=6 (second session, 6.59 / 9.23 s) and 0.89 at N=1 (first session, 35.2 /
  39.5 s).
- The XC stage is NOT like-for-like (different point sets and counts: ferric's point count is not
  printed; PySCF 43141), so no XC ratio is quoted.  Within ferric: XC is 38% of the N=6 total and the
  one-time AO cache is 84% of XC (4.02 of 4.76 s).
- Parallel scaling N=1 -> 6: ferric 5.10x, PySCF 4.65x.

## Verdicts on the pre-registered hypotheses
- H8 (both cells pass, refinement shrinks the difference >= 3x): FAILS.  dryice fails gate part (a);
  diamond_prim passes both parts but shrinks 2.8x, not >= 3x.
- H9 (ferric/PySCF total in [0.3, 1.0] at N=6 on both cells): NOT CONFIRMED as stated (dryice
  untestable, gate); within the band on diamond_prim (0.57).  Pre-registered artifact checks:
  ferric XC grows with the grid (4.8 -> 17.8 s, not the signature); ferric PBE vs its own RHF on the
  same cell and N (earlier sessions' rows, different binaries 654087b4 / a50e9964, so indicative
  only): diamond_prim N=6 PBE 12.51 s vs RHF 7.48 s (second session), N=1 63.83 s vs 39.26 s (first
  session).  PBE is not below RHF; the extra ~5 s (N=6) / ~24 s (N=1) matches the measured XC stage.
- H10 (ferric N=1 -> 6 >= 4x on both cells): NOT CONFIRMED as stated (dryice untestable); within the
  band on diamond_prim (5.10x).

## Mismatches (next to every ratio above)
1. Grid construction and partition (pre-registration item 1): ferric TA-M4 x Lebedev, full atomic grids
   on the cell atoms, SSF partition with Bragg-Slater size adjustment over images; PySCF pbc
   BeckeGrids, same radial rule, every image atom's grid cut to the cell, ORIGINAL Becke partition
   with no size adjustment.  Same (75, 302) label, different point sets.
2. No pruning in either code.
3. omega: ferric 1.0 (default) vs PySCF automatic 0.417.  ferric's best omega for PBE was not measured.
4. Stopping rules: ferric density rms 1e-10 (10 iterations) vs PySCF conv_tol 1e-10 / grad 1e-6 (7).
5. Stage definitions: ferric builds its AO-on-grid cache once before the SCF; PySCF evaluates AOs on the
   grid inside every iteration; hence only total and DF-build ratios are quoted.
6. PySCF evaluates hcore twice per run (A4 of the second session); PySCF process wall includes the
   ~0.5 s python/pyscf import; PySCF cderi on disk (HDF5); precision 1e-11 vs ferric's fixed 1e-13/1e-14.
7. N=1 gate = `N1_GATE=foreign` (pre-registered this time); N=6 gate = foreign CPU < 12%.

## Open items of #228 after this session
dryice PBE rows (gate part (a) failed; needs a gate built on a converged per-code reference, see the
diagnostic); PBE omega sweep and best-omega PBE row; k-meshes beyond 2x2x2; best ferric omega on
dryice_112; N=1 rows for diamond_conv / diamond_prim222 / dryice_112; jkfit rows (aux-drop mismatch);
why PySCF's dryice hcore moved +38% between the first two sessions (carried forward; this session ran
no dryice RHF or PBE timing, so it adds no evidence).  The dryice_112 LR-scaling lead of the second
session is not carried: #508 (merged) addressed it (#507); dryice_112 was not re-timed here.

# ADDENDUM 2026-10-26: Session 4 pre-registration (harness hardening; PRE-REGISTERED)

Written and committed BEFORE any anchor or timed run of session 4 (branch
bench/harness-hardening-228, base origin/main 5a536617a).  No session-4 number exists at this commit;
the only runs made while writing it were seconds-long harness smoke checks on H2 / water STO-3G toy
cells (`--smoke` rows, never timing data) and a parse check of every new TOML (below).  Earlier
sessions' text is left as written; their stale dryice_112 ratios are marked inline.
Revised once in review of PR #560 (quoting rule, binary pinning, run_id, rep minimum, ladder
helper), still BEFORE any session-4 run; the text below is the revised version.

## Harness changes (this commit)
- `scatter_driver.py`: `--bin` is required and must be a PRIVATE copy (any `target/` path component,
  as given or after resolving symlinks, and any file with more than one hard link, is refused: a
  rebuild under a running job SIGBUSes it); its sha256 is taken at start and recorded on every row
  (`bin_sha256`); it is re-hashed before each ferric run (outside the timed window) and the driver
  EXITS if it changed.  `FERRIC_LINDEP_THRESH` in the environment is refused.  Each invocation stamps
  a `run_id` on its rows; an existing `--out` is refused unless `--append` (a top-up run).
  `--cpus` pins BOTH codes with `taskset -c` (defaults N=1 -> `1` (core 0 takes most interrupts),
  N=6 -> `0-5`: the i7-6800K's physical cores; SMT siblings 6-11 stay idle).  Every row carries `scaling_cur_freq` of all
  12 CPUs and the coretemp temperatures at the start and end of the timed window, and a `box_state`
  label (next section).  `--ferric-tags` runs several ferric inputs per rep, interleaved with PySCF;
  the sequence [ferric T1 .. Tk, PySCF] is rotated by the rep index.  `--pyscf-match-ferric-conv`,
  `--pyscf-args`, `--smoke` (no locks / waits, rows smoke=true, valid=false).
- `pyscf_bench.py --match-ferric-conv`: PySCF stops on ferric's periodic SCF rule (next sections);
  `max_cycle` defaults to 200 under it (ferric's periodic `max_iter`, config.rs:5341-5346).
- `summarize.py`: the statistics below; `test_bench_harness.py` unit-tests them (pairing, bootstrap,
  noisy label, binary refusal, pinning defaults, rotation, sysfs snapshots, the stopping rule).
- New inputs in `toml_ri/`: `diamond_prim_k333_rhf.toml`; `dryice_112_rhf_w{0.3,0.5,0.7,1.0}.toml`;
  `diamond_prim_pbe_w{0.3,0.5,0.7,1.0}.toml` (`w1.0` writes the default explicitly); dryice PBE grid
  ladder `dryice_pbe_150x302`, `_100x590`, `_200x590`, `_250x590` (with the existing `dryice_pbe`
  (75x302), `_150x434`, `_150x590`).  Parse check: each new file, with only `[molecule] xyz` swapped
  to H2 and `[basis]` to sto-3g, ran to `converged = true` on a private copy of an existing release
  build (commit 3843d65c8, dirty false); the printed `gdf_omega` and the grid-dependent energies show
  the keys took effect.

## Quiet-box preconditions (every row)
- One private binary per session; rows with different `bin_sha256` are never pooled or compared
  (`summarize.py` keys every group on it).  ferric
  `--version` must name a commit that contains #508 (fc0ccc041), checked before the first run.
- Pinning as above for both codes (`cpus` recorded).  Both heavy-slot locks held from before the
  quiet wait until the run exits (unchanged).
- `box_state = "quiet"` iff the 1-minute loadavg at the start of the run is < 1.5 AND the foreign-CPU
  fraction over the run (box CPU outside the run's own process tree / (wall x 12 logical CPUs)) is
  < 2%; otherwise `"ambient"`.  Ambient rows are kept and shown, but only quiet rows enter a QUOTED
  number (`summarize.py --quiet-only`).  If a group has no quiet rows, its ratio is not quoted.
  The existing `valid` gate (N=6 and N=1 with `N1_GATE=foreign`: foreign < 12%, start loadavg < 1.5,
  converged, energy parsed; <= 3 attempts) is unchanged and applies on top.
- Frequency / temperature snapshots are recorded, not gated; the write-up reports, per group, the
  range of the pinned CPUs' start/end `scaling_cur_freq` and the max package temperature.

## Repeat counts and order
- Timed rows: 5 valid reps per (cell, code, input) at N=6; 3 at N=1.  Order rotated per rep (default).
  A group (or a per-session group) with fewer valid reps is labelled "n too small": neither NOISY
  nor ok, and never quoted.
- Between-session scatter: every row to be quoted is run in at least 2 sessions (separate `--out`
  files, started >= 1 h apart, same binary sha256).

## Statistics (`summarize.py`, per (cell, N, code, input, box_state); ratios per binary sha256)
- median, min-max, IQR, IQR/median, relative MAD (median |x - median| / median); NOISY iff
  IQR/median of the quoted metric (total wall unless stated) > 10%; "n too small" below the rep
  minimum above.
- THE quoted ratio: ferric median / PySCF median over the PAIRED reps only (the (session, run_id,
  rep) keys with a valid row for both codes, last valid attempt each), pooled across all sessions
  of one binary, same cell, N and method; with a PAIRED bootstrap 95% percentile CI (10000
  resamples of pairs with replacement, `random.Random(228)`).  Because that bootstrap pools
  sessions, it does not see session-to-session drift; the between-session range below is the guard
  for that.
- Between sessions: per-session medians, CV of those medians (sample stdev / mean), max/min; and the
  per-session ferric/PySCF ratio with its max/min.

## Quoting rule
A ferric/PySCF ratio is quoted only if: same cell, N, method, stage definition and binary sha256;
EVERY contributing row has `box_state = "quiet"`; every per-session group of both codes is labelled
ok (not NOISY, not "n too small"); and the ratio's between-session range (max/min of the
per-session ratios - 1) is < 5% over >= 2 sessions.  `summarize.py` prints QUOTABLE only when all of
these hold (else "not quotable" with the failing conditions).  Every quoted ratio carries its
bootstrap CI and the mismatch list.

## SCF stopping rule (matched work)
Timed PySCF rows use `--match-ferric-conv`.  The hook reproduces ferric's rule as read from the code
(file:line in the `pyscf_bench.py` comment block): Gamma: dP_rms < 1e-10 AND dP_max < 1e-9 AND
|dE| < 1e-3, where dP = D_new - D_old over all nao^2 elements with D = 2 C_occ C_occ^T, then the
orbital-gradient veto max|X^T (FDS - SDF) X| <= 1e-3 (X: canonical orthogonaliser, absolute
eigenvalue cut 1e-6); k-mesh: |dE| < 1e-12 AND max over k of |X_k^H (FDS - SDF) X_k| < 1e-9.  PySCF's
extra `conv_check` cycle is switched off (ferric has none).  Under the same rule ferric `iterations` =
PySCF `cycles` + 1 (ferric counts the guess-energy pass).  Equivalence evidence (toy, untimed,
`--smoke`): water / STO-3G in an 8 Bohr cube, Gamma RHF, hcore guess, an unextrapolated first step in
either code: the first checked step agrees to the 4 digits ferric's `FERRIC_SCF_TRACE` prints
(ferric iteration 2: dE 1.704, dP_rms 5.200e-1, dP_max 1.708; PySCF cycle 1: 1.70437, 0.519993,
1.70816); final energies differ by 1.3e-12 Ha.  Later steps differ (different DIIS), so iteration
counts are reported, not assumed equal (ferric 10, PySCF 12 cycles = 13 ferric-equivalent there).
Gate on the switch: per cell, one untimed PySCF anchor under each rule; if they differ by more than
1e-8 Ha/cell, the matched-rule rows of that cell are not used.

## Rows, in priority order
1. dryice_112 Gamma RHF, N=6: ferric `--ferric-tags _w0.3 _w0.5 _w0.7 _w1.0` interleaved with PySCF
   (automatic omega, labelled).  Replaces the stale 2.20 / 7.5.
2. diamond_prim RHF, N=6: ferric k333 (`diamond_prim_k333`) vs PySCF `--kmesh 3 3 3`; N=1 if budget.
3. diamond_prim Gamma PBE 75x302, N=6: `--ferric-tags _w0.3 _w0.5 _w0.7 _w1.0` vs PySCF.
4. dryice PBE ladder anchors (untimed, N=6, one run per code per grid): 75x302, 150x302, 150x434,
   150x590, 100x590, 200x590, 250x590 (PySCF `--grid R A`, precision 1e-11).
5. If (4) passes the gate below: dryice PBE 75x302 timing, 5 reps N=6, 3 reps N=1.
6. Repeats of the earlier quoted rows (diamond_prim / dryice RHF N=6) for the between-session table.
   That table uses session-4-and-later rows only: rows of sessions 1-3 carry no `bin_sha256`,
   `box_state` or `run_id` (summarize.py labels them "no-sha" / "unlabelled", so they can never be
   QUOTABLE) and are shown for context only.

## dryice PBE ladder gate (replaces the one-step eps of the third session for this cell)
ref = 250x590.  The ref is valid for code c only if |E_c(250x590) - E_c(200x590)| < 1e-6 Ha/cell;
if either code fails, the gate is undecidable and no dryice PBE timing row is run.  Pass iff
  |E_f(75x302) - E_P(75x302)| <= |E_f(ref) - E_f(75x302)| + |E_P(ref) - E_P(75x302)| + 1e-8.
Computed by `summarize.py <anchor jsonl...> --ladder dryice` (`ladder_gate`, unit-tested; PySCF rows
at `--ladder-precision`, default 1e-11; refuses mixed ferric binaries).
The full ladder is reported per code (non-monotone steps are reported, not a failure).  Artifact vs
physics: if the cross-code 75x302 difference is grid quadrature, |E_f(ref) - E_P(ref)| is small next
to it; a cross-code difference that does NOT shrink at the ref points at a non-grid defect and is
investigated instead of timed, whatever the gate says.

## Hypotheses (before any session-4 number)
- H11: with #508, ferric/PySCF total at N=6 on dryice_112 at the default omega is < 1.0.  Artifact
  signature: a ferric `LR sink: J3 + J2 GEMMs` stage near the second session's 216.9 s means the
  binary lacks #508.
- H12: every quiet group has IQR/median <= 3%, and ferric's between-session max/min <= 1.05.  PySCF
  dryice hcore (+38% between the first two sessions) is checked again under pinning.
- H13: diamond_prim k333 ferric/PySCF total in [0.5, 2] at N=6 (k222: 0.55).
- H14: the dryice ladder gate is decidable and passes; |E_f(ref) - E_P(ref)| < 1e-6 Ha/cell.

## Mismatches known before running
The earlier lists still apply (precision, aux handling, hcore twice in PySCF, python import in
PySCF wall, s/iter definitions, omega labelled), except that the stopping rule is now matched
(trajectories and DIIS are not) and both codes are pinned to the same CPUs.
