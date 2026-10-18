# ferric vs PySCF matched timing (issue #228 core slice)

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
  on the FFT mesh at precision 1e-11), not "ferric's JK is faster": split by stage.
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
| dryice | -750.8211184382 | -750.821118438013 | 1.9e-10 PASS (true value in ~[1.4e-10,2.4e-10]: ferric prints 10 dp) | 2.2e-6 FAIL | 7.3e-8 FAIL | 6.6e-11 PASS |
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
  N=1, different algorithms: PySCF nuclear attraction on the FFT mesh at precision 1e-11).
- s/iteration: ferric faster, but iteration counts differ (stopping rules, below).
- Parallel scaling N=1 -> 6: ferric 5.2x (diamond_prim), 4.6x (dryice); PySCF 4.1x, 4.3x.
- ferric default vs its own unrotated baseline: 3.6x on diamond_prim at N=1 (3.5x at N=6: 26.82 s vs 7.60 s; the 5.2x is the unrotated build's own N=1->6 scaling).  Without SR
  column rotation ferric diamond_prim (140 s) is 3.0x SLOWER than PySCF (47 s).

## Mismatches that remain (read before quoting any ratio)
1. Precision knob: PySCF cell.precision 1e-11 (gated); ferric fixed (RS-GDF 1e-13, hcore 1e-14), tighter.
2. Stopping rules: ferric 10/20 iterations, PySCF 7/13 (same hcore guess, DIIS defaults of each
   code); PySCF conv_tol 1e-10 + grad 1e-6 vs ferric density rms 1e-10.  Iteration counts within 2x.
3. omega: ferric default fixed 1.0 Bohr^-1; PySCF auto 0.417 / 0.330 (ferric `auto`: 1.0 / 0.5).
4. hcore: different algorithms (see above); the hcore stage is 64% of PySCF's dryice setup.
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
