# ferric vs PySCF matched timing (issue #228 core slice)

Files: `pyscf_bench.py` (PySCF side, one fresh process per call), `scatter_driver.py` (F P F P
interleaved driver, load gate), `toml_ri/` (ferric inputs, see "Deviation"), `bench_results/*.jsonl`
(every run, including invalid and discarded ones).  Ferric side reuses `omega_sweep.py`'s box probes.

## Written BEFORE the timing session (the pass condition and hypotheses)

Disclosure: the energy anchors below were run first and printed wall times as a side effect (ferric
diamond_prim ~40 s and dryice ~92 s at 1 thread; PySCF 1e-12 builds of ~1-2.5 min).  Those are single
unrepeated, partly loaded runs; the hypotheses below were written after seeing them and are NOT
blind to that.

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
