# S66 benchmark: MP2-V and SCS-MP2(2terfc) at aug-cc-pVTZ (issue #278)

Driver: `scripts/validation/s66_terfc_benchmark.py`. Raw records: `*.jsonl`
(one line per dimer x calculation x method, with wall time and build id);
aggregate: `results.json` (from `report`).

## Pre-registered before any S66 number was seen

Pass condition ("matches"): on the FULL 66-dimer set, uncorrected (no-CP, the
fitting convention) RMSD against CCSD(T)/CBS within 2x of the published
SCS-MP2(2terfc) value of 0.228 kcal/mol (issue #278 quoting JCTC 11, 4159
(2015) Table 2), i.e. <= 0.456. A subset result is reported as a subset and is
never called "matches".

If the pipeline is right: uncorrected SCS-MP2(2terfc) and MP2-V errors are small
and not systematically signed; plain RI-MP2 (control) is more overbound than the
two fitted methods on dispersion-bound dimers.
If the pipeline is broken I expect: (a) ghost atoms carrying no basis -> CP
energy identical to uncorrected to ~1e-9 Ha (checked per row); (b) frozen core
omitted -> shifts of ~0.05-0.2 kcal/mol on every dimer; (c) CP applied to a
no-CP fit -> fitted methods become WORSE with CP, roughly by the size of the
BSSE (order 0.3-1 kcal/mol for H-bonded dimers at aTZ), mean signed error moving
positive (underbinding). That CP-worse-than-uncorrected is the negative control.

## Anchors (selftest) and fragment split
`selftest` reproduces committed PySCF/numpy references for H2O/aug-cc-pVDZ
(SCS-MP2(2terfc) total), water dimer/aug-cc-pVDZ (RHF, RI-MP2 corr, MP2-V E_nl).
Python run_* use the library-default SCF (density_conv 1e-6), the Rust tests use
1e-9, so the anchor bar is 5e-8 Ha, not the Rust 5e-11 bar.
`check-split` verifies all 66 fragment sizes (S66_FRAGA from c9_driver.rs)
against the independent monomer nuclear repulsion in benchmarks/grid/S66.py (all
66 monomers agree to < 1e-8 relative).

## Measurement (5 of 66 dimers: S66-01, 02, 08, 51, 59; kcal/mol; NOT the full set)
Run through `ferric-limited`, RAYON_NUM_THREADS=3, on a box with load average 13-22
(timings are upper bounds). Build 330d3013 (clean), script at 823f6155. Raw table:
`results.json`; records: `s66_atz.jsonl`.

| method | conv | MSE | MAE | RMSD | max |
|---|---|---|---|---|---|
| RI-MP2 | no CP | -0.314 | 0.314 | 0.320 | 0.413 |
| RI-MP2 | CP | +0.134 | 0.155 | 0.167 | 0.221 |
| SCS-MP2(2terfc) | no CP | -0.244 | 0.244 | 0.255 | 0.319 |
| SCS-MP2(2terfc) | CP | +0.266 | 0.266 | 0.270 | 0.311 |
| MP2-V | no CP | -0.148 | 0.148 | 0.173 | 0.250 |
| MP2-V | CP | +0.303 | 0.303 | 0.307 | 0.355 |

Wall time per dimer (3 methods x 5 calcs): 01 ~710 s, 59 ~1450 s, 02 ~2050 s,
51 ~2400 s, 08 ~2460 s. Water dimer energies reproduced to 1e-10 Ha across two
independent runs (pilot and main).

## Interpretation (provisional)
* Subset, 4 H-bonded/mixed dimers plus ethyne dimer: no pi-stack, so the MP2
  pi-stack overbinding control from the issue is NOT exercised.
* Uncorrected RMSD 0.255 (SCS-2terfc) and 0.173 (MP2-V) vs the published full-set
  0.228 for SCS-2terfc: different sets, no "matches" claim. The pre-registered
  criterion needs the full 66.
* CP moves the fitted methods from slight overbinding to underbinding: MP2-V
  clearly worse with CP (MAE 0.148 -> 0.303), SCS-2terfc marginally (0.244 -> 0.266),
  consistent with the fit being a no-CP one; CP != uncorrected in every row, so
  ghost atoms do carry basis.
* Not done: plain RI-MP2 interaction energies vs PySCF dfmp2 (issue anchor, <=1e-6
  kcal/mol); only the aDZ selftest anchors above were run.
* Cost: ~12-41 min per dimer under load for 6-10 atom dimers; three methods repeat
  the SCF (no shared-SCF Python entry point). Full 66 (up to 24 heavy atoms) is
  infeasible on this box at this cost without a shared-SCF API and RI-JK.
