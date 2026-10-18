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
**Outcome (added after the run): the "not systematically signed" prediction was MISSED.** All 5 uncorrected errors are negative for all three methods (every method overbinds every dimer). n=5 from one chemistry class, so weak evidence either way. The plain-MP2 dispersion control could not be tested (no dispersion-class dimer was run).
If the pipeline is broken I expect: (a) ghost atoms carrying no basis -> CP
energy identical to uncorrected to ~1e-9 Ha (checked per row); (b) frozen core
omitted -> shifts of ~0.05-0.2 kcal/mol on every dimer; (c) CP applied to a
no-CP fit -> fitted methods become WORSE with CP, roughly by the size of the
BSSE (order 0.3-1 kcal/mol for H-bonded dimers at aTZ), mean signed error moving
positive (underbinding). That CP-worse-than-uncorrected is the negative control.
**Correction (added after the run):** (c) was misfiled. CP-worse is the outcome expected for a correct pipeline, not a symptom of a broken one, and it cannot discriminate: BSSE here is 0.36-0.61 kcal/mol, which worsens any method whose uncorrected |MSE| < BSSE/2. The CP control is non-discriminating.

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
51 ~2400 s, 08 ~2460 s.

## Interpretation (provisional)
* Subset: 01, 02, 08 (H-bonded, class 1-23), 51 ethyne dimer and 59 ethyne-water (mixed, class 47-66). NO dispersion-class dimer (24-46) and no pi-stack, so the plain-MP2 pi-stack overbinding control from the issue is untestable here.
* 08 (MeOH-water) was run instead of the issue's suggested 03 (water-MeNH2) because it has fewer atoms (9 vs 10); this was my choice.
* Uncorrected RMSD 0.255 (SCS-2terfc) and 0.173 (MP2-V) vs the published full-set 0.228 for SCS-2terfc: different sets, no "matches" claim. The pre-registered criterion needs the full 66. MP2-V has NO pre-registered pass condition (the issue gives no number I could quote). The 2x band for SCS-2terfc is a judgement call: wide enough to tolerate SCF-convergence and geometry-set differences, narrow enough not to be vacuous.
* Counterpoise: SCS-2terfc RMSD goes 0.255 -> 0.270 while its max error improves 0.319 -> 0.311, so the issue's "visibly worse with CP" criterion is NOT met for SCS-2terfc. MP2-V does get worse (MAE 0.148 -> 0.303). RI-MP2 + CP is the best row of all (RMSD 0.167). The CP control is non-discriminating (see correction above). CP differs from uncorrected in every row, so the ghost atoms plausibly carry basis, but the selftest never exercises build_calc/ghost atoms, so the CP path has no external anchor.
* Geometry and reference are the BEGDB S66x8 "1.00" set (2-decimal references; e.g. water dimer O-O 2.972 A vs 2.924 A in the Psi4 S66 geometry; references differ by up to 0.028 kcal/mol from the Psi4 S66 values). The pipeline is internally consistent, but this is NOT like-for-like with the published 0.228 even for a full run; which set JCTC 2015 used is unverified.

## Acceptance criteria for #278 (status)
* UNMET: plain RI-MP2/aTZ interaction energies vs PySCF `dfmp2` (not run; only aDZ selftest anchors).
* UNMET: CP negative control (non-discriminating; SCS-2terfc criterion not met); no external anchor for the CP/ghost path.
* PARTIAL: subset statistics with per-dimer rows (5 dimers); no "matches" claim.
* NOT RUN: `#[ignore]` regression test; run-twice spread; quiet-box `run_slot.sh` run (box load average was 13-22); wiki update.
* DONE: ruff format/check on the driver; site validation text updated to current facts.

## Other caveats
* The 5e-8 Ha selftest bar is about 3.6x the largest measured diff (1.4e-8); it was not derived from the defect side.
* An uncommitted pilot of dimer 01 gave the same energies; I cannot establish it ran after the pre-registration commit, so a dimer-01 number may have been seen before it. No committed run-twice evidence exists.
* Cost: ~12-41 min per dimer under load for 6-10 atom dimers; three methods repeat the SCF (no shared-SCF Python entry point). Timings are loaded-box upper bounds. Whether the full 66 (up to 24 heavy atoms) is feasible is undetermined; it would likely need a shared-SCF entry point, and the bindings' `k_builder` options (link/cosx) were not evaluated.
