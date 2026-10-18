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
