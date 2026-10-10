# Pre-registration: refit of MP2-V `b` at sharp terfc seam (Phase D)

Written and committed BEFORE any production number (no A24/S22 fragment has
been run with this harness; only water-dimer/aDZ validation runs exist, see
`validation_result.json`).

## Question

Matt's MP2-V (JCTC 11, 4159 (2015)) uses `terfc(r0)` with the Dutoi curvature
link `r0*omega = 1/sqrt2` and b = 11.0 (r0 = 1.00 A), fitted to S66 non-CP,
frozen core, aTZ. The 2026 decoupling lets omega be sharper. Phase B (Ne2/aDZ/CP)
found the sharp-omega well ~40% shallower at b = 11.0. Can a refit of b (along
Table 1's valley in r0) make a sharp-omega MP2-V match the linked one?

## Reference data (inventory, provenance)

| set | n | file | reference | trust |
|---|---|---|---|---|
| A24 | 24 | `sets.json` (from psi4 `A24.py`, master branch as fetched by `benchmarks/grid/run_grid.py`) | CCSD(T)/CBS interaction energies, Rezac & Hobza JCTC 9, 2151 (2013) (psi4 file header says "in press"; DOI in the BIND comment is ct400057w) | good; values cross-checked equal to `benchmarks/grid/refs.json`. Geometries are the published ones. |
| S22 | 22 | `sets.json` (psi4 `S22.py`, `BIND_S22B`) | S22B revised CCSD(T)/CBS (Marshall/Sherrill/Hobza 2011) | good; NOTE psi4 also carries the older S22 (`BIND_S220`) values -- we use S22B, differences up to ~0.3 kcal/mol on large dimers |
| S66 | 66 | `sets.json` (psi4 `S66.py`) | CCSD(T)/CBS, Rezac, Riley, Hobza JCTC 7, 2427 (2011) | good, BUT S66 is the PUBLISHED method's training set: it is a hold-out for OUR fit only, and is the right comparison for "does the refit reproduce the published S66 RMSD 0.199" |
| stacked A24 #22-24 | 3 | subset of A24 | same | NOT an independent hold-out (part of A24 train); used only as a class |

Not trustworthy / flagged:
* `benchmarks/a24-subset/results*.json`, `stacked_*`: older cc-pVDZ/aDZ runs of
  RS-MP2-RPA (not MP2-V), different SCF aux and no frozen core; NOT reusable
  as MP2-V baselines.
* `benchmarks/grid/out/*` (main checkout, untracked): aQZ RS-MP2-RPA outputs,
  `trunc_thresh` 1e-3 era; only 144 aTZ files and NO `*_mp2v.out` files exist
  (the `atz_mp2v.toml` jobs were staged, never run). So there is NO existing
  MP2-V A24 baseline of any kind.
* `scripts/scan_a24_aqz_terfc_r0_mae.py::fc_count` counts Ar as 1 frozen core
  (should be 5); A24 #20/#21 are affected in that script. `harness.py::n_frozen_core`
  uses the correct 0 (H, He) / 1 (Li-Ne) / 5 (Na-Ar) rule.
* The published S66 numbers are non-CP; we have no published CP MP2-V numbers.
* a24-subset sampling bias (memory): the 7-system subset is the weakly bound
  tail; the FULL 24 is used here.

## Parameters

* Fixed: C = 0.0089, frozen core ON (rule above), terfc attenuator + Eq. 11 damping
  with the same (r0, omega) in both halves, aug-cc-pV{T,D}Z with the matching
  `-rifit` aux, SCF DF-JK `def2-universal-jkfit` (A24 grid convention), VV10 on
  the plain-HF density (post-HF variant), NLC grid (50,50).
* Arms: r0 in {0.85, 0.90, 0.95, 1.00, 1.05, 1.10} A (Table 1 points) x
  r0*omega in {linked (omega=None, = 0.7071), 2.0, 4.0} = 18 arms
  (`arms_primary.json`). 2.0 is the intermediate robustness arm (Dutoi-safe
  bound 2.07); 4.0 is the sharp arm. (omega = (r0*omega)/r0.)
* b grid: 5.00 .. 20.00 step 0.25 (61 values; `bgrid.json`), parabola refinement
  at the minimum; a minimum on the grid edge is reported as NOT FITTED.
* Objective: RMSD of CP-corrected interaction energies vs CCSD(T)/CBS,
  all 24 A24 systems. Primary convention CP. Non-CP stored and fitted in parallel
  (published convention).
* Table-1-style valley table: for each (r0, sharpness) the optimum b*, RMSD, MSE, MAE.

## Splits

* TRAIN: all 24 A24 (class balance; not the weak tail).
* HOLD-OUT: S22 (22, none used in fitting). Hold-out predictions use b* from A24 only, no refit.
  (Whether S66 is used at all is pending the user's decision; no S66 step is registered here.)
* Leave-one-system-out on A24: refit b without system j, predict j; report b* range and
  the system with max leverage.
* Run order: aDZ full A24 first (cheap, also used to test the harness at scale), then aTZ A24,
  then hold-out. No hold-out result may alter the A24 fit.

## Metrics

RMSD (primary), MAE, MSE, max|err|; by class (my a priori assignment in `analyze.py::A24_CLASS`:
hbond {1,2,3,4,5,9}, mixed {6,7,8,10-14,16}, dispersion {15,17-21}, repulsive stack {22,23,24});
a second class split by reference energy (< -3, -3..-1, > -1 kcal/mol) is reported to make the
class assignment non-load-bearing.

## Baselines (same SCF/aux/frozen core, same table)

1. plain RI-MP2 (Coulomb) CP and non-CP (`--coulomb-mp2`).
2. published linked MP2-V: r0 = 1.00, b = 11.0, omega = None.
3. MP2-V sharp (r0*omega = 4, and 2) with UNREFIT b = 11.0.
4. attMP2 alone (no VV10) per arm (shows how much VV10 must supply).

## Noise floor / effect size

* RI-fit noise (wiki a24 README): canonical vs RI = 0.0025 kcal/mol on H2/aTZ; aux swap
  (aug-cc-pvtz-rifit vs def2-tzvpp-rifit) 0.018 kcal/mol. Differences in RMSD smaller than
  0.02 kcal/mol between arms are NOT interpreted. The aDZ->aTZ basis shift is expected to be larger and is measured.
* Published S66 RMSD 0.199 kcal/mol; our own aQZ A24 MP2 RMSD is 0.129 (wiki). "Matches the linked one"
  is operationalised as: sharp-arm A24 RMSD (CP, refit b) within 0.02 kcal/mol of the linked arm's
  AND hold-out S22 RMSD within 0.05.

## Artifact hypotheses (physics vs implementation, stated before data)

* H-BSSE/basis: if the fitted b merely soaks up BSSE or basis incompleteness, then b*(CP) differs
  from b*(non-CP) by a large, system-class-dependent shift, and b* moves strongly between aDZ and aTZ.
  Expected if real: b*(CP) ~ b*(non-CP), small aDZ/aTZ shift. Flag if |b*(CP)-b*(nonCP)| > 2 or |b*(aDZ)-b*(aTZ)| > 3.
* VV10 grid: if E_nl is grid-limited, b* shifts with the NLC grid. Re-evaluate E_nl on one aDZ
  A24 subset (5 systems spanning classes) with (30,50), (50,50), (75,110): required |dE_int| < 0.005 kcal/mol,
  else the grid is part of the fit and (50,50) is not usable. (Water-dimer check in `validation_result.json`.)
* Frozen-core/aux: a wrong core count or aux would shift attMP2, not VV10; guard: Ar systems (#20, #21)
  checked individually (core = 5 per Ar) and compared with all-electron on #20 only.
* Single-system leverage: LOSO b* range > 3 means the fit rests on a few systems.
* Valley degeneracy: the (r0, b) valley is flat (Table 1 RMSD differences ~0.01); a flat RMSD(b) with b* on
  the edge, or RMSD within 0.01 of the minimum over a b interval wider than +-2, means b is not determined
  and no b* is quoted (interval instead).
* Exactness anchors already in place: scan == all-in-one (bitwise), omega=None == published (bitwise).

## Falsification

"A sharp-omega MP2-V can be refit to match the linked one" is FALSIFIED if, at aTZ with CP:
(a) at every r0 in the valley the best sharp-arm (4.0) A24 RMSD exceeds the linked arm's best by > 0.02
kcal/mol with b* strictly inside the grid; OR (b) the sharp b* sits on the grid edge at all r0 (no minimum);
OR (c) the sharp arm matches on A24 but its S22 hold-out RMSD exceeds the linked arm's by > 0.05 (overfit); OR
(d) the sharp fit holds only for the classes dominated by dispersion while H-bond MSE differs in sign
(the refit trades classes). Support requires none of (a)-(d) AND the artifact checks above to pass.
## Not decided here

Whether CP-fitted parameters should replace the published non-CP convention (reported side by side).
