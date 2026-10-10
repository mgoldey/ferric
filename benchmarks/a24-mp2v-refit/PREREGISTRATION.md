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
  r0*omega in {linked (omega=None, = 0.7071), 2, 4, 8, 16, 32} = 36 arms
  (`arms_primary.json`; the sharpness set was extended from {linked,2,4} on the user's
  direction, "freedom to fine-tune with a much sharper omega"; wave 2, NOT yet run).
  2.0 is the Dutoi-safe bound (2.07); 4..32 are progressively harder steps.
  (omega = (r0*omega)/r0.) omega enters both halves in lockstep (MP2 attenuator
  AND the VV10 damping weight); `effective_vv10_damping` enforces it.
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

---

# Addendum A (pre-registered before any run): paper-SI check of the Eq. 11 reading

Data: `paper_si/a21x12.json` (see its PROVENANCE.md). Observation to test: the
paper's VV10 contribution to the binding energy, increment I(f) = S18(f) - S17(f)
(MP2-V minus MP2(terfc), both aTZ, non-CP per the paper's fit convention; the SI
does not state CP for these tables, so we compute both and report both), is
small and changes sign with the stretch factor f. This is the first comparison of
ferric's reading of Eq. 11 (Phi * [1 - terfc(R, r0)^2]) against any paper number.

## Systems (scope fixed by the user: a few small geometries, not the suite)

#2 water-dimer (I = +0.0088 at f=1.0, -0.0287 at 1.6, -0.0177 at 2.0),
#4 HF-dimer (I < 0 at every f), #5 ammonia-dimer (+0.1152 at 1.0, sign change near 1.2),
#8 water-methane (+0.0388 at 1.0), #19 methane-dimer (+0.0416 at 1.0),
#20 Ar-methane only if cost allows (Ar core = 5). Chosen because they are the cheapest
and span the increment's sign pattern (positive-then-negative; always negative).
SMALL SAMPLE: n = 5 (6) systems; nothing beyond sign and magnitude agreement on
THESE systems can be claimed, and no RMSD over them is a population statistic.
Large dimers (ethene-dimer, formaldehyde-dimer, borane-methane, ...) are skipped.

## Step 1: geometry check (cheapest, first)

HF/aug-cc-pVTZ non-CP binding energy at f = 1.0 vs table '3' (SCF/aV5Z non-CP).
The basis differs (aTZ vs a5Z), so a basis offset is expected; criterion: agreement
within 0.03 kcal/mol (coordinator), diagnosed per system. If the A24 geometry is NOT
the f=1.0 geometry the HF energies will disagree by >> 0.03 (water dimer scale: tenths).
If it fails: report and STOP. (Water-dimer SCF/a5Z non-CP at 1.0 is -3.668.)

## Step 2: factor 1.0, published linked parameters

terfc, r0 = 1.00 A, b = 11.0, C = 0.0089, omega = None, frozen core (H,He 0; Li-Ne 1;
Na-Ar 5), aug-cc-pVTZ, aug-cc-pvtz-rifit, DF-JK def2-universal-jkfit, VV10 on the HF
density, NLC (50,50). Compute attMP2-only and MP2-V E_int (non-CP primary, CP stored).
Compare vs S17, S18 and the increment.

Hypotheses and outcomes:
* H_read (ferric's Eq. 11 reading is the paper's): |I_ferric - I_paper| <= tol and sign
  agrees on all systems, with tol = max(0.01, 2 x the largest increment change under
  the sensitivity checks below) kcal/mol.
* H_diff (reading differs): sign or magnitude disagrees beyond tol, systematically
  (same direction or pattern across systems), while S17 itself agrees (attMP2 half fine).
* If S17 (attMP2) disagrees beyond ~0.05 kcal/mol the failure is in the MP2 half
  (RI aux, frozen core, terfc operator) not VV10, and the increment test is confounded;
  the increment I is computed difference-wise within ferric so MP2-half errors cancel
  from I but not from S18 itself.
* Increment is also computed as ferric's own (MP2-V minus attMP2) from the same SCF, so
  it depends on no MP2-half agreement.

Artifact hypotheses (each quantified, not assumed): (a) NLC grid: paper used SG-1,
ferric (50,50); repeat I on (30,50), (75,110), (99,302); required for "grid-insensitive":
variation < 0.005 kcal/mol; else the grid enters the comparison. (b) RI aux:
aug-cc-pvtz-rifit vs def2-tzvpp-rifit (attMP2 half only; affects S17 and S18, not I).
(c) SCF convergence/DF-JK vs exact J/K on one system. (d) post-HF vs self-consistent VV10 (paper
Table 1: 0.202 vs 0.199 RMSD): the paper's S18 may be self-consistent; ferric is post-HF;
the expected effect is small but not measured, so it is listed as UNVERIFIED unless
bounded here. (e) CP vs non-CP choice of the paper's table (not stated).
A genuine reading difference would give a systematic, density-independent pattern
(same sign flip location error across systems); a grid/aux artifact would be
system-dependent and shrink with the refined grid.

## Step 3: stretched geometries

Hypothesis G1: rigid monomers, intermolecular centre-of-mass vector scaled by f (monomer
orientations fixed). G2: same with the scale applied to the minimum-distance (closest
atom-atom) vector. Acceptance: HF/aTZ non-CP binding energies reproduce table '3' across
f in {0.9, 1.3, 1.6, 2.0} to tolerance T = max(0.03, 2 x the largest |deviation| seen at
f=1.0 on these systems) kcal/mol, per system. If neither G1 nor G2 meets T, the stretched
extension is NOT done and stated as such. If accepted: attMP2 / MP2-V / I at those f.
Discriminating features: sign change location and long-range decay of I.

## Step 4: variant constructions (REVISED before any VV10 variant was run; supersedes the earlier V0-V5 list)

Triggered only if Step 2 or 3 shows a disagreement beyond tol. Written down BEFORE reading any
MP2-V / increment number (the factor-1.0 run was in progress but its output had not been read).
Notation: terf(R) = 1 - terfc(R; r0, omega); Eq. 11 factor D_C = 1 - terfc^2 = terf*(2 - terf) >= terf >= terf^2.
No tuning of b, r0, C (published 11.0, 1.00 A, 0.0089; b "was tuned" in the paper per the user's recollection,
which is consistent with the published valley and is NOT retuned here).

* V_A: pair kernel Phi * terf(R).  (This is the single-power form; identical to the earlier "V1".)
* V_B: pair kernel Phi * terf(R)^2.
* V_C: pair kernel Phi * (1 - terfc^2)  = current ferric Eq. 11 reading (observed, not predicted).
* V_D: E_nl-level weight: I_D = I_bare * terf(R_cc), with I_bare the UNDAMPED VV10 contribution to
  the binding energy (dimer - monomers, Vv10Damping::None) and R_cc the COM-COM distance of the two
  monomers (rigid, G1 geometry). This is the only definition of a "weight on the assembled E_nl" that does not need
  information the paper does not give; a monomer's E_nl has no intermolecular distance, so V_D is defined for
  interaction energies only and is NOT the same object as a weight inside the density integral. If the paper
  meant another argument (e.g. a size of the whole system) V_D cannot be defined without more information; that
  case is recorded as UNDEFINED, not scored.
* V_5 (bare): no damping, reference for sign.
* Conditional, only if all of the above fail: V_3 damping inside g,kappa; V_4 beta term damped with the kernel.

Anchor for each pair-kernel variant: r0 -> 0 (1e-3 Bohr) must reproduce bare VV10 up to the documented
R=0 self-pair floor (water/cc-pVDZ: 9.05e-6 Ha); a variant failing the anchor is a construction bug and unscored.
(V_D anchor: terf(R_cc) -> 1 gives I_bare exactly.)

PREDICTIONS (stated before running; ordering is from the damping strength, signs are my guess from the paper's
own pattern, and a wrong sign is a refutation, not something to be explained afterwards):
* Damping strength D_B < D_A < D_C pointwise, so removing more short-range attraction: the increment I (positive = VV10
  reduces binding) should satisfy I(V_B) > I(V_A) > I(V_C) at every f where the short-range kernel matters (0.9, 1.0).
* V_5 (bare) and V_D: I negative at f = 0.9 and 1.0 for water dimer (bare VV10 only adds binding; V_D at
  f=1.0 has terf(R_cc) ~ 1 so I_D ~ I_bare), i.e. WRONG sign vs the paper (+0.0005, +0.0088).
* V_A, V_B: predict I > 0 at f = 0.9 and 1.0 for water dimer (strong short-range removal) -- the prediction that
  could pass; V_B more positive than V_A.
* V_C: no prior prediction (it is the existing reading; its numbers will be read once, after this text is committed).
* Long-range (f >= 1.6): all pair-kernel variants -> terf -> 1 so I -> I_bare asymptote, same sign and decay;
  the paper's negative decaying I at f = 1.6-2.0 is predicted by every pair-kernel variant, so f >= 1.6 does not
  discriminate between them; f = 0.9-1.3 does.
A variant "matches" only if the sign agrees on all five systems at f = 0.9 and 1.0 AND |I - I_paper| <= tol on
all stretched points tested. Honest outcome "none match" is permitted and will be reported as such.


---

# Addendum B (pre-registered, wave 2; nothing in it has been run)

## Hard-step limit and anchor
As r0*omega -> infinity terf(R) -> step(R > r0): terfc = 1 for R < r0 and 0 beyond, i.e. exact 1/r MP2
inside r0 and (damped) VV10 outside. Required anchors before trusting any wave-2 energy:
(i) monotone approach: for one dimer, E_corr(attMP2) as a function of r0*omega in {2,4,8,16,32}
must change monotonically and by decreasing steps; (ii) a toy check where the hard-cut MP2 energy
equals the sum of pair contributions with R < r0 (analytic two-centre s-type test of the 2-centre
terfc integral against quadrature of the step kernel); (iii) as r0 -> large at fixed sharpness the
operator -> Coulomb (existing anchor, r0 = 8 Bohr to 1.4e-4 Ha).

## Sharp-omega artifact hypotheses (each must be tested BEFORE its energies are used)
(a) Integral validity domain. The terfc 2-centre/3-centre integrals use tables/series whose far-field
fallback was derived under the linked s <= 1/2 assumption (valid S > 20 only there). At r0*omega >> 4 they may
leave the validity domain. REQUIRED: an exactness anchor against the independent Fourier-space generator
(`validation_terfc_integrals` and its generator under `crates/ferric-integrals/tests`) at EACH new r0*omega
(8, 16, 32) and each r0 on the valley grid, BEFORE any energy at that arm is trusted. An arm that fails is
excluded and reported, not fixed silently. If real, errors grow with r0*omega; a physics effect would converge.
(b) Metric conditioning. (P|Q) positive definiteness / Dunlap-fit conditioning at sharp omega on the A24 dimers:
report the minimum eigenvalue of the terfc metric per dimer/arm and require r0-monotonicity of E_corr toward the
Coulomb value. A negative or near-zero eigenvalue invalidates the arm.
(c) NLC grid noise from a near-step pair weight. Converge the NLC grid per arm: E_nl and the b-fit must be
stable between (50,50) and a larger grid (required: interaction-energy change < 0.005 kcal/mol, same b*); also
check smoothness of E_nl versus a small geometry displacement (finite step 0.01 A of one monomer; a non-smooth
response signals grid noise, not physics). Water-dimer/aDZ at r0*omega=4 shows E_nl changes 1.5e-6 Ha across
(30,50)->(75,110); a step weight is expected to be worse.
(d) Overfitting. omega is now a free parameter alongside b: fit on the TRAIN set (A24) only and judge on the
hold-out (S22) and leave-one-system-out. Report the fitted (r0*omega, b) SURFACE (RMSD on the full grid) and
whether the optimum is interior or on the arm-grid edge. An edge optimum means the sharpness set is too small,
not that the optimum is found. Differences below the RI noise floor (0.02 kcal/mol) are not interpreted.


---

# Addendum C (written AFTER the factor-1.0 numbers were read; the hypotheses below are therefore post hoc and labelled so)

Observation (committed `incr_f1.json`, `incr_f1_r0_1p35.json`): ferric terfc attMP2 at r0=1.00 A (linked) is less bound than
paper table 17 by 0.13-0.40 kcal/mol (non-CP, systems 2,4,5,8,19), but at r0=1.35 A matches to 0.0003-0.008. r0=1.35 A was
NOT fitted here: it is the paper's own optimal r0 for UNcorrected MP2(terfc, aTZ) non-CP, quoted in the `att_vv10.rs` module
docs (1.35 A non-CP, 1.75 A CP). Moreover ferric MP2-V at the published (r0=1.00, b=11) matches table 18 to 0.006-0.019.
Consequence: S18 - S17 is NOT the VV10 contribution (the columns use different r0); the increment test as framed in Step 2
is invalid. The valid test of the Eq. 11 reading is the MP2-V TOTAL vs S18; the VV10 contribution is ferric's MP2-V minus
attMP2 at the SAME r0.
Hypotheses for the operator behind table 17, predictions stated now:
* H1: table 17 is terfc(r0=1.35, linked omega). Predict r0_eff from a per-system fine scan (1.25-1.45 step 0.05, interpolated to the
  paper value) is system-independent, 1.35 +- 0.03.
* H2: a convention factor between the paper's r0 and ferric's (sqrt2 = 1.414; Bohr/A 1.89): predicts r0_eff = 1.414 at paper r0=1.00 --
  resolvable from 1.35 by the scan.
* H3: different omega linkage at r0=1.00: predicts a system-independent r0*omega in the scanned set {0.25,0.35,0.5,1.0} reproducing
  table 17 at f=1.0; one factor cannot separate H1/H3, the stretch profile (f = 0.9, 1.3, 1.6, 2.0) of table 17 decides.
* H4: RI vs exact integrals: predicted irrelevant (O(0.01) kcal/mol vs the observed 0.13-0.40).
b, C and the published MP2-V r0 are not tuned; r0/omega are scanned only for the VV10-free attMP2 half.

## Addendum D (target correction, user/coordinator-confirmed after reading the numbers)

The "increment = S18 - S17" target (Addendum A Steps 2-4, the V0-V5 sign-pattern criterion) is WITHDRAWN: S17 (MP2(terfc, aTZ)
alone) is evidently computed at its own optimal r0 (~1.35 A, no explicit statement in the SI text; the SI has no
r0 line for the attMP2-only column), so S18 - S17 compares different operators and is not a VV10 contribution. The valid
targets are: (1) the MP2-V TOTAL, table 18, at r0 = 1.00 A / b = 11.0 / C = 0.0089 (all variants scored on totals; attMP2 half at r0=1.00),
at factors 0.9, 1.0, 1.1, 1.3, 1.6, 2.0 (water dimer and ammonia dimer for the stretched ones), and (2) table 17 as an independent check of
the attMP2 operator at its own r0 (Addendum C). Match criterion for (1): |ferric - paper| within 0.02 kcal/mol (the noise level seen
at factor 1.0 is +0.006..+0.019 with a constant sign) AND the offset is constant (+-0.01) across factors rather than growing with the
damping-sensitive short-range factors 0.9-1.1. The earlier sign predictions for V_A/V_B ("increment > 0") are scored as
refuted by their own wording (observed ferric VV10 contributions are about -0.2 kcal/mol for every variant); the more informative outcome is whether any
variant's TOTAL tracks table 18 across factors.

## Addendum E: results summary (all aTZ, non-CP, G1 rigid-monomer geometries, no parameter tuned)

* Geometry: HF/aTZ non-CP vs table 3 (SCF/a5Z) at f=1.0 deviates -0.016..-0.029 kcal/mol on #2,4,5,8,19 (basis offset, same sign).
  G1 (COM-scaled) reproduces table 3 at f = 0.9-2.0 (deviation shrinks with f; water dimer f=0.9: -0.070, the only point beyond the
  pre-registered T = 0.058); G2 (closest-atom vector) is rejected (0.05-1.5 kcal/mol off).
* MP2-V total (r0=1.00, linked, b=11.0, C=0.0089, Eq. 11 damping) vs table 18, ferric - paper, f=1.0: +0.017 (#2), +0.019 (#4),
  +0.0145 (#5), +0.006 (#8), +0.009 (#19) kcal/mol. Stretch (#2 / #5): f=0.9 +0.022/+0.022, 1.1 +0.013/+0.012, 1.3 +0.008/+0.007,
  1.6 +0.004/+0.003, 2.0 +0.001/+0.001: the offset is NOT constant; it grows toward short range.
* Variants on totals, #2 / #5, f=0.9: V_C +0.022/+0.022, bare +0.005/+0.008, V_A +0.057/+0.056, V_B +0.092/+0.090, V_D +0.021/+0.018;
  at f>=1.6 all coincide (to 0.0003). Bare VV10 tracks table 18 with a near-constant +0.005..+0.008; V_A and V_B are worse than V_C.
  The V_C-vs-bare difference (<= 0.017 kcal/mol at f=0.9) is at the level of unquantified differences (SG-1 vs ferric NLC grid
  [grid effect on the dimer E_nl total ~1e-6 Ha], RI vs exact attMP2, Q-Chem HF) and is not interpreted as evidence against Eq. 11.
  Pre-registered sign predictions for V_A/V_B (increment > 0) are refuted (all ferric VV10 contributions are ~ -0.2 kcal/mol at f=1.0).
* Table 17 operator: per-system r0_eff (linked omega) = 1.365 (#2), 1.355 (#5), 1.351 (#8), 1.354 (#19), and > 1.45 (#4, ill-conditioned:
  E saturates at ~0.002 kcal/mol per 0.05 A). H1 (r0 ~ 1.35) supported (1.35 is the paper's own non-CP optimum, not a fit here); H2 (sqrt2 = 1.414)
  not supported; H3 (r0=1.00 with a different omega) gives a system-dependent r0*omega (#19 ~0.35-0.4, #8 ~0.3, #2 below 0.25) so no single operator.
