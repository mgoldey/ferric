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


---

# Addendum F (pre-registered BEFORE any production number): full refit of attenuated MP2 + attenuated VV10

Scope change from Addenda A-E: the question is no longer "can b be refit at a sharp seam" but "what do ALL of the
method's parameters want to be, and is the extra flexibility real or overfit". Written before any production
(A24/S22/S66) energy exists. Numbers in sections F.11 and F.13 come from the cheap validation runs of this turn and
are labelled MEASURED; everything else is a pre-registered rule.

## F.0 Facts taken from the paper (papers/mp2v.pdf, JCTC 11, 4159; these supersede earlier recollections in A-E)

1. Eq. 11 is E_nlc = int rho(r) int rho(r') Phi_VV10(|r-r'|; b, C) (1 - terfc(|r-r'|, r0)^2): a MULTIPLICATIVE damping of the pair kernel
   with the SAME r0 as the MP2 half ("shared ... rather than adjusted separately"). `eq11` in this refit IS the paper's construction.
   `terf` (terf weight; Addendum A's V_A) and `bare` are NOT in the paper: they are CONTROLS scored under the same protocol to see
   whether the damping form matters, never "candidates for the paper's construction". b "controls the strength of short-range damping"
   and C "the magnitude of C6"; C was FIXED at the LC-VV10 value 0.0089 (LC-VV10's own b is 6.9) and never fitted.
2. Training protocol of the paper (what "the published parameters" means): S66 (66 points); r0 scanned in 0.05 A steps x b in 0.1 steps
   (Fig. 2's plotted grid is 0.5 in b, a 75x75 interpolation); converged HF densities, VV10 NON-self-consistent (the self-consistent
   variant gave RMSD 0.199 vs 0.202 kcal/mol); counterpoise NOT applied; frozen core; RI where an RI basis exists; the VV10 grid is SG-1
   (Q-Chem SG-1 = 50 radial x 194 Lebedev, pruned per radial shell). Our default NLC grid is 50 x 50 unpruned; see F.11 for the (50, 194) arm.
3. The paper's own error surface is a VALLEY (Fig. 2, Table 1): RMSD 0.199-0.212 kcal/mol across r0 = 0.85-1.10 A with the optimal b rising
   steeply with r0 ((0.85, 8.0), (0.90, 9.0), (0.95, 9.5), (1.00, 11.0), (1.05, 12.5), (1.10, 14.5)); "a range of r0 between approximately 0.8
   and 1.2 A appear to be viable". The attenuated MP2 ALONE was optimal at r0 = 1.35 A ("MP2(terfc, aTZ)", SI tables 14/17/20). USE: the
   prior box of the adaptive search is r0 in [0.8, 1.2] at the linked width (the coarse grid of F.5 still spans 0.6-2.0 so the prior is tested,
   not assumed), and the identifiability expectation of F.8 is exactly this valley.
4. omega is NOT a parameter of the paper (terfc has the single parameter r0 with the Dutoi link r0*omega = 1/sqrt2); the decoupled omega, the
   independent VV10 seam and C-free fits are NEW work. ANCHOR A1 (first thing run at any production stage, before any new degree of freedom is
   opened): on the LINKED line (s = 1/sqrt2) the six Table-1 (r0, b) pairs must reproduce the paper's valley: on S66, non-CP, frozen core, aTZ,
   the six RMSDs within 0.03 kcal/mol of Table 1 (0.206, 0.200, 0.212, 0.199, 0.201, 0.204), and the b*(r0) of our own 1-D b search within one
   0.5 of the paper's b at each r0. Failure means the harness differs from the paper (grid, RI aux, HF, frozen core) and NO refit result is
   interpreted until the difference is located (at aDZ, where Table 1 does not apply, only the valley's SHAPE is checked: b* rising with r0).
5. SCS-MP2-V(2terfc, aTZ) (trained jointly on S66 and W4-11 with the weighted-RMSD figure of merit of Eq. 13; r0 = 0.70/0.90 A, b = 14.0,
   cOS = 1.267, cSS = 4.444) is OUT OF SCOPE for this refit; recorded as a possible later extension (a different objective, 3 more parameters).

## F.1 Parameters (what is free)

| # | parameter | symbol | class | where it enters |
|---|---|---|---|---|
| 1 | MP2 attenuation length | r0 (A) | EXPENSIVE | terfc(r; r0, omega)/r in the MP2 operator |
| 2 | MP2 seam sharpness | s = r0*omega (dimensionless; omega = s/r0) | EXPENSIVE | same; `None` = Dutoi-linked s = 1/sqrt(2) |
| 3 | VV10 b | b | CHEAP | kernel (k = b*(3pi/2)(9pi)^(-1/6) rho^(1/6), beta = (3/b^2)^(3/4)/32) |
| 4 | VV10 C | C | CHEAP | omega_0^2 = C (grad rho/rho)^4 + 4pi rho/3 |
| 5 | VV10 damping seam length | r0_vv (A) | CHEAP | pair weight W(R; r0_vv, s_vv) |
| 6 | VV10 damping sharpness | s_vv | CHEAP | same |
| 7 | VV10 damping FORM (categorical; only `eq11` is the paper's, `terf`/`bare` are CONTROLS, F.0.1) | eq11 \| terf \| bare | CHEAP | W = 1-terfc^2 (published Eq. 11) \| terf = 1-terfc \| 1 |

EXPENSIVE = one attenuated RI-MP2 per fragment per (r0, s) node (orbitals reused). CHEAP = evaluated from the stored
density/grid state of the fragment, no SCF and no MP2, with ALL tuples of a call sharing one VV10 pair sum
(`Mp2vSession.vv10`, `att_mp2_vv10_decoupled_scan`). "Cheap" is relative and, MEASURED, the label is misleading for the VV10 half: b and C are nearly free once a damping weight's
pair sum is paid for (0.7-5 s per extra b), but every NEW damping seam (r0, s, or r0_vv, s_vv, or form) costs a full erf-bound pair sum,
10-30x the attMP2 of the same fragment (F.14). So in the current code "cheap" means "no SCF, no MP2", not "free"; the secondary
arms' Nelder-Mead over (r0_vv, s_vv) pays one pair sum per candidate seam per fragment until the tabulated damping factor of F.14 exists.

Lockstep (the published construction) is the PRIMARY arm: W = Eq. 11 with (r0_vv, s_vv) = (r0, s). The API guards it
(`effective_vv10_damping` errors on a diverging seam, unchanged); independent damping is an explicit opt-in
(`AttVv10Config::vv10_decoupling = Some(..)`, or the decoupled scan function) and is scored only as SECONDARY arms (F.7).
Parameters 4-7 are therefore NOT searched until the primary result exists.

## F.2 Primary loop structure (user's specification)

```
for each fragment f (dimer, monomers; CP ghosts when CP is scored):
    SCF once; keep orbitals, density, NLC grid state                       [Mp2vSession]
    for each MP2 seam node (r0, s):                                        [EXPENSIVE]
        e_c(f, r0, s) = attMP2 (frozen core, same aux)
        for each b (a Chebyshev node set in ln b):                         [CHEAP]
            e_nl(f, r0, s, b) = VV10 with Eq. 11 damping at the SAME (r0, s)
```
E_int(system; r0, s, b) = E(dimer) - E(mA) - E(mB) with (CP) or without (non-CP) ghost basis; objective as F.4.

## F.3 Objective and convention (both fitted, shift reported)

* Primary statistic: RMSD of E_int against CCSD(T)/CBS over the TRAIN set, unweighted (the paper's statistic).
* TWO complete fits: non-CP (the paper's convention; interaction energies from the monomers in their own basis) and CP
  (the project rule, memory `counterpoise-dominates-small-basis-fragments`). Neither is "primary"; both are reported with
  the full parameter vector. The parameter SHIFT between the two fits (in ln units, per parameter, with the Hessian-based
  1-sigma) is the BSSE-absorption diagnostic: a fit whose parameters move by several sigma between CP and non-CP is
  absorbing basis-set error into physics parameters. Prediction (before data): terfc attenuation removes diffuse-tail overlap,
  so a smaller r0 (more attenuation) absorbs BSSE in the non-CP fit; expected sign of (r0_CP - r0_nonCP) > 0, as for the
  uncorrected attMP2 (1.75 A CP vs 1.35 A non-CP, Huang/Goldey/Head-Gordon quoted in `att_vv10.rs`). A shift in the OTHER
  direction, or one that is 0 within 1 sigma, refutes this reading.
* Secondary statistics: MAE, MSE, max|err|, per class (F.4), per-system leverage.

## F.4 Data, classes and splits (S66 is the training set; true hold-outs are kept untouched)

Available locally (psi4 database modules; `sets.json` has A24/S22/S66 equilibria; S66by8 (528 pts) and S22by5 (110 pts) BIND tables
exist in `/home/matt/qc/psi4/psi4/share/psi4/databases/` and need an extractor extension -- NOT yet extracted; HBC6/NBC10 are
stored without BIND tables and are not used). Formula-pair overlap between the sets, MEASURED (monomer-formula pair equality, a
conservative proxy for "same chemical system"): A24 vs S66: 3 of 24 (A24 #2, 12, 24); S22 vs S66: 6 of 22 (#2, 5, 11, 13, 17, 20);
A24 vs S22: 6 of 24 (A24 #2, 5, 14, 19, 22, 23). A24 alone (24 pts, 22 of them <= 12 atoms) is too small for 6-7 parameters.

| tier | set | n | role | rule |
|---|---|---|---|---|
| TRAIN | S66 equilibria | 66 | all fitting | the paper's own training set; reproducing the published non-CP S66 RMSD (0.199 at aTZ, linked, b=11) with the PUBLISHED parameters is an anchor of the harness, not a result |
| TRAIN-extension (optional, stage 4) | S66by8 factors 0.9, 1.0, 1.25, 1.5, 2.0 | 330 | adds curvature information (seam sensitivity lives at short range) | decided only after stage 3 shows the fit unidentified on S66 equilibria |
| DEV (cross-validation arbiter) | A24 minus the 3 S66-overlapping systems | 21 | model selection (F.8), never fitted | 
| BLIND hold-out | S22 minus any complex whose formula pair is in S66 or A24 | 12 (#3,4,6,7,10,12,14,15,18,19,21,22) | ONE look at the end, per frozen decision | no parameter, form or box may change after it is read |

Class labels (a priori, from `analyze.py::A24_CLASS` for A24; S66: indices 1-23 hydrogen-bonded, 24-46 dispersion-dominated,
47-66 mixed per the S66 paper's own grouping -- VERIFY against the psi4 TAGL strings before use, this is recollection;
stacked/pi systems form a 4th class "pi-stack" taken from the TAGLs). A second class split by reference energy
(< -3, -3..-1, > -1 kcal/mol) keeps class assignment from being load-bearing.

Cross-validation (the generalisation estimate; each fold is a full nested refit): leave-one-SYSTEM-out (66 folds, cheap
parameters re-minimised; the MP2 node table is reused, so folds cost only cheap work) and leave-one-CLASS-out (4 folds).
Reported: out-of-fold RMSD, the spread of each fitted parameter across folds (a parameter whose across-fold range exceeds its
Hessian 1-sigma by > 3x is "unstable"), the max-leverage system.

## F.5 The adaptive nested search (rule fixed here; implemented in `session.py::adaptive_search`)

The curvature and the minima of the error surface are NOT known in advance, so the search is adaptive. Rule, all constants
pre-registered (the code defaults are `session.RULES`):

0. PRIOR (F.0.3): the search is expected to find its minimum near r0 in [0.8, 1.2] A on the linked line with b rising with r0; the
   coarse grid below spans far beyond that on purpose, and a minimum found OUTSIDE [0.8, 1.2] on the linked line contradicts the paper
   and triggers anchor A1 re-checks before it is believed.
1. COARSE GRID of expensive nodes: r0 in 7 log-spaced values 0.60..2.00 A (ratio 1.22) x s in {linked (1/sqrt2), 1, 2, 4, 8}
   = 35 nodes. Hard box r0 in [0.4, 3.0] A, s in [0.5, 8]. s > 8 is NOT searched until the integral-validity anchors of Addendum B(a)
   (Fourier-space generator at s = 16, 32) pass; until then s = 8 is a wall and a minimum there is reported as "box edge", not found.
2. At every node, b is found by the CHEAP rule: the per-system errors are evaluated on 9 Chebyshev nodes in ln b over [2, 64]
   (one shared VV10 pair sum per fragment), cubic-spline interpolated per system, the pooled RMSD minimised on a 4001-point ln b grid,
   and the optimum VERIFIED by one exact evaluation (spline-vs-exact RMSD must agree to 0.002 kcal/mol, else the point is added and the
   spline refit, <= 3 times). Window shifts x4 once or twice if the minimum is on the window edge; at b in [0.4, 400] an edge is
   reported as "b edge" and the node's RMSD is the edge value, flagged.
3. EDGE EXTENSION: if the best coarse node is on the r0 boundary of the coarse grid, extend by one ratio-1.22 row, at most 2 times
   (within the hard box).
4. LOCAL REFINEMENT around the best `n_basins` = 2 nodes that are separated by > 1.5 grid spacings: a 3x3 stencil in (ln r0, ln s)
   with half the coarse spacing, recentre on the best stencil node, halve the step, repeat. STOP a basin when step < (0.02, 0.04) in
   (ln r0, ln s), or when a round gains < 0.005 kcal/mol RMSD (after the first), or after 4 rounds.
5. MAX EXPENSIVE EVALUATIONS: 80 (r0, s) nodes per fit set (each = one attMP2 per distinct fragment in the set). If the cap is hit the
   result is labelled "budget-limited", not converged.
6. MAP: the full table of (r0, s, b*, RMSD(b*)) over every evaluated node is stored (JSON); contour map and the valley b*(r0) per s are
   reported. Minimum interior/edge flags: r0_low/high, s_low/high, b_edge, "on the coarse boundary".
7. CURVATURE (amended after the dry run, F.15): the REPORTED identifiability matrix is the Gauss-Newton information matrix J^T J of the
   per-system errors in w = (ln r0, ln s, ln b) (first derivatives by central differences, step h = 0.10 in ln units), with covariance
   sigma^2 (J^T J)^-1; it is positive semi-definite by construction and reports a rank-deficient direction as `inf`, never as a number.
   The finite-difference Hessian of F = RMSD^2 is still computed and stored, as a cross-check only: in the planted dry run it was INDEFINITE in 7 of 7
   seeds (softest eigenvalue < 0), so it cannot be the reported matrix. Original text of this item: the finite-difference Hessian of F = RMSD^2 in w = (ln r0, ln s, ln b), step h = 0.10 in ln units
   (a 3x3x3 stencil: 9 expensive nodes, the b direction is cheap), eigenvalues, eigenvectors, condition number, and the implied
   1-sigma of each eigen-direction for sigma_noise (F.9). A stencil step that leaves the box is shifted inward and flagged.
   Step-size check: the Hessian is recomputed at h = 0.05; eigenvalues that change by > 30% flag the surface as non-quadratic at that
   scale (then only the profile curves in F.9 are interpretable).

Planted-parameter recovery (F.10) must use exactly this nested search; the code path is the same function.

## F.6 Secondary arms (scored on top of the primary result; CHEAP only, no new MP2)

Only after the primary fit is written down. At the best (r0, s) node (and at 2 further nodes on its valley) the MP2 energies are fixed
and the cheap parameters are searched with the stored density state:
* S1: C free (b, C), lockstep damping.
* S2: independent damping seam (r0_vv, s_vv) with b, form = Eq. 11.
* S3: form in {eq11, terf, bare} (categorical; each form gets its own continuous refit). `eq11` is the paper's; `terf`/`bare` are controls: a control that
  beats `eq11` out of fold is a finding about the damping form, not a replacement of the published construction until it also passes the BLIND set.
* S4: everything cheap free: (b, C, r0_vv, s_vv) x form.
Optimiser: Nelder-Mead in ln space, multi-start (5 starts: the published seam, the primary optimum, 3 jittered), tolerance 1e-4 in F,
bounded to the same boxes (r0_vv in [0.4, 3] A, s_vv in [0.5, 8], C in [0, 0.1], b in [0.4, 400]). Each is reported with its
Hessian, and is compared to the primary arm by the SAME cross-validation (F.4), not by training RMSD. A secondary arm is "adopted"
only if its out-of-fold RMSD beats the primary arm's by more than the noise floor (F.9, 0.02 kcal/mol) AND the BLIND hold-out
decision (one look) is made on the arm frozen at that point.

## F.7 Model selection / regularisation

Out-of-fold RMSD (leave-one-system-out and leave-one-class-out on TRAIN, evaluated on DEV A24) is the primary arbiter; a parameter-count
penalty is secondary: AICc = n ln(MSE) + 2k + 2k(k+1)/(n-k-1) with k = number of free continuous parameters (primary 3, S1 4, S2 5,
S4 6, plus 1 for a non-default form), reported next to the CV numbers but not used to override them. No ridge/shrinkage term: the
search box edges are the regulariser, and the identifiability analysis (F.8) says which parameters are not determined.
A parameter that is not determined (its 1-sigma exceeds half the box width in ln units, or it sits on an edge) is FIXED at the published
value in the adopted model and reported as "undetermined", never quoted as a fitted number.

## F.8 Identifiability (the Hessian is reported, not just the minimum)

* Hessian of F in (ln r0, ln s, ln b) at the minimum (F.5.7) and, for the secondary arms, in all of their parameters.
* Prediction, stated before data, from the paper's own Table 1: along the published valley b rises steeply with r0
  ((0.85, 8.0) -> (1.10, 14.5)), d ln b / d ln r0 = ln(14.5/8.0)/ln(1.10/0.85) = 2.31 with RMSD flat to 0.013 kcal/mol across it.
  So (a) the SOFTEST Hessian eigenvector must lie near (d ln r0, d ln b) proportional to (1, 2.31) with small ln s component (< 0.3);
  (b) the condition number lambda_max/lambda_min must exceed 20; (c) RMSD must be flat (within the noise floor) along that
  line over at least +-0.15 in ln r0. If (a) fails (softest direction is elsewhere) or (c) fails (the surface is NOT a valley), the
  paper's valley was a feature of its restricted (linked, s = 0.707) family and the "r0-b degeneracy" is not the relevant
  identifiability limit here. Implication when it holds: r0 and b are NOT separately identified; only the valley coordinate
  is; reporting (r0*, b*) as two numbers is a misrepresentation -- the deliverable is the valley and its transversal minimum.
* Other candidate degeneracies to test explicitly (each with the eigenvector): r0 vs s (sharpness trades against length: the integral
  of the weight (1-terfc^2) changes little when both move), r0_vv vs b and C vs b in the secondary arms, and r0_vv vs r0 when
  the damping is decoupled (they are separated only by the MP2/VV10 split, which the data constrain only through the sum).
* Profile curves: RMSD along each coordinate (others re-optimised) with the sigma_noise band; the profile width at +sigma_noise
  is reported next to the Hessian prediction (they must agree to a factor 2 or the quadratic picture is wrong).
* sigma_noise for the 1-sigma conversion: sqrt(0.02^2 [RI aux] + 0.05^2 [CCSD(T)/CBS reference, typical stated uncertainty] +
  grid^2 [F.11]) ~ 0.055 kcal/mol; the 0.02-only value is also printed. Formula: delta_i = sqrt(2 sigma^2 / (n lambda_i)) for
  F = RMSD^2 (chi^2 = n F / sigma^2, Delta chi^2 = 1).

## F.9 Baselines (same SCF, aux, frozen core, same table; every row both CP and non-CP)

1. MP2 (Coulomb).  2. Published linked MP2-V (terfc, r0 = 1.00 A, b = 11.0, C = 0.0089).  3. Published parameters at a SHARP seam,
UNREFIT (s = 4, 8): expected to be badly wrong (a sharp terfc doubles E_corr, measured) -- it is the control that shows how
much refitting b must supply.  4. MP2-V with only b free (r0, s at published).  5. Primary lockstep refit (r0, s, b all free).
6. Secondary arms S1-S4.  7. attMP2 alone (no VV10) at its own optimum, to show what VV10 contributes.
The headline comparison is 4 vs 5 vs 6 by OUT-OF-FOLD RMSD on DEV, with the BLIND set read once at the end.

## F.10 Planted-parameter recovery (dry run on synthetic data; must pass before any production fit is trusted)

`planted.py`: the attMP2 half is a spline-interpolated surrogate over a REAL tabulated (r0, s) node set (cc-pVDZ, non-CP,
water/HF/NH3 dimers at 4 COM stretches each = 12 pseudo-systems, shared monomers); the VV10 half is the real cheap evaluation.
"Experimental" energies = model(planted) + Gaussian noise at the measured floor; the UNMODIFIED nested search + Hessian must
recover the planted (r0, s, b) = (1.15 A, 3.0, 13.0) over 12 noise seeds. Pass criteria, stated before the run:
(i) median |ln(recovered/planted)| < the Hessian-predicted 1-sigma for every parameter (the optimiser is not biased);
(ii) the across-seed sd of each recovered ln-parameter within a factor 2 of the median Hessian 1-sigma (the identifiability numbers are honest);
(iii) the softest eigen-direction of the Hessian carries the largest across-seed scatter (the degenerate pair is flagged by the Hessian, not just present);
(iv) no seed ends on an edge. Failure of (i) -> the search is biased or the b rule is wrong; failure of (ii)/(iii) -> the Hessian
cannot be used to report identifiability for the real fit. The interpolation error of the surrogate on held-out REAL nodes is reported
(it bounds how far the planted test is from the real surface; it does not make it a physics test).

## F.11 Noise floor and the NLC grid (MEASURED this turn)

NLC-grid sensitivity of the VV10 contribution to the NON-CP interaction energy (`gridsens.py`, aDZ, HF density, plain
50x50 = ferric's default NLC grid; kcal/mol; column = damping spec at b = 11, C = 0.0089, r0_vv = 1.00 A; results in
`results/gridsens_adz_a24_02.json`, `..._19.json`). CP-fragment grids were not repeated (they cost as much as the dimer and the
ghost grid is the same construction); the CP-vs-non-CP grid difference is therefore UNMEASURED.

| system | grid (radial, angular) | pts (dimer) | eq11 linked | eq11 s=4 | terf linked | bare |
|---|---|---|---|---|---|---|
| A24 #2 water dimer | (30, 50) | 9000 | -0.23129 | -0.23539 | -0.20840 | -0.24091 |
| | **(50, 50)** | 15000 | **-0.23139** | -0.23549 | -0.20849 | -0.24101 |
| | (75, 50) | 22500 | -0.23139 | -0.23549 | -0.20849 | -0.24101 |
| | (50, 110) | 33000 | -0.23153 | -0.23565 | -0.20861 | -0.24116 |
| | (30, 110) | 19800 | -0.23144 | -0.23556 | -0.20853 | -0.24107 |
| | (30, 302) | 54360 | -0.23162 | -0.23574 | -0.20872 | -0.24124 |
| A24 #19 methane dimer | (30, 50) | 15000 | -0.25541 | -0.25779 | -0.24235 | -0.25890 |
| | **(50, 50)** | 25000 | **-0.25528** | -0.25766 | -0.24222 | -0.25876 |
| | (75, 50) | 37500 | -0.25533 | -0.25771 | -0.24227 | -0.25881 |
| | (50, 110) | 55000 | -0.25594 | -0.25820 | -0.24318 | -0.25926 |

Reading (MEASURED, two small systems, one basis; NOT a guarantee for the 34-atom S66 systems or for step-like weights s >= 8):
radial 50 -> 75 changes E_int by <= 5e-5 kcal/mol; angular 50 -> 110 by 1.4e-4 (water dimer) and 6.6e-4 (methane dimer);
(30, 50) differs from (50, 50) by 1.0e-4 and 1.3e-4. Every difference is below the pre-registered 0.005 kcal/mol criterion by a
factor >= 7, and 40x below the RI floor (0.02). DECISION: the production NLC grid is (50, 50) -- it is converged for the
interaction energy to < 1e-3 kcal/mol on these systems and the damping choice (eq11 / terf / bare, linked / s=4) does not change the
grid sensitivity beyond 2e-5. (30, 50) is admissible for the EARLY coarse stages of the search (2.7x fewer pair evaluations) but the
final b and the Hessian are recomputed at (50, 50). Required at stage 1 (not yet done): repeat on one S66 system with >= 25 atoms,
and with s = 8 (the steepest weight allowed in the box).

SG-1-LIKE ARM (the paper's VV10 grid; ferric has no SG-1, so (50 radial, 194 Lebedev) UNPRUNED is the nearest available -- it has
more points than SG-1 because it is not shell-pruned; `results/gridsens_adz_a24_02_sg1like.json`): water dimer aDZ, non-CP VV10 contribution
(50, 194): eq11 -0.23170, eq11 s=4 -0.23583, terf -0.20880, bare -0.24132 kcal/mol, i.e. -3.1e-4 from (50, 50) (58200 vs 15000 points,
1616 s vs 97 s on the loaded box). Same conclusion: the interaction-energy grid error is < 1e-3 kcal/mol.

DOES THE GRID EXPLAIN THE SHORT-RANGE OFFSET TO PAPER SI TABLE 18? (`rescore_t18.py`, aTZ, non-CP, G1 stretched geometries, published linked
parameters r0 = 1.00 A, b = 11.0, C = 0.0089; value = ferric MP2-V total minus paper table 18, kcal/mol; `results/rescore_t18_atz.json`.)
The (50, 50) rows reproduce Addendum E exactly (eq11 +0.022/+0.022 at f = 0.9 for #2/#5; bare +0.005/+0.008).

| system, f | (50, 50) eq11 | (50, 194) eq11 | change | (50, 50) bare | (50, 194) bare | change |
|---|---|---|---|---|---|---|
| #2 water dimer, 0.9 | +0.0219 | +0.0211 | -0.0008 | +0.0051 | +0.0043 | -0.0008 |
| #2 water dimer, 1.0 | +0.0171 | +0.0167 | -0.0004 | +0.0075 | +0.0071 | -0.0004 |
| #5 NH3 dimer, 0.9 | +0.0218 | +0.0198 | -0.0020 | +0.0076 | +0.0056 | -0.0020 |
| #5 NH3 dimer, 1.0 and #4 HF dimer | (50, 194) NOT run (cost; stopped) | | | | | |

The grid moves BOTH constructions by the same amount (the eq11 minus bare gap is +0.0168 -> +0.0168 for #2 at 0.9, +0.0142 -> +0.0142 for #5 at
0.9): the refinement shifts the offset by <= 0.002 kcal/mol and does NOT change the eq11-vs-bare ordering, so the earlier reading stands -- the
0.017 kcal/mol eq11-bare gap at f = 0.9 is not a grid effect. (n = 2 systems at (50, 194); the other three systems were not re-run.)
It is also below the 0.02 kcal/mol RI floor, so it is not interpreted as evidence for or against Eq. 11 (Addendum E's conclusion).

Frozen core: H, He 0; Li-Ne 1; Na-Ar 5 (Ar = 5; `scripts/scan_a24_aqz_terfc_r0_mae.py` counts Ar as 1 and is not used). Aux: matching `-rifit`
(RI-fit noise 0.0025 kcal/mol canonical-vs-RI on H2/aTZ; aux swap 0.018). SCF: DF-JK def2-universal-jkfit (A24 grid convention) --
the same for every arm.

## F.12 Artifact hypotheses and falsification (the extra flexibility is overfit or an artifact if ANY of these holds)

Artifact hypotheses, each paired with the physics hypothesis and what each predicts:
* H-BSSE: physics -> CP and non-CP fits agree within ~1 sigma on the MP2-V-specific parameters and both beat the published parameters; artifact ->
  the shift is >> 1 sigma and the non-CP fit's RMSD is better than the CP fit's only because BSSE cancels a basis-set error (the paper
  fitted non-CP). Test: fit aDZ and aTZ; parameter shift aDZ->aTZ and CP->non-CP are compared; an artifact-absorbing parameter moves
  with the basis, a physical one does not. Flag |ln(r0_CP/r0_nCP)| > 0.15 or |ln b shift| > 0.3.
* H-sharp: at s >= 4 the kernel overshoots in k-space (E_corr ~ 2x Coulomb MP2, measured), so any b that "works" there is absorbing
  an MP2 overshoot: physics -> fitted b at sharp seams is far off the linked valley AND the transferable total is still good; artifact ->
  the sharp-seam optimum has RMSD no better than the linked one, i.e. b only cancels the overshoot.
* H-grid: b* shifts with the NLC grid. Required: the production grid of F.11.
* H-valley-as-artifact: the paper's r0-b valley might be the attMP2 half's own r0 preference (table-17 fit r0 = 1.35 vs 1.00) rather than the
  VV10 coupling. Test: b* at fixed r0 across r0_vv decoupling (S2): if b* then ceases to move with r0, the valley was coupling of the halves.
* Unquantified differences vs the paper (SG-1 grid, Q-Chem HF, possible self-consistent VV10) are LISTED and not bounded; the refit is
  of ferric's MP2-V, not a reproduction of the paper's numbers.

FALSIFIED (the flexible refit is not supported) if any of:
(a) BLIND-set (or DEV) out-of-fold RMSD of the all-free primary arm is not better than the b-only refit by more than 0.02 kcal/mol, or is worse;
(b) the optimum is on the box edge (r0, s, or b) in either CP or non-CP fit, or the search was budget-limited;
(c) any fitted parameter's across-fold range (leave-one-system-out or class-out) exceeds 3x its Hessian 1-sigma, or the sign of
    a CP-vs-non-CP shift contradicts F.3;
(d) the planted-parameter recovery (F.10) fails;
(e) the parameters move beyond 1 sigma between aDZ and aTZ on the stage-3 overlap set (the fit is basis-specific);
(f) in a secondary arm, a gain in TRAIN RMSD is not accompanied by an out-of-fold gain (classic overfit signature);
(g) CP-fit class MSEs differ in sign between hydrogen-bonded and dispersion classes by more than the noise floor (the refit trades classes).

## F.13 Basis and staged production plan (cost numbers are in F.14; this fixes the order, nothing here has been run)

Bases: aTZ is the paper's; aDZ is the cheap robustness arm. The box cannot do S66 at aTZ with the terfc engine in reasonable time, so:
* Stage 0 (this turn, done): harness, anchors, noise floor, planted test, grid decision.
* Stage 1: aDZ, stratified A24 SUBSET (8 systems: hydrogen-bond 2, 5; mixed 8, 10; dispersion 15, 19; stack 22; + 4 as a strongly bound one) --
  smoke test of the whole nested search at the production grid; budget cap 80 nodes; non-CP and CP both. No hold-out is touched.
* Stage 2: aDZ, S66 stratified subset (24 systems, 8 per class, <= ~20 atoms) as TRAIN; A24 DEV (21).
* Stage 3: aTZ on the SMALL systems only (the 22 A24 systems with <= 12 atoms and S66 systems <= 12 atoms); the aDZ->aTZ parameter shift is
  measured on this overlap set and decides whether aDZ results carry any transferable meaning (F.12(e)).
* Stage 4: S66 full at aDZ (66), leave-one-out/class-out CV, secondary arms S1-S4; aTZ for as many S66 systems as the cost model
  allows; BLIND set read once.
The coordinator's go is required before any of stages 1-4 starts.

## F.14 Cost model

MEASURED (aDZ, dimer fragment, ferric-limited, RAYON_NUM_THREADS = 2, box load average 11-20 from another heavy job, so every wall time below
is an UPPER bound; a quiet box is expected to be 2-4x faster, UNVERIFIED). `results/timing_adz.json`, `timing_erf_share_adz_a24_02.json`:

| fragment | atoms | NLC pts | SCF + grid + session | attMP2 (terfc, one seam) | VV10 pair sum, Eq. 11, 1 seam x 1 b | + 8 more b (same seam) |
|---|---|---|---|---|---|---|
| A24 #2 water dimer | 6 | 15000 | 0.9 s | 1.8 s | 31 s (bare weight, no erf: 2.1 s) | +16 s |
| A24 #19 methane dimer | 10 | 25000 | 1.4 s | 3.7 s (+ sharp 4.0 s) | 86 s | +49 s |
| A24 #13 | 10 | 25000 | 2.1 s | 6.0 s | 82 s | +29 s |

Session memory: 46-92 MB per dimer-class fragment session (measured RSS delta; includes the held SCF state, grid state and basis).
`Pool` refuses to build a fragment that would exceed its budget (default 4 GiB) and sessions are droppable.

THE FINDING THAT CHANGES THE PLAN: the "cheap" half is the dominant cost, not the attMP2. The VV10 pair sum is erf-bound: on the
water dimer a bare (undamped) kernel costs 2.1 s, and each additional UNIQUE damping weight costs ~17-19 s because the factor
1 - terfc^2 evaluates two erf per pair (npts = 15000, all pairs inside the 40 Bohr cutoff, so the work is quadratic in npts:
31 -> 86 s for 15000 -> 25000 points). Extra b values at the same seam cost only ~0.7-5 s each. Consequences:
(1) b (and C) really are free once the seam's damping weight is paid for; (2) a (r0, s) NODE costs one damping-weight pair sum per fragment,
~10-30x its attMP2; batching nodes into one call (`Pool.prefetch_nl`) shares the pair geometry but NOT the erf (5 seams x 9 b = 120 s vs
1 seam x 9 b = 32 s), so batching saves ~15%, not the 5x one would hope; (3) the lever is a tabulated damping factor D(R; r0, s)
(cubic interpolation of the 1-D function of the pair distance; erf is then paid once per table, not per pair). Expected effect: a
node drops from ~30 s to ~(2-9) s on #2 and the per-extra-seam cost from ~17 s to ~1 s, i.e. ~4-10x on the dominant cost; NOT
implemented, NOT measured -- it must reproduce the exact-erf E_nl to < 1e-10 Ha before it is used (the bit-for-bit anchors stay on
the exact path). RECOMMENDATION: do this first (stage 0b); a lower NLC cutoff (40 Bohr now) is a second, independent lever to be
quantified separately.

Model (loaded-box seconds, N = atoms of the dimer, CP fragments carry the dimer's grid, monomer grids are ~N/2 atoms = 1/4 the pair work):
  t_nl(N) ~ 40 (N/6)^2 s per fragment per seam node (exact erf, includes 9 b-nodes + verification);  t_mp2(N) ~ 2 (N/6)^1.4 s.
  per system per node: CP + non-CP = 3.5 t_nl(N); non-CP only = 1.5 t_nl(N).
Per-node cost of a fit set = sum over systems; nodes per fit <= 80 (typically 50-65: the planted run used 61):

| stage | systems (mean N) | convention | per node (loaded, exact erf) | x ~60 nodes | with tabulation (est. /4-10, UNVERIFIED) |
|---|---|---|---|---|---|
| 1 aDZ A24 stratified subset | 8 (N~9) | CP + non-CP | 8 x 3.5 x 90 s = 42 min | ~42 h | ~5-10 h |
| 1 same | 8 | non-CP only | 18 min | ~18 h | ~2-4 h |
| 2 aDZ S66 stratified subset | 24 (N~18) | non-CP only | 24 x 1.5 x 360 s = 3.6 h | ~9 days | ~1-2 days |
| 3 aTZ smallest systems | 20 (N~8) | non-CP only | MP2 x ~15 (aTZ), pair sum same | ~14 h-1 day | ~4-8 h |
| 4 aDZ S66 full | 66 (N~20) | non-CP only | 66 x 1.5 x 440 s = 12 h | ~30 days | ~3-7 days |

The numbers say the full S66 refit is NOT affordable without the tabulation (and probably a lower cutoff); the aDZ stratified-subset
stages are. They also say CP doubles the cost, so the CP fit should reuse the SAME seam nodes the non-CP search visited (the search
is run on the non-CP objective; the CP objective is then re-minimised only over the cheap b and over the SAME node set, with the
local refinement repeated for the CP objective only around the non-CP optimum). Memory is not the constraint: <= 100 MB per
fragment session, ~1.5 GB for a 24-system pool of the 5 fragments (budget guard refuses past 4 GiB).

Recommended ordering: (0b) tabulated damping factor + anchor; (1) aDZ A24 stratified subset non-CP (end-to-end smoke test of the
search at the production grid, ~hours); (1b) its CP re-minimisation; (2) aDZ S66 stratified subset non-CP; (3) aTZ small-system overlap
set (the basis-transfer check, F.12(e)); (4) the S66-wide CV and the secondary arms (cheap: they reuse stored densities, the damping
tuples go through the same tabulated pair sum); BLIND set last. Do not start stage 2 before stage 1's planted/leave-one-out
diagnostics are read.


## F.15 Dry-run results of this turn (planted-parameter recovery, grid, timing) -- MEASURED, written after the runs

Planted-parameter recovery (`planted.py`, `results/planted_dz_sigma0p02.json`; cc-pVDZ, non-CP, NLC grid (20, 26) -- a MECHANICS test, not a physics
test; 12 pseudo-systems = A24 #2, #4, #5 at COM stretches 0.9, 1.0, 1.3, 1.8 with shared monomers; attMP2 = bicubic spline over a real 4 x 4
(r0, s) node table; VV10 = real evaluation; planted (r0, s, b) = (1.15 A, 3.0, 13.0), C = 0.0089, noise sigma = 0.02 kcal/mol; the UNMODIFIED
nested adaptive search + curvature code; 7 noise seeds completed of the 12 planned -- the run was stopped for cost; 53-61 nodes per seed,
no seed on an edge):

| quantity | ln r0 | ln s | ln b |
|---|---|---|---|
| mean ln(recovered/planted) | +0.0001 (0.0%) | +0.033 (+3.4%) | +0.078 (+8.2%) |
| across-seed sd | 0.010 | 0.017 | 0.145 |
| Gauss-Newton 1-sigma (median) | 0.006 | 0.033 | 0.086 |
| sd / GN sigma | 1.7 | 0.5 | 1.7 |

* (i) bias below the Hessian-predicted 1-sigma: PASS for r0 and b (0.0%, 0.9 sigma), s +3.4% vs 1-sigma 3.3% (marginal; the s step floor of the
  search is 0.04 in ln, so a ~4% s bias is the search resolution, not a physics bias).
* (ii) across-seed scatter within a factor 2 of the GN 1-sigma: PASS for all three (1.7, 0.5, 1.7).
* (iii) the softest information eigen-direction is ln b (|component| 0.985, eigenvalue 0.053) and it carries the largest across-seed scatter
  (sd 0.144 vs GN 0.087); the stiffest is ln r0 (sd 0.0025 vs 0.0017): PASS. GN condition number median 2.5e3.
* (iv) no seed on an edge: PASS.
* FINITE-DIFFERENCE HESSIAN OF F: FAIL as a reported matrix -- softest eigenvalue negative in 7 of 7 seeds (h = 0.10, eigenvector mostly ln s);
  replaced by the Gauss-Newton matrix (F.5.7). The interpolation error of the surrogate at the E_int level on 3 held-out REAL off-grid
  (r0, s) nodes (36 pseudo-system x node values) is rms 0.020, max 0.054 kcal/mol against a real seam-dependence range of 8 kcal/mol (the
  per-fragment total correlation energy error is up to 19.7 kcal/mol and cancels in E_int); it is at the noise floor, so the planted test
  cannot resolve differences of that size.
* In THIS synthetic problem r0 is tight (1%) and b loose (14%); (ln r0, ln b) deviations correlate at 0.95, slope d ln b / d ln r0 = 13.8 -- a valley, but
  not the paper's (2.31); the synthetic surface is not the real one and nothing about the paper's valley follows from it.

Grid decision: production NLC grid (50, 50) (F.11). Timing and cost: F.14. Session memory 46-92 MB per dimer fragment. Not done this turn: the 5 unpruned
(50, 194) rescorings beyond #2 and #5 at f = 0.9/1.0, a (75, 302)/(99, 302) corner, CP-fragment grid checks, any step-like weight (s >= 8) grid check,
the Table-1 anchor A1 (needs S66 at aTZ), 5 of the 12 planned planted seeds.


---

# Addendum G: Stage 0b (tabulated VV10 damping) and Stage 1 launch, pre-registered BEFORE the stage-1 production run

Written after the stage-0b gate/cost measurements (section G.1, MEASURED) and before any stage-1 fit. Everything in G.2-G.6 is a rule or a prediction; nothing in them
has been looked at in data.

## G.1 Stage 0b: tabulated damping gate and cost split (MEASURED, `fastdamp_gate.py`, `results/fastdamp_gate_adz.json`)

Opt-in `fast_damping` (default OFF) is plumbed through `compute_vv10_tuple_scan_opts` -> `att_mp2_vv10_decoupled_scan_opts` ->
`Mp2vSession.vv10(tuples, fast_damping=)` / `run_mp2_v_decoupled_scan(..., fast_damping=)` -> `session.Pool(fast_damping=)` (cache keys carry `|fast`, so exact and
tabulated values can never mix). Table: 4-point Lagrange on a spacing `min(0.01 Bohr, width/250)` (width = r0*sqrt2 linked, 1/omega decoupled), so a sharp seam is
resolved at the same relative level as the linked one; beyond `r0 + 5 width` the weight is 1 (Eq. 11 bit-exactly; `terf` to <= 1e-12). The bit-exact skip of #372 is
already active in the exact path of the tuple scan (same `Vv10Damping::factor_from_r2`).

GATE (pre-registered: |E_nl(tabulated) - E_nl(exact erf)| <= 1e-10 Ha, every form, linked seam and r0*omega in {2, 4, 8}):
water dimer and methane dimer, aDZ, NLC (50, 50), forms eq11/terf/bare, r0 in {1.0, 1.35} A, b in {11, 6}, C = 0.0089 (33 tuples each).

| system | pts | max abs error (Ha) | worst relative | worst seam | gate |
|---|---|---|---|---|---|
| A24 #2 water dimer | 15000 | 4.5e-14 | 4.9e-13 | eq11 r0*omega=2 | PASS (2000x margin) |
| A24 #19 methane dimer | 25000 | 4.7e-14 | 5.5e-13 | eq11 linked | PASS (2000x margin) |

The error decreases toward sharp seams (eq11 s=8: 1.5e-16) because the spacing refines with the width. No seam had to be excluded: the tabulated damping is used at
ALL seams of the search box. Rust side: `fast_damping_tuple_scan_matches_exact` asserts the gate on a synthetic density for both forms at linked/2/4/8 and that
`fast_damping=false` is byte-identical to the plain scan; MUTATION-CHECKED (table spacing x25 coarser -> the test fails with 3.3e-9). CP-fragment and aTZ grids were not
re-checked (same code path).

COST SPLIT, RE-MEASURED from scratch (same fragments and settings as the earlier table, exact and tabulated interleaved in one process, two repeats each, load averages
recorded in the JSON). The BOX IS STILL LOADED by other sessions (load average 11-19 during the runs), so absolute times remain upper bounds; interleaved ratios are the
trustworthy part. The earlier cost table (F.14: 31 s / 86 s per damping weight, attMP2 1.8 / 3.7 s, stage estimates 42 h / 9 d / 30 d) was taken on a thermally throttled CPU
(fan fixed 2026-10-10) and PREDATES the fan fix; it is superseded by this table.

| fragment | NLC pts | pair sum 1 seam x 1 b: OLD (throttled) | now exact erf | now tabulated | 1 seam x 9 b exact / tab. | 5 seams x 9 b exact / tab. | bare kernel, 1 b |
|---|---|---|---|---|---|---|---|
| A24 #2 | 15000 | 31 s | 12.5 s | 3.0 s (4.2x) | 15.4 / 6.2 s (2.5x) | 44.7 / 15.6 s (2.9x) | 1.8 s |
| A24 #19 | 25000 | 86 s | 37 s | 6.9 s (5.4x) | 48 / 20.5 s (2.3x) | 125-181 / 54-69 s (2.2-3.3x) | 6.5 s |

Reading (MEASURED, single runs, loaded box): the exact path is already ~2.5x faster than the throttled numbers (fan fix + #372's bit-exact skip together; not separable here).
Tabulating removes most of the remaining erf cost: 4-5x for a single kernel, 2.3-2.9x for the 9-b Chebyshev batches the search actually issues, because the pair sum is now
dominated by the per-(b, C) denominators (9 b cost ~2x one b at a seam). attMP2 per seam: 1.4 s (#19) to 4.5 s (#2, includes first-call warm-up); no longer negligible.

RE-DERIVED stage estimates (model, UNVERIFIED until the first stage-1 nodes are timed; dimer pair-sum coefficient 3.1 s per seam-node at 15000 pts, i.e. 0.0138 s per kpoint^2;
monomer fragments 1/4 of a dimer; attMP2 ~3 s per dimer + 1 s per monomer; loaded box): per (r0, s) node over the 8 stage-1 systems and one convention ~55 s pair sums + ~55 s MP2
= ~110 s; 80 nodes ~2.5 h per convention, ~5 h for CP + non-CP (earlier estimate 42 h). Stage 2 (24 systems, N~18, non-CP only): ~23 min/node, ~1 day (earlier 9 days). Stage 4
(66 systems, non-CP): ~80 min/node, ~4 days (earlier 30 days). Stage 3 (aTZ, small systems): MP2 dominates (~10-15x aDZ), ~1-2 days. These will be replaced by the measured
per-node cost reported at the end of stage 1.

## G.2 The stage-1 system subset (fixed before launch)

A24 systems {2, 4, 5, 6, 8, 19, 20, 24}, 4-10 atoms (mean 7.1; all fragment sessions, CP + non-CP, ~2 GB).

| class (analyze.py A24_CLASS) | systems | why |
|---|---|---|
| hydrogen-bonded (3) | #2 water dimer (-5.014), #4 HF dimer (-4.572), #5 ammonia dimer (-3.157) | the three smallest H-bonded dimers; span 1.9 kcal/mol; #4 (4 atoms) is the cheapest set |
| mixed (2) | #6 CH4-HF (-1.679), #8 CH4-H2O (-0.672) | polar + dispersion; the two smallest |
| dispersion (2) | #19 CH4 dimer (-0.538), #20 CH4-Ar (-0.408) | pure dispersion; #20 exercises Ar (frozen core 5) |
| repulsive stack (1) | #24 C2H4 dimer (+1.075) | the only positive-energy member small enough; tests that the fit does not over-bind |

Class balance 3/2/2/1 against 6/9/6/3 in A24 (the smallest members of each class). The stack class has ONE member, so its class RMSD is a single point and is reported as such.
Monomer-formula overlap with S66 (already analysed: A24 #2, 12, 24): #2 and #24 of the subset are in it, the other six are not. Overlap with S22 (A24 #2, 5, 14, 19, 22, 23): #2, 5,
19 -- those S22 complexes are already excluded from the BLIND set (F.4). Stage 1 trains on these 8 only and touches no S66/S22 system, so no hold-out is consumed. No cross-validation
at n = 8: ALL numbers are in-sample and are reported as such; the purpose of stage 1 is mechanics, identifiability and cost, not a generalisation claim.

Fixed settings (Addendum F): primary arm lockstep Eq. 11 (VV10 damping shares (r0, s)), C = 0.0089 fixed, b free; the F.5 adaptive rule with node cap 80 PER convention and the box wall
at s = 8; NLC grid (50, 50); aDZ (aug-cc-pVDZ, aug-cc-pVDZ-rifit); frozen core H/He 0, Li-Ne 1, Na-Ar 5 (Ar = 5); RHF densities, non-self-consistent VV10; tabulated damping ON (gate G.1);
both conventions (non-CP, CP), each a FULL independent search sharing the evaluation cache. Gauss-Newton sigma_noise = 0.055 kcal/mol (F.8).

## G.3 Predictions (stated before data)

Physics hypothesis P: the paper's valley carries over to aDZ on these 8 systems. Null/artifact hypothesis N: the 8-point, 3-parameter problem is under-determined and the optimum is
set by one or two systems.
1. VALLEY (anchor A1-lite, G.4): on the linked line, the optimal b at r0 = 0.85, 0.90, 0.95, 1.00, 1.05, 1.10 A INCREASES monotonically with r0 in both conventions; the log-log slope
   d ln b / d ln r0 lies in [1.2, 4.0] (paper Table 1: 2.31 at aTZ/S66); the RMSD varies by < 0.06 kcal/mol across the six r0 (paper: 0.013). Under N the slope is unconstrained or
   negative and the RMSD varies. Monotone-and-in-range separates P from N (a random b ordering is monotone with probability 1/720).
2. Position: full-refit r0 in [0.8, 1.2] A and b in [6, 20]; s NOT at the wall s = 8 (a minimum at the wall is reported as "edge, not found", not as a result).
3. CP vs non-CP: ln(r0_CP / r0_nonCP) > 0 (CP removes the BSSE overbinding that a smaller r0 absorbs; same sign as plain attMP2, 1.75 CP vs 1.35 non-CP A). Refuted if <= 0 with the shift
   larger than its Gauss-Newton sigma.
4. Curvature: the softest Gauss-Newton direction has same-sign (ln r0, ln b) components both > 0.3, ratio ~2.3, and |ln s| component < 0.5 (the paper's valley). A softest direction dominated by
   ln b alone or ln s would mean the 8-system data do not constrain r0 (the planted dry run found exactly that on a synthetic surface).
5. In-sample nesting (arithmetic, not a finding): RMSD(full refit) <= RMSD(b-only refit at the published seam) <= RMSD(published, b = 11); full-vs-b-only gain expected < 0.03 kcal/mol; a larger
   in-sample gain is NOT evidence of generalisation.
6. Published parameters at a sharp seam, unrefit (s = 4, 8; r0 = 1.0, b = 11): clearly worse than linked (a sharp terfc roughly doubles E_corr, measured earlier): the control showing b must move with s.

## G.4 Anchor A1-lite (what stage 1 can and cannot claim)

Run on the linked line r0*omega = 1/sqrt2 at the paper's six r0 on THESE 8 systems at aDZ: optimal b per r0 (1-D search, same rule as the main search) compared with the paper's b in direction
and slope only. This is NOT anchor A1: A1 (S66, non-CP, aTZ, frozen core, the six Table-1 RMSDs within 0.03 kcal/mol and b* within 0.5) is a LATER stage and is NOT claimed here. A1-lite can
fail (non-monotone b, slope outside [1.2, 4.0]); if it fails, stage 1 stops and reports.

## G.5 Run protocol

`stage1.py`, detached (`nohup setsid nice -n 10 scripts/ferric-limited --max=12G --high=10G`, RAYON_NUM_THREADS = 4, OPENBLAS_NUM_THREADS = 1, one production process). Every attMP2 and E_nl is cached to
`results/stage1_cache_adz.json` keyed by (fragment, params) (uncommitted bulk); `results/stage1/heartbeat.json` and `stage1.log` are rewritten after every node (nodes done / cap, ETA upper bound,
best RMSD so far, load); per-node surfaces `nodes_{ncp,cp}.json` (r0, s, b*, RMSD, per-system errors, b-edge flags); per-phase JSONs (search, valley, curv, base, summary). Re-running the same
command resumes. Interim reports at the first node batch and at ~50% of the nodes.

## G.6 Stop conditions (report immediately, no further stage)

(a) the A1-lite valley check fails (G.3.1); (b) the optimum of either convention is on a box edge (r0, s or b) or the search is budget-limited at 80 nodes; (c) the Gauss-Newton matrix reports an
undetermined parameter (inf sigma) at the optimum; (d) the in-sample ordering of G.3.5 is violated (a harness error, not a result); (e) any test of the code in use fails after launch.


---

# Addendum H (2026-10-10): stage-1 non-CP search result, stop condition G.6(b) tripped, decision

## H.1 Measurement (non-CP search, `results/stage1/search_ncp.json`, `nodes_ncp.json`; aDZ, NLC (50, 50), lockstep Eq. 11, C = 0.0089, tabulated damping, 8 systems of G.2; all RMSDs in-sample)

The F.5 adaptive search ended at 69 of the 80-node cap (not budget-limited). Process note: the first run was killed at node 9 by SIGBUS because the shared `target/release/libferric.so`
was rebuilt under the running process (not a code defect; cause established from the core: fault address inside the mapped .so, .so mtime equal to the crash time); the run was resumed
from the cache with a private copy of the library and reproduced nodes 1-9 identically.

| node | r0 (A) | s = r0*omega | b* | RMSD (kcal/mol) | b status |
|---|---|---|---|---|---|
| best overall (node 60) | 1.152 | 0.522 | 251.99 | 0.0890 | ON THE UPPER b EDGE (window shifted twice; the next shift would pass the 400 limit) |
| best with b* inside the pre-registered range b <= 20 (node 43) | 0.991 | 1.354 | 10.95 | 0.0930 | interior |
| node 65 (lowest RMSD among nodes the code flags interior) | 1.123 | 0.522 | 72.96 | 0.0904 | interior by the flag, but b* = 73 is outside G.3.2 and on the runaway trend toward the edge |
| node 67 | 1.152 | 0.563 | 67.05 | 0.0934 | interior |
| node 68 / 69 | 1.181 | 0.522 / 0.563 | 251.99 | 0.0914 / 0.0917 | edge |

Gain of the edge optimum over the best interior node: 0.0930 - 0.0890 = 0.0040 kcal/mol, below the 0.02 kcal/mol noise floor (F.8). 10 of the first 24 nodes (the coarse grid) had b* on the
edge, mostly at sharp seams (s >= 4, RMSD 0.19-4.2) and at r0 >= 1.34 A. Linked-seam (s = 0.707) coarse values: r0 = 0.600, 0.733, 0.896, 1.095 A give b* = 5.6, 6.6, 9.2, 18.8 and RMSD
0.159, 0.135, 0.125, 0.117. Along the b* values of the interior nodes near the optimum, b* rises with r0 at fixed s (s = 0.522: 73 at r0 = 1.123, edge at 1.152).

Stop condition G.6(b) (optimum on a box edge) is TRIPPED for the non-CP convention. The job was stopped at the valley phase (one valley point computed: r0 = 0.85, b* = 8.25, paper 8.0, RMSD 0.127).

## H.2 Decision (Matt, 2026-10-10, relayed by the coordinator)

The interior basin (r0 = 0.991 A, s = 1.354, b* = 10.95, RMSD 0.0930) is the stage-1 NON-CP result. b is reported as NOT IDENTIFIED at the wide-seam edge. The b search window is NOT widened and the
pre-registered rules (F.5, G.2) stand unchanged. Any later optimum on an edge (including the CP search) is again a stop condition. Curvature and baselines are evaluated at the interior basin
(the lowest-RMSD node with an interior b* AND b* <= 20, the G.3.2 range: this is node 43, as the decision states; the code's weaker 'interior' flag alone would pick node 65, b* = 73, RMSD 0.0904, which sits on the same runaway trend and is NOT what the decision names); the raw edge optimum and node 65 are reported alongside, flagged. This node-43 rule was fixed after seeing the nodes (correction of the first draft of H.1, which mislabelled node 43 as the best interior node).

## H.3 Interpretation (separate from the measurement; PROVISIONAL, untested)

* The two regions are within 0.004 kcal/mol, inside the noise floor, so the data do not prefer the edge optimum; the sharper (s = 1.35) interior basin and the wider (s ~ 0.52) edge region are not
  distinguished by 8 systems.
* A flat valley that runs toward large b at wide seams (b rising steeply with r0 near s ~ 0.5) is one reading of the 73 -> 252 trend. It is NOT tested: the b > 252 region was not searched, the Gauss-Newton
  matrix at the edge was not computed, and an unconverged b search at a wide seam could also be an artefact of the 9-node Chebyshev spline over a window whose far end is poorly sampled.
* Against prediction G.3.2 (b in [6, 20]): the interior basin satisfies it; the raw optimum does not. Prediction G.3.1 (valley) is evaluated by the resumed A1-lite phase, not by these nodes.


---

# Addendum I (written 2026-10-24): stage-1 counterpoise (CP) search, and the non-CP A1-lite valley that Addendum H did not report

Written AFTER every number below was read. Sections I.1-I.5 are MEASUREMENT (raw values copied from the committed JSONs);
I.6 scores them mechanically against the rules pre-registered in F and G; I.7 is INTERPRETATION and is labelled
provisional; I.8 lists what is NOT decided here. Nothing in this addendum changes a rule, a box, or a prediction.

## I.0 Provenance

* Run: `stage1.py` (driver as committed in the lane's 80346d0a9 on `feat/mp2v-full-refit`; Rust tree identical to 328f2afbe,
  because 80346d0a9 touched no Rust). The ferric `.so` build stamp (`ferric.__build__`) was NOT recorded by the driver; that is a
  provenance gap, and the claim above rests on the Rust tree being unchanged between those two commits.
* Timeline (EDT, `results/stage1/stage1.log`): non-CP A1-lite valley resumed 2026-10-10 22:24-22:40, non-CP Gauss-Newton 22:45-22:50,
  attempt 1 exited rc = 1 at 22:52 (a logging `KeyError` in `run_base` after `base_ncp.json` was written); attempt 2 ran the CP search
  23:46 (2026-10-10) to 02:54 (2026-10-11) and exited rc = 1 at the same logging line after `base_cp.json` was written; attempt 3
  (02:57:31) loaded the 2-line fix (skip the `raw_search_optimum` key in the log loop) and finished from the cache with zero new MP2
  (`heartbeat.json`: `n_mp2 = 0`). The fix touches only the log loop, so no number depends on it.
* Settings: exactly G.2 / H.1 (A24 {2, 4, 5, 6, 8, 19, 20, 24}; aug-cc-pVDZ / aug-cc-pVDZ-rifit; frozen core H/He 0, Li-Ne 1, Ar 5;
  RHF densities, non-self-consistent VV10; NLC grid (50, 50); lockstep Eq. 11, C = 0.0089, b free; tabulated damping ON (gate G.1);
  F.5 rule, cap 80 nodes). CP = Boys-Bernardi, each monomer in the full dimer basis (`@` ghost atoms, `harness.py`); CP fragments carry the
  dimer's NLC grid (F.14). The CP search is a FULL independent search sharing the evaluation cache (G.2), not a re-minimisation over the non-CP nodes.
* All RMSDs are in-sample over n = 8, kcal/mol, error = calc - ref (positive = underbound). Raw data: `results/stage1/{search,nodes,valley,curv,base}_{cp,ncp}.json`,
  `summary.json`, `stage1.log`, `stdout.log`. The evaluation caches `results/stage1_cache_adz.json` (4.1 MB) and `results/planted_cache_dz.json` (2.2 MB)
  are NOT committed; they stay in the lane worktree `.claude/worktrees/agent-af29ac89265471080/benchmarks/a24-mp2v-refit/results/`.

## I.1 CP search (MEASURED)

75 of 80 nodes, not budget-limited; every edge flag false (r0, s, b, coarse boundary). Lowest-RMSD nodes:

| r0 (A) | s = r0*omega | b* | RMSD |
|---|---|---|---|
| **1.408** | **1.164** | **8.71** | **0.0780** (search optimum) |
| 1.408 | 1.354 | 9.85 | 0.0783 |
| 1.480 | 1.354 | 13.42 | 0.0795 |
| 1.339 | 1.354 | 7.86 | 0.0824 |
| 1.480 | 1.164 | 10.66 | 0.0837 |
| 1.556 | 1.354 | 22.26 | 0.0846 |
| 1.211 | 1.000 | 5.63 | 0.0943 |

The second basin explored by the F.5 rule (around r0 ~ 1.0 A, s ~ 0.8) bottomed at 0.1095 (r0 = 1.068, s = 0.888, b* = 4.68).
Coarse-grid b* reached the 252 edge only at s >= 2 with r0 >= 1.64 A (RMSD 0.45-3.4).

## I.2 Linked-line valley, A1-lite (MEASURED; the non-CP column was computed 2026-10-10 22:24-22:40 and is reported here for the first time)

| r0 (A) | b* CP | RMSD CP | b* non-CP | RMSD non-CP | paper Table 1 b (aTZ, S66, non-CP) |
|---|---|---|---|---|---|
| 0.85 | 3.93 | 0.1286 | 8.25 | 0.1268 | 8.0 |
| 0.90 | 4.11 | 0.1231 | 9.29 | 0.1252 | 9.0 |
| 0.95 | 4.32 | 0.1192 | 10.65 | 0.1237 | 9.5 |
| 1.00 | 4.54 | 0.1174 | 12.49 | 0.1219 | 11.0 |
| 1.05 | 4.78 | 0.1180 | 15.12 | 0.1196 | 12.5 |
| 1.10 | 5.03 | 0.1210 | 19.27 | 0.1166 | 14.5 |
| d ln b / d ln r0 (6-point fit) | **0.96** | | **3.23** | | 2.27 (same fit on Table 1; F.8's 2.31 is the end-point ratio) |
| RMSD spread across the six r0 | | 0.011 | | 0.010 | 0.013 |

No row is on a b edge. Both conventions give b* increasing monotonically with r0.

## I.3 Gauss-Newton curvature (MEASURED; w = (ln r0, ln s, ln b), sigma_noise = 0.055)

| convention, evaluated at | eigenvalues (h = 0.10) | softest eigenvector (ln r0, ln s, ln b) | 1-sigma per parameter (ln r0, ln s, ln b) | condition |
|---|---|---|---|---|
| CP, search optimum (1.408, 1.164, 8.71) | 0.0274, 0.827, 24.9 | (-0.246, -0.098, -0.964) | 0.083, 0.068, 0.321 | 911 |
| non-CP, interior basin of H.2 (0.991, 1.354, 10.95) | 0.0738, 0.874, 15.4 | (-0.218, -0.367, -0.904) | 0.047, 0.092, 0.184 | 209 |

h = 0.05 changes every GN eigenvalue by <= 3% in both conventions. The finite-difference Hessian cross-check (F.5.7, not the reported matrix):
CP softest eigenvalue 0.0179 (h = 0.10) vs 0.0116 (h = 0.05), a 35% change, so the CP surface is flagged non-quadratic at that scale along its softest direction.

## I.4 Baselines (MEASURED; RMSD / MSE)

| row | CP | non-CP |
|---|---|---|
| MP2 (Coulomb) | 0.431 / +0.352 | 0.303 / -0.256 |
| published linked MP2-V (r0 = 1.00 A, b = 11.0) | 0.568 / +0.545 | 0.129 / -0.064 |
| published, sharp s = 4, unrefit | 0.908 / +0.821 | 0.388 / -0.200 |
| published, sharp s = 8, unrefit | 0.958 / +0.844 | 0.486 / -0.285 |
| b-only refit at r0 = 1.00 A, linked | 0.117 / +0.013 (b = 4.54) | 0.122 / -0.023 (b = 12.49) |
| full refit (CP: search optimum; non-CP: interior basin) | 0.078 / +0.019 | 0.093 / -0.005 |

Per-class MSE of the full refit: CP hbond +0.020, mixed +0.074, dispersion +0.038, stack (n = 1) -0.137; non-CP hbond +0.016, mixed -0.051, dispersion +0.007, stack -0.003.

## I.5 CP minus non-CP parameter shift (MEASURED, `summary.json`)

Against the H.2 interior basin: ln r0 +0.351, ln s -0.152, ln b -0.229. In combined GN sigmas (sqrt(sigma_CP^2 + sigma_nCP^2) = 0.096, 0.114, 0.370):
+3.7, -1.3, -0.6. Against the raw non-CP edge optimum (1.152, 0.522, b on the edge): ln r0 +0.20, ln s +0.80, b not defined.
On the linked line at r0 = 1.00 A: ln(b*_CP / b*_nCP) = -1.01.

## I.6 Scoring against the pre-registered rules (mechanical)

| rule | CP | non-CP (interior basin) |
|---|---|---|
| G.3.1 valley: monotone | PASS | PASS |
| G.3.1 valley: slope in [1.2, 4.0] | **FAIL (0.96)** | PASS (3.23) |
| G.3.1 valley: RMSD spread < 0.06 | PASS | PASS |
| G.3.2 r0 in [0.8, 1.2] A | **FAIL (1.408)** | PASS (0.991) |
| G.3.2 b in [6, 20] | PASS (8.71) | PASS (10.95) |
| G.3.2 s not at the wall | PASS | PASS |
| G.3.3 ln(r0_CP / r0_nCP) > 0 | CONSISTENT: +0.351, 3.7 combined sigma | |
| G.3.4 softest direction: ln r0 and ln b components both > 0.3 | **FAIL (0.246)**; ln b / ln r0 = 3.9 | **FAIL (0.218)**; ratio 4.1 |
| G.3.4 softest direction: abs(ln s) component < 0.5 | PASS (0.098) | PASS (0.367) |
| G.3.5 nesting full <= b-only <= published | holds (0.078 <= 0.117 <= 0.568); full-vs-b-only gain 0.039, above the expected < 0.03 (in-sample, not evidence) | holds (0.093 <= 0.122 <= 0.129); gain 0.029 |
| G.3.6 sharp unrefit worse than linked | holds | holds |
| G.6(a) A1-lite valley check | **TRIPPED** (slope) | not tripped |
| G.6(b) optimum on an edge / budget-limited | not tripped | tripped for the raw optimum (H.1); H.2 took the interior basin |
| G.6(c) undetermined parameter (inf sigma) | not tripped | not tripped |
| G.6(d) in-sample ordering violated | not tripped | not tripped |
| F.12 H-BSSE flag: abs(ln r0 shift) > 0.15 | **FLAGGED** (0.351) | |
| F.12 H-BSSE flag: abs(ln b shift) > 0.3 | not flagged at the optima (0.229); 1.01 on the linked line at r0 = 1.00 A | |

Stop condition G.6(a) is therefore TRIPPED for the CP convention. Under G.6 that means "report immediately, no further stage"; this addendum is that report.

## I.7 Interpretation (PROVISIONAL, 2026-10-24, untested; separate from I.1-I.6)

* The CP valley sits a factor 2.0-2.9 below the paper's b (3.9-5.0 against 8.0-14.5); the non-CP aDZ valley lands on the paper's numbers to within
  0.3 at r0 <= 0.90 A and drifts above them with a steeper slope (3.23 vs 2.27). Two settings differ between the CP column and the paper at once
  (CP vs non-CP, and aDZ vs aTZ), so this table alone cannot say which one moves b.
* One reading (physics of the convention): the paper fitted at aTZ without CP, and MP2-V's MP2 half is the attenuated (terfc) operator, which was
  designed in the attenuated-MP2 line to absorb basis-set superposition error rather than to be counterpoise-corrected. Under CP at aDZ the method
  underbinds (published parameters MSE +0.545, Coulomb MP2 +0.352), and the fit restores attraction by lowering b (weaker short-range damping in the
  VV10 kernel, more nonlocal correlation). On this reading the low CP b* is VV10 absorbing aDZ basis incompleteness, not a property of the method.
  It matches the earlier de-risk finding (memory `derisk-cp-error-cancellation`) that bare attenuated SR-MP2 is over-repulsive when CP-corrected.
* Artifact hypothesis, stated with what separates it: if the CP shift is basis incompleteness, the CP valley must move UP toward the non-CP one at aTZ
  (F.12 H-BSSE: an artifact-absorbing parameter moves with the basis). If the CP-vs-non-CP gap persists at aTZ, then the non-CP parameters are absorbing
  BSSE, which is the F.12 artifact reading. No aTZ CP number exists, so neither is tested. The stage-3 aTZ overlap set (F.13) is the measurement that decides.
* The flatter CP slope (0.96) means the paper's valley SHAPE is not reproduced under CP at aDZ. The softest GN direction is mostly ln b in BOTH conventions,
  so at n = 8 the data constrain r0 better than b and do not show the paper's r0-b valley as the softest direction; at n = 8 that may be a sample-size effect.
* None of this bears on the published method as published: the paper's protocol (aTZ, S66, non-CP, frozen core) is a coherent convention for an attenuated
  method, and this addendum is a refit study of ferric's implementation of it, not a reproduction or a correction of the paper.

## I.8 Not decided here (for Matt)

* Matt's decision of 2026-10-10 (H.2) made the non-CP interior basin the stage-1 non-CP result, and F.14 plans stages 2-4 as non-CP only for cost. F.3 registers
  neither convention as primary. This addendum does NOT decide whether the non-CP plan stands in the light of I.6 (CP trips G.6(a) and the F.12 H-BSSE flag),
  whether a CP arm is carried into stage 2 (about 2x the cost), or whether a small aTZ CP + non-CP check (the I.7 discriminator) runs before stage 2.
  These are Matt's decisions.
* Whether G.6(a) tripping on the CP convention alone blocks stage 2 for both conventions, or only the CP arm, is not stated by G.6; it needs Matt's ruling.
