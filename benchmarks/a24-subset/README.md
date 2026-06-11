# A24-subset: RS-MP2-RPA decisive experiment (2026-06-10)

The pre-registered success criterion from
`docs/superpowers/specs/2026-06-09-sr-mp2-lr-rpa-design.md`:

> B at one fixed ω (0.2–0.3 Bohr⁻¹) beats plain MP2 MAE on the
> dispersion-bound dimers without per-system fitting.

**Verdict (chronological — see sections below): falsified at cc-pVDZ and
aug-cc-pVDZ** (B never beats MP2; basis-incompleteness underbinding dominates),
**criterion MET marginally at aug-cc-pVTZ** (B 0.139 vs MP2 0.143 at ω=0.42,
the predicted π-overbinding mechanism). The naive sum (A) is worse everywhere.

## Setup

- Systems: A24 #2 (H₂O·H₂O), #5 (NH₃·NH₃), #14 (C₂H₄·C₂H₄ C2v), #19 (CH₄·CH₄ D3d).
  Geometries + CCSD(T)/CBS refs from the psi4 A24 database (Řezáč & Hobza).
- cc-pVDZ / cc-pVDZ-RI, full-rank dRPA (trunc_thresh = 0), ω in Å⁻¹.
- Counterpoise via ghost atoms (`@`-notation); non-CP values in parentheses in
  the data file. 4 dimers × 5 ω × 5 fragments = 100 runs (`run_a24.py`).

## CP MAE vs CCSD(T)/CBS (kcal/mol)

| ω (Å⁻¹) | MP2 | naive A | Δ-form B |
|---|---|---|---|
| 0.1 | 0.847 | 0.851 | 0.847 |
| 0.2 | 0.847 | 0.882 | 0.847 |
| 0.3 | 0.847 | 0.956 | 0.849 |
| 0.42 | 0.847 | 1.053 | 0.859 |
| 0.6 | 0.847 | 1.108 | 0.893 |

## Reading

At CP/cc-pVDZ every correlated method *underbinds* these dimers (basis
incompleteness dominates the MAE: water −4.04 vs ref −5.01). The Δ-correction
(dRPA[erf] − dMP2[erf]) is screening — it strictly *removes* long-range binding
— so on systems that already need more binding it can only hurt. The regime
where screening should win is where MP2's tail genuinely overbinds (π-stacks
near the basis-set limit, e.g. benzene dimer at aug-cc-pVTZ); that is outside
DZ scope and would be the only justified revival experiment.

Combined with ACONF (`../gmtkn30/README.md`): no fixed-ω win anywhere tested.
Per the spec's falsifier clause, the method is parked and documented; the code
(driver, limit tests, component diagnostics, erf-safe metric, ghost atoms)
stays — it is validated infrastructure.

## Stacked systems + formulation T (2026-06-10, evening)

Stacked/saddle subset (A24 #22 ethene·ethyne, #23 ethene·ethene D2h, #24
ethyne·ethyne D2h — the MP2-overbinding regime), CP, aug-cc-pVDZ + RI-JK,
comparing MP2 / Δ-form B / coupled-rings T (`stacked_adz_mp2_B_T.txt`):

| ω (Å⁻¹) | MP2 MAE | B MAE | T MAE |
|---|---|---|---|
| 0.2 | 0.071 | 0.071 | 0.159 |
| 0.3 | 0.071 | 0.074 | 0.355 |
| 0.42 | 0.071 | 0.104 | 0.619 |
| 0.6 | 0.071 | 0.245 | 0.888 |

Findings: (i) on the one genuinely MP2-overbound system (#24, −0.031), B at
ω=0.3 improves it (−0.021) but T overshoots past the reference already at
ω=0.2 (+0.086) — T's mixed-ring correction is first-order in v_lr where B's
pure-LR rings are third-order, so **B and T do not share an ω scale**; T's
operating window, if any, is ω ≲ 0.1–0.15 Å⁻¹. (ii) At (a)DZ with CP the
dominant residual is basis-incompleteness underbinding, which any screening
correction worsens. The decisive test for T is stacked systems near the basis
limit (aug-cc-pVTZ run queued).

## aug-cc-pVTZ: success criterion MET (marginally) (2026-06-10, night)

Bound 4-dimer subset, CP, aug-cc-pVTZ (`a24_rsmp2rpa_atz.txt`, `results_atz.json`).
CP MAE vs CCSD(T)/CBS (kcal/mol):

| ω (Å⁻¹) | MP2 | naive A | Δ-form B |
|---|---|---|---|
| 0.1 | 0.145 | 0.142 | 0.145 |
| 0.2 | 0.166 | 0.155 | 0.166 |
| 0.3 | 0.143 | 0.275 | **0.142** |
| 0.42 | 0.143 | 0.452 | **0.139** |
| 0.6 | 0.143 | 0.591 | 0.176 |

First fixed-ω win, at the 2012 erfc-optimal ω=0.42 Å⁻¹ = 0.222 Bohr⁻¹, inside
the pre-registered window. Mechanism as designed: at the basis limit the ethene
dimer overbinds at MP2 (−1.197 vs −1.110) and B halves that error (−1.150 at
ω=0.42) while leaving the H-bonded systems nearly untouched. The effect is
small (~3% of MAE) because only 1 of 4 systems is in the overbinding regime —
stacked π systems at aTZ are the growth case (run in progress).

Caveats (read before quoting):
1. **SCF reproducibility noise on the ω=0.1/0.2 rows.** The ethene fragments
   (368 bf) converged to SCF points differing by 2–4e-5 Ha between runs
   (loose-convergence DIIS path dependence); the CP RHF wobbles ±0.03 kcal/mol
   on those two rows — excluded from conclusions pending rerun. Rows 0.3–0.6
   share identical SCF solutions, and the B-vs-MP2 comparison within every row
   uses the same SCF, so the win rows are internally consistent. Future sweeps
   should pin `[scf] energy_conv`.
2. **This sweep used exact 4-index J/K SCF, not RI-JK** (runner patch race);
   slower but strictly more accurate. The aDZ rows and the stacked-aTZ run use
   RI-JK (def2-universal-jkfit).
3. **All-electron correlation** (frozen_core = 0) vs frozen-core CCSD(T)/CBS
   references — a small protocol mismatch, common to every method column, so it
   largely cancels in the method-vs-method comparison.

## aug-cc-pVTZ stacked: formulation T validated (2026-06-10, late)

#23/#24, CP, aug-cc-pVTZ + RI-JK (`stacked_atz_mp2_B_T.txt`; corrected MAEs —
the runner's printed MAE divides by 3 with 2 systems): at the basis limit both
stacks overbind at MP2 (−0.110/−0.258) and **T at ω=0.2 Å⁻¹ cuts the MAE 2.5×
(0.075 vs MP2 0.184)**, ethene stack to −0.010. B moves ≤0.010 (third-order
correction too weak — the ω-scale law of §5 of the methodology doc). T's
operating window confirmed at ω≈0.2 Å⁻¹; no damping needed (required c ≈ 1).
Verdict: **B for mixed bound systems at ω≈0.42; T for π-stack overbinding at
ω≈0.2** — two regimes, one derivation. Scale-up test (benzene dimer) next.

## Stacked systems at aug-cc-pVTZ: T's regime found (2026-06-10, midnight)

A24 #23 (ethene·ethene D2h) + #24 (ethyne·ethyne D2h), CP, aug-cc-pVTZ +
RI-JK (def2-universal-jkfit), MP2/B/T (`stacked_atz_mp2_B_T.txt`,
`stacked_atz.json`). NOTE: the .txt's printed MAE/RMSE divide by a hardcoded
N=3 from the aDZ runner — corrected N=2 values:

| ω (Å⁻¹) | MP2 | Δ-form B | coupled T |
|---|---|---|---|
| 0.1 | 0.184/0.198 | 0.184/0.198 | 0.170/0.185 |
| 0.2 | 0.184/0.198 | 0.183/0.197 | **0.075/0.100** |
| 0.3 | 0.184/0.198 | 0.174/0.189 | 0.124/0.141 |

(MAE/RMSE, kcal/mol.) At the basis limit both stacks overbind at MP2
(#23 −0.110, #24 −0.258) and **T at ω=0.2 Å⁻¹ cuts the error 2.4×**
(#23: −0.010 — essentially exact; #24: −0.141). B barely moves at these ω
(its pure-LR ring correction is 3rd-order in v_lr). T's optimum sits at
ω≈0.2 Å⁻¹, below B's 0.42 — confirming the separate-ω-scales reading from
aDZ, now with the sign working *for* T instead of against it.

Caveats: N=2, both systems same error sign, both are saddle-point stacked
geometries (not minima); all-electron correlation vs frozen-core refs.
Next falsifiable step: a real π-stack minimum at aTZ (S22 benzene dimer PD)
at ω=0.2, plus a check that ω=0.2 T does not damage the bound-dimer set.
