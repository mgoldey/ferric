# A24-subset: RS-MP2-RPA decisive experiment (2026-06-10)

The pre-registered success criterion from
`docs/superpowers/specs/2026-06-09-sr-mp2-lr-rpa-design.md`:

> B at one fixed ω (0.2–0.3 Bohr⁻¹) beats plain MP2 MAE on the
> dispersion-bound dimers without per-system fitting.

**Verdict: FALSIFIED — the method is parked.** The Δ-form (B) never beats MP2
at any ω on the four dispersion-bound A24 dimers; it ties MP2 in the ω→0 limit
and degrades monotonically. The naive sum (A) is worse and degrades faster.

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
