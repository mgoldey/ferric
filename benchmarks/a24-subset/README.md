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
