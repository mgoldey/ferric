# GMTKN30 / ACONF benchmark (ferric)

ACONF: 18 alkane conformers (butane/pentane/hexane), 15 conformer-energy
reactions with W1h-val **CCSD(T)/CBS** references (Gruzman, Karton, Martin,
*J. Phys. Chem. A* 2009, 113, 11974). Geometries from the GMTKN30 database
(Goerigk & Grimme, *JCTC* 2011, 7, 291), Bonn mirror.

## Run

```
OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=6 \
  python benchmarks/gmtkn30/run_aconf.py rimp2 cc-pvdz
```

methods: `rhf`, `rimp2`, `ccsd_t` (ccsd_t only feasible for ~H2O-sized systems —
the dense (T) kernel needs ~TB RAM at butane scale; see docs/VALIDATION.md).

## Result (RI-MP2 / cc-pVDZ, ferric, 2026-06-05)

15/15 reactions vs CCSD(T)/CBS: **MAE 0.129, MD +0.094, RMSD 0.187, MAX 0.512 kcal/mol**.

cc-pVDZ RI-MP2 reproduces ACONF conformer ordering and sub-tenth-kcal MAE on the
small reactions; error grows on the most gauche-strained conformers (rxn 15,
H_ttt→H_x+g-x+, +0.51) — the expected DZ basis + MP2-dispersion limitation, not
a code error. A larger basis (cc-pVTZ/aug-cc-pVTZ) would tighten it.

## Result (RS-MP2-RPA / cc-pVDZ, ferric, 2026-06-10)

ω-scan (Δ-form, `rs_mp2_rpa`): 15/15 reactions vs CCSD(T)/CBS, cc-pVDZ / cc-pVDZ-RI,
full-rank dRPA (`trunc_thresh=0`), serial legs, OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=6.

| Method | ω (Å⁻¹) | MAE | MD | RMSD | MAX (kcal/mol) |
|---|---|---|---|---|---|
| RI-MP2 | — | 0.129 | +0.094 | 0.187 | 0.512 |
| RS-MP2-RPA (Δ-form) | 0.1 | 0.129 | +0.094 | 0.187 | 0.512 |
| RS-MP2-RPA (Δ-form) | 0.2 | 0.129 | +0.094 | 0.187 | 0.512 |
| RS-MP2-RPA (Δ-form) | 0.3 | 0.130 | +0.097 | 0.189 | 0.517 |
| RS-MP2-RPA (Δ-form) | 0.42 | 0.135 | +0.113 | 0.201 | 0.545 |
| RS-MP2-RPA (Δ-form) | 0.6 | 0.187 | +0.186 | 0.265 | 0.672 |

The Δ-form reproduces RI-MP2 exactly in the ω→0 limit (as the unit tests guarantee) and
degrades monotonically with ω; at ω=0.6 Å⁻¹ the MD is +0.186 kcal/mol (under-binding).
Alkane conformer energetics are short-range-dominated: MP2's long-range direct ring term
is adequate here, and replacing it with screened dRPA only removes correlation binding.
There is no fixed-ω win on ACONF; errors at ω ≤ 0.3 Å⁻¹ are negligible relative to
the RI-MP2 baseline.

The spec's pre-registered decisive success criterion — "RS-MP2-RPA beats plain MP2 MAE
on dispersion-bound dimers (A24 subset)" — has **not yet been run**; that experiment
remains open.
