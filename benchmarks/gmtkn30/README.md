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

methods: `rhf`, `rimp2`, `ccsd_t`. The (T) triples correction itself is now a
cheap per-triple-block streaming kernel (2026-07-19 rewrite, see
`docs/VALIDATION.md`) and is no longer the memory bottleneck. The remaining
bottleneck is the **spin-orbital CCSD step's dense VVVV block** (`ccsd.rs`,
unchanged by the (T) rewrite): at cc-pVDZ, even the smallest ACONF system
(butane) needs ~24 GB peak RSS — more than this project's usual 23-24 GB
shared dev box has available, so a cc-pVDZ CCSD(T) run on ACONF was not
attempted. At **6-31G**, CCSD(T) is comfortably tractable in memory (a few
GB peak) but still genuinely O(N^7)-expensive in wall time: ~9-10 min per
butane conformer, ~43 min per pentane conformer on this box — see the
2026-07-19 result below for what was actually run and why a full 15-reaction
sweep wasn't attempted in one session.

## Result (CCSD(T) / 6-31G, ferric, 2026-07-19 — partial, streaming-rewrite validation)

Validates the 2026-07-19 per-triple-block streaming CCSD(T) rewrite (#89) on
real ACONF systems, beyond the synthetic H2O/butane-STO-3G demos it shipped
with. **Not a full benchmark run** — stopped deliberately after 3 of 18
conformers to conserve shared-box CPU once enough evidence had accumulated;
see scope note below.

| Conformer | Basis | RHF (Ha) | RI-MP2 total (Ha) | CCSD(T) total (Ha) | Wall time |
|---|---|---|---|---|---|
| B_G (butane) | 6-31G | −157.232793 | −157.607696 | −157.670854 | 9.9 min |
| B_T (butane) | 6-31G | −157.234361 | −157.608915 | −157.672048 | 8.9 min |
| P_TT (pentane) | 6-31G | −196.252685 | −196.719776 | −196.797025 | 43.0 min |

All three conformers converged cleanly with no memory or numerical issues —
this is the first time ferric's CCSD(T) has run on a system this size
end-to-end. B_G/B_T form one of the 15 real ACONF reactions (literature
CCSD(T)/CBS relative energy 0.598 kcal/mol, `run_aconf.py`'s `REACTIONS`
list); at 6-31G (nowhere near CBS), ferric gives **B_T − B_G = −0.749
kcal/mol (CCSD(T)) / −0.765 kcal/mol (RI-MP2)** — the wrong *sign* relative
to the CBS reference, but ferric's own CCSD(T) and RI-MP2 numbers agree
tightly with each other (−0.749 vs −0.765), which is the more informative
consistency check here: this is very plausibly an ordinary small-basis
artifact on a genuinely small (~0.6 kcal/mol) conformer energy gap, not a
CCSD(T)-specific bug — RI-MP2 (already validated elsewhere, unrelated to
this rewrite) shows the same sign at the same basis. Not enough data to
separate "6-31G is just too small for this particular reaction" from
anything more interesting; a cc-pVDZ or larger-basis rerun would resolve it
but hits the CCSD memory ceiling described above.

**Scope note**: hexane conformers (12 of the 18 ACONF conformers) were not
attempted — extrapolating the butane→pentane wall-time growth puts each
hexane conformer at roughly 1.5-2+ hours, making a full 15-reaction sweep
impractical in a single session, and the run was stopped after 3 conformers
once they had demonstrated the streaming rewrite works correctly on
real, non-trivial systems (the actual validation goal) rather than running
the full set to exhaustion. A complete ACONF/CCSD(T) sweep, if wanted later,
would need either a much larger basis-appropriate memory budget (for
cc-pVDZ) or acceptance of multi-hour-per-conformer wall time at 6-31G.

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
on dispersion-bound dimers (A24 subset)" — was subsequently run (see `../a24-subset/README.md`): at (a)DZ the falsifier
fired, but at aug-cc-pVTZ the criterion is MET marginally (B 0.139 vs MP2
0.143 kcal/mol at ω=0.42) via the π-overbinding mechanism.
