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
