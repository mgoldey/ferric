# Periodic boundary conditions: Python prototypes

These are small numpy/PySCF scripts written to work out how ferric should do
periodic (crystal) calculations **before** any Rust was written. They are kept
because each one is a short, runnable explanation of one idea. Read them in the
order below.

The scripts are small and slow on purpose. None of the timings here say
anything about ferric's speed.

## What you need

- Python with `numpy`, `scipy`, `pyscf` (2.x) and `pytest`.
- Nothing from ferric. PySCF's **molecular** integrals stand in for libint2.
  PySCF's **periodic** module (`pyscf.pbc`) is used only as an independent
  reference to check against.

Run everything from this directory:

```
cd prototypes/pbc
OPENBLAS_NUM_THREADS=1 python -m pytest -q test_prototype.py   # ~35 s; PBC_SLOW=1 adds 3 slow tests
```

## Reading order

1. **`pbc_gamma.py`**: Γ-point Hartree–Fock for a crystal, built from
   ordinary molecular integrals. The ideas it shows:
   - **Ewald splitting.** Every Coulomb interaction 1/r splits into a
     short-range part erfc(ωr)/r and a long-range part erf(ωr)/r. The
     short-range part is summed over nearby copies of the cell in real space.
     The long-range part is summed in reciprocal space.
   - **Analytic Fourier transforms of Gaussian pair densities**, via
     McMurchie–Davidson Hermite coefficients (`hermite_E`, `pair_ft`). This is
     the one genuinely new integral a molecular code needs for crystals.
   - **The G = 0 term.** Each piece is infinite on its own, but they cancel for
     a neutral cell. The code shows exactly which constant π/(ω²Ω) gets
     subtracted where.
   - **The exchange divergence.** `exxdiv="none"` versus `"ewald"` (the
     Madelung correction). Only `"ewald"` approaches the isolated-molecule
     answer quickly: the error falls as a⁻³ rather than 1/a (see
     `run_molecular_limit.py`).
   - **The Γ point is a molecule.** At Γ the whole calculation reduces to an
     ordinary molecular SCF on lattice-summed S, h and ERIs.
2. **`run_h2.py`, `run_h2_oracle.py`, `run_triclinic_p.py`,
   `run_h2_exxdiv.py`**: checks against PySCF's periodic code. Energies agree
   to about 1e-12 Ha.
3. **`run_molecular_limit.py`**: a check that uses no PySCF periodic code. As
   the box grows, the periodic energy should approach the molecular energy.
   The residual should go as c₃/a³, with c₃ = −(4π/3)σ², where σ² is the
   spread of the occupied orbital. It does, to 6e-6 relative.
4. **`pbc_gdf.py`**: range-separated Gaussian density fitting (RS-GDF), which
   is how J and K scale past toy systems. Things to look for:
   - how charged auxiliary functions are handled at G = 0;
   - why the metric is eigendecomposed with a linear-dependence cut rather
     than Cholesky-factored;
   - the exactness check, which uses an auxiliary basis that spans every pair
     product exactly, so the fit must be exact.
5. **`test_prototype.py`**: the tests. Several exist to catch a specific,
   plausible bug (a sign flip, a missing Madelung term, the G = 0 term applied
   to only one side); the comments say which one.

## Design notes

- **`FINDINGS.md`**: the lab notebook: what was measured, when, and what it
  meant. Measurements and interpretation are kept apart. Some early
  conclusions were later corrected; the corrections are dated and left in
  place so the reasoning can be followed. This file is a snapshot of the
  working copy.
- **`stage1-design.md`**: how the prototype was mapped onto ferric's Rust
  crates.

## Lessons worth knowing

- **A sum rule cannot check a parameter.** libint2 2.7.2's
  `erf_nuclear`/`erfc_nuclear` integrals effectively use 2ω instead of ω. The
  check "erf part + erfc part = plain nuclear attraction" still passes to
  1e-13, because the sum does not depend on ω. Only a closed-form check on a
  single Gaussian caught it. `libint-erf-nuclear-bug/` has the reproducer.
- **A code can agree with itself and still be wrong.** Matching PySCF to 1e-12
  proves the Fourier-transform kernel correct, not the physics convention,
  because both codes use the same algorithm and the same convention. The
  box-limit comparison against a molecular calculation is the check that could
  have caught a convention error.
- **PySCF 2.13 RSJK has a bug for 3-D cells.** It uses the molecular nuclear
  repulsion instead of the Ewald value, because of an `and`/`or` precedence
  slip at `pbc/scf/hf.py:760`. The eigenvalues are right; the total energy is
  off by a constant.

## Where this went

The Rust implementation is in `crates/ferric-pbc`. Its tests use the numbers
these prototypes produced.
