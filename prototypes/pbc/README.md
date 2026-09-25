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
5. **`pbc_mp2.py`** (plus the `run_mp2_*.py` scripts): Γ-point MP2 built on
   the periodic density-fitting tensor B. The lesson is about the orbital
   energies in the denominators:
   - with the Madelung-shifted occupied energies, the result approaches
     molecular MP2 as a⁻³, with a coefficient that can be predicted in
     advance;
   - with unshifted energies it converges only as 1/a;
   - in small boxes the wrong choice looks better. `run_mp2_box_limit.py`
     shows why a single small cell would pick it.
6. **`pbc_rpa.py`** (plus the `run_rpa_*.py` scripts): Γ-point direct RPA
   from the same tensor B. Three independent routes to one number are shown
   side by side: the plasmon formula, the ring-CCD Riccati equation and
   frequency quadrature. The second-order term is checked against direct MP2,
   which pins the factor of 4 and the 1/(2π). There is also a numerical
   detail: computing ln det(1+Π) − tr Π directly loses accuracy at high
   frequency, so the code sums log1p(λ) − λ over the eigenvalues instead.
   `run_rpa_c3_prediction.py` predicts the a⁻³ box-limit coefficient from a
   molecular calculation with an added harmonic kernel.
7. **`pbc_uhf.py`** (plus the `run_uhf_*.py` scripts): open-shell Γ-point
   UHF.
   - The Madelung correction needs no spin factor: it is the same v_M applied
     to each spin's density.
   - There is a new SCF trap. Under `exxdiv="ewald"` every occupied level
     drops by v_M, so a state that breaks the aufbau principle without the
     correction can become self-consistent with it.
   - `run_uhf_guess.py` shows the fix: converge without the correction
     first, then turn it on.
8. **`pbc_lmp2.py`, `pbc_supercell.py`** (plus the `run_lmp2_*.py` scripts):
   local MP2 in a Γ-point supercell.
   - **Localization.** Ordinary Boys localization breaks translation
     symmetry, because the position operator isn't periodic. The periodic
     Resta/Berghold functional, built on ⟨e^{ib·r}⟩, keeps equivalent
     molecules equivalent to 1e-15.
   - **A hidden coupling.** At Γ every orbital pair carries a
     distance-independent coupling, −(4π/Ω)μμ. It is the missing q = 0
     term of the equivalent k-mesh, a finite-size artifact.
   - **Why it matters.** It makes an integral threshold keep all N² pairs
     until the supercell is large.
   - **The fix.** `run_lmp2_uniform.py` shows how removing it restores
     locality.
9. **`pbc_ump2.py`** (plus the `run_ump2_*.py` scripts): open-shell MP2 and
   RPA.
   - **The per-spin convention is measured, not assumed.** Both spins'
     occupied levels get the same v_M. Two plausible wrong shifts (v_M/2 per
     spin, or shifting alpha only) leave a 1/a error in the box limit.
   - **Test systems must exercise every spin block.** The triplet's
     same-spin block is identically zero, so a 5-H doublet with all three
     blocks nonzero is used as well.
10. **`pbc_dft.py`** (plus the `run_dft_*.py` scripts): Γ-point Kohn–Sham
    DFT on a periodic Becke/SSF grid.
    - **Grid construction.** Atom-centred grids are weighted against
      neighbouring image atoms, and the atomic orbitals are summed over
      lattice images.
    - **Partition error dominates.** The main source of grid error is the
      Becke space partition, not the periodic machinery.
    - **Madelung on exact exchange only.** For hybrids the Madelung
      correction applies only to the exact-exchange fraction, and the box
      limit confirms it.
11. **`pbc_uks.py`** (plus the `run_uks_*.py` scripts): spin-polarized UKS.
    - **A blind spot in energy tests.** Two plausible bugs, dropping the
      spin cross term of the density gradient or giving beta alpha's
      potential, pass every energy test. Only a finite-difference check on
      the XC potential catches them.
    - **The trap in hybrids.** The Ewald trap criterion scales with the
      exact-exchange fraction (gap ≥ α·v_M).
12. **`pbc_kpts.py`** (plus the `run_kpts_*.py` scripts): k-point RHF.
    - **The key check.** An N₁×N₂×N₃ k-mesh gives exactly the Γ-point energy
      of the matching supercell, because the k-shifted G vectors together
      form the supercell's reciprocal lattice.
    - **Why three k-points.** A mesh needs at least three k-points along
      some axis before the sign of the Bloch phase becomes visible.
13. **`pbc_kgdf.py`, `pbc_kcorr.py`, `pbc_kuhf.py`**: k-point density
    fitting, MP2/dRPA and UHF.
    - **Exact supercell equivalence.** A k-mesh reproduces the Γ-point
      supercell exactly for fitted HF, correlation and open-shell HF. For
      fitting, the kept auxiliary functions per q add up to the supercell's
      count.
    - **The per-k aufbau bug.** It is invisible on every system with the same
      number of occupied orbitals at each k. The "zchain" case, with both α
      electrons at Γ, is there to catch it.
14. **`test_prototype.py`**: the tests. Several exist to catch a specific,
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

## Forces, stress and later work

Read these after the energy prototypes; each has its own FINDINGS iteration.

| File | FINDINGS | What it is |
|---|---|---|
| `pbc_grad.py` | Iteration 16 | Γ-point RHF forces, dense J/K |
| `pbc_grad_open.py` | Iteration 17 | UHF, RKS and UKS forces, with grid response |
| `pbc_grad_gdf.py` | Iteration 18 | RS-GDF forces, including the Loewner metric term |
| `pbc_stress.py` | Iteration 19 | Γ-point stress tensor (frozen G/image index sets) |
| `pbc_grad_ro.py` | Iteration 20 | ROHF/ROKS forces (W equals the UHF form at convergence) |
| `pbc_kgrad.py`, `pbc_kgrad_gdf.py` | Iteration 21 | k-point RHF/UHF forces (force on one supercell copy) |
| `pbc_grad_ecp.py` | Iteration 22 | periodic ECP force term (partial; PySCF ECP derivative defects) |
| `roks_replica.py`, `run_roks_trap*.py` | ROKS PBE0 CI diagnosis | numpy replica of ferric's ROHF/ROKS loop |
| `run_kcorr_head_anomaly.py` | q = 0 head investigation | why the head removed ~100% of MP2's finite-size term |
| `pbc_lindep.py`, `pbc_ecp.py` | Iterations 13–14 | linear dependence; periodic ECPs |

The `run_*` scripts are the anchors and oracles for each; their docstrings state
the predictions made before running them.

## Where this went

The Rust implementation is in `crates/ferric-pbc`. Its tests use the numbers
these prototypes produced.
