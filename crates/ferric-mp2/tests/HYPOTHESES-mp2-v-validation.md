# MP2-V validation (issue #273): hypotheses recorded before measuring

Dated 2026-10-05. Written before any reference number was generated.

## What is shared and what is independent

* Undamped VV10 vs PySCF `_vv10nlc`: ferric's kernel (`vv10.rs`) is a port of
  PySCF's `_vv10nlc` (same `beta + F/2` split, same `1e-8` density threshold).
  Agreement therefore checks the AO-on-grid density, the grid, the threshold and
  the summation; it does NOT independently check the VV10 formula.
* The numpy kernel in `gen_mp2_v.py` is written from Vydrov & Van Voorhis, JCP
  133, 244103 (2010) (kappa = b v_F^2 / omega_p, omega_0 = sqrt(omega_g^2 +
  omega_p^2/3), Phi = -3/(2 g g' (g + g'))). That is the independent check of the
  formula, and the damped reference is built on it (PySCF has no damping).
* The density fed to ferric's E_nl is PySCF's exact-J/K RHF density permuted into
  ferric AO order, evaluated on the grid by ferric's own AO code. The grid is
  rebuilt in numpy from PySCF primitives (TA-M4 radial, Lebedev, ferric's Becke
  size adjustment). Shared with ferric: the grid DEFINITION (by design; the
  issue says "on ferric's grid").
* The erfc-attenuator control reuses `gen_attenuated_mp2.py`'s numpy RI-MP2 on
  PySCF `with_range_coulomb(-omega)` integrals; independent of ferric's
  integrals.
* The terfc MP2 half is compared against the row-119 reference
  (`scs_mp2_2terfc/*.json`, k-space terfc integrals), reused, not rebuilt. Its
  terfc definition is copied from ferric's `shim.cc`, so it does not test the
  paper's omega = 1/(r0 sqrt 2); only the r0 -> 0 / infinity limits do.

## Hypotheses

H1 undamped E_nl, ferric on the PySCF density vs `_vv10nlc`.
  Real: |d| < 1e-11 Ha (same arithmetic; the 40 Bohr cell-list cutoff drops no
  active pair, since active points (rho >= 1e-8) of these systems span < 40 Bohr).
  Broken AO permutation / basis normalisation: |d| >= 1e-4. Broken grid replica
  (radii, Becke adjustment, Lebedev normalisation): |d| >= 1e-7.
H2 numpy-from-paper kernel with factor 1 vs `_vv10nlc`: |d| < 1e-12 relative-
  scale summation noise. A transcription slip in kappa/omega_0/beta: >= 1e-5.
H3 damped E_nl, ferric vs numpy at r0 = 1.00 A (b 11.0) and 0.85 A (b 8.0):
  |d| < 1e-11. Damped minus undamped: >= 1e-4 Ha on every system (the damping is
  live). A r0 unit slip (A vs Bohr) moves E_nl by >= 1e-4; inverting the factor
  moves it by O(E_nl).
H4 erfc-attenuator E_c through `att_mp2_vv10` (frozen core 1 for first-row) vs
  numpy RI-MP2 at omega = 1/(r0 sqrt 2) Bohr^-1: |d| <= 1e-10 (the attenuated row
  measured 5.9e-12). omega built from r0 in A instead of Bohr: miss >= 1e-2. Frozen
  core ignored: miss >= 1e-3.
H5 full chain (ferric's own RHF density) E_nl vs the reference: |d| ~ 1e-9 or
  less (first-order in the SCF density error, density_conv 1e-9).
H6 UHF singlet collapsed onto RHF through `u_att_mp2_vv10` reproduces H1, H3, H4
  at the SCF-collapse floor (~1e-9).
H7 grid-mismatch negative control: ferric on a (75, 110) grid with the same
  density misses the (50, 50) reference by >= 1e-7 Ha, i.e. far above the H1 bar.
  If instead it agrees, the H1 bar is not measuring the grid match.

## Terfc MP2 half (H8): written ALONGSIDE the measurement, not before it

The first measurement (max |d| 2.5e-12 Ha) and the mutants below were run before
this section was written; it records the reasoning honestly, it is not a
pre-registration.

H8 terfc E_c (and E_OS, E_SS) through `att_mp2_vv10` with `mp2_v_terfc_atz` and
  frozen core 1 vs the row-119 numpy reference at r0 = 1.00 A.
  Physics/wiring right: |d| at the integral-quadrature floor (<= ~1e-11 Ha;
  measured 2.5e-12). Artifact/wiring broken, each expected to miss by far more:
  * terfc replaced by erfc in `mp2_operator`: ~2e-2 Ha (measured 2.2e-2).
  * r0 passed in Angstrom instead of Bohr (omega from the wrong r0): ~2e-2
    (measured 2.6e-2).
  * Coulomb instead of terfc fitting metric: ~3e-4 (measured 3.0e-4).
  * frozen core dropped: ~1e-3 (the E_c(erfc) mutant already shows 2.3e-3; the
    reference block is selected by frozen_core so a drop misses it).
  Not distinguished: a misreading of omega = 1/(r0 sqrt 2) that ferric and the
  reference share (both take the definition from `shim.cc`); a common
  misreading is invisible to this comparison.
