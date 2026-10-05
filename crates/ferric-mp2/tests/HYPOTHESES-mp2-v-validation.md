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
* The terfc MP2 half depends on the row-119 reference (#275), which does not
  exist on main at the time of writing. It is NOT built here.

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
