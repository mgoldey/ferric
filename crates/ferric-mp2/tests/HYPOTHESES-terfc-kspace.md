# Issue #275: terf/terfc integrals and SCS-MP2(2terfc) vs a Fourier-space reference

**Written BEFORE any measurement on this branch.** Dated 2026-10-05, against
`origin/main` @ `c28cfb7f`. Every number below is a PREDICTION.

## The construction

Reference: (P|K|μν) = (2π)⁻³ ∫d³k K̂(k) χ̂_P(k)* ρ̂_μν(k), with PySCF `ft_ao` /
`ft_aopair` on a radial Gauss–Legendre × Lebedev k-grid and
K̂_terf(k) = (4π/k²)·exp(−k²/4w²)·cos(k r₀). terfc = PySCF analytic Coulomb
`int3c2e`/`int2c2e` − terf(k-space). Shared with ferric: the basis JSON, the
geometry, the Coulomb half of terfc (PySCF's own Coulomb vs ferric's MD
Coulomb pass; independent codes). NOT shared: the MD recursion, the Hermite
sign conventions, the G_{m,0}(S,s) tables and their interpolation, the series
and asymptotic branches.

## H-REF (reference harness)

* If the k-space harness is right: with K̂_erf at ω = w it reproduces PySCF
  `with_range_coulomb(w)` int3c2e/int2c2e to ≤1e-10 once the grid converges.
  Because K̂_erf and K̂_terf both carry exp(−k²/4w²) (w ≈ 0.36–0.50 Bohr⁻¹),
  the radial range needed is small (k ≲ 8); the angular order is driven by
  k·|R_P − R_C| ≲ 30 rad plus l ≤ 7, so I expect Lebedev ≈ 590–2030.
* If broken: a missing (2π)⁻³ or 4π is an O(1) ratio in EVERY class; a
  cart/spherical mismatch fails d/f only; a wrong e^{±ik·r} convention is
  INVISIBLE (Re[A* B] = Re[A B*]), so it is not something the anchor can or
  needs to catch.

## H-PHYS (ferric is right)

ferric's terf and terfc agree with the reference element-wise at the table
interpolation floor, predicted 1e-11–1e-9 absolute, with NO dependence on
shell class (ss, sp, pd, df... all at the same floor relative to magnitude).

## H-ART (ferric's MD pass is broken in a way the existing tests cannot see)

A shared MD error (ket-side (−1)^(t+u+v) sign, an l > 0 normalization, a
Hermite index slip) is invisible in (ss|ss)-type checks and in terf + terfc =
Coulomb (both pieces share `compute_cart_eri3`). Signature: ss classes pass at
the floor, p/d/f classes miss by O(0.01–1) relative. A table-interpolation
error instead shows in EVERY class, ss included, scaling with magnitude.
These two predictions differ, so the per-class comparison distinguishes them.

## Energies

* E_OS(r₀₁), E_SS(r₀₁), E_SS(r₀₂) vs numpy DF-MP2 on the reference tensors
  (terfc metric, Cholesky, PySCF exact-J/K RHF orbitals): predicted ≤1e-10 Ha,
  the same floor as the attenuated-MP2 row (5.9e-12) plus the integral error
  propagated (≲1e-10).
* If ferric fitted with a Coulomb metric: miss ~1e-5 Ha (the size of the RI
  error). If r₀ were taken in Å: miss ~mHa.
* Negative controls, predicted to MISS by ≥1e-4 Ha: terfc(r₀₁) vs the r₀₂
  reference; erfc(ω = 1/(r₀√2)) vs terfc(r₀).
* The terfc metric (P|terfc|Q) is positive definite for cc-pvdz-ri and
  aug-cc-pvdz-rifit (smallest eigenvalue small but > 0); if it is not, report,
  do not regularize.
