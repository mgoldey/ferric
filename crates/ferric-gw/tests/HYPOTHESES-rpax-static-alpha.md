# Hypotheses: RPAx static polarizability validation (issue #274)

Recorded 2026-10-05, BEFORE any measurement, on `fix/p2-274` off main
`c28cfb7f`. Results are appended below the line at the end; the hypotheses
above it are not edited after measuring.

## What is being validated

`ferric_gw::bse::run_rpax_static_polarizability`: a statically screened
RPAx (BSE-like) kernel on a restricted reference,

    (A+B) = Δε δ + 4(ia|jb) − (ab|W|ij) − (ib|W|aj)
    (A−B) = Δε δ + (ib|W|aj) − (ab|W|ij)
    α_ij  = 4 μ_iᵀ (A−B)[(A−B)(A+B)]⁻¹ μ_j

with W(0) from the reference's own static PDEP dielectric (unshifted
mean-field energies), and `scissor` added to every virtual energy on the
A±B diagonal only.

## A pre-measurement algebraic observation (a prediction, not a result)

The code solves `sysm t = (A−B) μ` with `sysm = (A−B)(A+B)`, so
`t = (A+B)⁻¹ (A−B)⁻¹ (A−B) μ = (A+B)⁻¹ μ`. The static α therefore depends
ONLY on A+B; A−B cancels exactly (up to the conditioning of the solve).

PREDICTION: the issue's mutation "swap the (ab|W|ij) and (ib|W|aj) roles in
A−B" is an IDENTITY on α. It will SURVIVE every α comparison (the change in α
will sit at roundoff, ~1e-12 relative, not at the bar). Detecting it needs an
observable of A−B itself. The validation therefore also compares the minimum
eigenvalue of A−B (and of A+B) against the numpy reference, through a
`#[doc(hidden)]` hook that does not change the public function.

## Physics vs artifact hypotheses

1. Screened α vs the independent numpy screened α (same DF aux, same kind of
   reference SCF, full-rank W).
   - Real (kernel right): agreement limited by SCF convergence and integral
     engines, ≲1e-8 relative per element.
   - Broken: a wrong Coulomb factor (2 vs 4) moves α by tens of percent;
     W → v moves α to the bare control (expected several percent at HF);
     a wrong w_red or Π spin factor moves α by a fraction of the screening
     effect. All ≫ 1e-8. X ≠ Y, so the comparison discriminates.
2. Bare limit (W → v) at an HF reference equals CPHF α.
   - Real: numpy bare α on a DF-RHF equals PySCF DF-CPHF
     (`scf.cphf.solve` + `mf.gen_response(hermi=1)`, same aux) to ≤1e-9
     relative; ferric's bare hook equals numpy bare on exact-J/K orbitals to
     ~1e-8 relative.
   - Broken numpy (layout, factor, exchange index order): misses CPHF by
     O(1%) or more. This anchors the numpy kernel to an independent response
     code BEFORE any ferric comparison.
3. The documented physics verdict: at PBE/cc-pVDZ, scissor 0.36 Ha, water
   iso ≈ 5.2 a.u. (46% below DOSD 9.64).
   - If that deficit is PHYSICS of this kernel: the independent numpy kernel
     also gives ≈ 5.2 (exact value depends on aux/J choice: the documented
     number came from the Python path, DF-J/K def2-universal-jkfit, so a
     difference of a few percent between that and this test's exact-J
     reference is expected and is NOT a discrepancy).
   - If it is an IMPLEMENTATION artifact: numpy gives a different value
     (e.g. near the bare/CPHF-like ~8-9 a.u.).
4. Scissor 0 at PBE (the documented "excitonic instability").
   - If physics: the numpy kernel ALSO gives a negative α diagonal element
     there (stored in the reference), and ferric refuses via
     `check_alpha_diagonal_positive`.
   - If a ferric assembly bug: numpy α diagonal is positive at scissor 0.
5. A matching static α does NOT validate the dynamic response or C6
   (`run_bse_c6_ks`, ~63% low). Nothing here touches ω > 0.

## Negative controls planned

- ferric screened α vs numpy BARE α: must miss by > 10× the bar.
- truncated PDEP (trunc_thresh 1e-2) vs the full-rank reference: must miss.
- aux swap (def2-universal-jkfit) vs the same-aux reference: must miss.
- scissor 0 at PBE: refusal (record the result either way).

## What the two paths share (independence statement)

Shared: the geometry file, the orbital basis and aux basis JSONs (same
contraction renormalization), the DF Coulomb-metric factorization
L = V^{-1/2}(P|pq) (product-invariant), the formulas above (derived by
reading bse.rs — so a CONCEPTUAL error common to both, e.g. a wrong physical
kernel definition, is NOT caught by the screened comparison; only the bare
limit vs CPHF checks the convention against an independent code), and for
PBE the grid recipe (75,110) and the 1e-10 density floor.
Not shared: integral engines (libint vs libcint), SCF solvers, the W
construction (PDEP eigen-decomposition + redressing in ferric vs a direct
(I+Π)⁻¹ in numpy), the A±B assembly, and the linear solve.
Both screened paths are density fitted with the same aux; the CPHF anchor
is density fitted (same aux) in one comparison and exact-ERI in another.

---
## Results (appended after measurement)
