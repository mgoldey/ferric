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

Measured 2026-10-05 (fix/p2-274; generator `gen_rpax_alpha.py`, test
`validation_rpax_alpha.rs`, `--release`, OPENBLAS_NUM_THREADS=1).

### Anchors (generator)

| system | numpy bare vs PySCF DF-CPHF | numpy exact-ERI bare vs exact CPHF | DF fitting gap (bare DF vs exact CPHF) | PySCF iterative `cphf.solve` vs dense |
|---|---|---|---|---|
| H2O / aug-cc-pVDZ | 3.1e-15 | 5.3e-15 | 9.0e-5 | 5.5e-8 |
| NH3 / aug-cc-pVDZ | 7.6e-15 | 1.9e-14 | 1.5e-5 | 1.7e-8 |

The first attempt anchored to `scf.cphf.solve` (tol 1e-13) and missed by
5.5e-8. Diagnosis: the CPHF matrix assembled from PySCF's own
`gen_response(hermi=1)` equals the numpy A+B to 1.3e-14, and its dense solve
equals the numpy α to the printed digits; the gap is the iterative solver's
stopping rule. The anchor now uses the dense matrix from PySCF's response
operator, and the iterative value is recorded beside it.

### ferric vs numpy (worst element |d| / max|α|; iso relative; Ha for eigenvalues)

| case | screened α | iso | bare α | min eig A+B | min eig A−B | E_SCF |
|---|---|---|---|---|---|---|
| H2O / aug-cc-pVDZ @RHF | 5.3e-9 | 2.8e-9 | 5.4e-9 | 2.7e-10 | 2.4e-10 | 3.7e-12 |
| NH3 / aug-cc-pVDZ @RHF | 2.1e-9 | 2.4e-10 | 2.1e-9 | 1.0e-10 | 9.5e-11 | 5.2e-12 |
| H2O / cc-pVDZ @PBE, scissor 0.36 | 2.3e-9 | 2.2e-9 | — | 7.5e-10 | 7.5e-10 | 4.3e-12 |

Bars: 5e-8 (α, iso, bare), 1e-8 Ha (eigenvalues), 1e-9 Ha (E_SCF).

Values: screened iso 7.585204 (H2O/aug), 11.754072 (NH3/aug), 5.203987
(H2O/cc-pVDZ@PBE+0.36). numpy route identity |(A−B)(A+B) route − (A+B)⁻¹
route| / max ≤ 2.9e-16: A−B cancels from α, as predicted.

### Hypotheses, verdicts

1. Screened kernel: REAL agreement (≤ 5.3e-9), far below every artifact
   signature.
2. Bare limit vs CPHF: holds (anchors above); ferric's bare hook vs exact-ERI
   CPHF differs by exactly the generator's DF fitting gap (9.013e-5 /
   1.509e-5).
3. Physics of the 46% deficit: the independent kernel gives iso 5.203987 at
   PBE + 0.36 Ha, the documented 5.20. The deficit belongs to the kernel
   (provisional: one system), not to ferric's assembly.
4. Scissor 0 at PBE: the independent kernel has diag (−2.680881, +14.600511,
   +16.492205); ferric refuses with α_xx = −2.680881. The instability is the
   kernel's.
5. Not tested: anything at ω > 0, C6.

### Negative controls (must exceed 5e-7)

screened vs bare 7.9e-2 / 8.0e-2 / 1.6e-1; trunc_thresh 1e-2 vs full rank
7.6e-5; aux def2-universal-jkfit vs aug-cc-pvdz-rifit 3.9e-5.

### Mutation ledger (kill-safe harness, committed state 2fe615e1; each mutant
compiled; `--ignored` run, 6 tests run, 0 ignored)

| mutant (in `rpax_static_polarizability_with_screening` only) | result |
|---|---|
| M1 A−B roles swapped: `amb = w_abij − w_ibaj` | 3 failed / 3 passed. α UNCHANGED (worst 5.354e-9, the unmutated value, as predicted); killed ONLY by `min eig A-B` (|d| 0.40 / 0.66 / 0.35 Ha) |
| M2 `4.0 * coul` → `2.0 * coul` in A+B | 4 failed: α off by 0.29 rel; the scissor-0 test also fails (ferric no longer refuses) |
| M3 `w_red` → 0 (W → v) | 3 failed: screened α off by 0.086 / 0.086 / 0.19 |
| M4 α prefactor 4 → 2 | 3 failed: α off by 0.50 |

Source restored after the run; `git diff` on bse.rs clean.
