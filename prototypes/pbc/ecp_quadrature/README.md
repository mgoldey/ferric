# ferric-owned ECP integrals — design prototype (FINDINGS "Iteration 25", 2026-09-27)

Replacement design for libecpint (semi-local type-2 AND local type-1), Python/NumPy only. PySCF is used ONLY as a test
oracle (`oracle.py`, the tests), never by the engine (`ecpq.py`).

## Method (ecpq.py)
- Semi-local: `<A|U_l P_l|B> = Σ_m ∫ r² U_l(r) F^A_lm(r) F^B_lm(r) dr`, with the angular projection done ANALYTICALLY:
  `F_lm(r) = 4π e^{-α(r-|a|)²} Σ_{N,λ} T[c,m,N,λ] r^N k̃_λ(2α|a|r)`, `k̃_n(z) = e^{-z} i_n(z)` (exp-scaled modified
  spherical Bessel; bounded, no cancellation at any distance, exact on-centre: `k̃_n(0) = δ_n0`).
  `T` (binomial shift × real harmonics × monomial sphere integrals) depends only on (shell, ECP centre, l), so all
  primitive/term/node work accumulates one radial tensor `R[N+N', λ, λ']`, contracted with `T^A T^B` once.
- Local: the product Gaussian `exp(-(α+β)r² + 2k·r)` expanded in the same `k̃_λ` against monomials.
- Radial: every (primitive pair, ECP term) integrand is `exp(-p(r-r0)²)` × analytic. Gauss–Hermite (20 nodes) when
  `r0√p ≥ 6.5`, else Gauss–Legendre on `[max(0, r0-6/√p), r_pk+6/√p]` with `clamp(⌈3.4·len·√p⌉, 16, 40)` nodes.
- Derivatives: raised/lowered shells (`2α c` for l+1, `-i c` for l-1); centre = −(bra + ket) per triple.
- API mirrors `ferric_integrals::ecp`: `ecp_block_spherical`, `ecp_block_deriv_spherical`, `ecp_matrix_spherical`
  (bare-Cartesian coefficients, CCA order, the ecp.rs cart2sph tables — asserted equal by a test).

## Files
| file | what |
|---|---|
| `ecpq.py` | the engine |
| `fixtures.py` | HI/LANL2DZ (= `reference/pbc/ecp_accuracy`) and AuH/def2-SVP + def2-ECP data |
| `oracle.py` | PySCF `ECPscalar_sph` (+ 1e-3 zero-weight screen guard) and an independent radial × Lebedev quadrature |
| `test_ecpq.py` | 19 tests incl. 3 negative controls (`python -m pytest -q test_ecpq.py`, ~4-5 min) |
| `measure.py` | FINDINGS tables: `python measure.py values lebedev smooth deriv target oncentre` |
| `radial_conv.py`, `radial_scheme.py` | radial rule convergence / choice (per-window vs over-converged reference) |
| `deriv_hscan.py` | h-scan separating FD truncation from value roundoff on the worst derivative blocks |
| `cost.py` | triiodobenzene/def2-SVP: work counts, libecpint timing (ctypes shim), writes `windows.txt` |
| `radial_kernel.cc` | C++ stand-in of the Rust hot loop, timed on `windows.txt` |

Run everything with `OPENBLAS_NUM_THREADS=1` in the PySCF venv (`/home/matt/qc/ferric/.venv/bin/python`).
`cost.py` needs the ctypes shim library built as in `reference/pbc/ecp_accuracy/README.txt`.
