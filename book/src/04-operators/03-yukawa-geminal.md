# 4.3 Yukawa, Slater geminal and fitted geminal kernels

<!-- section: 4.3; status: gap; registry: none;
     theory: crates/ferric-integrals/theory.md#"Yukawa and Slater-geminal kernels" -->

Three kernel families decay exponentially in $r_{12}$. The Yukawa kernel is a
screened Coulomb interaction; the Slater geminal $e^{-\zeta r}$ and its
Gaussian fit are the correlation factors of explicitly correlated (F12)
methods. All three come from libint2.

| Entry point | Location |
|---|---|
| Rust | `ferric-integrals/src/operator.rs::{yukawa,slater_geminal,stg}`, `ferric-integrals/src/engine.rs::new_2e_geminal` |
| Consumer | `crates/ferric-mp2/src/f12.rs` (CABS lane; section 6.4) |

<!-- include: crates/ferric-integrals/theory.md#"Yukawa and Slater-geminal kernels" -->

## 4.3.1 Limits a caller must respect

| Region | Stock libint2 | With ferric's `boys.h` overrides |
|---|---|---|
| $U = \zeta^2/4\rho > 10^3$ | out-of-range table read (2.7.2) or a Gaussian-fit fallback (2.13.1) | rejected with an error status (2.7.2) |
| $T > 1024$ and $U + \zeta R \gtrsim 709$ | NaN, silently | the underflowed true value, below $10^{-200}$ |
| Yukawa, small $T$ with $U < 10^{-7}$ | the upward recursion loses all digits at high $m$ | a Gauss–Legendre quadrature of the defining integral |
| $\zeta \le 0$ | invalid | invalid (`debug_assert`) |

The overrides live in `crates/ferric-integrals/shim/libint2_overrides/`; a
libint2 build that does not receive them keeps the stock behaviour, and the
build prints a warning. Ferric builds no Schwarz table for these kernels
(`ferric-integrals/src/schwarz.rs::check_operator_supported` errors), so no
screen hides a bad value.

## 4.3.2 Numerical anchors

| What | Test | Bar |
|---|---|---|
| Yukawa and Slater-geminal auxiliary functions at the edges of libint2's table, against a quadrature reference that is itself anchored | `crates/ferric-integrals/tests/tenno_table_edge.rs` | per point, derived from the reference value |
| small-$T$ behaviour of the Yukawa recursion | `crates/ferric-integrals/tests/yukawa_urr_small_t.rs` | per test |

No proof script covers these kernels. The Fourier transform
$\widehat{e^{-\zeta r}/r} = 4\pi/(k^2 + \zeta^2)$ shows the Yukawa kernel is
positive definite, so a Schwarz bound would be valid; the code does not build
one. A proof script and a committed test of the fitted geminal against the
exact one would move this section out of `gap`.

## Sources

- D. P. Tew and W. Klopper, *J. Chem. Phys.* **123**, 074101 (2005).
- S. Ten-no, *Chem. Phys. Lett.* **398**, 56 (2004).
- E. F. Valeev, Libint, <https://github.com/evaleev/libint>.
