# 3.2 Two-, three- and four-center integrals

<!-- section: 3.2; status: edit; registry: none;
     theory: crates/ferric-integrals/theory.md#"Two-, three- and four-center integrals" -->

Four-center integrals feed exact J and K, three-center integrals $(P|\mu\nu)$
and the two-center metric $(P|Q)$ feed every density-fitted method
(chapter 6). One `Engine` type serves all three classes and every kernel.

| Entry point | Location |
|---|---|
| Rust | `ferric-integrals/src/engine.rs::Engine` (`new_2e`, `compute_quartet`, `new_3center`, `compute_eri3`, `new_2center`, `compute_eri2`) |
| Rust, whole tensors | `ferric-integrals/src/threeindex.rs::{eri3_tensor,coulomb_metric_2c}` |
| Python | `compute_eri3`, `compute_metric_2c` (`crates/ferric-python/ferric.pyi`) |

<!-- include: crates/ferric-integrals/theory.md#"Two-, three- and four-center integrals" -->

## 3.2.1 Which engine serves which kernel

| Kernel | `OperatorKind` | Engine |
|---|---|---|
| $1/r$ | `Coulomb` | libint2 |
| $\operatorname{erf}(\omega r)/r$, $\operatorname{erfc}(\omega r)/r$ | `ErfCoulomb`, `ErfcCoulomb` | libint2 |
| $e^{-\zeta r}/r$, $e^{-\zeta r}$ | `Yukawa`, `SlaterGeminal` | libint2 (section 4.3) |
| fitted geminals | `Cgtg`, `CgtgCoulomb`, `Delcgtg2` | libint2, `Engine::new_2e_geminal` |
| $\operatorname{terf}/r$, $\operatorname{terfc}/r$ | `Terf`, `Terfc` | the table engine (section 4.2) |

## 3.2.2 What a caller must respect

| Rule | Why |
|---|---|
| A `None` block means screened, not zero by symmetry | libint2 skips a quartet below the engine precision |
| Use the `_checked` calls on a table engine | a refusal (angular momentum past the table depth) is an error, not a zero block |
| Build bound tables with precision $0$ | a skipped diagonal would make a bound lower than its integral (section 5.1) |
| One engine per thread | libint2 engines hold scratch; the code builds a pool (`crates/ferric-integrals/src/engine_pool.rs`) |
| Exceptions never cross the C ABI | every `scf_compute_*` in `crates/ferric-integrals/shim/shim.cc` catches and returns a negative status; the Rust side asserts it |

## 3.2.3 Numerical anchors

| What | Test | Bar |
|---|---|---|
| terf and terfc $(P\vert\mu\nu)$, $(P\vert Q)$ against an independent Fourier-space construction (H2O, NH3 / cc-pVDZ; H2O / aug-cc-pVDZ) | `crates/ferric-integrals/tests/validation_terfc_integrals.rs` (`scs-mp2-2terfc` row of `site/src/reference/validation.md`) | $2\times10^{-12}$ (3-index), $3\times10^{-11}$ (2-index) |
| the four-center table engine against libint2 Coulomb | `crates/ferric-integrals/tests/eri4_vs_libint.rs` | per test |
| permutational symmetry of the quartets | `crates/ferric-integrals/tests/eri_permutational_symmetry.rs` | rounding |
| a C++ throw returns a status, not undefined behaviour | `crates/ferric-integrals/tests/ffi_exception_safety.rs` | — |

## Sources

- E. F. Valeev, Libint, <https://github.com/evaleev/libint>.
- L. E. McMurchie and E. R. Davidson, *J. Comput. Phys.* **26**, 218 (1978).
