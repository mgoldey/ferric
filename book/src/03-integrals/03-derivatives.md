# 3.3 Derivative integrals and buffer sizing

<!-- section: 3.3; status: edit; registry: none;
     theory: crates/ferric-integrals/theory.md#"Derivative integrals and buffer sizing" -->

Analytic gradients and Hessians (chapters on SCF and MP2 gradients) contract
derivative integrals with densities. The derivatives themselves are libint2's;
what a caller gets wrong is the number of blocks an engine writes.

| Entry point | Location |
|---|---|
| Rust | `ferric-integrals/src/engine.rs::Engine` (`new_1e_deriv`, `compute_1e_deriv_block`, `compute_1e_deriv_block_n`, `new_2e_deriv`, `compute_eri_deriv_quartet`, `new_3center_deriv`, `compute_eri3_deriv`, `new_2center_deriv`, `compute_eri2_deriv`) |
| Second derivatives | `new_1e_deriv2`, `new_2e_deriv2`; pair order `ferric-integrals/src/engine.rs::{deriv2_pair_index,deriv2_pairs}` |

<!-- include: crates/ferric-integrals/theory.md#"Derivative integrals and buffer sizing" -->

## 3.3.1 Translational invariance in practice

The derivatives with respect to all centres of one integral sum to zero. A
gradient code can use that to skip one centre's block and recover it as minus
the sum of the others, and the ECP derivative does exactly that
(section 3.5). The engines still return every centre, and a code that skips
one must not assume a smaller buffer.

## 3.3.2 Numerical anchors

| What | Test | Bar |
|---|---|---|
| shifted-centre 3- and 2-center derivatives against finite differences | `crates/ferric-scf/tests/df_hessian_stencil.rs` | per test |
| point-charge derivative buffer sized for the charge count | `crates/ferric-scf/tests/qmmm_polarizable.rs` | — |
| four-center derivatives at shifted centres | `crates/ferric-integrals/tests/eri_deriv_shifted.rs` | per test |

## Sources

- T. Helgaker and P. R. Taylor, in *Modern Electronic Structure Theory*,
  edited by D. R. Yarkony (World Scientific, 1995), part II, chapter 12.
