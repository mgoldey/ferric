# 3.7 Basis functions and their derivatives on a grid

<!-- section: 3.7; status: edit; registry: none;
     theory: crates/ferric-integrals/theory.md#"Basis functions on a grid" -->

Density functionals, seminumerical exchange, Becke partitions and cube files
all need basis functions at points. One module evaluates values, gradients and
Hessians, and the DFT crate re-exports it.

| Entry point | Location |
|---|---|
| Rust | `ferric-integrals/src/ao_grid.rs::{eval_basis_on_points,eval_basis_and_grad_on_points,eval_basis_grad_hess_on_points,check_ao_grid_budget}` |
| Third derivatives | `ferric-integrals/src/ao_grid_third.rs::eval_basis_third_on_points` |
| Re-export | `ferric_dft::ao_grid` |

<!-- include: crates/ferric-integrals/theory.md#"Basis functions on a grid" -->

## 3.7.1 Memory

| Evaluation | Planes of $N \times N_g$ doubles |
|---|---|
| values | 1 |
| values and gradients | 4 |
| values, gradients and Hessians | 13 |
| third derivatives | 10 |

`check_ao_grid_budget` multiplies these by $8 N N_g$ bytes and errors before
allocating when the product exceeds the resolved memory budget. A caller that
evaluates in batches of points keeps $N_g$ small; the planes count is per
batch.

## 3.7.2 Numerical anchors

| What | Test | Bar |
|---|---|---|
| $g$ functions: grid-integrated overlap against the analytic overlap (normalization and libint2 order at once), and their gradients and Hessians | `crates/ferric-dft/tests/gfunc.rs` | per test |
| meta-GGA gradients built on the AO Hessians, against PySCF with grid response | the meta-GGA gradient tests listed in `site/src/reference/validation.md` | per row |

## Sources

- T. Helgaker, P. Jørgensen and J. Olsen, *Molecular Electronic-Structure
  Theory* (Wiley, 2000), section 6.6.
