# 3.5 Effective core potentials

<!-- section: 3.5; status: edit; registry: none;
     theory: crates/ferric-integrals/theory.md#"Effective core potentials" -->

An ECP replaces the core electrons of a heavy atom by a semi-local operator.
The atom's nuclear charge drops by the number of core electrons, and the
operator is added to the core Hamiltonian. ferric evaluates the operator by its
own quadrature, with libecpint kept as a cross-check.

| Entry point | Location |
|---|---|
| Rust | `ferric-integrals/src/ecp.rs::{ecp_matrix_spherical,ecp_matrix_deriv_spherical}`, `ferric-integrals/src/oneelectron.rs::hcore_ecp` |
| Backend | `FERRIC_ECP_BACKEND` = `quadrature` (default) or `libecpint` |
| Bases | def2 sets (`def2-ecp.json`), cc-pV$n$Z-PP for $Z \ge 37$ |

<!-- include: crates/ferric-integrals/theory.md#"Effective core potentials" -->

## 3.5.1 The order of operations

| Step | What happens | Where |
|---|---|---|
| 1 | `mol.apply_ecp(bs)` records $n_{\mathrm{core}}$ per atom | `ferric-core/src/mol.rs` |
| 2 | electron count and nuclear repulsion use $Z - n_{\mathrm{core}}$ | `Molecule::nelec`, `Molecule::nuclear_repulsion` |
| 3 | $V$ uses the same effective charges | `ferric-integrals/src/oneelectron.rs::nuclear` |
| 4 | $h = T + V + \hat U$ | `ferric-integrals/src/oneelectron.rs::hcore_ecp` |

Calling step 2 before step 1 counts the core electrons twice; the order is
part of the API.

## 3.5.2 Numerical anchors

| What | Test | Bar |
|---|---|---|
| RHF and UHF energies and gradients with def2 ECPs (HI, CH3I, RbH, I, SnH4) against PySCF and ORCA | `crates/ferric-scf/tests/validation_ecp.rs` (row in `site/src/reference/validation.md`) | measured $\le 2.9\times10^{-8}$ Ha, $\le 2.3\times10^{-8}$ Ha/Bohr |
| spherical ECP blocks against PySCF `ECPscalar` | `crates/ferric-integrals/tests/ecp_matrix.rs` | per test |
| quadrature against libecpint, with broken constructions that must fail | `crates/ferric-integrals/tests/ecp_quadrature.rs` | per test |
| derivative blocks | `crates/ferric-integrals/tests/ecp_matrix_deriv.rs`, `crates/ferric-integrals/tests/ecp_deriv_block.rs` | per test |

## Sources

- L. E. McMurchie and E. R. Davidson, *J. Comput. Phys.* **44**, 289 (1981).
- R. A. Shaw and J. G. Hill, *J. Chem. Phys.* **147**, 074108 (2017).
