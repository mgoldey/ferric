# 3.1 One-electron integrals and multipoles

<!-- section: 3.1; status: edit; registry: none;
     theory: crates/ferric-integrals/theory.md#"Conventions", "One-electron integrals" -->

The overlap, kinetic and nuclear-attraction matrices are the first thing any
calculation builds, and the multipole matrices give dipoles, quadrupoles and
orbital spreads. All of them come from libint2 shell-pair blocks.

| Entry point | Location |
|---|---|
| Rust | `ferric-integrals/src/oneelectron.rs::{overlap,kinetic,nuclear,hcore,hcore_ecp,dipole,second_moment,r2_moment}` |
| Python | `AoIntegrals` (`crates/ferric-python/ferric.pyi`) |

## 3.1.1 Basis functions and the integrals

<!-- include: crates/ferric-integrals/theory.md#"Conventions" -->

## 3.1.2 The one-electron matrices

<!-- include: crates/ferric-integrals/theory.md#"One-electron integrals" -->

## 3.1.3 Options that change the matrices

| Option | Effect |
|---|---|
| ECP atoms (`mol.apply_ecp`) | $Z_C \to Z_C - n_{\mathrm{core}}$ in $V$; $\hat U_{\mathrm{ECP}}$ added by `hcore_ecp` (section 3.5) |
| ghost atoms | $Z_C = 0$; the basis functions stay |
| external point charges and field | folded into $h$ once before the SCF (`ferric-integrals/src/oneelectron.rs::hcore_with_external`); `None` leaves $h$ bit-identical |

## 3.1.4 Numerical anchors

| What | Test | Bar |
|---|---|---|
| every bundled basis set loads to the published contractions (RHF and RI-MP2 against PySCF) | `crates/ferric-mp2/tests/basis_coverage_vs_pyscf.rs`, `crates/ferric-mp2/tests/validation_basis_coverage.rs`; the basis row of `site/src/reference/validation.md` | per row |
| translation identity of the second moments | `ferric-integrals/src/oneelectron.rs::second_moment_translational_identity` | rounding |
| shifted-origin $S$, $T$, $V$ blocks: zero shift is bitwise the plain block | `crates/ferric-integrals/tests/one_electron_shifted.rs` | bitwise |

No proof script covers this section: the integrals are libint2's, and the
section adds conventions, not derivations. The translation identity of the
second moments is the one identity the code relies on, and the unit tests
check it directly.

## Sources

- libint2: E. F. Valeev, Libint, a library for the evaluation of molecular
  integrals over Gaussian functions, <https://github.com/evaleev/libint>.
- T. Helgaker, P. Jørgensen and J. Olsen, *Molecular Electronic-Structure
  Theory* (Wiley, 2000), chapter 9.
