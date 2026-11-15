# 17.6 SC-NEVPT2

<!-- section: 17.6; status: gap; registry: nevpt2;
     theory: crates/ferric-ci/theory.md#"SC-NEVPT2 (`src/nevpt2.rs`)" -->

A CASSCF energy has the static correlation of the active space and none of
the dynamic correlation outside it. Strongly contracted NEVPT2 adds that
second part by perturbation theory with Dyall's zeroth-order Hamiltonian,
which keeps the two-electron interaction inside the active space. One
perturber per external label, $\Psi_l = P_lH\Psi_0$, makes the energy a sum
over eight excitation classes, each a ratio of a norm to a shifted
denominator.

| Symbol | Meaning | Code name |
|---|---|---|
| $l$ | label: the external (core and virtual) orbitals of a class | |
| $N_l$, $K_l$ | $\langle\Psi_l|\Psi_l\rangle$, $\langle\Psi_l|H_\text{act} - E_0|\Psi_l\rangle$ | |
| $\Delta_l$ | $\sum_{r\in l}\varepsilon_r - \sum_{i\in l}\varepsilon_i$ | |
| classes | `Sr`, `Si`, `Sijrs`, `Sijr`, `Srsi`, `Srs`, `Sij`, `Sir` | `sc_nevpt2` result |

| Entry point | Location |
|---|---|
| Rust | `ferric-ci/src/nevpt2.rs::sc_nevpt2` |
| CLI | `method.kind = "nevpt2"` with `[cas]` (`examples/n2-nevpt2.toml`); `[cas] reference = "casscf"` (default) or `"casci"` |
| Python | `run_nevpt2(reference=)` |

## 17.6.1 The method

<!-- include: crates/ferric-ci/theory.md#"SC-NEVPT2 (`src/nevpt2.rs`)" -->

This section has no verified box. The class formulas are a transcription of
PySCF `mrpt.nevpt2`, and their derivation from Dyall's Hamiltonian is in a
working note (`scripts/prototypes/nevpt2_derivation.md`), not in
`theory.md`. A sympy proof needs the 3-RDM and the 4-RDM contractions of
section 17.4.2 on a model space, so the registry lists `nevpt2` as a gap.
The evidence below is two independent constructions: the numpy prototype's
Wick construction and PySCF.

## 17.6.2 The algorithm

| Step | Quantity | Function |
|---|---|---|
| 1 | a converged CASSCF or CAS-CI root with $n_\alpha = n_\beta$; $\lVert Hc - Ec\rVert \le$ `max_ci_residual`, $\lVert c\rVert = 1$ | `sc_nevpt2` |
| 2 | canonical core and virtual orbitals in $F^I + F^A$ (PySCF `CASCI.canonicalize`); active orbitals as given | |
| 3 | NEVPT RDMs: E-product 2- and 3-RDMs, $f^\text{ca}$, $f^\text{ac}$, hole RDMs | `ferric-ci/src/rdm3.rs::{nevpt_rdms,hole_rdms,e_product_rdms}` |
| 4 | external integral blocks from the dense AO block (or the RI $B$) | |
| 5 | the eight classes, $E_\text{class} = -\sum_l N_l/(\Delta_l + K_l/N_l)$ over $|N_l| > 10^{-14}$ | `make_a16`, `make_a22`, … |

The memory of the AO block, the MO blocks, twelve $n_a^6$ intermediates and
the per-label tensors is reserved through one memory plan before the AO
block is built.

## 17.6.3 Numerical anchors

| Quantity | Systems | Error | Test |
|---|---|---|---|
| class energies vs the numpy prototype on the same orbitals and CI vector | H2O CAS(4,4), N2 CAS(6,6) at 1.0977 and 2.2 Å / 6-31G | $\le 6.2\times10^{-14}$ Ha | `crates/ferric-ci/tests/nevpt2.rs` |
| vs PySCF `mrpt.NEVPT` on PySCF's own CASSCF | N2 (H2O $1.3\times10^{-8}$, stretched N2 $2.1\times10^{-8}$) | $\le 3.2\times10^{-8}$ Ha | `crates/ferric-ci/tests/nevpt2.rs` |
| one closed-shell active orbital vs MP2 pair sums | H2O / 6-31G | $8.1\times10^{-15}$ Ha | `crates/ferric-ci/tests/nevpt2.rs` |
| classes with an empty external space | H2, LiH | exactly 0 | `crates/ferric-ci/tests/nevpt2.rs` |

The PySCF bar propagates each code's CASSCF convergence into the class
energies; inside a degenerate group of core or virtual orbitals the pair
classes depend on the eigenvectors chosen, which moves N2's `Srs` by up to
$1.7\times10^{-7}$ Ha between codes.

## 17.6.4 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-ci/theory.md`, "SC-NEVPT2" | definitions, orbitals, evaluation, cost |
| Angeli, Cimiraglia and Malrieu, JCP 117, 9138 (2002) | the method |
| `scripts/prototypes/nevpt2_derivation.md`, `nevpt2_proto.py` | the derivation note and the Wick-construction reference |
| PySCF `mrpt/nevpt2.py` | the factorization transcribed |

**Known limits (2026-11-15).** Single root, closed-shell active spaces only
($n_\alpha = n_\beta$; open-shell active spaces are refused). On a CAS-CI
reference the certificate is not graded. Dense $O(n_a^8)$ intermediates
bound the active-space size.
