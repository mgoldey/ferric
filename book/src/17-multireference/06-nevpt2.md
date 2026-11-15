# 17.6 SC-NEVPT2

<!-- section: 17.6; status: ready; registry: nevpt2;
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

> **Verified equation 17.6** · the eight class energies
>
> On a CAS(4,4) model (two core, four active, two virtual orbitals, random
> positive semidefinite $(pq|rs)$, core and virtual orbitals canonical in
> $F^I + F^A$), a numpy transcription of every class and intermediate
> function of `nevpt2.rs` gives each class norm $\sum_l N_l$ and energy
> $E_\text{class} = -\sum_l N_l/(\Delta_l + K_l/N_l)$ equal to the
> definition evaluated in the full 4900-determinant space:
> $\Psi_l = P_lH\Psi_0$ grouped by external occupation, $K_l$ from Dyall's
> $H_\text{act}$ on active indices only. It holds on the lowest singlet and on
> the lowest $M_s = 0$ triplet root of the $(2, 2)$ sector. Anchors:
> $\langle\Psi_0|H|\Psi_0\rangle = E_\text{core} + E_0$ and
> $\sum_l N_l = |QH\Psi_0|^2$. The transcription is pinned to the Rust file:
> every `einsum` specification and permutation of those functions, in order,
> is parsed from `nevpt2.rs` and compared (claim T).
>
> | | |
> |---|---|
> | Proof | claims C and T of `scripts/nevpt2_sympy.py` (claim D: the single-perturber energy, sympy) |
> | Negative control | the runner's mutations: a sign flip of the `f3ca` term of `make_a16` (`Sr`) and of one `make_a12` term (`Sir`), the `Sijr` pair diagonal not halved, $\theta$ of `Sijrs` replaced by $v$, the sign of the `Sij` denominator, an $i \leftrightarrow j$ swap of the `Sijr` block (seen by claim T only: `Sijr` is symmetric in the pair) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-ci/src/nevpt2.rs::{sr,si,sijrs,sijr,srsi,srs,sij,sir,norm_to_energy}` |
> | Code test | `matches_prototype_h2o_cas44` in `crates/ferric-ci/tests/nevpt2.rs` |
>
> <!-- verified: id=ci.nevpt2.classes; script=scripts/nevpt2_sympy.py; claim=C; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-ci/src/nevpt2.rs::{sr,si,sijrs,sijr,srsi,srs,sij,sir,norm_to_energy}; rust_test=ferric-ci/tests/nevpt2.rs::matches_prototype_h2o_cas44; registry=nevpt2 -->

> **Verified equation 17.7** · the RDM inputs
>
> On the same model's CI vector, each input of the class formulas by two
> routes: the normal-ordered 1- to 3-RDMs (Gram matrices of annihilated
> vectors) against the E-product $\langle EE\rangle$, $\langle EEE\rangle$
> through the reorder formula and `e_product_rdms`; `hole_rdms` against
> $\langle aa\,a^\dagger a^\dagger\rangle$ and
> $\langle aaa\,a^\dagger a^\dagger a^\dagger\rangle$ from created vectors;
> $f^\text{ca}$, $f^\text{ac}$ by the `rdm3.rs` construction
> ($\langle E_{dc}X_{ba}|R_{ef}\rangle$) against the contraction of a stored
> E-product 4-RDM with $(aa|aa)$.
>
> | | |
> |---|---|
> | Proof | claim R of `scripts/nevpt2_sympy.py` |
> | Negative control | the runner's mutations: the $\delta_{qt}$ term of `e_product_rdms` dropped; a coefficient 2 → 1 in `hole_rdms`; $R'$ built with the index order of $R$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-ci/src/rdm3.rs::{e_product_rdms,hole_rdms,nevpt_rdms}` |
> | Code test | `rdm3_and_f3_match_pyscf_h2o_cas44` in `crates/ferric-ci/tests/casci_rdm3.rs` |
>
> <!-- verified: id=ci.nevpt2.rdms; script=scripts/nevpt2_sympy.py; claim=R; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-ci/src/rdm3.rs::{e_product_rdms,hole_rdms,nevpt_rdms}; rust_test=ferric-ci/tests/casci_rdm3.rs::rdm3_and_f3_match_pyscf_h2o_cas44; registry=nevpt2 -->

The proof checks the factorization against the definition on a model; the
Rust tests below check the Rust code against the numpy prototype's Wick
construction and against PySCF on molecules.

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
| `crates/ferric-ci/theory.md`, "SC-NEVPT2" | definitions, orbitals, the classes and intermediates as coded, cost |
| `scripts/nevpt2_sympy.py` | claims D, R, T and C |
| Angeli, Cimiraglia and Malrieu, JCP 117, 9138 (2002) | the method |
| `scripts/prototypes/nevpt2_derivation.md`, `nevpt2_proto.py` | the derivation note and the Wick-construction reference |
| PySCF `mrpt/nevpt2.py` | the factorization transcribed |

**Known limits (2026-11-15).** Single root, closed-shell active spaces only
($n_\alpha = n_\beta$; open-shell active spaces are refused). On a CAS-CI
reference the certificate is not graded. Dense $O(n_a^8)$ intermediates
bound the active-space size.
