# 17.2 Slater–Condon rules as implemented

<!-- section: 17.2; status: ready; registry: casci;
     theory: crates/ferric-ci/theory.md#"Slater–Condon rules as implemented" -->

The matrix element between two determinants depends only on how many
orbitals they differ in, and on a sign from reordering the operators. The
textbook rules state the integrals; the signs are what the code has to get
right, and they depend on the convention of section 17.1. ferric's
production CI never visits a determinant pair (section 17.3), so
`general_element` is the reference that the direct sigma is tested against.

| Class | Condition | Function |
|---|---|---|
| diagonal | $d_\alpha = d_\beta = 0$ | `diagonal_element` |
| single | $d = 2$ in one channel | `single_excitation_same_spin` |
| same-spin double | $d = 4$ in one channel | `double_same_spin`, `double_phase_same_spin` |
| opposite-spin double | $d_\alpha = d_\beta = 2$ | `double_opposite_spin` |

| Entry point | Location |
|---|---|
| Rust | `ferric-ci/src/hamiltonian.rs::{general_element,diagonal_element,dense_hamiltonian,sigma_naive}` |

## 17.2.1 The rules

<!-- include: crates/ferric-ci/theory.md#"Slater–Condon rules as implemented" -->

The opposite-spin double carries one phase per channel and no exchange
term:

<!-- excerpt: ferric-ci/src/hamiltonian.rs::double_opposite_spin -->
```rust
// α phase
let pa_i = phase_below(a_ket, i);
let pa_a = phase_below(a_ket & !(1u64 << i), a);
// β phase
let pb_j = phase_below(b_ket, j);
let pb_b = phase_below(b_ket & !(1u64 << j), b);

pa_i * pa_a * pb_j * pb_b * h2(ints, a, i, b, j)
```

| Symbol | Code |
|---|---|
| $\epsilon_\alpha = {\rm pb}(A,t)\,{\rm pb}(A\setminus t, v)$ | `pa_i * pa_a` |
| $\epsilon_\beta = {\rm pb}(B,u)\,{\rm pb}(B\setminus u, w)$ | `pb_j * pb_b` |
| $(vt|wu)$ | `h2(ints, a, i, b, j)` |

> **Verified equation 17.2** · every element against the Jordan–Wigner $H$
>
> For every bra–ket pair in the sectors $(n, N_\alpha, N_\beta) =$
> (4,2,2), (5,3,2), (5,2,3) and (5,3,3), a line-for-line port of
> `general_element` equals the matrix element of the explicit Jordan–Wigner
> Hamiltonian as an exact linear form in symbolic $h_{pq}$ and
> $(pq|rs)$. The odd-$N_\alpha$ sectors are the ones that test the omitted
> α parity on β operators. Four sign mutations each mismatch, and the
> interleaved order mismatches element-wise with the same spectrum.
>
> | | |
> |---|---|
> | Proof | `scripts/casci_slater_condon_phases_sympy.py`, claim 3 |
> | Negative control | claim 4, cases a–e |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-ci/src/hamiltonian.rs::{general_element,diagonal_element,double_opposite_spin}` |
> | Code test | `general_element_matches_second_quantization_on_random_integrals` in `crates/ferric-ci/tests/slater_condon_brute_force.rs` |
>
> <!-- verified: id=ci.slater-condon.elements; script=scripts/casci_slater_condon_phases_sympy.py; claim=3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-ci/src/hamiltonian.rs::{general_element,diagonal_element,double_opposite_spin}; rust_test=ferric-ci/tests/slater_condon_brute_force.rs::general_element_matches_second_quantization_on_random_integrals; registry=casci -->

## 17.2.2 Why an energy test is not enough

An FCI energy against PySCF checks the spectrum only. A $\pm1$ similarity
inserted into `general_element` passes every energy test, and H2 in a
minimal basis has no nontrivial sign at all. The element-wise checks above,
and the Rust test against an independent bit-loop second-quantized $H$ on
random asymmetric integrals, are what pin the convention.

## 17.2.3 Numerical anchors

| Quantity | Systems | Error | Test |
|---|---|---|---|
| `general_element`, `sigma` and the diagonal vs an independent bit-loop $H$ on random asymmetric integrals | CAS(4,4) to CAS(6,6), odd and even $N_\alpha$ | $\le 8.9\times10^{-15}$ | `crates/ferric-ci/tests/slater_condon_brute_force.rs` |

## 17.2.4 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-ci/theory.md`, "Slater–Condon rules as implemented" | the four classes and their phases |
| `scripts/casci_slater_condon_phases_sympy.py` | claims 1–4 |
| `crates/ferric-ci/theory.md`, "Checks" | what each test constrains |

**Known limits (2026-11-15).** `dense_hamiltonian` and `sigma_naive` visit
every determinant pair and are test and benchmark oracles; `run_cas_ci`
calls neither.
