# CAS-CI

ferric runs complete-active-space configuration interaction (CAS-CI) from a
closed-shell RHF reference. It is a Rust library routine only: the CLI and
the Python bindings do not expose it.

The Hamiltonian, the determinant phase convention and the Slater-Condon rules
are derived in `crates/ferric-ci/theory.md`. The phase conventions are checked
by `scripts/casci_slater_condon_phases_sympy.py`.

## Scope

| | Implemented |
|---|---|
| Space | full CI inside `n_active` spatial orbitals; no RAS1/RAS3 restrictions |
| Reference | restricted (closed-shell) RHF, a singlet with `n_alpha == n_beta` in the active space |
| Roots | the single lowest root |
| Solver | CI-native Davidson, diagonal preconditioner `1/(H_ii - λ)` |
| Entry points | Rust only: `ferric_ci::run_cas_ci` (re-exported as `ferric::ci`) |
| Not implemented | open-shell or non-singlet references, S² control, multiple roots, CLI `method.kind`, Python function |

With no inactive orbitals the method is FCI in the basis. A call that fails
the bookkeeping check, `2 * active_start + n_alpha + n_beta == nelec`, returns
an error. So does a non-restricted reference, `n_active == 0`, or
`n_active > 63` (the `u64` string limit).

## Energy

$$
E_\text{CASCI} = E_\text{core} + \lambda_\text{min}(H_\text{act}),
$$

where $E_\text{core}$ is the nuclear repulsion plus the energy of the
`active_start` doubly occupied inactive orbitals, and $H_\text{act}$ carries
the inactive-core Fock potential in its one-electron part. `e_active` is
`e_total - e_core`, the quantity PySCF reports as `CASCI.e_cas`.

## Library use

```rust,ignore
use ferric_ci::{run_cas_ci, CasCiConfig};

// `mol`, `prep` and `rhf` come from a converged restricted solve_rhf
// with exact J/K (no density fitting), as in the validation tests.
let cfg = CasCiConfig {
    n_active: 6,
    n_elec_active: (3, 3),
    active_start: 4,          // MOs 0..4 are doubly occupied and inactive
    ..Default::default()
};
let res = run_cas_ci(&mol, &prep, &rhf, cfg)?;
println!("{res}");            // energy, determinant count, iterations
```

`CasCiResult` holds `e_total`, `e_active`, `e_core`, `n_determinants`,
`ci_vector` (index `ia * n_beta + ib`), `converged` and `iterations`.
Defaults: `conv_thresh = 1e-9`, `max_subspace = 50`, `max_iter = 200`.

## Limits

- The active-space integrals come from a dense AO ERI tensor, `O(nbasis⁴)`
  memory. Use it on minimal and double-zeta basis sets.
- The sigma build $\sigma = Hc$ is the string-driven direct algorithm
  (Knowles-Handy/Olsen): β-β, α-α and α-β contractions over per-string
  single-excitation lists, `O(N_det · n_exc_α · n_exc_β)` per call with
  `n_exc = k(n-k)+k`, parallel over α strings. It never visits determinant
  pairs. The `O(N_det²)` pair loop (`hamiltonian::sigma_naive`) and the
  dense Hamiltonian remain as test oracles.
- The Davidson subspace needs `2 * max_subspace * N_det * 8` bytes.
  `memory_budget_bytes` bounds the integral transform and the subspace;
  `None` resolves through `FERRIC_MEM_BUDGET_GB`, then 0.8 of available RAM,
  then a 2 GiB floor.

## Validation

Compared with PySCF `mcscf.CASCI` on an exact-integral RHF with the same
orbital window, and with a dense diagonalization of the full active-space
Hamiltonian:

| System | Active space | Basis | Largest difference |
|---|---|---|---|
| N2, r = 1.10 and 2.20 Å | CAS(6,6) | cc-pVDZ | E_CASCI 1.4e-11 Ha |
| H2O | CAS(4,4) | 6-31G | e_core and e_active 2.6e-10 Ha |

Each system is also run with the window moved down one orbital. The test
tolerance is 3e-9 Ha. The stretched N2 reference is the symmetric RHF, which
is internally unstable toward a symmetry-broken RHF 0.19 Ha lower; both codes
use the symmetric state.

The sign convention is checked three ways, because energy comparisons cannot
detect a diagonal $\pm1$ similarity of $H$:

1. `tests/slater_condon_brute_force.rs` compares `general_element`, the
   direct `sigma` and the diagonal with an independent bit-loop second-quantized $H$ on
   random asymmetric integrals, CAS(4,4) to CAS(6,6), odd and even $N_\alpha$.
   The largest difference is 8.9e-15.
2. `scripts/casci_slater_condon_phases_sympy.py` builds the
   Jordan-Wigner operators symbolically, verifies the anticommutation
   relations, and shows that a line-for-line port of `general_element` equals
   the explicit $H$ as an exact linear form in $h_{pq}$ and $(pq|rs)$ for four
   $(n, N_\alpha, N_\beta)$ sectors. Four sign mutations and the interleaved
   orbital order each fail.
3. `tests/direct_sigma_vs_naive.rs` compares the direct sigma with the
   pair-loop `sigma_naive` on random vectors: ten random-integral sectors
   (unequal $N_\alpha, N_\beta$, empty and full channels) and real
   H2O/STO-3G integrals (full CI, 441 determinants; 8 electrons in 6
   orbitals, 225; four unequal-spin sectors). The largest difference is
   9.9e-14 Ha on vectors with $|\sigma| \approx 74$ Ha. Dropping one phase
   factor, the α-β term, the β-β term, the ½ on same-spin products, the
   $-\tfrac12\sum_r (pr|rq)$ one-body correction, or the β sign in the
   α-β gather each fail it.
4. The PySCF energies above fix the spectrum.

The grade is on [Capabilities and validation](../reference/validation.md).
Sources: `crates/ferric-ci/tests/validation_casci.rs` and
`scripts/validation/gen_casci.py`.
