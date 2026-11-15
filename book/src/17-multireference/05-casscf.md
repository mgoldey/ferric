# 17.5 Two-step CASSCF

<!-- section: 17.5; status: ready; registry: casscf;
     theory: crates/ferric-ci/theory.md#"The two-step method", "Second-order check" -->

CASSCF minimizes the CAS-CI energy over the orbitals as well as the CI
vector. ferric does it in two steps per macro iteration: a CAS-CI in the
current orbitals (sections 17.1–17.4), then an orbital step
$C \leftarrow C\exp(K)$ from the gradient at those fixed RDMs. The orbital
gradient is a generalized Fock matrix built from $\gamma$, $\Gamma$ and
the core and active Fock matrices; the step is an augmented-Hessian Newton
step on the exact fixed-RDM orbital Hessian (default) or an inverse BFGS
step.

| Symbol | Meaning | Code name |
|---|---|---|
| $F^I$, $F^A$ | inactive and active Fock matrices | `active_fock` |
| $F_{pq}$ | generalized Fock matrix | `CasScfProblem::generalized_fock` |
| $\kappa_{pq}$, $K$ | rotation parameters, $K_{pq} = \kappa_{pq} = -K_{qp}$ for $p > q$ | `rotate`, `expm_antisymmetric` |
| $g$ | $\partial E/\partial\kappa = 2(F_{pq} - F_{qp})$ over core-active, core-virtual and active-virtual pairs | `CasScfPoint` |

| Entry point | Location |
|---|---|
| Rust | `ferric-ci/src/casscf.rs::{run_casscf,optimize,CasScfProblem}`, `ferric-ci/src/casscf_newton.rs::{orbital_hessian,ah_davidson}`, `ferric-ci/src/casscf_stability.rs::stability` |
| CLI | `method.kind = "casscf"` with `[cas]`, `[cas.ci]` (`examples/water-casscf.toml`, `examples/o2-casscf-triplet.toml`) |
| Python | `run_casscf` |

## 17.5.1 Energy, gradient and step

<!-- include: crates/ferric-ci/theory.md#"The two-step method" -->

> **Verified equation 17.5** · the orbital gradient is $2(F - F^\mathsf{T})$
>
> On a six-orbital model (one core, three active, two virtual orbitals) with
> random N-representable RDMs from a random CI vector, the central
> difference of the CASSCF energy at fixed RDMs along every rotation
> $p > q$ of $C\exp K$ equals $2(F_{pq} - F_{qp})$ with
> $F_{pi} = 2(F^I + F^A)_{pi}$,
> $F_{pt} = \sum_u F^I_{pu}\gamma_{ut} + \sum_{uvw}(pu|vw)\Gamma_{tuvw}$ and
> $F_{pa} = 0$; the core-virtual and active-virtual blocks reduce to
> $4(F^I + F^A)_{ai}$ and $2F_{at}$. For active $t, u$,
> $2(F_{tu} - F_{ut}) = -2c^\mathsf{T}(E_{tu} - E_{ut})r$ with
> $r = Hc - \langle c|H|c\rangle c$, so the active-active gradient vanishes
> at a CI eigenvector.
>
> | | |
> |---|---|
> | Proof | `scripts/ci_rdm_casscf_gradient_sympy.py`, checks `casscf-gradient`, `redundant-residual` |
> | Negative control | the runner's mutations: the factor 2 of $F_{pi}$ set to 1; the exchange of $F^A$ dropped. A core factor of 1 is also a printed negative control |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-ci/src/casscf.rs::{generalized_fock,active_fock,gradient_at_fixed_rdms}` |
> | Code test | `orbital_gradient_matches_fd_h2o_cas44_631g` in `crates/ferric-ci/tests/casscf.rs` |
>
> <!-- verified: id=ci.casscf.gradient; script=scripts/ci_rdm_casscf_gradient_sympy.py; claim=casscf-gradient; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-ci/src/casscf.rs::{generalized_fock,active_fock,gradient_at_fixed_rdms}; rust_test=ferric-ci/tests/casscf.rs::orbital_gradient_matches_fd_h2o_cas44_631g; registry=casscf -->

The box is at fixed RDMs, which is what the formula claims: at a converged
CI root the RDM response does not enter the gradient, because the CI energy
is stationary in $c$. The Rust test checks the analytic gradient against a
finite difference of the CI-relaxed energy per rotation class. The
augmented-Hessian products and the step control are not covered by a proof
script; their anchors are the PySCF comparisons below.

## 17.5.2 The macro iteration

| Step | Quantity | Function |
|---|---|---|
| 1 | $F^I$, $E_\text{core}$, $h_\text{eff} = C_a^\mathsf{T}F^IC_a$, $(tu|vw)$ by the dense, direct or RI route | `CasScfProblem::evaluate` |
| 2 | CAS-CI with RDMs; a root that misses the CI threshold is an error | `run_cas_ci_on_integrals` |
| 3 | $F^A$, the generalized Fock matrix, $g$ | `generalized_fock` |
| 4 | converged if $\lVert g\rVert_2 <$ `grad_tol` and $|\Delta E| <$ `energy_tol` at the same iteration | `optimize` |
| 5 | step $s$: AH Davidson on the fixed-RDM Hessian with trust radius and curvature stretch, or inverse BFGS | `ah_davidson`, `optimize` |
| 6 | $C \leftarrow C\exp(K(s))$ | `rotate`, `expm_antisymmetric` |
| 7 | after convergence, optional canonicalization and second-order check | `canonicalize`, `stability` |

## 17.5.3 The second-order check

<!-- include: crates/ferric-ci/theory.md#"Second-order check" -->

The CI response matters for the sign: without it the check reads the
fixed-RDM Hessian, which is stiffer, and can call a saddle a minimum. The
check costs about as much as the CASSCF itself, so it is off by default.

## 17.5.4 Options

| Option (`[cas]`; Python) | Values | Default | Effect |
|---|---|---|---|
| `n_active`, `n_elec_active` | integer, $[n_\alpha, n_\beta]$ | required | active space and sector |
| `n_core` | integer | the electrons the active space leaves, halved | doubly occupied core |
| `grad_tol`; `casscf_grad_tol` | | $10^{-7}$ | gradient-norm threshold |
| `energy_tol`; `casscf_energy_tol` | Ha | $10^{-12}$ | energy-change threshold |
| `max_iter`; `casscf_max_iter` | integer | 300 | macro iterations |
| `max_step`; `casscf_max_step` | rad | 0.2 | cap on the largest step element |
| `optimizer`; `casscf_optimizer` | `ah`, `bfgs` | `ah` | orbital step |
| `integrals`; `casscf_integrals` | `auto`, `direct`, `dense`, `ri` | `auto` | integral route; `ri` needs `auxbasis` and is a different Hamiltonian |
| `integral_thresh`; `casscf_integral_thresh` | | $10^{-12}$ | Schwarz threshold of `direct` |
| `stability`; `casscf_stability` | `off`, `check`, `follow` | `off` | second-order check |
| `canonicalize` | bool | false | canonical core and virtual, natural active orbitals |
| `[cas.ci] conv_thresh`; `ci_conv_thresh` | residual | $10^{-10}$ | CI root acceptance |
| `[cas.ci] max_subspace`, `max_iter` | integer | 50, 200 | CI Davidson |

## 17.5.5 Numerical anchors

| Quantity | Systems | Error | Test |
|---|---|---|---|
| $E$ vs PySCF `mcscf.CASSCF` on ferric's AO integrals | H2O CAS(4,4) / 6-31G | $\le 1.5\times10^{-11}$ Ha | `crates/ferric-ci/tests/casscf_vs_pyscf.rs` |
| the same | N2 CAS(6,6) at 1.0977 and 1.6 Å, O2 CAS(8,6) $(5,3)$ / 6-31G | $\le 5.8\times10^{-13}$ Ha | `crates/ferric-ci/tests/casscf_vs_pyscf.rs` |
| full space: $E$ vs `run_cas_ci` | H2O ($M_s$ = 0, 1), N2, O2 ($M_s$ = 1) / STO-3G | $\le 10^{-13}$ Ha | `crates/ferric-ci/tests/casscf.rs` |
| full space: $2(F - F^\mathsf{T})$ | same | $\le 7.7\times10^{-11}$ | `crates/ferric-ci/tests/casscf.rs` |
| analytic vs finite-difference gradient | | $\le 4.6\times10^{-10}$ | `crates/ferric-ci/tests/casscf.rs` |
| canonicalization: change of $E$ | | $\le 5.7\times10^{-14}$ Ha | `crates/ferric-ci/tests/casscf_canonical.rs` |
| RI route vs PySCF `density_fit()` CASSCF | the three 6-31G systems, def2-universal-jkfit | $\le 1.2\times10^{-12}$ Ha | `crates/ferric-ci/tests/casscf_ri.rs` |

## 17.5.6 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-ci/theory.md`, "Two-step CASSCF" | the method; its measurement records follow the method text there |
| Chaban, Schmidt and Gordon, TCA 97, 88 (1997) | the approximate diagonal Hessian |
| `scripts/prototypes/casscf_proto.py`, `casscf_derivation.md` | the reference implementation |
| `scripts/ci_rdm_casscf_gradient_sympy.py` | `casscf-gradient`, `redundant-residual` |

**Known limits (2026-11-15).** Lowest root, no state averaging; one
restricted orbital set with a doubly occupied core, from an exact-J/K RHF
with an even electron count. Both optimizers are two-step: no orbital–CI
coupling in the step. An unconverged run or a saddle found by the check
caps the certificate to not graded.
