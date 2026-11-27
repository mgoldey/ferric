# Fock hooks {#fock-hooks}

<!-- status: edit; registry: -->

A new one-body or density-functional term (an embedding potential, a
penalty, a relativistic correction) changes the energy and therefore the
Fock matrix, and nothing else about the SCF. A Fock hook adds one such
caller-supplied term to the energy functional without changing the solver.
It is how Python prototypes a new term before it has a Rust implementation.

This section derives what the hook must return and what its nuclear gradient
needs (@sec:fock-hooks-equations), then where it enters the loop, what it refuses and how it is
tested.

| Symbol | Meaning | Code name |
|---|---|---|
| $E_h[D]$ | the hook's energy | the first value the hook returns |
| $V = \partial E_h/\partial D$ | the hook's potential, added to $F$ | the second value |
| $\partial E_h/\partial\mathbf R\vert_D$ | the explicit nuclear derivative at fixed $D$ | `fock_hook_grad` |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/rhf.rs::{solve_rhf_with_hook,validate_fock_hook}`, `ferric-scf/src/uhf.rs::solve_uhf_with_hook`, `ferric-scf/src/rohf.rs::solve_rohf_with_hook` |
| Python | `fock_hook=` on `run_rhf`, `run_dft`, `run_uhf`, `run_rohf`; `fock_hook_grad=` with `with_gradient=True` (`crates/ferric-python/python/ferric/ferric.pyi`) |
| CLI | none |

| Solver | Hook signature |
|---|---|
| RHF, RKS | `hook(D_total) -> (E, V)` |
| UHF, ROHF | `hook(D_a, D_b) -> (E, V_a, V_b)` |

## The equations {#fock-hooks-equations}

### The hooked functional

A hook (`rhf::solve_rhf_with_hook`, `uhf::solve_uhf_with_hook`,
`rohf::solve_rohf_with_hook`; Python `fock_hook=`) adds one caller-supplied
term to the energy functional:

$$ E(\mathbf R) = E_0[D;\mathbf R] + E_h[D;\mathbf R], \qquad V = \partial E_h/\partial D . $$

$E_0$ is the built-in HF or KS energy. $V$ is added to the Fock matrix every
iteration, with $V^\sigma = \partial E_h/\partial D^\sigma$ per spin for UHF
and ROHF. Why the derivative? Section @sec:energy-functionals showed that the
Fock matrix is $\partial E/\partial D$; differentiating the sum gives
$F = F_0 + V$, so the SCF that diagonalizes the hooked $F$ makes the hooked
$E$ stationary: $FC_o = SC_o\varepsilon$ with the hooked $F$.

### The nuclear gradient

Section @sec:fock-hook-gradient derives the gradient of the hooked energy and
proves it. Everything the hook does through the density is already in the
built-in gradient, whose energy-weighted density comes from the hooked
orbitals and eigenvalues. The one term the caller supplies is
$\partial E_h/\partial\mathbf R\vert_D$, the derivative at fixed AO density
(`fock_hook_grad=` in Python,
`ferric-scf/src/gradient.rs::with_fock_hook_gradient` in Rust). That section
lists what the term must and must not include, the exactness anchor
$E_h = \lambda\operatorname{tr}[DS]$, and when the formula holds.

## Where the hook enters the iteration {#fock-hooks-hook-enters-iteration}

$V$ is added to the Fock matrix before DIIS sees it, and $E_h$ is added
outside the $\tfrac12\operatorname{tr}[D(h+F)]$ trace, as $E_{xc}$ is. The
converged energy is therefore stationary in the hooked functional when $V$ is
the derivative of $E_h$. Nothing checks that it is: a $V$ that is not
$\partial E_h/\partial D$ converges to a self-consistent solution whose energy
is not variational, and its gradient is wrong.

Why outside the trace? $\tfrac12\operatorname{tr}[D(h+F)]$ counts $h$ at
full weight and every other term of $F$ at half weight. That is right for $J$
and $K$, whose energies are quadratic in $D$ (section @sec:energy-functionals). A hook energy can
have any form, so its energy is taken as returned and its potential only
enters $F$.

The closed-shell application, `apply_fock_hook` in
`crates/ferric-scf/src/rhf.rs`, with the early return for a run without a
hook removed:

```rust
let (e, v) = h(d)?;                          // (E_h, V) = hook(D)
if v.dim() != f.dim() || !e.is_finite() {    // wrong shape or non-finite E_h
    return Err(FerricError::General(format!(
        "solve_rhf_with_hook: hook returned V of shape {:?} (expected {:?}) and E = {e}",
        v.dim(),
        f.dim()
    )));
}
*f += &v;                                    // F = F_0 + V
Ok(e)                                        // E_h, added to the energy outside the trace
```

## What is refused {#fock-hooks-refused}

Every setting that builds an orbital Hessian or a blocked solve would not see
the hook, so `validate_fock_hook` refuses it, naming the field:

| Field | Reason |
|---|---|
| `symmetry`, `irrep_nelec` | the point-group-blocked solve (section @sec:scf-exits-symmetry) does not call the hook |
| `newton_trigger` | the Newton orbital Hessian does not include the hook |
| `ah_trigger` | the augmented-Hessian step does not include the hook |
| `aurora.enabled` | the AURORA curvature model (section @sec:scf-exits-aurora) does not include the hook |
| `trah_trigger` | the TRAH orbital Hessian does not include the hook |
| `check_stability`, `scf_stability_descent` | the stability Hessian does not include the hook |

Each of these needs the response of the potential to a change of the
density, $\partial V/\partial D$, which the hook interface does not return.

In Python, a gradient run (`with_gradient=True`) with a `fock_hook` but no
`fock_hook_grad` is a `ValueError` before the SCF.

A hooked run records a `fock-hook` approximation in its certificate.

## Numerical anchors {#fock-hooks-numerical-anchors}

| Check | Agreement | Test |
|---|---|---|
| a fixed point charge entered as a hook, against the same charge as `RhfConfig.external_potential` (an independent construction of the same functional): RHF and RKS (LDA) water, UHF and ROHF NH2, 6-31G | max $\lvert\Delta g\rvert$ from 1.2e-14 (RHF) to 6.2e-11 (UHF) Ha/Bohr; bar 2e-7 Ha/Bohr | `crates/ferric-scf/tests/fock_hook_gradient.rs` (`point_charge_hook_gradient_matches_external_potential_{rhf,rks,uhf,rohf}`) |
| the constant hook $\lambda\operatorname{tr}[DS]$ leaves the gradient unchanged; hooks built from $S$, $h$ and $D^2$ against finite differences of the hooked energy, including a KS case | asserted per test | `crates/ferric-python/tests/test_fock_hook_gradient.py` (`test_constant_trace_hook_leaves_the_gradient_unchanged`, `test_hooked_gradient_matches_finite_differences`, `test_hooked_ks_gradient_matches_finite_differences`) |

## Status {#fock-hooks-status}

The gradient's derivation, its proof script and its verified equation are in
section @sec:fock-hook-gradient. This section adds the hooked functional,
where the hook enters the loop, and what is refused.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/src/rhf.rs` | the entry point and the refusal list |
| `crates/ferric-python/python/ferric/ferric.pyi` | the Python signatures |
| section @sec:fock-hook-gradient | the gradient derivation and its proof |
