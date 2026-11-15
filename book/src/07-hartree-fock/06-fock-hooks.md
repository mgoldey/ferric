# 7.6 Fock hooks

<!-- section: 7.6; status: edit; registry: ;
     theory: crates/ferric-scf/theory.md#"Fock hooks and their nuclear gradient" -->

A Fock hook adds one caller-supplied term to the energy functional without
changing the solver. It is how Python prototypes a new one-body or
density-functional term (an embedding potential, a penalty, a relativistic
correction) before it has a Rust implementation.

| Symbol | Meaning | Code name |
|---|---|---|
| $E_h[D]$ | the hook's energy | the first value the hook returns |
| $V = \partial E_h/\partial D$ | the hook's potential, added to $F$ | the second value |
| $\partial E_h/\partial\mathbf R\vert_D$ | the explicit nuclear derivative at fixed $D$ | `fock_hook_grad` |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/rhf.rs::{solve_rhf_with_hook,validate_fock_hook}`, `ferric-scf/src/uhf.rs::solve_uhf_with_hook`, `ferric-scf/src/rohf.rs::solve_rohf_with_hook` |
| Python | `fock_hook=` on `run_rhf`, `run_dft`, `run_uhf`, `run_rohf`; `fock_hook_grad=` with `with_gradient=True` (`crates/ferric-python/ferric.pyi`) |
| CLI | none |

| Solver | Hook signature |
|---|---|
| RHF, RKS | `hook(D_total) -> (E, V)` |
| UHF, ROHF | `hook(D_a, D_b) -> (E, V_a, V_b)` |

## 7.6.1 The equations

<!-- include: crates/ferric-scf/theory.md#"Fock hooks and their nuclear gradient" -->

## 7.6.2 Where the hook enters the iteration

$V$ is added to the Fock matrix before DIIS sees it, and $E_h$ is added
outside the $\tfrac12\operatorname{tr}[D(h+F)]$ trace, as $E_{xc}$ is. The
converged energy is therefore stationary in the hooked functional when $V$ is
the derivative of $E_h$. Nothing checks that it is: a $V$ that is not
$\partial E_h/\partial D$ converges to a self-consistent solution whose energy
is not variational, and its gradient is wrong.

## 7.6.3 What is refused

Every setting that builds an orbital Hessian or a blocked solve would not see
the hook, so `validate_fock_hook` refuses it, naming the field:

| Field | Reason |
|---|---|
| `symmetry`, `irrep_nelec` | the point-group-blocked solve does not call the hook |
| `newton_trigger` | the Newton orbital Hessian does not include the hook |
| `ah_trigger` | the augmented-Hessian step does not include the hook |
| `aurora.enabled` | the AURORA curvature model does not include the hook |
| `trah_trigger` | the TRAH orbital Hessian does not include the hook |
| `check_stability`, `scf_stability_descent` | the stability Hessian does not include the hook |

In Python, a gradient run (`with_gradient=True`) with a `fock_hook` but no
`fock_hook_grad` is a `ValueError` before the SCF. A hooked run records a `fock-hook` approximation in its
certificate.

## 7.6.4 Numerical anchors

`crates/ferric-scf/tests/fock_hook_gradient.rs` checks the three-term gradient
of the included section with a point-charge hook: the hooked gradient must
equal the gradient of the same point charge supplied as an external potential,
for RHF and RKS. That is an independent construction of the same functional.

## 7.6.5 Status

The section stays `edit`. The hook's gradient has a committed derivation and
an independent-construction test, but no sympy proof script: the identity
that a hook depending on $D$ alone has a zero explicit term, and the
$\lambda\operatorname{tr}[DS]$ cancellation of the included derivation, are
the natural claims for one.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/theory.md`, "Fock hooks and their nuclear gradient" | the equations |
| `crates/ferric-scf/src/rhf.rs` | the entry point and the refusal list |
| `crates/ferric-python/ferric.pyi` | the Python signatures |
