# Fock hooks {#fock-hooks}

<!-- status: edit; registry: ;
     theory: crates/ferric-scf/theory.md#"Fock hooks and their nuclear gradient" -->

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
| Python | `fock_hook=` on `run_rhf`, `run_dft`, `run_uhf`, `run_rohf`; `fock_hook_grad=` with `with_gradient=True` (`crates/ferric-python/ferric.pyi`) |
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

$$ E(\mathbf R) = E_0[\mathbf D;\mathbf R] + E_h[\mathbf D;\mathbf R], \qquad \mathbf V = \partial E_h/\partial\mathbf D . $$

$E_0$ is the built-in HF or KS energy. $\mathbf V$ is added to the Fock
matrix every iteration, with $\mathbf V_\sigma = \partial E_h/\partial\mathbf D_\sigma$
per spin for UHF and ROHF. Why the derivative? Section @sec:energy-functionals showed that the
Fock matrix is $\partial E/\partial D$; differentiating the sum gives
$F = F_0 + \mathbf V$, so the SCF that diagonalizes the hooked $F$ makes the
hooked $E$ stationary.

The SCF makes $E$ stationary under orbital rotations, subject to
orthonormality $\mathbf C^{\mathsf T}\mathbf S\mathbf C = \mathbf 1$. Its
Lagrangian is

$$ \mathcal L = E[\mathbf D(\mathbf C);\mathbf R] - \sum_{ij}\epsilon_{ij}\left(\mathbf C^{\mathsf T}\mathbf S(\mathbf R)\mathbf C - \mathbf 1\right)_{ij} . $$

At convergence $\partial\mathcal L/\partial\mathbf C = 0$ gives
$\mathbf F\mathbf C_{\rm occ} = \mathbf S\mathbf C_{\rm occ}\boldsymbol\epsilon$
with the hooked Fock matrix $\mathbf F = \mathbf F_0 + \mathbf V$.

### The nuclear gradient

Move a nucleus. $\mathcal L$ depends on $\mathbf R$ directly and through
$\mathbf C$, but $\partial\mathcal L/\partial\mathbf C = 0$ at convergence, so
the orbital response drops out. The total derivative is the partial
derivative of $\mathcal L$ at fixed $\mathbf C$:

$$ \frac{dE}{d\mathbf R} = \left.\frac{\partial E_0}{\partial\mathbf R}\right|_{\mathbf D} + \left.\frac{\partial E_h}{\partial\mathbf R}\right|_{\mathbf D} - \operatorname{Tr}\!\left[\mathbf W\,\frac{\partial\mathbf S}{\partial\mathbf R}\right], \qquad \mathbf W = \sum_\sigma\mathbf C_{\sigma,\rm occ}\boldsymbol\epsilon_\sigma\mathbf C_{\sigma,\rm occ}^{\mathsf T} . $$

The closed-shell $\mathbf W$ carries the factor 2 of the double occupation:
$\mathbf W = 2\mathbf C_{\rm occ}\boldsymbol\epsilon\mathbf C_{\rm occ}^{\mathsf T} = \tfrac12\mathbf D\mathbf F\mathbf D$.
To see the second form, substitute $\mathbf D = 2\mathbf C_{\rm occ}\mathbf C_{\rm occ}^{\mathsf T}$
and use $\mathbf C_{\rm occ}^{\mathsf T}\mathbf F\mathbf C_{\rm occ} = \boldsymbol\epsilon$:
$\tfrac12\mathbf D\mathbf F\mathbf D = 2\mathbf C_{\rm occ}\boldsymbol\epsilon\mathbf C_{\rm occ}^{\mathsf T}$.

The three pieces have three owners:

1. $\partial E_0/\partial\mathbf R|_{\mathbf D}$: the built-in gradient's
   integral-derivative terms (nuclear repulsion,
   $\operatorname{Tr}[\mathbf D\,\partial\mathbf h/\partial\mathbf R]$, the
   two-electron and XC derivatives) at the hooked density. Unchanged code.
2. The Pulay term $-\operatorname{Tr}[\mathbf W\,\partial\mathbf S/\partial\mathbf R]$.
   $\mathbf W$ is built from the converged orbitals and eigenvalues, which
   diagonalize the hooked $\mathbf F$, so the $\mathbf V$ part of the Lagrange
   multipliers is included without the hook doing anything. Since
   $\mathbf W = \tfrac12\mathbf D\mathbf F\mathbf D$ is linear in
   $\mathbf F$, its $\mathbf V$ part is $\tfrac12\mathbf D\mathbf V\mathbf D$
   (closed shell). For ROHF,
   $\mathbf W = \operatorname{sym}(\mathbf D_\alpha\mathbf F_\alpha\mathbf D_\alpha + \mathbf D_\alpha\mathbf F_\beta\mathbf D_\beta)$
   uses the converged spin Fock matrices, which hold $\mathbf V_\alpha$,
   $\mathbf V_\beta$.
3. The explicit term $\partial E_h/\partial\mathbf R|_{\mathbf D}$, the
   derivative of the hook's energy with the AO density matrix held fixed. Only
   the hook knows it, so the caller supplies it (`fock_hook_grad=` in Python,
   `gradient::with_fock_hook_gradient` in Rust).

### What the explicit term must and must not include

It must include every dependence of $E_h$ on the nuclear positions at fixed
AO density matrix: integrals over AO functions centred on the atoms (the
derivative of $\langle\mu|\hat v|\nu\rangle$ with respect to the centres of
$\mu$ and $\nu$), operator centres that sit on atoms, and purely classical
terms such as a charge–nuclear repulsion. For example, a point charge $q$ at
fixed $\mathbf r_c$ has
$E_h = \operatorname{Tr}[\mathbf D\mathbf V_q] + \sum_AqZ_A/|\mathbf R_A - \mathbf r_c|$.
Its explicit term is $\operatorname{Tr}[\mathbf D\,\partial\mathbf V_q/\partial\mathbf R_A]$
(basis-centre derivatives only; the charge does not move) plus the derivative
of the classical sum.

It must not include any change of $\mathbf D$ with geometry (the response of
the orbitals is handled by the Lagrangian above), nor the Pulay term of
$\mathbf V$ (it is already in $\mathbf W$). A hook that depends only on
$\mathbf D$, for example $E_h = \lambda\sum_{\mu\nu}D_{\mu\nu}^2$, has an
explicit term of zero.

A hook built from the overlap or another geometry-dependent AO matrix
depends on $\mathbf R$ through that matrix. Take
$E_h = \lambda\operatorname{Tr}[\mathbf D\mathbf S]$. Its potential is
$\mathbf V = \lambda\mathbf S$ and its explicit term is
$\lambda\operatorname{Tr}[\mathbf D\,\partial\mathbf S/\partial\mathbf R]$.
Its Pulay part is
$-\operatorname{Tr}[\tfrac12\lambda\mathbf D\mathbf S\mathbf D\,\partial\mathbf S/\partial\mathbf R]$.
For the closed shell $\mathbf D\mathbf S\mathbf D = 2\mathbf D$, because
$\mathbf C_{\rm occ}^{\mathsf T}\mathbf S\mathbf C_{\rm occ} = \mathbf 1$, so
the Pulay part is $-\lambda\operatorname{Tr}[\mathbf D\,\partial\mathbf S/\partial\mathbf R]$
and cancels the explicit term exactly. It must:
$\lambda\operatorname{Tr}[\mathbf D\mathbf S] = \lambda N$ is a constant.
The tests use it as their exactness anchor: both halves nonzero, their sum
known in closed form.

The formula holds for the converged hooked solution only (stationarity was
used). It assumes integer aufbau occupations of the hooked $\mathbf F$, as
the un-hooked gradients do, and that $\mathbf V$ is exactly
$\partial E_h/\partial\mathbf D$. A hook whose $\mathbf V$ is not the
derivative of its $E_h$ converges to a non-stationary point of $E$, and no
gradient of the reported energy follows from this formula.

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
| `symmetry`, `irrep_nelec` | the point-group-blocked solve does not call the hook |
| `newton_trigger` | the Newton orbital Hessian does not include the hook |
| `ah_trigger` | the augmented-Hessian step does not include the hook |
| `aurora.enabled` | the AURORA curvature model does not include the hook |
| `trah_trigger` | the TRAH orbital Hessian does not include the hook |
| `check_stability`, `scf_stability_descent` | the stability Hessian does not include the hook |

Each of these needs the response of the potential to a change of the
density, $\partial V/\partial D$, which the hook interface does not return.

In Python, a gradient run (`with_gradient=True`) with a `fock_hook` but no
`fock_hook_grad` is a `ValueError` before the SCF. A hooked run records a `fock-hook` approximation in its
certificate.

## Numerical anchors {#fock-hooks-numerical-anchors}

`crates/ferric-scf/tests/fock_hook_gradient.rs` checks the three-term gradient
of @sec:fock-hooks-equations with a point-charge hook: the hooked gradient must equal the gradient
of the same point charge supplied as an external potential, for RHF and RKS.
That is an independent construction of the same functional.

## Status {#fock-hooks-status}

The section stays `edit`. The hook's gradient has a committed derivation and
an independent-construction test, but no sympy proof script: the identity
that a hook depending on $D$ alone has a zero explicit term, and the
$\lambda\operatorname{tr}[DS]$ cancellation of @sec:fock-hooks-equations, are the natural claims
for one. The steps $\mathbf W = \tfrac12\mathbf D\mathbf F\mathbf D$ and
$\mathbf D\mathbf S\mathbf D = 2\mathbf D$ were checked numerically on a
random model for this section.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/theory.md`, "Fock hooks and their nuclear gradient" | the equations |
| `crates/ferric-scf/src/rhf.rs` | the entry point and the refusal list |
| `crates/ferric-python/ferric.pyi` | the Python signatures |
