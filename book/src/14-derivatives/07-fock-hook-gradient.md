# Fock-hook gradients {#fock-hook-gradient}

<!-- status: edit; registry: none -->

A Fock hook adds your own term to the SCF energy: at every iteration the SCF
calls your function with the current density matrix, and you return an
energy $E_h[D;\mathbf R]$ and its density derivative
$V = \partial E_h/\partial D$, which ferric adds to the Fock matrix. To
optimize a geometry with the hook you need its nuclear gradient, and most of
it ferric already has. The built-in gradient of section @sec:hf-gradient is built from
the converged orbitals and eigenvalues of the hooked Fock matrix, so
everything the hook does through the density is already in it. What ferric
cannot know is how your $E_h$ depends on the nuclei when the density is held
fixed, through integrals over atom-centred functions, operators on atoms or
classical terms. You supply that one term.

| Symbol | Meaning | Code name |
|---|---|---|
| $E_h$, $V$ | the hook's energy and its density derivative | the hook callback |
| $\partial E_h/\partial\mathbf R\vert_D$ | explicit term at fixed AO density | `explicit`, `fock_hook_grad=` |
| $W$ | energy-weighted density of the hooked orbitals | as in section @sec:hf-gradient |

| Entry point | Location |
|---|---|
| Rust, SCF | `ferric-scf/src/rhf.rs::solve_rhf_with_hook`, `ferric-scf/src/uhf.rs::solve_uhf_with_hook`, `ferric-scf/src/rohf.rs::solve_rohf_with_hook` |
| Rust, gradient | `ferric-scf/src/gradient.rs::with_fock_hook_gradient`, added to the result of `restricted_scf_gradient` or its open-shell siblings |
| Python | `fock_hook=` and `fock_hook_grad=` on `run_rhf`, `run_uhf`, `run_rohf`, `run_dft`, `run_optimize` |

In Python, `run_rhf` and `run_dft` call `hook(D)` with the total density and
expect `(E, V)`; `run_uhf` and `run_rohf` call `hook(D_a, D_b)` and expect
`(E, V_a, V_b)`. The gradient callback is `grad(D, mol)`, or
`grad(D_a, D_b, mol)` for the open shell, and returns $\partial
E_h/\partial\mathbf R|_D$ as an array of shape `(natoms, 3)` in
Hartree/Bohr.

## The gradient {#fock-hook-gradient-gradient}

### The Lagrangian

The hook adds one term to the energy functional,

$$ E(\mathbf{R}) = E_0[\mathbf{D}; \mathbf{R}] + E_h[\mathbf{D}; \mathbf{R}],
\qquad \mathbf{V} = \partial E_h / \partial \mathbf{D} , $$

where $E_0$ is the built-in HF or KS energy. $\mathbf V$ is added to the Fock
matrix every iteration, and for UHF and ROHF there is one per spin,
$\mathbf{V}_\sigma = \partial E_h/\partial\mathbf{D}_\sigma$.

The SCF makes $E$ stationary under orbital rotations, subject to
orthonormality $\mathbf{C}^\top \mathbf{S} \mathbf{C} = \mathbf{1}$. Enforce
the constraint with multipliers $\epsilon_{ij}$:

$$ \mathcal{L} = E[\mathbf{D}(\mathbf{C}); \mathbf{R}]
 - \sum_{ij} \epsilon_{ij} \left(\mathbf{C}^\top \mathbf{S}(\mathbf{R}) \mathbf{C}
 - \mathbf{1}\right)_{ij} . $$

Set $\partial\mathcal L/\partial\mathbf C = 0$. The derivative of $E$ with
respect to $\mathbf D$ is the Fock matrix, and since $E$ contains $E_h$ that
is the hooked Fock matrix $\mathbf{F} = \mathbf{F}_0 + \mathbf{V}$. The
result is the Roothaan equation with that matrix:
$\mathbf{F}\mathbf{C}_\text{occ} = \mathbf{S}\mathbf{C}_\text{occ}\boldsymbol{\epsilon}$.

Now differentiate with respect to $\mathbf R$. At the solution
$\mathcal L = E$, and $\mathcal L$ is stationary in $\mathbf C$, so the change
of the orbitals with geometry drops out. The total derivative is the
partial derivative of $\mathcal L$ at fixed $\mathbf C$. Three terms depend on
$\mathbf R$ explicitly: $E_0$, $E_h$, and $\mathbf S$ in the constraint.

$$ \frac{dE}{d\mathbf{R}} =
 \left.\frac{\partial E_0}{\partial \mathbf{R}}\right|_{\mathbf{D}}
 + \left.\frac{\partial E_h}{\partial \mathbf{R}}\right|_{\mathbf{D}}
 - \operatorname{Tr}\!\left[\mathbf{W}\, \frac{\partial \mathbf{S}}{\partial \mathbf{R}}\right],
\qquad
\mathbf{W} = \sum_\sigma \mathbf{C}_{\sigma,\text{occ}} \boldsymbol{\epsilon}_\sigma \mathbf{C}_{\sigma,\text{occ}}^\top . $$

The closed-shell $\mathbf W$ carries the factor 2 of the double occupation,
$\mathbf{W} = 2\mathbf{C}_\text{occ}\boldsymbol{\epsilon}\mathbf{C}_\text{occ}^\top$.
To write it with the density, use $\mathbf D = 2\mathbf C_\text{occ}\mathbf C_\text{occ}^\top$
and $\mathbf C_\text{occ}^\top\mathbf F\mathbf C_\text{occ} = \boldsymbol\epsilon$
(multiply the Roothaan equation by $\mathbf C_\text{occ}^\top$ and use
orthonormality). Then
$\mathbf{D}\mathbf{F}\mathbf{D} = 4\mathbf C_\text{occ}\boldsymbol\epsilon\mathbf C_\text{occ}^\top$, so
$\mathbf{W} = \tfrac12 \mathbf{D}\mathbf{F}\mathbf{D}$.

### Three pieces, three owners

1. $\partial E_0 / \partial \mathbf{R}|_{\mathbf{D}}$ is the built-in
   gradient's integral-derivative terms: nuclear repulsion,
   $\operatorname{Tr}[\mathbf{D}\,\partial \mathbf{h}/\partial \mathbf{R}]$,
   the two-electron and XC derivatives, all evaluated at the hooked density.
   The code is unchanged.
2. The Pulay term $-\operatorname{Tr}[\mathbf{W}\,\partial\mathbf{S}/\partial\mathbf{R}]$.
   $\mathbf{W}$ is built from the converged orbitals and eigenvalues, which
   diagonalize the hooked $\mathbf{F}$, so the $\mathbf{V}$ part of the
   Lagrange multipliers is included without the hook doing anything. By
   $\mathbf W = \tfrac12\mathbf D\mathbf F\mathbf D$, that $\mathbf V$ part is
   $\tfrac12\mathbf{D}\mathbf{V}\mathbf{D}$ for a closed shell. For ROHF,
   $\mathbf{W} = \operatorname{sym}(\mathbf{D}_\alpha\mathbf{F}_\alpha\mathbf{D}_\alpha +
   \mathbf{D}_\alpha\mathbf{F}_\beta\mathbf{D}_\beta)$ uses the converged spin
   Fock matrices, which hold $\mathbf{V}_\alpha$ and $\mathbf{V}_\beta$.
3. The explicit term $\partial E_h / \partial \mathbf{R}|_{\mathbf{D}}$, the
   derivative of the hook's energy with the AO density matrix held fixed.
   Only the hook knows it, so the caller supplies it: `fock_hook_grad=` in
   Python, `ferric-scf/src/gradient.rs::with_fock_hook_gradient` in Rust.

### What the explicit term must and must not include

It must include every dependence of $E_h$ on the nuclear positions at fixed
AO density matrix:

- integrals over AO functions centred on the atoms, that is the derivative of
  $\langle\mu|\hat{v}|\nu\rangle$ with respect to the centres of $\mu$ and
  $\nu$;
- operator centres that sit on atoms;
- classical terms, such as a charge–nuclear repulsion.

For example, a point charge $q$ at a fixed position $\mathbf{r}_c$ has
$E_h = \operatorname{Tr}[\mathbf{D}\mathbf{V}_q] + \sum_A q Z_A / |\mathbf{R}_A - \mathbf{r}_c|$.
Its explicit term is
$\operatorname{Tr}[\mathbf{D}\,\partial\mathbf{V}_q/\partial\mathbf{R}_A]$,
with basis-centre derivatives only because the charge does not move, plus
the derivative of the classical sum.

It must not include any change of $\mathbf{D}$ with geometry: the response of
the orbitals is handled by the Lagrangian above. Nor may it include the
Pulay term of $\mathbf{V}$, which is already in $\mathbf{W}$. A hook that
depends only on $\mathbf{D}$, for example
$E_h = \lambda \sum_{\mu\nu} D_{\mu\nu}^2$, has an explicit term of zero.

A hook built from the overlap or another geometry-dependent AO matrix
depends on $\mathbf R$ through that matrix. That case is the test below.

> **Verified equation {#eq:scf-hook-gradient}** · the hooked gradient
>
> On a three-function closed-shell model whose $h$, $S$ and $(\mu\nu\vert\lambda\sigma)$ depend on a
> parameter, with the hook $E_h = \kappa\,\mathrm{tr}[DB] + \lambda\,\mathrm{tr}[DS] + \tfrac\gamma2\sum D^2_{\mu\nu}$,
> the formula $\mathrm dE/\mathrm dx = \mathrm{tr}[Dh^x] + \sum\Gamma(\mu\nu\vert\lambda\sigma)^x + \partial E_h/\partial x\vert_D - \mathrm{tr}[W S^x]$,
> with $W$ from the hooked orbitals and eigenvalues, equals a fourth-order finite
> difference of the re-solved hooked SCF energy (claim K1). Dropping the explicit term, or building $W$
> from the un-hooked Fock matrix, misses (claim K2). With $E_h = \lambda\,\mathrm{tr}[DS]$ alone the explicit
> term and the hook's part of the Pulay term cancel, because $DSD = 2D$ for a normalized
> doubly occupied orbital, so the hooked gradient equals the plain one (claim K3).
>
> | | |
> |---|---|
> | Proof | `scripts/fock_hook_gradient_sympy.py`, claims K1, K2 and K3 |
> | Negative control | the runner drops the $\lambda\,\mathrm{tr}[DS^x]$ part of the explicit term and, separately, builds $W$ from the un-hooked Fock matrix (claim K1 fails each time) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/gradient.rs::with_fock_hook_gradient` |
> | Code test | `point_charge_hook_gradient_matches_external_potential_rhf` in `crates/ferric-scf/tests/fock_hook_gradient.rs` |
>
> <!-- verified: id=scf.hook.gradient; script=scripts/fock_hook_gradient_sympy.py; claim=K1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/gradient.rs::with_fock_hook_gradient; rust_test=ferric-scf/tests/fock_hook_gradient.rs::point_charge_hook_gradient_matches_external_potential_rhf -->

### The exactness anchor

Take $E_h = \lambda\,\operatorname{Tr}[\mathbf D\mathbf S]$ (claim K3). For a
normalized density $\operatorname{Tr}[\mathbf D\mathbf S] = N$, the number of
electrons, so $E_h = \lambda N$ is a constant. The hook shifts the energy and
must leave the gradient unchanged.

Work out both halves. The explicit term is
$\lambda\operatorname{Tr}[\mathbf D\,\partial\mathbf S/\partial\mathbf R]$.
The hook's $\mathbf V$ is $\lambda\mathbf S$, so its part of the Pulay term is

$$ -\operatorname{Tr}\big[\tfrac12\lambda\,\mathbf D\mathbf S\mathbf D\,\partial\mathbf S/\partial\mathbf R\big] = -\lambda\operatorname{Tr}\big[\mathbf D\,\partial\mathbf S/\partial\mathbf R\big] , $$

using $\mathbf D\mathbf S\mathbf D = 2\mathbf D$. That identity is
$\mathbf C_\text{occ}^\top\mathbf S\mathbf C_\text{occ} = \mathbf 1$ again:
$\mathbf D\mathbf S\mathbf D = 4\mathbf C_\text{occ}(\mathbf C_\text{occ}^\top\mathbf S\mathbf C_\text{occ})\mathbf C_\text{occ}^\top = 2\mathbf D$.
The two halves cancel exactly.

Why is this a good test? Both halves are nonzero and their sum is known in
closed form, zero. A gradient that dropped the explicit term, or built
$\mathbf W$ from the un-hooked Fock matrix, would be off by
$\lambda\operatorname{Tr}[\mathbf D\,\partial\mathbf S/\partial\mathbf R]$,
about $10^{-2}$ Ha/Bohr at $\lambda = 0.1$. The Python tests build this
constant-trace hook
(`test_constant_trace_hook_leaves_the_gradient_unchanged` in
`crates/ferric-python/tests/test_fock_hook_gradient.py`), and an optimization
with it must reach the plain geometry
(`test_optimize_with_a_constant_hook_reaches_the_plain_geometry`).

### When the formula holds

The formula holds for the converged hooked solution only, because
stationarity of $\mathcal L$ was used. It assumes integer aufbau occupations
of the hooked $\mathbf{F}$, as the un-hooked gradients do. It also needs
$\mathbf{V}$ to be exactly $\partial E_h/\partial\mathbf{D}$. A hook whose
$\mathbf{V}$ is not the derivative of its $E_h$ converges to a point where
$E$ is not stationary, and no gradient of the reported energy follows from
this formula.

## The algorithm {#fock-hook-gradient-algorithm}

| Step | Work | Function |
|---|---|---|
| 1 | SCF with the hook added to $F$ at each iteration; UHF and ROHF hooks return $V_\sigma$ per spin | `solve_rhf_with_hook`, `solve_uhf_with_hook`, `solve_rohf_with_hook` |
| 2 | the built-in gradient at the hooked result: $W$ from the hooked orbitals, so the $V$ part of the multipliers is included with nothing done by the hook | `ferric-scf/src/gradient.rs::restricted_scf_gradient` and siblings |
| 3 | add the caller's explicit term, after checking its shape is `(natoms, 3)` and that every entry is finite | `ferric-scf/src/gradient.rs::with_fock_hook_gradient` |

For ROHF, $W = \mathrm{sym}(D_\alpha F_\alpha D_\alpha + D_\alpha F_\beta D_\beta)$
uses the converged spin Fock matrices. Those must be recorded after the hook
is added, or $W$ lacks $V$. A mutation that records them before is one of
the checks of the Rust test.

Step 3 is `with_fock_hook_gradient`. Its last lines, after the shape check,
are:

<!-- excerpt: ferric-scf/src/gradient.rs::with_fock_hook_gradient -->
```rust
if explicit.iter().any(|v| !v.is_finite()) {
    return Err(FerricError::General(
        "Fock-hook gradient: the explicit term is not finite".into(),
    ));
}
Ok(grad + explicit)
```

The addition is the second term of $dE/d\mathbf R$. The checks before it,
this one and the shape check above it, turn a mis-shaped or non-finite
explicit term into an error instead of a wrong optimization that runs to
completion.

## Reproducibility and refusals {#fock-hook-gradient-reproducibility-refusals}

- A hooked gradient request without `fock_hook_grad` is an error naming the
  keyword (`test_a_hooked_gradient_without_fock_hook_grad_names_the_kwarg`);
  `fock_hook_grad` without a hook, or without a gradient request, is an error.
- A hook whose explicit term raises, has the wrong shape or is not finite is an
  error that reaches the caller.

## Options that change the equation {#fock-hook-gradient-options-change-equation}

| Option | Effect |
|---|---|
| closed shell | one $V$; the closed-shell $W = \tfrac12DFD$ |
| UHF, ROHF | $V_\sigma = \partial E_h/\partial D_\sigma$ per spin |
| Kohn–Sham | the hook adds to the KS Fock matrix. The tests cover RKS with an LDA functional and, in Python, `run_dft` |
| optimization | `run_optimize(..., fock_hook=, fock_hook_grad=)`; with a geometry-dependent hook the optimum is stationary on the hooked surface (`test_optimize_with_a_geometry_dependent_hook_reaches_a_stationary_point`) |
| relativistic X2C | the gradient of an X2C SCF is refused (section @sec:hf-gradient), whether or not a hook is present |

## Numerical anchors {#fock-hook-gradient-numerical-anchors}

| Check | Reference | Agreement | Test |
|---|---|---|---|
| a fixed point charge entered as a hook $V = h_{\mathrm{ext}} - h$, against the same charge as `RhfConfig.external_potential`: RHF and RKS (LDA) water, UHF and ROHF NH2, all 6-31G | the built-in external-potential gradient | energy at most $2.8\times10^{-14}$ Ha; gradient at most $1.2\times10^{-14}$ (RHF), $4.7\times10^{-15}$ (RKS), $6.2\times10^{-11}$ (UHF), $2.0\times10^{-11}$ (ROHF) Ha/Bohr; bar $2\times10^{-7}$ Ha/Bohr, the density-convergence bound | `crates/ferric-scf/tests/fock_hook_gradient.rs` |
| the same test with the explicit term dropped | | misses by 3.7e-2 to 4.7e-2 Ha/Bohr, so the test sees the bug it guards | same |
| the same test with the ROHF spin Fock matrices recorded before the hook, so $W$ lacks $V$ | | misses by 3.0e-2 Ha/Bohr | same |
| hooked gradient against finite differences of the hooked energy, including a KS case | finite differences of the hooked energy | `test_hooked_gradient_matches_finite_differences` and `test_hooked_ks_gradient_matches_finite_differences` assert it | `crates/ferric-python/tests/test_fock_hook_gradient.py` |

The point-charge row is a second independent route. ferric has a built-in
gradient for an external point charge, so the same physics entered as a
hook must give the same number. It does, to rounding for RHF and RKS.

The first three rows are recorded in the header of
`crates/ferric-scf/tests/fock_hook_gradient.rs`; the hook is not graded on the
validation page. The status of this section is `edit`: the derivation is
tested and proved by `scripts/fock_hook_gradient_sympy.py`, but the registry
has no entry for it.

## Sources

| Source | Used for |
|---|---|
| section @sec:fock-hooks-equations | the derivation and the rules for the explicit term |
| `scripts/fock_hook_gradient_sympy.py`, claims K1 to K3 | verified equation @eq:scf-hook-gradient |
| `crates/ferric-scf/tests/fock_hook_gradient.rs` | the point-charge anchor and its mutations |
| `crates/ferric-python/tests/test_fock_hook_gradient.py` | the exactness anchor and the error cases |
| `crates/ferric-python/ferric.pyi`, the `FockHook` and `FockHookGrad` aliases | the Python call signatures |
