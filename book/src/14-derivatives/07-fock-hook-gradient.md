# 14.7 Fock-hook gradients

<!-- section: 14.7; status: edit; registry: none;
     theory: crates/ferric-scf/theory.md#"Fock hooks and their nuclear gradient" -->

A Fock hook adds one caller-supplied term $E_h[D;\mathbf R]$ to the SCF energy
and its derivative $V = \partial E_h/\partial D$ to the Fock matrix at every
iteration. The built-in gradient of section 14.1 already contains everything
about the hook that depends on the density, because it is built from the
converged orbitals and eigenvalues of the hooked Fock matrix. What it cannot
know is the hook's explicit dependence on the nuclei at a fixed density. The
caller supplies that one term.

| Symbol | Meaning | Code name |
|---|---|---|
| $E_h$, $V$ | the hook's energy and its density derivative | the hook callback |
| $\partial E_h/\partial\mathbf R\vert_D$ | explicit term at fixed AO density | `explicit`, `fock_hook_grad=` |
| $W$ | energy-weighted density of the hooked orbitals | as in section 14.1 |

| Entry point | Location |
|---|---|
| Rust, SCF | `ferric-scf/src/rhf.rs::solve_rhf_with_hook`, `ferric-scf/src/uhf.rs::solve_uhf_with_hook`, `ferric-scf/src/rohf.rs::solve_rohf_with_hook` |
| Rust, gradient | `ferric-scf/src/gradient.rs::with_fock_hook_gradient`, added to the result of `restricted_scf_gradient` or its open-shell siblings |
| Python | `fock_hook=` and `fock_hook_grad=` on `run_rhf`, `run_uhf`, `run_rohf`, `run_dft`, `run_optimize` |

## 14.7.1 The gradient

<!-- include: crates/ferric-scf/theory.md#"Fock hooks and their nuclear gradient" -->

> **Verified equation 14.19** · the hooked gradient
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

**The exactness anchor** (claim K3). Take $E_h = \lambda\,\mathrm{tr}[D S]$. Its explicit term
is $\lambda\,\mathrm{tr}[D\,\partial S/\partial\mathbf R]$, and the hook's part of
the Pulay term is $-\tfrac12\lambda\,\mathrm{tr}[DSD\,\partial S/\partial\mathbf R]
= -\lambda\,\mathrm{tr}[D\,\partial S/\partial\mathbf R]$, since $DSD = 2D$ for a
closed shell. They cancel exactly, as they must, because
$\lambda\,\mathrm{tr}[DS] = \lambda N$ is a constant. Both halves are nonzero and
their sum is known in closed form, which is why the Python tests build the
constant-trace hook (`test_constant_trace_hook_leaves_the_gradient_unchanged` in
`crates/ferric-python/tests/test_fock_hook_gradient.py`) and an optimization with a
constant hook must reach the plain geometry
(`test_optimize_with_a_constant_hook_reaches_the_plain_geometry`).

**What the caller must and must not include.** It must include every dependence
of $E_h$ on the nuclei at fixed AO density matrix: basis-centre derivatives of
integrals the hook uses, operator centres that sit on atoms, and classical
terms. It must not include any change of $D$ with geometry or the Pulay term
of $V$. A hook that depends only on $D$ has an explicit term of zero. The
formula holds for a converged hooked solution with integer aufbau
occupations, and only when $V$ is exactly $\partial E_h/\partial D$: a hook whose
$V$ is not the derivative of its $E_h$ converges to a non-stationary point of
$E$, and no gradient of the reported energy follows.

## 14.7.2 The algorithm

| Step | Work | Function |
|---|---|---|
| 1 | SCF with the hook added to $F$ at each iteration; UHF and ROHF hooks return $V_\sigma$ per spin | `solve_rhf_with_hook`, `solve_uhf_with_hook`, `solve_rohf_with_hook` |
| 2 | the built-in gradient at the hooked result: $W$ from the hooked orbitals, so the $V$ part of the multipliers is included with nothing done by the hook | `ferric-scf/src/gradient.rs::restricted_scf_gradient` and siblings |
| 3 | add the caller's explicit term, after checking its shape is `(natoms, 3)` and that every entry is finite | `ferric-scf/src/gradient.rs::with_fock_hook_gradient` |

For ROHF, $W = \mathrm{sym}(D_\alpha F_\alpha D_\alpha + D_\alpha F_\beta D_\beta)$ uses the
converged spin Fock matrices, which hold $V_\alpha$ and $V_\beta$. The spin Fock
matrices must be recorded after the hook is added. A mutation that records them
before is one of the checks of the Rust test.

The third step is one line.

<!-- excerpt: ferric-scf/src/gradient.rs::with_fock_hook_gradient -->
```rust
Ok(grad + explicit)
```

The work is in the two checks before it, which turn a mis-shaped or non-finite
explicit term into an error instead of a silently wrong optimization.

## 14.7.3 Reproducibility and refusals

- A hooked gradient request without `fock_hook_grad` is an error naming the
  keyword (`test_a_hooked_gradient_without_fock_hook_grad_names_the_kwarg`);
  `fock_hook_grad` without a hook, or without a gradient request, is an error.
- A hook whose explicit term raises, has the wrong shape or is not finite is an
  error that reaches the caller.

## 14.7.4 Options that change the equation

| Option | Effect |
|---|---|
| closed shell | one $V$; the closed-shell $W = \tfrac12DFD$ |
| UHF, ROHF | $V_\sigma = \partial E_h/\partial D_\sigma$ per spin |
| Kohn–Sham | the hook adds to the KS Fock matrix. The tests cover RKS with an LDA functional and, in Python, `run_dft` |
| optimization | `run_optimize(..., fock_hook=, fock_hook_grad=)`; with a geometry-dependent hook the optimum is stationary on the hooked surface (`test_optimize_with_a_geometry_dependent_hook_reaches_a_stationary_point`) |
| relativistic X2C | the gradient of an X2C SCF is refused (section 14.1), whether or not a hook is present |

## 14.7.5 Numerical anchors

| Check | Reference | Agreement | Test |
|---|---|---|---|
| a fixed point charge entered as a hook $V = h_{\mathrm{ext}} - h$, against the same charge as `RhfConfig.external_potential`: RHF and RKS (LDA) water, UHF and ROHF NH2, all 6-31G | the built-in external-potential gradient | energy at most $2.8\times10^{-14}$ Ha; gradient at most $1.2\times10^{-14}$ (RHF), $4.7\times10^{-15}$ (RKS), $6.2\times10^{-11}$ (UHF), $2.0\times10^{-11}$ (ROHF) Ha/Bohr; bar $2\times10^{-7}$ Ha/Bohr, the density-convergence bound | `crates/ferric-scf/tests/fock_hook_gradient.rs` |
| the same test with the explicit term dropped | | misses by 3.7e-2 to 4.7e-2 Ha/Bohr, so the test sees the bug it guards | same |
| the same test with the ROHF spin Fock matrices recorded before the hook, so $W$ lacks $V$ | | misses by 3.0e-2 Ha/Bohr | same |
| hooked gradient against finite differences of the hooked energy, including a KS case | finite differences of the hooked energy | `test_hooked_gradient_matches_finite_differences` and `test_hooked_ks_gradient_matches_finite_differences` assert it | `crates/ferric-python/tests/test_fock_hook_gradient.py` |

The first three rows are recorded in the header of
`crates/ferric-scf/tests/fock_hook_gradient.rs`; the hook is not graded on the
validation page. The status of this section is `edit`: the derivation in `theory.md` is tested and
now has `scripts/fock_hook_gradient_sympy.py`, but the script is not yet named in
`theory.md` or the registry.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/theory.md`, "Fock hooks and their nuclear gradient" | the derivation and the rules for the explicit term |
| `scripts/fock_hook_gradient_sympy.py`, claims K1 to K3 | verified equation 14.19 |
| `crates/ferric-scf/tests/fock_hook_gradient.rs` | the point-charge anchor and its mutations |
| `crates/ferric-python/tests/test_fock_hook_gradient.py` | the exactness anchor and the error cases |
