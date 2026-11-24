# Basis functions and their derivatives on a grid {#ao-grid}

<!-- status: edit; registry: none -->

Some quantities have no closed-form integral over Gaussians. An
exchange-correlation functional is a nonlinear function of the density, so
its matrix elements are computed by quadrature: evaluate the basis functions
at many points, build the density there, and sum with weights. Seminumerical
exchange, Becke partitions and cube files need basis functions at points for
the same reason. A GGA needs the gradient of each basis function as well, and
meta-GGA gradients and analytic Hessians need second derivatives.

One module evaluates values, gradients and Hessians, and the DFT crate
re-exports it. This section needs only the basis functions and their
normalization from section @sec:one-electron; nothing from sections
@sec:two-electron to @sec:pvp-x2c is used. The plan: factor a basis function
into an angular and a radial part, differentiate the product by the chain
rule, read the radial loop, and count the memory.

| Entry point | Location |
|---|---|
| Rust | `ferric-integrals/src/ao_grid.rs::{eval_basis_on_points,eval_basis_and_grad_on_points,eval_basis_grad_hess_on_points,check_ao_grid_budget}` |
| Third derivatives | `ferric-integrals/src/ao_grid_third.rs::eval_basis_third_on_points` |
| Re-export | `ferric_dft::ao_grid` |

## The factorization {#ao-grid-factorization}

With coordinates relative to the shell centre, a contracted function of
(@eq:basis-sets-1) factors into an angular polynomial and a radial function
of $r^2$,

$$ \chi(\mathbf r) = A(x, y, z)\, \mathcal R(r^2), \qquad \mathcal R(r^2) = \sum_k c_k N(\alpha_k, l)\, e^{-\alpha_k r^2}, $$

with $A$ a Cartesian monomial or a solid harmonic and $N(\alpha, l)$ the
factor of section @sec:one-electron-basis-functions-integrals. All components
of a shell share $\mathcal R$, because $N(\alpha, l)$ is one factor per
shell, so the radial sum is done once per shell and point, and only the cheap
polynomial $A$ changes from component to component.

The derivatives of $\mathcal R$ with respect to its argument $s = r^2$ are
sums over the same primitives:

$$ \mathcal R' = \frac{d\mathcal R}{d(r^2)} = -\sum_k c_k N \alpha_k e^{-\alpha_k r^2}, \qquad \mathcal R'' = \sum_k c_k N \alpha_k^2 e^{-\alpha_k r^2} . $$

## The chain rule {#ao-grid-chain-rule}

Differentiate $\chi = A\,\mathcal R(r^2)$ with respect to $x_a$. The product
rule gives one term from each factor, and $\partial_a(r^2) = 2x_a$:

$$ \partial_a \chi = (\partial_a A)\,\mathcal R + 2 x_a\, A\, \mathcal R' . $$

Differentiate again with respect to $x_b$. The first term gives
$(\partial_a\partial_b A)\mathcal R + 2x_b(\partial_a A)\mathcal R'$. The
second gives three terms: $2\delta_{ab}A\mathcal R'$ from $x_a$,
$2x_a(\partial_b A)\mathcal R'$ from $A$, and $4x_ax_bA\mathcal R''$ from
$\mathcal R'$. Collect the $\mathcal R'$ terms:

$$ \partial_a\partial_b \chi = (\partial_a\partial_b A)\,\mathcal R + 2\big[x_a\, \partial_b A + x_b\, \partial_a A + \delta_{ab} A\big]\mathcal R' + 4 x_a x_b\, A\, \mathcal R'' . $$

Check it on an $s$ function ($A = 1$) at its centre: every term with an $x$
vanishes, and $\partial_a\partial_b\chi = 2\delta_{ab}\mathcal R'(0) = -2\delta_{ab}\sum_k c_kN\alpha_k$.
That is the curvature at the top of a sum of Gaussians: negative, and the same
in every direction.

## The radial loop {#ao-grid-radial-loop}

`radial_and_d_d2` accumulates all three radial factors in one pass over the
primitives, so each exponential is evaluated once. This is its loop, with
comments added (`radial_and_d` is the same loop without `d2rad`):

```rust
for (a, c) in shell.exponents.iter().zip(shell.coefficients.iter()) {
    let n = (2.0 * a / pi).powf(0.75) * (4.0 * a).powi(l).sqrt() / dbl_fact.sqrt();  // N(alpha, l)
    let e = (-a * r2).exp();                 // one exp per primitive and point
    rad += c * n * e;                        // R
    drad += -c * n * a * e;                  // R'  = dR/d(r^2)
    d2rad += c * n * a * a * e;              // R'' = d^2R/d(r^2)^2
}
```

Two points about it:

- **The stored coefficients do not contain $N$.** The loop multiplies
  $N(\alpha, l)$ in, with $(2l-1)!!$ taken from a table for $l \le 4$. That is
  the same factor libint2 uses, so values on the grid and analytic integrals
  agree.
- **Derivatives are with respect to $r^2$, not $r$.** That keeps the formulas
  free of $1/r$, so nothing is singular at the shell centre.

Components come in libint2 order: $p$ as $x, y, z$, pure shells $m = -l..l$,
Cartesian shells in CCA order. The evaluators support $l \le 4$ and return
`UnsupportedL` above (`eval_basis_on_points`, `eval_basis_and_grad_on_points`,
`eval_basis_grad_hess_on_points`).

## Memory {#ao-grid-memory}

| Evaluation | Planes of $N \times N_g$ doubles |
|---|---|
| values | 1 |
| values and gradients | 4 |
| values, gradients and Hessians | 13 |
| third derivatives | 10 |

The counts are $1 + 3$ and $1 + 3 + 9$: the Hessian is returned as the full
$3 \times 3$ tensor, so a value-gradient-Hessian evaluation holds 13 planes at
peak rather than the $1 + 3 + 6$ that its symmetry would need. The
third-derivative evaluator returns only the 10 unique components.

`check_ao_grid_budget` multiplies these by $8 N N_g$ bytes and errors before
allocating when the product exceeds the resolved memory budget. A caller that
evaluates in batches of points keeps $N_g$ small; the planes count is per
batch. For scale: 500 basis functions on $10^5$ points with Hessians is
$13 \times 8 \times 500 \times 10^5$ bytes, 5.2 GB, which is why grid code
works in batches.

## Numerical anchors {#ao-grid-numerical-anchors}

| What | Test | Bar |
|---|---|---|
| $g$ functions: grid-integrated overlap against the analytic overlap (normalization and libint2 order at once), and their gradients and Hessians | `crates/ferric-dft/tests/gfunc.rs` | per test |
| meta-GGA gradients built on the AO Hessians, against PySCF with grid response | the meta-GGA gradient tests listed in `site/src/reference/validation.md` | per row |

No proof script covers this section. The chain-rule Hessian above was checked
symbolically for an arbitrary polynomial $A$ and radial function $\mathcal R$.

## Known limits

Checked 2026-11-19.

- Shells up to $g$ ($l \le 4$): the evaluators return `UnsupportedL` above.
- The Hessian evaluation holds all nine components, not the six unique ones
  (section @sec:ao-grid-memory).

## Sources

- T. Helgaker, P. Jørgensen and J. Olsen, *Molecular Electronic-Structure
  Theory* (Wiley, 2000), section 6.6.
