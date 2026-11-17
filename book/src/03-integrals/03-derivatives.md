# Derivative integrals and buffer sizing {#integrals-derivatives}

<!-- status: edit; registry: none;
     theory: crates/ferric-integrals/theory.md#"Derivative integrals and buffer sizing" -->

A force on a nucleus is the derivative of the energy with respect to that
nucleus's position. The basis functions sit on the nuclei and move with them,
so every integral changes when an atom moves. Analytic gradients and Hessians
(the SCF and MP2 gradient chapters) contract these derivative integrals with
densities.

The derivatives themselves are libint2's. What a caller gets wrong is the
number of blocks an engine writes, and this section is mostly about that
count. The plan: derive the derivative of a Gaussian, use translational
invariance to see how many independent derivatives an integral has, then
count the blocks libint2 returns.

| Entry point | Location |
|---|---|
| Rust | `ferric-integrals/src/engine.rs::Engine` (`new_1e_deriv`, `compute_1e_deriv_block`, `compute_1e_deriv_block_n`, `new_2e_deriv`, `compute_eri_deriv_quartet`, `new_3center_deriv`, `compute_eri3_deriv`, `new_2center_deriv`, `compute_eri2_deriv`) |
| Second derivatives | `new_1e_deriv2`, `new_2e_deriv2`; pair order `ferric-integrals/src/engine.rs::{deriv2_pair_index,deriv2_pairs}` |

### The derivative of a basis function

A basis function on centre $\mathbf A$ depends on $\mathbf A$ only through
$\mathbf r - \mathbf A$. Moving the centre by $+\delta$ is the same as moving
the point by $-\delta$, so

$$ \frac{\partial \chi_\mu}{\partial A_x} = -\frac{\partial \chi_\mu}{\partial x} . $$

Apply the product rule to one Cartesian factor. With $x_A = x - A_x$,
$\partial x_A/\partial A_x = -1$, so the polynomial gives $-l\,x_A^{l-1}$ and
the Gaussian gives $+2\alpha x_A$:

$$ \frac{\partial}{\partial A_x}\big(x_A^{l} e^{-\alpha r_A^2}\big) = 2\alpha\, x_A^{l+1} e^{-\alpha r_A^2} - l\, x_A^{l-1} e^{-\alpha r_A^2} . $$

The derivative of a function of angular momentum $l$ is a combination of
functions of $l + 1$ and $l - 1$ with the same exponent. So a derivative
integral costs one higher angular momentum, and an engine that can do the
integral at $l+1$ can do its derivative. For an $s$ function ($l = 0$) only
the raised term survives: the derivative of a $1s$ function is a $p$
function. Section @sec:ecp uses this formula directly for the ECP derivatives.

### Counting the blocks

Translate every centre of an integral by the same vector $\boldsymbol\delta$.
The integral is over all space, so it does not change:
$I(\mathbf A + \boldsymbol\delta, \mathbf B + \boldsymbol\delta, \dots) = I(\mathbf A, \mathbf B, \dots)$.
Differentiate with respect to $\delta_x$ at $\boldsymbol\delta = 0$:

$$ \sum_{\text{centres } K} \frac{\partial I}{\partial K_x} = 0 . $$

This is translational invariance: the derivatives with respect to all centres
of an integral sum to zero. An integral with $k$ centres has $3k$ derivatives
and $3(k-1)$ independent ones. For the nuclear attraction the point charges
are centres too, because the integral depends on where they are.

Libint2 nevertheless returns every centre's derivative, and the number of
blocks a derivative engine writes is:

| Integral | Blocks | Function |
|---|---|---|
| overlap, kinetic | $6$: $\partial_{A}$, $\partial_{B}$ | `Engine::compute_1e_deriv_block` |
| nuclear attraction with $n_c$ point charges | $3(2 + n_c)$: bra, ket, then each charge | `Engine::compute_1e_deriv_block_n` |
| four-center | $12$ | `Engine::compute_eri_deriv_quartet` |
| three-center | $9$ | `Engine::compute_eri3_deriv` |
| two-center | $6$ | `Engine::compute_eri2_deriv` |

Each block holds $n_1 n_2$ doubles for a one-electron pair, and
$n_1 n_2 n_3 n_4$ for a quartet; the counts are three Cartesian directions
per centre.

The nuclear count grows with the number of charges, so a caller that adds
external charges (`Engine::set_point_charges_extra`) must size the buffer for
the actual charge count with `compute_1e_deriv_block_n`. Here is how
`ferric-integrals/src/engine.rs::Engine::compute_1e_deriv_block` sizes its
buffer, with the FFI call and the screened-block return removed:

```rust
let n = prep.shell_dims()[sh1] * prep.shell_dims()[sh2];   // n1 n2 per block
// Worst case is the nuclear operator: 3*(2 + natoms) blocks. Sizing for
// it unconditionally keeps the shim's write (nderiv * n doubles) in
// bounds for every 1e engine kind.
let total = 3 * (2 + prep.atoms().len()) * n;              // bra, ket, each nucleus
if self.buf.len() < total {
    self.buf.resize(total, 0.0);
}
// ...
```

The buffer is sized for the nuclear operator even on an overlap engine,
which writes only 6 blocks, so one sizing rule covers every one-electron
engine kind. `compute_1e_deriv_block_n` is the same function with
`prep.atoms().len()` replaced by an explicit `n_charges`. The atom count is
the wrong number as soon as external charges are set: with $n_{\mathrm{extra}}$
extra charges, libint2 writes $3n_{\mathrm{extra}}$ more blocks than a buffer
sized for the atoms holds.

### Second derivatives

Second derivatives are returned as the upper triangle of coordinate pairs
(`deriv2_pair_index`, `deriv2_pairs`). With $n$ coordinates, the pairs
$(i, j)$ with $i \le j$ are stored row by row. Row $k$ holds $n - k$ pairs, so
the rows before $i$ hold $\sum_{k<i}(n - k) = in - i(i-1)/2$ entries, and
$(i, j)$ is $j - i$ further on:

$$ \text{index}(i, j) = in - \frac{i(i-1)}{2} + (j - i) = \frac{i(2n - i - 1)}{2} + j, \qquad i \le j . $$

Check it on $n = 6$ (one-electron overlap, two centres): $(0,0) \to 0$, $(0,5) \to 5$,
$(1,1) \to 6$, $(2,4) \to 13$, $(5,5) \to 20$, the last of the 21 pairs.
`deriv2_pair_index` sorts its arguments, so $(j, i)$ gives the same index.
Section @sec:pvp-x2c uses it to pick the bra–ket blocks $(a, 3 + a)$ of the nuclear
second derivative.

## Translational invariance in practice {#integrals-derivatives-translational-invariance-practice}

The derivatives with respect to all centres of one integral sum to zero. A
gradient code can use that to skip one centre's block and recover it as minus
the sum of the others, and the ECP derivative does this
(section @sec:ecp). The engines still return every centre, and a code that skips
one must not assume a smaller buffer.

## Numerical anchors {#integrals-derivatives-numerical-anchors}

| What | Test | Bar |
|---|---|---|
| shifted-centre 3- and 2-center derivatives against finite differences | `crates/ferric-scf/tests/df_hessian_stencil.rs` | per test |
| point-charge derivative buffer sized for the charge count | `crates/ferric-scf/tests/qmmm_polarizable.rs` | — |
| four-center derivatives at shifted centres | `crates/ferric-integrals/tests/eri_deriv_shifted.rs` | per test |

No committed proof script covers this section. The Gaussian derivative
formula is the one in `crates/ferric-integrals/theory.md`; the pair order is
libint2's, and the doc comment of `deriv2_pair_index` cites the libint2
headers it was read from. The engines' derivatives are tested against finite
differences in the files above.

## Sources

- T. Helgaker and P. R. Taylor, in *Modern Electronic Structure Theory*,
  edited by D. R. Yarkony (World Scientific, 1995), part II, chapter @sec:oo-laplace.
