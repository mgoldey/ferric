# 14.3 XC gradient: AO-derivative and grid-response terms

<!-- section: 14.3; status: ready; registry: ks-gradient;
     theory: crates/ferric-dft/theory.md#"XC nuclear gradient: AO-derivative and grid-response terms", "Becke partition and its nuclear derivative" -->

The Kohn–Sham gradient is the Hartree–Fock gradient of section 14.1 with the
exact-exchange term scaled by the hybrid fraction and one new piece, the
derivative of the XC energy. That derivative has two parts because the XC
energy is a sum over a grid that itself depends on the nuclei. The
AO-derivative part moves the basis functions under fixed grid points. The
grid-response part moves the points and their weights. Dropping the second
leaves a gradient that is not the derivative of the energy the SCF computes,
and an optimization on it stalls near, not at, the minimum.

| Symbol | Meaning | Code name |
|---|---|---|
| $\chi_\mu$, $\partial_\alpha\chi_\mu$, $\partial^2_{\alpha b}\chi_\mu$ | AO and its first and second derivatives at a grid point | `chi`, `dchi`, `ddchi` |
| $m_\mu = (D\chi)_\mu$, $(D\partial_b\chi)_\mu$ | density matrix applied to the AOs | `m`, `mdchi` |
| $v_\rho$, $v_\sigma$, $v_\tau$ | libxc derivatives of $\rho\epsilon$ | `vrho`, `vsigma`, `vtau` |
| $c_b = 2wv_\sigma\partial_b\rho$ | GGA weight column | `c`, `gga_weight_columns` |
| $w^A$, $\partial w^A/\partial\mathbf R_C$ | Becke weight of atom $A$ and its derivative | `weight1` |
| $A$ | home atom of a grid point | `gp.home_atom` |

| Entry point | Location |
|---|---|
| Rust, XC term | `ferric-dft/src/gradient.rs::{xc_gradient_closed_lda_from_density,xc_gradient_closed_gga_from_density,xc_gradient_closed_mgga_from_density,xc_gradient_uks_from_density,xc_gradient_uks_mgga_from_density}` |
| Rust, full gradient | `ferric-scf/src/ks_gradient.rs::{ks_gradient_closed,ks_gradient_uks,ks_gradient_roks}` |
| CLI | `task = "optimize"` with `method.kind = "ksdft"`, or `rhf`/`uhf`/`rohf` with `[dft] functional` |
| Python | `run_dft(..., with_gradient=True)`, `run_optimize(..., xc=...)` |

## 14.3.1 The two terms

<!-- include: crates/ferric-dft/theory.md#"XC nuclear gradient: AO-derivative and grid-response terms" -->

> **Verified equation 14.6** · the AO-derivative terms, with the $\tfrac12$ of $\tau$
>
> On two real Gaussians (a $p_x$ on one nucleus, an $s$ on another) at an
> exact rational point, with a symmetric density matrix: $\partial\tau/\partial R_{A,\alpha}$
> at fixed $(\mathbf r, D)$ equals $-\sum_b\sum_{\mu\in A}\partial^2_{\alpha b}\chi_\mu(D\partial_b\chi)_\mu$,
> so the $\tfrac12$ in $\tau$ cancels against the $\mu\leftrightarrow\nu$ doubling.
> The assembled AO-derivative term, $-2\times$ the sum of the code's
> `gga_ao_partials` and `mgga_tau_partials`, equals $\partial E/\partial R_{A,\alpha}$
> computed from the exact derivatives of $\rho$, $\sigma$ and $\tau$.
>
> | | |
> |---|---|
> | Proof | `scripts/dft_mgga_tau_gradient_sympy.py`, claims 1 and 4 |
> | Negative control | keeping the $\tfrac12$ of $\tau$ must fail (the script's own control); the runner also drops the factor 2 in the GGA weight column (claim 4 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-dft/src/gradient.rs::{gga_ao_partials,mgga_tau_partials,scatter_partials}` |
> | Code test | `scan_gradient_h2o_sto3g_vs_fd` in `crates/ferric-scf/tests/dft_gradient_mgga.rs` |
>
> <!-- verified: id=ks.grad.ao-derivative; script=scripts/dft_mgga_tau_gradient_sympy.py; claim=4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-dft/src/gradient.rs::{gga_ao_partials,mgga_tau_partials,scatter_partials}; rust_test=ferric-scf/tests/dft_gradient_mgga.rs::scan_gradient_h2o_sto3g_vs_fd; registry=ks-gradient -->

> **Verified equation 14.7** · the grid-response term
>
> The home-atom translation column of the integrand, $\partial_\alpha(\partial_b\rho)$
> (`hess_col`) and $\partial_\alpha\tau$ (`tau_spatial_grad`), equals the exact
> electron-coordinate derivative, and $\rho$, $\sigma$ and $\tau$ are
> translation invariant: $\partial q/\partial R_A + \partial q/\partial R_B + \nabla_{\mathbf r}q = 0$.
> That invariance is what makes the home-translation term the correct
> correction for a point that rides its home atom.
>
> | | |
> |---|---|
> | Proof | `scripts/dft_mgga_tau_gradient_sympy.py`, claims 2, 3 and 5 |
> | Negative control | a halved spatial gradient must fail claim 2, and the runner drops the factor 2 of `hess_col` (claim 5 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-dft/src/gradient.rs::{tau_spatial_grad,mgga_closed_batch}` |
> | Code test | `scan_gradient_h2o_sto3g_vs_fd` in `crates/ferric-scf/tests/dft_gradient_mgga.rs` |
>
> <!-- verified: id=ks.grad.grid-response; script=scripts/dft_mgga_tau_gradient_sympy.py; claim=5; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-dft/src/gradient.rs::{tau_spatial_grad,mgga_closed_batch}; rust_test=ferric-scf/tests/dft_gradient_mgga.rs::scan_gradient_h2o_sto3g_vs_fd; registry=ks-gradient -->

## 14.3.2 The Becke weight derivative

<!-- include: crates/ferric-dft/theory.md#"Becke partition and its nuclear derivative" -->

> **Verified equation 14.8** · the weight derivative and its translation invariance
>
> The assembled $\partial w^A/\partial\mathbf R_C$ of a three-atom,
> unequal-radius cluster equals the exact sympy derivative; the weights form
> a partition of unity, so $\sum_A\partial w^A/\partial\mathbf R_C = 0$; and
> $\nabla_{\mathbf r}w^A = -\sum_C\partial w^A/\partial\mathbf R_C$.
>
> | | |
> |---|---|
> | Proof | `scripts/dft_becke_weight_grad_sympy.py`, claims assembly, pou and translation |
> | Negative control | dropping the $\mathrm dT/T^2$ term must be detected (check 5 of the script) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-dft/src/becke.rs::becke_weights_and_grad` |
> | Code test | `becke_weights_grad_finite_difference_ch` in `crates/ferric-dft/src/becke.rs` |
>
> <!-- verified: id=ks.grad.becke-weights; script=scripts/dft_becke_weight_grad_sympy.py; claim=assembly; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-dft/src/becke.rs::becke_weights_and_grad; rust_test=ferric-dft/src/becke.rs::becke_weights_grad_finite_difference_ch; registry=ks-gradient -->

## 14.3.3 The algorithm

The production path is the batched, shell-screened gradient of
`ferric-dft/src/gradient/screened.rs`, behind every `xc_gradient_*_from_density`
entry point. The whole-grid bodies in `gradient.rs` (`gga_closed_batch`,
`mgga_closed_batch`) are kept as the reference it is anchored to. Both
evaluate the same expressions.

| Step | Quantity | Function |
|---|---|---|
| 1 | spatial batches of at most `DEFAULT_MAX_BATCH_PTS` points, the same bisection the SCF's $V_{xc}$ uses | `ferric-dft/src/xc_batch.rs::partition_points` |
| 2 | per batch, keep a shell if the largest of $\lvert\chi\rvert$, $\lvert\nabla\chi\rvert$, $\lvert\nabla\nabla\chi\rvert$ over the batch exceeds `DEFAULT_GRAD_SCREEN_THRESH` | `ferric-dft/src/gradient/screened.rs::screened_xc_gradient` |
| 3 | $\rho$, $\nabla\rho$, $\sigma$, $\tau$ on the batch, then libxc once for $\epsilon$, $v_\rho$, $v_\sigma$, $v_\tau$ | `ferric-dft/src/density_on_grid.rs::eval_density_closed` |
| 4 | AO-derivative partials per basis function, scattered to atoms | `gga_ao_partials`, `mgga_tau_partials`, `scatter_partials` |
| 5 | weight response $\sum_g(\partial w_g/\partial\mathbf R_b)\,\rho_g\epsilon_g$ | `weight1` |
| 6 | home-translation of the integrand, added to the point's home atom | `hess_col`, `tau_spatial_grad` |
| 7 | add the exact-exchange, Coulomb, one-electron, ECP and Pulay terms of section 14.1 | `ferric-scf/src/ks_gradient.rs::ks_gradient_closed_for_density` |

The screened path forms one integrand per $(\mu, g)$ and reduces it two ways,
over points for the AO partials and over functions for the grid-response
columns, in place of the whole-grid path's separate passes. Batches are
grouped into at most 64 contiguous groups by a function of the grid alone,
each group sums its batches in ascending order, and the group results are
added in ascending order.

The scatter of step 4 is where the factor $-2$ of the formula lives.

<!-- excerpt: ferric-dft/src/gradient.rs::scatter_partials -->
```rust
for (axis, part) in partials.iter().enumerate() {
    for mu in 0..nbf {
        let atom = bf_to_atom_map[mu];
        grad[(atom, axis)] -= 2.0 * part[mu];
    }
}
```

| Code | Equation |
|---|---|
| `part[mu]` | the partial sum over points for basis function $\mu$, for one axis |
| `bf_to_atom_map[mu]` | the atom $A$ carrying $\mu$ |
| `-= 2.0 *` | the $-2$ in front of the AO-derivative sum, from $\mu\leftrightarrow\nu$ |

`mgga_tau_partials` returns half of the $\tau$ sum
(`p.mapv_inplace(|v| 0.5 * v)`), and the same scatter applies $-2\times$, which
reproduces equation 14.6. A reader who re-adds the $\tfrac12$ halves the
$\tau$ term; the proof script's control is exactly that mistake.

## 14.3.4 Reproducibility and memory

- The result is bit-identical across rayon thread counts: the reduction groups
  and their order do not depend on the thread count.
- The screened path is not bit-identical to the whole-grid path: every grid sum
  is re-associated, and screening drops terms below the threshold. Both
  effects are measured in `crates/ferric-scf/tests/ks_gradient_screened.rs`.
  A threshold of zero or below keeps every shell, the exactness anchor.
- The batch size is resolved against the memory budget once, and the charge is
  committed before the batches run; the AO planes of one batch are $(n_{\mathrm{act}}, n_b)$,
  not whole-grid.

## 14.3.5 Options that change the equation

| Option | Effect |
|---|---|
| LDA, GGA, meta-GGA | $v_\sigma = v_\tau = 0$ for LDA, $v_\tau = 0$ for GGA; one code path with fewer terms |
| UKS, ROKS | the same per spin with $D\to D^\sigma$ and $\mathbf G^\sigma$ in place of $2v_\sigma\nabla\rho$ |
| hybrid, range-separated hybrid | exact exchange scaled by $c_K$, or split into erfc and erf pieces, through the routes of section 14.2 |
| VV10 | `ferric-dft/src/gradient.rs::vv10_gradient_from_density` has no Becke-weight response by design, and is not finite-difference validated |
| grid | the gradient differentiates the grid the SCF ran on: `ferric-scf/src/ks_gradient.rs::KsGradGrids` carries it. A default-grid gradient paired with an SCF on another grid misses |
| meta-GGAs that need the density Laplacian (for example `MGGA_X_SCANL`, `MGGA_X_BR89`) | refused in `ferric-dft/src/libxc.rs::xc_def_from_name` (test `laplacian_metagga_rejected_by_flag`); the $\tau$-only functionals run |

The Hessian of the same energy, with the same two kinds of term, is
section 14.5.

## 14.3.6 Numerical anchors

| Check | Reference | Agreement | Test |
|---|---|---|---|
| Meta-GGA gradient, closed shell (SCAN, r2SCAN): H2O, NH3 / 6-31G, def2-SVP; RI-J and exact J | PySCF `RKS` `grid_response=True` on the same grid; FD of ferric's own energy | at most 1.3e-9 Ha/Bohr vs PySCF; analytic vs own FD 1.0e-9; bar 1e-8 | `crates/ferric-scf/tests/validation_mgga_gradients.rs` |
| Meta-GGA gradient, open shell: UKS HO2, NH2 / 6-31G; ROKS NH2 / 6-31G | PySCF `UKS` `grid_response=True`; central FD of PySCF `ROKS` | UKS at most 6.2e-9 Ha/Bohr, bar 3e-8; ROKS 2.2e-10 vs FD, bar 1e-7 | same |
| KS gradient on a custom or angularly pruned grid: water, NH2 / 6-31G | central FD of the energy on the same grid | 0.9e-7 to 1.4e-7 Ha/Bohr, bar 3e-7; the default-grid gradient paired with the same SCF misses by 2.8e-4 to 6.0e-4 | `crates/ferric-scf/tests/ks_gradient_scf_grid.rs` |

<!-- figures: site/src/reference/validation.md "Meta-GGA gradient, closed shell (SCAN, r2SCAN)"; site/src/reference/validation.md "Meta-GGA gradient, open shell (SCAN, r2SCAN)"; site/src/reference/validation.md "KS gradient on a custom or angularly pruned XC grid" -->

The third row is the check on the grid-response term: the same gradient
without it is the "paired with the default grid" miss.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-dft/theory.md`, "XC nuclear gradient: AO-derivative and grid-response terms", "Becke partition and its nuclear derivative" | equations, tests |
| `scripts/dft_mgga_tau_gradient_sympy.py`, `scripts/dft_becke_weight_grad_sympy.py` | verified equations 14.6 to 14.8 |
| `crates/ferric-dft/src/gradient.rs`, `crates/ferric-dft/src/gradient/screened.rs` | the algorithm and reproducibility rules |
| `site/src/reference/validation.md` | the anchor rows |
