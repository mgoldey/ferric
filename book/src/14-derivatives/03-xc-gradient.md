# XC gradient: AO-derivative and grid-response terms {#xc-gradient}

<!-- status: ready; registry: ks-gradient;
     theory: crates/ferric-dft/theory.md#"XC nuclear gradient: AO-derivative and grid-response terms", "Becke partition and its nuclear derivative" -->

The Kohn–Sham gradient is the Hartree–Fock gradient of section @sec:hf-gradient with the
exact-exchange term scaled by the hybrid fraction and one new piece, the
derivative of the XC energy. That energy is a weighted sum over grid
points, and the grid is built around the nuclei: when an atom moves, its
basis functions move, its grid points move with it, and the Becke weights
of every nearby point change. The AO-derivative term covers the first, the
grid-response terms the other two. Dropping the grid response leaves a
gradient that is not the derivative of the energy the SCF computes, and an
optimization on it stalls near, not at, the minimum.

The plan: write the total derivative of the grid sum, work out the
AO-derivative term by the chain rule, add the two grid-response terms, and
check that the three together give forces that sum to zero.

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

## The two terms {#xc-gradient-terms}

Section @sec:vxc writes the XC energy on the grid as
$E_{xc} = \sum_g w_g\,e(\rho_g, \sigma_g, \tau_g)$ with $e = \rho\epsilon$,
$\rho = \sum D_{\mu\nu}\chi_\mu\chi_\nu$, $\sigma = \lvert\nabla\rho\rvert^2$ and
$\tau = \tfrac12\sum_b\sum D_{\mu\nu}\partial_b\chi_\mu\partial_b\chi_\nu$. The
SCF is stationary, so as in section @sec:hf-gradient we differentiate at fixed $D$. Three
things depend on a nuclear coordinate $R_{C,\alpha}$:

- the AOs on atom $C$, at a fixed point $\mathbf r$;
- the point $\mathbf r_g$ itself, which rides its home atom $A$:
  $\mathbf r_g = \mathbf R_A + \text{offset}$;
- the weight $w_g$, through the Becke partition.

Apply the chain rule to each grid term, with $q$ running over
$\rho, \sigma, \tau$:

$$ \frac{\mathrm dE_{xc}}{\mathrm dR_{C,\alpha}} = \sum_g \frac{\partial w_g}{\partial R_{C,\alpha}}\,e_g
+ \sum_g w_g\sum_q v_q\,\frac{\partial q}{\partial R_{C,\alpha}}\Big|_{\mathbf r}
+ \sum_{g:\,\mathrm{home}=C} w_g\sum_q v_q\,\partial_\alpha q . $$

The middle sum is the AO-derivative term. The first and last are the grid
response.

**The AO-derivative term.** A basis function on atom $A$ depends on
$\mathbf R_A$ only through $\mathbf r - \mathbf R_A$, so
$\partial\chi_\mu/\partial R_{A,\alpha} = -\partial_\alpha\chi_\mu$ for $\mu\in A$
(section @sec:integrals-derivatives). Differentiate $\rho$. Each of the two $\chi$ factors gives a
term, and they are equal because $D$ is symmetric:

$$ \frac{\partial\rho}{\partial R_{A,\alpha}} = -2\sum_{\mu\in A}\partial_\alpha\chi_\mu\,m_\mu . $$

Differentiate $\partial_b\rho = 2\sum D_{\mu\nu}\,\partial_b\chi_\mu\,\chi_\nu$ the same
way. The nuclear derivative hits either factor:

$$ \frac{\partial(\partial_b\rho)}{\partial R_{A,\alpha}} = -2\sum_{\mu\in A}\Big[\partial^2_{\alpha b}\chi_\mu\,m_\mu + \partial_\alpha\chi_\mu\,(D\partial_b\chi)_\mu\Big], $$

and $\partial\sigma = 2\sum_b\partial_b\rho\,\partial(\partial_b\rho)$. For $\tau$
the nuclear derivative hits one of the two $\partial_b\chi$ factors. The two
cases map onto each other under $\mu\leftrightarrow\nu$, so they are equal,
and their sum cancels the $\tfrac12$ in the definition of $\tau$:

$$ \frac{\partial\tau}{\partial R_{A,\alpha}} = -\sum_b\sum_{\mu\in A}\partial^2_{\alpha b}\chi_\mu\,(D\partial_b\chi)_\mu . $$

Multiply by $w\,v_q$, sum over $q$, and pull out a common $-2$. The $\sigma$
term collects into the GGA weight column $c_b = 2wv_\sigma\partial_b\rho$, and
the $\tau$ term keeps a $\tfrac12$ because its derivative had no factor 2:

$$\frac{\partial E_{xc}}{\partial R_{A,\alpha}}\Big|_{\mathrm{AO}} = -2\sum_g\sum_{\mu\in A}\Big[w v_\rho\,\partial_\alpha\chi_\mu m_\mu
+ \sum_b c_b\big(\partial_\alpha\chi_\mu (D\partial_b\chi)_\mu + \partial^2_{\alpha b}\chi_\mu m_\mu\big)
+ \tfrac12 w v_\tau\sum_b\partial^2_{\alpha b}\chi_\mu (D\partial_b\chi)_\mu\Big] . $$

The GGA path (`xc_gradient_closed_gga_from_density`) is the $v_\tau = 0$ case
of this formula and the LDA path (`xc_gradient_closed_lda_from_density`) the
$v_\sigma = v_\tau = 0$ case; the meta-GGA path is
`xc_gradient_closed_mgga_from_density` with batch body `mgga_closed_batch`.

> **Verified equation {#eq:ks-grad-ao-derivative}** · the AO-derivative terms, with the $\tfrac12$ of $\tau$
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

**The grid-response terms.** The first is the weight term,
$\sum_g(\partial w_g/\partial\mathbf R_b)\,\rho_g\epsilon_g$, with the "weight1"
derivative of section @sec:xc-gradient-becke-weight-derivative. The second is the home-atom translation of the
integrand,

$$ \sum_{g:\,\mathrm{home}=A} w_g\big[v_\rho\,\partial_\alpha\rho + v_\sigma\,\partial_\alpha\sigma + v_\tau\,\partial_\alpha\tau\big], $$

which needs the electron-coordinate derivatives of $\rho$, $\sigma$ and
$\tau$ at the point. Differentiate $\partial_b\rho$ and $\tau$ with respect to
$r_\alpha$; now every basis function moves relative to the point, so the sums
run over all $\mu$, and the $\mu\leftrightarrow\nu$ doubling works as before:

$$ \partial_\alpha(\partial_b\rho) = 2\Big[\sum_\mu\partial_\alpha\chi_\mu(D\partial_b\chi)_\mu + \sum_\mu m_\mu\,\partial^2_{\alpha b}\chi_\mu\Big]\ (\texttt{hess\_col}),\qquad
\partial_\alpha\tau = \sum_b\sum_\mu\partial^2_{\alpha b}\chi_\mu (D\partial_b\chi)_\mu\ (\texttt{tau\_spatial\_grad}) . $$

Why is the home translation the right correction? The density at a point is
unchanged when every nucleus and the point move together, so for
$q = \rho, \sigma, \tau$

$$ \sum_C\frac{\partial q}{\partial\mathbf R_C}\Big|_{\mathbf r} + \nabla_{\mathbf r}q = 0 . $$

A point that rides atom $A$ sees $\mathbf r$ change with $\mathbf R_A$, and the
term $\nabla_{\mathbf r}q$ is that change.

**Check translational invariance.** Sum the XC gradient over all atoms $C$.
The weight term vanishes, because the weight1 rows sum to zero for every
point (section @sec:grids-becke-partition). At each point the AO-derivative term contributes
$w_g\sum_q v_q\sum_C\partial q/\partial\mathbf R_C$ and the home term
$w_g\sum_q v_q\nabla_{\mathbf r}q$; by the identity above they cancel. The XC
forces sum to zero exactly on the grid. Drop the home term and the sum
becomes $-\sum_g w_g\nabla_{\mathbf r}e(\mathbf r_g)$: the grid's estimate of
$-\int\nabla e\,\mathrm d\mathbf r$, which is zero in exact integration but not
on a finite grid. That residual is a net force on the molecule from the
quadrature error, and it is what the grid response removes.

> **Verified equation {#eq:ks-grad-grid-response}** · the grid-response term
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

The proof script uses $p$- and $s$-Gaussians on two centres at an exact
rational point; its controls (the $\tfrac12$ of $\tau$ kept, a halved spatial
gradient) fail. The open-shell path (`xc_gradient_uks_mgga_from_density`;
LDA and GGA `xc_gradient_uks_from_density`) is the same per spin with
$D\to D^\sigma$ and $\mathbf G^\sigma$ in place of $2v_\sigma\nabla\rho$.

The Rust tests of the whole term are `ferric-scf/tests/dft_gradient_mgga.rs`
(SCAN, r2SCAN and TPSS against finite differences of the SCF energy; SCAN
and r2SCAN against PySCF `grid_response=True`; UKS NH$_2$ and OH),
`ferric-scf/tests/dft_gradient_gga.rs` and `dft_gradient_lda.rs` (against
finite differences), and
`ferric-python/tests/test_xc_blocks.py::test_xc_grad_matches_finite_difference_at_fixed_density`
(PBE and SCAN, finite differences at fixed $D$, closed and polarized).

## The Becke weight derivative {#xc-gradient-becke-weight-derivative}

The weight of grid point $g$ is $w_g = w_rw_l\,w^A(\mathbf r_g)$: radial and
angular quadrature weights times the Becke partition weight of its home
atom $A$. Section @sec:grids-becke-partition builds $w^A$ in four steps and differentiates it by
the chain rule through the same steps
(`ferric-dft/src/becke.rs::becke_weights_and_grad`). The pieces, with
$r_X = |\mathbf r-\mathbf R_X|$ and $R_{AB} = |\mathbf R_A-\mathbf R_B|$:

$$ \mu_{AB} = \frac{r_A-r_B}{R_{AB}},\qquad
\nu_{AB} = \mu_{AB} + a_{AB}(1-\mu_{AB}^2),\qquad
s_{AB} = \tfrac12\bigl(1 - f^{(3)}(\nu_{AB})\bigr),\quad f(x)=\tfrac12(3x-x^3), $$

$$ a_{AB} = \operatorname{clip}_{[-1/2,\,1/2]}\frac{u}{u^2-1},\quad u=\frac{\chi-1}{\chi+1},\ \chi = R^{\mathrm{BS}}_A/R^{\mathrm{BS}}_B,\qquad
P_A = \prod_{B\ne A}s_{AB},\qquad w^A = P_A/T,\ T=\sum_C P_C . $$

The loop differentiates $\mu_{AB}$ with respect to both nuclei,

$$ \frac{\partial\mu_{AB}}{\partial R_A^\alpha} = -\frac{(\mathbf{r}-\mathbf{R}_A)^\alpha}{r_A R_{AB}} - \frac{\mu_{AB}}{R_{AB}^2}(\mathbf{R}_A-\mathbf{R}_B)^\alpha,\qquad
\frac{\partial\mu_{AB}}{\partial R_B^\alpha} = +\frac{(\mathbf{r}-\mathbf{R}_B)^\alpha}{r_B R_{AB}} + \frac{\mu_{AB}}{R_{AB}^2}(\mathbf{R}_A-\mathbf{R}_B)^\alpha, $$

then passes through $\nu$ and the three iterations of $f$, and finishes with
the quotient rule on $w^A = P_A/T$:

$$ \frac{\partial\nu}{\partial\mu} = 1-2a_{AB}\mu,\qquad
\frac{df^{(3)}}{d\nu} = f_2' f_1' f_0',\quad f_k' = \tfrac32(1-f_k^2),\qquad
\frac{\partial w^A}{\partial R_C} = \frac{1}{T}\frac{\partial P_A}{\partial R_C} - \frac{P_A}{T^2}\sum_D\frac{\partial P_D}{\partial R_C} . $$

Every $\mu_{AB}$ depends only on differences of positions, so it is
invariant under a rigid translation of nuclei and point together. So is
$w^A$:

$$ \sum_C \frac{\partial w^A}{\partial \mathbf{R}_C}\Big|_{\mathbf{r}} + \nabla_{\mathbf{r}} w^A = 0 . $$

`ferric-dft/src/grid.rs::build_atomic_grid_with_response` uses this to form
PySCF's "weight1" derivative, in which each point rides its home atom:

$$ \frac{\partial w_g}{\partial \mathbf{R}_b} = w_rw_l\Big[\frac{\partial w^{A}}{\partial\mathbf{R}_b} + \delta_{b,A}\nabla_{\mathbf{r}}w^{A}\Big], \qquad \sum_b \frac{\partial w_g}{\partial\mathbf{R}_b} = 0 . $$

The $\delta_{b,A}$ term is the weight's version of the home translation of
@sec:xc-gradient-terms, and the sum rule is the reason the weight term drops out of the
translational-invariance check there.

> **Verified equation {#eq:ks-grad-becke-weights}** · the weight derivative and its translation invariance
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

The script checks each derivative symbolically, the assembled gradient of
the three-atom cluster against sympy to $10^{-61}$, the partition of unity
and the translation identity. The Rust tests are
`becke::tests::becke_weights_grad_finite_difference_ch` (finite differences,
$<10^{-7}$), `becke::tests::becke_weights_grad_sums_to_zero_h2` and
`becke::tests::partition_weight_derivative_matches_fd` (the neighbour-list
kernel); the KS gradients below test the chain end to end.

## The algorithm {#xc-gradient-algorithm}

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
| 7 | add the exact-exchange, Coulomb, one-electron, ECP and Pulay terms of section @sec:hf-gradient | `ferric-scf/src/ks_gradient.rs::ks_gradient_closed_for_density` |

Step 2 screens on second derivatives as well as values, because the
AO-derivative and home terms contain $\partial^2_{\alpha b}\chi_\mu$. The
screened path forms one integrand per $(\mu, g)$ and reduces it two ways,
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
| `bf_to_atom_map[mu]` | the atom $A$ carrying $\mu$, which restricts the sum to $\mu\in A$ |
| `-= 2.0 *` | the $-2$ in front of the AO-derivative sum, from $\mu\leftrightarrow\nu$ |

The scatter is a serial pass over $\mu$, so the summation order over
functions is fixed whatever the thread count. Because the $-2$ is applied
here for every term, `mgga_tau_partials` must return half of the $\tau$ sum,
and it does, with its last step:

<!-- excerpt: ferric-dft/src/gradient.rs::mgga_tau_partials -->
```rust
let mut p = partial.expect("3 axes");
p.mapv_inplace(|v| 0.5 * v);
p
```

Together they reproduce the $\tfrac12 w v_\tau$ term of the formula. A reader
who re-adds the $\tfrac12$ of $\tau$ halves the $\tau$ term; the proof
script's control is exactly that mistake.

## Reproducibility and memory {#xc-gradient-reproducibility-memory}

- The result is bit-identical across rayon thread counts: the reduction groups
  and their order do not depend on the thread count.
- The screened path is not bit-identical to the whole-grid path: every grid sum
  is re-associated, and screening drops terms below the threshold. Both
  effects are measured in `crates/ferric-scf/tests/ks_gradient_screened.rs`.
  A threshold of zero or below keeps every shell, the exactness anchor.
- The batch size is resolved against the memory budget once, and the charge is
  committed before the batches run; the AO planes of one batch are $(n_{\mathrm{act}}, n_b)$,
  not whole-grid.

## Options that change the equation {#xc-gradient-options-change-equation}

| Option | Effect |
|---|---|
| LDA, GGA, meta-GGA | $v_\sigma = v_\tau = 0$ for LDA, $v_\tau = 0$ for GGA; one code path with fewer terms |
| UKS, ROKS | the same per spin with $D\to D^\sigma$ and $\mathbf G^\sigma$ in place of $2v_\sigma\nabla\rho$ |
| hybrid, range-separated hybrid | exact exchange scaled by $c_K$, or split into erfc and erf pieces, through the routes of section @sec:df-jk-gradient |
| VV10 | `ferric-dft/src/gradient.rs::vv10_gradient_from_density` has no Becke-weight response by design, and is not finite-difference validated |
| grid | the gradient differentiates the grid the SCF ran on: `ferric-scf/src/ks_gradient.rs::KsGradGrids` carries it. A default-grid gradient paired with an SCF on another grid misses |
| meta-GGAs that need the density Laplacian (for example `MGGA_X_SCANL`, `MGGA_X_BR89`) | refused in `ferric-dft/src/libxc.rs::xc_def_from_name` (test `laplacian_metagga_rejected_by_flag`); the $\tau$-only functionals run |

The Hessian of the same energy, with the same two kinds of term, is
section @sec:ks-hessian.

## Numerical anchors {#xc-gradient-numerical-anchors}

| Check | Reference | Agreement | Test |
|---|---|---|---|
| Meta-GGA gradient, closed shell (SCAN, r2SCAN): H2O, NH3 / 6-31G, def2-SVP; RI-J and exact J | PySCF `RKS` `grid_response=True` on the same grid; FD of ferric's own energy | at most 1.3e-9 Ha/Bohr vs PySCF; analytic vs own FD 1.0e-9; bar 1e-8 | `crates/ferric-scf/tests/validation_mgga_gradients.rs` |
| Meta-GGA gradient, open shell: UKS HO2, NH2 / 6-31G; ROKS NH2 / 6-31G | PySCF `UKS` `grid_response=True`; central FD of PySCF `ROKS` | UKS at most 6.2e-9 Ha/Bohr, bar 3e-8; ROKS 2.2e-10 vs FD, bar 1e-7 | same |
| KS gradient on a custom or angularly pruned grid: water, NH2 / 6-31G | central FD of the energy on the same grid | 0.9e-7 to 1.4e-7 Ha/Bohr, bar 3e-7; the default-grid gradient paired with the same SCF misses by 2.8e-4 to 6.0e-4 | `crates/ferric-scf/tests/ks_gradient_scf_grid.rs` |

<!-- figures: site/src/reference/validation.md "Meta-GGA gradient, closed shell (SCAN, r2SCAN)"; site/src/reference/validation.md "Meta-GGA gradient, open shell (SCAN, r2SCAN)"; site/src/reference/validation.md "KS gradient on a custom or angularly pruned XC grid" -->

The first two rows check the grid-response terms against an independent
code: PySCF with `grid_response=True` differentiates the same moving grid.
The third row checks that the gradient differentiates the grid the SCF ran
on. Its miss of 2.8e-4 to 6.0e-4 Ha/Bohr is what pairing two different grids
costs, and it is at least 2000 times the finite-difference agreement of
0.9e-7 to 1.4e-7.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-dft/theory.md`, "XC nuclear gradient: AO-derivative and grid-response terms", "Becke partition and its nuclear derivative" | equations, tests |
| `scripts/dft_mgga_tau_gradient_sympy.py`, `scripts/dft_becke_weight_grad_sympy.py` | verified equations @eq:ks-grad-ao-derivative to @eq:ks-grad-becke-weights |
| `crates/ferric-dft/src/gradient.rs`, `crates/ferric-dft/src/gradient/screened.rs` | the algorithm and reproducibility rules |
| `crates/ferric-scf/tests/ks_gradient_scf_grid.rs` | what the custom-grid row tests |
| `site/src/reference/validation.md` | the anchor rows |
