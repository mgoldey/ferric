# XC gradient: AO-derivative and grid-response terms {#xc-gradient}

<!-- status: ready; registry: ks-gradient, vv10-gradient -->

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

`ks_gradient_roks` evaluates the XC term of the UKS route at the ROKS spin
densities $D_\alpha$, $D_\beta$, and builds $W$ in the ROHF form of section
@sec:hf-gradient with the Kohn–Sham spin Fock matrices.

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
$\sum_g(\partial w_g/\partial\mathbf R_C)\,e_g$, with the "weight1"
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

PBE on a distorted water in 6-31G, from the `3f612e0b3` build, on the
default grid. The forces sum to zero to $10^{-14}$. The central differences
($h = 10^{-3}$ Bohr) match the analytic gradient to 7.0e-8 and 1.1e-7
Ha/Bohr, the $O(h^2)$ error of the stencil and the 1e-7 of the custom-grid
row in the anchors table below. Every run stores the RI-J tensor in f64,
as in section @sec:df-jk-gradient. Without `ri_j_precision="f64"` the
energy-only runs keep the f32 default, and the same comparison misses by
1.1e-5 and 1.3e-5 Ha/Bohr: the f32 rounding of the fitted Coulomb energy,
divided by $2h$. The default also moves the analytic gradient itself by
1.6e-7 and 1.3e-7 Ha/Bohr, because `run_dft(with_gradient=True)` does not
switch RI-J to f64 on its own; that gradient is the derivative of the
rounded functional (section @sec:df-jk-gradient-options-change-equation).

```python
import numpy as np
import ferric

sym = ["O", "H", "H"]
x0 = np.array([[0.0, 0.0, 0.25], [0.0, 1.50, -0.85], [0.0, -1.35, -0.95]])
bs = ferric.BasisSet.bundled("6-31g")


def run(x, grad=False):
    mol = ferric.Molecule.from_arrays(sym, x, units="bohr")
    # f64 RI-J storage on every run: energy-only runs default to f32
    return ferric.run_dft(mol, bs, functional="PBE", energy_conv=1e-11, density_conv=1e-9,
                          ri_j_precision="f64", with_gradient=grad)


g = np.asarray(run(x0, grad=True).gradient())
print("PBE gradient (Ha/Bohr):\n", g)
print("sum over atoms:", g.sum(axis=0))
h = 1e-3
for atom, axis in ((0, 2), (2, 1)):
    xp = x0.copy(); xp[atom, axis] += h
    xm = x0.copy(); xm[atom, axis] -= h
    fd = (run(xp).total_energy - run(xm).total_energy) / (2 * h)
    print(f"atom {atom} axis {axis}: analytic {g[atom, axis]:.10f}  FD {fd:.10f}  diff {abs(fd - g[atom, axis]):.1e}")
```

```text
PBE gradient (Ha/Bohr):
 [[-8.14069181e-15 -2.16675889e-02 -9.43171220e-03]
 [-5.33206590e-16 -9.23320282e-03 -5.06386320e-03]
 [-6.04595639e-16  3.09007918e-02  1.44955754e-02]]
sum over atoms: [-9.27849404e-15  1.07587550e-14  2.38697950e-15]
atom 0 axis 2: analytic -0.0094317122  FD -0.0094317825  diff 7.0e-08
atom 2 axis 1: analytic 0.0309007918  FD 0.0309008985  diff 1.1e-07
```

The proof script uses $p$- and $s$-Gaussians on two centres at an exact
rational point; its controls (the $\tfrac12$ of $\tau$ kept, a halved spatial
gradient) fail. The open-shell path (`xc_gradient_uks_mgga_from_density`;
LDA and GGA `xc_gradient_uks_from_density`) is the same per spin with
$D\to D^\sigma$. The functional depends on
$\sigma_{st} = \nabla\rho^s\cdot\nabla\rho^t$, so the vector that multiplies
$\partial(\nabla\rho^\sigma)$ is

$$ \mathbf G^\alpha = 2v_{\sigma_{\alpha\alpha}}\nabla\rho^\alpha + v_{\sigma_{\alpha\beta}}\nabla\rho^\beta , $$

with $\mathbf G^\beta$ the same with $\alpha$ and $\beta$ swapped. It takes
the place of the closed-shell $2v_\sigma\nabla\rho$, and the GGA weight
column becomes $w\,G^\sigma_b$. Section @sec:ks-hessian uses the same
vector with the grid weight folded in.

The Rust tests of the whole term are `ferric-scf/tests/dft_gradient_mgga.rs`
(SCAN, r2SCAN and TPSS against finite differences of the SCF energy; SCAN
and r2SCAN against PySCF `grid_response=True`; UKS NH$_2$ and OH),
`ferric-scf/tests/dft_gradient_gga.rs` and `dft_gradient_lda.rs` (against
finite differences), and
`ferric-python/tests/test_xc_blocks.py::test_xc_grad_matches_finite_difference_at_fixed_density`
(PBE and SCAN, finite differences at fixed $D$, closed and polarized).

### VV10 nonlocal correlation {#xc-gradient-vv10}

The VV10 energy of section @sec:vv10 is a double sum over its own grid, the
NLC grid (default $50\times50$ per atom, unpruned). Like the main grid it is
built around the nuclei, so moving an atom changes three things: the AOs
under every point, the position of every point that rides that atom (and so
its distance to every point that rides another atom), and the Becke weights.
ferric differentiates the first and drops the other two. This section derives
all three, shows that the dropped part is a grid error that vanishes when the
quadrature is exact, and measures it on water.

Keep the notation of section @sec:vv10-energy-potentials: $W_i$ is the weight
of NLC point $i$ (the $w_g$ of section @sec:xc-gradient-terms), and

$$ E_{nl} = \sum_i W_i\rho_i\Bigl[\beta + \tfrac12 f_i\Bigr], \qquad f_i = \sum_p W_p\rho_p\,\Phi_{ip}, \qquad \Phi_{ip} = -\frac{3}{2\,g_ig_p(g_i+g_p)}, \qquad g_x = \omega_{0,x}R_{ip}^2 + \kappa_x . $$

$\rho_i$ and $\sigma_i$ enter through $\omega_0$ and $\kappa$; the pair
distance $R_{ip}$ enters only through $g$. Write $\mathrm{home}(i)$ for the
atom point $i$ rides, $\mathbf r_i = \mathbf R_{\mathrm{home}(i)} + \text{offset}$,
and $g_t = g_i + g_p$.

**Four terms.** Differentiate with respect to $R_{C,\alpha}$ at fixed $D$
(the SCF is stationary, as in section @sec:xc-gradient-terms). $E_{nl}$
depends on the nuclei through $\rho_i$ and $\sigma_i$ at a fixed point,
through the point's motion, which changes $\rho_i$, $\sigma_i$ and $R_{ip}$,
and through $W_i$. The partial derivatives of $E_{nl}$ with respect to
$\rho_i$ and $\sigma_i$ are $W_iv_{\rho,i}$ and $W_iv_{\sigma,i}$, the per-point
potentials of section @sec:vv10-energy-potentials. They already include the
factor 2 from $\rho_i$ appearing both as a row and as a partner. The chain
rule gives

$$ \frac{\mathrm dE_{nl}}{\mathrm dR_{C,\alpha}} = G^{\mathrm{AO}}_{C\alpha} + G^{\mathrm{home}}_{C\alpha} + G^{\mathrm{ker}}_{C\alpha} + G^{\mathrm{wt}}_{C\alpha} . $$

1. **The AO term.** The points stay where they are and the AOs on $C$ move:
   $$ G^{\mathrm{AO}}_{C\alpha} = \sum_i W_i\Bigl[v_{\rho,i}\frac{\partial\rho_i}{\partial R_{C,\alpha}}\Big|_{\mathbf r} + v_{\sigma,i}\frac{\partial\sigma_i}{\partial R_{C,\alpha}}\Big|_{\mathbf r}\Bigr] . $$
   It has the form of the GGA AO-derivative term of section @sec:xc-gradient-terms ($v_\tau = 0$),
   with the VV10 potentials in place of libxc's. The same $-2\sum_{\mu\in C}$
   assembly follows from the same derivatives of $\rho$ and $\partial_b\rho$.
   With the grid held fixed in space it is the whole derivative.
2. **The home term.** A point that rides $C$ sees the integrand move under
   it. Its $\rho_i$ and $\sigma_i$ change by their spatial gradients:
   $$ G^{\mathrm{home}}_{C\alpha} = \sum_{i:\,\mathrm{home}(i)=C} W_i\bigl[v_{\rho,i}\,\partial_\alpha\rho_i + v_{\sigma,i}\,\partial_\alpha\sigma_i\bigr], $$
   the semilocal home term of section @sec:xc-gradient-terms with the VV10 potentials.
3. **The kernel term.** A point that rides $C$ also moves relative to every
   point that rides another atom. Differentiate $R_{ip}^2$:
   $\partial R_{ip}^2/\partial R_{C,\alpha} = 2(\mathbf r_i - \mathbf r_p)_\alpha(\delta_{\mathrm{home}(i),C} - \delta_{\mathrm{home}(p),C})$.
   The pair term of $E_{nl}$ is $\tfrac12\sum_{i,p}W_i\rho_iW_p\rho_p\Phi_{ip}$,
   so the $\tfrac12$ cancels the 2:
   $$ G^{\mathrm{ker}}_{C\alpha} = \sum_{i,p} W_i\rho_i\,W_p\rho_p\,\frac{\partial\Phi_{ip}}{\partial R^2}\,(\mathbf r_i - \mathbf r_p)_\alpha\bigl(\delta_{\mathrm{home}(i),C} - \delta_{\mathrm{home}(p),C}\bigr) . $$
   $R^2$ enters $\Phi$ through $g_i$ and $g_p$, with $\partial g_i/\partial R^2 = \omega_{0,i}$,
   and $\partial\Phi/\partial g_i$ is the derivative of section @sec:vv10-energy-potentials:
   $$ \frac{\partial\Phi_{ip}}{\partial R^2} = \frac{3}{2\,g_ig_pg_t}\Bigl[\omega_{0,i}\Bigl(\frac1{g_i} + \frac1{g_t}\Bigr) + \omega_{0,p}\Bigl(\frac1{g_p} + \frac1{g_t}\Bigr)\Bigr] . $$
   Check the limits. A pair whose two points ride the same atom moves
   rigidly, and its bracket of Kronecker deltas is zero; the self pair
   $p = i$ is such a pair. On one atom $G^{\mathrm{ker}} = 0$.
4. **The weight term.** $W_i$ appears in $E_{nl}$ as a row prefactor and as a
   partner weight, exactly as $\rho_i$ does, so
   $$ \frac{\partial E_{nl}}{\partial W_i} = \rho_i\bigl(\beta + f_i\bigr), \qquad G^{\mathrm{wt}}_{C\alpha} = \sum_i \rho_i\bigl(\beta + f_i\bigr)\frac{\partial W_i}{\partial R_{C,\alpha}}, $$
   with the "weight1" derivative of section @sec:xc-gradient-becke-weight-derivative.
   The factor is $\beta + f_i$, not the energy per particle
   $\beta + \tfrac12 f_i$ that the semilocal weight term would suggest: the
   partner weight doubles the pair part.

**Check translational invariance.** Sum over all atoms $C$. The weight term
vanishes, because the weight1 rows sum to zero for every point. The kernel
term vanishes, because $\sum_C(\delta_{\mathrm{home}(i),C} - \delta_{\mathrm{home}(p),C}) = 0$
for every pair. The AO and home terms cancel by the identity
$\sum_C\partial q/\partial\mathbf R_C|_{\mathbf r} + \nabla_{\mathbf r}q = 0$
of section @sec:xc-gradient-terms. The four terms together give forces that
sum to zero on any grid.

**What the code drops.** `ferric-dft/src/gradient.rs::vv10_gradient_from_density`
computes $G^{\mathrm{AO}}$ alone. The dropped part is

$$ \Delta_{C\alpha} = G^{\mathrm{home}}_{C\alpha} + G^{\mathrm{ker}}_{C\alpha} + G^{\mathrm{wt}}_{C\alpha} . $$

What is it? Write the energy as a quadrature rule applied to an integrand,
$E_{nl} = Q_{\mathbf R}[F_{\mathbf R}]$. The rule $Q_{\mathbf R}$ is the set of
points and weights, which depend on the nuclei; the integrand $F_{\mathbf R}$
is built from the density, which depends on them through the AOs. The
derivative splits into the change of the integrand under a fixed rule, which
is $G^{\mathrm{AO}}$, and the change of the rule under a fixed integrand,

$$ \Delta = \frac{\mathrm d}{\mathrm d\mathbf R}\,Q_{\mathbf R}\bigl[F_{\mathbf R_0}\bigr]\Big|_{\mathbf R = \mathbf R_0} . $$

A rule that integrates $F_{\mathbf R_0}$ exactly gives the same number,
$\iint F_{\mathbf R_0}$, wherever the points sit, so $\Delta = 0$. On a finite
grid $\Delta$ is the derivative of the quadrature error, a grid error, and
it goes to zero as the grid is refined.

Two limits make this concrete.

- **One atom.** All AOs and all points ride the same nucleus, so the
  moving-grid energy does not depend on its position, and the exact
  gradient is zero. $G^{\mathrm{ker}} = G^{\mathrm{wt}} = 0$, and
  $G^{\mathrm{home}} = -G^{\mathrm{AO}}$ by the translation identity: ferric's
  VV10 gradient of a lone atom is pure grid error.
- **The net force.** Summed over atoms, the dropped part is
  $\sum_C\Delta_C = \sum_i W_i(v_{\rho,i}\nabla\rho_i + v_{\sigma,i}\nabla\sigma_i) = -\sum_C G^{\mathrm{AO}}_C$.
  This sum is the grid's estimate of the derivative of $E_{nl}$ under a rigid
  shift of the density with the grid held still. The continuous double
  integral does not change under that shift, because $\Phi$ depends only on
  $\lvert\mathbf r - \mathbf r'\rvert$ and on local values of $\rho$ and $\sigma$.
  So ferric's VV10 forces sum to zero only to the grid error, and their net
  force is a cheap measure of $\Delta$.

**What it means.** On a toy model (two atoms, five Gaussians, VV10 with
$b = 6$, $C = 0.01$, a product grid riding each atom and a smooth two-atom
partition; `scripts/dft_vv10_gradient_sympy.py`), the largest gradient
component is $3.3\times10^{-2}$ and the dropped part shrinks with the grid:

| Points | $\max\lvert\Delta\rvert$ | $\max\lvert\sum_C G^{\mathrm{AO}}_C\rvert$ |
|---|---|---|
| 384 | $4.3\times10^{-3}$ | $8.2\times10^{-4}$ |
| 1152 | $1.1\times10^{-3}$ | $2.4\times10^{-4}$ |
| 3072 | $1.2\times10^{-4}$ | $2.4\times10^{-5}$ |

On water in 6-31G, from the `3f612e0b3` build, with the default grids
(production main grid, $50\times50$ NLC grid), the size of $\Delta$ is
measured two ways.

- At a fixed density matrix (the converged ωB97X-V density), the XC gradient
  (`AoIntegrals.xc_functional(...).grad`) against a central difference of
  `eval` with $h = 10^{-4}$ Bohr: ωB97X-V misses by $5.4\times10^{-6}$
  Ha/Bohr (the O $z$ component), ωB97M-V, which carries the same VV10
  parameters, by the same $5.4\times10^{-6}$, and ωB97X without VV10
  (`HYB_GGA_XC_WB97X`) by $1.0\times10^{-9}$. The analytic ωB97X-V forces sum
  to $3.4\times10^{-6}$ Ha/Bohr; the ωB97X ones to $1\times10^{-14}$.
- End to end, `run_dft(..., with_gradient=True)` against a central difference
  of the re-converged energy ($h = 10^{-3}$ Bohr, `energy_conv=1e-12`,
  `density_conv=1e-10`, `ri_j_precision="f64"`): ωB97X-V misses by
  $5.4\times10^{-6}$ Ha/Bohr and ωB97X by $1.6\times10^{-7}$, the
  $O(h^2)$ error of the difference.

So the VV10 part of the ωB97X-V water gradient carries a grid error of
$5.4\times10^{-6}$ Ha/Bohr, about $3\times10^{-4}$ of the largest gradient
component ($2.0\times10^{-2}$), and nothing else in the gradient comes close.

**The code.** `vv10_gradient_from_density` builds the NLC grid with
`build_atomic_grid_pruned` from the same configuration the SCF used (it is
not stored on the result), and without the weight derivative. It evaluates
$\chi$, $\nabla\chi$ and the AO Hessian on it, then:

<!-- excerpt: ferric-dft/src/gradient.rs::vv10_gradient_from_density -->
```rust
// Density on the NLC grid.
let dens = crate::density_on_grid::eval_density_closed(d_total, &chi, &dchi);

// VV10 potentials from the pair-sum.
let (vrho, vsig) = crate::vv10::compute_vv10_potentials(&grid, &dens, params);
```

`ferric-dft/src/vv10.rs::compute_vv10_potentials` runs the same pair sum as
the SCF energy (the 40 Bohr cutoff and the $\rho < 10^{-8}$ skip included),
so $v_\rho$ and $v_\sigma$ are the derivatives of the energy the SCF
computed. `ferric-dft/src/gradient.rs::gga_gradient_from_potentials` then
assembles $G^{\mathrm{AO}}$ with $t_\rho = Wv_\rho$ and the weight column
$c_b = 2Wv_\sigma\partial_b\rho$, zeroing points with $\rho \le 10^{-10}$. The
open-shell gradients call the same function on the total density
(`ferric-scf/src/ks_gradient.rs::add_xc_gradient`), because $E_{nl}$ is a
functional of the total density.

**Design choices.** Two of the dropped terms cost more than the kept one.
$G^{\mathrm{ker}}$ is a vector-valued pair sum over the whole NLC grid, of the
same $O(N^2)$ order as the energy, and $G^{\mathrm{wt}}$ needs the weight1
derivative of every NLC point with respect to every atom, which the NLC path
does not build. $G^{\mathrm{home}}$ alone is cheap (the AO Hessian is already
evaluated), but adding it without the other two would not give the
derivative of any energy. What the omission costs is measured above: ωB97X-V
and ωB97M-V gradients match finite differences of their own energy to a few
$10^{-6}$ Ha/Bohr instead of $10^{-7}$, and their forces sum to zero only to
that level.

> **Verified equation {#eq:ks-grad-vv10-ao}** · the code's VV10 term is the derivative at a fixed grid
>
> On a two-atom Gaussian density with ferric's VV10 constants and a grid
> whose points ride the atoms: the code-form $v_\rho$, $v_\sigma$ (the
> $u_i$, $w_i$ sums) times $W_i$ are the per-point derivatives of $E_{nl}$,
> and $G^{\mathrm{AO}}$ assembled as `gga_gradient_from_potentials` does
> equals the central difference of $E_{nl}$ with the points and weights
> held fixed in space, to $1.1\times10^{-10}$ on a gradient of $3.3\times10^{-2}$.
>
> | | |
> |---|---|
> | Proof | `scripts/dft_vv10_gradient_sympy.py`, claims potentials and code |
> | Negative control | the runner builds $v_\sigma$ from $u_i$ in place of $w_i$ (claim code fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-dft/src/gradient.rs::{vv10_gradient_from_density,gga_gradient_from_potentials}` |
> | Code test | `wb97xv_h2o_ccpvdz_vs_pyscf_grad` in `crates/ferric-scf/tests/dft_gradient_vs_pyscf.rs` |
>
> <!-- verified: id=ks.grad.vv10.ao; script=scripts/dft_vv10_gradient_sympy.py; claim=code; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-dft/src/gradient.rs::{vv10_gradient_from_density,gga_gradient_from_potentials}; rust_test=ferric-scf/tests/dft_gradient_vs_pyscf.rs::wb97xv_h2o_ccpvdz_vs_pyscf_grad; registry=vv10-gradient -->

> **Verified equation {#eq:ks-grad-vv10-dropped}** · the full VV10 gradient and the part ferric drops
>
> On the same model: $\partial E_{nl}/\partial W_i = \rho_i(\beta + f_i)$ and
> the closed form of $\partial\Phi/\partial R^2$ (sympy, exact);
> $G^{\mathrm{AO}} + G^{\mathrm{home}} + G^{\mathrm{ker}} + G^{\mathrm{wt}}$ equals the
> central difference of the moving-grid energy to $1.5\times10^{-10}$ and
> sums to zero over atoms to $10^{-17}$, while $G^{\mathrm{AO}}$ alone has a net
> force of $8.2\times10^{-4}$; on one atom $G^{\mathrm{AO}} = -G^{\mathrm{home}}$ to
> $4\times10^{-14}$ relative; and the dropped part falls from
> $4.3\times10^{-3}$ to $1.2\times10^{-4}$ as the grid goes from 384 to 3072
> points.
>
> | | |
> |---|---|
> | Proof | `scripts/dft_vv10_gradient_sympy.py`, claims weights, kernel, full, invariance, limit |
> | Negative control | the script drops $G^{\mathrm{ker}}$ and uses $\rho_i(\beta + \tfrac12 f_i)$ in the weight term (each misses by more than $10^{-3}$); the runner halves $G^{\mathrm{ker}}$ and uses $\rho_i(\beta + \tfrac12 f_i)$ (claim full fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-dft/src/gradient.rs::vv10_gradient_from_density` |
>
> <!-- verified: id=ks.grad.vv10.dropped; script=scripts/dft_vv10_gradient_sympy.py; claim=full; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-dft/src/gradient.rs::vv10_gradient_from_density; registry=vv10-gradient -->

The proof covers the derivation on a model grid. It does not run ferric's
Becke partition or its NLC grid; the water numbers above are the measurement
of $\Delta$ on those. Nothing in ferric computes $G^{\mathrm{home}}$,
$G^{\mathrm{ker}}$ or $G^{\mathrm{wt}}$ for VV10.

## The Becke weight derivative {#xc-gradient-becke-weight-derivative}

The weight of grid point $g$ is $w_g = w_rw_l\,w^A(\mathbf r_g)$: radial and
angular quadrature weights times the Becke partition weight of its home
atom $A$. Section @sec:grids-becke-partition builds $w^A$ in four steps and
differentiates it by the chain rule through the same steps
(`ferric-dft/src/becke.rs::becke_weights_and_grad`); the derivation of
$\partial w^A/\partial\mathbf R_C$ is there. What the gradient needs from it
is one property. Every cell function depends only on differences of
positions, so $w^A$ is invariant under a rigid translation of nuclei and
point together:

$$ \sum_C \frac{\partial w^A}{\partial \mathbf{R}_C}\Big|_{\mathbf{r}} + \nabla_{\mathbf{r}} w^A = 0 . $$

`ferric-dft/src/grid.rs::build_atomic_grid_with_response` uses this to form
PySCF's "weight1" derivative, in which each point rides its home atom:

$$ \frac{\partial w_g}{\partial \mathbf{R}_C} = w_rw_l\Big[\frac{\partial w^{A}}{\partial\mathbf{R}_C} + \delta_{C,A}\nabla_{\mathbf{r}}w^{A}\Big], \qquad \sum_C \frac{\partial w_g}{\partial\mathbf{R}_C} = 0 . $$

The $\delta_{C,A}$ term is the weight's version of the home translation of
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
| 5 | weight response $\sum_g(\partial w_g/\partial\mathbf R_C)\,e_g$ | `weight1` |
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

- Given a density matrix, the result is bit-identical across rayon thread
  counts: the reduction groups and their order do not depend on the thread
  count. An end-to-end run also inherits the SCF's dependence on the BLAS
  thread count.
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
| UKS, ROKS | the same per spin with $D\to D^\sigma$ and $\mathbf G^\sigma$ of section @sec:xc-gradient-terms in place of $2v_\sigma\nabla\rho$ |
| hybrid, range-separated hybrid | exact exchange scaled by $c_K$, or split into erfc and erf pieces, through the routes of section @sec:df-jk-gradient |
| VV10 | only the AO term of the nonlocal energy; the NLC grid's point-motion, pair-distance and weight terms are dropped, a grid error of about 5e-6 Ha/Bohr on water (section @sec:xc-gradient-vv10) |
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
| sections @sec:xc-gradient-terms and @sec:grids-becke-partition | equations, tests |
| `scripts/dft_mgga_tau_gradient_sympy.py`, `scripts/dft_becke_weight_grad_sympy.py` | verified equations @eq:ks-grad-ao-derivative to @eq:ks-grad-becke-weights |
| `scripts/dft_vv10_gradient_sympy.py` | section @sec:xc-gradient-vv10, verified equations @eq:ks-grad-vv10-ao and @eq:ks-grad-vv10-dropped |
| `crates/ferric-dft/src/gradient.rs`, `crates/ferric-dft/src/gradient/screened.rs` | the algorithm and reproducibility rules |
| `crates/ferric-scf/tests/ks_gradient_scf_grid.rs` | what the custom-grid row tests |
| `site/src/reference/validation.md` | the anchor rows |
