# Kohn–Sham Hessian on a fixed and a moving grid {#ks-hessian}

<!-- status: ready; registry: ks-hessian -->

Move a nucleus and the density at every grid point changes twice: the basis
functions on that nucleus move with it, and then the orbitals relax to the
new geometry. Each change alters the XC potential, and how the XC potential
responds to a change of density is the kernel
$f_{xc} = \delta^2E_{xc}/\delta\rho^2$ of section @sec:fxc. The KS Hessian
therefore needs $f_{xc}$ twice: once at fixed orbitals, once in the
coupled-perturbed equations. On the production grid a third change appears,
because the grid points and their weights move with the nuclei.

The plan: the density derivatives and the fixed-grid terms (@sec:ks-hessian-xc-terms), the
moving grid and the open shell, then the algorithm (@sec:ks-hessian-algorithm), the refusals
(@sec:ks-hessian-reproducibility-refusals), the options (@sec:ks-hessian-options-change-equation) and the anchors (@sec:ks-hessian-numerical-anchors).

Everything else is the Hartree–Fock Hessian of section @sec:hf-hessian, with the
exact-exchange terms scaled by the hybrid fraction. The XC energy adds three
pieces: a skeleton at fixed density, a first-order XC Fock matrix, and the
kernel $f_{xc}$ in the CPHF operator. One more choice decides what those
pieces are: which energy is differentiated. On a grid fixed in space the
points and weights do not move. On the production grid each point rides its
home atom and the Becke weights move with the nuclei, which adds weight
derivatives to every XC term.

| Symbol | Meaning | Code name |
|---|---|---|
| $E_{xc} = \sum_g W_g f(\rho_g,\sigma_g)$ | the XC energy on the grid, $\sigma = \lvert\nabla\rho\rvert^2$ | `FixedGridKs` |
| $\rho_x$, $\mathbf g_x$, $\rho_{xy}$, $\mathbf g_{xy}$ | first and second nuclear derivatives of $\rho$ and $\nabla\rho$ at fixed $D$ | `rho_derivative`, `grad_rho_derivative`, `add_same_centre`, `add_cross_centre` |
| $F = Wf_\rho$, $\mathbf G = 2Wf_\sigma\mathbf g$ | per-point weights | `weights_response` |
| $V[F,\mathbf G]$ | the XC matrix built from per-point weights | `kernel_part` |
| $W'$, $W''$ | nuclear derivatives of the Becke weight at a fixed point | `ferric-dft/src/becke_hessian.rs::HomeWeightDerivs` |
| $T$, $G_h$ | translation map and the point-fixed second derivative of home atom $h$'s batches | `translation_fill` |

| Entry point | Location |
|---|---|
| Rust, fixed grid | `ferric-scf/src/ks_fixed_grid.rs::FixedGridKs::{skeleton_hessian,first_order_fock}`, library entry only |
| Rust, moving grid | `ferric-scf/src/ks_fixed_grid.rs::FixedGridKs::{skeleton_hessian_moving,first_order_fock_moving}`, `ferric-dft/src/becke_hessian.rs::becke_home_weight_hess` |
| Rust, open shell | `ferric-scf/src/ks_fixed_grid/polarized.rs`, `ferric-scf/src/hessian/uks.rs::uks_moving_grid_hessian` |
| CLI, Python | `task = "frequencies"` and `run_frequencies(xc=...)` under `hessian = "auto"` or `"analytic-moving-grid"` |

The fixed-grid Hessian differentiates a frozen-grid energy (PySCF
`grid_response=False`). The production SCF does not minimize that energy,
so the fixed-grid Hessian is a library entry that `hessian = "auto"` never
selects. The moving-grid Hessian differentiates the energy that the
production KS SCF minimizes and that the KS gradient of section @sec:xc-gradient
differentiates.

## The XC terms {#ks-hessian-xc-terms}

The code is `ferric-scf/src/ks_fixed_grid.rs`, for closed-shell LDA, GGA and
global hybrids. The exchange and Coulomb Hessian terms are in
`ferric-scf/src/hessian/ks.rs`. `FixedGridKs::from_points` accepts only LDA,
GGA and hybrid-GGA components. A meta-GGA is refused before the SCF, with
the message "tau-dependent XC nuclear derivatives" from
`ferric-scf/src/hessian/ks.rs`, so no $\tau$ terms appear below. An LDA is
the case $f_\sigma = 0$: then $\mathbf G = 0$ and every $\mathbf g_x$ term
drops out. The first-order identities used here, $V_{xc} = \partial
E_{xc}/\partial D$, the per-point linearization of $f_{xc}$ and the Becke
weight derivatives, are proved in the scripts cited by the verified boxes of
chapter @sec:kohn-sham and section @sec:xc-gradient.

### How the density moves with a nucleus

The XC energy is a sum over grid points,

$$ E_{xc} = \sum_g W_g\, f(\rho_g, \sigma_g), \qquad \sigma = \lvert\mathbf g\rvert^2, \qquad \mathbf g = \nabla\rho . $$

At each point the density is $\rho = \sum_{\mu\nu}D_{\mu\nu}\chi_\mu\chi_\nu$.
Two short names keep the formulas readable: $P_\mu = (D\chi)_\mu$ and
$Q_{\mu\gamma} = (D\,\partial_\gamma\chi)_\mu$, where $\partial_\gamma$ is the
derivative with respect to the electron coordinate $\gamma$. Write a nuclear
coordinate as $x = (B,\alpha)$, atom $B$ and Cartesian direction $\alpha$,
and a second one as $y = (C,\beta)$.

Differentiate $\rho$ with respect to $x$ at fixed $D$. Only the AOs centred
on $B$ move. A Gaussian centred at $\mathbf R_B$ depends on
$\mathbf r - \mathbf R_B$, so moving the centre by $+\delta$ is the same as
moving the point by $-\delta$: $\partial\chi_\mu/\partial R_{B\alpha} =
-\partial_\alpha\chi_\mu$ for $\mu \in B$. $D$ is symmetric, so the two
factors of $\chi_\mu\chi_\nu$ give the same contribution and the sum carries
a factor 2. The density gradient follows the same way:

$$ g_\gamma = 2\sum_\mu \partial_\gamma\chi_\mu P_\mu, \qquad
\rho_x = -2\sum_{\mu\in B}\partial_\alpha\chi_\mu P_\mu, \qquad
g_{\gamma,x} = -2\sum_{\mu\in B}\big[\partial_\gamma\partial_\alpha\chi_\mu P_\mu + \partial_\alpha\chi_\mu Q_{\mu\gamma}\big] . $$

Differentiate once more with respect to $y$. If $B = C$ the same AO is
differentiated twice. If $\mu \in B$ and $\nu \in C$, both AOs move, one
for each coordinate:

$$ \rho_{xy} = 2\delta_{BC}\sum_{\mu\in B}\partial_\alpha\partial_\beta\chi_\mu P_\mu
 + 2\sum_{\mu\in B,\nu\in C} D_{\mu\nu}\,\partial_\alpha\chi_\mu\,\partial_\beta\chi_\nu , $$

$$ g_{\gamma,xy} = 2\delta_{BC}\sum_{\mu\in B}\big[\partial_\gamma\partial_\alpha\partial_\beta\chi_\mu P_\mu + \partial_\alpha\partial_\beta\chi_\mu Q_{\mu\gamma}\big]
 + 2\sum_{\mu\in B,\nu\in C} D_{\mu\nu}\big[\partial_\gamma\partial_\alpha\chi_\mu\,\partial_\beta\chi_\nu + \partial_\alpha\chi_\mu\,\partial_\gamma\partial_\beta\chi_\nu\big] . $$

The two terms of $\rho_{xy}$ are the code's `add_same_centre` and
`add_cross_centre`.

### Where $f_{xc}$ enters

Start with the gradient on a grid fixed in space, $W_g = w_g$ constant.
Apply the chain rule through $\rho$ and $\sigma$, with
$\sigma_x = 2\,\mathbf g\cdot\mathbf g_x$:

$$ \frac{\partial E}{\partial x} = \sum_g W\big(f_\rho\rho_x + 2f_\sigma\,\mathbf g\cdot\mathbf g_x\big) = \sum_g F\rho_x + \mathbf G\cdot\mathbf g_x, \qquad F = Wf_\rho, \quad G_\gamma = 2Wf_\sigma g_\gamma . $$

$F$ and $\mathbf G$ are the per-point weights of the XC potential, the same
ones the SCF builds $V_{xc}$ from (section @sec:vxc). For an LDA, $G = 0$.

Now differentiate the gradient with respect to $y$, still at fixed $D$. The
product rule gives two kinds of term. Differentiating $\rho_x$ and
$\mathbf g_x$ gives $F\rho_{xy} + \mathbf G\cdot\mathbf g_{xy}$. Differentiating
$F$ and $\mathbf G$ asks how the potential weights change when the density
changes by $(\rho_y, \mathbf g_y)$. That is the kernel. For a general change
$(\rho_1, \mathbf g_1)$, apply the chain rule to $F = Wf_\rho(\rho,\sigma)$
and $\mathbf G = 2Wf_\sigma(\rho,\sigma)\,\mathbf g$ with
$\delta\sigma = 2\,\mathbf g\cdot\mathbf g_1$:

$$ \delta F[\rho_1,\mathbf g_1] = W\big(f_{\rho\rho}\rho_1 + 2f_{\rho\sigma}\,\mathbf g\cdot\mathbf g_1\big), \qquad
\delta G_\gamma[\rho_1,\mathbf g_1] = 2W\big[(f_{\rho\sigma}\rho_1 + 2f_{\sigma\sigma}\,\mathbf g\cdot\mathbf g_1)g_\gamma + f_\sigma g_{1\gamma}\big] . $$

The last term of $\delta G$ is the product rule acting on $\mathbf g$
itself. Collecting both kinds gives the fixed-grid skeleton:

$$ \frac{\partial^2 E}{\partial x\,\partial y}\Big|_D = \sum_g \rho_x\,\delta F[y] + \mathbf g_x\cdot\delta\mathbf G[y] + F\rho_{xy} + \mathbf G\cdot\mathbf g_{xy} . $$

Check it on an LDA. Then $\mathbf G = 0$, $\delta\mathbf G = 0$, and the
skeleton is $\sum_g W(f_{\rho\rho}\rho_x\rho_y + f_\rho\rho_{xy})$: the
second derivative of $W f(\rho)$ by the chain rule, term for term.

### The first-order Fock matrix and the CPHF kernel

The CPHF equations of section @sec:hf-hessian need two more XC objects. Both are built
with one matrix-valued map from per-point weights to an AO matrix:

$$ V_{\mu\nu}[F,G] = \sum_g F\chi_\mu\chi_\nu + G_\gamma(\partial_\gamma\chi_\mu\chi_\nu + \chi_\mu\partial_\gamma\chi_\nu) . $$

With the SCF's own $F$ and $\mathbf G$ this is $V_{xc}$. The first-order Fock
matrix is the derivative of $V_{xc}$ with respect to $x$ at fixed $D$. Two
things in $V$ depend on $x$: the weights $F,\mathbf G$, through the density,
and the AOs of atom $B$ in the matrix element itself. The first gives the
kernel matrix $V[\delta F[x],\delta G[x]]$. The second gives a row term for
every $\mu \in B$ and, by symmetry, a column term for every $\nu \in B$:

$$ \frac{\partial V_{\mu\nu}}{\partial x}\Big|_D = V_{\mu\nu}[\delta F[x],\delta G[x]] - \big(T^{\alpha}_{\mu\nu}[\mu\in B] + T^{\alpha}_{\nu\mu}[\nu\in B]\big),\qquad
T^\alpha_{\mu\nu} = \sum_g F\,\partial_\alpha\chi_\mu\chi_\nu + G_\gamma\big(\partial_\gamma\partial_\alpha\chi_\mu\chi_\nu + \partial_\alpha\chi_\mu\partial_\gamma\chi_\nu\big) . $$

The minus sign is the same $\partial\chi/\partial R = -\partial\chi$ as above.

The CPHF operator needs the change of $V_{xc}$ when the orbitals relax and
the density matrix changes by $D_1$. The AOs do not move, so only the weights
change:

$$ V^{(1)}[D_1] = V\big[\delta F[\rho_1,\mathbf g_1],\,\delta G[\rho_1,\mathbf g_1]\big], \qquad \rho_1 = \sum D_{1,\mu\nu}\chi_\mu\chi_\nu, \quad \mathbf g_1 = 2\sum D_{1,\mu\nu}\nabla\chi_\mu\chi_\nu . $$

$V^{(1)}$ is the closed-shell form of the linearized potential of section @sec:fxc-linearized-potential.
It is the XC part of the KS orbital Hessian; section @sec:second-order derives the
Hartree–Fock part. All three objects here are the derivatives of
PySCF's `grid_response=False` surface.

> **Verified equation {#eq:ks-hess-fixed-grid}** · the fixed-grid skeleton, first-order Fock matrix and kernel
>
> On a three-atom model with $s$, $p$ and $d_{xy}$ Gaussians, a symmetric
> density matrix and a GGA-like $f(\rho,\sigma)$ with nonzero $f_{\rho\sigma}$
> and $f_{\sigma\sigma}$, each code-form quantity, assembled term by term from
> the electron-coordinate AO derivatives, equals sympy's exact nuclear
> derivative at 50 digits: $\rho_x$, $\mathbf g_x$, $\rho_{xy}$, $\mathbf g_{xy}$
> (claims 1 to 4); the second-order contraction (claim 5); the kernel
> weights $\delta F$, $\delta G$ (claim 6); the skeleton (claim 7); the
> gradient (claim 8); the first-order Fock matrix at frozen $D$ (claim 9); and
> the kernel response $V^{(1)}[D_1] = \mathrm d V(D+tD_1)/\mathrm dt$ (claim 10).
>
> | | |
> |---|---|
> | Proof | `scripts/ks_fixed_grid_hessian_sympy.py`, claims 1 to 10 (cited by claims 7 and 9 here) |
> | Negative control | the runner drops the factor 2 of the $f_{\sigma\sigma}$ coefficient (claim 6, with 7, 9 and 10, fails) and negates a term of the AO-derivative Fock rows (claim 9 fails); the script carries a control for each factor |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` (validation tier, about 12 minutes) |
> | Code | `ferric-scf/src/ks_fixed_grid.rs::FixedGridKs::{skeleton_hessian,first_order_fock}` |
> | Code test | `gga_xc_pieces_match_frozen_density_differences` in `crates/ferric-scf/tests/ks_fixed_grid_hessian.rs` |
>
> <!-- verified: id=ks.hess.fixed-grid; script=scripts/ks_fixed_grid_hessian_sympy.py; claim=7; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/ks_fixed_grid.rs::FixedGridKs::skeleton_hessian,ferric-scf/src/ks_fixed_grid.rs::FixedGridKs::first_order_fock; rust_test=ferric-scf/tests/ks_fixed_grid_hessian.rs::gga_xc_pieces_match_frozen_density_differences; registry=ks-hessian -->

### The moving grid

On the production grid each point rides its home atom $h$:
$\mathbf r_g = \mathbf R_h + \mathbf u_g$, with $\mathbf u_g$ fixed. The
weight is $W_g = s_g\,w^h(\mathbf r_g;\mathbf R)$, the radial and angular
quadrature weight $s_g$ times the Becke cell weight of section @sec:grids-becke-partition. Moving
atom $h$ now moves the point. Moving any other atom changes $w^h$ through
the Becke partition, which is the derivative $W'$ of section @sec:xc-gradient-becke-weight-derivative.

Differentiating with respect to $\mathbf R_h$ directly would mean following
the point. Translation invariance avoids that. Move every nucleus and every
point by the same vector: the AO values, the density and the Becke weight do
not change, so the contribution $E_g$ of one point depends only on the
differences $\mathbf R_B - \mathbf R_h$. Its derivatives with respect to the
atoms therefore sum to zero, and the home atom's derivative is minus the sum
of the others:

$$ \frac{\partial E_g}{\partial R_{h\alpha}} = -\sum_{B\ne h}\frac{\partial E_g}{\partial R_{B\alpha}} . $$

The derivatives with respect to $B \ne h$ hold $\mathbf R_h$ and so the point
fixed. They are the point-fixed derivatives of the fixed grid plus the
weight terms. Write the map from the $B\ne h$ coordinates to all coordinates
as $T$, with $T_{x,x} = 1$ and $T_{(h,\alpha),(B,\alpha)} = -1$ for
$B \ne h$. Applying it on both sides fills the home rows and columns of the
second derivative:

$$ H^{(h)} = T\,G_h\,T^{\mathsf T},\qquad
G_h = \sum_{g\in h}\big[W\,\partial^2 f + W'(\partial f)^{\mathsf T} + \partial f\,W'^{\mathsf T} + W'' f\big]_{\mathbf r\ \text{fixed}},
\quad \partial_x f = f_\rho\rho_x + 2f_\sigma\,\mathbf g\cdot\mathbf g_x , $$

over the atoms $B, C \ne h$. The four terms of $G_h$ are the product rule
on $W f$: both factors differentiated twice, or each once. The first is the
fixed-grid skeleton above.

The first-order Fock matrix of a home-$h$ batch follows the same rule. For
$x \notin h$ it is the point-fixed derivative, which now includes the weight
term $V[W'_x f_\rho,\,2W'_x f_\sigma\mathbf g]$. The home rows come from
translation invariance: $V^{(h,\alpha)} = -\sum_{B\neq h}V^{(B,\alpha)}$.

> **Verified equation {#eq:ks-hess-moving-grid}** · the moving-grid skeleton and first-order Fock matrix
>
> For points $\mathbf r_g = \mathbf R_h + \mathbf u_g$ riding home atom $h$ and
> a generic translation-invariant weight $W(\mathbf r;\mathbf R)$,
> $\partial^2E/\partial\mathbf R\,\partial\mathbf R = T\,G_h\,T^{\mathsf T}$ with
> $G_h$ the point-fixed second derivative
> $W\,\partial^2f + W'(\partial f)^{\mathsf T} + \partial f\,W'^{\mathsf T} + W''f$,
> and the first-order Fock matrix of atoms other than $h$ is the point-fixed
> derivative including $V[W'_xf_\rho,\,2W'_xf_\sigma\mathbf g]$, with the home
> rows filled by translation (claims 11 and 12). The script proves the
> riding and translation-fill algebra for any such $W$.
>
> | | |
> |---|---|
> | Proof | `scripts/ks_fixed_grid_hessian_sympy.py`, claims 11 and 12 |
> | Negative control | the validation page records seven moving-grid mutants (dropping $W''$, the $W'f'$ cross terms, the weight term of the Fock matrix, the translation fill, or flipping its sign) that miss the frozen-density anchors by $10^{6}$ to $10^{9}$ bars |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` (validation tier) |
> | Code | `ferric-scf/src/ks_fixed_grid.rs::FixedGridKs::{skeleton_hessian_moving,first_order_fock_moving},ferric-scf/src/ks_fixed_grid.rs::translation_fill` |
> | Code test | `moving_grid_mutants_each_fail_a_frozen_density_anchor` in `crates/ferric-scf/tests/ks_moving_grid_hessian.rs` |
>
> <!-- verified: id=ks.hess.moving-grid; script=scripts/ks_fixed_grid_hessian_sympy.py; claim=11; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/ks_fixed_grid.rs::FixedGridKs::{skeleton_hessian_moving,first_order_fock_moving},ferric-scf/src/ks_fixed_grid.rs::translation_fill; rust_test=ferric-scf/tests/ks_moving_grid_hessian.rs::moving_grid_mutants_each_fail_a_frozen_density_anchor; registry=ks-hessian -->

$W'$ and $W''$ are the nuclear derivatives of the Becke weight at a fixed
point. The first derivative is section @sec:xc-gradient's equation @eq:ks-grad-becke-weights. The second
follows from the same chain, differentiated once more: the pair coordinate
$\mu$, the step function $s(\mu)$, the cell product $P_A$ and the quotient
$w^h = P_h/T$.

> **Verified equation {#eq:ks-hess-becke-second-derivative}** · the Becke weight second derivative
>
> The pair quantities $\mu_x = N_x/R - \mu R_x/R$ and
> $\mu_{xy} = N_{xy}/R - (N_xR_y+N_yR_x)/R^2 - \mu R_{xy}/R + 2\mu R_xR_y/R^2$,
> with $N = r_A - r_B$ and $R = R_{AB}$, equal the exact derivatives with respect to
> the six nuclear coordinates of one pair, and $s_{xy} = s_{\mu\mu}\mu_x\mu_y + s_\mu\mu_{xy}$
> (claim W1). The cell weight $w^h = P_h/T$ with $P_A = \prod_Bs_{AB}$ has
> $P' = PL'$, $P'' = P(L''+L'L'^{\mathsf T})$ with $L = \sum_B\ln s_{AB}$, and
> $w'' = P_h''/T - (P_h'T'^{\mathsf T}+T'P_h'^{\mathsf T})/T^2 - P_hT''/T^2 + 2P_hT'T'^{\mathsf T}/T^3$;
> this assembly equals the exact second derivative of the quotient for the Becke
> three-iteration step function on three nuclei, to 40 digits (claim W2).
>
> | | |
> |---|---|
> | Proof | `scripts/dft_becke_weight_hessian_sympy.py`, claims W1 and W2. The first derivative is `scripts/dft_becke_weight_grad_sympy.py`, which is section @sec:xc-gradient's equation @eq:ks-grad-becke-weights |
> | Negative control | the runner drops the $2\mu R_xR_y/R^2$ term (claim W1 fails) and the $2P_hT'T'^{\mathsf T}/T^3$ term (claim W2 fails). The script also requires the quantities it compares to be of order one, because a point deep inside one cell makes every derivative zero to 40 digits and every formula pass |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-dft/src/becke_hessian.rs::becke_home_weight_hess` |
> | Code test | `partition_weight_derivative_matches_fd` in `crates/ferric-dft/src/becke.rs` |
>
> <!-- verified: id=ks.hess.becke-second-derivative; script=scripts/dft_becke_weight_hessian_sympy.py; claim=W2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-dft/src/becke_hessian.rs::becke_home_weight_hess; rust_test=ferric-dft/src/becke.rs::partition_weight_derivative_matches_fd; registry=ks-hessian -->

The collinear three-nucleus geometry of claim W2 proves the assembly, which is
dimension-free. The three-dimensional pair derivatives are claim W1. The pair
size adjustment $a_{AB}$ is held constant, as in the code.

**The exactness anchor** is a single atom. Every point rides the only atom,
there is no $B \ne h$, and $E_g$ depends on no difference of positions. The
weight is identically 1, and the moving-grid XC skeleton and first-order
Fock matrix are exactly zero, where the fixed-grid ones are not. That anchor
cannot see the weight terms, because a single atom has none. The
frozen-density finite differences of the validation page carry the weight
derivatives.

### The code

The translation fill is `ferric-scf/src/ks_fixed_grid.rs::translation_fill`.
It builds $T$ explicitly and forms $T\,G_h\,T^{\mathsf T}$. The excerpt is the
body after the `n3` declaration:

<!-- excerpt: ferric-scf/src/ks_fixed_grid.rs::translation_fill -->
```rust
let mut t = Array2::<f64>::zeros((n3, n3));
for x in (0..n3).filter(|x| x / 3 != home) {
    t[(x, x)] = 1.0;
    t[(3 * home + x % 3, x)] = fill;
}
t.dot(&g).dot(&t.t())
```

| Code | Equation |
|---|---|
| `x / 3 != home` | the coordinates of atoms $B \ne h$ |
| `t[(x, x)] = 1.0` | $T_{x,x} = 1$ |
| `t[(3 * home + x % 3, x)] = fill` | $T_{(h,\alpha),(B,\alpha)} = -1$; `fill` is $-1$ in production and a mutation parameter in the tests |
| `g` | $G_h$, with zero home rows and columns |

The kernel weights $\delta F$ and $\delta\mathbf G$ are the GGA branch of
`ferric-scf/src/ks_fixed_grid.rs::weights_response` (the LDA branch, which
returns $W f_{\rho\rho}\rho_1$ alone, is not shown):

<!-- excerpt: ferric-scf/src/ks_fixed_grid.rs::weights_response -->
```rust
let gg1: Array1<f64> = (g * g1).sum_axis(Axis(0));
let df = w * &(&(&k.frr * rho1) + &(2.0 * &(&k.frs * &gg1)));
let coef = 2.0 * &(w * &(&(&k.frs * rho1) + &(2.0 * &(&k.fss * &gg1))));
let wfs = 2.0 * &(w * &k.fs);
let mut dg = Array2::<f64>::zeros(g.dim());
for gam in 0..3 {
    let row = &(&coef * &g.row(gam)) + &(&wfs * &g1.row(gam));
    dg.row_mut(gam).assign(&row);
}
```

| Code | Equation |
|---|---|
| `gg1` | $\mathbf g\cdot\mathbf g_1$, one value per point |
| `df` | $\delta F = W(f_{\rho\rho}\rho_1 + 2f_{\rho\sigma}\,\mathbf g\cdot\mathbf g_1)$ |
| `coef` | $2W(f_{\rho\sigma}\rho_1 + 2f_{\sigma\sigma}\,\mathbf g\cdot\mathbf g_1)$, the coefficient of $\mathbf g$ in $\delta\mathbf G$ |
| `wfs` | $2Wf_\sigma$, the coefficient of $\mathbf g_1$ |
| `dg` | $\delta\mathbf G$, three rows (one per direction $\gamma$) by points |

Each array runs over the points of one batch, so the formula is evaluated
for every point at once. The factor 2 in `df` is the $\delta\sigma =
2\,\mathbf g\cdot\mathbf g_1$ of the chain rule; dropping the factor 2 of
the $f_{\sigma\sigma}$ coefficient is one of the runner's mutations of box
@eq:ks-hess-fixed-grid.

### Open shell (UKS)

`ferric-scf/src/hessian/uks.rs::uks_moving_grid_hessian` runs every expression
above once per spin with $D \to D^\sigma$. The derivatives $\rho^\sigma_x$,
$\mathbf g^\sigma_x$, $\rho^\sigma_{xy}$, $\mathbf g^\sigma_{xy}$ keep their
form exactly, with $P^\sigma = D^\sigma\chi$ and $Q^\sigma = D^\sigma\partial\chi$.
The factors 2 come from the $\mu\leftrightarrow\nu$ symmetry of $D$, not from
spin, so they stay. The functional is
$f(\rho_\alpha,\rho_\beta,\sigma_{\alpha\alpha},\sigma_{\alpha\beta},\sigma_{\beta\beta})$
with $\sigma_{st} = \mathbf g^s\cdot\mathbf g^t$. Its potential weights are

$$ F^\sigma = Wf_{\rho_\sigma},\qquad \mathbf G^\alpha = W(2f_{\sigma_{\alpha\alpha}}\mathbf g^\alpha + f_{\sigma_{\alpha\beta}}\mathbf g^\beta) , $$

and $\mathbf G^\beta$ swaps $\alpha$ and $\beta$. A perturbation
$(\rho_1^t, \mathbf g_1^t)$ of both spins changes the three $\sigma$
variables by the product rule,

$$ \delta\sigma_{\alpha\alpha} = 2\mathbf g^\alpha\cdot\mathbf g_1^\alpha,\quad \delta\sigma_{\alpha\beta} = \mathbf g^\alpha\cdot\mathbf g_1^\beta + \mathbf g^\beta\cdot\mathbf g_1^\alpha,\quad \delta\sigma_{\beta\beta} = 2\mathbf g^\beta\cdot\mathbf g_1^\beta , $$

and the weights by the chain rule:

$$ \delta F^\sigma = W\Big[\sum_t f_{\rho_\sigma\rho_t}\rho_1^t + \sum_u f_{\rho_\sigma\sigma_u}\delta\sigma_u\Big],\qquad
\delta\mathbf G^\alpha = W\big[2\mathbf g^\alpha\,\delta f_{\sigma_{\alpha\alpha}} + \mathbf g^\beta\,\delta f_{\sigma_{\alpha\beta}} + 2f_{\sigma_{\alpha\alpha}}\mathbf g_1^\alpha + f_{\sigma_{\alpha\beta}}\mathbf g_1^\beta\big] , $$

with $\delta f_v = \sum_t f_{v\rho_t}\rho_1^t + \sum_u f_{v\sigma_u}\delta\sigma_u$.
The second derivatives come from libxc's `v2rho2`, `v2rhosigma` and
`v2sigma2`, with 3, 6 and 6 components. The skeleton is

$$ \sum_\sigma\big[\rho^\sigma_x\delta F^\sigma[y] + \mathbf g^\sigma_x\cdot\delta\mathbf G^\sigma[y] + F^\sigma\rho^\sigma_{xy} + \mathbf G^\sigma\cdot\mathbf g^\sigma_{xy}\big] , $$

where $\delta F^\sigma[y]$ responds to the perturbation
$(\rho^t_y,\mathbf g^t_y)$ of both spins. The first-order Fock matrix is
$\partial V^\sigma/\partial x|_D = V[\delta F^\sigma[x],\delta\mathbf G^\sigma[x]]$
plus the AO-derivative rows of $V[F^\sigma,\mathbf G^\sigma]$.

On the moving grid the $T\,G_h\,T^{\mathsf T}$ fill and the weight terms are
spin-blind, apart from two places: $\partial_x f =
\sum_\sigma(F^\sigma\rho^\sigma_x + \mathbf G^\sigma\cdot\mathbf g^\sigma_x)/W$,
and the per-spin Fock weight term $V[W'_xf_{\rho_\sigma},\,W'_x\mathbf G^\sigma/W]$.

The two-electron part is the UHF Hessian with exchange scaled by the
global-hybrid fraction $a$:
$\Gamma = \tfrac12DD - \tfrac a2\sum_\sigma D^\sigma D^\sigma$. Where the SCF
fitted them, RI-J runs on the total density and RI-K on one channel per spin
with weight $-a/2$. The coupled UCPHF uses

$$ G^\sigma[D_1] = J[D_1^\alpha + D_1^\beta] - aK[D_1^\sigma] + V^{(1)\sigma}_{xc}[D_1^\alpha, D_1^\beta] , $$

whose XC part is `ferric-scf/src/ks_fixed_grid/polarized.rs::PolarizedXcResponse`,
with $\alpha\alpha$, $\alpha\beta$ and $\beta\beta$ blocks. Each spin's
potential pieces are gated on its own $\rho_\sigma$, as in `ferric_dft`'s
polarized $V_{xc}$.

`scripts/uks_xc_hessian_sympy.py` proves these term by term against exact
sympy derivatives with $D^\alpha\neq D^\beta$ and $D_1^\alpha\neq D_1^\beta$,
including the closed-shell limit. It also shows an error a singlet anchor
cannot see. A "total-density" kernel, which replaces each spin's
perturbation by the spin average, reproduces the closed-shell limit exactly
and misses the polarized response.

> **Verified equation {#eq:ks-hess-polarized}** · the spin-polarized algebra
>
> With two different density matrices $D^\alpha\neq D^\beta$ and two different
> response densities, on an $f$ that is not spin-blind: the per-spin weights
> and gradient (claim 1), the polarized kernel $\delta F^\sigma$,
> $\delta\mathbf G^\sigma$ for a perturbation of both spins (claim 2), the
> skeleton (claim 3), the first-order Fock matrices (claim 4), the kernel
> response (claim 5), the moving-grid forms (claim 6) and the closed-shell
> limit (claim 7), each against sympy's exact derivative at 50 digits.
>
> | | |
> |---|---|
> | Proof | `scripts/uks_xc_hessian_sympy.py`, claims 1 to 7 (cited by claim 5 here) |
> | Negative control | a "total-density" kernel, each spin's perturbation replaced by the spin average, reproduces the closed-shell limit (claim 7) exactly and misses claim 5. A singlet anchor therefore cannot see it, which is why the anchors use unequal spins |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` (validation tier, about 30 minutes) |
> | Code | `ferric-scf/src/ks_fixed_grid/polarized.rs::{PolarizedKs,PolarizedXcResponse}` |
> | Code test | `polarized_kernel_matches_density_differences_for_unequal_spin_changes` in `crates/ferric-scf/tests/uks_xc_pieces.rs` |
>
> <!-- verified: id=ks.hess.polarized; script=scripts/uks_xc_hessian_sympy.py; claim=5; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/ks_fixed_grid/polarized.rs::{PolarizedKs,PolarizedXcResponse}; rust_test=ferric-scf/tests/uks_xc_pieces.rs::polarized_kernel_matches_density_differences_for_unequal_spin_changes; registry=ks-hessian -->

ROKS is not implemented: its response is Roothaan-coupled.
`HessianMethod::Analytic` and `AnalyticMovingGrid` refuse it before the SCF,
and ROKS frequencies are central differences of the analytic gradient.

### ECPs

$V_{\rm ECP}$ is part of $h$ and does not enter $E_{xc}$. It therefore adds
the same two terms to the RKS and UKS Hessians as to the RHF and UHF ones:
the skeleton $\sum_{\mu\nu} D_{\mu\nu}\,\partial^2 V^{\rm ECP}_{\mu\nu}/\partial x\partial y$,
built in `ferric-scf/src/hessian_ecp.rs` as a stencil of the first-derivative
blocks with $D$ the total density, and $\partial V^{\rm ECP}/\partial x$ in
every first-order Fock matrix $F^x$, the same for both spins. The XC terms
above do not change.

### Where each equation is tested

| Equation | Rust | Numerical test |
|---|---|---|
| $g$, $\rho_x$, $\mathbf g_x$ | `gradient_densities`, `rho_derivative`, `grad_rho_derivative` | `ferric-scf/tests/ks_fixed_grid_hessian.rs::gradient_matches_central_difference_of_the_fixed_grid_energy` |
| $F\rho_{xy}+\mathbf G\cdot\mathbf g_{xy}$, $\delta F$, $\delta G$ | `add_same_centre`, `add_cross_centre`, `weights_response`, `kernel_part` | `xc_pieces_match_frozen_density_differences`, `gga_xc_pieces_match_frozen_density_differences` |
| $\partial V/\partial x$ | `first_order_fock`, `add_ao_derivative_fock` | same two tests (FD of $V_{xc}$ at displaced nuclei) |
| $V^{(1)}[D_1]$ | `XcResponse::apply` | `xc_kernel_response_matches_difference_of_vxc_in_the_density`, `gga_kernel_response_…` |
| polarized $F^\sigma,\mathbf G^\sigma,\delta F^\sigma,\delta\mathbf G^\sigma$, skeleton, $\partial V^\sigma/\partial x$, $V^{(1)\sigma}$ | `ks_fixed_grid/polarized.rs` (`PolarizedKs`, `pol_response`, `PolarizedXcResponse`) | `uks_xc_pieces.rs` (closed-shell limit, frozen-density and frozen-grid FD, kernel FD with $D_1^\alpha\neq D_1^\beta$, mutants); full Hessian `uks_moving_grid_hessian.rs` |
| $T G_h T^{\mathsf T}$, moving $\partial V/\partial x$ | `skeleton_hessian_moving`, `translation_fill`, `first_order_fock_moving`, `weight_terms` | `ferric-scf/tests/ks_moving_grid_hessian.rs::moving_xc_pieces_match_frozen_density_differences`, `moving_grid_mutants_each_fail_a_frozen_density_anchor` |
| ECP skeleton and $\partial V^{\rm ECP}/\partial x$ in the KS Hessian | `extra_skeleton`, `add_extra_first_order` (`hessian.rs`), called from `hessian/ks.rs` and `hessian/uks.rs` | `crates/ferric-scf/tests/ks_ecp_hessian.rs` (RKS HI, UKS H-I-H vs FD of the production gradient, both terms as negative controls); `ferric-scf/tests/validation_ks_fixed_grid_hessian.rs::fixed_grid_pbe_hessian_with_an_ecp_matches_pyscf_on_the_same_grid` |

## The algorithm {#ks-hessian-algorithm}

The Coulomb, exchange, overlap and one-electron terms and the CPHF solver are
those of section @sec:hf-hessian. The XC pieces are added at three places.

| Step | XC piece | Function |
|---|---|---|
| 1 | skeleton $\sum_g\rho_x\delta F[y]+\mathbf g_x\!\cdot\!\delta\mathbf G[y]+F\rho_{xy}+\mathbf G\!\cdot\!\mathbf g_{xy}$ (plus the weight terms on the moving grid) | `FixedGridKs::skeleton_hessian`, `skeleton_hessian_moving` |
| 2 | first-order XC Fock matrix $\partial V_{xc}/\partial x$ at frozen $D$, added to $F^y$ | `FixedGridKs::first_order_fock`, `first_order_fock_moving` |
| 3 | kernel $V^{(1)}[D_1]$ in every CPHF operator application | `ferric-scf/src/ks_fixed_grid.rs::XcResponse` |
| 4 | moving grid: per home atom, $W'$ and $W''$ contracted point by point, never stored | `ferric-dft/src/becke_hessian.rs::becke_home_weight_hess` |
| 5 | moving grid: home-atom rows and columns by translation invariance, $H^{(h)} = T\,G_h\,T^{\mathsf T}$ | `ferric-scf/src/ks_fixed_grid.rs::translation_fill` |

Step 4 is where the moving grid costs more. The $W''$ of one point is a
$3N\times3N$ matrix, so the code contracts it with $f$ at that point and adds
it to a running sum per batch, instead of storing it per point. The Becke
weight second derivative costs $O(n^3)$ per point for $n$ atoms, and the
translation fill is a dense $(3N)^3$ product per batch, so the cost ratio
against the fixed grid grows with $N$. The validation page records 1.07 to
1.10 on three atoms and no measurement above that.

## Reproducibility and refusals {#ks-hessian-reproducibility-refusals}

- The moving-grid route needs a grid with home atoms. A grid built from
  explicit points is refused (`a_grid_without_home_atoms_is_refused` in
  `crates/ferric-scf/tests/ks_moving_grid_hessian.rs`).
- `hessian = "auto"` takes the analytic KS Hessian wherever
  `ferric-scf/src/hessian/ks.rs::rks_moving_grid_hessian_available` accepts the
  system, and finite differences otherwise. `"analytic"` reaches neither KS
  Hessian.
- A failed UKS response solve under `"auto"` runs finite differences with a
  warning. The recorded case is the OH $^2\Pi$ exact $\pi$-hole zero mode.

## Options that change the equation {#ks-hessian-options-change-equation}

| Option | Effect |
|---|---|
| LDA, GGA, global hybrids | covered. $f_\sigma = 0$ for LDA |
| meta-GGA | refused before the SCF: no $\tau$ terms appear above |
| VV10, range-separated hybrids, ROKS | finite differences of the analytic gradient |
| RI-J, RI-K | the RHF fitted Hessians of section @sec:hf-hessian supply the Coulomb and exchange parts |
| ECP | supported on the RKS and UKS routes: the skeleton and $\partial V^{\rm ECP}/\partial x$ terms above |
| external potential (point charges, a field, smeared charges) | refused on the KS routes |
| grid | every atom-centred grid is supported, pruned or flat, with or without a weight cutoff: each point still rides its home atom |

## Numerical anchors {#ks-hessian-numerical-anchors}

| Check | Reference | Agreement | Test |
|---|---|---|---|
| Moving-grid RKS Hessian (LDA, GGA, global hybrids; exact or RI-J, RI-K): distorted H2O / STO-3G, 6-31G, cc-pVDZ, 75x110 grid | central FD (Richardson) of the production gradient with a production SCF at every displaced geometry | 7.9e-11 (LDA) and 4.8e-11 (PBE) at STO-3G; 1.7e-10 (LDA / 6-31G, default grid); 1.05e-9 (PBE / 6-31G); 2.6e-10 (LDA / cc-pVDZ); harmonic frequencies at most 5.0e-7 cm⁻¹ against the symmetrized FD Hessian | `crates/ferric-scf/tests/validation_ks_moving_grid_hessian.rs`, `crates/ferric-scf/tests/ks_moving_grid_hessian.rs` |
| the same system with the fixed-grid Hessian | the same FD | misses by 1.1e-3 Ha/Bohr² and 3.0 cm⁻¹ | same |
| Fixed-grid RKS Hessian (library only): distorted H2O / STO-3G, 6-31G, cc-pVDZ | FD of the fixed-grid gradient; PySCF `dft.RKS` Hessian with `grid_response=False` on ferric's own grid | vs FD of the fixed-grid gradient 5.7e-11 (STO-3G), 1.9e-8 (6-31G), 8.1e-9 (cc-pVDZ); vs PySCF 4.1e-8 (6-31G) and 9.3e-7 (cc-pVDZ) | `crates/ferric-scf/tests/validation_ks_fixed_grid_hessian.rs` |
| UKS moving-grid Hessian: NH2, CH3 doublets, O2 triplet / STO-3G, 6-31G; LDA, PBE, B3LYP | Richardson FD of the production UKS gradient | at most 8.5e-10 Ha/Bohr², bar 2e-9; singlet UKS equals RKS to 2.7e-12; every spin-polarized mutant misses by at least 6.6e6 bars | `crates/ferric-scf/tests/uks_moving_grid_hessian.rs`, `crates/ferric-scf/tests/uks_xc_pieces.rs` |

<!-- figures: site/src/reference/validation.md "Analytic closed-shell KS Hessian on the MOVING (production) grid"; site/src/reference/validation.md "Analytic closed-shell KS Hessian (LDA, GGA, and global hybrids of both)"; site/src/reference/validation.md "Open-shell KS frequencies" -->

The second row is the reason the production route is the moving grid. The
fixed-grid Hessian is the exact derivative of a different energy, and on
distorted water it misses the production surface's curvature by 3.0 cm⁻¹.

On PySCF's analytic cc-pVDZ Hessian: about 9.2e-7 of it is not the
derivative of PySCF's own gradient, and that accounts for all of the 9.3e-7
by which the fixed-grid Hessian differs from it. The cause on the PySCF side
is not isolated, so the fixed-grid agreement is stated at the FD level.

## Sources

| Source | Used for |
|---|---|
| `scripts/ks_fixed_grid_hessian_sympy.py`, `scripts/uks_xc_hessian_sympy.py` | verified equations @eq:ks-hess-fixed-grid to @eq:ks-hess-polarized |
| `scripts/dft_becke_weight_hessian_sympy.py`, claims W1 and W2; `crates/ferric-dft/src/becke_hessian.rs` module documentation | verified equation @eq:ks-hess-becke-second-derivative |
| `crates/ferric-scf/src/hessian/ks.rs` module and `embedding_refusal` documentation | the ECP, external-potential and grid rows of @sec:ks-hessian-options-change-equation |
| `site/src/reference/validation.md` | the anchor rows |
