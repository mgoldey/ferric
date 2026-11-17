# Radial and angular grids, the Becke partition, pruning {#grids}

<!-- status: ready; registry: ks-grid;
     theory: crates/ferric-dft/theory.md#"Integration Grid" "Becke partition and its nuclear derivative" -->

The XC energy is an integral over all space of a function of the density,
and for a functional like PBE there is no closed form for it. So it is done
numerically. The integrand is sharply peaked at every nucleus (the density
has a cusp there) and decays smoothly between them. A single grid over the
whole molecule handles that badly. ferric splits the integral into one
piece per atom, each a smooth function centred on its own nucleus, and
integrates each piece on a spherical grid around that atom.

The plan:

1. Build one atom's grid: a radial rule times a Lebedev sphere, with the
   angular order pruned by radius (@sec:grids-atomic-grid).
2. Glue the atomic grids together with Becke's fuzzy partition of space
   (@sec:grids-becke-partition).
3. Assemble the molecular grid and drop the points that carry no weight
   (@sec:grids-building-molecular-grid).

| Symbol | Meaning | Code name |
|---|---|---|
| $n_r$, $n_\Omega$ | radial shells, Lebedev points per shell (before pruning) | `n_radial`, `n_angular` |
| $x_k$, $r_k$, $w^{\rm rad}_k$ | Chebyshev node, radius, radial weight with $4\pi r^2$ | `x`, `r`, `w_r` |
| $\xi$ | Treutler–Ahlrichs scale per element | `TA_XI` |
| $w^\Omega_j$ | Lebedev weight, $\sum_j w^\Omega_j = 1$ | |
| $w^A(\mathbf r)$ | Becke weight of atom $A$ | `becke_weights_all` |
| $R^{\rm BS}_A$ | Bragg–Slater radius | `bragg_slater_bohr` |

| Entry point | Location |
|---|---|
| Rust | `ferric-dft/src/grid.rs::build_atomic_grid`, `ferric-dft/src/radial.rs::treutler_ahlrichs_m4`, `ferric-quadrature/src/lebedev.rs::lebedev`, `ferric-dft/src/prune.rs::angular_orders_for_atom`, `ferric-dft/src/becke.rs::{becke_weights_all,becke_weights_and_grad}` |
| CLI | `[dft] grid_radial`, `grid_angular`, `grid_prune`, `grid_weight_cut` |
| Python | the same keywords on `run_dft`, `run_ksdft`, `run_tddft`; `dft_grid_point_count` |

## The atomic grid {#grids-atomic-grid}

Each atom carries a Treutler–Ahlrichs radial grid times a Lebedev angular
grid. The atomic grids are combined with the Becke partition weights
$w^A(\mathbf r)$ of section @sec:grids-becke-partition, so a point on atom $A$ has weight
$w^{\rm rad}_k\,w^\Omega_j\,w^A(\mathbf r)$.

**The radial rule.** We need $\int_0^\infty 4\pi r^2f(r)\,dr$ for an $f$
that is large near $r = 0$ and decays slowly at large $r$. Start from a rule
on the finite interval $[-1, 1]$ and map it onto $[0, \infty)$. The nodes are
the second-kind Chebyshev points,

$$ x_k = \cos\theta_k, \qquad \theta_k = \frac{\pi k}{n+1}, \qquad k = 1, \dots, n . $$

Treutler and Ahlrichs' M4 map sends $x$ to a radius:

$$ r(x) = -\frac{\xi}{\ln 2}\,(1+x)^{0.6}\ln\frac{1-x}{2} . $$

Check the ends. As $x \to -1$ the factor $(1+x)^{0.6}$ goes to zero and
$r \to 0$. As $x \to 1$ the logarithm diverges and $r \to \infty$. In between
$\tfrac{1-x}{2} \in (0, 1)$, so the logarithm is negative and $r > 0$. The
scale $\xi$ stretches the map per element (`TA_XI`: H 0.8, C 1.1, N 0.9,
O 0.9, ...; 1.5 past the table).

Change variables twice. First $r \to x$, which brings in $dr/dx$. Product
rule on the map:

$$ \frac{dr}{dx} = \frac{\xi}{\ln 2}(1+x)^{0.6}\Bigl[-\frac{0.6}{1+x}\ln\frac{1-x}{2} + \frac{1}{1-x}\Bigr] . $$

Then $x = \cos\theta$, which turns
$\int_{-1}^1 g(x)\,dx$ into $\int_0^\pi g(\cos\theta)\sin\theta\,d\theta$.
The trapezoid rule in $\theta$ with step $\pi/(n+1)$ gives the weight. With
the $4\pi r^2$ of the volume element folded in:

$$ w^{\rm rad}_k = \frac{\pi}{n+1}\sin\theta_k\,\Bigl|\frac{dr}{dx}\Bigr|_{x_k}\,4\pi r_k^2, \qquad \int_0^\infty 4\pi r^2 f(r)\,dr \approx \sum_k w^{\rm rad}_k f(r_k) . $$

The weight is $\sin\theta_k$, not $\sin^2\theta_k$. The $\sin^2\theta_k$
weight belongs to the Chebyshev rule for $\int f(x)\sqrt{1-x^2}\,dx$, a
different integral. The verified box below uses it as the negative control.

> **Verified equation {#eq:ks-grid-radial}** · the Treutler–Ahlrichs M4 radial rule
>
> The loop's $dr/dx$ is the derivative of the M4 map, and the 75-point rule
> with weight $\tfrac{\pi}{n+1}\sin\theta_k\,|dr/dx|\,4\pi r^2$ integrates
> $4\pi r^2e^{-ar^2}$ for $a = 0.3$ to 40 to $(\pi/a)^{3/2}$ within
> $10^{-12}$ (H, O and a third scale); the radii come out smallest first.
>
> | | |
> |---|---|
> | Proof | `scripts/dft_becke_weight_grad_sympy.py`, claim radial (check 6) |
> | Negative control | the runner squares the Chebyshev weight, $\sin\theta_k \to \sin^2\theta_k$ (the rule for $\int f\sqrt{1-x^2}$), and the check fails at $2.6\times10^{-1}$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-dft/src/radial.rs::treutler_ahlrichs_m4` |
> | Code test | `treutler_xi_matches_pyscf` in `crates/ferric-dft/src/radial.rs` |
>
> <!-- verified: id=ks.grid.radial; script=scripts/dft_becke_weight_grad_sympy.py; claim=radial; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-dft/src/radial.rs::treutler_ahlrichs_m4; rust_test=ferric-dft/src/radial.rs::treutler_xi_matches_pyscf; registry=ks-grid -->

The loop body of `ferric-dft/src/radial.rs::treutler_ahlrichs_m4` is the
three formulas in order:

<!-- excerpt: ferric-dft/src/radial.rs::treutler_ahlrichs_m4 -->
```rust
let r = -ln2 * one_plus.powf(alpha) * log_term;
// dr/dx = ln2 * (1+x)^α * [-α/(1+x) * ln((1-x)/2) + 1/(1-x)]
let dr_dx = ln2 * one_plus.powf(alpha) * (-alpha / one_plus * log_term + 1.0 / one_minus);
// Chebyshev-2 integration weight: |dx/dθ| · dθ = sin(θ) · (π/(n+1))
let w_cheb = (pi / np1) * theta.sin();
let w_r = w_cheb * dr_dx * 4.0 * pi * r * r;
```

| Symbol | Code |
|---|---|
| $\xi/\ln2$ | `ln2` (the name is the code's) |
| $0.6$ | `alpha` |
| $1 \pm x_k$, $\ln\frac{1-x_k}{2}$ | `one_plus`, `one_minus`, `log_term` |
| $w^{\rm rad}_k$ | `w_r` |

Two points about it:

- The loop runs $k = 1, \dots, n$, which is large $x$ (large $r$) first, and
  then reverses both arrays. The radii are returned smallest first, as
  PySCF returns them.
- `w_cheb * dr_dx` has no absolute value. $dr/dx > 0$ on $(-1, 1)$: both
  terms in the bracket are positive, since $\ln\frac{1-x}{2} < 0$.

**The angular rule.** On each radial shell, `ferric-quadrature/src/lebedev.rs::lebedev`
supplies a Lebedev–Laikov rule, with the weights normalized to
$\sum_j w^\Omega_j = 1$. The rules live in `ferric_quadrature::lebedev`, which
the PCM cavity spheres (section @sec:cavity-and-bem) also use; `ferric_dft::lebedev`
(`crates/ferric-dft/src/lebedev.rs`) re-exports it, so code that names
the `ferric_dft` path gets the same tables. The $4\pi$ of the sphere already sits in the radial
weight. A point's weight is therefore
$w_g = w^{\rm rad}_k\,w^\Omega_j\,w^A(\mathbf r_g)$, and
$\sum_g w_gf(\mathbf r_g) \approx \int f\,d\mathbf r$.

**Pruning.** Near the nucleus the density is close to spherical, and far
out it is small. Neither region needs the full angular order. Each radial
shell therefore takes its Lebedev order from a five-region table. The
region of a shell at radius $r$ is the number of boundaries $a$ with
$r/R^{\rm BS}_A > a$, where $R^{\rm BS}$ is the Bragg–Slater radius:

| Element row | Region boundaries $a$ |
|---|---|
| H, He | 0.25, 0.5, 1.0, 4.5 |
| Li to Ne | 0.1667, 0.5, 0.9, 3.5 |
| Na and heavier | 0.1, 0.4, 0.8, 2.5 |

That count is one line in `ferric-dft/src/prune.rs::angular_orders_for_atom`:

<!-- excerpt: ferric-dft/src/prune.rs::angular_orders_for_atom -->
```rust
let x = r / r_atom;
let region = alphas.iter().filter(|&&a| x > a).count();
table[region]
```

The table of orders per region (`ferric-dft/src/prune.rs::region_orders`)
starts from NWChem's request $[\ell_1, \ell_3, \ell_{p-1}, \ell_p, \ell_{p-1}]$
on the ladder $\ell = (38, 50, 74, 86, 110, 146, \dots)$, with $\ell_p$ the
requested order. ferric then puts the innermost region on order 26 and
snaps the other four up onto its own rule set
$\{6, 14, 26, 50, 110, 194, 302, 434, 590, \dots\}$, never past $\ell_p$.
At $\ell_p = 110$ the regions get $(26, 110, 110, 110, 110)$; at 302,
$(26, 110, 302, 302, 302)$. Orders below 50 are not pruned. Order 50 is
refused: its snapped table would be larger than the flat grid.

The default scheme, `NwchemRadial`, also shortens the radial grid of the
light elements. `ferric-dft/src/prune.rs::radial_shells` gives
$\lfloor(n_r m + 37)/75\rfloor$ shells, with $m = 50$ for H and He and
$m = 66$ for Li to Ne (so 50 and 66 at $n_r = 75$), and the full $n_r$
beyond Ne. The `Sgx` and ORCA `DefGrid` schemes use their own tables
(`sgx_region_orders`, `defgrid_region_orders`).

**The weight cut.** A point whose final weight $\lvert w_g\rvert$ is below
`weight_cut` is dropped. The KS default is $10^{-14}$ (`DEFAULT_WEIGHT_CUT`,
ORCA's `WeightCut`).

**Four grids.** The constructors of `AtomicGridConfig` name the grids the
code uses:

| Constructor | Radial $\times$ angular | Prune | Weight cut | Used for |
|---|---|---|---|---|
| `AtomicGridConfig::production()` | $75\times110$ | `NwchemRadial` | $10^{-14}$ | every KS path when no grid is set |
| `AtomicGridConfig::default()` | $75\times110$ | none | none | analysis grids, flat-grid tests |
| `AtomicGridConfig::default_nlc()` | $50\times50$ | none | none | the VV10 nonlocal grid |
| `AtomicGridConfig::meta_gga_response()` | $150\times434$ | `NwchemRadial` | $10^{-14}$ | `response_default_grid` when any component is a meta-GGA (TDDFT) |

A Kohn–Sham calculation with no grid set (`RhfConfig::dft_grid = None`, the
CLI `[dft]` grid keys absent, the Python grid keywords `None`) runs on
`production()`. Every KS path (SCF, gradients, Hessians, response, TDDFT)
resolves `None` to this grid, so they all integrate on the same points.
`default()` is the flat, uncut $75\times110$ grid. It is the quadrature for
analysis code (Hirshfeld volumes, partitions, property grids) and for tests
that pin flat-grid numbers; it is not the KS default. The VV10 grid is a
separate flat $50\times50$ grid. The prune table has no entry at angular
order 50, so that grid is never pruned.

Setting only a size keeps the default scheme and cutoff at that size;
`grid_prune = "none"` with `grid_weight_cut = 0` is the flat grid. The
analytic gradient differentiates the energy on the grid the SCF ran on,
pruning and weight cut included, by dropping exactly the same points.

The region tables and point counts are tested in `prune::tests`, the $\xi$
table against PySCF in `radial::tests`, and the finer meta-GGA response grid
in `grid::tests::response_default_grid_is_finer_for_meta_gga_only`.

![The default grid of water's atoms: the radius of each Treutler–Ahlrichs shell and its Lebedev order after pruning, for O (66 shells) and H (50 shells), with the region boundaries in units of the Bragg–Slater radius. Water's pruned grid has 14,060 points against 24,750 flat (weight cut off), 43% fewer; the script checks its transcription of the grid against `ferric.dft_grid_point_count`. Script: `book/src/figures/09-kohn-sham/atomic_grid.py`.](../figures/09-kohn-sham/atomic_grid.svg)

## The Becke partition {#grids-becke-partition}

A point $\mathbf r$ on atom $A$'s grid also lies near atom $B$. If both
atomic grids integrated the full integrand, the overlap region would be
counted twice. Becke's fix is to split the integrand: give every atom a
weight function $w^A(\mathbf r) \ge 0$ with $\sum_A w^A(\mathbf r) = 1$
everywhere, and let atom $A$'s grid integrate only $w^A f$. The pieces add
up to $f$, and each piece is concentrated around its own nucleus, where
that atom's grid is fine.

The construction goes in four steps, each a smooth version of "is
$\mathbf r$ closer to $A$ than to $B$?".

1. **Elliptical coordinate.** With $r_X = \lvert\mathbf r - \mathbf R_X\rvert$
   and $R_{AB} = \lvert\mathbf R_A - \mathbf R_B\rvert$,
   $$ \mu_{AB} = \frac{r_A - r_B}{R_{AB}} . $$
   By the triangle inequality $\mu_{AB} \in [-1, 1]$: $-1$ at nucleus $A$,
   $+1$ at nucleus $B$, 0 on the bisecting plane.
2. **Size adjustment.** Unequal atoms should not split at the midpoint. With
   $\chi = R^{\rm BS}_A/R^{\rm BS}_B$ and $u = (\chi - 1)/(\chi + 1)$,
   $$ a_{AB} = \operatorname{clip}_{[-1/2,\,1/2]}\frac{u}{u^2-1}, \qquad \nu_{AB} = \mu_{AB} + a_{AB}(1 - \mu_{AB}^2) . $$
   The added term vanishes at $\mu = \pm1$, so the nuclei stay at
   $\nu = \pm1$. Its derivative is $\partial\nu/\partial\mu = 1 - 2a_{AB}\mu$,
   which the clip keeps non-negative on $\lvert\mu\rvert \le 1$: $\nu$ stays
   monotone in $\mu$.
3. **Cell function.** Sharpen the step with three iterations of
   $f(x) = \tfrac12(3x - x^3)$:
   $$ s_{AB} = \tfrac12\bigl(1 - f^{(3)}(\nu_{AB})\bigr) . $$
   $f$ fixes $-1$, 0 and 1, and $f'(\pm1) = 0$, so $f^{(3)}$ goes from $-1$
   to 1 with zero slope at both ends. $s_{AB}$ goes from 1 at $A$ to 0 at
   $B$, and is $\tfrac12$ on the bisecting plane when $a_{AB} = 0$.
4. **Normalize.** Multiply the cell functions and divide by the sum over
   atoms:
   $$ P_A = \prod_{B\ne A}s_{AB}, \qquad w^A = \frac{P_A}{T}, \qquad T = \sum_C P_C . $$

Check the limit at a nucleus. At $\mathbf r = \mathbf R_A$, every
$\mu_{AB} = -1$, so $s_{AB} = 1$ and $P_A = 1$. Every other atom $C$ has
$\mu_{CA} = +1$, so $s_{CA} = 0$ and $P_C = 0$. Then $w^A = 1$: the point
belongs entirely to $A$.

The four steps are the inner loop of `ferric-dft/src/becke.rs::becke_weights_all`:

<!-- excerpt: ferric-dft/src/becke.rs::becke_weights_all -->
```rust
let mu = (r_dists[a] - r_dists[b]) / r_ab;
let chi = r_a_bs / r_b_bs;
let u = (chi - 1.0) / (chi + 1.0);
let a_corr = (u / (u * u - 1.0)).clamp(-0.5, 0.5);
let nu = mu + a_corr * (1.0 - mu * mu);
let smoothed = becke_smoothing(nu, 3);
let s_ab = 0.5 * (1.0 - smoothed);
p_cell[a] *= s_ab;
```

| Symbol | Code |
|---|---|
| $\mu_{AB}$, $\chi$, $u$, $a_{AB}$, $\nu_{AB}$ | `mu`, `chi`, `u`, `a_corr`, `nu` |
| $f^{(3)}(\nu)$ | `becke_smoothing(nu, 3)` |
| $s_{AB}$, $P_A$ | `s_ab`, `p_cell[a]` |

After the loop the function divides each `p_cell` by their sum, and returns
all zeros if the sum is below $10^{-30}$.

**The nuclear derivative.** Gradients and Hessians need
$\partial w^A/\partial\mathbf R_C$, because moving a nucleus moves the
partition. `ferric-dft/src/becke.rs::becke_weights_and_grad` codes it by the
chain rule through the same four steps. Differentiate $\mu_{AB}$ with
respect to each nucleus (the quotient rule, with
$\partial R_{AB}/\partial R_A^\alpha = (\mathbf R_A - \mathbf R_B)^\alpha/R_{AB}$):

$$ \frac{\partial\mu_{AB}}{\partial R_A^\alpha} = -\frac{(\mathbf{r}-\mathbf{R}_A)^\alpha}{r_A R_{AB}} - \frac{\mu_{AB}}{R_{AB}^2}(\mathbf{R}_A-\mathbf{R}_B)^\alpha, \qquad \frac{\partial\mu_{AB}}{\partial R_B^\alpha} = +\frac{(\mathbf{r}-\mathbf{R}_B)^\alpha}{r_B R_{AB}} + \frac{\mu_{AB}}{R_{AB}^2}(\mathbf{R}_A-\mathbf{R}_B)^\alpha . $$

Then through $\nu$ and the three iterations of $f$:

$$ \frac{\partial\nu}{\partial\mu} = 1 - 2a_{AB}\mu, \qquad \frac{df^{(3)}}{d\nu} = f_2'f_1'f_0', \qquad f_k' = \tfrac32(1 - f_k^2), $$

where $f_k$ is the $k$-th iterate. Last, the quotient rule on $w^A = P_A/T$:

$$ \frac{\partial w^A}{\partial R_C} = \frac{1}{T}\frac{\partial P_A}{\partial R_C} - \frac{P_A}{T^2}\sum_D\frac{\partial P_D}{\partial R_C} . $$

**Translation invariance.** Every $\mu_{AB}$ depends only on differences of
positions, so it does not change when all nuclei and the point move
together. The same holds for $w^A$, which gives

$$ \sum_C \frac{\partial w^A}{\partial \mathbf{R}_C}\Big|_{\mathbf{r}} + \nabla_{\mathbf{r}} w^A = 0 . $$

`ferric-dft/src/grid.rs::build_atomic_grid_with_response` uses this to form
PySCF's "weight1" derivative, in which each grid point rides along with its
home atom $A$:

$$ \frac{\partial w_g}{\partial \mathbf{R}_b} = w_rw_l\Bigl[\frac{\partial w^{A}}{\partial\mathbf{R}_b} + \delta_{b,A}\nabla_{\mathbf{r}}w^{A}\Bigr], \qquad \sum_b \frac{\partial w_g}{\partial\mathbf{R}_b} = 0 . $$

The sum rule follows from the invariance above: summing the bracket over
$b$ gives $\sum_b\partial w^A/\partial\mathbf R_b + \nabla_{\mathbf r}w^A = 0$.

> **Verified equation {#eq:ks-grid-becke}** · Becke weights: partition of unity and their nuclear derivative
>
> For a three-atom cluster with unequal radii at a generic point, the
> loop's quotient-rule assembly of $\partial w^A/\partial\mathbf R_C$ equals
> sympy's derivative of the definition to $10^{-40}$ (60-digit geometry),
> $\sum_A\partial w^A/\partial\mathbf R_C = 0$, and
> $\nabla_{\mathbf r}w^A = -\sum_C\partial w^A/\partial\mathbf R_C$.
>
> | | |
> |---|---|
> | Proof | `scripts/dft_becke_weight_grad_sympy.py`, claims assembly, pou, translation |
> | Negative control | the runner drops the $\tfrac12$ of $ds/d\mu$, and flips the sign of the $\mu/R_{AB}^2$ term of $d\mu/dR_B$; the script's own controls drop the $dT/T^2$ term and the $\mu/R_{AB}^2$ term |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-dft/src/becke.rs::becke_weights_and_grad`, `ferric-dft/src/grid.rs::build_atomic_grid_with_response` |
> | Code test | `becke_weights_grad_finite_difference_ch` in `crates/ferric-dft/src/becke.rs` |
>
> <!-- verified: id=ks.grid.becke; script=scripts/dft_becke_weight_grad_sympy.py; claim=assembly; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-dft/src/becke.rs::becke_weights_and_grad; rust_test=ferric-dft/src/becke.rs::becke_weights_grad_finite_difference_ch; registry=ks-grid -->

The script checks each derivative symbolically and the assembled gradient
of the three-atom cluster against sympy to $10^{-61}$. The Rust tests are
`becke::tests::becke_weights_grad_finite_difference_ch` (finite differences,
$<10^{-7}$), `becke::tests::becke_weights_grad_sums_to_zero_h2` and
`becke::tests::partition_weight_derivative_matches_fd` (the neighbour-list
kernel); the KS gradients test the whole chain end to end. The sum rule
$\sum_b\partial w_g/\partial\mathbf R_b = 0$ for the assembled weight1 rows
has no direct Rust test; only the KS gradients cover it.

## Building a molecular grid {#grids-building-molecular-grid}

| Step | What | Function |
|---|---|---|
| 1 | per atom: $n_r$ shells (fewer for H–Ne under `NwchemRadial`) | `radial_shells`, `treutler_ahlrichs_m4` |
| 2 | per shell: the Lebedev order of its region | `angular_orders_for_atom`, `region_orders` |
| 3 | per point: $w_g = w^{\rm rad}_kw^\Omega_jw^A(\mathbf r_g)$ | `BeckeTable::weight` (bit-identical to `becke_weights_all`) |
| 4 | drop $\lvert w_g\rvert$ below `weight_cut` | `build_atomic_grid` |
| 5 | for gradients and Hessians: $\partial w_g/\partial\mathbf R$ on the same points | `build_atomic_grid_with_response` |

The cost of building the grid is the Becke weight, $O(N_{\rm atoms}^2)$ per
point: every cell product runs over every other atom. `BeckeTable`
precomputes $R_{AB}$ and $a_{AB}$ once per molecule and stops a cell product
$P_A$ once it is an exact zero. Its module documentation proves that the
early stop gives the same bits as `becke_weights_all`: no cell factor is
negative, so a product that reached $+0.0$ cannot turn into $-0.0$.

**Reproducibility.** The weight cut is part of the grid: gradients and
Hessians differentiate the energy on exactly the points the SCF used. The
Bragg–Slater radii of He, Ne and Ar are ferric's own (0.30, 0.45, 0.71 Å),
not PySCF's, because the cDFT populations of section @sec:cdft are defined on them.
A KS energy with those atoms therefore agrees with PySCF only to the grid
error.

## Options {#grids-options}

| Option | Values | Default |
|---|---|---|
| `grid_radial`, `grid_angular` | any $n_r$; a supported Lebedev order | 75, 110 |
| `grid_prune` | `nwchem-radial`, `nwchem`, `sgx`, `defgrid1`–`defgrid3`, `none` | `nwchem-radial` |
| `grid_weight_cut` | a non-negative number; 0 keeps every point | $10^{-14}$ |
| TDDFT with a meta-GGA | `response_default_grid` raises the grid to $150\times434$ unless set | |

The value tables are in `site/src/reference/input.md`.

## Numerical anchors {#grids-numerical-anchors}

| Quantity | Value | Source |
|---|---|---|
| Becke weight derivative vs FD | $<10^{-7}$ | `becke_weights_grad_finite_difference_ch`, `crates/ferric-dft/src/becke.rs` |
| KS gradient on custom and pruned grids vs FD on the same grid | $0.9$ to $1.4\times10^{-7}$ Ha/Bohr | `crates/ferric-scf/tests/ks_gradient_scf_grid.rs` (validation page, "KS gradient on a custom or angularly pruned XC grid") |
| the same gradient against a different (default) grid | misses by $2.8$ to $6.0\times10^{-4}$ Ha/Bohr | the same |

The last two rows are why the gradient uses the SCF's own grid. On that
grid the analytic gradient matches finite differences to about $10^{-7}$
Ha/Bohr. Against another grid the mismatch is the grid error, more than
$10^3$ times larger.

## Sources {#grids-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-dft/theory.md`, "Integration Grid", "Becke partition and its nuclear derivative" | the reference statement of the equations |
| `crates/ferric-dft/src/prune.rs` module documentation | the measurements behind the region table (why the outermost region rounds up) |
| Treutler and Ahlrichs, J. Chem. Phys. 102, 346 (1995) | the M4 map and $\xi$ |
| Becke, J. Chem. Phys. 88, 2547 (1988) | the partition |
| Lebedev and Laikov, Dokl. Math. 59, 477 (1999) | the angular rules |
