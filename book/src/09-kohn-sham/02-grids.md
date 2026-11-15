# 9.2 Radial and angular grids, the Becke partition, pruning

<!-- section: 9.2; status: ready; registry: ks-grid;
     theory: crates/ferric-dft/theory.md#"Integration Grid" "Becke partition and its nuclear derivative" -->

Every XC integral in ferric is a sum over an atom-centred grid: a radial grid
times a Lebedev sphere on each atom, the atomic grids glued together by
Becke's fuzzy partition of space. The default grid is pruned: the angular
order of a shell depends on its radius, and the light elements get fewer
radial shells.

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

## 9.2.1 The atomic grid

<!-- include: crates/ferric-dft/theory.md#"Integration Grid" -->

> **Verified equation 9.2** · the Treutler–Ahlrichs M4 radial rule
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
| $1 \pm x_k$, $\ln\frac{1-x_k}{2}$ | `one_plus`, `one_minus`, `log_term` |
| $w^{\rm rad}_k$ | `w_r` |

![The default grid of water's atoms: the radius of each Treutler–Ahlrichs shell and its Lebedev order after pruning, for O (66 shells) and H (50 shells), with the region boundaries in units of the Bragg–Slater radius. Water's pruned grid has 14,060 points against 24,750 flat (weight cut off), 43% fewer; the script checks its transcription of the grid against `ferric.dft_grid_point_count`. Script: `book/src/figures/09-kohn-sham/atomic_grid.py`.](../figures/09-kohn-sham/atomic_grid.svg)

## 9.2.2 The Becke partition

<!-- include: crates/ferric-dft/theory.md#"Becke partition and its nuclear derivative" -->

> **Verified equation 9.3** · Becke weights: partition of unity and their nuclear derivative
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

## 9.2.3 Building a molecular grid

| Step | What | Function |
|---|---|---|
| 1 | per atom: $n_r$ shells (fewer for H–Ne under `NwchemRadial`) | `radial_shells`, `treutler_ahlrichs_m4` |
| 2 | per shell: the Lebedev order of its region | `angular_orders_for_atom`, `region_orders` |
| 3 | per point: $w_g = w^{\rm rad}_kw^\Omega_jw^A(\mathbf r_g)$ | `BeckeTable::weight` (bit-identical to `becke_weights_all`) |
| 4 | drop $\lvert w_g\rvert$ below `weight_cut` | `build_atomic_grid` |
| 5 | for gradients and Hessians: $\partial w_g/\partial\mathbf R$ on the same points | `build_atomic_grid_with_response` |

The cost of building the grid is the Becke weight, $O(N_{\rm atoms}^2)$ per
point. `BeckeTable` precomputes $R_{AB}$ and $a_{AB}$ once per molecule and stops a
cell product $P_A$ once it is an exact zero; its module documentation proves
that the early stop gives the same bits as `becke_weights_all`.

**Reproducibility.** The weight cut is part of the grid: gradients and
Hessians differentiate the energy on exactly the points the SCF used. The
Bragg–Slater radii of He, Ne and Ar are ferric's own (0.30, 0.45, 0.71 Å),
not PySCF's, because the cDFT populations of section 9.7 are defined on them;
a KS energy with those atoms therefore agrees with PySCF only to the grid
error.

## 9.2.4 Options

| Option | Values | Default |
|---|---|---|
| `grid_radial`, `grid_angular` | any $n_r$; a supported Lebedev order | 75, 110 |
| `grid_prune` | `nwchem-radial`, `nwchem`, `sgx`, `defgrid1`–`defgrid3`, `none` | `nwchem-radial` |
| `grid_weight_cut` | a non-negative number; 0 keeps every point | $10^{-14}$ |
| TDDFT with a meta-GGA | `response_default_grid` raises the grid to $150\times434$ unless set | |

The value tables are in `site/src/reference/input.md`.

## 9.2.5 Numerical anchors

| Quantity | Value | Source |
|---|---|---|
| Becke weight derivative vs FD | $<10^{-7}$ | `becke_weights_grad_finite_difference_ch`, `crates/ferric-dft/src/becke.rs` |
| KS gradient on custom and pruned grids vs FD on the same grid | $0.9$ to $1.4\times10^{-7}$ Ha/Bohr | `crates/ferric-scf/tests/ks_gradient_scf_grid.rs` (validation page, "KS gradient on a custom or angularly pruned XC grid") |
| the same gradient against a different (default) grid | misses by $2.8$ to $6.0\times10^{-4}$ Ha/Bohr | the same |

## 9.2.6 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-dft/theory.md`, "Integration Grid", "Becke partition and its nuclear derivative" | the equations |
| `crates/ferric-dft/src/prune.rs` module documentation | the measurements behind the region table (why the outermost region rounds up) |
| Treutler and Ahlrichs, J. Chem. Phys. 102, 346 (1995) | the M4 map and $\xi$ |
| Becke, J. Chem. Phys. 88, 2547 (1988) | the partition |
| Lebedev and Laikov, Dokl. Math. 59, 477 (1999) | the angular rules |
