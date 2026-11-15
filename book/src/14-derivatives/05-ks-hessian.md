# 14.5 Kohn–Sham Hessian on a fixed and a moving grid

<!-- section: 14.5; status: ready; registry: ks-hessian;
     theory: crates/ferric-scf/theory.md#"Kohn-Sham XC Hessian terms (fixed and moving grid)", "Hartree-Fock nuclear Hessian" -->

The Kohn–Sham Hessian is the Hartree–Fock Hessian of section 14.4 with the
exact-exchange terms scaled by the hybrid fraction, plus the XC terms: a
skeleton at fixed density, a first-order XC Fock matrix, and the
second-order XC kernel $f_{xc}$ in the CPHF operator. Which energy is being
differentiated decides what the XC skeleton is. On a grid fixed in space the
points and weights do not move. On the production grid the points ride their
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
`grid_response=False`). It is not what the production SCF minimizes, so it is
a library entry that `hessian = "auto"` never selects. The moving-grid Hessian
differentiates the energy the production KS SCF and the KS gradient of
section 14.3 differentiate.

## 14.5.1 The XC terms

<!-- include: crates/ferric-scf/theory.md#"Kohn-Sham XC Hessian terms (fixed and moving grid)" -->

> **Verified equation 14.11** · the fixed-grid skeleton, first-order Fock matrix and kernel
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

> **Verified equation 14.12** · the moving-grid skeleton and first-order Fock matrix
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

> **Verified equation 14.13** · the spin-polarized algebra
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

> **Verified equation 14.14** · the Becke weight second derivative
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
> | Proof | `scripts/dft_becke_weight_hessian_sympy.py`, claims W1 and W2. The first derivative is `scripts/dft_becke_weight_grad_sympy.py`, which is section 14.3's equation 14.8 |
> | Negative control | the runner drops the $2\mu R_xR_y/R^2$ term (claim W1 fails) and the $2P_hT'T'^{\mathsf T}/T^3$ term (claim W2 fails). The script also requires the quantities it compares to be of order one, because a point deep inside one cell makes every derivative zero to 40 digits and every formula pass |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-dft/src/becke_hessian.rs::becke_home_weight_hess` |
> | Code test | `partition_weight_derivative_matches_fd` in `crates/ferric-dft/src/becke.rs` |
>
> <!-- verified: id=ks.hess.becke-second-derivative; script=scripts/dft_becke_weight_hessian_sympy.py; claim=W2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-dft/src/becke_hessian.rs::becke_home_weight_hess; rust_test=ferric-dft/src/becke.rs::partition_weight_derivative_matches_fd; registry=ks-hessian -->

The collinear three-nucleus geometry of claim W2 proves the assembly, which is
dimension-free. The three-dimensional pair derivatives are claim W1. The pair
size adjustment $a_{AB}$ is held constant, as in the code.

The exactness anchor of the moving-grid terms is a single atom: the point
rides its only atom, the weight is identically 1, and the moving-grid XC
skeleton and first-order Fock matrix are exactly zero, where the fixed-grid
ones are not. That anchor is blind to the weight terms, which is why the
frozen-density finite differences of the validation page carry the weight
derivatives.

## 14.5.2 The algorithm

The Coulomb, exchange, overlap and one-electron terms and the CPHF solver are
those of section 14.4. The XC pieces are added at three places.

| Step | XC piece | Function |
|---|---|---|
| 1 | skeleton $\sum_g\rho_x\delta F[y]+\mathbf g_x\!\cdot\!\delta\mathbf G[y]+F\rho_{xy}+\mathbf G\!\cdot\!\mathbf g_{xy}$ (plus the weight terms on the moving grid) | `FixedGridKs::skeleton_hessian`, `skeleton_hessian_moving` |
| 2 | first-order XC Fock matrix $\partial V_{xc}/\partial x$ at frozen $D$, added to $F^y$ | `FixedGridKs::first_order_fock`, `first_order_fock_moving` |
| 3 | kernel $V^{(1)}[D_1]$ in every CPHF operator application | `ferric-scf/src/ks_fixed_grid.rs::XcResponse` |
| 4 | moving grid: per home atom, $W'$ and $W''$ contracted point by point, never stored | `ferric-dft/src/becke_hessian.rs::becke_home_weight_hess` |
| 5 | moving grid: home-atom rows and columns by translation invariance, $H^{(h)} = T\,G_h\,T^{\mathsf T}$ | `ferric-scf/src/ks_fixed_grid.rs::translation_fill` |

The Becke weight second derivative costs $O(n^3)$ per point for $n$ atoms and
the translation fill is a dense $(3N)^3$ product per batch, so the cost ratio
against the fixed grid grows with $N$. The validation page records 1.07 to 1.10
on three atoms and no measurement above that.

## 14.5.3 Reproducibility and refusals

- The moving-grid route needs a grid with home atoms. A grid built from
  explicit points is refused (`a_grid_without_home_atoms_is_refused` in
  `crates/ferric-scf/tests/ks_moving_grid_hessian.rs`).
- `hessian = "auto"` takes the analytic KS Hessian wherever
  `ferric-scf/src/hessian/ks.rs::rks_moving_grid_hessian_available` accepts the
  system, and finite differences otherwise. `"analytic"` reaches neither KS
  Hessian.
- A failed UKS response solve under `"auto"` runs finite differences with a
  warning. The recorded case is the OH $^2\Pi$ exact $\pi$-hole zero mode.

## 14.5.4 Options that change the equation

| Option | Effect |
|---|---|
| LDA, GGA, global hybrids | covered. $f_\sigma = 0$ for LDA |
| meta-GGA | refused before the SCF: no $\tau$ terms appear above |
| VV10, range-separated hybrids, ROKS | finite differences of the analytic gradient |
| RI-J, RI-K | the RHF fitted Hessians of section 14.4 supply the Coulomb and exchange parts |
| ECP, external potential | refused on the KS routes |
| unpruned grid | the default pruned grid is refused by the moving-grid RKS Hessian where its pruning is not supported; see the availability probe for the current rule |

## 14.5.5 Numerical anchors

| Check | Reference | Agreement | Test |
|---|---|---|---|
| Moving-grid RKS Hessian (LDA, GGA, global hybrids; exact or RI-J, RI-K): distorted H2O / STO-3G, 6-31G, cc-pVDZ, 75x110 grid | central FD (Richardson) of the production gradient with a production SCF at every displaced geometry | 7.9e-11 (LDA) and 4.8e-11 (PBE) at STO-3G; 1.7e-10 (LDA / 6-31G, default grid); 1.05e-9 (PBE / 6-31G); 2.6e-10 (LDA / cc-pVDZ); harmonic frequencies at most 5.0e-7 cm⁻¹ against the symmetrized FD Hessian | `crates/ferric-scf/tests/validation_ks_moving_grid_hessian.rs`, `crates/ferric-scf/tests/ks_moving_grid_hessian.rs` |
| the same system with the fixed-grid Hessian | the same FD | misses by 1.1e-3 Ha/Bohr² and 3.0 cm⁻¹ | same |
| Fixed-grid RKS Hessian (library only): distorted H2O / STO-3G, 6-31G, cc-pVDZ | FD of the fixed-grid gradient; PySCF `dft.RKS` Hessian with `grid_response=False` on ferric's own grid | vs FD of the fixed-grid gradient 5.7e-11 (STO-3G), 1.9e-8 (6-31G), 8.1e-9 (cc-pVDZ); vs PySCF 4.1e-8 (6-31G) and 9.3e-7 (cc-pVDZ) | `crates/ferric-scf/tests/validation_ks_fixed_grid_hessian.rs` |
| UKS moving-grid Hessian: NH2, CH3 doublets, O2 triplet / STO-3G, 6-31G; LDA, PBE, B3LYP | Richardson FD of the production UKS gradient | at most 8.5e-10 Ha/Bohr², bar 2e-9; singlet UKS equals RKS to 2.7e-12; every spin-polarized mutant misses by at least 6.6e6 bars | `crates/ferric-scf/tests/uks_moving_grid_hessian.rs`, `crates/ferric-scf/tests/uks_xc_pieces.rs` |

<!-- figures: site/src/reference/validation.md "Analytic closed-shell KS Hessian on the MOVING (production) grid"; site/src/reference/validation.md "Analytic closed-shell KS Hessian (LDA, GGA, and global hybrids of both)"; site/src/reference/validation.md "Open-shell KS frequencies" -->

On PySCF's analytic cc-pVDZ Hessian: about 9.2e-7 of it is not the derivative of
PySCF's own gradient, and that is the whole 9.3e-7 by which the fixed-grid
Hessian differs from it. The cause on the PySCF side is not isolated, so the
fixed-grid agreement is stated at the FD level.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/theory.md`, "Kohn-Sham XC Hessian terms (fixed and moving grid)" | equations, the test table |
| `scripts/ks_fixed_grid_hessian_sympy.py`, `scripts/uks_xc_hessian_sympy.py` | verified equations 14.11 to 14.13 |
| `scripts/dft_becke_weight_hessian_sympy.py`, claims W1 and W2; `crates/ferric-dft/src/becke_hessian.rs` module documentation | verified equation 14.14 |
| `site/src/reference/validation.md` | the anchor rows |
