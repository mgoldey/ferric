# Density Functional Theory (DFT)

This module implements Kohn-Sham DFT with numerical integration on an
atom-centred grid. Each section below states the equations in the code's own
conventions, names the Rust function that implements them, the sympy script
that proves the identity the code relies on, and the Rust/Python test that
checks it numerically. Every script exits nonzero on a mismatch and carries
negative controls (a mutated formula that must fail):

| Script | Proves |
|---|---|
| `scripts/dft_becke_weight_grad_sympy.py` | Becke weight derivatives, translational invariance |
| `scripts/dft_vxc_matrix_elements_sympy.py` | LDA/GGA/meta-GGA $V_{xc}$ is $\partial E_{xc}/\partial D$; integration by parts |
| `scripts/dft_fxc_kernel_sympy.py` | GGA $f_{xc}$ response is the linearization of $V_{xc}$ |
| `scripts/dft_vv10_potential_sympy.py` | VV10 per-point potentials are $\partial E_{nl}/\partial(\rho,\sigma)$ |
| `scripts/dft_mgga_tau_gradient_sympy.py` | GGA/meta-GGA nuclear-gradient AO-derivative and grid-response terms |

Run any of them with `python3 scripts/<name>.py` (needs `sympy`).

The second nuclear derivatives of $E_{xc}$ (the analytic KS Hessian's XC
skeleton, first-order Fock matrix and CPHF kernel, fixed and moving grid) are
in `crates/ferric-scf/theory.md` ("Kohn-Sham XC Hessian terms"), proved by
`scripts/ks_fixed_grid_hessian_sympy.py`.

## Kohn-Sham Equations
The DFT energy is:
$$E_{DFT} = T_s + V_{ext} + J + E_{xc}$$
where $E_{xc}$ is the exchange-correlation functional.

## Numerical Integration
The exchange-correlation energy is a quadrature over grid points
$\{\mathbf{r}_g, w_g\}$:
$$E_{xc} = \int \rho\,\epsilon_{xc}(\rho, \sigma, \tau)\,d\mathbf{r} \approx \sum_g w_g\,\rho_g\,\epsilon_{xc}(\rho_g, \sigma_g, \tau_g)$$
with, for a closed shell with total density matrix $D$ and real AOs $\chi_\mu$,
$$\rho = \sum_{\mu\nu} D_{\mu\nu}\chi_\mu\chi_\nu,\qquad
\sigma = |\nabla\rho|^2,\qquad
\tau = \tfrac12\sum_b\sum_{\mu\nu} D_{\mu\nu}\,\partial_b\chi_\mu\,\partial_b\chi_\nu .$$
$\tau = \tfrac12\sum_i|\nabla\phi_i|^2$ is built from $D$ directly
(`eval_tau_closed`, `src/density_on_grid.rs`; checked against the explicit
MO sum by `density_on_grid::tests::tau_from_density_matrix_matches_explicit_mo_sum`).
No density Laplacian is used: meta-GGAs are evaluated $\tau$-only.

Spin-polarized quantities follow libxc's interleaved packing:
$\rho = (\rho_\alpha, \rho_\beta)$,
$\sigma = (\sigma_{\alpha\alpha}, \sigma_{\alpha\beta}, \sigma_{\beta\beta})$ with
$\sigma_{\alpha\beta} = \nabla\rho_\alpha\cdot\nabla\rho_\beta$,
$\tau = (\tau_\alpha, \tau_\beta)$, $\tau_\sigma = \tfrac12\sum_b\sum D^\sigma_{\mu\nu}\partial_b\chi_\mu\partial_b\chi_\nu$.
libxc returns $v_\rho = \partial(\rho\epsilon)/\partial\rho$,
$v_\sigma = \partial(\rho\epsilon)/\partial\sigma$ (not $2\times$),
$v_\tau = \partial(\rho\epsilon)/\partial\tau$, per point.

## Functionals
Functionals are evaluated through `libxc`: LDA $\epsilon(\rho)$, GGA
$\epsilon(\rho,\sigma)$, hybrids and range-separated hybrids (the exact-exchange
fraction is built by the SCF), $\tau$-only meta-GGAs (SCAN, r2SCAN, TPSS) and
VV10 nonlocal correlation.

## Integration Grid
Each atom carries a Treutler-Ahlrichs radial grid times a Lebedev angular grid; atomic grids are combined with Becke partition weights $w_A(\mathbf{r})$, so the weight of a point on atom $A$ is $w_r\,w_\Omega\,w_A(\mathbf{r})$.

**Default grid.** A Kohn-Sham calculation with no grid set (`RhfConfig::dft_grid = None`, CLI `[dft]` grid keys absent, Python grid kwargs `None`) runs on `AtomicGridConfig::production()`: 75 radial $\times$ 110 angular points per atom, pruned with `NwchemRadial` (NWChem angular regions; 50 radial shells for H/He and 66 for Li-Ne), with points of final weight below $10^{-14}$ dropped (`DEFAULT_WEIGHT_CUT`, ORCA's `WeightCut`). Every KS path (SCF, gradients, Hessians, response, TDDFT) resolves `None` to this grid, so they integrate on the same points.

`AtomicGridConfig::default()` is the flat, uncut 75 $\times$ 110 grid. It is the quadrature for analysis code (Hirshfeld volumes, partitions, property grids) and for tests that pin flat-grid numbers; it is not the KS default. The VV10 nonlocal grid is a separate flat 50 $\times$ 50 grid (`default_nlc()`); pruning has no region table at angular order 50, so it is never pruned.

**Config keys.** CLI `[dft]`: `grid_radial`, `grid_angular`, `grid_prune`, `grid_weight_cut`. Python (`run_dft`, `run_ksdft`, ...): the same names as keyword arguments. Setting only a size keeps the default scheme and cutoff at that size; `grid_prune = "none"` with `grid_weight_cut = 0` is the flat grid. See `site/src/reference/input.md` for the value tables.

**Gradients.** The analytic gradient differentiates the energy on the grid the SCF ran on, including pruning and the weight cut, by dropping exactly the same points.

## Becke partition and its nuclear derivative

`becke_weights_and_grad` (`src/becke.rs`). For atoms $A, B$ with
$r_X = |\mathbf{r}-\mathbf{R}_X|$, $R_{AB} = |\mathbf{R}_A-\mathbf{R}_B|$:
$$\mu_{AB} = \frac{r_A-r_B}{R_{AB}},\qquad
\nu_{AB} = \mu_{AB} + a_{AB}(1-\mu_{AB}^2),\qquad
s_{AB} = \tfrac12\bigl(1 - f^{(3)}(\nu_{AB})\bigr),\quad f(x)=\tfrac12(3x-x^3),$$
$$a_{AB} = \operatorname{clip}_{[-1/2,\,1/2]}\frac{u}{u^2-1},\quad u=\frac{\chi-1}{\chi+1},\ \chi = R^{BS}_A/R^{BS}_B,\qquad
P_A = \prod_{B\ne A}s_{AB},\qquad w^A = P_A/T,\ T=\sum_C P_C .$$
Derivatives the loop codes (`becke_weights_and_grad`):
$$\frac{\partial\mu_{AB}}{\partial R_A^\alpha} = -\frac{(\mathbf{r}-\mathbf{R}_A)^\alpha}{r_A R_{AB}} - \frac{\mu_{AB}}{R_{AB}^2}(\mathbf{R}_A-\mathbf{R}_B)^\alpha,\qquad
\frac{\partial\mu_{AB}}{\partial R_B^\alpha} = +\frac{(\mathbf{r}-\mathbf{R}_B)^\alpha}{r_B R_{AB}} + \frac{\mu_{AB}}{R_{AB}^2}(\mathbf{R}_A-\mathbf{R}_B)^\alpha,$$
$$\frac{d f^{(3)}}{d\nu} = f_2' f_1' f_0',\quad f_k' = \tfrac32(1-f_k^2),\qquad
\frac{\partial\nu}{\partial\mu} = 1-2a_{AB}\mu,\qquad
\frac{\partial w^A}{\partial R_C} = \frac{1}{T}\frac{\partial P_A}{\partial R_C} - \frac{P_A}{T^2}\sum_D\frac{\partial P_D}{\partial R_C}.$$
Every $\mu_{AB}$ is invariant under a rigid translation of nuclei and point,
so
$$\sum_C \frac{\partial w^A}{\partial \mathbf{R}_C}\Big|_{\mathbf{r}} + \nabla_{\mathbf{r}} w^A = 0 .$$
`build_atomic_grid_with_response` (`src/grid.rs`) uses this to form the
PySCF "weight1" derivative, in which each point rides its home atom:
$\partial w_g/\partial \mathbf{R}_b = w_r w_l[\partial w^{A}/\partial\mathbf{R}_b + \delta_{b,A}\nabla_{\mathbf{r}}w^{A}]$, so $\sum_b \partial w_g/\partial\mathbf{R}_b = 0$.

- Proof: `scripts/dft_becke_weight_grad_sympy.py` (each derivative symbolically;
  the assembled gradient of a 3-atom unequal-radius cluster vs sympy to
  $10^{-61}$; partition of unity; $\nabla_{\mathbf r}w = -\sum_C\partial w/\partial\mathbf R_C$).
- Tests: `becke::tests::becke_weights_grad_finite_difference_ch` (FD, $<10^{-7}$),
  `becke::tests::becke_weights_grad_sums_to_zero_h2`,
  `becke::tests::partition_weight_derivative_matches_fd` (neighbour-list kernel);
  end to end through the KS gradients below.

## Semilocal $V_{xc}$ matrix elements

`semilocal_vxc_closed_scratch` (`src/vxc.rs`). Treating $D_{\mu\nu}$ and
$D_{\nu\mu}$ as independent,
$$\frac{\partial E_{xc}}{\partial D_{\mu\nu}} = \sum_g w_g\Bigl[v_\rho\,\chi_\mu\chi_\nu
+ 2v_\sigma\,\nabla\rho\cdot(\chi_\mu\nabla\chi_\nu + \chi_\nu\nabla\chi_\mu)
+ \tfrac12 v_\tau \sum_b \partial_b\chi_\mu\,\partial_b\chi_\nu\Bigr].$$
The code builds the GGA piece as $M + M^\top$ with
$M_{\mu\nu} = \sum_g f_{b,g}\chi_\mu\partial_b\chi_\nu$, $f_b = 2 w v_\sigma \partial_b\rho$, the $\tau$ piece with
$\tfrac12 w v_\tau$, then $\tfrac12(V+V^\top)$. An LDA has $v_\sigma = v_\tau = 0$ (only the first
term is built), a GGA $v_\tau = 0$. Points with $\rho$ at or below the functional's
`DensityFloor` are dropped (`DensityFloor::for_xc`: $10^{-14}$ for LDA/GGA and their hybrids,
$10^{-10}$ when any component is a meta-GGA).

Spin-polarized (`semilocal_vxc_polarized_scratch`, `src/vxc.rs`):
$$\frac{\partial E_{xc}}{\partial D^\alpha_{\mu\nu}} = \sum_g w_g\Bigl[v_{\rho_\alpha}\chi_\mu\chi_\nu
+ \mathbf{G}^\alpha\cdot(\chi_\mu\nabla\chi_\nu+\chi_\nu\nabla\chi_\mu) + \tfrac12 v_{\tau_\alpha}\sum_b\partial_b\chi_\mu\partial_b\chi_\nu\Bigr],\qquad
\mathbf{G}^\alpha = 2v_{\sigma_{\alpha\alpha}}\nabla\rho_\alpha + v_{\sigma_{\alpha\beta}}\nabla\rho_\beta,$$
and $\alpha\leftrightarrow\beta$. The cross coefficient is 1 because
$\partial\sigma_{\alpha\beta}/\partial\nabla\rho_\alpha = \nabla\rho_\beta$.
Each spin block is floored on its own $\rho_\sigma$.

The matrix-element form and the functional-derivative potential
$v_{xc} = v_\rho - 2\nabla\cdot(v_\sigma\nabla\rho)$ differ by a pure divergence,
$$v_\rho\chi_\mu\chi_\nu + 2v_\sigma\nabla\rho\cdot\nabla(\chi_\mu\chi_\nu) - v_{xc}\chi_\mu\chi_\nu = \nabla\cdot\bigl(2v_\sigma\nabla\rho\,\chi_\mu\chi_\nu\bigr),$$
which integrates to zero for decaying AOs, so both define the same Fock matrix;
the code uses the left form (no second derivatives of $\rho$ needed).

- Proof: `scripts/dft_vxc_matrix_elements_sympy.py` (generic $e(\rho,\sigma,\tau)$
  and generic 7-variable polarized $e$; the divergence identity in 3D; a central
  difference of $E_{xc}(D)$ vs $\mathrm{Tr}[VX]$ for a concrete meta-GGA-like $e$ on a
  4-point, 3-AO model, closed and polarized, residual $\le 7\times10^{-17}$).
- Tests: `ferric-python/tests/test_xc_blocks.py::test_vxc_is_the_derivative_of_exc`
  (FD of $E_{xc}$ along a random symmetric $X$ vs $\mathrm{Tr}[VX]$, PBE / SCAN / ωB97X,
  closed and polarized, $10^{-9}$ relative),
  `test_polarized_equals_closed_at_equal_spin_densities`,
  `ferric-dft/tests/mgga_polarized_packing.rs` (packing, $\tau$-channel independence,
  vs PySCF `eval_xc`).

## $f_{xc}$ response kernels

`GgaFxcKernel::apply_with_ref` (`src/fxc.rs`) returns
$\delta V^\sigma$ for a perturbation $(\delta D^\alpha, \delta D^\beta)$. Because
$V^\sigma$ enters the AO matrix only through $v_{\rho_\sigma}$ and $\mathbf{G}^\sigma$,
and the back-projection is linear in them, $\delta V^\sigma$ uses the same
back-projection with
$$u^\sigma = \delta v_{\rho_\sigma},\qquad \mathbf{w}^\sigma = \delta\mathbf{G}^\sigma,$$
$$\delta v_{\rho_\alpha} = f_{\rho_\alpha\rho_\alpha}\delta\rho_\alpha + f_{\rho_\alpha\rho_\beta}\delta\rho_\beta + \sum_c f_{\rho_\alpha\sigma_c}\delta\sigma_c,\qquad
\delta v_{\sigma_c} = \sum_{\sigma'} f_{\sigma_c\rho_{\sigma'}}\delta\rho_{\sigma'} + \sum_{c'} f_{\sigma_c\sigma_{c'}}\delta\sigma_{c'},$$
$$\delta\sigma_{\alpha\alpha} = 2\nabla\rho_\alpha\cdot\nabla\delta\rho_\alpha,\quad
\delta\sigma_{\alpha\beta} = \nabla\rho_\alpha\cdot\nabla\delta\rho_\beta + \nabla\rho_\beta\cdot\nabla\delta\rho_\alpha,$$
$$\mathbf{w}^\alpha = 2\,\delta v_{\sigma_{\alpha\alpha}}\nabla\rho_\alpha + \delta v_{\sigma_{\alpha\beta}}\nabla\rho_\beta
+ 2 v_{\sigma_{\alpha\alpha}}\nabla\delta\rho_\alpha + v_{\sigma_{\alpha\beta}}\nabla\delta\rho_\beta .$$
libxc's second derivatives are read packed: `v2rho2` $=[\alpha\alpha,\alpha\beta,\beta\beta]$,
`v2rhosigma` $=[\alpha\cdot\alpha\alpha,\alpha\cdot\alpha\beta,\alpha\cdot\beta\beta,\beta\cdot\alpha\alpha,\beta\cdot\alpha\beta,\beta\cdot\beta\beta]$,
`v2sigma2` the upper triangle $[\alpha\alpha\cdot\alpha\alpha,\alpha\alpha\cdot\alpha\beta,\alpha\alpha\cdot\beta\beta,\alpha\beta\cdot\alpha\beta,\alpha\beta\cdot\beta\beta,\beta\beta\cdot\beta\beta]$.
The LDA kernel (`LdaFxcKernel::apply_with_ref`) is the
$\sigma$-free case $u^\alpha = f_{\alpha\alpha}\delta\rho_\alpha + f_{\alpha\beta}\delta\rho_\beta$, $\mathbf{w}=0$.
The $\tau$-only meta-GGA kernel (`MggaFxcKernel::apply_with_ref`) extends the
per-point Hessian to the 7 variables $(\rho_\alpha,\rho_\beta,\sigma_{\alpha\alpha},\sigma_{\alpha\beta},\sigma_{\beta\beta},\tau_\alpha,\tau_\beta)$
(`point_hessians`; libxc `v2rhotau` $=[\rho_\alpha\tau_\alpha,\rho_\alpha\tau_\beta,\rho_\beta\tau_\alpha,\rho_\beta\tau_\beta]$,
`v2sigmatau` $=[\sigma_{\alpha\alpha}\tau_\alpha,\sigma_{\alpha\alpha}\tau_\beta,\sigma_{\alpha\beta}\tau_\alpha,\dots,\sigma_{\beta\beta}\tau_\beta]$ and
`v2tau2` $=[\alpha\alpha,\alpha\beta,\beta\beta]$ fill the $\tau$ rows, `scatter_tau`). With $\tau_\sigma = \tfrac12\sum_b\sum_{\mu\nu} D^\sigma_{\mu\nu}\,\partial_b\chi_\mu\,\partial_b\chi_\nu$
linear in $D$,
$$\delta\tau_\sigma = \tfrac12\sum_b\sum_{\mu\nu}\delta D^\sigma_{\mu\nu}\,\partial_b\chi_\mu\,\partial_b\chi_\nu,\qquad
\delta v_{x_i} = \sum_j f_{x_i x_j}\,\delta x_j,$$
and the response adds one back-projected term to $u$ and $\mathbf w$ (`linearize_point`;
`MggaFxcKernel::backproject`):
$$\delta V^\sigma_{\mu\nu} = \sum_g w_g\Big[u^\sigma\chi_\mu\chi_\nu + \mathbf w^\sigma\cdot\nabla(\chi_\mu\chi_\nu)
+ \tfrac12\, t^\sigma\,\nabla\chi_\mu\cdot\nabla\chi_\nu\Big],\qquad t^\sigma = \delta v_{\tau_\sigma}.$$
$v_\tau$ itself does not appear (only $\delta v_\tau$), because $V^\tau$ is linear in $v_\tau$
and $\tau$ is linear in $D$. Every term is zeroed at points where the reference
$\rho_\sigma$ is at or below the functional's `DensityFloor`, as $V^\sigma$ is. None of these
kernels contains the VV10 response; that is `vv10_potential_tangent` (`src/vv10.rs`).

Consumers. `ferric_scf::rohf::FxcKernelStore` (`ferric-scf/src/rohf.rs`) dispatches
LDA / GGA / meta-GGA for the orbital-Hessian users: RKS and UKS Newton steps,
RKS/UKS internal stability, and the RKS/UKS/ROKS Z-vector
(`zvector_ks`). `lr_kernel::SingletKernel` (`src/lr_kernel.rs`) does the same for
the singlet $(ia|f_{\alpha\alpha}+f_{\alpha\beta}|jb)$ block of TDA / Casida TDDFT
(`ferric_tddft::run_tddft`, `ferric_gw::tddft::run_tda_dft`). Refused for meta-GGA:
ROKS Newton / augmented-Hessian and TRAH (DIIS is kept; not anchored) and the analytic nuclear Hessian (its $\tau$-dependent
$V_{xc}$ nuclear derivatives are not implemented).

- Proof: `scripts/dft_fxc_kernel_sympy.py` ($\delta\sigma$, $u$, $\mathbf w$ for a
  generic cubic $e$ whose first and second derivatives are independent symbols;
  packing-swap control; the meta-GGA `linearize_point` $(u,\mathbf w,t)$ with libxc-packed
  second derivatives vs a central difference of $(v_\rho,\mathbf G,v_\tau)$ for a concrete
  7-variable $e$, residual $6\times10^{-18}$).
- Tests: `fxc::tests::gga_fxc_matches_finite_difference_of_vxc` (OH doublet, all
  spin channels), `fxc::tests::gga_kernel_matches_lda_kernel_on_lda_functional`,
  `fxc::tests::mgga_fxc_matches_finite_difference_of_production_vxc_polarized` /
  `_closed`, `fxc::tests::mgga_kernel_on_gga_functional_equals_gga_kernel`,
  `ferric-scf/tests/uhf_newton_smoke.rs::uks_pbe_newton_engages_gga_fxc_and_matches_diis` (end to end: UKS/PBE Newton reaches the DIIS energy).
- Meta-GGA consumers (r2SCAN, 6-31G, matched (75,110) grid, references from
  `scripts/validation/gen_mgga_response.py` and `gen_tddft_refs.py`):
  `ferric-scf/tests/mgga_response_vs_pyscf.rs` (UKS NH2 and RKS water dense orbital-Hessian
  spectra and stability $\lambda_{\min}$ vs PySCF `gen_g_hop_uhf` / `gen_g_hop_rhf`; RKS/UKS
  Newton vs DIIS energy), `ferric-rpa/tests/mbd_scf_gradient{,_uks,_roks}.rs::*_r2scan`
  (Z-vector: MBD@rsSCS gradient vs FD of the full SCF pipeline),
  `ferric-tddft/tests/validation_tddft.rs::water_631g_r2scan` (TDA / Casida vs PySCF `get_ab`).

## VV10 nonlocal correlation

`vv10_internal_screened` (`src/vv10.rs`), closed shell, total density:
$$E_{nl} = \sum_i W_i\rho_i\Bigl[\beta + \tfrac12 f_i\Bigr],\qquad
f_i = \sum_p W_p\rho_p\,\Phi_{ip},\qquad
\Phi_{ip} = -\frac{3}{2\,g_i g_p (g_i+g_p)},\qquad g_x = \omega_0(\mathbf r_x) R_{ip}^2 + \kappa(\mathbf r_x),$$
$$\omega_0 = \sqrt{C\,(\sigma/\rho^2)^2 + \tfrac{4\pi}{3}\rho},\qquad
\kappa = b\,\tfrac{3\pi}{2}\Bigl(\frac{\rho}{9\pi}\Bigr)^{1/6},\qquad
\beta = \tfrac{1}{32}\bigl(3/b^2\bigr)^{3/4}.$$
The pair sum includes the self pair $p=i$ ($R=0$). With
$t_{ip} = W_p\rho_p/(g_i g_p g_t)$, $g_t = g_i+g_p$,
$u_i = \sum_p t_{ip}(1/g_i + 1/g_t)$ and $w_i = \sum_p t_{ip}(1/g_i+1/g_t)R_{ip}^2$, the
per-point potentials are
$$v_{\rho,i} = \beta + f_i + \tfrac32\Bigl(u_i\,\rho\frac{\partial\kappa}{\partial\rho} + w_i\,\rho\frac{\partial\omega_0}{\partial\rho}\Bigr)_i,\qquad
v_{\sigma,i} = \tfrac32\,w_i\,\Bigl(\rho\frac{\partial\omega_0}{\partial\sigma}\Bigr)_i,$$
so that $W_i v_{\rho,i} = \partial E_{nl}/\partial\rho_i$ and $W_i v_{\sigma,i} = \partial E_{nl}/\partial\sigma_i$.
The code's `dk_dr`, `dw0_dr`, `dw0_dg` are $\rho$ times the
partial derivatives: the $\rho_i$ prefactor of $E_{nl}$ is folded in.
$V_{nl}$ is assembled by `add_vv10_scratch` with the closed-shell
GGA form $w v_\rho\chi_\mu\chi_\nu + 2 w v_\sigma\nabla\rho\cdot(\chi_\mu\nabla\chi_\nu+\chi_\nu\nabla\chi_\mu)$. Points with $\rho < 10^{-8}$ are excluded from the pair sum.
$\rho\,\partial\omega_0/\partial\sigma = C\sigma/(\rho^3\omega_0)$ is kept at every active
point, including $\sigma \le 10^{-8}$: $\omega_0$ depends on $\sigma$ wherever $\rho$ is
active, so $V_{nl} = \partial E_{nl}/\partial D$.

- Proof: `scripts/dft_vv10_potential_sympy.py` (two-point grid with self pairs,
  exact; the constants; module-doc $\Phi$ form).
- Tests: `vv10::tests::truncated_potentials_are_the_derivative_of_the_truncated_energy`
  (per-point FD, $10^{-8}$), `vv10::tests::potential_tangent_is_the_derivative_of_production_vnl`,
  `vv10::tests::screen_matches_dense_exactly_in_the_trivial_limit`,
  `ferric-scf/tests/dft_gradient_wb97xv.rs` (ωB97X-V nuclear gradient vs FD).
- `vv10::tests::vsig_is_the_sigma_derivative_at_low_gradient_points` (FD of
  $E_{nl}$ along $\sigma$ at active low-gradient points, $10^{-5}$ relative) and
  `test_xc_blocks.py::test_vxc_is_the_derivative_of_exc`, which includes
  ωB97X-V: $\mathrm{Tr}[V_{nl}X]$ equals the FD derivative of $E_{nl}$.

## XC nuclear gradient: AO-derivative and grid-response terms

`xc_gradient_closed_mgga_from_density` (`src/gradient.rs`, batch body
`mgga_closed_batch`). The GGA path (`xc_gradient_closed_gga_from_density`) is the
$v_\tau = 0$ case and the LDA path (`xc_gradient_closed_lda_from_density`) the
$v_\sigma = v_\tau = 0$ case.
Translating nucleus $A$ moves the AOs centred on $A$:
$\partial\chi_\mu/\partial R_{A,\alpha} = -\partial_\alpha\chi_\mu$ for $\mu\in A$.
At fixed $D$ and fixed grid, with $m = D\chi$ and $(D\partial_b\chi)_\mu = \sum_\nu D_{\mu\nu}\partial_b\chi_\nu$,
$$\frac{\partial\tau}{\partial R_{A,\alpha}} = -\sum_b\sum_{\mu\in A}\partial^2_{\alpha b}\chi_\mu\,(D\partial_b\chi)_\mu .$$
The $\tfrac12$ in $\tau$ cancels against the $\mu\leftrightarrow\nu$ doubling.
`mgga_tau_partials` returns half of this sum and `scatter_partials` applies
$-2\times$, which reproduces it. The $\rho,\sigma$ partials (`gga_ao_partials`) with
$c_b = 2w v_\sigma\partial_b\rho$:
$$\frac{\partial E_{xc}}{\partial R_{A,\alpha}}\Big|_{\rm AO} = -2\sum_g\sum_{\mu\in A}\Bigl[w v_\rho\,\partial_\alpha\chi_\mu m_\mu
+ \sum_b c_b\bigl(\partial_\alpha\chi_\mu (D\partial_b\chi)_\mu + \partial^2_{\alpha b}\chi_\mu m_\mu\bigr)
+ \tfrac12 w v_\tau\sum_b\partial^2_{\alpha b}\chi_\mu (D\partial_b\chi)_\mu\Bigr].$$
Grid response: the weight term $\sum_g (\partial w_g/\partial\mathbf R_b)\,\rho_g\epsilon_g$
(weight1, above) plus the home-atom translation of the integrand,
$\sum_{g:\,{\rm home}=A} w_g[v_\rho\partial_\alpha\rho + v_\sigma\partial_\alpha\sigma + v_\tau\partial_\alpha\tau]$, with
$$\partial_\alpha(\partial_b\rho) = 2\Bigl[\sum_\mu\partial_\alpha\chi_\mu(D\partial_b\chi)_\mu + \sum_\mu m_\mu\,\partial^2_{\alpha b}\chi_\mu\Bigr]\ (\texttt{hess\_col}),\qquad
\partial_\alpha\tau = \sum_b\sum_\mu\partial^2_{\alpha b}\chi_\mu (D\partial_b\chi)_\mu\ (\texttt{tau\_spatial\_grad}).$$
The home-translation term is the correct correction because $\rho$, $\sigma$
and $\tau$ are translation invariant:
$\sum_C\partial q/\partial\mathbf R_C + \nabla_{\mathbf r}q = 0$. The UKS path
(`xc_gradient_uks_mgga_from_density`; LDA/GGA `xc_gradient_uks_from_density`) is the same per spin
with $D\to D^\sigma$, $\mathbf{G}^\sigma$ in place of $2v_\sigma\nabla\rho$.

- Proof: `scripts/dft_mgga_tau_gradient_sympy.py` (p- and s-Gaussians on two
  centres, exact rational point; the $\tfrac12$-kept and halved-spatial controls
  fail).
- Tests: `ferric-scf/tests/dft_gradient_mgga.rs` (SCAN/r2SCAN/TPSS vs FD of the
  SCF energy; SCAN/r2SCAN vs PySCF `grid_response=True`; UKS NH$_2$, OH),
  `ferric-scf/tests/dft_gradient_gga.rs` and `dft_gradient_lda.rs` (vs FD),
  `ferric-python/tests/test_xc_blocks.py::test_xc_grad_matches_finite_difference_at_fixed_density`
  (PBE, SCAN; fixed-$D$ FD, closed and polarized).

## Exact exchange in hybrid and range-separated hybrid functionals

A hybrid functional adds a fraction of exact (Hartree-Fock) exchange to a
semilocal functional. A range-separated hybrid uses a different fraction at
short and long electron-electron distance. libxc supplies the semilocal part
and the fractions; the SCF builds the exact exchange itself. The proof script
is `scripts/dft_hybrid_rsh_sympy.py`.

**Operator split.** With $\mathrm{erf}+\mathrm{erfc}=1$ the Coulomb operator
splits at the range-separation parameter $\omega$ (Bohr$^{-1}$):
$$\frac1r = \frac{\mathrm{erfc}(\omega r)}{r} + \frac{\mathrm{erf}(\omega r)}{r}.$$
The CAM form of Yanai, Tew and Handy writes the exact-exchange kernel as
$[a + b\,\mathrm{erf}(\omega r)]/r$ and gives the rest,
$[1 - a - b\,\mathrm{erf}(\omega r)]/r$, to the semilocal exchange; the two
sum to $1/r$. ferric writes every exact-exchange kernel as
$$v_x^{\rm HF}(r) = c_{\rm sr}\,\frac{\mathrm{erfc}(\omega r)}{r} + c_{\rm lr}\,\frac{\mathrm{erf}(\omega r)}{r},$$
so $c_{\rm sr}$ is the exact-exchange fraction as $r\to0$ and $c_{\rm lr}$ the
fraction as $r\to\infty$. In the Yanai notation $c_{\rm sr} = a$ and
$c_{\rm lr} = a + b$.

**libxc convention as ferric reads it.** libxc's `xc_hyb_cam_coef` returns
$(\omega, \alpha, \beta)$ for the kernel $\alpha/r + \beta\,\mathrm{erfc}(\omega r)/r$:
`cam_alpha` multiplies the full-range operator and `cam_beta` the short-range
one. Matching the two forms term by term,
$$c_{\rm sr} = \alpha + \beta,\qquad c_{\rm lr} = \alpha$$
(`XcFunctional::cam_coefficients`, stored in `CamCoeffs { omega, c_sr, c_lr }`).
A functional with $\omega = 0$ has no `CamCoeffs`; its global fraction
$a$ is `xc_hyb_exx_coef` (`XcFunctional::exact_exchange_mix`, stored in
`XcDef::b3lyp_mix`). PySCF's `dft.libxc.rsh_coeff` returns
$(\omega, c_{\rm lr}, c_{\rm sr} - c_{\rm lr})$, the same split. The sign
of `cam_beta` tells the two families apart:

| Functional | libxc $(\alpha, \beta, \omega)$ | $c_{\rm sr}$ | $c_{\rm lr}$ | Kind |
|---|---|---|---|---|
| ωB97X | $(1, -0.842294, 0.3)$ | 0.157706 | 1 | long-range corrected |
| ωB97X-V | $(1, -0.833, 0.3)$ | 0.167 | 1 | long-range corrected |
| CAM-B3LYP | $(0.65, -0.46, 0.33)$ | 0.19 | 0.65 | long-range corrected |
| LC-ωPBE | $(1, -1, 0.4)$ | 0 | 1 | long-range only |
| HSE06 | $(0, 0.25, 0.11)$ | 0.25 | 0 | screened (short-range only) |
| B3LYP, PBE0 | $\omega = 0$, $a = 0.2$, $0.25$ | $a$ | $a$ | global hybrid |

$\beta < 0$ means more exact exchange at long range, $\beta > 0$ more at
short range. libxc's semilocal energy and potential for these functionals
already carry the complementary semilocal exchange (scaled, and for an RSH
attenuated by the matching short- or long-range factor), so ferric adds only
the exact exchange.

**Mixing coefficients.** `k_mix_from_xc_def` and `KsXc::k_mix` /
`KsXcUks::k_mix` (`src/ks.rs`) turn the definition into
`KMix { sr, lr, omega }` (`src/xc_trait.rs`):
$(c_{\rm sr}, c_{\rm lr}, \omega)$ for an RSH, $(a, a, 0)$ for a global hybrid,
$(0, 0, 0)$ for a pure functional, and $(1, 1, 0)$ for HF (no functional).
A meta-GGA built from several libxc components (`metagga_def_from_handles`)
sums the components' `xc_hyb_exx_coef` into one global $a$ (M06-2X: 0.54
from the exchange component, 0 from correlation), takes `CamCoeffs` from at
most one component, and refuses a definition that has both.
`xc_def_from_name_nspin_omega` overrides $\omega$ in every component's
`_omega` parameter and in `CamCoeffs.omega` together, leaving $\alpha$ and
$\beta$ unchanged.

**Exchange matrices.** For an AO density $D$ and kernel $v$,
$K[v](D)_{\mu\nu} = \sum_{\lambda\sigma}(\mu\lambda|v|\nu\sigma)D_{\lambda\sigma}$
and $J(D)_{\mu\nu} = \sum_{\lambda\sigma}(\mu\nu|\lambda\sigma)D_{\lambda\sigma}$.
The exchange matrix the SCF subtracts is
$$K_{\rm tot}(D) = c_{\rm sr}\,K[\mathrm{erfc}](D) + c_{\rm lr}\,K[\mathrm{erf}](D)$$
(`fock_assembly::subtract_rsh_exchange`). For $\omega > 0$ the two matrices
come from two density-fitting builders over the same auxiliary basis
(`config.df_k_aux`, default `def2-universal-jkfit`), one per operator
(`fock_assembly::build_rsh_dfk_pair`, `Operator::erfc(ω)` and
`Operator::erf(ω)`). For $\omega = 0$, $K_{\rm tot} = a\,K[1/r]$ from whichever
K builder the run selected.

**Closed shell** (`rhf.rs` `solve_rhf`, $D$ the total density,
$D = 2C_{\rm occ}C_{\rm occ}^{\mathsf T}$):
$$E_x^{\rm HF} = -\tfrac14\,\mathrm{Tr}[D\,K_{\rm tot}(D)],\qquad
F = h + J(D) - \tfrac12 K_{\rm tot}(D) + V_{xc}.$$
$K_{\rm tot}$ is linear in $D$ and $(\mu\lambda|v|\nu\sigma)$ is symmetric
under $(\mu\nu)\leftrightarrow(\lambda\sigma)$, so
$\mathrm d E_x^{\rm HF}/\mathrm dD = -\tfrac12 K_{\rm tot}(D)$: the exchange
part of $F$ is the derivative of the exchange energy.

**Open shell** (`uhf.rs` `solve_uhf`, `rohf.rs` `solve_rohf`, per-spin
densities $D^\sigma = C^\sigma_{\rm occ}C^{\sigma\,\mathsf T}_{\rm occ}$):
$$E_x^{\rm HF} = -\tfrac12\sum_\sigma \mathrm{Tr}[D^\sigma K_{\rm tot}(D^\sigma)],\qquad
F_\sigma = h + J(D^\alpha + D^\beta) - K_{\rm tot}(D^\sigma) + V_{xc}^\sigma.$$
ROHF assembles the same $F_\alpha$, $F_\beta$ and then couples them into its
Roothaan operator. With $D^\alpha = D^\beta = D/2$ both the energy and
$F_\sigma$ equal the closed-shell ones.

**Total energy.** The SCF evaluates the one- and two-electron energy from the
Fock matrix before $V_{xc}$ is added, and adds $E_{xc}$ from the grid:
$$E = \tfrac12\mathrm{Tr}[D(h + F_{\rm no\,xc})] + E_{xc} + E_{nn}
= \mathrm{Tr}[Dh] + \tfrac12\mathrm{Tr}[DJ] - \tfrac14\mathrm{Tr}[D K_{\rm tot}] + E_{xc} + E_{nn},$$
with $F_{\rm no\,xc} = h + J - \tfrac12 K_{\rm tot}$ (closed shell; open
shell $\tfrac12\sum_\sigma\mathrm{Tr}[D^\sigma(h + F_{\sigma,\rm no\,xc})]$).
The $\tfrac12$ appears once, on the trace; $E_{xc}$ is not
$\mathrm{Tr}[D V_{xc}]$, which is why $V_{xc}$ is added to $F$ after the trace.

**Proof** (`scripts/dft_hybrid_rsh_sympy.py`): symbolically, the erf/erfc
and CAM operator splits, the libxc-to-ferric map and the Yanai map, and
$\mathrm dE_x/\mathrm dD = F_x$ for the closed- and open-shell forms with
generic 8-fold-symmetric integrals for both kernels, plus their closed-shell
equality. Numerically, on four s-type Gaussians: the erf integrals in closed
form against quadrature of
$1/r = (2/\sqrt\pi)\int_0^\infty e^{-t^2r^2}\,\mathrm dt$ split at
$t = \omega$, the limits $\omega\to\infty$ (Coulomb) and $\omega\to0$ (zero),
$K_{\rm tot}$ against a K built on the CAM kernel directly, and central
differences of the total electronic energy against $\mathrm{Tr}[F\,\delta D]$
(RHF RSH, global hybrid, HF; UHF RSH). The libxc values in the table map to
the published short- and long-range fractions. With
`DERIV_CROSSCHECK=1` (PySCF and ferric importable) it also checks PySCF's `rsh_coeff`, ferric's
`XcFunctional.exchange_mix`, and ferric's `AoIntegrals.veff` energy against
its potential by central differences (ωB97X-V, CAM-B3LYP, HSE06 on
H$_2$/STO-3G).

## Follow-ups (identities with no direct Rust test)

- $\sum_b \partial w_g/\partial\mathbf R_b = 0$ for the assembled weight1 rows
  (`build_atomic_grid_with_response`); covered only end to end through the KS gradients.
