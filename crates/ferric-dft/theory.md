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
(`eval_tau_closed`, `src/density_on_grid.rs:266`; checked against the explicit
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

`becke_weights_and_grad` (`src/becke.rs:365`). For atoms $A, B$ with
$r_X = |\mathbf{r}-\mathbf{R}_X|$, $R_{AB} = |\mathbf{R}_A-\mathbf{R}_B|$:
$$\mu_{AB} = \frac{r_A-r_B}{R_{AB}},\qquad
\nu_{AB} = \mu_{AB} + a_{AB}(1-\mu_{AB}^2),\qquad
s_{AB} = \tfrac12\bigl(1 - f^{(3)}(\nu_{AB})\bigr),\quad f(x)=\tfrac12(3x-x^3),$$
$$a_{AB} = \operatorname{clip}_{[-1/2,\,1/2]}\frac{u}{u^2-1},\quad u=\frac{\chi-1}{\chi+1},\ \chi = R^{BS}_A/R^{BS}_B,\qquad
P_A = \prod_{B\ne A}s_{AB},\qquad w^A = P_A/T,\ T=\sum_C P_C .$$
Derivatives the loop codes (`becke.rs:422-434`):
$$\frac{\partial\mu_{AB}}{\partial R_A^\alpha} = -\frac{(\mathbf{r}-\mathbf{R}_A)^\alpha}{r_A R_{AB}} - \frac{\mu_{AB}}{R_{AB}^2}(\mathbf{R}_A-\mathbf{R}_B)^\alpha,\qquad
\frac{\partial\mu_{AB}}{\partial R_B^\alpha} = +\frac{(\mathbf{r}-\mathbf{R}_B)^\alpha}{r_B R_{AB}} + \frac{\mu_{AB}}{R_{AB}^2}(\mathbf{R}_A-\mathbf{R}_B)^\alpha,$$
$$\frac{d f^{(3)}}{d\nu} = f_2' f_1' f_0',\quad f_k' = \tfrac32(1-f_k^2),\qquad
\frac{\partial\nu}{\partial\mu} = 1-2a_{AB}\mu,\qquad
\frac{\partial w^A}{\partial R_C} = \frac{1}{T}\frac{\partial P_A}{\partial R_C} - \frac{P_A}{T^2}\sum_D\frac{\partial P_D}{\partial R_C}.$$
Every $\mu_{AB}$ is invariant under a rigid translation of nuclei and point,
so
$$\sum_C \frac{\partial w^A}{\partial \mathbf{R}_C}\Big|_{\mathbf{r}} + \nabla_{\mathbf{r}} w^A = 0 .$$
`build_atomic_grid_with_response` (`src/grid.rs:673`) uses this to form the
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

`semilocal_vxc_closed_scratch` (`src/vxc.rs:262`). Treating $D_{\mu\nu}$ and
$D_{\nu\mu}$ as independent,
$$\frac{\partial E_{xc}}{\partial D_{\mu\nu}} = \sum_g w_g\Bigl[v_\rho\,\chi_\mu\chi_\nu
+ 2v_\sigma\,\nabla\rho\cdot(\chi_\mu\nabla\chi_\nu + \chi_\nu\nabla\chi_\mu)
+ \tfrac12 v_\tau \sum_b \partial_b\chi_\mu\,\partial_b\chi_\nu\Bigr].$$
The code builds the GGA piece as $M + M^\top$ with
$M_{\mu\nu} = \sum_g f_{b,g}\chi_\mu\partial_b\chi_\nu$, $f_b = 2 w v_\sigma \partial_b\rho$
(`vxc.rs:338`), the $\tau$ piece with $\tfrac12 w v_\tau$ (`vxc.rs:370`), then
$\tfrac12(V+V^\top)$. Points with $\rho \le 10^{-10}$ (`DENSITY_FLOOR`) are dropped.

Spin-polarized (`semilocal_vxc_polarized_scratch`, `src/vxc.rs:542`):
$$\frac{\partial E_{xc}}{\partial D^\alpha_{\mu\nu}} = \sum_g w_g\Bigl[v_{\rho_\alpha}\chi_\mu\chi_\nu
+ \mathbf{G}^\alpha\cdot(\chi_\mu\nabla\chi_\nu+\chi_\nu\nabla\chi_\mu) + \tfrac12 v_{\tau_\alpha}\sum_b\partial_b\chi_\mu\partial_b\chi_\nu\Bigr],\qquad
\mathbf{G}^\alpha = 2v_{\sigma_{\alpha\alpha}}\nabla\rho_\alpha + v_{\sigma_{\alpha\beta}}\nabla\rho_\beta,$$
and $\alpha\leftrightarrow\beta$. The cross coefficient is 1 because
$\partial\sigma_{\alpha\beta}/\partial\nabla\rho_\alpha = \nabla\rho_\beta$ (`vxc.rs:629-639`).
Each spin block is floored on its own $\rho_\sigma$.

The matrix-element form and the functional-derivative potential
$v_{xc} = v_\rho - 2\nabla\cdot(v_\sigma\nabla\rho)$ differ by a pure divergence,
$$v_\rho\chi_\mu\chi_\nu + 2v_\sigma\nabla\rho\cdot\nabla(\chi_\mu\chi_\nu) - v_{xc}\chi_\mu\chi_\nu = \nabla\cdot\bigl(2v_\sigma\nabla\rho\,\chi_\mu\chi_\nu\bigr),$$
which integrates to zero for decaying AOs, so both define the same Fock matrix;
the code uses the left form (no second derivatives of $\rho$ needed).

- Proof: `scripts/dft_vxc_matrix_elements_sympy.py` (generic $e(\rho,\sigma,\tau)$
  and generic 7-variable polarized $e$; the divergence identity in 3D).
- Tests: `ferric-python/tests/test_xc_blocks.py::test_vxc_is_the_derivative_of_exc`
  (FD of $E_{xc}$ along a random symmetric $X$ vs $\mathrm{Tr}[VX]$, PBE / SCAN / ωB97X,
  closed and polarized, $10^{-9}$ relative),
  `test_polarized_equals_closed_at_equal_spin_densities`,
  `ferric-dft/tests/mgga_polarized_packing.rs` (packing, $\tau$-channel independence,
  vs PySCF `eval_xc`).

## $f_{xc}$ response kernels

`GgaFxcKernel::apply_with_ref` (`src/fxc.rs:314`) returns
$\delta V^\sigma$ for a perturbation $(\delta D^\alpha, \delta D^\beta)$. Because
$V^\sigma$ enters the AO matrix only through $v_{\rho_\sigma}$ and $\mathbf{G}^\sigma$,
and the back-projection is linear in them, $\delta V^\sigma$ uses the same
back-projection with
$$u^\sigma = \delta v_{\rho_\sigma},\qquad \mathbf{w}^\sigma = \delta\mathbf{G}^\sigma,$$
$$\delta v_{\rho_\alpha} = f_{\rho_\alpha\rho_\alpha}\delta\rho_\alpha + f_{\rho_\alpha\rho_\beta}\delta\rho_\beta + \sum_c f_{\rho_\alpha\sigma_c}\delta\sigma_c,\qquad
\delta v_{\sigma_c} = \sum_{\sigma'} f_{\sigma_c\rho_{\sigma'}}\delta\rho_{\sigma'} + \sum_{c'} f_{\sigma_c\sigma_{c'}}\delta\sigma_{c'},$$
$$\delta\sigma_{\alpha\alpha} = 2\nabla\rho_\alpha\cdot\nabla\delta\rho_\alpha,\quad
\delta\sigma_{\alpha\beta} = \nabla\rho_\alpha\cdot\nabla\delta\rho_\beta + \nabla\rho_\beta\cdot\nabla\delta\rho_\alpha\quad(\texttt{fxc.rs:415}),$$
$$\mathbf{w}^\alpha = 2\,\delta v_{\sigma_{\alpha\alpha}}\nabla\rho_\alpha + \delta v_{\sigma_{\alpha\beta}}\nabla\rho_\beta
+ 2 v_{\sigma_{\alpha\alpha}}\nabla\delta\rho_\alpha + v_{\sigma_{\alpha\beta}}\nabla\delta\rho_\beta\quad(\texttt{fxc.rs:478, 534}).$$
libxc's second derivatives are read packed: `v2rho2` $=[\alpha\alpha,\alpha\beta,\beta\beta]$,
`v2rhosigma` $=[\alpha\cdot\alpha\alpha,\alpha\cdot\alpha\beta,\alpha\cdot\beta\beta,\beta\cdot\alpha\alpha,\beta\cdot\alpha\beta,\beta\cdot\beta\beta]$,
`v2sigma2` the upper triangle $[\alpha\alpha\cdot\alpha\alpha,\alpha\alpha\cdot\alpha\beta,\alpha\alpha\cdot\beta\beta,\alpha\beta\cdot\alpha\beta,\alpha\beta\cdot\beta\beta,\beta\beta\cdot\beta\beta]$.
The LDA kernel (`LdaFxcKernel::apply_with_ref`, `fxc.rs:121`) is the
$\sigma$-free case $u^\alpha = f_{\alpha\alpha}\delta\rho_\alpha + f_{\alpha\beta}\delta\rho_\beta$, $\mathbf{w}=0$.
The $\tau$-only meta-GGA kernel (`MggaFxcKernel`, `fxc.rs:817`) extends the
per-point Hessian to the 7 variables $(\rho_\alpha,\rho_\beta,\sigma_{\alpha\alpha},\sigma_{\alpha\beta},\sigma_{\beta\beta},\tau_\alpha,\tau_\beta)$
(`point_hessians`, `fxc.rs:709`; libxc `v2rhotau`, `v2sigmatau`, `v2tau2` fill the
$\tau$ rows). With $\tau_\sigma = \tfrac12\sum_\kappa\sum_{\mu\nu} D^\sigma_{\mu\nu}\,\partial_\kappa\chi_\mu\,\partial_\kappa\chi_\nu$
linear in $D$,
$$\delta\tau_\sigma = \tfrac12\sum_\kappa\sum_{\mu\nu}\delta D^\sigma_{\mu\nu}\,\partial_\kappa\chi_\mu\,\partial_\kappa\chi_\nu,\qquad
\delta v_{x_i} = \sum_j f_{x_i x_j}\,\delta x_j,$$
and the response adds one back-projected term to $u$ and $\mathbf w$ (`linearize_point`,
`fxc.rs:746`; `backproject`, `fxc.rs:960`):
$$\delta V^\sigma_{\mu\nu} = \sum_g w_g\Big[u^\sigma\chi_\mu\chi_\nu + \mathbf w^\sigma\cdot\nabla(\chi_\mu\chi_\nu)
+ \tfrac12\, t^\sigma\,\nabla\chi_\mu\cdot\nabla\chi_\nu\Big],\qquad t^\sigma = \delta v_{\tau_\sigma}.$$
$v_\tau$ itself does not appear (only $\delta v_\tau$), because $V^\tau$ is linear in $v_\tau$
and $\tau$ is linear in $D$. Every term is zeroed at points where the reference
$\rho_\sigma$ is below the production `DENSITY_FLOOR`, as $V^\sigma$ is. None of these
kernels contains the VV10 response; that is `vv10_potential_tangent` (`src/vv10.rs:1555`).

Consumers. `ferric_scf::rohf::FxcKernelStore` (`ferric-scf/src/rohf.rs:113`) dispatches
LDA / GGA / meta-GGA for the orbital-Hessian users: RKS and UKS Newton steps,
RKS/UKS internal stability, and the RKS/UKS/ROKS Z-vector
(`zvector_ks`). `lr_kernel::SingletKernel` (`src/lr_kernel.rs:87`) does the same for
the singlet $(ia|f_{\alpha\alpha}+f_{\alpha\beta}|jb)$ block of TDA / Casida TDDFT
(`ferric_tddft::run_tddft`, `ferric_gw::tddft::run_tda_dft`). Refused for meta-GGA:
ROKS Newton / augmented-Hessian and TRAH (DIIS is kept; not anchored) and the analytic nuclear Hessian (its $\tau$-dependent
$V_{xc}$ nuclear derivatives are not implemented).

- Proof: `scripts/dft_fxc_kernel_sympy.py` ($\delta\sigma$, $u$, $\mathbf w$ for a
  generic cubic $e$ whose first and second derivatives are independent symbols;
  packing-swap control).
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

`vv10_internal_screened` (`src/vv10.rs:908`), closed shell, total density:
$$E_{nl} = \sum_i W_i\rho_i\Bigl[\beta + \tfrac12 f_i\Bigr],\qquad
f_i = \sum_p W_p\rho_p\,\Phi_{ip},\qquad
\Phi_{ip} = -\frac{3}{2\,g_i g_p (g_i+g_p)},\qquad g_x = \omega_0(\mathbf r_x) R_{ip}^2 + \kappa(\mathbf r_x),$$
$$\omega_0 = \sqrt{C\,(\sigma/\rho^2)^2 + \tfrac{4\pi}{3}\rho},\qquad
\kappa = b\,\tfrac{3\pi}{2}\Bigl(\frac{\rho}{9\pi}\Bigr)^{1/6},\qquad
\beta = \tfrac{1}{32}\bigl(3/b^2\bigr)^{3/4}.$$
The pair sum includes the self pair $p=i$ ($R=0$). With
$t_{ip} = W_p\rho_p/(g_i g_p g_t)$, $g_t = g_i+g_p$,
$u_i = \sum_p t_{ip}(1/g_i + 1/g_t)$ and $w_i = \sum_p t_{ip}(1/g_i+1/g_t)R_{ip}^2$, the
per-point potentials (`vv10.rs:1150-1151`) are
$$v_{\rho,i} = \beta + f_i + \tfrac32\Bigl(u_i\,\rho\frac{\partial\kappa}{\partial\rho} + w_i\,\rho\frac{\partial\omega_0}{\partial\rho}\Bigr)_i,\qquad
v_{\sigma,i} = \tfrac32\,w_i\,\Bigl(\rho\frac{\partial\omega_0}{\partial\sigma}\Bigr)_i,$$
so that $W_i v_{\rho,i} = \partial E_{nl}/\partial\rho_i$ and $W_i v_{\sigma,i} = \partial E_{nl}/\partial\sigma_i$.
The code's `dk_dr`, `dw0_dr`, `dw0_dg` (`vv10.rs:966`) are $\rho$ times the
partial derivatives: the $\rho_i$ prefactor of $E_{nl}$ is folded in.
$V_{nl}$ is assembled by `add_vv10_scratch` (`vv10.rs:1373`) with the closed-shell
GGA form $w v_\rho\chi_\mu\chi_\nu + 2 w v_\sigma\nabla\rho\cdot(\chi_\mu\nabla\chi_\nu+\chi_\nu\nabla\chi_\mu)$
(`vv10.rs:1425`). Points with $\rho < 10^{-8}$ are excluded from the pair sum;
$\partial\omega_0/\partial\sigma$ is set to zero where $\sigma \le 10^{-8}$.

- Proof: `scripts/dft_vv10_potential_sympy.py` (two-point grid with self pairs,
  exact; the constants; module-doc $\Phi$ form).
- Tests: `vv10::tests::truncated_potentials_are_the_derivative_of_the_truncated_energy`
  (per-point FD, $10^{-8}$), `vv10::tests::potential_tangent_is_the_derivative_of_production_vnl`,
  `vv10::tests::screen_matches_dense_exactly_in_the_trivial_limit`,
  `ferric-scf/tests/dft_gradient_wb97xv.rs` (ωB97X-V nuclear gradient vs FD).
- **Open issue (T015).** The AO matrix $V_{nl}$ is not exactly $\partial E_{nl}/\partial D$:
  on NH$_2$/6-31G, $\mathrm{FD}(E) - \mathrm{Tr}[VX] = -1.31\times10^{-7}$
  ($3.6\times10^{-6}$ relative), constant over step sizes, while ωB97X without VV10
  agrees to $8.6\times10^{-14}$ (`test_xc_blocks.py::test_vxc_is_the_derivative_of_exc`,
  strict xfail). The per-point partials above are exact, so the residual lies
  elsewhere on the $D\to(\rho,\sigma)\to V$ path; cause not identified.

## XC nuclear gradient: AO-derivative and grid-response terms

`xc_gradient_closed_mgga_from_density` (`src/gradient.rs:1312`, batch body
`mgga_closed_batch`, `gradient.rs:1370`); the GGA path is the $v_\tau = 0$ case.
Translating nucleus $A$ moves the AOs centred on $A$:
$\partial\chi_\mu/\partial R_{A,\alpha} = -\partial_\alpha\chi_\mu$ for $\mu\in A$.
At fixed $D$ and fixed grid, with $m = D\chi$ and $(D\partial_b\chi)_\mu = \sum_\nu D_{\mu\nu}\partial_b\chi_\nu$,
$$\frac{\partial\tau}{\partial R_{A,\alpha}} = -\sum_b\sum_{\mu\in A}\partial^2_{\alpha b}\chi_\mu\,(D\partial_b\chi)_\mu .$$
The $\tfrac12$ in $\tau$ cancels against the $\mu\leftrightarrow\nu$ doubling.
`mgga_tau_partials` (`gradient.rs:232`) returns half of this sum and
`scatter_partials` (`gradient.rs:108`) applies $-2\times$, which reproduces it.
The $\rho,\sigma$ partials (`gga_ao_partials`, `gradient.rs:149`) with
$c_b = 2w v_\sigma\partial_b\rho$:
$$\frac{\partial E_{xc}}{\partial R_{A,\alpha}}\Big|_{\rm AO} = -2\sum_g\sum_{\mu\in A}\Bigl[w v_\rho\,\partial_\alpha\chi_\mu m_\mu
+ \sum_b c_b\bigl(\partial_\alpha\chi_\mu (D\partial_b\chi)_\mu + \partial^2_{\alpha b}\chi_\mu m_\mu\bigr)
+ \tfrac12 w v_\tau\sum_b\partial^2_{\alpha b}\chi_\mu (D\partial_b\chi)_\mu\Bigr].$$
Grid response: the weight term $\sum_g (\partial w_g/\partial\mathbf R_b)\,\rho_g\epsilon_g$
(weight1, above) plus the home-atom translation of the integrand,
$\sum_{g:\,{\rm home}=A} w_g[v_\rho\partial_\alpha\rho + v_\sigma\partial_\alpha\sigma + v_\tau\partial_\alpha\tau]$, with
$$\partial_\alpha(\partial_b\rho) = 2\Bigl[\sum_\mu\partial_\alpha\chi_\mu(D\partial_b\chi)_\mu + \sum_\mu m_\mu\,\partial^2_{\alpha b}\chi_\mu\Bigr]\ (\texttt{hess\_col},\ \texttt{gradient.rs:1483}),\qquad
\partial_\alpha\tau = \sum_b\sum_\mu\partial^2_{\alpha b}\chi_\mu (D\partial_b\chi)_\mu\ (\texttt{tau\_spatial\_grad},\ \texttt{gradient.rs:266}).$$
The home-translation term is the correct correction because $\rho$, $\sigma$
and $\tau$ are translation invariant:
$\sum_C\partial q/\partial\mathbf R_C + \nabla_{\mathbf r}q = 0$. The UKS path
(`xc_gradient_uks_mgga_from_density`, `gradient.rs:1844`) is the same per spin
with $D\to D^\sigma$, $\mathbf{G}^\sigma$ in place of $2v_\sigma\nabla\rho$.

- Proof: `scripts/dft_mgga_tau_gradient_sympy.py` (p- and s-Gaussians on two
  centres, exact rational point; the $\tfrac12$-kept and halved-spatial controls
  fail).
- Tests: `ferric-scf/tests/dft_gradient_mgga.rs` (SCAN/r2SCAN/TPSS vs FD of the
  SCF energy; SCAN/r2SCAN vs PySCF `grid_response=True`; UKS NH$_2$, OH),
  `ferric-scf/tests/dft_gradient_gga.rs` and `dft_gradient_lda.rs` (vs FD),
  `ferric-python/tests/test_xc_blocks.py::test_xc_grad_matches_finite_difference_at_fixed_density`
  (PBE, SCAN; fixed-$D$ FD, closed and polarized).

## Follow-ups (identities with no direct Rust test)

- $\sum_b \partial w_g/\partial\mathbf R_b = 0$ for the assembled weight1 rows
  (`grid.rs:673`); covered only end to end through the KS gradients.
- The VV10 AO-matrix residual (T015, above).
