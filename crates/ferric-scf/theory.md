# Self-Consistent Field (SCF) Theory

This module implements the Restricted Hartree-Fock (RHF) method and associated optimization techniques.

## Hartree-Fock Equations
The RHF energy is given by:
$$E_{HF} = 2 \sum_i H_{ii} + \sum_{ij} [2(ii|jj) - (ij|ij)] + V_{nn}$$
where $H$ is the core Hamiltonian.

The Fock matrix in the AO basis is:
$$F_{\mu\nu} = H_{\mu\nu} + \sum_{\lambda\sigma} D_{\lambda\sigma} [2(\mu\nu|\lambda\sigma) - (\mu\lambda|\nu\sigma)]$$
where $D_{\lambda\sigma} = \sum_i C_{\lambda i} C_{\sigma i}$ is the density matrix.

## Convergence Acceleration (DIIS)
Direct Inversion in the Iterative Subspace (DIIS) minimizes the error vector:
$$\mathbf{e}_k = \mathbf{F}_k \mathbf{D}_k \mathbf{S} - \mathbf{S} \mathbf{D}_k \mathbf{F}_k$$
A linear combination of past Fock matrices is formed:
$$\mathbf{F}_{k+1} = \sum_i c_i \mathbf{F}_i$$
where the coefficients $c_i$ are found by solving the DIIS system.

## Continuous Fast Multipole Method (CFMM)
To avoid the $O(N^4)$ cost of direct ERI evaluation, CFMM treats the long-range Coulomb interaction using multipole expansions:
$$J(\mathbf{r}) \approx \sum_{l,m} M_{lm} \frac{Y_{lm}(\theta, \phi)}{r^{l+1}}$$
A hierarchical FMM (M2M/L2L translations) makes the far-field Coulomb build
$O(N)$. ferric's `cfmm` module is behind the off-by-default `cfmm-incomplete`
feature: its far field is a flat leaf-to-leaf M2L with no M2M/L2L pass, so it
does not have that asymptotic scaling, and it is slower than `DirectJ`. It
reproduces the direct J to ~1e-14 in the all-near-field limit.

## Kohn-Sham XC Hessian terms (fixed and moving grid)
Implemented in `src/ks_fixed_grid.rs` (closed-shell LDA/GGA and global
hybrids; the exchange and Coulomb Hessian terms live in `src/hessian/ks.rs`).
Proof: `scripts/ks_fixed_grid_hessian_sympy.py`. The first-order identities
$V_{xc}=\partial E_{xc}/\partial D$, the per-point $f_{xc}$ linearization and
the Becke weight derivatives are proved in the scripts listed in
`crates/ferric-dft/theory.md`.

The energy is $E_{xc}=\sum_g W_g f(\rho_g,\sigma_g)$, $\sigma=|\mathbf g|^2$,
$\mathbf g=\nabla\rho$. With $P_\mu=(D\chi)_\mu$, $Q_{\mu\gamma}=(D\,\partial_\gamma\chi)_\mu$,
$\partial_\alpha$ the electron-coordinate derivative, and $x=(B,\alpha)$,
$y=(C,\beta)$ nuclear coordinates:
$$g_\gamma = 2\sum_\mu \partial_\gamma\chi_\mu P_\mu,\qquad
\rho_x = -2\sum_{\mu\in B}\partial_\alpha\chi_\mu P_\mu,\qquad
g_{\gamma,x} = -2\sum_{\mu\in B}\big[\partial_\gamma\partial_\alpha\chi_\mu P_\mu + \partial_\alpha\chi_\mu Q_{\mu\gamma}\big],$$
$$\rho_{xy} = 2\delta_{BC}\sum_{\mu\in B}\partial_\alpha\partial_\beta\chi_\mu P_\mu
 + 2\sum_{\mu\in B,\nu\in C} D_{\mu\nu}\,\partial_\alpha\chi_\mu\,\partial_\beta\chi_\nu,$$
$$g_{\gamma,xy} = 2\delta_{BC}\sum_{\mu\in B}\big[\partial_\gamma\partial_\alpha\partial_\beta\chi_\mu P_\mu + \partial_\alpha\partial_\beta\chi_\mu Q_{\mu\gamma}\big]
 + 2\sum_{\mu\in B,\nu\in C} D_{\mu\nu}\big[\partial_\gamma\partial_\alpha\chi_\mu\,\partial_\beta\chi_\nu + \partial_\alpha\chi_\mu\,\partial_\gamma\partial_\beta\chi_\nu\big].$$
With $F=W f_\rho$ and $G_\gamma = 2W f_\sigma g_\gamma$ ($G=0$ for an LDA):
$$\delta F[\rho_1,\mathbf g_1] = W\big(f_{\rho\rho}\rho_1 + 2f_{\rho\sigma}\,\mathbf g\cdot\mathbf g_1\big),\qquad
\delta G_\gamma[\rho_1,\mathbf g_1] = 2W\big[(f_{\rho\sigma}\rho_1 + 2f_{\sigma\sigma}\,\mathbf g\cdot\mathbf g_1)g_\gamma + f_\sigma g_{1\gamma}\big],$$
$$V_{\mu\nu}[F,G] = \sum_g F\chi_\mu\chi_\nu + G_\gamma(\partial_\gamma\chi_\mu\chi_\nu + \chi_\mu\partial_\gamma\chi_\nu).$$
On a grid fixed in space ($W_g=w_g$ constant):
$$\frac{\partial E}{\partial x} = \sum_g F\rho_x + \mathbf G\cdot\mathbf g_x,\qquad
\frac{\partial^2 E}{\partial x\,\partial y}\Big|_D = \sum_g \rho_x\,\delta F[y] + \mathbf g_x\cdot\delta\mathbf G[y] + F\rho_{xy} + \mathbf G\cdot\mathbf g_{xy},$$
$$\frac{\partial V_{\mu\nu}}{\partial x}\Big|_D = V_{\mu\nu}[\delta F[x],\delta G[x]] - \big(T^{\alpha}_{\mu\nu}[\mu\in B] + T^{\alpha}_{\nu\mu}[\nu\in B]\big),\quad
T^\alpha_{\mu\nu} = \sum_g F\,\partial_\alpha\chi_\mu\chi_\nu + G_\gamma\big(\partial_\gamma\partial_\alpha\chi_\mu\chi_\nu + \partial_\alpha\chi_\mu\partial_\gamma\chi_\nu\big),$$
and the CPHF kernel is $V^{(1)}[D_1]=V[\delta F[\rho_1,\mathbf g_1],\delta G[\rho_1,\mathbf g_1]]$ with
$\rho_1=\sum D_{1,\mu\nu}\chi_\mu\chi_\nu$, $\mathbf g_1 = 2\sum D_{1,\mu\nu}\nabla\chi_\mu\chi_\nu$.
These are the derivatives of PySCF's `grid_response=False` surface.

On the moving (production) grid the points ride their home atom,
$\mathbf r_g=\mathbf R_h+\mathbf u_g$, and $W_g=s_g w^h(\mathbf r_g;\mathbf R)$.
$E_g$ is invariant under a joint translation of $\mathbf r$ and $\mathbf R$, so it depends only
on $\mathbf R_B-\mathbf R_h$, and with $T_{x,x}=1$, $T_{(h,\alpha),(B,\alpha)}=-1$ ($B\neq h$):
$$H^{(h)} = T\,G_h\,T^{\mathsf T},\qquad
G_h = \sum_{g\in h}\big[W\,\partial^2 f + W'(\partial f)^{\mathsf T} + \partial f\,W'^{\mathsf T} + W'' f\big]_{\mathbf r\ \text{fixed}},
\quad \partial_x f = f_\rho\rho_x + 2f_\sigma\,\mathbf g\cdot\mathbf g_x,$$
over the atoms $B,C\neq h$. The first-order Fock matrix of a home-$h$ batch is the
point-fixed derivative for $x\notin h$, including the weight term
$V[W'_x f_\rho,\,2W'_x f_\sigma\mathbf g]$, and $V^{(h,\alpha)} = -\sum_{B\neq h}V^{(B,\alpha)}$.

**Open shell (UKS).** `hessian::uks_moving_grid_hessian` runs every expression
above once per spin with $D\to D^\sigma$ ($\rho^\sigma_x$, $\mathbf g^\sigma_x$,
$\rho^\sigma_{xy}$, $\mathbf g^\sigma_{xy}$ keep their form exactly, with $P^\sigma=D^\sigma\chi$,
$Q^\sigma=D^\sigma\partial\chi$: the factors 2 come from the $\mu\leftrightarrow\nu$ symmetry of
$D$, not from spin), on $f(\rho_\alpha,\rho_\beta,\sigma_{\alpha\alpha},\sigma_{\alpha\beta},\sigma_{\beta\beta})$, $\sigma_{st}=\mathbf g^s\cdot\mathbf g^t$:
$$F^\sigma=Wf_{\rho_\sigma},\qquad \mathbf G^\alpha=W(2f_{\sigma_{\alpha\alpha}}\mathbf g^\alpha+f_{\sigma_{\alpha\beta}}\mathbf g^\beta)\ (\beta:\alpha\leftrightarrow\beta),$$
$$\delta\sigma_{\alpha\alpha}=2\mathbf g^\alpha\cdot\mathbf g_1^\alpha,\quad \delta\sigma_{\alpha\beta}=\mathbf g^\alpha\cdot\mathbf g_1^\beta+\mathbf g^\beta\cdot\mathbf g_1^\alpha,\quad \delta\sigma_{\beta\beta}=2\mathbf g^\beta\cdot\mathbf g_1^\beta,$$
$$\delta F^\sigma=W\Big[\sum_t f_{\rho_\sigma\rho_t}\rho_1^t+\sum_u f_{\rho_\sigma\sigma_u}\delta\sigma_u\Big],\qquad
\delta\mathbf G^\alpha=W\big[2\mathbf g^\alpha\,\delta f_{\sigma_{\alpha\alpha}}+\mathbf g^\beta\,\delta f_{\sigma_{\alpha\beta}}+2f_{\sigma_{\alpha\alpha}}\mathbf g_1^\alpha+f_{\sigma_{\alpha\beta}}\mathbf g_1^\beta\big],$$
with $\delta f_v=\sum_t f_{v\rho_t}\rho_1^t+\sum_u f_{v\sigma_u}\delta\sigma_u$ (libxc `v2rho2`/`v2rhosigma`/`v2sigma2`,
3/6/6 components). The skeleton is
$\sum_\sigma[\rho^\sigma_x\delta F^\sigma[y]+\mathbf g^\sigma_x\cdot\delta\mathbf G^\sigma[y]+F^\sigma\rho^\sigma_{xy}+\mathbf G^\sigma\cdot\mathbf g^\sigma_{xy}]$,
where $\delta F^\sigma[y]$ responds to the perturbation $(\rho^t_y,\mathbf g^t_y)$ of BOTH spins; $\partial V^\sigma/\partial x|_D=V[\delta F^\sigma[x],\delta\mathbf G^\sigma[x]]$
plus the AO-derivative rows of $V[F^\sigma,\mathbf G^\sigma]$. On the moving grid the $T\,G_h\,T^{\mathsf T}$ fill and the weight terms are
spin-blind apart from $\partial_x f=\sum_\sigma(F^\sigma\rho^\sigma_x+\mathbf G^\sigma\cdot\mathbf g^\sigma_x)/W$ and the per-spin
Fock weight term $V[W'_xf_{\rho_\sigma},W'_x\mathbf G^\sigma/W]$. Two-electron part: the UHF Hessian with exchange
scaled by the global-hybrid fraction $a$ ($\Gamma=\tfrac12DD-\tfrac a2\sum_\sigma D^\sigma D^\sigma$; RI-J on the total density and one RI-K
channel per spin with weight $-a/2$ where the SCF fitted them), and the coupled UCPHF with
$G^\sigma[D_1]=J[D_1^\alpha+D_1^\beta]-aK[D_1^\sigma]+V^{(1)\sigma}_{xc}[D_1^\alpha,D_1^\beta]$
(`ks_fixed_grid::PolarizedXcResponse`: the $\alpha\alpha$, $\alpha\beta$ and $\beta\beta$ blocks). Each spin's potential pieces are gated on its
own $\rho_\sigma$ (as `ferric_dft`'s polarized $V_{xc}$). Proved term by term against exact sympy derivatives with $D^\alpha\neq D^\beta$,
$D_1^\alpha\neq D_1^\beta$ in `scripts/uks_xc_hessian_sympy.py`, including the closed-shell limit; that script also shows the
artifact a singlet anchor cannot see: a "total-density" kernel (each spin's perturbation replaced by the spin average) reproduces
the closed-shell limit exactly and misses the polarized response. ROKS is not implemented (Roothaan-coupled response;
`HessianMethod::Analytic` and `AnalyticMovingGrid` refuse it before the SCF; ROKS frequencies are central differences of the
analytic gradient).

| Equation | Rust | Numerical test |
|---|---|---|
| $g$, $\rho_x$, $\mathbf g_x$ | `gradient_densities`, `rho_derivative`, `grad_rho_derivative` | `ks_fixed_grid_hessian.rs::gradient_matches_central_difference_of_the_fixed_grid_energy` |
| $F\rho_{xy}+\mathbf G\cdot\mathbf g_{xy}$, $\delta F$, $\delta G$ | `add_same_centre`, `add_cross_centre`, `weights_response`, `kernel_part` | `xc_pieces_match_frozen_density_differences`, `gga_xc_pieces_match_frozen_density_differences` |
| $\partial V/\partial x$ | `first_order_fock`, `add_ao_derivative_fock` | same two tests (FD of $V_{xc}$ at displaced nuclei) |
| $V^{(1)}[D_1]$ | `XcResponse::apply` | `xc_kernel_response_matches_difference_of_vxc_in_the_density`, `gga_kernel_response_…` |
| polarized $F^\sigma,\mathbf G^\sigma,\delta F^\sigma,\delta\mathbf G^\sigma$, skeleton, $\partial V^\sigma/\partial x$, $V^{(1)\sigma}$ | `ks_fixed_grid/polarized.rs` (`PolarizedKs`, `pol_response`, `PolarizedXcResponse`) | `uks_xc_pieces.rs` (closed-shell limit, frozen-density and frozen-grid FD, kernel FD with $D_1^\alpha\neq D_1^\beta$, mutants); full Hessian `uks_moving_grid_hessian.rs` |
| $T G_h T^{\mathsf T}$, moving $\partial V/\partial x$ | `skeleton_hessian_moving`, `translation_fill`, `first_order_fock_moving`, `weight_fock` | `ks_moving_grid_hessian.rs::moving_xc_pieces_match_frozen_density_differences`, `moving_grid_mutants_each_fail_a_frozen_density_anchor` |

## Fock hooks and their nuclear gradient

A Fock hook (`rhf::solve_rhf_with_hook`, `uhf::solve_uhf_with_hook`,
`rohf::solve_rohf_with_hook`; Python `fock_hook=`) adds one caller-supplied
term to the energy functional,

$$E(\mathbf{R}) = E_0[\mathbf{D}; \mathbf{R}] + E_h[\mathbf{D}; \mathbf{R}],
\qquad \mathbf{V} = \partial E_h / \partial \mathbf{D},$$

with $\mathbf{V}$ added to the Fock matrix every iteration ($\mathbf{V}_\sigma =
\partial E_h / \partial \mathbf{D}_\sigma$ per spin for UHF/ROHF). $E_0$ is the
built-in HF or KS energy. The SCF makes $E$ stationary under orbital rotations,
subject to orthonormality $\mathbf{C}^\top \mathbf{S} \mathbf{C} = \mathbf{1}$.
Its Lagrangian is

$$\mathcal{L} = E[\mathbf{D}(\mathbf{C}); \mathbf{R}]
 - \sum_{ij} \epsilon_{ij} \left(\mathbf{C}^\top \mathbf{S}(\mathbf{R}) \mathbf{C}
 - \mathbf{1}\right)_{ij}.$$

At convergence $\partial \mathcal{L} / \partial \mathbf{C} = 0$ gives
$\mathbf{F} \mathbf{C}_\text{occ} = \mathbf{S} \mathbf{C}_\text{occ}
\boldsymbol{\epsilon}$ with the HOOKED Fock matrix $\mathbf{F} = \mathbf{F}_0 +
\mathbf{V}$. The total derivative is the partial derivative of $\mathcal{L}$
at fixed $\mathbf{C}$:

$$\frac{dE}{d\mathbf{R}} =
 \left.\frac{\partial E_0}{\partial \mathbf{R}}\right|_{\mathbf{D}}
 + \left.\frac{\partial E_h}{\partial \mathbf{R}}\right|_{\mathbf{D}}
 - \operatorname{Tr}\!\left[\mathbf{W}\, \frac{\partial \mathbf{S}}{\partial \mathbf{R}}\right],
\qquad
\mathbf{W} = \sum_\sigma \mathbf{C}_{\sigma,\text{occ}} \boldsymbol{\epsilon}_\sigma \mathbf{C}_{\sigma,\text{occ}}^\top .$$

(The closed-shell $\mathbf{W}$ carries the factor 2 of the double occupation:
$\mathbf{W} = 2\mathbf{C}_\text{occ}\boldsymbol{\epsilon}\mathbf{C}_\text{occ}^\top
= \tfrac12 \mathbf{D}\mathbf{F}\mathbf{D}$.) Three pieces, three owners:

1. $\partial E_0 / \partial \mathbf{R}|_{\mathbf{D}}$: the built-in gradient's
   integral-derivative terms (nuclear repulsion, $\operatorname{Tr}[\mathbf{D}\,
   \partial \mathbf{h}/\partial \mathbf{R}]$, the two-electron and XC
   derivatives) at the hooked density. Unchanged code.
2. The Pulay term $-\operatorname{Tr}[\mathbf{W}\,\partial\mathbf{S}/\partial
   \mathbf{R}]$: $\mathbf{W}$ is built from the converged orbitals and
   eigenvalues, which diagonalize the hooked $\mathbf{F}$, so the $\mathbf{V}$
   part of the Lagrange multipliers is included without the hook doing
   anything. Its $\mathbf{V}$ part is $\tfrac12\mathbf{D}\mathbf{V}\mathbf{D}$
   (closed shell). For ROHF, $\mathbf{W} =
   \operatorname{sym}(\mathbf{D}_\alpha\mathbf{F}_\alpha\mathbf{D}_\alpha +
   \mathbf{D}_\alpha\mathbf{F}_\beta\mathbf{D}_\beta)$ uses the converged spin
   Fock matrices, which hold $\mathbf{V}_\alpha$, $\mathbf{V}_\beta$.
3. The EXPLICIT term $\partial E_h / \partial \mathbf{R}|_{\mathbf{D}}$, the
   derivative of the hook's energy with the AO density matrix held fixed. Only
   the hook knows it, so the caller supplies it (`fock_hook_grad=` in Python,
   [`gradient::with_fock_hook_gradient`] in Rust).

What the explicit term must and must not include:

- It MUST include every dependence of $E_h$ on the nuclear positions at fixed
  AO density matrix: integrals over AO functions centred on the atoms (the
  derivative of $\langle\mu|\hat{v}|\nu\rangle$ with respect to the centres of
  $\mu$ and $\nu$), operator centres that sit on atoms, and purely classical
  terms such as a charge-nuclear repulsion. Example: a point charge $q$ at
  fixed $\mathbf{r}_c$ has $E_h = \operatorname{Tr}[\mathbf{D}\mathbf{V}_q] +
  \sum_A q Z_A / |\mathbf{R}_A - \mathbf{r}_c|$; its explicit term is
  $\operatorname{Tr}[\mathbf{D}\,\partial\mathbf{V}_q/\partial\mathbf{R}_A]$
  (basis-centre derivatives only; the charge does not move) plus the
  derivative of the classical sum.
- It must NOT include any change of $\mathbf{D}$ with geometry (the response
  of the orbitals is handled by the Lagrangian above) nor the Pulay term of
  $\mathbf{V}$ (it is already in $\mathbf{W}$). A hook that depends only on
  $\mathbf{D}$, for example $E_h = \lambda \sum_{\mu\nu} D_{\mu\nu}^2$, has an
  explicit term of ZERO.
- A hook built from the overlap or another geometry-dependent AO matrix
  depends on $\mathbf{R}$ through that matrix. $E_h = \lambda
  \operatorname{Tr}[\mathbf{D}\mathbf{S}]$ has explicit term $\lambda
  \operatorname{Tr}[\mathbf{D}\,\partial\mathbf{S}/\partial\mathbf{R}]$. Its
  Pulay part, $-\operatorname{Tr}[\tfrac12\lambda\mathbf{D}\mathbf{S}\mathbf{D}
  \,\partial\mathbf{S}/\partial\mathbf{R}] = -\lambda\operatorname{Tr}[\mathbf{D}
  \,\partial\mathbf{S}/\partial\mathbf{R}]$ ($\mathbf{D}\mathbf{S}\mathbf{D} =
  2\mathbf{D}$), cancels it exactly, as it must: $\lambda\operatorname{Tr}
  [\mathbf{D}\mathbf{S}] = \lambda N$ is a constant. This is the exactness
  anchor of the tests: both halves nonzero, their sum known in closed form.

The formula holds for the converged hooked solution only (stationarity was
used). It assumes integer aufbau occupations of the hooked $\mathbf{F}$, as
the un-hooked gradients do, and that $\mathbf{V}$ is exactly $\partial E_h /
\partial \mathbf{D}$: a hook whose $\mathbf{V}$ is not the derivative of its
$E_h$ converges to a non-stationary point of $E$, and no gradient of the
reported energy follows from this formula.
