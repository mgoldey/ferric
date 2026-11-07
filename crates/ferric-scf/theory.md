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
This allows for $O(N)$ scaling in the Coulomb build.

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

**Open shell: not implemented.** There is no analytic UKS or ROKS Hessian,
fixed or moving grid. Open-shell KS frequencies are central differences of
the analytic UKS/ROKS gradient; `HessianMethod::Analytic` and
`AnalyticMovingGrid` refuse an open-shell `xc` before the SCF
(`hessian.rs::xc_refusal`, `frequencies.rs::harmonic_frequencies_ks_moving_grid`;
tests `frequency_hessian_dispatch.rs::analytic_kohn_sham_requests_name_the_open_shell_gap`,
`ks_moving_grid_frequencies.rs::every_refusal_is_named`). Building it means
running every expression above once per spin with $D\to D^\sigma$
($\rho^\sigma_x$, $\mathbf g^\sigma_x$, $\rho^\sigma_{xy}$, $\mathbf g^\sigma_{xy}$ keep their form
exactly, with $P^\sigma=D^\sigma\chi$, $Q^\sigma=D^\sigma\partial\chi$: the factors 2 come from the
$\mu\leftrightarrow\nu$ symmetry of $D$, not from spin),
$f(\rho_\alpha,\rho_\beta,\sigma_{\alpha\alpha},\sigma_{\alpha\beta},\sigma_{\beta\beta})$,
$F^\sigma=Wf_{\rho_\sigma}$, $\mathbf G^\sigma=W(2f_{\sigma_{\sigma\sigma}}\mathbf g^\sigma+f_{\sigma_{\alpha\beta}}\mathbf g^{\bar\sigma})$,
and a $\delta F^\sigma,\delta\mathbf G^\sigma$ that couple both spins' $\rho_1,\mathbf g_1$
(the polarized `v2rho2`/`v2rhosigma`/`v2sigma2` that `fxc.rs::GgaFxcKernel`
already evaluates for UKS Newton and stability). The UKS CPHF is the UHF one
(`ucphf_response`) with $G^\sigma[D_1]=J[D_1^\alpha+D_1^\beta]-aK[D_1^\sigma]+V^{(1)\sigma}_{xc}[D_1^\alpha,D_1^\beta]$;
the moving-grid $T\,G_h\,T^{\mathsf T}$ fill and the weight terms are
spin-blind apart from $\partial_x f=\sum_\sigma(F^\sigma\rho^\sigma_x+\mathbf G^\sigma\cdot\mathbf g^\sigma_x)/W$.

| Equation | Rust | Numerical test |
|---|---|---|
| $g$, $\rho_x$, $\mathbf g_x$ | `gradient_densities`, `rho_derivative`, `grad_rho_derivative` | `ks_fixed_grid_hessian.rs::gradient_matches_central_difference_of_the_fixed_grid_energy` |
| $F\rho_{xy}+\mathbf G\cdot\mathbf g_{xy}$, $\delta F$, $\delta G$ | `add_same_centre`, `add_cross_centre`, `weights_response`, `kernel_part` | `xc_pieces_match_frozen_density_differences`, `gga_xc_pieces_match_frozen_density_differences` |
| $\partial V/\partial x$ | `first_order_fock`, `add_ao_derivative_fock` | same two tests (FD of $V_{xc}$ at displaced nuclei) |
| $V^{(1)}[D_1]$ | `XcResponse::apply` | `xc_kernel_response_matches_difference_of_vxc_in_the_density`, `gga_kernel_response_…` |
| $T G_h T^{\mathsf T}$, moving $\partial V/\partial x$ | `skeleton_hessian_moving`, `translation_fill`, `first_order_fock_moving`, `weight_fock` | `ks_moving_grid_hessian.rs::moving_xc_pieces_match_frozen_density_differences`, `moving_grid_mutants_each_fail_a_frozen_density_anchor` |
