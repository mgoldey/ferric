# Random Phase Approximation (RPA) Theory

This crate computes the direct Random Phase Approximation (dRPA) correlation energy, closed and open shell, its analytic gradient, range-separated MP2 + RPA, and RPA polarizabilities and C6 coefficients. The proofs of the identities below are `scripts/rpa_drpa_pdep_sympy.py`, `scripts/rpa_rs_mp2_rpa_sympy.py`, `scripts/rpa_gradient_sympy.py` and `scripts/rpa_c6_polarizability_sympy.py`.

## RPA Correlation Energy

The direct RPA (dRPA) correlation energy has three equivalent forms. ferric
evaluates all three; they agree to round-off on a fixed problem
(`scripts/rpa_drpa_pdep_sympy.py`).

**Definition and conventions.** Closed shell, canonical RHF orbitals, $i,j$
active occupied, $a,b$ virtual, $e_{ia} = \varepsilon_a - \varepsilon_i > 0$,
compound index $ia = i\,n_v + a$. The RI factor
$\tilde B^P_{ia} = \sum_Q (V^{-1/2})_{PQ}(Q|ia)$ gives $(ia|jb) = \sum_P \tilde B^P_{ia}\tilde B^P_{jb}$,
$K_{ia,jb} = (ia|jb)$. In spin orbitals the direct RPA (no exchange) is
$\mathbf A = \boldsymbol\Delta + \mathbf K$, $\mathbf B = \mathbf K$ over every
spin-orbital ov pair, and $E_c = \tfrac12\sum_n(\Omega_n - A_{nn})$. Spin
adaptation splits it into a singlet block, $\mathbf A = \boldsymbol\Delta + 2\mathbf K$,
$\mathbf B = 2\mathbf K$, and a triplet block with no direct coupling
($\Omega = \Delta$, contributing zero).

**Plasmon formula** (`ferric_mp2::drpa_amplitude::canonical_plasmon_drpa`):

$$ E_c = \tfrac12\Big(\sum_n \Omega_n - \operatorname{tr}\mathbf A\Big), \qquad \Omega_n^2 = \operatorname{eig}\big[(\mathbf A - \mathbf B)(\mathbf A + \mathbf B)\big], \qquad \mathbf A = \boldsymbol\Delta + 2\mathbf K,\ \mathbf B = 2\mathbf K . $$

The occupied and virtual Fock blocks are re-diagonalized first, because
the energy is first order in any Fock inconsistency.

**drCCD Riccati** (`drpa_amplitude::amplitude_drpa_dense`,
`riccati_masked_solve`, `fock_super`). With $\mathbf B = 2\mathbf K$ the ring
amplitudes solve

$$ \mathbf R(\mathbf T) = \mathbf B + \mathcal F(\mathbf T) + \mathbf B\mathbf T + \mathbf T\mathbf B + \mathbf T\mathbf B\mathbf T = 0, \qquad
\mathcal F(\mathbf T) = \hat\Delta\mathbf T + \mathbf T\hat\Delta, \quad \hat\Delta = \mathbf 1_o\otimes F_{vv} - F_{oo}\otimes\mathbf 1_v, $$

and $E_c = \tfrac12\sum_{ia,jb}B_{ia,jb}T_{ia,jb}$. $\hat\Delta$ is
$\operatorname{diag}(e_{ia})$ in canonical orbitals; written with the Fock
blocks the equation holds in any rotated occupied and virtual basis, which is
why the solver runs in Boys-localized occupied and VV-HV virtual orbitals.
The solver is the damped fixed point $\mathbf T \leftarrow \mathbf T - \mathbf R(\mathbf T)/D$,
$D_{ia,jb} = F_{aa} + F_{bb} - F_{ii} - F_{jj}$, optionally DIIS-accelerated.
In one mode ($\mathbf A = D + b$, $\mathbf B = b$) the root continuous at
$b = 0$ of $b + 2(D + b)t + bt^2 = 0$ gives $\tfrac12 bt = \tfrac12(\sqrt{D(D+2b)} - D - b)$,
the plasmon value.

**Frequency integral** (the ACFD form; `energy.rs`):

$$ E_c = \frac{1}{2\pi}\int_0^\infty \Big[\ln\det(\mathbf 1 + \boldsymbol\Pi(i\omega)) - \operatorname{tr}\boldsymbol\Pi(i\omega)\Big]\,d\omega,
\qquad \boldsymbol\Pi(i\omega) = 4\,\tilde B\,\operatorname{diag}\!\Big(\frac{e_{ia}}{\omega^2 + e_{ia}^2}\Big)\tilde B^{\mathsf T}. $$

$\boldsymbol\Pi = -\tilde{\boldsymbol\chi}_0\mathbf v$ in the Coulomb-metric
basis, so $\tilde{\boldsymbol\epsilon} = \mathbf 1 + \boldsymbol\Pi$ is the
symmetrized dielectric matrix, all eigenvalues $\ge 1$. The prefactor 4 is 2
for spin and 2 for the $\pm\omega$ poles of $\chi_0$
(`sternheimer::build_scale_factors`, which scales by
$\sqrt{4e_{ia}/(\omega^2 + e_{ia}^2)}$, and
`dielectric_matrix_from_projection_into`). In one mode
$\frac{1}{2\pi}\int_0^\infty[\ln(1 + \frac{2bD}{\omega^2 + D^2}) - \frac{2bD}{\omega^2 + D^2}]d\omega = \tfrac12(\sqrt{D(D+2b)} - D - b)$
with $b = 2K$. By Sylvester's identity
$\ln\det(\mathbf 1 + \tilde Bg\tilde B^{\mathsf T}) = \ln\det(\mathbf 1 + g^{1/2}\mathbf Kg^{1/2})$,
so the energy depends on $\tilde B$ only through $\mathbf K$ and any factor of
$V^{-1}$ (Cholesky $L^{-1}$ or symmetric $V^{-1/2}$) gives the same result.

The per-frequency summand is taken two ways with the same value:
`trace_log_summands_of_projection` uses the LU log-determinant,
$\ln\det\tilde{\boldsymbol\epsilon} + \operatorname{tr}(\mathbf 1 - \tilde{\boldsymbol\epsilon})$;
the eigenvalue route (`rpa_correlation_energy`, `trace_log_sum`) sums
$f(\lambda) = \ln\lambda + 1 - \lambda$ over the eigenvalues, with
`trace_log_summand_term` switching to the series
$f = -\sum_{k\ge2}(-x)^k/k$, $x = \lambda - 1$, for $|x| < 0.25$ to avoid the
cancellation of the literal form near $\lambda = 1$.

**Quadrature.** `quadrature::gauss_legendre_nodes` maps Gauss-Legendre nodes
$x_k$ to $\omega_k = u_0(1 + x_k)/(1 - x_k)$ with weights
$w_k^{\rm GL}\,2u_0/(1 - x_k)^2$, so $\int_0^\infty f\,d\omega \approx \sum_k w_kf(\omega_k)$
and $E_c \approx \frac{1}{2\pi}\sum_k w_k S(i\omega_k)$. The grid depends only
on $(n, u_0)$, never on the molecule. The exact `drpa` kind
(`energy_route::exact_drpa_config`) uses $n = 48$, $u_0 = 0.5$, the aux-basis
log-det route and no truncation; `run_drpa` without `local` solves the
Riccati equation at $\varepsilon = 0$.

**Second order.** Expanding the plasmon energy in $b$ gives
$-b^2/(4D) + b^3/(4D^2) - \dots$; in many modes the first Riccati amplitude is
$\mathbf T_1 = -\mathbf B/D$, so

$$ E_c^{(2)} = \tfrac12\sum \mathbf B\circ\mathbf T_1 = -2\sum_{ijab}\frac{(ia|jb)^2}{e_{ia} + e_{jb}} = 2E_{\rm OS}, $$

twice the opposite-spin MP2 energy (`ferric_mp2::rimp2::pair_energy`
accumulates $E_{\rm OS} = \sum(ia|jb)^2/D_{ij}^{ab}$ with
$D = \varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b < 0$).
dRPA has no second-order exchange term.

**Amplitude-threshold local scheme** (`amplitude_drpa`, `amplitude_drpa_scan`,
CLI `[local] scheme = "amplitude-threshold"`). In the localized basis the
amplitudes are kept only on the pattern $|B_{ia,jb}| = 2|(ia|jb)| > \varepsilon$
(symmetric, so swap-closed): $\mathbf R$ and $\mathbf T$ are projected onto it
every iteration and $E_c = \tfrac12\sum_{\rm pattern}B\circ T$. At
$\varepsilon = 0$ the pattern is everything and the energy is the exact one.
The masked energy is not variational; its error is roughly linear in
$\varepsilon$.

## The PDEP Eigenpotential Approach

PDEP replaces the aux basis by a smaller, system-adapted one taken from the
static dielectric matrix (`run_pdep_rpa`, `lib.rs`).

1. **Static eigenpotentials.** $\tilde{\boldsymbol\epsilon}(0) = \mathbf 1 + 4\tilde B\,\operatorname{diag}(1/e_{ia})\tilde B^{\mathsf T}$
   is diagonalized iteratively (block Lanczos by default,
   `Eigensolver::Lanczos`; Davidson is selectable):
   $\tilde{\boldsymbol\epsilon}(0)\mathbf u_\alpha = \lambda_\alpha(0)\mathbf u_\alpha$.
   The modes with $|\lambda_\alpha(0) - 1| > \texttt{trunc\_thresh}$ are kept
   (`PdepRpaConfig::n_kept_modes`); `trunc_thresh = 0` keeps all of them.

2. **Response matvec.** The Sternheimer equation
   $(\hat H_0 - \epsilon_i + i\omega)|\Delta\Psi_i\rangle = -P_{\rm virt}\hat V|\Psi_i\rangle$
   is the formulation; ferric's RI-MO implementation sums over every
   occupied-virtual pair instead. With $\mathbf y = U^{\mathsf T}\tilde B$
   ($m \times n_{ov}$, frequency independent),
   $U^{\mathsf T}\tilde{\boldsymbol\epsilon}(i\omega)U = \mathbf 1 + 4\,\mathbf y\,\operatorname{diag}(e_{ia}/(\omega^2 + e_{ia}^2))\,\mathbf y^{\mathsf T}$
   (`dielectric_matrix_from_projection_into`).
   `sternheimer::chi_from_trial_potential` is a test helper that returns the
   single-spin value $-2\sum_{ia} e_{ia}/(\omega^2 + e_{ia}^2)\,(\sum_P V_P\tilde B^P_{ia})^2$,
   half the closed-shell $\langle V|\tilde\chi_0|V\rangle$.

3. **Frequency-dependent eigenvalues.** The static basis does not diagonalize
   $\tilde{\boldsymbol\epsilon}(i\omega)$. At each node the projected matrix
   $U^{\mathsf T}\tilde{\boldsymbol\epsilon}(i\omega_k)U$ is diagonalized
   (`eval_eigenvalues_at_frequencies_budgeted`, `eigvalsh_dc`), and
   $\lambda_\alpha(i\omega_k)$ are its eigenvalues, not its diagonal
   $\langle\mathbf u_\alpha|\tilde{\boldsymbol\epsilon}(i\omega_k)|\mathbf u_\alpha\rangle$.
   The diagonal alone would drop the off-diagonal coupling (2.8e-2 Ha on the
   test model).

4. **Energy.**
   $$ E_c = \frac{1}{2\pi}\sum_k w_k\sum_\alpha\big[\ln\lambda_\alpha(i\omega_k) + 1 - \lambda_\alpha(i\omega_k)\big]
   = \frac{1}{2\pi}\sum_k w_k\big[\ln\det\tilde{\boldsymbol\epsilon}_U(i\omega_k) + \operatorname{tr}(\mathbf 1 - \tilde{\boldsymbol\epsilon}_U(i\omega_k))\big], $$
   $\tilde{\boldsymbol\epsilon}_U = U^{\mathsf T}\tilde{\boldsymbol\epsilon}U$
   (`rpa_correlation_energy`). Both terms are basis-invariant traces, so
   with every mode kept ($U$ orthogonal, $m = n_{\rm aux}$) this is the
   aux-basis log-det energy of the previous section exactly. That is why the
   full-rank route (`RpaEnergyRoute::AuxBasisLogDet`) skips the static
   eigensolve. With truncation the energy is that of the projected
   dielectric and converges to the full one as `trunc_thresh` goes to 0.

The PDEP basis avoids forming and diagonalizing the full
$n_{\rm aux}\times n_{\rm aux}$ dielectric at every frequency when few modes
are kept.

## Open-Shell (Spin-Summed) Dielectric

`run_u_pdep_rpa` (`crates/ferric-rpa/src/lib.rs`) takes a UHF reference, or a ROHF reference semi-canonicalized per spin. Each spin $\sigma$ has its own RI tensor $B^P_{i_\sigma a_\sigma}$ and gaps $e_{ia}^\sigma = \epsilon_a^\sigma - \epsilon_i^\sigma$ in one shared aux basis, and the symmetrized dielectric is

$$ \tilde{\mathbf{\epsilon}}_U(i\omega) = \mathbf{1} + \mathbf{\Pi}_\alpha(i\omega) + \mathbf{\Pi}_\beta(i\omega), \qquad \Pi^\sigma_{PQ}(i\omega) = 2 \sum_{ia} \frac{e^\sigma_{ia}}{\omega^2 + (e^\sigma_{ia})^2} B^P_{i_\sigma a_\sigma} B^Q_{i_\sigma a_\sigma}. $$

The closed-shell $\mathbf{\Pi}$ carries the prefactor 4; with identical spin channels $\mathbf{\Pi}_\alpha + \mathbf{\Pi}_\beta$ equals it, so a UHF singlet reproduces `run_pdep_rpa`. With different $\alpha$ and $\beta$ orbitals, gaps and occupations, $\frac{1}{2\pi}\int[\ln\det\tilde{\boldsymbol\epsilon}_U - \operatorname{tr}(\boldsymbol\Pi_\alpha + \boldsymbol\Pi_\beta)]d\omega$ equals the spin-orbital definition of the "RPA Correlation Energy" section, $\tfrac12\sum_n(\Omega_n - A_{nn})$ with $\mathbf A = \boldsymbol\Delta + \mathbf K$, $\mathbf B = \mathbf K$ over both spin blocks, $K_{i_\sigma a_\sigma, j_\tau b_\tau} = (i_\sigma a_\sigma|j_\tau b_\tau)$ (`scripts/rpa_drpa_pdep_sympy.py`). The per-spin prefactor 2 is the $\pm\omega$ pole pair alone (`build_scale_factors_with_prefactor(.., 2.0)`). The PDEP eigensolve, truncation, frequency projection and trace-log integration are those of the closed-shell path applied to $\tilde{\mathbf{\epsilon}}_U$; the matvec $\tilde{\mathbf{\epsilon}}_U \mathbf{V} = \mathbf{V} + \mathbf{\Pi}_\alpha \mathbf{V} + \mathbf{\Pi}_\beta \mathbf{V}$ is two closed-shell-shaped applications (`sternheimer::dielectric_apply_unrestricted`). The Laplace $\chi_0$ backend builds one Laplace quadrature per spin on that spin's gap range. `run_diagnostics` assembles $\tilde{\mathbf{\epsilon}}_U(i\omega_k)$ densely at every quadrature node (`diagnostics::u_ri_drpa_energy`).

The Boys-screened $\chi_0$ is closed-shell only: its screened tiles and localized orbital energies are built from the restricted MOs and Fock matrix.

## Dispersion: Casimir-Polder C6

`dispersion::casimir_polder_c6` turns per-atom dynamic polarizabilities on the
imaginary-frequency quadrature $(\omega_k, w_k)$ into pair coefficients:

$$ C_6^{AB} = \frac{3}{\pi}\int_0^\infty \bar\alpha^A(i\omega)\,\bar\alpha^B(i\omega)\,d\omega \approx \frac{3}{\pi}\sum_k w_k\,\bar\alpha^A(i\omega_k)\,\bar\alpha^B(i\omega_k),\qquad \bar\alpha = \tfrac13\operatorname{Tr}\alpha, $$

and, element by element (no tensor contraction), the anisotropic
$C^{AB}_{6,ij} = \frac{3}{\pi}\sum_k w_k\,\alpha^A_{ij}(i\omega_k)\,\alpha^B_{ij}(i\omega_k)$.
Two sources feed it.

**Tkatchenko-Scheffler single pole** (`ts_dynamic_polarizability`,
`mbd::ts_atom_params`). With the Hirshfeld volume ratio $r_A$ and the
free-atom $\alpha^{\rm free}$, $C_6^{\rm free}$ (TS PRL 2009 Table I for
$Z \le 18$; Gould and Bučko 2016 for $Z = 19$–$54$, Pd α from Jerabek et al. 2018; other $Z$ is an error):

$$ \alpha_A = r_A\,\alpha^{\rm free},\quad C_{6,A} = r_A^2\,C_6^{\rm free},\quad \bar\alpha_A(i\omega) = \frac{\alpha_A}{1+(\omega/\omega_A)^2},\quad \omega_A = \frac{4}{3}\frac{C_{6,A}}{\alpha_A^2}. $$

$\omega_A$ follows from the single-pole integral
$\frac{3}{\pi}\int_0^\infty \alpha_A^2(1+\omega^2/\omega_A^2)^{-2}\,d\omega = \frac34\alpha_A^2\omega_A = C_{6,A}$.
The tensor is $\bar\alpha_A(i\omega)$ times the static tensor normalized to
unit isotropic average, so only the shape comes from the molecular calculation.

**PDEP-RPA** (`pdep_dynamic_polarizability`,
`properties::pdep_polarizability_becke_dynamic` / `_hirshfeld_dynamic`). The
per-atom tensor is the Krishtal-Senet-Van Alsenoy intrinsic polarizability
(JCP 125, 034312 (2006)),
$\alpha^A_{dj}(i\omega) = \langle m^A_d|R(i\omega)|\mu_j\rangle$, with the
atom-weighted dipole $m^A_d = w_A(\mathbf r)(r-R_A)_d$ on the bra and the
molecular dipole on the ket. $R(i\omega)$ is the RI response with
$g_{ia}(\omega) = e_{ia}/(\omega^2+e_{ia}^2)$ and the screened dielectric
$\tilde\epsilon(\omega) = \mathbf 1 + 4\tilde B\,\mathrm{diag}(g(\omega))\,\tilde B^{\mathsf T}$
(closed shell). At $\omega = 0$, $g_{ia} = 1/e_{ia}$ and the static per-atom
tensor is recovered. The molecular tensor $\alpha_{\rm mol} = \langle\mu|R|\mu\rangle$
is not the atom sum: $\alpha_{\rm CT} = \alpha_{\rm mol} - \sum_A\alpha^A$ is the
field-induced charge transfer between atoms (`charge_transfer_remainder`,
`_dynamic`).

**Per-atom kernel** (`properties/per_atom.rs`, `closed_per_atom_alpha`). With
$G = \operatorname{diag}(g_{ia})$, $\mathbf K = \tilde B^{\mathsf T}\tilde B$ and
the ov vectors $m^A_{d,ia} = \langle i|m^A_d|a\rangle$,
$\mu_{j,ia} = \langle i|r_j|a\rangle$, the closed-shell dRPA response is
$R = (G^{-1}/4 + \mathbf K)^{-1}$, which Woodbury turns into the aux-basis form
the code evaluates:

$$ \alpha^A_{dj}(i\omega) = 4\,m^A_d\cdot(g\circ\mu_j) - 16\,\big(\tilde B(g\circ m^A_d)\big)^{\mathsf T}\tilde{\boldsymbol\epsilon}^{-1}\big(\tilde B(g\circ\mu_j)\big), \qquad \tilde{\boldsymbol\epsilon} = \mathbf 1 + 4\tilde BG\tilde B^{\mathsf T}. $$

The same $R$ is the sum over the dRPA (Casida) excitations,
$\alpha(i\omega) = \sum_n f_n/(\Omega_n^2 + \omega^2)$ with
$\Omega_n^2\mathbf Z_n = \Delta^{1/2}(\Delta + 4\mathbf K)\Delta^{1/2}\mathbf Z_n$ and
$f_n = 4\,(m^{\mathsf T}\Delta^{1/2}\mathbf Z_n)(\mathbf Z_n^{\mathsf T}\Delta^{1/2}\mu)$.
The static drivers use $g_{ia} = 1/e_{ia}$, the $\omega \to 0$ limit. The code
then symmetrizes each tensor, $\alpha^A \to \tfrac12(\alpha^A + \alpha^{A\mathsf T})$
(`symmetrize_tensor`). For partition weights with $\sum_Aw_A = 1$ the bras
sum to $\sum_A m^A_d = \mu_d - \sum_A R_{A,d}\,w_A$ (and $\sum_A\langle i|w_A|a\rangle = 0$), so by linearity
$\alpha_{\rm CT} = \alpha_{\rm mol} - \sum_A\operatorname{sym}\alpha^A = \operatorname{sym}\langle\sum_AR_Aw_A|R|\mu\rangle$.

**Quadrature.** The C6 sums run on the RPA frequency grid of
`quadrature.rs` (default Gauss-Legendre, 20 points, $u_0 = 0.5$). For the
single-pole model the closed form is
$\frac{3}{\pi}\int_0^\infty\bar\alpha_A\bar\alpha_B\,d\omega = \frac32\,\frac{\alpha_A\alpha_B\omega_A\omega_B}{\omega_A + \omega_B}$
(London), and the 20-point sum reproduces it to $3\times10^{-14}$ relative
for $\omega_{A,B} \approx 0.43\ E_h$ in the test model; the mapped rule converges geometrically on this rational integrand. For a pole expansion
$\alpha(i\omega) = \sum_n f_n/(\Omega_n^2 + \omega^2)$,
$C_6^{AB} = \frac32\sum_{mn}f^A_mf^B_n/(\Omega_m\Omega_n(\Omega_m + \Omega_n))$.
`scripts/rpa_c6_polarizability_sympy.py` checks these identities, the kernel
against the Casida sum and the charge-transfer sum rule.

## Range-separated MP2 + RPA

`rs_mp2_rpa::rs_mp2_lr_rpa` combines RI-MP2 and dRPA with the Coulomb
operator split as $1/r = v_{\rm sr} + v_{\rm lr}$ ($\operatorname{erfc}/\operatorname{erf}$
at $\omega$, or terfc/terf at $r_0$), so $(ia|jb)_{\rm C} = (ia|jb)_{\rm sr} + (ia|jb)_{\rm lr}$.
With the spin components of `ferric_mp2::rimp2::pair_energy`,

$$ E_{\rm OS}[v] = \sum_{ijab}\frac{(ia|jb)_v^2}{D_{ij}^{ab}}, \qquad
E_{\rm SS}[v] = \sum_{ijab}\frac{(ia|jb)_v[(ia|jb)_v - (ib|ja)_v]}{D_{ij}^{ab}}, \qquad
E_{\rm MP2}[v] = E_{\rm OS} + E_{\rm SS}, $$

($D = \varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b$; $E_{\rm OS} + E_{\rm SS}$ is the
spin-orbital $\frac14\sum|\langle ij||ab\rangle|^2/D$), the two formulations are

$$ E^{\rm B}_c = E_{\rm MP2}[\mathrm C] + \big(E_{\rm dRPA}[\mathrm{lr}] - 2E_{\rm OS}[\mathrm{lr}]\big) \quad (\texttt{DeltaLr}, \text{default}), $$

$$ E^{\rm T}_c = E_{\rm MP2}[\mathrm C] + \big(E_{\rm dRPA}[\mathrm C] - 2E_{\rm OS}[\mathrm C]\big) - \big(E_{\rm dRPA}[\mathrm{sr}] - 2E_{\rm OS}[\mathrm{sr}]\big) \quad (\texttt{CoupledRings}). $$

Each $E_{\rm dRPA}$ is a `run_pdep_rpa_from_intermediates` call (full rank by
default, `RsMp2RpaConfig::default`) on the same dressed $\tilde B$ the MP2
components use. The subtraction
$2E_{\rm OS}$ is exactly the second-order term of the dRPA it accompanies
(section "RPA Correlation Energy": $E^{(2)}_{\rm dRPA} = 2E_{\rm OS}$; on the
$n$-point frequency grid it holds to the quadrature error), so every bracket
starts at third order and both formulations equal $E_{\rm MP2}[\mathrm C]$ through second
order, exchange included: nothing is counted twice. Scaling every integral by
$\lambda$, the third-order terms are

$$ E^{\rm B\,(3)}_c = E^{(3)}_{\rm dRPA}[\mathrm{lr}], \qquad E^{\rm T\,(3)}_c = E^{(3)}_{\rm dRPA}[\mathrm C] - E^{(3)}_{\rm dRPA}[\mathrm{sr}], $$

with $E^{(3)}_{\rm dRPA} = \tfrac12\sum\mathbf B\circ\mathbf T_2$,
$D\circ\mathbf T_2 = -(\mathbf B\mathbf T_1 + \mathbf T_1\mathbf B)$. In one mode
($E_{\rm dRPA}(b) = -b^2/(4D) + b^3/(4D^2) + O(b^4)$, $b = b_s + b_l$) the
difference is $E^{\rm T\,(3)} - E^{\rm B\,(3)} = 3b_sb_l(b_s + b_l)/(4D^2)$: the
mixed short-range by long-range rings that B drops and T keeps. Both forms
reduce to $E_{\rm MP2}[\mathrm C]$ when $v_{\rm lr} = 0$ ($\omega \to 0$) and
to $E_{\rm MP2}[\mathrm C] + E_{\rm dRPA}[\mathrm C] - 2E_{\rm OS}[\mathrm C]$
when $v_{\rm sr} = 0$. `scripts/rpa_rs_mp2_rpa_sympy.py` checks these
statements, the ring orders against Cauchy-integral Taylor coefficients of the
plasmon energy, and the spin components against spin-orbital MP2.

## Analytic nuclear gradient (closed-shell RI-dRPA, HF reference)

`gradient_analytic.rs` differentiates the energy `run_pdep_rpa` evaluates at
full rank (`trunc_thresh = 0`), on the fixed $n$-point imaginary-frequency
grid $(\omega_k, w_k)$ of `quadrature.rs`. Every scheme there (Gauss-Legendre,
MiniMax, ChebyshevTan) has nodes and weights that depend only on $(n, u_0)$,
never on the molecule, so the grid has no geometry derivative and the
gradient is the exact derivative of the $n$-point sum, not of the converged
integral: at $n = 6$ it must match a finite difference of the 6-point energy,
which differs from the 40-point one by $3 \times 10^{-4}$ Ha/Bohr on distorted
H2O. The derivation follows the frequency-integrated relaxed-density route
of Burow, Bates, Furche and Eshuis (JCTC 10, 180 (2014)); the ring-CCD
Lagrangian of Rekkedal et al. (JCP 139, 081101 (2013)) gives the same
gradient for the direct-ring energy and is not used.

### The energy as a function of B and F

With $B^P_{ia} = \sum_Q (V^{-1/2})_{PQ}(Q|ia)$ (the RI factor, $V_{PQ} = (P|Q)$),
the closed-shell dielectric of the Open-Shell section and $\Pi = \tilde\epsilon - \mathbf 1$,

$$ E_c = \frac{1}{2\pi}\sum_k w_k\, S(\Pi_k), \qquad S(\Pi) = \ln\det(\mathbf 1 + \Pi) - \operatorname{tr}\Pi,
\qquad \Pi_k = B\, g_k(\Delta)\, B^{\mathsf T}, \qquad g_k(e) = \frac{4e}{\omega_k^2 + e^2}. $$

At full rank $S$ is basis-invariant, so the eigenvalue and log-det routes of
`energy.rs` are this expression. $\Delta$ is the occupied-virtual
orbital-energy-difference operator,
$\Delta_{ia,jb} = F_{ab}\delta_{ij} - F_{ij}\delta_{ab}$, which is
$\operatorname{diag}(e_{ia})$, $e_{ia} = \varepsilon_a - \varepsilon_i$, in
canonical orbitals. Written with the Fock blocks instead of the orbital
energies, $E_c(B, F_{oo}, F_{vv})$ is invariant to rotations among the
occupied and among the virtual orbitals ($B \to O^{\mathsf T}B$,
$F_{oo} \to O^{\mathsf T}F_{oo}O$ leaves $\Pi$ unchanged). Non-canonical
dependence is therefore handled by never leaving this invariant form: the
orbital-energy derivative is the derivative with respect to the full
symmetric Fock blocks, and the occupied-occupied and virtual-virtual
rotations need no response equation. $E_c$ depends on $B$ only through the
fitted integrals $B^{\mathsf T}B$ (Sylvester: $\ln\det(\mathbf 1 + BgB^{\mathsf T})
= \ln\det(\mathbf 1 + g^{1/2}B^{\mathsf T}Bg^{1/2})$), so any factor of $V^{-1}$
gives the same energy and gradient.

### First variation

With $Q_k = (\mathbf 1 + \Pi_k)^{-1} - \mathbf 1$ (symmetric, naux x naux),
$\delta S = \operatorname{tr}[Q_k\,\delta\Pi_k]$ and
$\delta\Pi_k = \delta B\,g_kB^{\mathsf T} + Bg_k\,\delta B^{\mathsf T} + B\,\delta g_k\,B^{\mathsf T}$:

$$ \delta E_c = \sum_{P,ia}\Gamma^P_{ia}\,\delta B^P_{ia} + \sum_{ij}P_{ij}\,\delta F_{ij} + \sum_{ab}P_{ab}\,\delta F_{ab}. $$

**Three-index two-particle density** (frequency-integrated):

$$ \Gamma^P_{ia} = \frac{1}{\pi}\sum_k w_k\, g_k(e_{ia})\,(Q_kB)^P_{ia}. $$

**Occupied-occupied and virtual-virtual densities.** With
$M_k = B^{\mathsf T}Q_kB$ (nov x nov), $\delta E_c|_B = \frac{1}{2\pi}\sum_k w_k
\operatorname{tr}[M_k\,\delta g_k(\Delta)]$. The derivative of the matrix
function $g_k(\Delta)$ in the eigenbasis of $\Delta$ (Daleckii-Krein) is
$\delta g_k(\Delta)_{ia,jb} = g_k^{[1]}(e_{ia}, e_{jb})\,\delta\Delta_{ia,jb}$ with
the divided difference

$$ g^{[1]}(x, y) = \frac{g(x) - g(y)}{x - y} = \frac{4(\omega^2 - xy)}{(\omega^2 + x^2)(\omega^2 + y^2)}, $$

which is smooth through $x = y$ (it is $g'(x)$ there), so degenerate or
near-degenerate orbitals need no special case. Inserting
$\delta\Delta_{ia,jb} = \delta_{ij}\delta F_{ab} - \delta_{ab}\delta F_{ij}$:

$$ P_{ab} = \frac{1}{2\pi}\sum_k w_k \sum_i (M_k)_{ia,ib}\, g_k^{[1]}(e_{ia}, e_{ib}), \qquad
P_{ij} = -\frac{1}{2\pi}\sum_k w_k \sum_a (M_k)_{ia,ja}\, g_k^{[1]}(e_{ia}, e_{ja}). $$

Both are symmetric. Their diagonals are $\partial E_c/\partial\varepsilon_a$ and
$\partial E_c/\partial\varepsilon_i$ of the canonical expression; the
off-diagonal elements are what the canonical expression alone cannot
supply. Only the blocks $(M_k)_{ia,ib}$ and $(M_k)_{ia,ja}$ are needed:
per frequency, $W_k = Q_kB$ (naux$^2$ nov) and then one
$(n_v \times n_{aux})(n_{aux} \times n_v)$ product per occupied orbital and one
$(n_o \times n_{aux})(n_{aux} \times n_o)$ per virtual, the RI-MP2 density cost
times $n$.

**The second-order term.** Replacing $S$ by its leading term
$-\tfrac12\operatorname{tr}\Pi^2$ gives $Q_k = -\Pi_k$ and, as $n \to \infty$
($\int_0^\infty \frac{xy}{(\omega^2+x^2)(\omega^2+y^2)}d\omega = \frac{\pi}{2(x+y)}$),
the direct-ring MP2 energy $E^{(2)}_{\mathrm{dr}} = 2\sum_{iajb}(ia|jb)^2/D_{ij}^{ab}$,
$D = \varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b$: twice the
opposite-spin MP2 energy, i.e. SCS-MP2 with $(c_{\mathrm{OS}}, c_{\mathrm{SS}}) = (2, 0)$.
In that limit $\Gamma$, $P_{ij}$, $P_{ab}$ become
$8\sum_{jb}t_{ij}^{ab}B^P_{jb}$, $-4\sum_{kab}t_{ik}^{ab}t_{jk}^{ab}$ and
$4\sum_{ijc}t_{ij}^{ac}t_{ij}^{bc}$ with $t = (ia|jb)/D$, the
$(2, 0)$-scaled densities of ferric-mp2 theory section 9.2 in this
section's normalization ($\Gamma = 4x$, $P = P^{\mathrm{mp2}} + P^{\mathrm{mp2}\,\mathsf T}$).
This is the exactness anchor of the gradient.

### Lagrangian and Z-vector

The orbitals are canonical RHF orbitals: $F_{ai} = 0$, orthonormal in the
moving AO basis. Parametrize the orbital change as $C(x) = C\,U(x)$ with
$U + U^{\mathsf T} = -S^{(x)}$ ($S^{(x)} = C^{\mathsf T}\partial_xS\,C$). By the
invariance above the symmetric choice $U_{oo} = -\tfrac12S^{(x)}_{oo}$,
$U_{vv} = -\tfrac12S^{(x)}_{vv}$ is exact, and only $U_{ai}$ needs a response
equation. Collect the orbital derivative of $E_c$ as $\sum_{pq}Y_{pq}U_{pq}$:

$$ Y_{pi} = \sum_{Pa}\Gamma^P_{ia}B^P_{pa}, \qquad Y_{pa} = \sum_{Pi}\Gamma^P_{ia}B^P_{ip}, \qquad
Y_{pq} \mathrel{+}= 2P_{pq}\varepsilon_p \ (pq \in oo, vv), $$

the last term from $\delta F_{pq} \ni U_{qp}\varepsilon_q + U_{pq}\varepsilon_p$.
The Fock blocks also move through the density: with $G[X] = J[X] - \tfrac12K[X]$,
$\sum_{pq}P_{pq}(C^{\mathsf T}G[D^{(1)}]C)_{pq} = \operatorname{tr}(G[P^{\mathrm{AO}}]D^{(1)})$,
$P^{\mathrm{AO}} = C_oP_{oo}C_o^{\mathsf T} + C_vP_{vv}C_v^{\mathsf T}$ and
$D^{(1)} = 2(C_vU_{vo}C_o^{\mathsf T} + \mathrm{h.c.}) - 2C_oS^{(x)}_{oo}C_o^{\mathsf T}$.
The coefficient of $U_{ai}$ (using $U_{ia} = -S^{(x)}_{ia} - U_{ai}$) is the
Z-vector right-hand side

$$ L_{ai} = Y_{ai} - Y_{ia} + 4\,(C_v^{\mathsf T}G[P^{\mathrm{AO}}]C_o)_{ai}. $$

The CPHF equation of the reference, $A\,U^{(x)} = R^{(x)}$ with
$(AU)_{ai} = (\varepsilon_a - \varepsilon_i)U_{ai} + (C_v^{\mathsf T}G[2(C_vUC_o^{\mathsf T} + \mathrm{h.c.})]C_o)_{ai}$
and $R^{(x)}_{ai} = -F^{[x]}_{ai} + \varepsilon_iS^{(x)}_{ai} + (C_v^{\mathsf T}G[2C_oS^{(x)}_{oo}C_o^{\mathsf T}]C_o)_{ai}$,
has a symmetric $A$, so the $3N$ responses are replaced by one solve
$A z = L$ and $\sum_{ai}L_{ai}U^{(x)}_{ai} = \sum_{ai}z_{ai}R^{(x)}_{ai}$.
$F^{[x]} = C^{\mathsf T}(\partial_xh + G^{[x]}[D^{\mathrm{HF}}])C$ is the
skeleton Fock derivative at fixed density.

### Gradient

With the relaxed correlation density
$D_2 = P^{\mathrm{AO}} - \tfrac12(C_vzC_o^{\mathsf T} + \mathrm{h.c.})$:

$$ \frac{dE}{dx} = \frac{dE_{\mathrm{HF}}}{dx}
+ \operatorname{tr}\!\big[D_2(\partial_xh + G^{[x]}[D^{\mathrm{HF}}])\big]
+ \sum_{pq}W_{pq}S^{(x)}_{pq}
+ \sum_{Q\mu\nu}\tilde\Gamma^Q_{\mu\nu}\,\partial_x(Q|\mu\nu)
- \tfrac12\sum_{PQ}\gamma_{PQ}\,\partial_xV_{PQ}, $$

$$ W_{oo} = -\tfrac12Y_{oo} - 2C_o^{\mathsf T}G[D_2]C_o, \quad W_{vv} = -\tfrac12Y_{vv}, \quad
W_{ia} = -Y_{ia}, \quad W_{ai} = z_{ai}\varepsilon_i, $$

$\tilde\Gamma^Q_{ia} = \sum_P(V^{-1/2})_{QP}\Gamma^P_{ia}$ back-transformed to
AOs, and $\gamma_{PQ} = \sum_{ia}\tilde\Gamma^P_{ia}c^Q_{ia}$ with
$c = V^{-1}(Q|ia)$. (With $U_{oo}$ fixed at $-\tfrac12S^{(x)}_{oo}$ the
$G[D^{(1)}]$ overlap term of $P^{\mathrm{AO}}$ and the one of the Z-vector
combine into the single $G[D_2]$ term of $W_{oo}$.)

`scripts/rpa_gradient_sympy.py` checks this gradient on a toy closed-shell
system with $x$-dependent $S$, $h$ and exactly factorized integrals
$(\mu\nu|\lambda\sigma) = \sum_PL_{P,\mu\nu}L_{P,\lambda\sigma}$ (unit metric, so
the $\gamma$ term vanishes): the Lagrangian expression above, with
$\Gamma = 4x$, $P = 2p$ from the transcribed `rpa_ri_densities`, equals a
Richardson central difference of $E_{\rm RHF} + E_c$ with the RHF re-solved at
$x \pm h$ to $2\times10^{-12}$ (and $3\times10^{-11}$ with one frozen core
orbital). It also checks $\Gamma$ and $P$ directly as derivatives of
$E_c(B, F_{oo}, F_{vv})$ with off-diagonal Fock blocks, the divided difference
and the second-order integral.

### Frozen core

With core orbitals $I$ frozen, $\Gamma$ and $P_{ij}$ run over active
occupied orbitals only, and core-active rotations are no longer a symmetry
of $E_c$. The canonical condition $F_{Ij} = 0$ fixes them
($U_{Ij}(\varepsilon_I - \varepsilon_j) = -F^{(x)}_{Ij} + \varepsilon_jS^{(x)}_{Ij}$
without the orbital terms), which adds the symmetric core-active block

$$ P_{Ij} = P_{jI} = -\frac{Y_{Ij} - Y_{jI}}{2(\varepsilon_I - \varepsilon_j)} = -\frac{Y_{Ij}}{2(\varepsilon_I - \varepsilon_j)} $$

to $P_{oo}$ ($Y_{jI} = 0$: no core column enters $B$; the factor $\tfrac12$
because $\sum_{ij}P_{ij}\delta F_{ij}$ visits the pair twice). From there it is an
ordinary part of $P$: it enters $Y$'s $2P\varepsilon$ term, $P^{\mathrm{AO}}$,
$L$, $D_2$ and $W$. This is the `frozen_core_block` of ferric-mp2 (section 9,
$D_{kI} = I_{kI}/(\varepsilon_k - \varepsilon_I)$, and its $I_{qi} =
-2\sum_{Pa}x^P_{ia}B^P_{qa}$ is $-\tfrac12 Y_{qi}$ at $x = \Gamma/4$). It needs a gap between the core and the
active occupied orbitals, which every frozen-core choice of
`ferric_core` has.

### What is reused from ferric-mp2

The gradient is the RI-MP2 relaxed-density gradient (ferric-mp2 theory
section 9) with a different $(\Gamma, P_{oo}, P_{vv})$. Unchanged:
the Lagrangian matrix $I$ from the three-index density (`build_imat_ri`,
which takes $x = \Gamma/4$), the frozen core-active block, the Z-vector
solve on the reference J/K route (`solve_zvector_timed`), the relaxed and
energy-weighted density assembly and its one-electron, overlap and
two-electron derivative terms (`mp2_relaxed_lagrangian_gradient`), and the
3-centre/2-centre RI response (`integral_response_gradient_3c2c_timed`). The
RPA-specific code is the frequency loop that builds $\Gamma$, $P_{oo}$,
$P_{vv}$ from $B$, the orbital energies and the grid. The MP2 pipeline
reads the pair amplitudes only through $x$, so the RPA intermediates carry
$x$ directly and no $t_2$.

## Literature Citations

1. **PDEP Eigenpotentials:** F. Gygi and G. Galli, *Phys. Rev. B* **65**, 220102(R) (2002).
2. **PDEP-RPA Correlation Energy:** D. Rocca, D. Lu, and G. Galli, *J. Chem. Phys.* **133**, 164109 (2010).
3. **MiniMax Quadrature:** A. Takatsuka, S. Ten-no, and W. Hackbusch, *J. Chem. Phys.* **129**, 044112 (2008).
4. **RI-RPA Review:** H. Eshuis, J. E. Bates, and F. Furche, *Theor. Chem. Acc.* **131**, 1084 (2012).
