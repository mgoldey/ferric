# Random Phase Approximation (RPA) Theory

This crate computes the direct Random Phase Approximation (dRPA) correlation energy, closed and open shell, its analytic gradient, range-separated MP2 + RPA, and RPA polarizabilities and C6 coefficients. The proofs of the identities below are `scripts/rpa_drpa_pdep_sympy.py`, `scripts/rpa_rs_mp2_rpa_sympy.py`, `scripts/rpa_gradient_sympy.py`, `scripts/rpa_c6_polarizability_sympy.py` and `scripts/rpa_response_modes_sympy.py`.

Notation (book/notation.md): $u$ is the imaginary frequency, $\chi_0(iu)$, so
$\omega$ stays the range-separation parameter; $\xi, \eta$ label PDEP
eigenpotentials (dielectric modes); $\Delta_i^a = \varepsilon_a - \varepsilon_i > 0$
and $\Delta_{ij}^{ab} = \Delta_i^a + \Delta_j^b$, with every correlation
energy written as an explicit minus sign over a positive denominator.

## RPA Correlation Energy

The direct RPA (dRPA) correlation energy has three equivalent forms. ferric
evaluates all three; they agree to round-off on a fixed problem
(`scripts/rpa_drpa_pdep_sympy.py`).

**Definition and conventions.** Closed shell, canonical RHF orbitals, $i,j$
active occupied, $a,b$ virtual, compound index $ia = i\,n_v + a$. The RI
factor is $B^P_{ia} = \sum_Q M_{PQ}(Q|ia)$ with $M^{\mathsf T}M = V^{-1}$,
$V_{PQ} = (P|Q)$, so $(ia|jb) = \sum_P B^P_{ia}B^P_{jb}$ and
$K_{ia,jb} = (ia|jb)$. The code's dressing matrix (`RpaIntermediates::v_inv_sqrt`,
built by `ferric_mp2::rimp2::metric_inverse_sqrt`) is the Cholesky inverse
$M = L^{-1}$, $V = LL^{\mathsf T}$, for the Coulomb and erfc metrics, and the
symmetric $M = U\lambda^{-1/2}U^{\mathsf T}$ with eigenvalues below
$10^{-10}$ dropped (`eigh_inverse_sqrt`, so $M^{\mathsf T}M = V^{+}$) for the
erf and terf metrics, which can be indefinite in a Coulomb-fitted auxiliary
basis. In spin orbitals the direct RPA (no exchange) is
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
$\operatorname{diag}(\Delta_i^a)$ in canonical orbitals; written with the Fock
blocks the equation holds in any rotated occupied and virtual basis, which is
why the solver runs in Boys-localized occupied and VV-HV virtual orbitals.
The solver is the damped fixed point $\mathbf T \leftarrow \mathbf T - \mathbf R(\mathbf T)/\bar\Delta$,
$\bar\Delta_{ia,jb} = F_{aa} + F_{bb} - F_{ii} - F_{jj}$, optionally DIIS-accelerated.
In one mode ($\mathbf A = \Delta + b$, $\mathbf B = b$, $\Delta > 0$) the root
continuous at $b = 0$ of $b + 2(\Delta + b)t + bt^2 = 0$ gives
$\tfrac12 bt = \tfrac12(\sqrt{\Delta(\Delta+2b)} - \Delta - b)$, the plasmon value.

**Frequency integral** (the ACFD form; `energy.rs`):

$$ E_c = \frac{1}{2\pi}\int_0^\infty \Big[\ln\det(\mathbf 1 + \boldsymbol\Pi(iu)) - \operatorname{tr}\boldsymbol\Pi(iu)\Big]\,du,
\qquad \boldsymbol\Pi(iu) = 4\,B\operatorname{diag}\!\Big(\frac{\Delta_i^a}{u^2 + (\Delta_i^a)^2}\Big)B^{\mathsf T}. $$

$\boldsymbol\Pi = -\tilde{\boldsymbol\chi}_0\mathbf v$ in the Coulomb-metric
basis, so $\tilde{\boldsymbol\epsilon} = \mathbf 1 + \boldsymbol\Pi$ is the
symmetrized dielectric matrix, all eigenvalues $\ge 1$. The prefactor 4 is 2
for spin and 2 for the $\pm u$ poles of $\chi_0$
(`sternheimer::build_scale_factors`, which scales by
$\sqrt{4\Delta_i^a/(u^2 + (\Delta_i^a)^2)}$, and
`dielectric_matrix_from_projection_into`). In one mode
$\frac{1}{2\pi}\int_0^\infty[\ln(1 + \frac{2b\Delta}{u^2 + \Delta^2}) - \frac{2b\Delta}{u^2 + \Delta^2}]du = \tfrac12(\sqrt{\Delta(\Delta+2b)} - \Delta - b)$
with $b = 2K$. By Sylvester's identity
$\ln\det(\mathbf 1 + BgB^{\mathsf T}) = \ln\det(\mathbf 1 + g^{1/2}\mathbf Kg^{1/2})$,
so the energy depends on $B$ only through $\mathbf K = B^{\mathsf T}B$, and
the Cholesky and the symmetric dressing give the same result.

The per-frequency summand is taken two ways with the same value:
`trace_log_summands_of_projection` uses the LU log-determinant,
$\ln\det\tilde{\boldsymbol\epsilon} + \operatorname{tr}(\mathbf 1 - \tilde{\boldsymbol\epsilon})$;
the eigenvalue route (`rpa_correlation_energy`, `trace_log_sum`) sums
$f(\lambda) = \ln\lambda + 1 - \lambda$ over the eigenvalues, with
`trace_log_summand_term` switching to the series
$f = -\sum_{k\ge2}(-x)^k/k$, $x = \lambda - 1$, for $|x| < 0.25$ to avoid the
cancellation of the literal form near $\lambda = 1$.

**Quadrature.** $E_c \approx \frac{1}{2\pi}\sum_k w_k S(iu_k)$ on a grid
$(u_k, w_k)$ that depends only on the scheme, $n$ and $u_0$, never on the
molecule (section "Imaginary-frequency quadratures"). The default
(`QuadratureConfig::default`) is mapped Gauss-Legendre with $n = 20$,
$u_0 = 0.5$. The exact `drpa` kind (`energy_route::exact_drpa_config`) uses
$n = 48$, $u_0 = 0.5$, the aux-basis log-det route and no truncation;
`run_drpa` without `local` solves the Riccati equation at $\varepsilon = 0$.

**Second order.** Expanding the one-mode plasmon energy in $b$ gives
$-b^2/(4\Delta) + b^3/(4\Delta^2) - \dots$; in many modes the first Riccati
amplitude is $\mathbf T_1 = -\mathbf B/\bar\Delta$, so

$$ E_c^{(2)} = \tfrac12\sum \mathbf B\circ\mathbf T_1 = -2\sum_{ijab}\frac{(ia|jb)^2}{\Delta_{ij}^{ab}} = 2E_{\rm OS}, $$

twice the opposite-spin MP2 energy $E_{\rm OS} = -\sum_{ijab}(ia|jb)^2/\Delta_{ij}^{ab}$
(`ferric_mp2::rimp2::pair_energy` accumulates it by dividing by the negative
`denom` $= \varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b$).
dRPA has no second-order exchange term.

**Amplitude-threshold local scheme** (`amplitude_drpa`, `amplitude_drpa_scan`,
CLI `[local] scheme = "amplitude-threshold"`). In the localized basis the
amplitudes are kept only on the pattern $|B_{ia,jb}| = 2|(ia|jb)| > \varepsilon$
(symmetric, so swap-closed): $\mathbf R$ and $\mathbf T$ are projected onto it
every iteration and $E_c = \tfrac12\sum_{\rm pattern}B\circ T$. At
$\varepsilon = 0$ the pattern is everything and the energy is the exact one.
The masked energy is not variational. Its error has been positive (less
correlation) at every point measured and grows faster than linearly in
$\varepsilon$, because the dropped fraction itself grows quickly
(`site/src/methods/rpa-gw.md` gives the n-octane / 6-31G table); there is no
error bound.

## The PDEP Eigenpotential Approach

PDEP replaces the aux basis by a smaller, system-adapted one taken from the
static dielectric matrix (`run_pdep_rpa`, `lib.rs`).

1. **Static eigenpotentials.** $\tilde{\boldsymbol\epsilon}(0) = \mathbf 1 + 4B\,\operatorname{diag}(1/\Delta_i^a)B^{\mathsf T}$
   is diagonalized, $\tilde{\boldsymbol\epsilon}(0)\mathbf U_\xi = \lambda_\xi(0)\mathbf U_\xi$
   (`run_pdep_rpa_eigensolve`). The default solver, `Eigensolver::Lanczos`
   on the dense $\chi_0$, asks for all $N_{\rm aux}$ pairs, and that request
   is one dense `eigh` of the assembled matrix
   (`lanczos::run_lanczos_full_rank_budgeted`, which builds
   $\tilde{\boldsymbol\epsilon}(0)$ in column panels); no Krylov iteration
   runs. The block Lanczos iteration (`run_lanczos_seeded`) runs only with the
   Boys-screened $\chi_0$, from a Boys-orbital seed. `Eigensolver::Davidson` is
   iterative, from an atom-localized seed when `trunc_thresh > 0` and
   $N_{\rm aux} > 4N_{\rm at}$ and from the identity otherwise, and is the only
   solver that reads `eigensolver_conv_thresh` and `eigensolver_max_vecs`.
   The modes come ordered by $|\lambda_\xi(0) - 1|$, descending, and those
   with $|\lambda_\xi(0) - 1| > \texttt{trunc\_thresh}$ are kept
   (`PdepRpaConfig::n_kept_modes`, default threshold $10^{-4}$, at least one
   mode); `trunc_thresh` $\le 0$ keeps all of them. The physical auxiliary
   coefficients of a mode are $c_\xi = M^{\mathsf T}\mathbf U_\xi$, which
   satisfy $c_\xi^{\mathsf T}Vc_\eta = \delta_{\xi\eta}$.

2. **Response matvec.** The Sternheimer equation
   $(\hat H_0 - \epsilon_i + iu)|\Delta\Psi_i\rangle = -P_{\rm virt}\hat V|\Psi_i\rangle$
   is the formulation; ferric's RI-MO implementation sums over every
   occupied-virtual pair instead. With $\mathbf y = \mathbf U^{\mathsf T}B$
   ($m \times n_{ov}$, frequency independent),
   $\mathbf U^{\mathsf T}\tilde{\boldsymbol\epsilon}(iu)\mathbf U = \mathbf 1 + 4\,\mathbf y\,\operatorname{diag}(\Delta_i^a/(u^2 + (\Delta_i^a)^2))\,\mathbf y^{\mathsf T}$
   (`dielectric_matrix_from_projection_into`).
   `sternheimer::chi_from_trial_potential` is a test helper that returns the
   single-spin value $-2\sum_{ia} \Delta_i^a/(u^2 + (\Delta_i^a)^2)\,(\sum_P V_PB^P_{ia})^2$,
   half the closed-shell $\langle V|\tilde\chi_0|V\rangle$.

3. **Frequency-dependent eigenvalues.** The static basis does not diagonalize
   $\tilde{\boldsymbol\epsilon}(iu)$. At each node the projected matrix
   $\mathbf U^{\mathsf T}\tilde{\boldsymbol\epsilon}(iu_k)\mathbf U$ is diagonalized
   (`eval_eigenvalues_at_frequencies_budgeted`, `eigvalsh_dc`), and
   $\lambda_\xi(iu_k)$ are its eigenvalues, not its diagonal
   $\langle\mathbf U_\xi|\tilde{\boldsymbol\epsilon}(iu_k)|\mathbf U_\xi\rangle$.
   The diagonal alone would drop the off-diagonal coupling (2.8e-2 Ha on the
   test model). When the caller asks for neither the per-frequency eigenvalues
   nor the inverse dielectric (`need_eigenvalues_freq`,
   `need_inv_dielectric_freq` both false) and the dense $\chi_0$ is in use,
   the projected matrix is not diagonalized: its LU log-determinant gives the
   summand directly (`eval_trace_log_summands_budgeted`).

4. **Energy.**
   $$ E_c = \frac{1}{2\pi}\sum_k w_k\sum_\xi\big[\ln\lambda_\xi(iu_k) + 1 - \lambda_\xi(iu_k)\big]
   = \frac{1}{2\pi}\sum_k w_k\big[\ln\det\tilde{\boldsymbol\epsilon}_U(iu_k) + \operatorname{tr}(\mathbf 1 - \tilde{\boldsymbol\epsilon}_U(iu_k))\big], $$
   $\tilde{\boldsymbol\epsilon}_U = \mathbf U^{\mathsf T}\tilde{\boldsymbol\epsilon}\mathbf U$
   (`rpa_correlation_energy`, `rpa_correlation_energy_from_summands`). Both
   terms are basis-invariant traces, so with every mode kept ($\mathbf U$
   orthogonal, $m = N_{\rm aux}$) this is the aux-basis log-det energy of the
   previous section exactly. With truncation the energy is that of the
   projected dielectric and converges to the full one as `trunc_thresh` goes
   to 0.

**Routes.** `energy_route::run_pdep_rpa_energy` skips the static eigensolve
and assembles $\tilde{\boldsymbol\epsilon}(iu_k)$ in the auxiliary basis
itself (`RpaEnergyRoute::AuxBasisLogDet`) when the caller wants the energy
only, `trunc_thresh` $\le 0$, the $\chi_0$ is dense and not Laplace, and no
per-frequency eigenvalues or inverse dielectric are requested
(`static_route_required`); otherwise it calls `run_pdep_rpa`
(`RpaEnergyRoute::StaticEigenbasis`). The CLI `drpa` kind takes the first
route, a `pdep-rpa` run with the default `trunc_thresh` the second. With the
Boys-screened $\chi_0$ only the static eigensolve uses the screened tiles;
the frequency loop uses the dense $B$.

The PDEP basis avoids forming and diagonalizing the full
$N_{\rm aux}\times N_{\rm aux}$ dielectric at every frequency when few modes
are kept.

## Open-Shell (Spin-Summed) Dielectric

`run_u_pdep_rpa` (`crates/ferric-rpa/src/lib.rs`) takes a UHF reference, or a ROHF reference semi-canonicalized per spin. Each spin $\sigma$ has its own RI tensor $B^P_{i_\sigma a_\sigma}$ and gaps $\Delta_{i_\sigma}^{a_\sigma} = \epsilon_{a_\sigma} - \epsilon_{i_\sigma}$ in one shared aux basis, built with one dressing $M$, and the symmetrized dielectric is

$$ \tilde{\mathbf{\epsilon}}_U(iu) = \mathbf{1} + \mathbf{\Pi}_\alpha(iu) + \mathbf{\Pi}_\beta(iu), \qquad \Pi^\sigma_{PQ}(iu) = 2 \sum_{ia} \frac{\Delta^{a_\sigma}_{i_\sigma}}{u^2 + (\Delta^{a_\sigma}_{i_\sigma})^2} B^P_{i_\sigma a_\sigma} B^Q_{i_\sigma a_\sigma}. $$

The closed-shell $\mathbf{\Pi}$ carries the prefactor 4; with identical spin channels $\mathbf{\Pi}_\alpha + \mathbf{\Pi}_\beta$ equals it, so a UHF singlet reproduces `run_pdep_rpa`. With different $\alpha$ and $\beta$ orbitals, gaps and occupations, $\frac{1}{2\pi}\int[\ln\det\tilde{\boldsymbol\epsilon}_U - \operatorname{tr}(\boldsymbol\Pi_\alpha + \boldsymbol\Pi_\beta)]du$ equals the spin-orbital definition of the "RPA Correlation Energy" section, $\tfrac12\sum_n(\Omega_n - A_{nn})$ with $\mathbf A = \boldsymbol\Delta + \mathbf K$, $\mathbf B = \mathbf K$ over both spin blocks, $K_{i_\sigma a_\sigma, j_\tau b_\tau} = (i_\sigma a_\sigma|j_\tau b_\tau)$ (`scripts/rpa_drpa_pdep_sympy.py`). The per-spin prefactor 2 is the $\pm u$ pole pair alone (`build_scale_factors_with_prefactor(.., 2.0)`).

The open-shell driver (`u_pdep_rpa_core`) always takes the static-eigenbasis route. The static eigensolve is the dense full-rank `eigh` of $\tilde{\mathbf{\epsilon}}_U(0)$ (default `Eigensolver::Lanczos`) or Davidson from the identity seed; the matvec $\tilde{\mathbf{\epsilon}}_U \mathbf{V} = \mathbf{V} + \mathbf{\Pi}_\alpha \mathbf{V} + \mathbf{\Pi}_\beta \mathbf{V}$ is two closed-shell-shaped applications (`sternheimer::dielectric_apply_unrestricted`). Truncation follows the closed-shell rule. The frequency loop then always diagonalizes the projected $\mathbf U^{\mathsf T}\tilde{\mathbf{\epsilon}}_U(iu_k)\mathbf U$ (`eval_eigenvalues_at_frequencies_unrestricted`) and sums $\ln\lambda + 1 - \lambda$ over the eigenvalues: there is no aux-basis log-det route and no LU log-det branch on the open-shell path. The Laplace $\chi_0$ backend builds one Laplace quadrature per spin on that spin's gap range. `run_diagnostics` assembles $\tilde{\mathbf{\epsilon}}_U(iu_k)$ densely at every quadrature node (`diagnostics::u_ri_drpa_energy`). `run_u_pdep_rpa_from_parts` (pre-built intermediates) accepts only full rank, the dense $\chi_0$ and `Eigensolver::Lanczos`, and refuses $\alpha$ and $\beta$ intermediates with different dressings.

The Boys-screened $\chi_0$ is closed-shell only: its screened tiles and localized orbital energies are built from the restricted MOs and Fock matrix, and `run_u_pdep_rpa` refuses it.

## Dispersion: Casimir-Polder C6

The direct-RPA response that screens the correlation energy also gives the
molecular and per-atom polarizabilities at imaginary frequency, and from them
the $C_6$ coefficients. Throughout, $g_{ia}(u) = \Delta_i^a/(u^2 + (\Delta_i^a)^2)$,
$G = \operatorname{diag}(g_{ia})$, $\mu_{d,ia} = \langle i|r_d|a\rangle$ ($d = x, y, z$,
about the origin), and closed shell unless stated.

**Molecular polarizability, dielectric form** (`properties::pdep_polarizability_static`,
`properties::molecular_dynamic_polarizability`). In singlet direct RPA
$\mathbf A - \mathbf B = \boldsymbol\Delta$ and $\mathbf A + \mathbf B = \boldsymbol\Delta + 4\mathbf K$,
$\mathbf K = B^{\mathsf T}B$, so the response is

$$ \alpha_{de}(iu) = 4\,\mu_d^{\mathsf T}\big[\boldsymbol\Delta(\boldsymbol\Delta + 4\mathbf K) + u^2\big]^{-1}\boldsymbol\Delta\,\mu_e . $$

The prefactor 4 is 2 for spin and 2 for the pair of poles. The Woodbury
identity turns the $n_on_v$-sized inverse into one of size $N_{\rm aux}$,
which is what the code solves:

$$ \alpha_{de}(iu) = 4\,\mu_d^{\mathsf T}G\,\mu_e - 16\,w_d^{\mathsf T}\,\tilde{\boldsymbol\epsilon}(iu)^{-1}\,w_e, \qquad w_d = B\,(g\circ\mu_d), \qquad \tilde{\boldsymbol\epsilon}(iu) = \mathbf 1 + 4BGB^{\mathsf T} $$

(`tensor[i][j] = 4.0 * bare - 16.0 * coupled`, with `coupled` from one SPD
solve `solve_dielectric_3` for the three right-hand sides). The 16 is the
outer 4 times the 4 of the dielectric. At $u = 0$, $g_{ia} = 1/\Delta_i^a$
and this is the static polarizability; the dynamic driver evaluates the same
expression at each quadrature node. The tensor is then symmetrized. The
dipole is the molecular dipole on both sides, so the tensor does not depend
on the origin.

**Mode form** (`response_modes::response_modes`,
`ResponseModes::polarizability_from_modes`). With
$\tilde{\boldsymbol\epsilon}(iu) = \sum_\xi\lambda_\xi\mathbf U_\xi\mathbf U_\xi^{\mathsf T}$
(a dense eigensolve at the requested $u$),

$$ \alpha(iu) = 4\,\mu^{\mathsf T}G\,\mu - \sum_\xi\frac{s_\xi s_\xi^{\mathsf T}}{\lambda_\xi}, \qquad s_{\xi,d} = 4\,\mathbf U_\xi\cdot w_d . $$

With every mode kept this is the dielectric form exactly; keeping the
$n_{\rm modes}$ largest-$\lambda_\xi$ modes truncates the screening sum. At
$u = 0$ the modes are the PDEP eigenpotentials, with physical coefficients
$c_\xi = M^{\mathsf T}\mathbf U_\xi$, $(V + \Pi)c_\xi = \lambda_\xi Vc_\xi$.
`response_modes` is closed shell only.

**Open shell** (the unrestricted branches of `pdep_polarizability_static_unrestricted`
and `molecular_dynamic_polarizability`; ROHF/ROKS semi-canonicalized per
spin). The screening is shared and the bare terms are per spin:

$$ \tilde{\boldsymbol\epsilon}_U = \mathbf 1 + 2\sum_\sigma B_\sigma G_\sigma B_\sigma^{\mathsf T}, \qquad
\alpha_{de} = \sum_\sigma 2\,\mu_{\sigma,d}^{\mathsf T}G_\sigma\,\mu_{\sigma,e} - 4\,w_d^{\mathsf T}\tilde{\boldsymbol\epsilon}_U^{-1}w_e, \qquad
w_d = \sum_\sigma B_\sigma(g_\sigma\circ\mu_{\sigma,d}) , $$

the closed-shell form when the spins coincide ($w$ doubles, so
$4\,(2w_\sigma)^{\mathsf T}\cdots(2w_\sigma) = 16\,w_\sigma^{\mathsf T}\cdots w_\sigma$, and
$2 + 2 = 4$ in the bare term). It equals the direct solve of the
spin-orbital response $2\mu_{\rm so}^{\mathsf T}[\boldsymbol\Delta(\boldsymbol\Delta + 2\mathbf K_{\rm so}) + u^2]^{-1}\boldsymbol\Delta\mu_{\rm so}$
for unequal $\alpha$ and $\beta$ spaces.

`scripts/rpa_response_modes_sympy.py` checks these three forms: the
dielectric form against the direct solve at $u = 0$ and $0.9$ (claim M1),
the mode form at full rank and its truncation (M2), the static limit (M3),
and the open-shell form against the spin-orbital solve (M4).

**$C_6$.** `dispersion::casimir_polder_c6` turns per-atom dynamic polarizabilities on the
imaginary-frequency quadrature $(u_k, w_k)$ into pair coefficients:

$$ C_6^{AB} = \frac{3}{\pi}\int_0^\infty \bar\alpha^A(iu)\,\bar\alpha^B(iu)\,du \approx \frac{3}{\pi}\sum_k w_k\,\bar\alpha^A(iu_k)\,\bar\alpha^B(iu_k),\qquad \bar\alpha = \tfrac13\operatorname{Tr}\alpha, $$

and, element by element (no tensor contraction), the anisotropic
$C^{AB}_{6,ij} = \frac{3}{\pi}\sum_k w_k\,\alpha^A_{ij}(iu_k)\,\alpha^B_{ij}(iu_k)$.
Two sources feed it.

**Tkatchenko-Scheffler single pole** (`ts_dynamic_polarizability`,
`mbd::ts_atom_params`). With the Hirshfeld volume ratio $r_A$ and the
free-atom $\alpha^{\rm free}$, $C_6^{\rm free}$ (TS PRL 2009 Table I for
$Z \le 18$; Gould and Bučko 2016 for $Z = 19$–$54$, Pd α from Jerabek et al. 2018; other $Z$ is an error):

$$ \alpha_A = r_A\,\alpha^{\rm free},\quad C_{6,A} = r_A^2\,C_6^{\rm free},\quad \bar\alpha_A(iu) = \frac{\alpha_A}{1+(u/u_A)^2},\quad u_A = \frac{4}{3}\frac{C_{6,A}}{\alpha_A^2}. $$

$u_A$ follows from the single-pole integral
$\frac{3}{\pi}\int_0^\infty \alpha_A^2(1+u^2/u_A^2)^{-2}\,du = \frac34\alpha_A^2u_A = C_{6,A}$.
The tensor is $\bar\alpha_A(iu)$ times the static tensor normalized to
unit isotropic average, so only the shape comes from the molecular calculation.

**PDEP-RPA** (`pdep_dynamic_polarizability`,
`properties::pdep_polarizability_becke_dynamic` / `_hirshfeld_dynamic`). The
per-atom tensor is the Krishtal-Senet-Van Alsenoy intrinsic polarizability
(JCP 125, 034312 (2006)),
$\alpha^A_{dj}(iu) = \langle m^A_d|R(iu)|\mu_j\rangle$, with the
atom-weighted dipole $m^A_d = w_A(\mathbf r)(r-R_A)_d$ on the bra and the
molecular dipole on the ket. $R(iu)$ is the RI response with
$g_{ia}(u)$ and the screened dielectric
$\tilde{\boldsymbol\epsilon}(iu) = \mathbf 1 + 4BGB^{\mathsf T}$
(closed shell). At $u = 0$, $g_{ia} = 1/\Delta_i^a$ and the static per-atom
tensor is recovered. The molecular tensor $\alpha_{\rm mol} = \langle\mu|R|\mu\rangle$
is not the atom sum: $\alpha_{\rm CT} = \alpha_{\rm mol} - \sum_A\alpha^A$ is the
field-induced charge transfer between atoms (`charge_transfer_remainder`,
`_dynamic`).

**Per-atom kernel** (`properties/per_atom.rs`, `closed_per_atom_alpha`). With
the ov vectors $m^A_{d,ia} = \langle i|m^A_d|a\rangle$, the closed-shell dRPA
response is $R = (G^{-1}/4 + \mathbf K)^{-1}$, which Woodbury turns into the
aux-basis form the code evaluates (the molecular form above with the bra
$\mu_d$ replaced by $m^A_d$):

$$ \alpha^A_{dj}(iu) = 4\,m^A_d\cdot(g\circ\mu_j) - 16\,\big(B(g\circ m^A_d)\big)^{\mathsf T}\tilde{\boldsymbol\epsilon}^{-1}\big(B(g\circ\mu_j)\big). $$

The same $R$ is the sum over the dRPA (Casida) excitations,
$\alpha(iu) = \sum_n f_n/(\Omega_n^2 + u^2)$ with
$\Omega_n^2\mathbf Z_n = \boldsymbol\Delta^{1/2}(\boldsymbol\Delta + 4\mathbf K)\boldsymbol\Delta^{1/2}\mathbf Z_n$ and
$f_n = 4\,(m^{\mathsf T}\boldsymbol\Delta^{1/2}\mathbf Z_n)(\mathbf Z_n^{\mathsf T}\boldsymbol\Delta^{1/2}\mu)$.
The static drivers use $g_{ia} = 1/\Delta_i^a$, the $u \to 0$ limit. The code
then symmetrizes each tensor, $\alpha^A \to \tfrac12(\alpha^A + \alpha^{A\mathsf T})$
(`symmetrize_tensor`). For partition weights with $\sum_Aw_A = 1$ the bras
sum to $\sum_A m^A_d = \mu_d - \sum_A R_{A,d}\,w_A$ (and $\sum_A\langle i|w_A|a\rangle = 0$), so by linearity
$\alpha_{\rm CT} = \alpha_{\rm mol} - \sum_A\operatorname{sym}\alpha^A = \operatorname{sym}\langle\sum_AR_Aw_A|R|\mu\rangle$.

**Quadrature.** The C6 sums run on the RPA frequency grid of
`quadrature.rs` (default Gauss-Legendre, 20 points, $u_0 = 0.5$). For the
single-pole model the closed form is
$\frac{3}{\pi}\int_0^\infty\bar\alpha_A\bar\alpha_B\,du = \frac32\,\frac{\alpha_A\alpha_Bu_Au_B}{u_A + u_B}$
(London), and the 20-point sum reproduces it to $3\times10^{-14}$ relative
for $u_{A,B} \approx 0.43\ E_h$ in the test model; the mapped rule converges geometrically on this rational integrand. For a pole expansion
$\alpha(iu) = \sum_n f_n/(\Omega_n^2 + u^2)$,
$C_6^{AB} = \frac32\sum_{mn}f^A_mf^B_n/(\Omega_m\Omega_n(\Omega_m + \Omega_n))$.
`scripts/rpa_c6_polarizability_sympy.py` checks these identities, the kernel
against the Casida sum and the charge-transfer sum rule.

## Range-separated MP2 + RPA

`rs_mp2_rpa::rs_mp2_lr_rpa` combines RI-MP2 and dRPA with the Coulomb
operator split as $1/r = v_{\rm sr} + v_{\rm lr}$ ($\operatorname{erfc}/\operatorname{erf}$
at $\omega$, or terfc/terf at $r_0$), so $(ia|jb)_{\rm C} = (ia|jb)_{\rm sr} + (ia|jb)_{\rm lr}$.
With the spin components of `ferric_mp2::rimp2::pair_energy`,

$$ E_{\rm OS}[v] = -\sum_{ijab}\frac{(ia|jb)_v^2}{\Delta_{ij}^{ab}}, \qquad
E_{\rm SS}[v] = -\sum_{ijab}\frac{(ia|jb)_v[(ia|jb)_v - (ib|ja)_v]}{\Delta_{ij}^{ab}}, \qquad
E_{\rm MP2}[v] = E_{\rm OS} + E_{\rm SS}, $$

($E_{\rm OS} + E_{\rm SS}$ is the spin-orbital
$-\frac14\sum|\langle ij||ab\rangle|^2/\Delta_{ij}^{ab}$), the two formulations are

$$ E^{\rm B}_c = E_{\rm MP2}[\mathrm C] + \big(E_{\rm dRPA}[\mathrm{lr}] - 2E_{\rm OS}[\mathrm{lr}]\big) \quad (\texttt{DeltaLr}, \text{default}), $$

$$ E^{\rm T}_c = E_{\rm MP2}[\mathrm C] + \big(E_{\rm dRPA}[\mathrm C] - 2E_{\rm OS}[\mathrm C]\big) - \big(E_{\rm dRPA}[\mathrm{sr}] - 2E_{\rm OS}[\mathrm{sr}]\big) \quad (\texttt{CoupledRings}). $$

Each $E_{\rm dRPA}$ is a `run_pdep_rpa_from_intermediates` call (full rank by
default, `RsMp2RpaConfig::default`) on the same dressed $B$ the MP2
components use. The subtraction
$2E_{\rm OS}$ is exactly the second-order term of the dRPA it accompanies
(section "RPA Correlation Energy": $E^{(2)}_{\rm dRPA} = 2E_{\rm OS}$; on the
$n$-point frequency grid it holds to the quadrature error), so every bracket
starts at third order and both formulations equal $E_{\rm MP2}[\mathrm C]$ through second
order, exchange included: nothing is counted twice. Scaling every integral by
$\lambda$, the third-order terms are

$$ E^{\rm B\,(3)}_c = E^{(3)}_{\rm dRPA}[\mathrm{lr}], \qquad E^{\rm T\,(3)}_c = E^{(3)}_{\rm dRPA}[\mathrm C] - E^{(3)}_{\rm dRPA}[\mathrm{sr}], $$

with $E^{(3)}_{\rm dRPA} = \tfrac12\sum\mathbf B\circ\mathbf T_2$,
$\bar\Delta\circ\mathbf T_2 = -(\mathbf B\mathbf T_1 + \mathbf T_1\mathbf B)$. In one mode
($E_{\rm dRPA}(b) = -b^2/(4\Delta) + b^3/(4\Delta^2) + O(b^4)$, $b = b_s + b_l$) the
difference is $E^{\rm T\,(3)} - E^{\rm B\,(3)} = 3b_sb_l(b_s + b_l)/(4\Delta^2)$: the
mixed short-range by long-range rings that B drops and T keeps. Both forms
reduce to $E_{\rm MP2}[\mathrm C]$ when $v_{\rm lr} = 0$ ($\omega \to 0$) and
to $E_{\rm MP2}[\mathrm C] + E_{\rm dRPA}[\mathrm C] - 2E_{\rm OS}[\mathrm C]$
when $v_{\rm sr} = 0$. `scripts/rpa_rs_mp2_rpa_sympy.py` checks these
statements, the ring orders against Cauchy-integral Taylor coefficients of the
plasmon energy, and the spin components against spin-orbital MP2.

## Analytic nuclear gradient (closed-shell RI-dRPA, HF reference)

`gradient_analytic.rs` differentiates the energy `run_pdep_rpa` evaluates at
full rank (`trunc_thresh = 0`), on the fixed $n$-point imaginary-frequency
grid $(u_k, w_k)$ of `quadrature.rs`. Every scheme there (Gauss-Legendre,
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

With $B^P_{ia} = \sum_Q M_{PQ}(Q|ia)$ (the RI factor, $M^{\mathsf T}M = V^{-1}$, $V_{PQ} = (P|Q)$; the code's $M$ is the Cholesky $L^{-1}$ of the Coulomb metric),
the closed-shell dielectric of the Open-Shell section and $\Pi = \tilde\epsilon - \mathbf 1$,

$$ E_c = \frac{1}{2\pi}\sum_k w_k\, S(\Pi_k), \qquad S(\Pi) = \ln\det(\mathbf 1 + \Pi) - \operatorname{tr}\Pi,
\qquad \Pi_k = B\, g_k(\Delta)\, B^{\mathsf T}, \qquad g_k(e) = \frac{4e}{u_k^2 + e^2}. $$

At full rank $S$ is basis-invariant, so the eigenvalue and log-det routes of
`energy.rs` are this expression. $\Delta$ is the occupied-virtual
orbital-energy-difference operator,
$\Delta_{ia,jb} = F_{ab}\delta_{ij} - F_{ij}\delta_{ab}$, which is
$\operatorname{diag}(\Delta_i^a)$, $\Delta_i^a = \varepsilon_a - \varepsilon_i$, in
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

$$ \Gamma^P_{ia} = \frac{1}{\pi}\sum_k w_k\, g_k(\Delta_i^a)\,(Q_kB)^P_{ia}. $$

**Occupied-occupied and virtual-virtual densities.** With
$N_k = B^{\mathsf T}Q_kB$ (nov x nov), $\delta E_c|_B = \frac{1}{2\pi}\sum_k w_k
\operatorname{tr}[N_k\,\delta g_k(\Delta)]$. The derivative of the matrix
function $g_k(\Delta)$ in the eigenbasis of $\Delta$ (Daleckii-Krein) is
$\delta g_k(\Delta)_{ia,jb} = g_k^{[1]}(\Delta_i^a, \Delta_j^b)\,\delta\Delta_{ia,jb}$ with
the divided difference

$$ g^{[1]}(x, y) = \frac{g(x) - g(y)}{x - y} = \frac{4(u^2 - xy)}{(u^2 + x^2)(u^2 + y^2)}, $$

which is smooth through $x = y$ (it is $g'(x)$ there), so degenerate or
near-degenerate orbitals need no special case. Inserting
$\delta\Delta_{ia,jb} = \delta_{ij}\delta F_{ab} - \delta_{ab}\delta F_{ij}$:

$$ P_{ab} = \frac{1}{2\pi}\sum_k w_k \sum_i (N_k)_{ia,ib}\, g_k^{[1]}(\Delta_i^a, \Delta_i^b), \qquad
P_{ij} = -\frac{1}{2\pi}\sum_k w_k \sum_a (N_k)_{ia,ja}\, g_k^{[1]}(\Delta_i^a, \Delta_j^a). $$

Both are symmetric. Their diagonals are $\partial E_c/\partial\varepsilon_a$ and
$\partial E_c/\partial\varepsilon_i$ of the canonical expression; the
off-diagonal elements are what the canonical expression alone cannot
supply. Only the blocks $(N_k)_{ia,ib}$ and $(N_k)_{ia,ja}$ are needed:
per frequency, $W_k = Q_kB$ (naux$^2$ nov) and then one
$(n_v \times n_{aux})(n_{aux} \times n_v)$ product per occupied orbital and one
$(n_o \times n_{aux})(n_{aux} \times n_o)$ per virtual, the RI-MP2 density cost
times $n$.

**The second-order term.** Replacing $S$ by its leading term
$-\tfrac12\operatorname{tr}\Pi^2$ gives $Q_k = -\Pi_k$ and, as $n \to \infty$
($\int_0^\infty \frac{xy}{(u^2+x^2)(u^2+y^2)}du = \frac{\pi}{2(x+y)}$),
the direct-ring MP2 energy $E^{(2)}_{\mathrm{dr}} = -2\sum_{iajb}(ia|jb)^2/\Delta_{ij}^{ab}$:
twice the
opposite-spin MP2 energy, i.e. SCS-MP2 with $(c_{\mathrm{OS}}, c_{\mathrm{SS}}) = (2, 0)$.
In that limit $\Gamma$, $P_{ij}$, $P_{ab}$ become
$8\sum_{jb}t_{ij}^{ab}B^P_{jb}$, $-4\sum_{kab}t_{ik}^{ab}t_{jk}^{ab}$ and
$4\sum_{ijc}t_{ij}^{ac}t_{ij}^{bc}$ with $t_{ij}^{ab} = -(ia|jb)/\Delta_{ij}^{ab}$, the
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

$\tilde\Gamma^Q_{ia} = \sum_P M_{PQ}\Gamma^P_{ia}$ ($\tilde\Gamma = M^{\mathsf T}\Gamma$) back-transformed to
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

## Range-separated MP2 + RPA: which terms depend on $\omega$

`rs_mp2_lr_rpa` (formulations B and T) splits the Coulomb kernel as
$v = v^{\mathrm{lr}}_\omega + v^{\mathrm{sr}}_\omega$ with $v^{\mathrm{lr}}_\omega = \mathrm{erf}(\omega r)/r$
(or the tempered $\mathrm{terf}$ pair). With the closed-shell spin components
$E_{\mathrm{OS}}[v]$ and the direct-ring series $E_{\mathrm{dRPA}}[v]$ of a kernel $v$,

$$E_c^{B} = E_{\mathrm{MP2}}[v] + \big(E_{\mathrm{dRPA}}[v^{\mathrm{lr}}_\omega] - 2E_{\mathrm{OS}}[v^{\mathrm{lr}}_\omega]\big),$$

$$E_c^{T} = E_{\mathrm{MP2}}[v] + \big(E_{\mathrm{dRPA}}[v] - 2E_{\mathrm{OS}}[v]\big) - \big(E_{\mathrm{dRPA}}[v^{\mathrm{sr}}_\omega] - 2E_{\mathrm{OS}}[v^{\mathrm{sr}}_\omega]\big).$$

Every term is a functional of ONE kernel: the dressed intermediate
$B^{v}_{P,ia} = \sum_Q [V^{v}]^{-1/2}_{PQ}(Q|v|ia)$ built from that kernel's
3-centre integrals and 2-centre metric, and the fixed reference orbitals and
energies. No term mixes kernels, so a term depends on $\omega$ exactly when
its kernel does:

- independent of $\omega$: the reference SCF (full-Coulomb RHF), $B^{v}$,
  $E_{\mathrm{MP2}}[v]$, $E_{\mathrm{OS}}[v]$, and for T the full-Coulomb
  $E_{\mathrm{dRPA}}[v]$ (the most expensive single piece);
- dependent on $\omega$: $B^{\mathrm{lr}}_\omega$ and $B^{\mathrm{sr}}_\omega$
  with their spin components, B's $E_{\mathrm{dRPA}}[v^{\mathrm{lr}}_\omega]$ and
  T's $E_{\mathrm{dRPA}}[v^{\mathrm{sr}}_\omega]$.

B and T at the same $\omega$ use the same $B^{\mathrm{lr}}_\omega$ and
$B^{\mathrm{sr}}_\omega$, so a scan over $\omega$ and formulations builds each
once. The combination formulas above are unchanged; only the order of
evaluation is. Each kernel's intermediate is dropped before the next is built,
so the co-resident set is one call's worth for any number of $\omega$ points.
No new equation is introduced, so there is no new sympy proof: the evidence is
that every scan entry equals the single-$\omega$ call bit for bit
(`crates/ferric-rpa/tests/rs_mp2_rpa_scan.rs`).

## Literature Citations

1. **PDEP Eigenpotentials:** F. Gygi and G. Galli, *Phys. Rev. B* **65**, 220102(R) (2002).
2. **PDEP-RPA Correlation Energy:** D. Rocca, D. Lu, and G. Galli, *J. Chem. Phys.* **133**, 164109 (2010).
3. **MiniMax Quadrature:** A. Takatsuka, S. Ten-no, and W. Hackbusch, *J. Chem. Phys.* **129**, 044112 (2008).
4. **RI-RPA Review:** H. Eshuis, J. E. Bates, and F. Furche, *Theor. Chem. Acc.* **131**, 1084 (2012).
