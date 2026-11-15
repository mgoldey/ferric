# GW, COHSEX, BSE and RPAx Theory

This crate computes quasiparticle (QP) energies in the GW approximation
(G0W0, evGW₀, evGW, static COHSEX; closed and open shell) on a
resolution-of-identity (RI) screened interaction taken from `ferric-rpa`'s
PDEP dielectric eigenbasis, plus BSE-TDA excitation energies and the
statically screened TDHF/RPAx polarizability. Every equation below states the
convention the code uses and points at the function that implements it
(`path:line` on the commit that added this file) and the test that checks it.

Identities marked **[sympy n]** are proven or checked to ≥30 digits by claim
*n* of [`scripts/gw_self_energy_sympy.py`](../../scripts/gw_self_energy_sympy.py),
which exits nonzero on any mismatch and carries a negative control per claim.

Units are Hartree atomic units. Indices: $i,j$ occupied, $a,b$ virtual,
$p,q,r,s$ any active MO, $P,Q$ RI auxiliary functions, $\xi,\eta$ PDEP
modes. $u$ is an imaginary frequency and $E$ a real energy argument. "Active" means MO indices `frozen_core..nmo`.

## 1. RI factorization and exchange

With the Coulomb metric $V_{PQ} = (P|Q)$ and $M$ a factor with
$M^{\mathsf T} M = V^{-1}$ (Cholesky $L^{-1}$, or the symmetric eigh
$V^{-1/2}$ for attenuated operators), the dressed three-index tensor is

$$ B^P_{pq} = \sum_Q M_{PQ}\,(Q|pq), \qquad (pq|rs) \approx \sum_P B^P_{pq} B^P_{rs}, $$

held over the full active square, shape `(naux, n_act, n_act)`
(`mo_b.rs:206` `build_full_b`; per spin `mo_b.rs:243`, `:406`). The
exchange self-energy diagonal is the closed-shell HF exchange in this basis,

$$ \Sigma^x_p = -\sum_i \sum_P \left(B^P_{pi}\right)^2 $$

(`cohsex.rs:111` `sigma_x_diag`; per spin the same formula on that spin's
$B_\sigma$). It is computed from the full $B$, never from the
PDEP projection, so PDEP rank truncation affects only correlation. Checked
against PySCF's density-fitted exchange to ≤4.9e-9 Ha
(`tests/validation_gw.rs`, the Σx anchor).

## 2. Screened interaction on the imaginary axis

In the dressed basis the bare interaction is the identity. The closed-shell
independent-particle response and dielectric matrix are

$$ \Pi_{PQ}(iu) = 4 \sum_{ia} B^P_{ia}B^Q_{ia}\,\frac{\Delta_i^a}{u^2 + (\Delta_i^a)^2}, \qquad \tilde\varepsilon(iu) = \mathbf 1 + \Pi(iu), \qquad \Delta_i^a = \epsilon_a - \epsilon_i, $$

prefactor 4 = 2 (spin) × 2 (resonant + antiresonant)
(`ferric-rpa/src/sternheimer.rs:204` `build_scale_factors`, `:400`
`dielectric_matrix_from_projection`). The full screened interaction is
$\tilde W = \tilde\varepsilon^{-1}$; its static bare part gives $\Sigma^x$ and
the code carries only the dynamic part

$$ \tilde W_d(iu) = \tilde\varepsilon^{-1}(iu) - \mathbf 1 . $$

**PDEP basis.** The static matrix is diagonalized,
$\tilde\varepsilon(0)\,U_\xi = \lambda_\xi(0)\,U_\xi$, and the GW code
works in the fixed eigenbasis $\{U_\xi\}$. $\tilde W_d$ is built as the
**full** $N_\text{pdep}\times N_\text{pdep}$ matrix $[U^{\mathsf T}\tilde\varepsilon(iu_k)U]^{-1} - \mathbf 1$
at every quadrature node (`ferric-rpa/src/energy.rs:566`
`eval_inv_dielectric_matrices`; `PdepRpaResult.inv_dielectric_freq`). The
per-frequency scalar eigenvalues $\lambda_\xi(iu)$ live in an
$u$-dependent rotated eigenbasis and are valid only for basis-invariant
traces such as the RPA energy; the diagonal form $\sum_\xi(1/\lambda_\xi(iu)-1)$
is not used for $\Sigma_c$. At $u = 0$ the matrix is diagonal,
$w_\xi \equiv 1/\lambda_\xi(0) - 1$ (`w_pdep.rs:157` `static_weights`).

**Redressing.** `PdepRpaResult.eigenpotentials` are physical aux
coefficients $c = M_{\rm pdep}^{\mathsf T} U$. The dressed vectors matching
this crate's $M$ solve $M^{\mathsf T} \hat U = c$ (`w_pdep.rs:39`
`redress_eigenpotentials`; triangular solve for Cholesky, pseudo-inverse for
symmetric $M$, anything else an error). Tests:
`tests/redress_recovers_dressed_eigenvectors.rs`.

**Projection.** All self-energy and kernel contractions use

$$ B^\xi_{pq} = \sum_P \hat U^P_\xi\, B^P_{pq} $$

(`cohsex.rs:228` `project_b_into_pdep`, one GEMM at `cohsex.rs:291`). For a
complete orthonormal $\hat U$, $\sum_\xi B^\xi_{pq}B^\xi_{rs} = (pq|rs)$
**[sympy 8]**; a truncated $\hat U$ does not satisfy it.

The screened matrix element at node $k$ is the quadratic form

$$ W_{pq}(iu_k) = \sum_{\xi\eta} B^\xi_{pq}\,[\tilde W_d(iu_k)]_{\xi\eta}\,B^\eta_{pq} $$

(`sigma.rs:310`–`315`, one GEMM per node).

## 3. Single-pole structure

For one $(ia)$ pair with $B^P_{ia} = b$, $\Delta_i^a = e$:

$$ \tilde W_d(iu) = -\frac{4b^2 e}{u^2 + e^2 + 4b^2 e} = -\frac{2\Omega s}{u^2 + \Omega^2}, \qquad \Omega^2 = e(e + 4b^2) = (\mathbf A-\mathbf B)(\mathbf A+\mathbf B), \quad s = \frac{2b^2e}{\Omega}, $$

with $\mathbf A = e + 2K$, $\mathbf B = 2K$, $K = b^2$ the singlet direct-RPA matrices: the
pole of W is the direct-RPA excitation energy **[sympy 1]**. In general
$\tilde W_d(iu) = -\sum_s 2\Omega_s\,\rho_s\rho_s^{\mathsf T}/(u^2+\Omega_s^2)$,
negative definite on the imaginary axis because $\tilde\varepsilon \ge \mathbf 1$.
This model is the analytic reference for §4 and §5.

## 4. Correlation self-energy on the imaginary axis

The code evaluates, for any complex $z$ off the real axis,

$$ \Sigma^c_p(z) = -\frac{1}{\pi}\sum_q \int_0^\infty du'\; W_{pq}(iu')\,\frac{z - \epsilon_q}{(z - \epsilon_q)^2 + u'^2} $$

(`sigma.rs:274` `sigma_c_at_z`; the $-1/\pi$ at `sigma.rs:329`). The
integrand of the full $\int_{-\infty}^{\infty}$ form is even in $u'$, so
$\frac{1}{2\pi}\cdot 2 = \frac1\pi$. The sum over $q$ runs over every active
MO with no occupied/virtual sign: the time ordering is carried by the sign of
$\mathrm{Re}(z - \epsilon_q)$. $z$ is an absolute energy; `eps_act` (the
propagator energies) is a separate argument so evGW can replace it. The
integral uses the PDEP quadrature (`pdep.quad_freqs`, `pdep.quad_weights`),
scaled Gauss–Legendre $u = u_0(1+x)/(1-x)$
(`ferric-rpa/src/quadrature.rs:84` `gauss_legendre_nodes`).

**Closed form for a single-pole W [sympy 2].** With
$W_{pq}(iu) = -2\Omega s_{pq}/(u^2+\Omega^2)$,

$$ \Sigma^c_p(z) = \sum_{q:\,\mathrm{Re}(z-\epsilon_q)>0} \frac{s_{pq}}{z - \epsilon_q + \Omega} \;+\; \sum_{q:\,\mathrm{Re}(z-\epsilon_q)<0} \frac{s_{pq}}{z - \epsilon_q - \Omega}, $$

the textbook residue result: occupied poles at $\epsilon_i - \Omega$, virtual
poles at $\epsilon_a + \Omega$. On the continuation line $z = \epsilon_F + iu$
the occupied states are exactly those with $\mathrm{Re}(z-\epsilon_q) > 0$.
Proven symbolically for real arguments and checked at complex $z$ to 1e-30.
The Rust test `sigma::tests::sigma_c_single_pole_w_matches_closed_form_residues`
feeds this W through `sigma_c_at_z` on the 100-point grid and asserts the
closed form to 1e-8 Ha at $z = \epsilon_F + i\{0, 0.1, 0.6, 2\}$ (measured
2.3e-9, the quadrature error of that grid).

**Product integration.** On the continuation line
$z = \epsilon_F + iu$ the kernel $a/(a^2+u'^2)$, $a = z - \epsilon_q$,
peaks at $u' = u$ with width $|\epsilon_F - \epsilon_q|$. When a
state sits close to $\epsilon_F$ the plain Gauss–Legendre sum cannot resolve
that ridge; the per-spin mid-gap $\epsilon_{F,\sigma}$ of a UKS radical is the
extreme case (OH/cc-pVDZ@PBE: β HOMO and LUMO 0.015 Ha either side). Against
the grid-free U-dRPA pole sum the plain sum misses Σc at the Padé nodes by
0.11 Ha at 100 points and 0.02 Ha at 400, and the QP energies wander by
0.1–2 eV with $n_{\rm quad}$. The open-shell drivers therefore use
`sigma::sigma_c_at_z_product`: with $u' = u_0(1+x)/(1-x)$, exactly
$a/(a^2+u'^2)\,du' = \tfrac{i}{2}[1/(x-x_1) - 1/(x-x_2)]\,dx$,
$x_1 = (a - iu_0)/(a + iu_0)$, $x_2 = 1/x_1$, and each Cauchy integral is done
exactly for the degree-$(n-1)$ interpolant $\mathcal P$ of $W_{pq}$ through the GL nodes,
$\int_{-1}^{1} \mathcal P/(x-x_0) = \mathcal P(x_0)\ln\frac{1-x_0}{-1-x_0} + \sum_k w_k\frac{\mathcal P_k - \mathcal P(x_0)}{x_k - x_0}$
(the plain rule where $x_0$ is far from $[-1,1]$, $n\ln\rho(x_0) > \ln 10^6$,
whose error there is $\rho^{-2n}$). The only error left is W's interpolation
error. Measured (OH, NH2 / cc-pVDZ@PBE): node error vs the pole sum 2e-8 Ha at
40 points, 2e-13 at 100; frontier QP energies converged to 0.005 meV at 40.
`u0` is recovered from the nodes; a non-GL rule (Chebyshev tan-map) keeps the
plain sum. Tests: `sigma::tests::product_rule_*`; `tests/validation_gw.rs`
`u_g0w0_uks_pbe_*`; `tests/validation_gw_defaults.rs` `u_g0w0_uks_*`. The
closed-shell drivers (G0W0, evGW₀, evGW and the MPI sweep) use the same rule
through `solve_qp_for_mo`. On H2O/cc-pVDZ G0W0@PBE HOMO−2 the plain sum sits
+136, −102 and −9.7 meV from the exact-Σc root at 40, 100 and 200 points; the
product rule is within 0.004 meV from 40 points on.

Further checks: `sigma::tests::sigma_c_blas3_matches_scalar_reference`
(GEMM vs scalar loop); `tests/validation_gw.rs`
`sigma_c_imag_axis_h2o_pbe_and_hf_vs_pyscf`, `sigma_c_imag_axis_n2_pbe_vs_pyscf`
($\Sigma^c(\epsilon_F + iu)$ at the Padé nodes vs PySCF `gw_ac`, ≤8.6e-11 Ha).

## 5. COHSEX: the static limit

For a frequency-independent $W_{pq}$ the integral of §4 is elementary,
$-\frac1\pi\int_0^\infty W\,a/(a^2+u^2)\,du = -\tfrac12 W\,\mathrm{sgn}(a)$,
so for $z$ in the gap

$$ \Sigma^c_p = -\tfrac12\sum_i W_{pi} + \tfrac12\sum_a W_{pa} = \underbrace{-\sum_i W_{pi}}_{\Delta\Sigma^{\rm SEX}_p} + \underbrace{\tfrac12 \sum_q W_{pq}}_{\Sigma^{\rm COH}_p} . $$

The plasmon model of §4 reaches the same limit as $\Omega \to \infty$ at fixed
$W(0) = -2s/\Omega$ **[sympy 3]**. With $W_{pq}(0) = \sum_\xi w_\xi (B^\xi_{pq})^2$:

$$ \Delta\Sigma^{\rm SEX}_p = -\sum_i\sum_\xi w_\xi (B^\xi_{pi})^2, \qquad \Sigma^{\rm COH}_p = \tfrac12\sum_q\sum_\xi w_\xi (B^\xi_{pq})^2, \qquad w_\xi = \frac1{\lambda_\xi(0)} - 1 $$

(`cohsex.rs:311` `cohsex_pieces`). The full screened exchange is
$\Sigma^x + \Delta\Sigma^{\rm SEX} = -\sum_i\sum_\xi \lambda_\xi(0)^{-1} (B^\xi_{pi})^2$
for a complete PDEP basis **[sympy 8]**. COHSEX has no root search:

$$ \epsilon^{\rm QP}_p = \epsilon_p + \Delta\Sigma^{\rm SEX}_p + \Sigma^{\rm COH}_p \;[+\;\Sigma^x_p - v^{xc}_p \text{ at a KS reference}], \qquad Z = 1 $$

(`cohsex.rs:363` `run_cohsex`, assembly at `cohsex.rs:421`–`427`). The Rust
test `sigma::tests::sigma_c_static_w_in_gap_equals_cohsex_pieces` puts a
constant W through `sigma_c_at_z` at $z = \epsilon_F$ and asserts
`cohsex_pieces` to 1e-9 Ha. External checks: `tests/validation_gw.rs`
`cohsex_hf_h2o_n2_vs_numpy` (9.4e-10 Ha), `cohsex_pbe_h2o_n2_vs_numpy`
(1.2e-9 Ha); `tests/cohsex_ks_shift.rs`.

## 6. Analytic continuation

$\Sigma^c_p$ is sampled on the Fermi-shifted imaginary axis
$z_k = \epsilon_F + iu_k$, $\epsilon_F = \tfrac12(\epsilon_{\rm HOMO} + \epsilon_{\rm LUMO})$
of the active mean-field spectrum (`sigma.rs:246` `fermi_level`). The nodes
are 18 indices (or `GwConfig.pade_npts`) chosen from $\{0\}\cup$ GL(100,
$u_0 = 0.5$) with a decreasing step, step ratio 2/3, first index 1
(`sigma.rs:368`–`381`; `sigma.rs:181` `pade_node_indices`, a port of PySCF
`_get_ac_idx`, pinned by `sigma::tests::pade_node_indices_matches_pyscf_get_ac_idx`).
They are independent of the W quadrature.

**Thiele continued fraction** (`pade.rs:39` `PadeCF::fit`, `pade.rs:76` `eval`):

$$ C_N(z) = \cfrac{a_1}{1 + \cfrac{a_2(z - z_1)}{1 + \cfrac{a_3(z-z_2)}{1 + \cdots}}}, \qquad g_1(z_k) = f(z_k), \quad g_\ell(z_k) = \frac{g_{\ell-1}(z_{\ell-1}) - g_{\ell-1}(z_k)}{(z_k - z_{\ell-1})\,g_{\ell-1}(z_k)}, \quad a_\ell = g_\ell(z_\ell), $$

evaluated bottom-up. In exact arithmetic **[sympy 6]**: $C_N$ interpolates
its $N$ nodes; it reproduces a rational function of type $[L/M]$ identically
in $z$ once $N \ge L + M + 1$ (so a single-pole $\Sigma^c$ of two states,
type [1/2], is continued exactly from 4 imaginary-axis nodes to the real
axis); one surplus node gives $a_{N} = 0$, and a second drives the next
denominator to exactly zero, which `fit` refuses with `Err` rather than
emitting NaN (`pade::tests::fit_errors_on_repeated_support_node`). The
physical $\Sigma^c$ is a rational function of degree ~$n_{\rm act}\times n_{\rm quad}$,
so 18 nodes give an approximation, not a reconstruction; the measured
sensitivities (16 vs 18 nodes 0.97 meV on the H₂O@PBE HOMO) are recorded in
`tests/validation_gw.rs`'s module doc. Unit tests: `pade::tests::interpolation_identity`,
`shifted_nodes_far_eval_matches_exact`, `pole_reproduction`.

The real-axis derivative is a 4-point central difference with $h = 0.05$ Ha
(`pade.rs:90` `deriv_real`; `sigma.rs:402`), exact through degree 4 with
error $-h^4 f^{(5)}/30$ **[sympy 5]**.

## 7. Quasiparticle equation, Newton solve, renormalization

For state $p$ the code solves

$$ f(E) = E - \epsilon_p - \delta_p - \mathrm{Re}\,C_N(E) = 0, \qquad \delta_p = \begin{cases} 0 & \text{HF reference} \\ \Sigma^x_p - v^{xc}_p & \text{KS reference} \end{cases} $$

(residual at `sigma.rs:421`; `sigma.rs:340` `solve_qp_for_mo`). The static
shift is inside the equation, so $\Sigma^c$ is evaluated at the shifted root;
adding it after an unshifted solve lands 0.38–0.96 eV away on the open-shell
@PBE cases, which `tests/validation_gw.rs` requires ferric to miss. $v^{xc}_p$
comes from `vxc_mo.rs:36` `vxc_diagonal_mo` (the same `add_xc` path as the
SCF's KS Fock).

- **Renormalization factor**
  $Z_p = \left[1 - \partial_E \mathrm{Re}\,\Sigma^c_p(E)\right]^{-1}_{E=\epsilon_p}$,
  clamped to $[0, 1.5]$ (`sigma.rs:405`). `GwResult.z_factor` is this value
  at the **mean-field** energy, the linearization point, not at the converged
  root.
- **Start**: $E_0 = \epsilon_p + Z_p(\Sigma^c_p(\epsilon_p) + \delta_p)$
  (`sigma.rs:410`), which is exactly one undamped Newton step from
  $\epsilon_p$ **[sympy 4]**.
- **Newton**: $E_{k+1} = E_k - d\,f(E_k)/f'(E_k)$,
  $f' = 1 - \partial_E\mathrm{Re}\,\Sigma^c$, damping
  $d = $ `qp_newton_damp` clamped to $[0.1, 1]$ (`sigma.rs:412`); at most 30
  steps; converged when $|\text{step}| < 10^{-7}$ Ha (`sigma.rs:429`); stops
  unconverged if $|f'| < 10^{-3}$ (`sigma.rs:424`, a root near a pole of
  $C_N$).
- **Fallback** (`sigma::qp_root_by_weight`, only when Newton fails): every
  root of $f$ within $\epsilon_p + \delta_p \pm 1$ Ha (4001 samples, sign
  changes with $|f| < 0.5$ on both sides, bisection) and the one of largest
  weight $Z \in (0, 1]$ is returned. NH2/cc-pVDZ U-G0W0@UKS α HOMO: Newton
  fails at every $n_{\rm quad}$ (roots −11.80 eV, Z 0.62, and −10.09 eV, Z 0.24,
  on the exact real axis); the fallback gives −11.802 eV at every
  $n_{\rm quad} \ge 30$. No root with $Z \in (0,1]$ leaves the state unsolved:
  reported in `GwResult.qp_converged` and warned (`lib.rs:491` `warn_gw_unconverged`).

At a root $E^*$, $Z(E^*) = 1/(1-\Sigma'(E^*))$ is the residue of
the QP pole of $G(E) = [E - \epsilon - \delta - \Sigma(E)]^{-1}$;
for a single-pole $\Sigma$ the QP equation has two roots (quasiparticle and
satellite) and undamped Newton converges quadratically to the one nearest
$\epsilon$ **[sympy 4]**. Tests: `sigma::tests::qp_newton_converges_for_a_mild_self_energy`,
`qp_newton_flags_nonconvergence_when_iteration_budget_exhausted`;
`tests/validation_gw.rs` `g0w0_hf_{h2o,nh3,n2}_vs_pyscf_gw_ac`,
`g0w0_pbe_{h2o,n2}_vs_pyscf_gw_ac`, `g0w0_hf_frozen_core_h2o_vs_pyscf_gw_ac`,
`g0w0_hf_ecp_{i2,xe,ag2}_vs_pyscf_gw_ac`; `tests/g0w0_pbe_h2o.rs`.

Default QP window: HOMO−2 … LUMO+2 (`lib.rs:514`; open shell `lib.rs:379`,
around $\max(n_\alpha, n_\beta)$).

## 8. Eigenvalue self-consistency

Both loops make Jacobi updates of the QP window only; states outside the
window, $\epsilon_F$, $\epsilon_p$ in $f(E)$, and $\delta_p$ stay at
their mean-field values. Convergence: $\max_p|\Delta\epsilon^{\rm QP}_p| <$
`ev_conv_thresh` (default 1e-4 Ha) within `max_ev_iter` (default 20); the
flag is `GwResult.outer_converged`.

- **evGW₀** (`sigma.rs:602` `run_evgw0`): the propagator energies in §4 are
  replaced by the previous sweep's QP energies (`sigma.rs:669`); W is frozen
  at the mean-field PDEP result.
- **evGW** (`sigma.rs:776` `run_evgw`): additionally rebuilds W every
  iteration after the first by rerunning PDEP-RPA on a copy of the reference
  whose $\epsilon$ in the QP window are the QP energies (`sigma.rs:833`,
  `:837`); orbitals and $\tilde B$ are unchanged. Convergence needs at least
  one rebuilt W (`sigma.rs:914`, `it > 0`).

Tests: `tests/validation_gw.rs` `evgw0_evgw_hf_h2o_vs_iterated_pyscf` (PySCF
`get_sigma` iterated with this update rule, 5.7e-7 Ha);
`tests/h2o_g0w0_cohsex.rs`; `tests/evgw_pbe_h2o.rs` (convergence and the KS
shift only; no external evGW@KS reference exists).

## 9. Open shell

`lib.rs:262` `run_u_gw` accepts UHF/UKS directly and ROHF/ROKS after per-spin
semi-canonicalization (`lib.rs:280`, `ferric_scf::semicanonical::unrestricted_reference`:
each spin's Fock diagonalized in its occupied and virtual blocks). W is
spin-independent, from the spin-summed dielectric

$$ \tilde\varepsilon_U(iu) = \mathbf 1 + \Pi_\alpha(iu) + \Pi_\beta(iu), \qquad \Pi_\sigma = 2\sum_{i_\sigma a_\sigma} B_\sigma B_\sigma^{\mathsf T}\frac{\Delta_{i\sigma}^{a}}{u^2+(\Delta_{i\sigma}^{a})^2} $$

(`ferric-rpa/src/energy.rs:599`; prefactor 2 per spin), which equals the
closed-shell $\mathbf 1 + \Pi$ when the spins coincide **[sympy 1]**. Each
spin has its own projection $B^\xi_\sigma = \hat U^{\mathsf T}B_\sigma$, its own
$\Sigma^x_\sigma$, $\Sigma^c_\sigma$ (sum over that spin's MOs), its own
mid-gap $\epsilon_{F,\sigma}$ (`u_sigma.rs:168`) and its own shift
$\delta_{\sigma p} = \Sigma^x_{\sigma p} - v^{xc}_{\sigma p}$ (`u_sigma.rs:42`
`static_shifts`). Drivers: `u_sigma.rs:61` `run_u_g0w0`, `:249`
`run_u_evgw0`, `:454` `run_u_evgw`; `u_cohsex.rs:23` `run_u_cohsex`.
sc-COHSEX is not implemented (returns `Err`).

Tests: `tests/validation_gw.rs` `u_g0w0_{oh,ch3,nh2}_vs_pyscf_ugw_ac`,
`u_g0w0_triplets_o2_ch2_vs_pyscf_ugw_ac`, `u_g0w0_uks_pbe_*`;
`tests/validation_u_cohsex.rs` (incl. `u_cohsex_singlet_h2o_equals_closed_shell_cohsex`);
`tests/u_cohsex_evgw_closed_shell_limit.rs`; `tests/validation_rohf_reference.rs`;
`tests/u_gw_ks_shift.rs`.

## 10. PDEP as W

The only GW-specific use of PDEP is the basis $\{\hat U_\xi\}$ of §2.
`PdepRpaConfig.trunc_thresh` drops modes with $\lambda_\xi(0)$ within the
threshold of 1. Truncation removes the corresponding rows of $B^\xi_{pq}$ and of
$\tilde W_d$; $\Sigma^x$ is unaffected (§1). Every external comparison in §11
uses `trunc_thresh = 0`, where the PDEP basis is the full RI dielectric
eigenbasis and ferric's W is the ordinary RI W; **truncated GW is not
validated**. `run_gw`/`run_u_gw` force `need_inv_dielectric_freq = true`
(`lib.rs:69`). Memory: $B$ and $B^\xi$ are hard pool charges
(`mo_b.rs`, `cohsex.rs:56` `guard_m_proj`), both $\propto N_{\rm aux} n_{\rm act}^2$.

## 11. BSE-TDA

On G0W0@HF QP energies for **all** active MOs (`bse.rs:283` `run_bse_tda`),
the closed-shell singlet TDA matrix

$$ \mathbf A_{ia,jb} = (\epsilon^{\rm QP}_a - \epsilon^{\rm QP}_i)\,\delta_{ij}\delta_{ab} + 2(ia|jb) - (ab|W(0)|ij), $$

$$ (pq|W(0)|rs) = (pq|rs) + \sum_\xi w_\xi B^\xi_{pq}B^\xi_{rs} $$

(`bse.rs:395` `screened`; assembly `bse.rs:431`–`434`; dense `eigh` at
`bse.rs:463`). The bare part comes from $B$, so the kernel equals
$\sum_\xi \lambda_\xi(0)^{-1} B^\xi B^\xi$ for a complete basis **[sympy 8]** and
stays correct under truncation. The Hartree term uses bare $v$; the direct
term uses the static full W. With $W \to v$ and HF energies it is CIS
(`bse.rs:564` `run_cis_tda`). Oscillator strengths
$f_n = \tfrac23\Omega_n|\langle 0|\mathbf r|n\rangle|^2$,
$\langle 0|\mathbf r|n\rangle = \sqrt2\sum_{ia}X_{n,ia}\langle i|\mathbf r|a\rangle$
(`bse.rs:158`). Tests: `tests/validation_bse.rs`
(`bse_tda_{h2o,nh3,ch2o}_vs_numpy`, `cis_tda_anchor_vs_numpy_and_pyscf_tda`),
`tests/bse_oscillator_strength.rs`, `tests/h2o_bse_tda.rs`.

## 12. TDHF/RPAx polarizability

The statically screened RPAx kernel on a reference with energies $\epsilon$
(`bse.rs:1369` `run_rpax_static_polarizability`; dynamic, library-only:
`bse.rs:715` `run_bse_c6`, `bse.rs:966` `run_bse_c6_ks`):

$$ (\mathbf A+\mathbf B)_{ia,jb} = \Delta_i^a\delta + 4(ia|jb) - (ab|W|ij) - (ib|W|ja), \qquad (\mathbf A-\mathbf B)_{ia,jb} = \Delta_i^a\delta + (ib|W|ja) - (ab|W|ij), $$

$$ \alpha_{xy}(iu) = 4\,\mu_x^{\mathsf T}\big[(\mathbf A-\mathbf B)(\mathbf A+\mathbf B) + u^2\big]^{-1}(\mathbf A-\mathbf B)\,\mu_y = 4\,\mu_x^{\mathsf T}\big[(\mathbf A+\mathbf B) + u^2(\mathbf A-\mathbf B)^{-1}\big]^{-1}\mu_y, $$

computed as one solve of $[(\mathbf A-\mathbf B)(\mathbf A+\mathbf B)+u^2]\,t = (\mathbf A-\mathbf B)\mu$
(`bse.rs:1588`–`1601`, `bse.rs:899`–`910`). The operator
$[(\mathbf A-\mathbf B)(\mathbf A+\mathbf B)+u^2]^{-1}(\mathbf A-\mathbf B)$ is symmetric and equals the sum over states
$2\sum_n|\langle0|\mathbf r|n\rangle|^2\,\Omega_n/(\Omega_n^2+u^2)$ with
$\Omega_n^2$ the eigenvalues of $(\mathbf A-\mathbf B)(\mathbf A+\mathbf B)$; at $u = 0$,
$\alpha = 4\mu^{\mathsf T}(\mathbf A+\mathbf B)^{-1}\mu$ **[sympy 7]**. The reversed product
$\mu^{\mathsf T}(\mathbf A-\mathbf B)[\cdots]^{-1}\mu$ is a different number. `scissor` adds a
constant to every virtual energy. With $W \to v$ at an HF reference the kernel
is TDHF and $\alpha(0)$ is the CPHF polarizability
(`RpaxScreening::Bare`). A negative $\alpha$ diagonal is refused
(`bse.rs:1251` `check_alpha_diagonal_positive`). Tests:
`tests/validation_rpax_alpha.rs`, `tests/rpax_*.rs`.

## 13. Validation status

Grades as published on the site (`site/src/reference/validation.md`):

| Capability | Grade | What is established |
|---|---|---|
| G0W0, evGW₀, evGW, COHSEX (closed shell), U-G0W0, U-COHSEX | Proven (narrow) | At **matched settings** (`n_quad = 100`, `trunc_thresh = 0`), against the grid-free dRPA pole sum: G0W0@HF ≤1.2e-8 Ha, @PBE ≤1.8e-8, U-G0W0@UHF ≤1.2e-5, U-G0W0@UKS ≤2.7e-8 (§4, product integration); against PySCF `gw_ac`: G0W0@HF ≤1.6e-7; ECP ≤2.5e-6, COHSEX ≤1.2e-9, evGW ≤5.7e-7, U-COHSEX ≤3.3e-9. At the defaults (`n_quad = 40`, `trunc_thresh = 1e-4`) HOMO and LUMO stay within 10 meV of matched settings. Narrow: the analytic continuation limits states far from ε_F with split spectral weight. |
| BSE-TDA | Smoke | Lowest five singlets match a numpy BSE-TDA to 1.8e-10 Ha given the same QP energies (H₂O, NH₃, CH₂O). |
| RPAx static α | Smoke | Matches an independent numpy kernel to 5.3e-9 relative; the physics is not established (water/cc-pVDZ@PBE, scissor 0.36 Ha: 5.20 vs DOSD 9.64 a.u.); C6 from the dynamic kernel is ~63% low. |

Benchmark statistics against experiment (GW100, G0W0@HF MAE 0.353 eV at
aug-cc-pVDZ, 0.491 eV at aug-cc-pVTZ) are kept in the working notes, not as
pinned tests. The algebra in this file is exact; the numerical error budget
is the W quadrature, the 18-node Padé extrapolation (largest for KS
references, whose $\epsilon_{\rm HOMO}$ lies far from $\epsilon_F$), the RI
basis, and PDEP truncation when enabled.

## 14. References

1. L. Hedin, *Phys. Rev.* **139**, A796 (1965) — GW and COHSEX.
2. H. J. Vidberg and J. W. Serene, *J. Low Temp. Phys.* **29**, 179 (1977) — Thiele–Padé continuation.
3. X. Ren, P. Rinke, V. Blum *et al.*, *New J. Phys.* **14**, 053020 (2012) — RI GW on the imaginary axis.
4. J. Wilhelm, D. Golze, L. Talirz, J. Hutter, C. A. Pignedoli, *J. Phys. Chem. Lett.* **9**, 306 (2018) — analytic-continuation G0W0 for molecules.
5. H. F. Wilson, F. Gygi, G. Galli, *Phys. Rev. B* **78**, 113303 (2008) — PDEP dielectric eigenpotentials.
6. M. J. van Setten *et al.*, *J. Chem. Theory Comput.* **11**, 5665 (2015) — GW100.
7. X. Blase, I. Duchemin, D. Jacquemin, *Chem. Soc. Rev.* **47**, 1022 (2018) — BSE for molecules.
8. T. Zhu and G. K.-L. Chan, *J. Chem. Theory Comput.* **17**, 727 (2021) — Gaussian-basis analytic-continuation G0W0 (PySCF `gw_ac`, whose node selection and QP conventions are matched here).
