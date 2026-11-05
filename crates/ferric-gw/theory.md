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
$m,n,p,q$ any active MO, $P,Q$ RI auxiliary functions, $\alpha,\beta$ PDEP
modes. "Active" means MO indices `frozen_core..nmo`.

## 1. RI factorization and exchange

With the Coulomb metric $V_{PQ} = (P|Q)$ and $F$ a factor with
$F^{\mathsf T} F = V^{-1}$ (Cholesky $L^{-1}$, or the symmetric eigh
$V^{-1/2}$ for attenuated operators), the dressed three-index tensor is

$$ \tilde B^P_{mn} = \sum_Q F_{PQ}\,(Q|mn), \qquad (pq|rs) \approx \sum_P \tilde B^P_{pq} \tilde B^P_{rs}, $$

held over the full active square, shape `(naux, n_act, n_act)`
(`mo_b.rs:206` `build_full_b`; per spin `mo_b.rs:243`, `:406`). The
exchange self-energy diagonal is the closed-shell HF exchange in this basis,

$$ \Sigma^x_m = -\sum_i \sum_P \left(\tilde B^P_{mi}\right)^2 $$

(`cohsex.rs:111` `sigma_x_diag`; per spin the same formula on that spin's
$\tilde B_\sigma$). It is computed from the full $\tilde B$, never from the
PDEP projection, so PDEP rank truncation affects only correlation. Checked
against PySCF's density-fitted exchange to ≤4.9e-9 Ha
(`tests/validation_gw.rs`, the Σx anchor).

## 2. Screened interaction on the imaginary axis

In the dressed basis the bare interaction is the identity. The closed-shell
independent-particle response and dielectric matrix are

$$ \Pi_{PQ}(i\omega) = 4 \sum_{ia} \tilde B^P_{ia}\tilde B^Q_{ia}\,\frac{\epsilon_{ia}}{\omega^2 + \epsilon_{ia}^2}, \qquad \tilde\varepsilon(i\omega) = \mathbf 1 + \Pi(i\omega), \qquad \epsilon_{ia} = \epsilon_a - \epsilon_i, $$

prefactor 4 = 2 (spin) × 2 (resonant + antiresonant)
(`ferric-rpa/src/sternheimer.rs:204` `build_scale_factors`, `:400`
`dielectric_matrix_from_projection`). The full screened interaction is
$\tilde W = \tilde\varepsilon^{-1}$; its static bare part gives $\Sigma^x$ and
the code carries only the dynamic part

$$ \tilde W_d(i\omega) = \tilde\varepsilon^{-1}(i\omega) - \mathbf 1 . $$

**PDEP basis.** The static matrix is diagonalized,
$\tilde\varepsilon(0)\,U_\alpha = \lambda_\alpha(0)\,U_\alpha$, and the GW code
works in the fixed eigenbasis $\{U_\alpha\}$. $\tilde W_d$ is built as the
**full** $M\times M$ matrix $[U^{\mathsf T}\tilde\varepsilon(i\omega_k)U]^{-1} - \mathbf 1$
at every quadrature node (`ferric-rpa/src/energy.rs:566`
`eval_inv_dielectric_matrices`; `PdepRpaResult.inv_dielectric_freq`). The
per-frequency scalar eigenvalues $\lambda_\alpha(i\omega)$ live in an
$\omega$-dependent rotated eigenbasis and are valid only for basis-invariant
traces such as the RPA energy; the diagonal form $\sum_\alpha(1/\lambda_\alpha(i\omega)-1)$
is not used for $\Sigma_c$. At $\omega = 0$ the matrix is diagonal,
$w_\alpha \equiv 1/\lambda_\alpha(0) - 1$ (`w_pdep.rs:157` `static_weights`).

**Redressing.** `PdepRpaResult.eigenpotentials` are physical aux
coefficients $c = F_{\rm pdep}^{\mathsf T} U$. The dressed vectors matching
this crate's $F$ solve $F^{\mathsf T} V = c$ (`w_pdep.rs:39`
`redress_eigenpotentials`; triangular solve for Cholesky, pseudo-inverse for
symmetric $F$, anything else an error). Tests:
`tests/redress_recovers_dressed_eigenvectors.rs`.

**Projection.** All self-energy and kernel contractions use

$$ M_{\alpha,mn} = \sum_P V^P_\alpha\, \tilde B^P_{mn} $$

(`cohsex.rs:228` `project_b_into_pdep`, one GEMM at `cohsex.rs:291`). For a
complete orthonormal $V$, $\sum_\alpha M_{\alpha,pq}M_{\alpha,rs} = (pq|rs)$
**[sympy 8]**; a truncated $V$ does not satisfy it.

The screened matrix element at node $k$ is the quadratic form

$$ W_{mn}(i\omega_k) = \sum_{\alpha\beta} M_{\alpha,mn}\,[\tilde W_d(i\omega_k)]_{\alpha\beta}\,M_{\beta,mn} $$

(`sigma.rs:310`–`315`, one GEMM per node).

## 3. Single-pole structure

For one $(ia)$ pair with $\tilde B = b$, $\epsilon_{ia} = e$:

$$ \tilde W_d(i\omega) = -\frac{4b^2 e}{\omega^2 + e^2 + 4b^2 e} = -\frac{2\Omega s}{\omega^2 + \Omega^2}, \qquad \Omega^2 = e(e + 4b^2) = (A-B)(A+B), \quad s = \frac{2b^2e}{\Omega}, $$

with $A = e + 2K$, $B = 2K$, $K = b^2$ the singlet direct-RPA matrices: the
pole of W is the direct-RPA excitation energy **[sympy 1]**. In general
$\tilde W_d(i\omega) = -\sum_s 2\Omega_s\,\rho_s\rho_s^{\mathsf T}/(\omega^2+\Omega_s^2)$,
negative definite on the imaginary axis because $\tilde\varepsilon \ge \mathbf 1$.
This model is the analytic reference for §4 and §5.

## 4. Correlation self-energy on the imaginary axis

The code evaluates, for any complex $z$ off the real axis,

$$ \Sigma^c_m(z) = -\frac{1}{\pi}\sum_n \int_0^\infty d\omega'\; W_{mn}(i\omega')\,\frac{z - \epsilon_n}{(z - \epsilon_n)^2 + \omega'^2} $$

(`sigma.rs:274` `sigma_c_at_z`; the $-1/\pi$ at `sigma.rs:329`). The
integrand of the full $\int_{-\infty}^{\infty}$ form is even in $\omega'$, so
$\frac{1}{2\pi}\cdot 2 = \frac1\pi$. The sum over $n$ runs over every active
MO with no occupied/virtual sign: the time ordering is carried by the sign of
$\mathrm{Re}(z - \epsilon_n)$. $z$ is an absolute energy; `eps_act` (the
propagator energies) is a separate argument so evGW can replace it. The
integral uses the PDEP quadrature (`pdep.quad_freqs`, `pdep.quad_weights`),
scaled Gauss–Legendre $\omega = u_0(1+x)/(1-x)$
(`ferric-rpa/src/quadrature.rs:84` `gauss_legendre_nodes`).

**Closed form for a single-pole W [sympy 2].** With
$W_{mn}(i\omega) = -2\Omega s_{mn}/(\omega^2+\Omega^2)$,

$$ \Sigma^c_m(z) = \sum_{n:\,\mathrm{Re}(z-\epsilon_n)>0} \frac{s_{mn}}{z - \epsilon_n + \Omega} \;+\; \sum_{n:\,\mathrm{Re}(z-\epsilon_n)<0} \frac{s_{mn}}{z - \epsilon_n - \Omega}, $$

the textbook residue result: occupied poles at $\epsilon_i - \Omega$, virtual
poles at $\epsilon_a + \Omega$. On the continuation line $z = \epsilon_F + i\omega$
the occupied states are exactly those with $\mathrm{Re}(z-\epsilon_n) > 0$.
Proven symbolically for real arguments and checked at complex $z$ to 1e-30.
The Rust test `sigma::tests::sigma_c_single_pole_w_matches_closed_form_residues`
feeds this W through `sigma_c_at_z` on the 100-point grid and asserts the
closed form to 1e-8 Ha at $z = \epsilon_F + i\{0, 0.1, 0.6, 2\}$ (measured
2.3e-9, the quadrature error of that grid).

Further checks: `sigma::tests::sigma_c_blas3_matches_scalar_reference`
(GEMM vs scalar loop); `tests/validation_gw.rs`
`sigma_c_imag_axis_h2o_pbe_and_hf_vs_pyscf`, `sigma_c_imag_axis_n2_pbe_vs_pyscf`
($\Sigma^c(\epsilon_F + i\omega)$ at the Padé nodes vs PySCF `gw_ac`, ≤8.6e-11 Ha).

## 5. COHSEX: the static limit

For a frequency-independent $W_{mn}$ the integral of §4 is elementary,
$-\frac1\pi\int_0^\infty W\,a/(a^2+\omega^2)\,d\omega = -\tfrac12 W\,\mathrm{sgn}(a)$,
so for $z$ in the gap

$$ \Sigma^c_m = -\tfrac12\sum_i W_{mi} + \tfrac12\sum_a W_{ma} = \underbrace{-\sum_i W_{mi}}_{\Delta\Sigma^{\rm SEX}_m} + \underbrace{\tfrac12 \sum_p W_{mp}}_{\Sigma^{\rm COH}_m} . $$

The plasmon model of §4 reaches the same limit as $\Omega \to \infty$ at fixed
$W(0) = -2s/\Omega$ **[sympy 3]**. With $W_{mn}(0) = \sum_\alpha w_\alpha M_{\alpha,mn}^2$:

$$ \Delta\Sigma^{\rm SEX}_m = -\sum_i\sum_\alpha w_\alpha M^2_{\alpha,mi}, \qquad \Sigma^{\rm COH}_m = \tfrac12\sum_p\sum_\alpha w_\alpha M^2_{\alpha,mp}, \qquad w_\alpha = \frac1{\lambda_\alpha(0)} - 1 $$

(`cohsex.rs:311` `cohsex_pieces`). The full screened exchange is
$\Sigma^x + \Delta\Sigma^{\rm SEX} = -\sum_i\sum_\alpha \lambda_\alpha(0)^{-1} M^2_{\alpha,mi}$
for a complete PDEP basis **[sympy 8]**. COHSEX has no root search:

$$ \epsilon^{\rm QP}_m = \epsilon_m + \Delta\Sigma^{\rm SEX}_m + \Sigma^{\rm COH}_m \;[+\;\Sigma^x_m - v^{xc}_m \text{ at a KS reference}], \qquad Z = 1 $$

(`cohsex.rs:363` `run_cohsex`, assembly at `cohsex.rs:421`–`427`). The Rust
test `sigma::tests::sigma_c_static_w_in_gap_equals_cohsex_pieces` puts a
constant W through `sigma_c_at_z` at $z = \epsilon_F$ and asserts
`cohsex_pieces` to 1e-9 Ha. External checks: `tests/validation_gw.rs`
`cohsex_hf_h2o_n2_vs_numpy` (9.4e-10 Ha), `cohsex_pbe_h2o_n2_vs_numpy`
(1.2e-9 Ha); `tests/cohsex_ks_shift.rs`.

## 6. Analytic continuation

$\Sigma^c_m$ is sampled on the Fermi-shifted imaginary axis
$z_k = \epsilon_F + i\omega_k$, $\epsilon_F = \tfrac12(\epsilon_{\rm HOMO} + \epsilon_{\rm LUMO})$
of the active mean-field spectrum (`sigma.rs:246` `fermi_level`). The nodes
are 18 indices (or `GwConfig.pade_npts`) chosen from $\{0\}\cup$ GL(100,
$u_0 = 0.5$) with a decreasing step, step ratio 2/3, first index 1
(`sigma.rs:368`–`381`; `sigma.rs:181` `pade_node_indices`, a port of PySCF
`_get_ac_idx`, pinned by `sigma::tests::pade_node_indices_matches_pyscf_get_ac_idx`).
They are independent of the W quadrature.

**Thiele continued fraction** (`pade.rs:39` `PadeCF::fit`, `pade.rs:76` `eval`):

$$ C_N(z) = \cfrac{a_1}{1 + \cfrac{a_2(z - z_1)}{1 + \cfrac{a_3(z-z_2)}{1 + \cdots}}}, \qquad g_1(z_k) = f(z_k), \quad g_p(z_k) = \frac{g_{p-1}(z_{p-1}) - g_{p-1}(z_k)}{(z_k - z_{p-1})\,g_{p-1}(z_k)}, \quad a_p = g_p(z_p), $$

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

For state $m$ the code solves

$$ f(\omega) = \omega - \epsilon_m - \Delta_m - \mathrm{Re}\,C_N(\omega) = 0, \qquad \Delta_m = \begin{cases} 0 & \text{HF reference} \\ \Sigma^x_m - v^{xc}_m & \text{KS reference} \end{cases} $$

(residual at `sigma.rs:421`; `sigma.rs:340` `solve_qp_for_mo`). The static
shift is inside the equation, so $\Sigma^c$ is evaluated at the shifted root;
adding it after an unshifted solve lands 0.38–0.96 eV away on the open-shell
@PBE cases, which `tests/validation_gw.rs` requires ferric to miss. $v^{xc}_m$
comes from `vxc_mo.rs:36` `vxc_diagonal_mo` (the same `add_xc` path as the
SCF's KS Fock).

- **Renormalization factor**
  $Z_m = \left[1 - \partial_\omega \mathrm{Re}\,\Sigma^c_m(\omega)\right]^{-1}_{\omega=\epsilon_m}$,
  clamped to $[0, 1.5]$ (`sigma.rs:405`). `GwResult.z_factor` is this value
  at the **mean-field** energy, the linearization point, not at the converged
  root.
- **Start**: $\omega_0 = \epsilon_m + Z_m(\Sigma^c_m(\epsilon_m) + \Delta_m)$
  (`sigma.rs:410`), which is exactly one undamped Newton step from
  $\epsilon_m$ **[sympy 4]**.
- **Newton**: $\omega_{k+1} = \omega_k - d\,f(\omega_k)/f'(\omega_k)$,
  $f' = 1 - \partial_\omega\mathrm{Re}\,\Sigma^c$, damping
  $d = $ `qp_newton_damp` clamped to $[0.1, 1]$ (`sigma.rs:412`); at most 30
  steps; converged when $|\text{step}| < 10^{-7}$ Ha (`sigma.rs:429`); stops
  unconverged if $|f'| < 10^{-3}$ (`sigma.rs:424`, a root near a pole of
  $C_N$). Non-convergence is reported in `GwResult.qp_converged` and warned
  (`lib.rs:491` `warn_gw_unconverged`).

At a root $\omega^*$, $Z(\omega^*) = 1/(1-\Sigma'(\omega^*))$ is the residue of
the QP pole of $G(\omega) = [\omega - \epsilon - \Delta - \Sigma(\omega)]^{-1}$;
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
window, $\epsilon_F$, $\epsilon_m$ in $f(\omega)$, and $\Delta_m$ stay at
their mean-field values. Convergence: $\max_m|\Delta\epsilon^{\rm QP}_m| <$
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

$$ \tilde\varepsilon_U(i\omega) = \mathbf 1 + \Pi_\alpha(i\omega) + \Pi_\beta(i\omega), \qquad \Pi_\sigma = 2\sum_{i_\sigma a_\sigma} \tilde B_\sigma\tilde B_\sigma^{\mathsf T}\frac{\epsilon_{ia,\sigma}}{\omega^2+\epsilon_{ia,\sigma}^2} $$

(`ferric-rpa/src/energy.rs:599`; prefactor 2 per spin), which equals the
closed-shell $\mathbf 1 + \Pi$ when the spins coincide **[sympy 1]**. Each
spin has its own projection $M_\sigma = V^{\mathsf T}\tilde B_\sigma$, its own
$\Sigma^x_\sigma$, $\Sigma^c_\sigma$ (sum over that spin's MOs), its own
mid-gap $\epsilon_{F,\sigma}$ (`u_sigma.rs:168`) and its own shift
$\Delta_{\sigma m} = \Sigma^x_{\sigma m} - v^{xc}_{\sigma m}$ (`u_sigma.rs:42`
`static_shifts`). Drivers: `u_sigma.rs:61` `run_u_g0w0`, `:249`
`run_u_evgw0`, `:454` `run_u_evgw`; `u_cohsex.rs:23` `run_u_cohsex`.
sc-COHSEX is not implemented (returns `Err`).

Tests: `tests/validation_gw.rs` `u_g0w0_{oh,ch3,nh2}_vs_pyscf_ugw_ac`,
`u_g0w0_triplets_o2_ch2_vs_pyscf_ugw_ac`, `u_g0w0_uks_pbe_*`;
`tests/validation_u_cohsex.rs` (incl. `u_cohsex_singlet_h2o_equals_closed_shell_cohsex`);
`tests/u_cohsex_evgw_closed_shell_limit.rs`; `tests/validation_rohf_reference.rs`;
`tests/u_gw_ks_shift.rs`.

## 10. PDEP as W

The only GW-specific use of PDEP is the basis $\{V_\alpha\}$ of §2.
`PdepRpaConfig.trunc_thresh` drops modes with $\lambda_\alpha(0)$ within the
threshold of 1. Truncation removes the corresponding rows of $M$ and of
$\tilde W_d$; $\Sigma^x$ is unaffected (§1). Every external comparison in §11
uses `trunc_thresh = 0`, where the PDEP basis is the full RI dielectric
eigenbasis and ferric's W is the ordinary RI W; **truncated GW is not
validated**. `run_gw`/`run_u_gw` force `need_inv_dielectric_freq = true`
(`lib.rs:69`). Memory: $\tilde B$ and $M$ are hard pool charges
(`mo_b.rs`, `cohsex.rs:56` `guard_m_proj`), both $\propto n_{\rm aux} n_{\rm act}^2$.

## 11. BSE-TDA

On G0W0@HF QP energies for **all** active MOs (`bse.rs:283` `run_bse_tda`),
the closed-shell singlet TDA matrix

$$ A_{ia,jb} = (\epsilon^{\rm QP}_a - \epsilon^{\rm QP}_i)\,\delta_{ij}\delta_{ab} + 2(ia|jb) - (ab|W(0)|ij), $$

$$ (pq|W(0)|rs) = (pq|rs) + \sum_\alpha w_\alpha M_{\alpha,pq}M_{\alpha,rs} $$

(`bse.rs:395` `screened`; assembly `bse.rs:431`–`434`; dense `eigh` at
`bse.rs:463`). The bare part comes from $\tilde B$, so the kernel equals
$\sum_\alpha \lambda_\alpha(0)^{-1} M M$ for a complete basis **[sympy 8]** and
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

$$ (A+B)_{ia,jb} = \epsilon_{ia}\delta + 4(ia|jb) - (ab|W|ij) - (ib|W|aj), \qquad (A-B)_{ia,jb} = \epsilon_{ia}\delta + (ib|W|aj) - (ab|W|ij), $$

$$ \alpha_{xy}(i\omega) = 4\,\mu_x^{\mathsf T}\big[(A-B)(A+B) + \omega^2\big]^{-1}(A-B)\,\mu_y = 4\,\mu_x^{\mathsf T}\big[(A+B) + \omega^2(A-B)^{-1}\big]^{-1}\mu_y, $$

computed as one solve of $[(A-B)(A+B)+\omega^2]\,t = (A-B)\mu$
(`bse.rs:1588`–`1601`, `bse.rs:899`–`910`). The operator
$[(A-B)(A+B)+\omega^2]^{-1}(A-B)$ is symmetric and equals the sum over states
$2\sum_n|\langle0|\mathbf r|n\rangle|^2\,\Omega_n/(\Omega_n^2+\omega^2)$ with
$\Omega_n^2$ the eigenvalues of $(A-B)(A+B)$; at $\omega = 0$,
$\alpha = 4\mu^{\mathsf T}(A+B)^{-1}\mu$ **[sympy 7]**. The reversed product
$\mu^{\mathsf T}(A-B)[\cdots]^{-1}\mu$ is a different number. `scissor` adds a
constant to every virtual energy. With $W \to v$ at an HF reference the kernel
is TDHF and $\alpha(0)$ is the CPHF polarizability
(`RpaxScreening::Bare`). A negative $\alpha$ diagonal is refused
(`bse.rs:1251` `check_alpha_diagonal_positive`). Tests:
`tests/validation_rpax_alpha.rs`, `tests/rpax_*.rs`.

## 13. Validation status

Grades as published on the site (`site/src/reference/validation.md`):

| Capability | Grade | What is established |
|---|---|---|
| G0W0, evGW₀, evGW, COHSEX (closed shell), U-G0W0, U-COHSEX | Smoke | Implementation matches PySCF `gw_ac`/`ugw_ac` or a numpy build at **matched settings** (`n_quad = 100`, `trunc_thresh = 0`): G0W0@HF ≤1.3e-7 Ha, @PBE ≤6.2e-8, ECP ≤2.5e-6, U-G0W0@UHF ≤8.7e-7 (OH/cc-pVDZ 7.0e-6), U-G0W0@UKS ≤2.6e-8, COHSEX ≤1.2e-9, evGW ≤5.7e-7, U-COHSEX ≤3.3e-9. CLI defaults are coarser (the 20-point grid moves the H₂O@PBE HOMO by 13 meV); truncated PDEP is not validated. |
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
