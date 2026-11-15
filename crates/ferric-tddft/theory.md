# ferric-tddft Mathematical Foundations

Linear response on closed-shell references (singlet or triplet) and
spin-conserving response on UHF/UKS references: TDA (CIS for an HF
reference) and full Casida TDDFT (TDHF for an HF reference), on RI integrals,
for LDA, GGA, global-hybrid, τ-only meta-GGA and range-separated-hybrid
references. The driver is `run_tddft` (`src/tddft.rs`, `src/unrestricted.rs`);
the XC kernel block comes from `ferric_dft::lr_kernel` (equations in
`crates/ferric-dft/theory.md`). §8 derives the range-separated exchange, §9
the triplet spin combination, §10 the unrestricted problem, §11 the
⟨S²⟩ of its roots and §12 the matrix-free Davidson solvers.

---

## 1. Response matrices

Indices $i,j$ run over doubly occupied and $a,b$ over virtual spatial
orbitals of a restricted reference; the compound index $ia$ has dimension
$d = n_o n_v$, stored as `i * nvir + a`. Orbitals are real, so the
chemist-notation integrals carry no complex conjugates, $(pq|rs) = (qp|rs)$:

$$A_{ia,jb} = \delta_{ij}\delta_{ab}\,\Delta_i^a + 2(ia|jb) - c_{\rm HF}(ij|ab) + K_{ia,jb}$$

$$B_{ia,jb} = 2(ia|jb) - c_{\rm HF}(ib|ja) + K_{ia,jb}$$

with $\Delta_i^a = \varepsilon_a - \varepsilon_i$ (`build_a_matrix`, `build_b_matrix`; the Coulomb coefficient 2 is
`coulomb_factor`, and `add_fxc_block` adds $K$ to both.)

- The factor 2 on the Coulomb term is the singlet spin adaptation.
- $c_{\rm HF}$ is the exact-exchange fraction: 1 for HF, the functional's
  global fraction for a hybrid, 0 for LDA/GGA/meta-GGA.
- $K_{ia,jb} = (ia|f_{\alpha\alpha}+f_{\alpha\beta}|jb)$ is the singlet XC-kernel
  block built by `ferric_dft::lr_kernel::fxc_ov_block` with
  `ResponseSpin::Singlet` (`singlet_fxc_ov_block`). The same block enters
  $\mathbf A$ and $\mathbf B$.
  It is zero for an HF reference. For a spin-symmetric $e(\rho_\alpha,\rho_\beta)$,
  $f_{\alpha\alpha}+f_{\alpha\beta} = 2\,\partial^2 e(\rho/2,\rho/2)/\partial\rho^2$,
  so the block is twice the unpolarized kernel: PySCF `tdscf.rhf.get_ab`'s
  `iajb`, which carries the same factor 2 (§5).
- $\mathbf A$ and $\mathbf B$ share the Coulomb block $2\,B_{ov}^{\mathsf T}B_{ov}$.
  The exchange terms differ: $(ij|ab)$ in $\mathbf A$ and $(ib|ja)$ in
  $\mathbf B$, the $(ib, ja)$ element of the same Coulomb product. (The
  derivation produces $(ia|bj)$ and $(ib|aj)$ in $\mathbf B$; for real
  orbitals they equal $(ia|jb)$ and $(ib|ja)$.)

**RI.** Every integral is factorized in the Coulomb metric,
$(pq|rs) \approx \sum_P B^P_{pq}B^P_{rs}$ with
$B^P_{pq} = \sum_Q M_{PQ}(Q|pq)$, where $M^{\mathsf T}M = V^{-1}$ and
$V_{PQ} = (P|Q)$ over the auxiliary basis (`metric_inverse_sqrt`, which
returns the inverse Cholesky factor for the Coulomb metric;
`stream_dressed_mo_band`). The
driver builds the $ov$, $oo$ and $vv$ blocks $B_{ov}$, $B_{oo}$, $B_{vv}$.

---

## 2. Tamm-Dancoff approximation

Setting $\mathbf B = 0$ leaves the Hermitian eigenproblem

$$\mathbf A\,X_n = \Omega_n X_n,\qquad X_n^{\mathsf T}X_n = 1.$$

For an HF reference ($c_{\rm HF}=1$, $K=0$) this is CIS. `solve_tda` diagonalizes
the dense $\mathbf A$.

---

## 3. Casida equations

Full linear response solves

$$\begin{pmatrix}\mathbf A & \mathbf B\\ \mathbf B & \mathbf A\end{pmatrix}\begin{pmatrix}X\\ Y\end{pmatrix}
= \Omega\begin{pmatrix}1 & 0\\ 0 & -1\end{pmatrix}\begin{pmatrix}X\\ Y\end{pmatrix}.$$

Adding and subtracting the two block rows gives
$(\mathbf A+\mathbf B)(X+Y) = \Omega(X-Y)$ and $(\mathbf A-\mathbf B)(X-Y) = \Omega(X+Y)$, so
$(\mathbf A-\mathbf B)(\mathbf A+\mathbf B)(X+Y) = \Omega^2(X+Y)$. When $\mathbf A-\mathbf B$ is positive definite, the
symmetric form with $T = (\mathbf A-\mathbf B)^{1/2}$ is

$$M Z = \Omega^2 Z,\qquad M = T(\mathbf A+\mathbf B)T,$$

and $X+Y = c\,T Z$ solves the non-symmetric equation. The normalization
$|X|^2-|Y|^2 = (X+Y)^{\mathsf T}(X-Y) = (X+Y)^{\mathsf T}(\mathbf A+\mathbf B)(X+Y)/\Omega
= c^2 Z^{\mathsf T}MZ/\Omega = c^2\Omega$ for unit $Z$, so

$$(X+Y)_n = \Omega_n^{-1/2}\,T Z_n.$$

`solve_casida` builds $T$ from the eigendecomposition of $\mathbf A-\mathbf B$. A negative
eigenvalue of $\mathbf A-\mathbf B$ (an instability of the reference) is returned as an
error, not clamped. For an HF reference this is TDHF.

**Roots.** Both paths return the lowest `n_roots` eigenvalues above
$10^{-10}$ Ha, in ascending order (`lowest_positive_roots`).

---

## 4. Transition dipoles and oscillator strengths

In the length gauge, with the singlet spin-adaptation factor $\sqrt2$:

$$\langle 0|\mathbf r|n\rangle = \sqrt2\sum_{ia}(X+Y)_{ia,n}\langle i|\mathbf r|a\rangle,\qquad
f_n = \tfrac23\,\Omega_n\,|\langle 0|\mathbf r|n\rangle|^2,$$

with $Y = 0$ and $|X| = 1$ in TDA. This is PySCF's convention
(`oscillator_strengths`, dipole origin at the coordinate origin; the result
is origin-independent because $\langle i|a\rangle = 0$). The $\sqrt2$ is the
spin sum: the singlet amplitude is $X_{ia\alpha} = X_{ia\beta} = X_{ia}/\sqrt2$
and the spin-free dipole adds the two spins. With all roots, the
normalization of §3 gives the sum rule
$\sum_n f_n = \tfrac43\sum_{x}\mathbf r_x^{\mathsf T}(\mathbf A-\mathbf B)\,\mathbf r_x$
($\mathbf r_x$ the vector of $\langle i|x|a\rangle$), since
$\sum_n\Omega_n(X+Y)_n(X+Y)_n^{\mathsf T} = T\big(\sum_n Z_nZ_n^{\mathsf T}\big)T = \mathbf A-\mathbf B$.

---

## 5. Functional gate and kernel

`run_tddft` resolves the functional through `tddft.rs::resolve_response_xc`,
which calls `ferric_dft::lr_kernel::resolve_response_xc` for a named
functional. That maps it to the exchange coefficients
$(c_{\rm sr}, c_{\rm lr}, \omega)$ (a global hybrid has
$c_{\rm sr} = c_{\rm lr} = c_{\rm HF}$, $\omega = 0$) and a kernel, and refuses
functionals whose response would be incomplete. (The other entry point,
`resolve_singlet_response_xc`, also refuses $\omega \ne 0$; it serves drivers
that build only full-Coulomb exchange, not `run_tddft`.)

| Functional class | Kernel | Status |
|---|---|---|
| HF (no functional) | none | CIS / TDHF; a `c_hf` override other than 1 is refused |
| LDA, GGA, global hybrid | `GgaFxcKernel` | accepted |
| τ-only meta-GGA (SCAN, r2SCAN, TPSS) | `MggaFxcKernel` | accepted |
| range-separated hybrid (`c_sr`, `c_lr`, ω) | `GgaFxcKernel` | accepted; exchange of §8 |
| meta-GGA with no libxc second derivative | none | refused |
| VV10-carrying functionals | none | refused (nonlocal response not in the kernel) |

A `c_hf` override is refused on an HF reference unless it is 1, and on a
range-separated functional at any value (it would not say which range it
replaces); on a global functional it replaces $c_{\rm HF}$.

The kernel is evaluated on `TddftConfig.grid`, the same grid as the reference
SCF. For meta-GGAs the default is 150 × 434 with NWChem radial pruning
(`ferric_dft::grid::response_default_grid`), because the τ kernel is more
grid-sensitive than LDA/GGA.

Singlet and triplet ($M_S = 0$) excitations of a closed-shell reference are
both available (§9). UHF/UKS references take the spin-conserving problem of
§10 with the same functional gate; ROHF/ROKS references are refused (§10).

---

## 6. Cost

$\mathbf A$, $\mathbf B$, $\mathbf A\pm \mathbf B$, $T$ and $M$ are dense $d\times d$ with $d = n_o n_v$,
so memory grows as $N^4$ and each `eigh` as $N^6$; Casida performs two. The
matrix-free Davidson solver (`src/davidson.rs`) holds only the subspace.
The memory plan counts the co-resident
$d^2$ matrices and the RI tensors before allocating, and a run over budget is
an error.

---

## 7. Validation

- Cross-code: `tests/validation_tddft.rs` against PySCF `TDA`/`TDDFT`
  (reference generator `scripts/validation/gen_tddft_refs.py`); grades and
  measured errors are in `site/src/reference/validation.md`, rows `tda`
  and `tddft`.
- Range-separated hybrids and triplets: `tests/validation_tddft_rsh.rs`
  against PySCF `get_ab` (RHF for singlets, the UKS spin blocks for
  triplets) with the response integrals, the SCF fit and the grid matched
  (generator `scripts/validation/gen_tddft_rsh_refs.py`); the exact limits
  of §8 and the singlet + triplet identity of §9 are unit tests.
- Library anchor: TDA equals `ferric_gw::tddft::run_tda_dft`, which shares
  the kernel block; with $K = 0$ and $c_{\rm HF} = 1$ it reduces to CIS.
- Prototype: `scripts/proto_tddft.py` (hand-built TDA/Casida matrices from
  PySCF integrals against PySCF `tdscf`, water/cc-pVDZ).
- Proof: `scripts/tddft_response_sympy.py`. Symbolically: the singlet and
  triplet spin adaptation of $\mathbf A$ and $\mathbf B$ (§1, §9), the kernel factor 2 (§1),
  the scalar Casida reduction and its normalization (§3) and the
  range-separated exchange split (§8). Numerically, on random models with
  RI tensors and a grid kernel: the spin-orbital $\mathbf A$, $\mathbf B$ against the code's
  singlet and triplet blocks; CIS by brute force over determinants
  (Jordan-Wigner) against $\mathbf A$, closed and open shell; the Casida roots and
  $X+Y$ against the full non-Hermitian problem; the sum rule of §4; the
  unrestricted blocks of §10 and their closed-shell limit; the ⟨S²⟩ formula
  of §11 against a Fock-space $\lVert S_+\Psi\rVert^2$ and the Casida $X$
  extraction against the full eigenvector. With
  `DERIV_CROSSCHECK=1` and PySCF importable, H$_2$O/STO-3G RHF (TDA, TDHF, singlet and triplet) and
  H$_2$O$^+$/STO-3G UHF (TDA, TDHF) energies and oscillator strengths.
- Refusals: `tests/refusals.rs`; memory plan: `tests/memory_budget.rs`.

---

## 8. Range-separated hybrids

A range-separated hybrid splits the exchange operator at $\omega$:
$1/r = \mathrm{erfc}(\omega r)/r + \mathrm{erf}(\omega r)/r$, with exact
exchange fraction $c_{\rm sr}$ on the short-range and $c_{\rm lr}$ on the
long-range part (libxc `cam_alpha + cam_beta` and `cam_alpha`;
`ferric_dft::libxc::CamCoeffs`). The exact-exchange energy is

$$E_x^{\rm HF} = -\tfrac12\sum_{\sigma}\sum_{ij\in\sigma}
\left[c_{\rm sr}(ij|{\rm erfc}|ji) + c_{\rm lr}(ij|{\rm erf}|ji)\right].$$

Its second derivative gives the exchange terms of §1 with
$c_{\rm HF}(pq|rs)$ replaced by

$$c_{\rm sr}(pq|{\rm erfc}|rs) + c_{\rm lr}(pq|{\rm erf}|rs)
= c_{\rm sr}(pq|rs) + (c_{\rm lr}-c_{\rm sr})(pq|rs)_\omega ,$$

using ${\rm erfc} = 1 - {\rm erf}$ and writing $(pq|rs)_\omega$ for the
erf-attenuated integral. So, for both spins,

$$A_{ia,jb} \mathrel{-}= c_{\rm sr}(ij|ab) + \Delta c\,(ij|ab)_\omega,\qquad
B_{ia,jb} \mathrel{-}= c_{\rm sr}(ib|ja) + \Delta c\,(ib|ja)_\omega,\qquad
\Delta c = c_{\rm lr}-c_{\rm sr}.$$

The Coulomb ($2(ia|jb)$) and f_xc terms are unchanged; the f_xc kernel is
libxc's for the same functional, whose semilocal exchange is already the
short-range-attenuated one.

**Which integrals are fitted.** The full-Coulomb exchange reuses the
Coulomb-metric tensors $B_{ov}, B_{oo}, B_{vv}$ of §1, which the Coulomb term
needs anyway. The long-range term takes a second set,

$$B^{\omega,P}_{pq} = \sum_Q [\mathbf M_\omega]_{PQ}\,(Q|{\rm erf}|pq),\qquad
\mathbf M_\omega^{\mathsf T}\mathbf M_\omega = \mathbf V_\omega^{+},\quad
[\mathbf V_\omega]_{PQ} = (P|{\rm erf}|Q),$$

over the same auxiliary basis, with $\mathbf V_\omega^{+}$ the
pseudo-inverse that drops eigenvalues below $10^{-10}$
(`ferric_mp2::rimp2::metric_inverse_sqrt`, eigendecomposition for the erf
operator). Only the $oo$ and $vv$ blocks (TDA) and the $ov$ block (Casida
$\mathbf B$) are built. The fit is in the erf metric because:

* it is the variational (Dunlap) fit for the erf operator: the error of
  $(pq|{\rm erf}|rs) \approx \sum_P B^{\omega,P}_{pq}B^{\omega,P}_{rs}$ is
  second order in the fitting residual measured in the erf norm, and the
  approximate block is a Gram matrix, so it stays symmetric positive
  semidefinite as the exact one is;
* a Coulomb-metric fit of erf integrals has no symmetric $B^{\mathsf T}B$
  form (it needs $(pq|{\rm erf}|P)\,\mathbf V^{-1}(\ldots)\mathbf V^{-1}(Q|{\rm erf}|rs)$
  with the erf two-centre matrix in between), which neither the dense
  nor the matrix-free contraction is built for;
* PySCF's `with_range_coulomb` density fitting and ferric's long-range SCF
  fitter (`build_rsh_dfk_pair`) use the same metric.

$\mathbf V_\omega$ is nearly singular for a Coulomb-fitted auxiliary basis
(the erf kernel is smooth, so tight auxiliary functions are nearly
redundant), hence the eigendecomposition and cut rather than a Cholesky
factor.

The SCF fits the two ranges as $c_{\rm sr}K[{\rm erfc}] + c_{\rm lr}K[{\rm erf}]$
in the JK auxiliary basis; the response fits $c_{\rm sr}K[1/r] + \Delta c\,K[{\rm erf}]$
in the response auxiliary basis. The two are equal for exact integrals and
differ by fitting error, of the same kind as the existing difference between
the JK-fitted SCF and the RI-fitted response of a global hybrid.

**Limits that are exact.**

* $\omega = 0$: ${\rm erf}(0) = 0$, the long-range term is identically zero
  and the run is the global hybrid with $c_{\rm HF} = c_{\rm sr}$. The driver
  builds no erf tensors then, so the result is bitwise the global-hybrid one.
* $c_{\rm sr} = c_{\rm lr} = c$ at any $\omega$: $\Delta c = 0$ and the
  result is the global hybrid with $c_{\rm HF} = c$. The erf tensors are
  still built and contracted with the coefficient $0$, so the result is
  bitwise equal unless the long-range term is wrongly weighted (e.g. by
  $c_{\rm lr}$ instead of $\Delta c$).

**Products.** With $K^\omega_A(x)_{ia} = \sum_{jb}(ij|ab)_\omega x_{jb}$ and
$K^\omega_B$ likewise, the products of `matvec.rs` gain
$-\Delta c\,K^\omega_A$ in $Ax$, $-\Delta c\,(K^\omega_A+K^\omega_B)$ in
$(\mathbf A+\mathbf B)x$ and $-\Delta c\,(K^\omega_A-K^\omega_B)$ in $(\mathbf A-\mathbf B)x$, contracted
exactly as the Coulomb-metric exchange.

**Memory.** The second set adds $N_{\rm aux}(n_o^2+n_v^2)$
(and $N_{\rm aux}n_o n_v$ for Casida) resident doubles, declared
in the plan before the erf tensors are built.

---

## 9. Triplets

For a closed-shell reference the spin-orbital response matrices are, with
$\sigma,\tau\in\{\alpha,\beta\}$ and the same spatial orbitals for both spins,

$$A_{ia\sigma,jb\tau} = \delta_{ij}\delta_{ab}\delta_{\sigma\tau}\,\Delta_i^a
+ (ia|jb) + (ia|f_{\sigma\tau}|jb) - \delta_{\sigma\tau}\,c\,(ij|ab),$$

$$B_{ia\sigma,jb\tau} = (ia|jb) + (ia|f_{\sigma\tau}|jb) - \delta_{\sigma\tau}\,c\,(ib|ja),$$

($c(pq|rs)$ standing for the exchange combination of §8). Spin symmetry gives
$\mathbf A_{\alpha\alpha} = \mathbf A_{\beta\beta}$ and $\mathbf A_{\alpha\beta} = \mathbf A_{\beta\alpha}$, so
the $2d\times 2d$ problem block-diagonalizes on
$X^{S}=(X_\alpha+X_\beta)/\sqrt2$ (singlet) and
$X^{T}=(X_\alpha-X_\beta)/\sqrt2$ (the $M_S = 0$ triplet):

$$\mathbf A^{S} = \mathbf A_{\alpha\alpha}+\mathbf A_{\alpha\beta},\qquad \mathbf A^{T} = \mathbf A_{\alpha\alpha}-\mathbf A_{\alpha\beta},$$

and the same for $\mathbf B$. Written out,

$$A^{T}_{ia,jb} = \delta_{ij}\delta_{ab}\,\Delta_i^a - c\,(ij|ab) + K^{T}_{ia,jb},\qquad
B^{T}_{ia,jb} = -c\,(ib|ja) + K^{T}_{ia,jb},$$

$$K^{T}_{ia,jb} = (ia|f_{\alpha\alpha}-f_{\alpha\beta}|jb).$$

The Coulomb term cancels between the two spin blocks. Every term other than
the Coulomb term and the kernel is identical for the two spins, and
$\mathbf A-\mathbf B$ (where both cancel) is the same matrix for singlets and triplets.
The $M_S=\pm1$ components are degenerate with $M_S = 0$ and are not
computed.

**The kernel block.** `ferric_dft::lr_kernel` applies the spin-resolved AO
kernel to $(\delta D^\alpha,\delta D^\beta) = (\delta D, s\,\delta D)$ with
$s = +1$ (singlet) or $s = -1$ (triplet) and keeps
$\delta V_\alpha = \int (f_{\alpha\alpha} + s f_{\alpha\beta})\,\delta\rho$.
So $K^T$ is built by the same adapter as the singlet block with the sign of
the β perturbation flipped, and carries the singlet block's normalization.

**The matrix-free kernel.** At a restricted point, $\rho_\alpha=\rho_\beta$,
$\nabla\rho_\alpha = \nabla\rho_\beta = \mathbf g$. With the triplet
perturbation $\delta\rho_\beta = -\delta\rho_\alpha = -\rho_1$,
$\nabla\delta\rho_\beta = -\nabla\delta\rho_\alpha$,
$\delta\tau_\beta = -\delta\tau_\alpha = -\tau_1$ and
$\gamma_1 = \mathbf g\cdot\nabla\delta\rho_\alpha$, the variable perturbation in
$x = (\rho_\alpha,\rho_\beta,\sigma_{\alpha\alpha},\sigma_{\alpha\beta},\sigma_{\beta\beta},\tau_\alpha,\tau_\beta)$ is

$$\delta x = (\rho_1,\,-\rho_1,\,2\gamma_1,\,0,\,-2\gamma_1,\,\tau_1,\,-\tau_1)$$

($\delta\sigma_{\alpha\beta} = \mathbf g\cdot\nabla\delta\rho_\beta + \mathbf g\cdot\nabla\delta\rho_\alpha = 0$).
For the singlet it is $(\rho_1, \rho_1, 2\gamma_1, 2\gamma_1, 2\gamma_1, \tau_1, \tau_1)$. Both are
$\delta x = (\rho_1,\,s\rho_1,\,2\gamma_1,\,(1+s)\gamma_1,\,2s\gamma_1,\,\tau_1,\,s\tau_1)$, so row $m$ of
$\delta v = H\,\delta x$ is

$$\delta v_m = (H_{m\rho_\alpha} + sH_{m\rho_\beta})\,\rho_1
+ \left(2H_{m\sigma_{\alpha\alpha}} + (1+s)H_{m\sigma_{\alpha\beta}} + 2sH_{m\sigma_{\beta\beta}}\right) \gamma_1
+ (H_{m\tau_\alpha} + sH_{m\tau_\beta})\,\tau_1 .$$

The α potential's gradient coefficient is
$\mathbf w^\alpha = (2\delta v_{\sigma_{\alpha\alpha}} + \delta v_{\sigma_{\alpha\beta}})\,\mathbf g
+ 2v_{\sigma_{\alpha\alpha}}\nabla\delta\rho_\alpha + v_{\sigma_{\alpha\beta}}\nabla\delta\rho_\beta$,
that is $(2\delta v_{\sigma_{\alpha\alpha}} + \delta v_{\sigma_{\alpha\beta}})\,\mathbf g
+ (2v_{\sigma_{\alpha\alpha}} + s\,v_{\sigma_{\alpha\beta}})\nabla\delta\rho_\alpha$,
with $u^\alpha = \delta v_{\rho_\alpha}$ and $t^\alpha = \delta v_{\tau_\alpha}$ as
for the singlet (`ferric_dft::fxc::spin_coefficients`, pinned against the
spin-resolved `linearize_point` for both signs).

**Casida triplets and instability.** $\mathbf A^T+\mathbf B^T = \Delta - c(K_A+K_B) + 2K^T$
is not positive definite when the reference is triplet-unstable (RHF
ethylene is the textbook case), and the Casida problem then has
$\Omega^2 < 0$. Both solvers refuse a negative $\Omega^2$ instead of
dropping it from the root list (for TDA the matrix is $\mathbf A^T$ alone and has
no such failure mode).

**Oscillator strengths.** $\langle 0|\mathbf r|n\rangle$ is a spin-free
one-electron operator, so its matrix element with $X^T$ is
$\sum_{ia}(X_\alpha-X_\beta)/\sqrt2\cdot\langle i|\mathbf r|a\rangle$ summed over
equal and opposite spin amplitudes: zero. Triplet transition dipoles and
oscillator strengths are reported as exactly zero.

---

## 10. Unrestricted (UHF / UKS) references

For an unrestricted reference the α and β spin orbitals differ, and the
response is solved in the spin-conserving space of both spins at once:
compound indices $ia\alpha$ ($d_\alpha = n^\alpha_o n^\alpha_v$)
followed by $ia\beta$ ($d_\beta$), dimension $d_\alpha + d_\beta$. With all
orbitals of their own spin,

$$A_{ia\sigma,jb\tau} = \delta_{\sigma\tau}\delta_{ij}\delta_{ab}(\varepsilon^\sigma_a-\varepsilon^\sigma_i)
+ (i_\sigma a_\sigma|j_\tau b_\tau) + (i_\sigma a_\sigma|f_{\sigma\tau}|j_\tau b_\tau)
- \delta_{\sigma\tau}\,c\,(i_\sigma j_\sigma|a_\sigma b_\sigma),$$

$$B_{ia\sigma,jb\tau} = (i_\sigma a_\sigma|j_\tau b_\tau) + (i_\sigma a_\sigma|f_{\sigma\tau}|j_\tau b_\tau)
- \delta_{\sigma\tau}\,c\,(i_\sigma b_\sigma|j_\sigma a_\sigma),$$

$c(pq|rs)$ again standing for the exchange combination of §8 (global, HF or
range-separated). These are PySCF `tdscf.uhf.get_ab`'s blocks
($\mathbf A_{\alpha\alpha}$, $\mathbf A_{\alpha\beta}$, $\mathbf A_{\beta\beta}$, with
$\mathbf A_{\beta\alpha} = \mathbf A_{\alpha\beta}^{\mathsf T}$). TDA and the Casida
reduction of §2-3 apply unchanged to the $(d_\alpha+d_\beta)$-dimensional
matrices.

**Assembly.** Each same-spin block is the restricted builder of §1 over that
spin's orbitals and RI tensors with the Coulomb coefficient 1 instead of 2
(one $(ia|jb)$ per spin pair, no spin adaptation) and that spin's exchange.
The cross-spin Coulomb block is $B_{ov}^{\alpha\,\mathsf T}B_{ov}^\beta$ in
$\mathbf A$ and in $\mathbf B$ (real orbitals). There is no cross-spin exchange. The matrix-free
products use the same pieces: per spin the $\Delta$ and exchange products of
`matvec.rs` (Coulomb coefficient 0, no kernel), plus the spin-summed Coulomb
product $J(x)_\sigma = B_{ov}^{\sigma\,\mathsf T}(B_{ov}^\alpha x_\alpha + B_{ov}^\beta x_\beta)$
and the kernel; $(\mathbf A+\mathbf B)x$ takes $2J$ and $2F$, $(\mathbf A-\mathbf B)x$ neither.

**The kernel.** `ferric_dft::lr_kernel::UksKernel` is the spin-resolved AO
kernel at the UKS reference $(D^\alpha, D^\beta)$, screened per spin on its
own $\rho_\sigma$ as the UKS $V_{xc}$ is. `UksOvFxc` applies it per vector:
$\delta D^\sigma = \tfrac12(Y_\sigma + Y_\sigma^{\mathsf T})$ with
$Y_\sigma = C^\sigma_{\rm occ}X_\sigma C^{\sigma\,\mathsf T}_{\rm vir}$, then
$(Kx)_\sigma = C^{\sigma\,\mathsf T}_{\rm occ}\,\delta V_\sigma[\delta D^\alpha,\delta D^\beta]\,C^\sigma_{\rm vir}$.
A unit vector of spin $\tau$ gives column $jb\tau$ of both spin blocks,
$(ia\sigma|f_{\sigma\tau}|jb\tau)$, with the normalization of the restricted
block (§1): at a restricted point $K_{\alpha\alpha}\pm K_{\alpha\beta}$ is the
singlet / triplet block. The dense path applies the operator to the
identity, checks the asymmetry against `FXC_BLOCK_ASYMMETRY_TOL` and
symmetrizes. Every functional the restricted path accepts is accepted here,
τ-only meta-GGAs and hybrid meta-GGAs included (`MggaFxcKernel` is polarized);
VV10 stays refused.

**Occupations.** $n_\alpha, n_\beta$ come from the molecule's charge and
multiplicity, and the occupied orbitals are the lowest $n_\sigma$ of each
spin. A reference whose density is not built from those orbitals (a
non-aufbau MOM solution) is refused.

**Exact limit.** For a closed-shell molecule with $C_\alpha = C_\beta$ the
problem block-diagonalizes on $X_\alpha \pm X_\beta$ (§9): the unrestricted
roots are the union of the restricted singlets and triplets, with
$\langle S^2\rangle = 0$ and 2. The only differences are rounding (the
Coulomb term summed as $B^{\mathsf T}B + B^{\mathsf T}B$ instead of
$2B^{\mathsf T}B$) and the kernel's density screen (per spin instead of on
$\rho_\alpha+\rho_\beta$), which acts only at points at the density floor.

**ROHF / ROKS.** Refused. ROHF orbitals diagonalize one Roothaan-coupled
Fock operator, not separate α and β Fock operators, so the $\mathbf A$ above (with
orbital-energy differences on its diagonal) is not the response of that
reference; the ROKS response needs the ROHF orbital-rotation Hessian.

**Oscillator strengths.** Each spin is its own block, so there is no
$\sqrt2$: $\langle 0|\mathbf r|n\rangle = \sum_\sigma\sum_{ia}(X+Y)_{ia\sigma,n}\langle i_\sigma|\mathbf r|a_\sigma\rangle$,
$f_n = \tfrac23\Omega_n|\langle 0|\mathbf r|n\rangle|^2$ (PySCF `tdscf.uhf`).
For a closed shell this is §4's formula with $X_\alpha = X_\beta = X/\sqrt2$.

---

## 11. ⟨S²⟩ of each root

The roots of an unrestricted reference are not spin eigenfunctions. Each
root is assigned the wavefunction
$|\Psi\rangle = \sum_{ia\sigma}X_{ia\sigma}\,a^\dagger_{a\sigma}a_{i\sigma}|\Phi\rangle$
with unit-normalized $X$: the TDA eigenvector, or for Casida the $X$ part
$\tfrac12((X+Y)+(X-Y))$, $X-Y = (\mathbf A+\mathbf B)(X+Y)/\Omega$, renormalized (Casida
response defines no excited-state wavefunction; this is the particle-hole
part of the root). Then $\langle S^2\rangle = M^2 + M + \langle S_-S_+\rangle$,
$M = (n_\alpha-n_\beta)/2$.

With $O = C_\alpha^{\mathsf T}SC_\beta$ over all MOs ($S$ the AO overlap),
$S_+ = \sum_{p\in\alpha,q\in\beta}O_{pq}a^\dagger_{p\alpha}a_{q\beta}$, and
moving the β creator past the two α operators,

$$S_-S_+ = \sum_{sq\in\beta}P_{sq}E^\beta_{sq} - \sum_{pr\in\alpha}\sum_{sq\in\beta}O_{rs}O_{pq}\,E^\alpha_{pr}E^\beta_{sq},
\qquad P = O^{\mathsf T}O .$$

Matrix elements between single excitations follow from
$\langle\Phi_i^a|E_{pr}|\Phi_j^b\rangle = \delta_{ab}\delta_{ij}\gamma^0_{pr} + \delta_{ij}\delta_{pa}\delta_{rb} - \delta_{ab}\delta_{pj}\delta_{ri}$
within a spin, and from the factorization of an α-even times β-even operator
across spins: $\langle\Phi_{i\alpha}^{a}|E^\alpha_{pr}E^\beta_{sq}|\Phi_{\bar\jmath}^{\bar b}\rangle = \delta_{pa}\delta_{ri}\,\delta_{s\bar\jmath}\delta_{q\bar b}$,
with overbars marking β orbitals.
Writing $X_\alpha$ ($n_\alpha\times n^\alpha_{\rm vir}$) and $X_\beta$,
$w_\sigma = |X_\sigma|^2$ ($w_\alpha + w_\beta = 1$),
$Q = O_{:,{\rm occ}\beta}O_{:,{\rm occ}\beta}^{\mathsf T}$ (α×α),
$R = O_{{\rm occ}\alpha,:}^{\mathsf T}O_{{\rm occ}\alpha,:}$ (β×β), and for a
matrix $N$ over one spin $\pi(X, N) = {\rm tr}(X N_{vv} X^{\mathsf T}) - {\rm tr}(X^{\mathsf T}N_{oo}X)$:

$$\langle S_-S_+\rangle = \sum_{\bar\jmath}P_{\bar\jmath\bar\jmath} + \pi(X_\beta,P)
- \left[w_\alpha{\rm tr}\,Q_{oo} + \pi(X_\alpha,Q)\right]
- \left[w_\beta{\rm tr}\,R_{oo} + \pi(X_\beta,R)\right]
- 2\,{\rm tr}\!\left(X_\alpha^{\mathsf T}O_{oo}X_\beta O_{vv}^{\mathsf T}\right),$$

where $O_{oo}$ is the α-occupied × β-occupied and $O_{vv}$ the α-virtual ×
β-virtual block. With no excitation it reduces to the reference value
$M^2+M+\sum_{\bar\jmath}P_{\bar\jmath\bar\jmath} - \sum_{i,\bar\jmath}O_{i\bar\jmath}^2$ (PySCF
`UHF.spin_square`). For a closed shell ($O = 1$): singlets
($X_\alpha = X_\beta$) give 0 and triplets ($X_\alpha = -X_\beta$) give 2.

**Check.** The generator `scripts/validation/gen_tddft_open_shell_refs.py`
evaluates the formula in numpy and, for OH and NH2, also expands the same
$X$ into a full determinant vector over the UHF orbitals and takes
$\langle S^2\rangle$ with `pyscf.fci.spin_op.spin_square` (from the 2-RDMs,
an independent construction); it asserts agreement to $10^{-10}$ on every
root. The Rust test compares ferric's values to the recorded ones.
`scripts/tddft_response_sympy.py` (claims s2ref, s2, casidax) transcribes
`SpinOverlap::reference`, `SpinOverlap::excited` and `casida_x` and checks
them against $M^2 + M + \lVert S_+\Psi\rVert^2$ evaluated by Jordan-Wigner
on the determinant space, for a general $O$ (α and β MOs spanning different
AO subspaces, so $P \ne 1$) and an orthogonal one.

---

## 12. Matrix-free Davidson solvers

`src/davidson.rs` solves the same problems as `solve_tda` and
`solve_casida` with only products by $\mathbf A$ (TDA) or by
$\mathbf A\pm\mathbf B$ (Casida), supplied by `matvec.rs` (closed shell) or
the unrestricted operator. The preconditioner is the diagonal
$D_{ia} = \varepsilon_a - \varepsilon_i$.

**Solver choice.** `TddftSolver::Auto` takes the dense path when
$d \le 200$ (`AUTO_DENSE_MAX_DIM`) or $4 n_{\rm roots} \ge d$, and the
Davidson path otherwise; `dense` and `davidson` force one.

**TDA.** Rayleigh-Ritz on an orthonormal subspace $V$,
$H = V^{\mathsf T}\mathbf A V$, gives Ritz pairs $(\theta, x = Vy)$ with
residual $r = \mathbf A x - \theta x$. The correction is Olsen's,

$$t = P r - \epsilon P x,\qquad P = (\theta - D)^{-1},\qquad
\epsilon = \frac{x^{\mathsf T}Pr}{x^{\mathsf T}Px},$$

so $x^{\mathsf T}t = 0$ (`diagonal_correction`). The plain $Pr$ stalls on a
root whose eigenvector lies on a block of equal orbital differences equal to
$\theta$, where $Pr \approx -x$. $|\theta - D|$ is floored at
`PRECOND_FLOOR`.

**Casida.** One subspace $V$ carries both $X+Y$ and $X-Y$. With
$M_\pm = V^{\mathsf T}(\mathbf A\pm\mathbf B)V$, the Galerkin conditions
$M_+u = \Omega v$, $M_-v = \Omega u$ give $M_-M_+u = \Omega^2u$. With
$L = M_-^{1/2}$ and $u = Lz$,

$$L M_+ L\, z = \Omega^2 z,$$

the subspace image of the dense $T(\mathbf A+\mathbf B)T$ of §3. For unit
$z$, $u = \Omega^{-1/2}Lz$ and $v = M_+u/\Omega$ satisfy
$u^{\mathsf T}v = z^{\mathsf T}LM_+Lz/\Omega^2 = 1$, the normalization
$(X+Y)\cdot(X-Y) = 1$ of §3 (`casida_ritz`). A non-positive eigenvalue of
$M_-$ or a negative $\Omega^2$ is an RPA instability and an error, as on the
dense path. The residuals are

$$r_+ = (\mathbf A+\mathbf B)(X+Y) - \Omega(X-Y),\qquad
r_- = (\mathbf A-\mathbf B)(X-Y) - \Omega(X+Y),$$

and the reported residual is $\sqrt{(\lVert r_+\rVert^2 + \lVert r_-\rVert^2)/2}$:
with $r_X = (r_+ + r_-)/2$ and $r_Y = (r_+ - r_-)/2$ the residual of the
full problem, $\lVert r_X\rVert^2 + \lVert r_Y\rVert^2 = (\lVert r_+\rVert^2 + \lVert r_-\rVert^2)/2$.
The corrections solve the per-element model
$\begin{pmatrix}D & -\Omega\\ -\Omega & D\end{pmatrix}\delta = -r$ with
the Olsen projection against $q = (X-Y;\,X+Y)$ (`casida_correction`), and
both $\delta_+$ and $\delta_-$ join the subspace.

**Guesses and stopping.** The guesses are unit vectors on the lowest
$\max(2n_{\rm roots}, n_{\rm roots}+8)$ diagonal elements, extended over any
element tied with the last within $10^{-8}$ Ha so a degenerate pair is never
split, each given a small deterministic dense part so the subspace reaches
every point-group irrep (`guess_indices`, `guess_vectors`). The targets are
the lowest $n_{\rm roots}$ Ritz values above $10^{-10}$ Ha, the dense path's
cut. Defaults: `conv_tol` $= 10^{-6}$ on the residual norm, `max_iter`
$= 100$, a subspace of $\max(n_{\rm guess} + 10n_{\rm roots}, 50)$ columns
before collapse. An unconverged root after `max_iter` is an error.

**Check.** `tda_matches_dense_eigh`, `casida_matches_dense_solve`,
`root_in_a_block_no_guess_touches_is_found`, `unconverged_root_is_an_error`
and `non_positive_a_minus_b_is_an_rpa_instability` in `src/davidson.rs`;
the RSH, triplet and open-shell validation tests run both solvers against
PySCF.
