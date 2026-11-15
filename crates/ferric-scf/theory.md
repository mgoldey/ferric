# Self-Consistent Field (SCF) Theory

This module implements the Restricted Hartree-Fock (RHF) method and associated optimization techniques.

## Hartree-Fock Equations
The RHF energy is given by:
$$E_{HF} = 2 \sum_i H_{ii} + \sum_{ij} [2(ii|jj) - (ij|ij)] + V_{nn}$$
where $H$ is the core Hamiltonian.

The Fock matrix in the AO basis is:
$$F_{\mu\nu} = H_{\mu\nu} + \sum_{\lambda\sigma} D_{\lambda\sigma} [2(\mu\nu|\lambda\sigma) - (\mu\lambda|\nu\sigma)]$$
where $D_{\lambda\sigma} = \sum_i C_{\lambda i} C_{\sigma i}$ is the density matrix.

## Hartree-Fock energy functionals (RHF, UHF, ROHF)

These are the energy expressions, Fock matrices and stationarity conditions
that `solve_rhf` (`src/rhf.rs`), `solve_uhf` (`src/uhf.rs`) and `solve_rohf`
(`src/rohf.rs`) evaluate. Proof: `scripts/hf_energy_sympy.py`.

Conventions. Real AO basis $\{\chi_\mu\}$ with overlap $S$, core Hamiltonian
$h = T + V_{ne}$ (plus $V_{\rm ECP}$ and external terms when present), and
real two-electron integrals $(\mu\nu|\lambda\sigma)$ with 8-fold symmetry.
Orbitals $C$ are $S$-orthonormal, $C^{\mathsf T} S C = 1$. Indices $i,j$ run
over occupied orbitals, $a,b$ over virtual ones, $p,q$ over all. The Coulomb
and exchange builders (`fock.rs`, `JBuilder`/`KBuilder`) compute
$$J(D)_{\mu\nu} = \sum_{\lambda\sigma} (\mu\nu|\lambda\sigma) D_{\lambda\sigma},
\qquad K(D)_{\mu\nu} = \sum_{\lambda\sigma} (\mu\lambda|\nu\sigma) D_{\lambda\sigma}.$$
Both are linear and, by the 8-fold symmetry, satisfy
$\mathrm{tr}[A\,J(B)] = \mathrm{tr}[B\,J(A)]$ and the same for $K$.

**RHF.** The code's closed-shell density carries the occupation 2,
$D = 2\,C_o C_o^{\mathsf T}$ (so it is twice the $D$ of the section above).
The energy is
$$E[D] = \mathrm{tr}(D h) + \tfrac12 \mathrm{tr}[D\,J(D)] - \tfrac14 \mathrm{tr}[D\,K(D)] + E_{nn}.$$
Differentiating with the trace symmetry gives the Fock matrix
$$F = \frac{\partial E}{\partial D} = h + J(D) - \tfrac12 K(D),$$
and the energy in the form the loop evaluates (`e_elec_no_xc`):
$$E = \tfrac12 \mathrm{tr}[D (h + F)] + E_{nn}.$$
Making $E$ stationary under $C \to C + \delta C$ with the constraint
$C^{\mathsf T} S C = 1$ gives the Roothaan equations $F C = S C \varepsilon$,
equivalently $F D S - S D F = 0$ (the DIIS error vector below). In the MO
basis the energy is $2\sum_i h_{ii} + \sum_{ij}[2(ii|jj) - (ij|ij)] + E_{nn}$,
the expression of the section above.

**UHF.** Per-spin densities $D_\sigma = C_{\sigma o} C_{\sigma o}^{\mathsf T}$,
$D = D_\alpha + D_\beta$:
$$E = \mathrm{tr}(D h) + \tfrac12 \mathrm{tr}[D\,J(D)] - \tfrac12 \sum_\sigma \mathrm{tr}[D_\sigma K(D_\sigma)] + E_{nn},$$
$$F_\sigma = \frac{\partial E}{\partial D_\sigma} = h + J(D) - K(D_\sigma),\qquad
E = \tfrac12 \sum_\sigma \mathrm{tr}[D_\sigma (h + F_\sigma)] + E_{nn}.$$
Stationarity gives the Pople-Nesbet equations
$F_\sigma C_\sigma = S C_\sigma \varepsilon_\sigma$. At $D_\alpha = D_\beta = D/2$
the UHF energy is the RHF energy of $D$ and $F_\alpha = F_\beta = F$.

**ROHF.** One orbital set $C$ with columns closed ($c$), open ($o$, alpha
only) and virtual ($v$); occupation vectors $n_\alpha = (1_c, 1_o, 0_v)$,
$n_\beta = (1_c, 0_o, 0_v)$, and
$$D_\beta = C_c C_c^{\mathsf T},\qquad D_\alpha = D_\beta + C_o C_o^{\mathsf T},\qquad D_\sigma = C\,\mathrm{diag}(n_\sigma)\,C^{\mathsf T}.$$
The energy is the UHF functional at these densities, with the same
$F_\sigma = h + J(D_\alpha + D_\beta) - K(D_\sigma)$; `solve_rohf` evaluates
exactly the UHF trace formula. In MO integrals this is the high-spin energy
$$E = 2\sum_c h_{cc} + \sum_t h_{tt} + \sum_{cc'}[2J_{cc'} - K_{cc'}]
+ \sum_{ct}[2J_{ct} - K_{ct}] + \tfrac12 \sum_{tu}[J_{tu} - K_{tu}] + E_{nn},$$
with $t,u$ open, $J_{pq} = (pp|qq)$ and $K_{pq} = (pq|pq)$.

Orbital gradient. Write $C \to C e^{\kappa}$ with $\kappa$ real antisymmetric
and $f_\sigma = C^{\mathsf T} F_\sigma C$. In the MO basis
$\delta D_\sigma = \kappa\,n_\sigma - n_\sigma \kappa$, i.e.
$(\delta D_\sigma)_{pq} = (n_{\sigma q} - n_{\sigma p})\kappa_{pq}$, and
$\delta E = \sum_\sigma \mathrm{tr}(f_\sigma \delta D_\sigma)$. A rotation
inside $c$, $o$ or $v$ changes neither density. For the three remaining
blocks (row index the orbital rotated into)
$$\frac{\partial E}{\partial \kappa_{pq}} = 2 g_{pq},\qquad
g_{vc} = (f_\alpha + f_\beta)_{vc},\quad g_{vo} = (f_\alpha)_{vo},\quad g_{oc} = (f_\beta)_{oc}.$$
`rohf_gradient_mo` builds exactly these blocks, and the DIIS error is
$S C (g - g^{\mathsf T}) C^{\mathsf T} S$. ROHF convergence means $g = 0$.

Effective Fock. `roothaan_fock_inner` is PySCF's `get_roothaan_fock` with the
Guest-Saunders choice. With $F_c = (F_\alpha + F_\beta)/2$ and the projectors
$P_c = D_\beta S$, $P_o = (D_\alpha - D_\beta) S$, $P_v = 1 - D_\alpha S$,
$$R = M + M^{\mathsf T},\quad
M = \tfrac12 P_c^{\mathsf T} F_c P_c + \tfrac12 P_o^{\mathsf T} F_c P_o + \tfrac12 P_v^{\mathsf T} F_c P_v
+ P_o^{\mathsf T} F_\beta P_c + P_o^{\mathsf T} F_\alpha P_v + P_v^{\mathsf T} F_c P_c .$$
Since $C^{\mathsf T} S P_c^{\mathsf T} X P_c S C$ is the $cc$ block of $C^{\mathsf T} X C$
(and likewise for $o$, $v$), $C^{\mathsf T} R C$ has the block table

| | $c$ | $o$ | $v$ |
|---|---|---|---|
| $c$ | $F_c$ | $F_\beta$ | $F_c$ |
| $o$ | $F_\beta$ | $F_c$ | $F_\alpha$ |
| $v$ | $F_c$ | $F_\alpha$ | $F_c$ |

The off-diagonal blocks are $g/2$ ($vc$), $g$ ($vo$, $oc$), so
$R C = S C \varepsilon$ at self-consistency is the same condition as $g = 0$.
The diagonal blocks are a canonicalization choice: they fix the orbital
energies (eigenvalues of $R$) and the orbitals inside each block, but not the
energy or the densities. There is no user knob for the coupling.

| equation | code |
|---|---|
| $J$, $K$ conventions | `fock.rs` (`JBuilder`, `KBuilder`) |
| RHF $F$, $E$ | `rhf.rs` `solve_rhf` (`f.assign(&h)`, `e_elec_no_xc`) |
| UHF $F_\sigma$, $E$ | `uhf.rs` `solve_uhf` |
| ROHF $E$, $R$, $g$ | `rohf.rs` `solve_rohf`, `roothaan_fock_inner`, `rohf_gradient_mo` |

The script proves $F = \partial E/\partial D$ (RHF, both UHF spins)
symbolically and by central differences, the closed-shell limit, the
$\Gamma$ forms used by the gradient, the block table, $\partial E/\partial\kappa = 2g$
by differences of $E(C e^\kappa)$ at a random orbital set, the stationarity of
the converged ROHF orbitals under $c$-$o$, $c$-$v$ and $o$-$v$ rotations, the
high-spin MO energy, and $E_{\rm UHF} \le E_{\rm ROHF}$.

## Hartree-Fock nuclear gradient

`rhf_gradient`, `uhf_gradient` and `rohf_gradient` (`src/gradient.rs`) return
$\partial E/\partial x$ for every nuclear coordinate $x$ from a converged SCF.
Proof: `scripts/hf_gradient_sympy.py`. This section covers the exact-integral
route; with density-fitted J/K the two-electron term differentiates the
fitted energy instead (`src/df_gradient.rs`).

**Derivation.** The energy depends on $x$ through the integrals $h(x)$,
$(\mu\nu|\lambda\sigma)(x)$, $S(x)$, $E_{nn}(x)$ and through $C(x)$. Form the
Lagrangian $L = E[C] - \mathrm{tr}[\varepsilon\,(C_o^{\mathsf T} S C_o - 1)]$
with a symmetric multiplier $\varepsilon$ (one per occupied pair and spin).
At the SCF solution $L$ is stationary in $C$, so
$\mathrm d E/\mathrm d x = \partial L/\partial x$ at fixed $C$:
$$E^x = E_{nn}^x + \sum_{\mu\nu} D_{\mu\nu} h^x_{\mu\nu}
+ \sum_{\mu\nu\lambda\sigma} \Gamma_{\mu\nu\lambda\sigma} (\mu\nu|\lambda\sigma)^x
- \sum_{\mu\nu} W_{\mu\nu} S^x_{\mu\nu},$$
where $W$ is the multiplier in the AO basis and $\Gamma$ is the two-particle
density with $\sum \Gamma (\mu\nu|\lambda\sigma) = E_{2e}$:

| reference | $\Gamma_{\mu\nu\lambda\sigma}$ (`gamma`, `gamma_uhf`) | $W$ |
|---|---|---|
| RHF | $\tfrac12 D_{\mu\nu}D_{\lambda\sigma} - \tfrac14 D_{\mu\lambda}D_{\nu\sigma}$ | $2\,C_o\,\mathrm{diag}(\varepsilon_o)\,C_o^{\mathsf T}$ (`build_energy_weighted_density`) |
| UHF | $\tfrac12 D_{\mu\nu}D_{\lambda\sigma} - \tfrac12\sum_\sigma D^\sigma_{\mu\lambda}D^\sigma_{\nu\sigma}$ | $\sum_\sigma C_{\sigma}\,\mathrm{diag}(f_\sigma \varepsilon_\sigma)\,C_\sigma^{\mathsf T}$ (`build_energy_weighted_density_uhf`) |
| ROHF | UHF form at the ROHF $D_\alpha, D_\beta$ | $\mathrm{sym}(D_\alpha F_\alpha D_\alpha + D_\alpha F_\beta D_\beta)$ (`rohf_energy_weighted_density`) |

For RHF and UHF the canonical orbitals diagonalize the multiplier, so $W$ is
the occupation-weighted eigenvalue sum; $f_\sigma$ are the occupations
recovered from the stored density (1 for aufbau, fractions under
`fractional_occ`).

For ROHF one orbital set carries two Focks. Stationarity of $L$ gives
$\partial E/\partial C = \sum_\sigma 2 F_\sigma C\,\mathrm{diag}(n_\sigma) = 2 S C \varepsilon$,
so over occupied orbitals $\varepsilon_{ij} = \sum_\sigma n_{\sigma j} (f_\sigma)_{ij}$.
In the AO basis $W = C_{\rm occ}\,\varepsilon\,C_{\rm occ}^{\mathsf T} = \sum_\sigma D_\alpha F_\sigma D_\sigma$,
because every occupied orbital holds an alpha electron
($C_{\rm occ} C_{\rm occ}^{\mathsf T} = D_\alpha$). Its antisymmetric part is
$(\varepsilon_{tc} - \varepsilon_{ct}) = (f_\beta)_{oc}$, which is zero at
convergence; `sym()` only removes convergence noise. The eigenvalue form
$C\,\mathrm{diag}(2\varepsilon_c, \varepsilon_o)\,C^{\mathsf T}$ of $R$ is not
this multiplier: it drops the closed-open coupling and is wrong whenever the
open shell couples to the closed shells (the script's negative control misses
by $5\times10^{-2}$).

`oneelectron_gradient` evaluates $E_{nn}^x + \sum D h^x - \sum W S^x$ (with
$D = D_\alpha + D_\beta$ for open shells) and `twoelectron_gradient` /
`twoelectron_gradient_uhf` the $\Gamma$ term; `ecp_gradient` adds
$\sum D\,V_{\rm ECP}^x$.

**Check.** A toy system whose integrals are smooth (quadratic) functions of
two parameters, with a non-orthogonal $S$: the SCF is re-solved at displaced
parameters and the energy differenced with a fourth-order stencil. The
analytic formula with the code's $W$ and $\Gamma$ matches to $3\times10^{-11}$
(RHF), $2\times10^{-11}$ (UHF) and $2\times10^{-11}$ (ROHF).

## Hartree-Fock nuclear Hessian

`rhf_hessian`, `uhf_hessian` (`src/hessian.rs`) and `rohf_hessian`
(`src/hessian/rohf.rs`) differentiate the gradient above once more. Proof:
`scripts/hf_hessian_sympy.py`.

**Skeleton (terms 1-4).** At fixed $D$ and $W$:
$$H^{\rm skel}_{xy} = E_{nn}^{xy} + \sum D h^{xy} + \sum \Gamma\,(\mu\nu|\lambda\sigma)^{xy} - \sum W S^{xy},$$
with the $D$, $\Gamma$, $W$ of the gradient table (ROHF uses its own $W$).

**Orbital response (term 5), RHF and UHF.** Write $C^y = C U^y$. The
differentiated orthonormality is $U + U^{\mathsf T} + S^y_{\rm MO} = 0$ with
$S^y_{\rm MO} = C^{\mathsf T} S^y C$. The occupied-occupied antisymmetric part
does not change $D$, so the code fixes $U^y_{ij} = -\tfrac12 S^y_{ij}$. With
$\mathrm{occ} = 2$ (RHF, one spin) or $1$ (each UHF spin):
$$D_1(U) = \mathrm{occ}\,(C_v U C_o^{\mathsf T} + C_o U^{\mathsf T} C_v^{\mathsf T}),\qquad
D^y_{oo} = -\mathrm{occ}\;C_o S^y_{oo} C_o^{\mathsf T},$$
and the first-order Fock at fixed density
$F^y = C^{\mathsf T}(h^y + G^y(D))\,C_o$, $G^y$ built from the integral
derivatives. Differentiating the $vo$ block of the SCF condition
($F_{ai} = 0$) gives the CPHF equations solved by `solve_cphf_block`
(right-hand side `cphf_rhs`):
$$(\varepsilon_a - \varepsilon_i)\,U^y_{ai} + G(D_1(U^y))_{ai}
= -\big(F^y_{ai} - \varepsilon_i S^y_{ai} + G(D^y_{oo})_{ai}\big),$$
with $G = J - \tfrac12 K$ (RHF) or the spin-coupled
$G_\sigma(D_\alpha, D_\beta) = J(D_\alpha + D_\beta) - K(D_\sigma)$ (UHF, one linear
system in $(U_\alpha, U_\beta)$). The occupied first-order multiplier is
(`finish_cphf`)
$$\varepsilon^y_{ij} = F^y_{ij} + [C_o^{\mathsf T} G(D^y_{oo} + D_1(U^y)) C_o]_{ij} - \tfrac12 S^y_{ij}(\varepsilon_i + \varepsilon_j).$$
The response term (`assemble_response`, PySCF `hess_elec`) is
$$R_{xy} = \sum_\sigma \Big[ 2\,\mathrm{occ} \sum_{p,i} (F^x - S^x\varepsilon)_{pi}\,U^y_{pi}
- \mathrm{occ} \sum_{ij} S^x_{ij}\,\varepsilon^y_{ij} \Big],$$
$p$ over all orbitals (the $oo$ rows carry $U_{oo} = -\tfrac12 S_{oo}$). It
is $\mathrm d/\mathrm dy$ of $\sum D F^x - \sum W S^x$ with $D^y$ and $W^y$
written through $U^y$ and $\varepsilon^y$. For RHF this is
$4\sum (F^x - S^x\varepsilon) U^y - 2\sum S^x_{oo}\varepsilon^y$. A
closed-shell UHF reproduces the RHF Hessian term by term.

**ROHF.** `hessian/rohf.rs` keeps the rotation $\kappa$ (blocks $vc$, $vo$,
$oc$) separate from the overlap part: $U = \kappa - \tfrac12 S^y_{\rm MO}$ and
$$(D^y_\sigma)_{\rm MO} = U n_\sigma + n_\sigma U^{\mathsf T},\quad
(D^S_\sigma)_{pq} = -\tfrac12 S^y_{pq}(n_{\sigma p} + n_{\sigma q}),\quad
(D^\kappa_\sigma)_{pq} = (n_{\sigma q} - n_{\sigma p})\kappa_{pq}.$$
Differentiating $g = 0$ along $y$ with
$\mathrm d f_\sigma/\mathrm dy = -\tfrac12(S f_\sigma + f_\sigma S) + [f_\sigma, \kappa] + C^{\mathsf T}(F^y_\sigma + G_\sigma(D^y))C$
gives (`Response::apply`, `Response::rhs`)
$$\Phi_\sigma = C^{\mathsf T}F^y_\sigma C + C^{\mathsf T}G_\sigma(D^S)C - \tfrac12(S f_\sigma + f_\sigma S)
+ C^{\mathsf T}G_\sigma(D^\kappa)C + [f_\sigma, \kappa],$$
$$(\Phi_\alpha + \Phi_\beta)_{vc} = 0,\qquad (\Phi_\alpha)_{vo} = 0,\qquad (\Phi_\beta)_{oc} = 0,$$
a linear system $A\kappa = -B$ whose operator keeps the full commutator
$[f_\sigma, \kappa]$ (the $f$ are not diagonal). The response column is
(`Response::column`, `w_response`)
$$R_{xy} = \sum_\sigma \mathrm{tr}[D^y_\sigma F^x_\sigma] - \mathrm{tr}[W^y S^x],$$
$$W^y = D^y_\alpha F_\alpha D_\alpha + D_\alpha F^y_\alpha D_\alpha + D_\alpha F_\alpha D^y_\alpha
+ D^y_\alpha F_\beta D_\beta + D_\alpha F^y_\beta D_\beta + D_\alpha F_\beta D^y_\beta,$$
with $F^y_\sigma$ the total derivative $h^y + G^y_\sigma(D) + G_\sigma(D^y)$.

**Check.** On the gradient toy system (two parameters), the analytic Hessian
matches a fourth-order difference of the analytic gradient with the SCF
re-solved: $3\times10^{-11}$ (RHF), $1\times10^{-9}$ (UHF), $1\times10^{-9}$
(ROHF); the closed-shell UHF Hessian equals the RHF one to $4\times10^{-16}$.

## Harmonic frequencies

`harmonic_frequencies` and the analytic routes (`src/frequencies.rs`,
`hessian.rs` `analytic_frequencies*`; Python `run_frequencies`) turn a
Cartesian Hessian into wavenumbers in `frequencies_from_cartesian_hessian`.
Proof: `scripts/hf_frequencies_sympy.py`.

The Hessian $H$ (Hartree/Bohr$^2$) is the analytic one above or the central
difference of the analytic gradient (default step $\delta = 5\times10^{-3}$
Bohr), symmetrized as $(H + H^{\mathsf T})/2$. Near a minimum
$V \approx \tfrac12 \Delta x^{\mathsf T} H \Delta x$ and $T = \tfrac12 \dot x^{\mathsf T} M \dot x$.
In mass-weighted coordinates $q = M^{1/2}\Delta x$ the equations of motion
decouple into the eigenproblem of
$$\tilde H = M^{-1/2} H M^{-1/2},\qquad \tilde H u_k = \omega_k^2 u_k .$$
Masses are converted from u to electron masses
(`AMU_TO_ELECTRON_MASS` $= 1/5.48579909065\times10^{-4}$, CODATA 2018), so
$\omega_k$ comes out in $E_h/\hbar$. The wavenumber is
$\tilde\nu = \omega E_h/(2\pi\hbar c) = \omega\,E_h/(hc)$, i.e.
`AU_FREQ_TO_CM` $= 219474.6313632$ cm$^{-1}$. A negative eigenvalue is
reported as a negative (imaginary) wavenumber. For a diatomic spring of force
constant $k$ the nonzero eigenvalue is $k(1/m_1 + 1/m_2) = k/\mu$, so
$\omega = \sqrt{k/\mu}$.

Projection. `translation_rotation_basis` builds the mass-weighted
translations $t_k[3A + i] = \sqrt{m_A}\,\delta_{ik}$ and rotations
$r_k[3A + \cdot] = \sqrt{m_A}\,(e_k \times (R_A - R_{\rm com}))$, and
orthonormalizes them by modified Gram-Schmidt, dropping vectors that vanish
(the axial rotation of a linear molecule), so the rank is 6, or 5 if linear
(cross-checked against `detect_linear`). The span does not depend on the
reference point: replacing $R_{\rm com}$ by any point adds a translation to
each rotation. The projected Hessian is $P\tilde H P$ with
$P = 1 - \sum_k t_k t_k^{\mathsf T}$, applied as rank-1 sweeps
$\tilde H \leftarrow \tilde H - t(t^{\mathsf T}\tilde H) - (\tilde H t)t^{\mathsf T} + t\,t^{\mathsf T}(t^{\mathsf T}\tilde H t)$,
which equal $P\tilde H P$ because the $t_k$ are orthonormal. Away from a
stationary point the rotations carry curvature (the gradient term); the
projection sets those eigenvalues to zero and leaves the vibrations alone at
a minimum. Eigenvectors are classified as translation/rotation by their
overlap with the subspace ($> 0.5$), not by eigenvalue size, so a small
imaginary mode at a saddle is kept. Normal modes are returned as Cartesian
displacements $x = M^{-1/2} u$ (not renormalized).

The script checks both constants against CODATA 2018, the diatomic
wavenumber against the SI route $(2\pi c)^{-1}\sqrt{k/\mu}$ (relative
$<10^{-9}$), the projector (rank, orthonormality, idempotence, annihilation of
the rigid modes, distance-preserving rotations, rank 5 for a linear
triatomic), and that the projection leaves the vibrations of a pair-spring
model unchanged at its minimum and zeroes six modes away from it.

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
`FixedGridKs::from_points` accepts only LDA, GGA and hybrid-GGA components;
a meta-GGA is refused before the SCF (`hessian/ks.rs`, "tau-dependent XC
nuclear derivatives"), so no $\tau$ terms appear below. An LDA is the
$f_\sigma = 0$ case ($\mathbf G = 0$, no $\mathbf g_x$ terms).
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
| $T G_h T^{\mathsf T}$, moving $\partial V/\partial x$ | `skeleton_hessian_moving`, `translation_fill`, `first_order_fock_moving`, `weight_terms` | `ks_moving_grid_hessian.rs::moving_xc_pieces_match_frozen_density_differences`, `moving_grid_mutants_each_fail_a_frozen_density_anchor` |

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
