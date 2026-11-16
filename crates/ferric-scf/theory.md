# Self-Consistent Field (SCF) Theory

This module implements the Restricted Hartree-Fock (RHF) method and associated optimization techniques.

## Hartree-Fock Equations
The closed-shell density carries the occupation 2,
$D_{\lambda\sigma} = 2\sum_i C_{\lambda i} C_{\sigma i}$, as in the code.
The RHF energy is
$$E_{HF} = 2 \sum_i h_{ii} + \sum_{ij} [2(ii|jj) - (ij|ij)] + E_{nn}
= \tfrac12\,\mathrm{tr}[D(h + F)] + E_{nn},$$
where $h$ is the core Hamiltonian, and the Fock matrix in the AO basis is
$$F_{\mu\nu} = h_{\mu\nu} + \sum_{\lambda\sigma} D_{\lambda\sigma}
\left[(\mu\nu|\lambda\sigma) - \tfrac12(\mu\lambda|\nu\sigma)\right],$$
i.e. $F = h + J(D) - \tfrac12 K(D)$. The next section derives these and the
open-shell forms.

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
$D = 2\,C_o C_o^{\mathsf T}$.
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
| UHF | $\tfrac12 D_{\mu\nu}D_{\lambda\sigma} - \tfrac12\sum_\tau D^\tau_{\mu\lambda}D^\tau_{\nu\sigma}$ | $\sum_\sigma C_{\sigma}\,\mathrm{diag}(f_\sigma \varepsilon_\sigma)\,C_\sigma^{\mathsf T}$ (`build_energy_weighted_density_uhf`) |
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

## RRHO thermochemistry

`rrho_thermochemistry` (`src/thermo.rs`; `FrequencyResult::thermochemistry`;
CLI `task = "frequencies"`; Python `FrequencyResult.thermo`) evaluates the
ideal-gas rigid-rotor / harmonic-oscillator model at temperature $T$ and
pressure $P$. Proof: `scripts/rrho_thermo_sympy.py`. The conventions are
those of PySCF `pyscf.hessian.thermo`.

Per molecule, with $k = k_B$, the energies follow from the partition
function $q$ as $E = kT^2\,\partial\ln q/\partial T$ and the entropies as
$S = k\ln q + E/T$, with the indistinguishability term $k(1 - \ln N)$
(Stirling) carried by the translations.

Translation. With $q_V = (2\pi M kT/h^2)^{3/2} V$ and $V/N = kT/P$,
$$q_{\rm tr} = \left(\frac{2\pi M k T}{h^2}\right)^{3/2}\frac{kT}{P},\qquad
E_{\rm tr} = \tfrac32 kT,\quad H_{\rm tr} = \tfrac52 kT,\quad
S_{\rm tr} = k\left(\tfrac52 + \ln q_{\rm tr}\right),$$
the Sackur-Tetrode entropy; $C_V = \tfrac32 k$, $C_P = \tfrac52 k$.

Rotation. The principal moments $I_a$ of the inertia tensor about the centre
of mass give $B_a = h/(8\pi^2 I_a)$ (Hz). With the rotational symmetry number
$\sigma$,
$$q_{\rm rot}^{\rm lin} = \frac{kT}{\sigma h B},\quad
E = H = kT,\quad S = k(1 + \ln q);\qquad
q_{\rm rot}^{\rm nonlin} = \frac{\sqrt\pi}{\sigma}\left(\frac{kT}{h}\right)^{3/2}
\frac{1}{\sqrt{B_A B_B B_C}},\quad E = H = \tfrac32 kT,\quad S = k(\tfrac32 + \ln q).$$
These are the high-temperature (classical) limits of the rigid-rotor level
sums; for a linear rotor the level sum is
$kT/(\sigma hB)\,[1 + y/3 + y^2/15 + \dots]$, $y = hB/kT$. A single atom has
no rotational part. Linearity is `detect_linear`, the same test the
frequency projection uses. $\sigma$ is an input (default 1); with
$\sigma = 1$ for a molecule whose true value is $\sigma$, $S_{\rm rot}$ is
too large by $k\ln\sigma$ and $G$ too low by $kT\ln\sigma$.

Vibration. Each real mode with wavenumber $\tilde\nu$ has
$\theta = hc\tilde\nu/k$ (`C2` $= 100\,hc/k$ in K per cm$^{-1}$) and
$x = \theta/T$. From $q = e^{-x/2}/(1 - e^{-x})$,
$$E_{\rm vib} = H_{\rm vib} = \sum\left[\tfrac12 k\theta + kT\,\frac{x}{e^x - 1}\right],\quad
S_{\rm vib} = k\sum\left[\frac{x}{e^x - 1} - \ln(1 - e^{-x})\right],\quad
C_V = k\sum\frac{x^2 e^x}{(e^x - 1)^2}.$$
The first sum is the zero-point energy. As $x \to 0$, $E - E_{\rm ZPE} \to kT$,
$C_V \to k$ and $S \to k(1 - \ln x)$ (the classical oscillator); as
$x \to \infty$, $E \to E_{\rm ZPE}$ and $S \to 0$. Imaginary modes
(negative wavenumbers) are left out, with a warning. The code evaluates
$x/(e^x-1)$ with `exp_m1` and $C_V$ as $x^2 e^{-x}/(1-e^{-x})^2$ so neither
soft nor stiff modes lose precision or overflow.

Electronic. $E = H = E_0$ (the SCF energy, with any dispersion correction) and
$S = k\ln(2S+1)$ from the spin multiplicity alone.

Totals are sums of the four parts and $G = H - TS$ for each. Energies are in
Hartree and entropies in Hartree/K, so $k$ is used as
$k_B/E_h = 3.166811563\times10^{-6}$ Hartree/K. Constants are CODATA 2018;
lengths go to metres with ferric's own Bohr radius
(`units::BOHR_RADIUS_ANGSTROM`, the value that read the Angstrom input) and
masses are the isotope-averaged standard atomic weights of the frequency
calculation.

The script derives each closed form from its partition function in sympy,
takes the high- and low-temperature limits, checks the transcribed formulas
for water against $\ln Q(T)$ summed over the vibrational levels and
differentiated numerically (relative $10^{-10}$), the rigid-rotor level sum
against its Euler-Maclaurin series, and the argon entropy against
NIST-JANAF (154.845 J/(mol K) at 298.15 K and 1 bar). The Python test
`test_thermo_ir.py` compares the full result with PySCF's `thermo` on the
same frequencies, geometry, masses and constants.

## Infrared intensities

`ir_spectrum` and `ir_intensities` (`src/infrared.rs`; CLI `task =
"frequencies"`; Python `FrequencyResult.ir_intensities`) give the harmonic
intensity of every normal mode. Proof: `scripts/ir_intensity_sympy.py`.

Intensity. For the $0 \to 1$ band of a harmonic mode with normal coordinate
$Q$ (mass-weighted, $Q = \sum_i \sqrt{m_i}\,\Delta x_i\,u_i$), the integrated
cross section is $\int\sigma\,d\omega = \pi\omega|\langle0|\mu|1\rangle|^2/(3\varepsilon_0\hbar c)$
(isotropic average included) and $|\langle 0|Q|1\rangle|^2 = \hbar/(2\omega)$.
Integrating over wavenumber ($d\omega = 2\pi c\,d\tilde\nu$) and per mole,
$$A_k = \frac{N_A}{12\,\varepsilon_0 c^2}\left|\frac{\partial\boldsymbol\mu}{\partial Q_k}\right|^2,$$
the Napierian integrated absorption coefficient in m/mol. $\hbar$ and
$\omega$ cancel: the result is the classical one, and because the
$n \to n+1$ band is $n+1$ times stronger, the Boltzmann-weighted net
absorption $\sum_n (p_n - p_{n+1})(n+1) = 1$ makes it independent of $T$.

Units. With $\partial\mu/\partial Q$ in atomic units, $e/\sqrt{m_e}$,
$$A_k\,[\text{km/mol}] = C\,\left|\frac{\partial\boldsymbol\mu}{\partial Q_k}\right|^2_{\rm a.u.},\qquad
C = \frac{N_A e^2}{12\,\varepsilon_0 c^2 m_e}\times10^{-3} = 1\,777\,097.7313$$
(`AU_TO_KM_PER_MOL`, CODATA 2018), i.e. 974.880 km/mol per $e^2/u$ and
42.2561 km/mol per (D/Å)$^2$/u.

Normal modes. `FrequencyResult::normal_modes` stores
$x_k = M^{-1/2}u_k$ with $u_k$ the unit eigenvectors of the mass-weighted
Hessian in electron masses, so $\sum_i m_i x_{k,i}^2 = 1$ and
$x_k = \partial R/\partial Q_k$. Then
$$\frac{\partial\mu_a}{\partial Q_k} = \sum_i P_{ai}\,x_{k,i},\qquad P_{ai} = \frac{\partial\mu_a}{\partial R_i},$$
with no further mass factor. $P$ is the atomic polar tensor ($3\times3N$, in $e$).

Dipole derivative. In a uniform field $\mathbf F$ ferric's energy is
$E(\mathbf F) = E(0) - \boldsymbol\mu\cdot\mathbf F - \tfrac12\mathbf F\alpha\mathbf F + \dots$
(a charge $q$ at $\mathbf R$ gains $-q\mathbf F\cdot\mathbf R$, see
"External point charges and fields"), so $\boldsymbol\mu = -\partial E/\partial\mathbf F$ and
$$P_{ai} = -\frac{\partial^2 E}{\partial F_a\,\partial R_i}
\approx -\frac{g_i(+h\mathbf e_a) - g_i(-h\mathbf e_a)}{2h},$$
the production route (`dipole_derivatives_field`): six SCF plus analytic
gradient evaluations in fields $\pm h$ ($h = 10^{-3}$ a.u.), independent of
the molecule size. The gradient of an SCF converged in the field carries the
orbital relaxation, so this is the relaxed (response) dipole derivative of
the SCF energy. Truncation is $h^2/6\,|\partial^4E/\partial F^3\partial R|$;
SCF noise is at most $K n\,\delta_D/h$ ($K$ the gradient's density
sensitivity, $n$ the basis size, $\delta_D$ the density threshold), so the
field SCFs run at $\delta_D \le 10^{-9}$ (`IR_DENSITY_CONV`). The validation
route (`dipole_derivatives_dipole_fd`) differences the SCF dipole
$\mu_a = \sum_A Z_A R_{A,a} - \mathrm{tr}(D\,r_a)$ along each Cartesian
displacement ($6N$ SCFs). Both satisfy the translational sum rule
$\sum_A P_{a,(A,b)} = Q\,\delta_{ab}$ for molecular charge $Q$ without
external point charges; `IrSpectrum::sum_rule_residual` records the
deviation. The dipole origin is fixed at 0; the intensities do not depend on
it because vibrations conserve the centre of mass.

Scope. The SCF references (RHF, UHF, ROHF and their Kohn-Sham variants).
Correlated kinds are refused: their dipole derivative would be the field
derivative of the correlated gradient, which is not validated. With
`[dft] dispersion` the modes come from the corrected Hessian and the dipole
derivative from the SCF alone (D3 does not depend on the field; MBD's
density dependence is not included in $P$).

The script derives $C$ from CODATA, the $0\to1$ band strength from the
matrix element (classical limit), the temperature independence, and the
field-route identity on a model $E(R,F)$; it checks the transcribed
contraction against differencing $\mu$ along each mode of a bent triatomic
model, the closed form $A = C q^2/\mu_{\rm red}$ for point charges on a
diatomic, and the field route's $O(h^2)$ truncation. The Rust test
`tests/ir_intensity_routes.rs` compares the two routes for RHF, RKS, UHF,
UKS and ROHF against a bar built from Richardson truncation estimates and
measured SCF-noise bounds.

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

## CFMM: the near and far fields as coded

This section states what `CfmmJ` (`src/cfmm.rs`, feature `cfmm-incomplete`)
computes; the short section above gives the idea.

Shell pairs. Every ordered shell pair $(s_1, s_2)$ is placed at the weighted
centre of its primitive products, $\sum w\,\mathbf P/\sum w$ with
$w = |c_1 c_2|\,e^{-\alpha\beta|\mathbf{AB}|^2/p}$ (`pair_center`), and given
the extent $\max\sqrt{-\ln\epsilon/p}$ over primitives whose prefactor is at
least $\epsilon$ (`extent_thresh`, default $10^{-10}$). The pairs are binned
into the leaves of an octree of uniform depth `max_level` (default 3) whose
root is the bounding cube of the pair centres enlarged by 1.1.

Near and far. A pair of leaves $(a, b)$ is far when
$$|\mathbf C_a - \mathbf C_b| > e_a + e_b + w_s\,h,$$
with $e_x$ the largest pair extent in leaf $x$, $h$ the leaf width and
$w_s$ = `ws_factor` (default 2). A near pair is contracted exactly,
$J_{\mu\nu} \mathrel{+}= \sum (\mu\nu|\lambda\sigma) D_{\lambda\sigma}$ over every
bra pair in $a$ and ket pair in $b$, with no permutational folding and no
screening. A far pair goes through Cartesian multipoles about the leaf
centres:
$$M^b_{ijk} = \sum_{\lambda\sigma\in b} D_{\lambda\sigma}\int \chi_\lambda\chi_\sigma\,(\mathbf r - \mathbf C_b)^{ijk}\,d\mathbf r,$$
$$L^a_{ijk} = \frac{1}{i!\,j!\,k!}\sum_{i'+j'+k' \le l_{\max}} M^b_{i'j'k'}\,\frac{(-1)^{i'+j'+k'}}{i'!\,j'!\,k'!}\,
\partial_x^{i+i'}\partial_y^{j+j'}\partial_z^{k+k'}\frac{1}{r}\Big|_{\mathbf C_a - \mathbf C_b},$$
$$J_{\mu\nu} \mathrel{+}= \sum_{ijk} L^a_{ijk}\,M^{\mu\nu}_{ijk}(\mathbf C_a)$$
(`add_m2l`, $l_{\max}$ default 6). This is one flat level of leaf-to-leaf
M2L; there is no M2M or L2L pass and the far-field work is $O(n_{\mathrm{leaf}}^2)$.

Exact limit. Each pair sits in exactly one leaf, so with every leaf pair
treated as near (`force_all_near_field`) every quartet is counted once and $J$
is the exact Coulomb matrix (`cfmm_matches_direct_j_in_the_trivial_limit`,
$10^{-10}$). With the far field engaged the error falls with $l_{\max}$
(`cfmm_far_field_engaged_matches_direct_j_on_alkane10`). `CfmmConfig::thresh`
is not read by the code: the near field is unscreened.

## Orthogonalization and starting guesses

Canonical orthogonalization. The overlap is diagonalized once,
$S = U\lambda U^{\mathsf T}$, and the columns with $\lambda_i \ge \tau$ are kept:
$$X_{\mu c} = U_{\mu i_c}\,\lambda_{i_c}^{-1/2}, \qquad X^{\mathsf T} S X = 1_m,\quad m \le N .$$
$\tau$ is absolute, $10^{-6}$ (`LINDEP_THRESH`), overridden by the environment
variable `FERRIC_LINDEP_THRESH` (`lindep_thresh`). Each iteration diagonalizes
$\tilde F = X^{\mathsf T} F X = \tilde C \varepsilon \tilde C^{\mathsf T}$ and
sets $C = X\tilde C$, so $C^{\mathsf T} S C = 1_m$ and the orbitals span the
retained space only (`canonical_orthogonalizer` in `src/rhf.rs`,
`diagonalize_rect` in `src/driver.rs`; the $N - m$ dropped columns are padded
with zeros and energy $10^6$). A guess that needs more occupied orbitals than
$m$ is an error. ROHF uses the symmetric $X = U\lambda^{-1/2}U^{\mathsf T}$
when nothing is dropped (`rohf_orthogonalizer`).

Core guess (`hcore_guess`): diagonalize $X^{\mathsf T} h X$ and occupy the
lowest $n_o$ orbitals, $D = 2C_oC_o^{\mathsf T}$.

Default guess (`[scf] guess = "minao"`, alias `"sad"`;
`minao_projection_guess`). $D$ is block diagonal over atoms, with zero
inter-atomic blocks; each atomic block is built in the target basis (there is
no projection from a minimal basis, despite the name):

- $Z \le 20$ with no $g$ or higher shell: the density of a free-atom SCF in
  that atom's basis (RHF for a singlet ground state, otherwise UHF with MOM
  from iteration 5, then spherically averaged by replacing each diagonal
  shell block by its mean diagonal times the unit matrix and zeroing the
  cross-$l$ blocks).
- Otherwise: a generalized-Wolfsberg–Helmholz block,
  $$F_{\mu\mu} = h_{\mu\mu}, \qquad F_{\mu\nu} = \tfrac12 K S_{\mu\nu}(h_{\mu\mu} + h_{\nu\nu}),\quad K = 1.75,$$
  diagonalized in the atom's own orthogonalizer, with occupation 2 on the
  lowest doubly occupied orbitals and the $2S$ unpaired electrons spread
  equally over the orbitals within $10^{-4}$ Ha of the first partly filled
  one (`atomic_hcore_density`).

The blocks are those of neutral atoms, so $\operatorname{tr}(DS) = N_{\mathrm{el}}$
holds for a neutral molecule only; the first Fock build and the aufbau step
restore the electron count.

Open-shell guesses start from the same closed-shell $D$. UHF sets
$D^\alpha = D^\beta = D/2$, builds $F^\alpha = F^\beta = h + J[D] - K[D/2]$
without exchange–correlation, diagonalizes per spin, and for $n_\alpha = n_\beta$
mixes the $\beta$ HOMO and LUMO by 0.1 rad (`uhf_guess_mos`,
`mix_beta_homo_lumo`). ROHF scales $D^\sigma = D\,n_\sigma/N_{\mathrm{el}}$ and
diagonalizes the Roothaan effective Fock matrix of the section on energy
functionals (`rohf_guess_mos`).

Secondary basis (`[scf] guess_basis`, `src/basis2.rs`). With
$P = S_{bb}^{+} S_{bs}$, $S_{bb}^{+} = XX^{\mathsf T}$ the lindep-filtered
inverse of the target overlap, the Fock projection carries the converged
small-basis densities, $D^\sigma = P D^\sigma_{\mathrm{small}} P^{\mathsf T}$,
and the overlap projection carries the occupied orbitals,
$C = P C_{o,\mathrm{small}}$, re-orthonormalized as $C(C^{\mathsf T}S_{bb}C)^{-1/2}$
(refused when the smallest eigenvalue of $C^{\mathsf T}S_{bb}C$ is below
$10^{-8}$).

## Direct and incremental J/K builds

Unique quartets. The direct builder (`DirectJK`, `src/direct_jk.rs`) loops
over canonical shell quartets $s_1 \ge s_2$, $s_3 \le s_1$,
$s_4 \le (s_3 = s_1\ ?\ s_2 : s_3)$, computes the full block
$(\mu\nu|\lambda\sigma)$, $\mu\in s_1,\nu\in s_2,\lambda\in s_3,\sigma\in s_4$,
and scatters each element to every distinct image of the 8-fold symmetry.
With $\delta_{12} = [s_1\ne s_2]$, $\delta_{34} = [s_3\ne s_4]$ and
$\delta_{\rm bk} = [(s_1,s_2)\ne(s_3,s_4)]$, an element $v = (\mu\nu|\lambda\sigma)$ adds
$$J_{\mu\nu} \mathrel{+}= v\,D_{\lambda\sigma},\qquad K_{\mu\lambda} \mathrel{+}= v\,D_{\nu\sigma}$$
and, when the flag of the image is set, the same with $\mu\leftrightarrow\nu$
($\delta_{12}$), $\lambda\leftrightarrow\sigma$ ($\delta_{34}$), both, and the
bra–ket swap with its three sub-images ($\delta_{\rm bk}$): up to 16 adds per
element (`scatter_bra_pair`, `scatter_bra_pair_probed`, and the blocked
`digest_j`/`digest_k` for blocks of 16 or more integrals in
`src/quartet_scatter.rs`). Because each distinct shell-index image is added
once, there is no element-level degeneracy factor: the sum over unique
quartets reproduces $J[D]_{\mu\nu} = \sum_{\lambda\sigma}(\mu\nu|\lambda\sigma)D_{\lambda\sigma}$
and $K[D]_{\mu\nu} = \sum_{\lambda\sigma}(\mu\lambda|\nu\sigma)D_{\lambda\sigma}$
exactly for a symmetric $D$.

Screening. A quartet is skipped when
$$Q_{s_1s_2}\,Q_{s_3s_4}\,d_{\max} < \tau_{\rm int},\qquad Q_{PQ} = \max_{a\in P,\,b\in Q}\sqrt{|(ab|ab)|},$$
with $\tau_{\rm int}$ = `integral_thresh` (default $10^{-12}$) and $d_{\max}$ the
largest shell-block maximum $\max|D_{\mu\nu}|$ over the six pairings of the
four shells (`DensityScreen::SixPair`; the open shell uses $|D^\alpha| + |D^\beta|$).
By Cauchy–Schwarz $|(\mu\nu|\lambda\sigma)| \le Q_{s_1s_2}Q_{s_3s_4}$, so each
dropped contribution to any element of $J$ or $K$ is below $\tau_{\rm int}$.
`[scf] screening = "csb"` replaces the bound by
$\min(Q_{12}Q_{34}, M_{13}M_{24}, M_{14}M_{23})$; `"csam"` is not a rigorous
bound.

Fock matrix. $F = h + J - \tfrac12 c_x K$ ($c_x = 1$ for Hartree–Fock),
$F^\sigma = h + J[D^\alpha + D^\beta] - c_x K[D^\sigma]$ for UHF, and
$E = \tfrac12\operatorname{tr}[D(h+F)] + E_{nn}$.

Incremental builds. $J$ and $K$ are linear in $D$, so with
$\Delta D_n = D_n - D_{n-1}$,
$$J[D_n] = J[D_{n-1}] + J[\Delta D_n], \qquad K[D_n] = K[D_{n-1}] + K[\Delta D_n] .$$
The closed-shell loop of `solve_rhf` keeps $J$ and $K$ (not $F$) in buffers
and adds $J[\Delta D]$, $K[\Delta D]$ built by `DirectJK::build_incremental`.
The screen is the same absolute test with $d_{\max}$ taken from $\Delta D$, so
as the SCF converges more quartets drop. The buffers are rebuilt from the full
density on iterations $1, 9, 17, \dots$ (every 8), which bounds the number of
screened increments that accumulate. No other event (a DIIS reset, MOM, a
level-shift change) forces a rebuild. The scheme is on only when it can pay
and cannot reach the convergence target: the direct route (no density
fitting, no pluggable K, no in-core store), and
`energy_conv` $\ge 10^4\,\tau_{\rm int}$ and `density_conv` $\ge 10^3\,\tau_{\rm int}$
(`closed_shell_incremental_enabled` in `src/direct_jk.rs`; the environment
variable `FERRIC_SCF_INCREMENTAL` forces it on or off), and the builder has
no in-core store (`DirectJK::incremental_pays`). The open-shell loops build
increments only under `FERRIC_SCF_UHF_INCREMENTAL=1`.

In-core store (`[scf] eri`, `src/incore_jk.rs`). Below about 200 functions the
unscreened integrals of one geometry are stored as the packed lower triangle
of the pair matrix and $J$, $K$ come from matrix products; `"auto"` takes the
store when it fits a quarter of the memory budget and both $J$ and $K$ are
four-centre. The incremental build is off with the store.

Proof: `scripts/scf_jk_builders_sympy.py` claim N9 (accumulation with the
8-iteration rebuild reproduces $J[D]$, $K[D]$).

## Density-fitted Coulomb and exchange (RI-J, RI-K, Split-RI-J)

Notation: raw three-index integrals $(P|\mu\nu)$, metric $V_{PQ} = (P|Q)$,
dressed factor $B^P_{\mu\nu} = \sum_Q M_{PQ}(Q|\mu\nu)$ with $M^{\mathsf T}M = V^{-1}$.

RI-J (`DfJ`, `src/df_j.rs`). The fitted density $\tilde\rho = \sum_P c_P\chi_P$
minimizes the Coulomb self-energy of the residual $\rho - \tilde\rho$, which
gives
$$\gamma_P = \sum_{\mu\nu}(P|\mu\nu)D_{\mu\nu},\qquad V c = \gamma,\qquad J_{\mu\nu} = \sum_P(\mu\nu|P)\,c_P .$$
The code stores $(P|\mu\nu)$ undressed, as the packed lower triangle; with
$w_{\mu\mu} = D_{\mu\mu}$ and $w_{\mu\nu} = D_{\mu\nu} + D_{\nu\mu}$ ($\nu<\mu$),
$\gamma_P = \sum_{\mu\ge\nu}(P|\mu\nu)\,w_{\mu\nu}$ (`packed_density_weights`),
which uses only the symmetric part of $D$. $V c = \gamma$ is solved with the
Cholesky factor $V = LL^{\mathsf T}$ computed once (no explicit inverse, no
eigenvalue cut; a metric that is not positive definite is an error). The fit
error of the Coulomb energy is the self-interaction of the residual,
$$\tfrac12(\rho|\rho) - \tfrac12\gamma^{\mathsf T}V^{-1}\gamma = \tfrac12(\rho - \tilde\rho\,|\,\rho - \tilde\rho) \ \ge 0,$$
second order in the residual; the robust form $2\gamma^{\mathsf T}c' - c'^{\mathsf T}Vc'$
has the same value at $c' = c$ and is stationary there, which is why the
plain and robust (Dunlap) energies coincide for RI-J with one density.

f32 storage (`ri_j_precision = "f32"`, the single-point default). Each
$(P|\mu\nu)$ is rounded to f32 once; $w$, $\gamma$, $c$, the solve and $J$ are
f64. The SCF converges on the operator of the rounded tensor, so the energy
carries an error first order in the rounding. Geometry and frequency tasks
default to f64.

RI-K (`DfK`, `src/df_k.rs`). With the same expansion of each exchange
product,
$$K_{\mu\nu} = \sum_P\sum_{\lambda\sigma}B^P_{\mu\lambda}D_{\lambda\sigma}B^P_{\sigma\nu} = \sum_P\big(B^P D B^P\big)_{\mu\nu}.$$
The dressing for K is the symmetric $M = U_s\lambda_s^{-1/2}U_s^{\mathsf T}$ over
the eigenpairs of $V$ with $\lambda$ above a cut of about $10^{-10}$
(`v_inv_sqrt_lindep`, `metric_lindep_cut`): $M^{\mathsf T}M$ is the
pseudo-inverse of $V$ on the kept span. RI-J uses the full Cholesky inverse,
so the two fits differ only when modes are cut. Three contractions give the
same $K$:

- density route, $\sum_P B^P D B^P$, for any $D$ (`density_chunk_k`);
- occupied route, $K_{\rm occ}(X) = \sum_P (B^PX)(B^PX)^{\mathsf T}$, no
  occupation factor (`occ_chunk_k`, `occ_half_panel`). RHF multiplies by 2,
  since $D = 2C_oC_o^{\mathsf T}$; UHF passes $C^\sigma_o$ with factor 1;
- signed factor route for a symmetric $D$ without orbitals
  (`build_from_density_factor`): $D = W\Lambda W^{\mathsf T}$,
  $X_\pm = W_\pm|\Lambda_\pm|^{1/2}$ over eigenvalues above $+\epsilon$ and below
  $-\epsilon$ ($\epsilon = 10^{-11}\max|\Lambda|$), and
  $K = K_{\rm occ}(X_+) - K_{\rm occ}(X_-)$. It falls back to the density
  route when $D$ is not symmetric to $10^{-12}$ or the kept columns exceed $3N/4$.

The incremental and response densities, which are indefinite, use the density
route or the factored response form $K = G + G^{\mathsf T}$,
$G = \sum_P (B^P X)(B^P C_o)^{\mathsf T}$ for $D = XC_o^{\mathsf T} + C_oX^{\mathsf T}$.
A range-separated hybrid contracts two fitters,
$K = c_{\rm SR}K[\mathrm{erfc}(\omega)] + c_{\rm LR}K[\mathrm{erf}(\omega)]$,
each with its own metric and dressing on one auxiliary basis.

Split-RI-J (`SplitRiJ`, `src/split_rij.rs`). The same three equations,
$\gamma = (P|\mu\nu)D$, $Vc = \gamma$ by the same Cholesky factor,
$J = (\mu\nu|P)c$, with $(P|\mu\nu)$ never stored: each build recomputes the
integrals in a McMurchie–Davidson kernel that contracts $D$ into Hermite
densities of each primitive pair before the auxiliary side, and $c$ into
auxiliary Hermite vectors before the orbital side, so the
$n_{\rm cart}^3$ integral block is never formed. Two approximations are
added on top of RI-J: the large-argument asymptote of the Boys function where
its relative error is at most $10^{-14}$ per term (`FAR_EPS`), and a
Schwarz-type screen on (orbital pair, auxiliary shell) blocks at
`integral_thresh`. It applies to the Coulomb operator with orbital $l \le 4$
and auxiliary $l \le 6$.

Proof: `scripts/scf_jk_builders_sympy.py` (claims S1–S3, N1–N8): the fit is a
Coulomb-metric projection, the energy error is the residual self-energy,
RI-J and RI-K are exact for a complete auxiliary set, $M = L^{-1}$ and
$M = V^{-1/2}$ agree, the occupied and signed-factor routes equal the density
route, the cut metric fits the kept span, and the packed weights are exact.

## LinK and COSX exchange

LinK (`LinkK`, `src/link_k.rs`). The same quartets and scatter as the direct
builder, for $K$ only, with two pair lists in front of the quartet test. With
$Q_{ij}$ the Schwarz factor and $q_x = \max_y Q_{xy}$:

- significant pairs (geometry only): $j \in \mathrm{sp}(i)$ iff $Q_{ij}\,Q_{\max} > \tau$;
- density pairs (every iteration): $\sigma \in \mathrm{dp}(j)$ iff
  $D^{\rm blk}_{j\sigma}\,q_j\,q_\sigma > \tau$, $D^{\rm blk}$ the shell-block
  maximum of $|D|$.

The bra runs over $(i, j)$ with $j\in\mathrm{sp}(i)$, the third shell over
$\mathrm{dp}(i)\cup\mathrm{dp}(j)$, the fourth over $\mathrm{sp}$ of the third.
A quartet, canonicalized and deduplicated, is skipped when
$$Q_{s_1s_2}Q_{s_3s_4}\,\max\big(D^{\rm blk}_{13}, D^{\rm blk}_{14}, D^{\rm blk}_{23}, D^{\rm blk}_{24}\big) < \tau,$$
the four exchange pairings only (`DensityScreen::FourPairK`). The lists are
necessary conditions of this test when the bound factorizes as
$Q_{12}Q_{34}$ (Schwarz), so LinK is direct $K$ to the threshold. The skip
rule is exact; the asymptotic count of surviving quartets is a property of
the density, not of the code.

COSX (`CosxK`, `src/cosx_k.rs`). One electron of each exchange integral is
integrated on a Becke–Lebedev grid, the other analytically:
$$K_{\mu\nu} = \sum_{\lambda\sigma}D_{\lambda\sigma}\int\!\!\int\frac{\chi_\mu(\mathbf r_1)\chi_\sigma(\mathbf r_1)\,\chi_\nu(\mathbf r_2)\chi_\lambda(\mathbf r_2)}{r_{12}}
\approx \sum_g X_{\mu g}\,G_{\nu g},$$
$$X_{\mu g} = \sqrt{w_g}\,\chi_\mu(\mathbf r_g),\quad F_{\lambda g} = \sum_\sigma D_{\lambda\sigma}X_{\sigma g},\quad
A_{\nu\lambda}(g) = \int\frac{\chi_\nu\chi_\lambda}{|\mathbf r - \mathbf r_g|}\,d\mathbf r,\quad
G_{\nu g} = \sum_\lambda A_{\nu\lambda}(g)F_{\lambda g}.$$
$A(g)$ comes from the McMurchie–Davidson kernel (`cosx_backend = "md3c1e"`)
and is never assembled as a matrix: each shell pair's block is added into
$G$ for both orderings of the pair (`accumulate_pair`). The plain result is
$K = \tfrac12(\tilde K + \tilde K^{\mathsf T})$, $\tilde K = XG^{\mathsf T}$. The
overlap fit (`cosx_overlap_fit`, default on) replaces the left factor by its
projection that makes the numerical overlap exact,
$$K = \tfrac12\big(Q\tilde K + (Q\tilde K)^{\mathsf T}\big),\qquad Q = S\,S_{\rm num}^{-1},\quad S_{\rm num} = XX^{\mathsf T},$$
computed with a Cholesky factor of $S_{\rm num}$ (built once per geometry).
For a potential that is constant over the molecule, $A_{\nu\lambda}(g) = cS_{\nu\lambda}$,
the plain result is $cS_{\rm num}DS$ and the fitted one is $cSDS$, exact on
any grid: the fit removes the error of the grid's overlap. As the grid
becomes exact $S_{\rm num}\to S$, $Q \to 1$ and both forms tend to the exact $K$.

Screening. A shell pair is dropped for a sub-batch of points when
$\mathrm{est}\cdot\max(f_{s_1}, f_{s_2}) < \tau_{\rm COSX}$, with est an upper
bound on $\max_g|A^{(g)}|$ over the region (the minimum of a sphere and a
box bound, both never below the true value) and $f_s = \max|F_{\mu g}|$ over
the shell and the points. Each dropped pair perturbs an element of $G$ by at
most $\tau_{\rm COSX}$ per component it contributes, so the bound on one
element is the sum over the dropped partners. $\tau_{\rm COSX} = 0$ is
bitwise the unscreened build. The default sparse half transform keeps only
the shells whose $|X|$ and $|D|$ blocks exceed $10^{-10}$; zero thresholds
are bitwise the dense build.

Final pass. With `cosx_final_pass` the converged energy is corrected once,
non-self-consistently, on a larger grid:
$E = E_{\rm SCF} + \sum_s c_s\big(\operatorname{tr}[D_sK_{\rm final}(D_s)] - \operatorname{tr}[D_sK_{\rm SCF}(D_s)]\big)$,
$c = -c_x/4$ closed shell and $-c_x/2$ per spin.

Proof: `scripts/scf_jk_builders_sympy.py` claims C1 (one $s$ Gaussian: the
grid sum tends to $(ss|ss) = 2\sqrt{\alpha/\pi}$) and C2 (the fit is exact for
a constant potential, the plain form is not).

## DIIS: the error vector and the extrapolation

Stationarity. In the orthonormal MO basis, $f = C^{\mathsf T}FC$ and
$n = \mathrm{diag}(2_o, 0_v)$ (closed shell). Since $C^{\mathsf T}SC = 1$ and
$D = CnC^{\mathsf T}$,
$$C^{\mathsf T}(FDS - SDF)C = fn - nf, \qquad (fn - nf)_{ai} = -2f_{ai},\ \ (fn-nf)_{ia} = 2f_{ia}.$$
The error $e = FDS - SDF$ is antisymmetric and its occupied–virtual block is
$2f_{ia}$, half the orbital gradient
$\partial E/\partial\kappa_{ai} = 4f_{ai}$ of $E(Ce^{\kappa})$. The occupied–occupied
and virtual–virtual blocks of $fn - nf$ vanish. So $e = 0$ exactly when the
energy is stationary under every occupied–virtual rotation, which is the
Roothaan condition $FC = SC\varepsilon$ up to rotations within each block.
For UHF the same holds per spin with $e^\sigma = F^\sigma D^\sigma S - SD^\sigma F^\sigma$
and $\partial E/\partial\kappa^\sigma_{ai} = 2f^\sigma_{ai}$. ROHF uses the gradient
of the section on energy functionals mapped to the AO basis,
$e = SC(g - g^{\mathsf T})C^{\mathsf T}S$, and extrapolates the Roothaan effective
Fock matrix.

Extrapolation. Given the last $m$ pairs $(F_i, e_i)$ ($m \le$ `diis_size`,
default 8), Pulay DIIS minimizes $|\sum_i c_ie_i|^2$ subject to $\sum_ic_i = 1$.
With $B_{ij} = \sum_{\mu\nu}(e_i)_{\mu\nu}(e_j)_{\mu\nu}$ (the AO matrices,
not orthogonalized), the stationarity conditions of the Lagrangian are the
bordered system the code solves (`solve_diis_coeffs`),
$$\begin{pmatrix}\tilde B & \mathbf 1\\ \mathbf 1^{\mathsf T} & 0\end{pmatrix}\begin{pmatrix}c\\ \lambda\end{pmatrix} = \begin{pmatrix}0\\ 1\end{pmatrix},\qquad \tilde B = B/\max_{ij}|B_{ij}|,$$
whose solution is $c = \tilde B^{-1}\mathbf 1/(\mathbf 1^{\mathsf T}\tilde B^{-1}\mathbf 1)$
with $\lambda = -c^{\mathsf T}\tilde Bc$. The scaling leaves $c$ unchanged. The
next Fock matrix is $\bar F = \sum_ic_iF_i$. UHF solves one system with
$B_{ij} = \langle e^\alpha_i, e^\alpha_j\rangle + \langle e^\beta_i, e^\beta_j\rangle$
and applies the same $c$ to both spins. Gaussian elimination with partial
pivoting rejects a pivot below $10^{-13}$; the solve is then retried without
the oldest vector, down to $m = 2$, for that step only. With fewer than two
vectors the plain $F$ is used, so `diis_size = 1` is plain Roothaan iteration.

EDIIS and ADIIS (`EnergyDiis`) minimize a quadratic model over the simplex,
$c_i = t_i^2/\sum t^2$: EDIIS with $g_i = E_i$,
$H_{ij} = -\langle D_i - D_j, F_i - F_j\rangle$; ADIIS with
$g_i = 2\langle D_i - D_n, F_n\rangle$, $H_{ij} = 2\langle D_i - D_n, F_j - F_n\rangle$
($n$ the newest entry). The driver uses them while $\max|e| \ge$
`diis_switch_thresh` (default 0.1) and Pulay below it. Only the RHF solver
reads the flavor (`[scf] diis`, `RhfConfig::diis_flavor`).

Convergence test. The SCF exits when, between successive densities,
$\Delta D_{\rm rms} <$ `density_conv`, $\Delta D_{\max} < 10\,$`density_conv`
and $|\Delta E| <$ `energy_conv`, from iteration 2 on; the exit diagonalizes
the unextrapolated, unshifted $F$. An exit with
$\max|X^{\mathsf T}eX| > \max(\sqrt{\texttt{density\_conv}}, 10^{-3})$ is
refused and DIIS restarts (`veto_nonstationary_rhf_exit`).

Proof: `scripts/scf_diis_sympy.py` (claims S1–S3, N1–N6).

## Level shift, MOM and Fermi smearing

Level shift. After extrapolation the solvers add
$$\bar F \leftarrow \bar F + \mu_{\rm eff}\,SC_vC_v^{\mathsf T}S,$$
$C_v$ the virtual columns of the previous orbitals. In their MO basis,
$C^{\mathsf T}SC_vC_v^{\mathsf T}SC = \mathrm{diag}(0_o, 1_v)$, so the shift
adds $\mu_{\rm eff}$ to the virtual–virtual block and nothing else: the
occupied–virtual coupling is unchanged, the step is damped (the virtual
orbitals mix less into the occupied space), and at self-consistency the
eigenvectors and density are those of the unshifted $F$, with the virtual
energies raised by $\mu_{\rm eff}$. The ramp differs by solver: RHF uses
$\mu_{\rm eff} = \mu\min(\max|e|/0.1, 1)$, UHF and ROHF
$\mu\,\max|e|/(\max|e| + 10^{-3})$. $\mu$ is `level_shift`, or 0.5 Ha for a
meta-GGA when `level_shift` is 0 (`effective_level_shift`). The energy and
the convergence test use the unshifted $F$.

MOM (`mom_reorder`, `src/mom.rs`). With the previous occupied orbitals
$C^{\rm old}_o$ and the new eigenvectors $C$, the overlap
$O = C_o^{{\rm old}\,\mathsf T}SC$ scores each new orbital by
$p_j = \sum_iO_{ij}^2$, its squared projection on the old occupied space. The
$n_o$ highest scores are occupied (ties to the lower index); for ROHF the
open shell is then chosen the same way from the rest. The reference is the
previous iteration's orbitals, or for UHF optionally the initial guess
(`mom_reference = "initial"`). MOM starts after iteration `mom_after_iter`.

Fermi smearing (`[scf] smearing_sigma`, RHF only; `src/smearing.rs`).
$$f_p = \frac{1}{1 + e^{(\varepsilon_p - \mu_F)/\sigma}}, \qquad \sum_pf_p = N_{\rm el}/2, \qquad D = C\,\mathrm{diag}(2f)\,C^{\mathsf T},$$
$\sigma$ in Hartree, $\mu_F$ by bisection on the bracket
$[\varepsilon_{\min} - (20\sigma+1), \varepsilon_{\max} + (20\sigma+1)]$ to $10^{-14}$,
the exponent clamped at $\pm40$. The entropy
$S = -2\sum_p[f_p\ln f_p + (1-f_p)\ln(1-f_p)]$ is computed, but the returned
energy is $E[D]$, not the Mermin free energy $E - \sigma S$.

Proof: `scripts/scf_diis_sympy.py` claims L1 (the shift touches the
virtual–virtual block only and leaves the converged density) and F1 (the
Fermi level holds $N/2$ pairs; $\sigma\to0$ is aufbau).

## Orbital Hessians: Newton, augmented Hessian, trust region

Parametrization. $C \leftarrow CU(\kappa)$ with $\kappa$ antisymmetric, stored as
$\kappa_{ai}$, and the Cayley $U = (1 - \kappa/2)^{-1}(1 + \kappa/2)$
(`cayley_unitary`). $U$ equals $e^\kappa$ through second order, so first and
second derivatives of $E(CU(t\kappa))$ at $t = 0$ are those of the
exponential map.

RHF. $D = 2C_oC_o^{\mathsf T}$ moves by $\delta D = 2C(\kappa + \kappa^{\mathsf T})C^{\mathsf T}$
at first order ($\kappa + \kappa^{\mathsf T}$ is the symmetric matrix with
$\kappa_{ai}$ in both corners). The derivatives of $E(CU(t\kappa))$ are
$$\frac{dE}{dt} = 4\sum_{ai}f_{ai}\kappa_{ai},\qquad
\frac{d^2E}{dt^2} = 4\,\kappa\cdot(\mathbf H\kappa),$$
$$(\mathbf H\kappa)_{ai} = \sum_b f_{ab}\kappa_{bi} - \sum_j\kappa_{aj}f_{ji}
+ \sum_{bj}\big[4(ai|bj) - c_x\big((ab|ij) + (aj|bi)\big)\big]\kappa_{bj}
\quad(+\,\text{the } f_{xc} \text{ term for KS}).$$
The code's gradient is $g = f_{ai}$ and its product is $\mathbf H\kappa$, a
quarter of the true derivatives (`hessian_matvec_jk` in `src/rhf_newton.rs`):
it forms $\delta D$, $\delta F = J[\delta D] - \tfrac12c_xK[\delta D]$, takes the
$vo$ block of $C^{\mathsf T}\delta FC$, and adds the diagonal gap
$(f_{aa} - f_{ii})\kappa_{ai}$ for the Fock terms. That is exact only when
$f_{oo}$ and $f_{vv}$ are diagonal (canonical orbitals);
`offdiag_gap_correction` (`src/trah.rs`) adds $f_{vv}\kappa - \kappa f_{oo} - (f_{aa}-f_{ii})\kappa$
for the general case. At canonical orbitals $\mathbf H$ is the singlet
$\mathbf A + \mathbf B$ of linear response.

UHF. $\delta D^\sigma = C^\sigma(\kappa^\sigma + \kappa^{\sigma\mathsf T})C^{\sigma\mathsf T}$,
$\delta F^\sigma = J[\delta D^\alpha + \delta D^\beta] - c_xK[\delta D^\sigma]$, and
$$\frac{dE}{dt} = 2\sum_\sigma\sum_{ai}f^\sigma_{ai}\kappa^\sigma_{ai},\qquad\frac{d^2E}{dt^2} = 2\sum_\sigma\kappa^\sigma\cdot(\mathbf H\kappa)^\sigma,$$
$$(\mathbf H\kappa)^\sigma_{ai} = (f^\sigma_{aa} - f^\sigma_{ii})\kappa^\sigma_{ai} + \sum_{\tau,bj}\big[2(a_\sigma i_\sigma|b_\tau j_\tau) - \delta_{\sigma\tau}c_x\big((ab|ij)+(aj|bi)\big)\big]\kappa^\tau_{bj}$$
(`hessian_matvec` in `src/uhf_newton.rs`, half the true derivatives, canonical orbitals).

ROHF. One orbital set with occupations $n_\alpha$, $n_\beta$, packed rotations
$vc$, $vo$, $oc$. With $X$ the unpacked antisymmetric generator,
$X_\sigma = XN_\sigma - N_\sigma X$ ($(X_\sigma)_{pq} = X_{pq}(n_{\sigma q} - n_{\sigma p})$),
$\delta D^\sigma = CX_\sigma C^{\mathsf T}$, $\delta f^\sigma = C^{\mathsf T}(J[\delta D^\alpha + \delta D^\beta] - c_xK[\delta D^\sigma])C$,
$A_\sigma = f^\sigma X - Xf^\sigma$ and $\Delta n_{pq} = n_{\sigma p} - n_{\sigma q}$,
$$G = \sum_\sigma\tfrac12\Delta n\circ A_\sigma + \tfrac12(X_\sigma f^\sigma - f^\sigma X_\sigma) + \Delta n\circ\delta f^\sigma,\qquad (\mathbf H x)_{pq} = G_{qp} - G_{pq}$$
is the exact Hessian product of $E(CU(x))$ at any orbitals, not only at a
stationary point (`rohf_exact_hessian_matvec` in `src/rohf_trah.rs`; the
gradient is $2g$ with the blocks of the energy-functional section,
`rohf_exact_gradient`). The molecular Newton and AH matvec
(`src/rohf_newton.rs`) is half of it.

Newton (`rhf_newton_step`, `uhf_newton_step`, `rohf_newton_step`).
Preconditioned conjugate gradient on $(\mathbf H + \mu)\kappa = -g$ from
$\kappa^0 = -g/(\Delta + \mu)$, $\Delta$ the diagonal gap, at most 20 iterations;
the step is scaled so $\max|\kappa|$ is at most 0.2 (0.1 for ROHF). ROHF
truncates at negative curvature (Steihaug); RHF and UHF do not.

Augmented Hessian. The lowest eigenpair $(\mu, (v_0, v))$ of
$$\begin{pmatrix}0 & \alpha g^{\mathsf T}\\ \alpha g & \mathbf H\end{pmatrix}$$
gives $\kappa = v/(\alpha v_0)$. The lower rows read
$\mathbf Hv + \alpha gv_0 = \mu v$, so
$$(\mathbf H - \mu)\kappa = -g,$$
a Newton step with the level shift $-\mu \ge 0$ chosen by the eigenproblem,
and the top row gives $g^{\mathsf T}\kappa = \mu/\alpha^2$. The predicted change
$g^{\mathsf T}\kappa + \tfrac12\kappa^{\mathsf T}\mathbf H\kappa$ then equals
$\tfrac12(g^{\mathsf T}\kappa + \mu|\kappa|^2)$. ROHF's `rohf_ah_step` solves it
at $\alpha = 1$ with block Davidson.

Trust region (`solve_trust_region`, `src/trah.rs`; RHF and UHF). $\alpha$ is
adjusted (grown by 8 then bisected in $\log\alpha$, $1 \le \alpha \le 1000$)
until $|\kappa|_2$ lies within 5% below the radius $\Delta_{\rm tr}$. The
prediction is multiplied by 4 (RHF) or 2 (UHF) to be a true energy change.
With $\rho = \Delta E/\Delta E_{\rm pred}$: reject and shrink by 0.7 at
$\rho \le 0$; accept and shrink by 0.7 at $\rho \le 0.25$; grow by 1.2 to at
most 2.0 at $\rho > 0.75$; the region collapses below $10^{-4}$ and DIIS
takes over. An eigenvector with a vanishing first component (the hard case)
is deflated and the step completed along it to the boundary.

Proof: `scripts/scf_orbital_hessian_sympy.py` claims C1, H1–H6, T1: every
product against fourth-order differences of the energy itself.

## SCF stability analysis

The stability operators are the Newton products at the converged orbitals
(`src/stability.rs`), so their eigenvalues carry the code's scale: a quarter
of the RHF energy curvature, half of the UHF one.

| Analysis | Operator | Function |
|---|---|---|
| RHF internal (real, singlet) | $\mathbf H$ of RHF: $(\varepsilon_a-\varepsilon_i)\delta + 4(ai\vert bj) - c_x[(ab\vert ij)+(aj\vert bi)]$ | `rhf_internal_stability` |
| RHF external (RHF → UHF, triplet) | $(\varepsilon_a-\varepsilon_i)\delta - c_x[(ab\vert ij)+(aj\vert bi)]$ | `rhf_external_stability` |
| UHF internal | $\mathbf H$ of UHF | `uhf_internal_stability` |

The triplet operator is the UHF operator at the RHF point in the direction
$\kappa^\alpha = -\kappa^\beta$: the Coulomb terms $2(a_\alpha i_\alpha|b_\beta j_\beta)$
and $2(a_\alpha i_\alpha|b_\alpha j_\alpha)$ cancel, the exchange term stays.
In the direction $\kappa^\alpha = \kappa^\beta$ the UHF operator is the RHF
singlet operator, and the two channels do not couple. For a unit eigenvector
$v$ of the triplet operator with eigenvalue $\lambda_T$, the UHF energy along
$(v, -v)$ has curvature $4\lambda_T$, so $\lambda_T < 0$ means a UHF solution
lower than the RHF one. Real-to-complex, UHF-to-GHF and ROHF stability are not
implemented.

The lowest eigenvalues come from a block Davidson (`davidson_lowest`; 8
seed vectors, blocks of 6, subspace at most 40, residual $10^{-6}$, diagonal
preconditioner $\varepsilon_a - \varepsilon_i$; `StabilityConfig`). The verdict is unstable when
$\lambda < -\max(10^{-6}, \text{residual})$, marginal within that floor. The
descent (`stability_descent` in `src/uhf.rs`, `rhf_stability_descent` in
`src/rhf.rs`) rotates the orbitals by $\epsilon v$ for
$\epsilon \in \{0.4, 0.8, 1.2\}$, reconverges each, and keeps the lowest energy
below the incumbent, for up to three rounds.

Proof: `scripts/scf_orbital_hessian_sympy.py` claims S1 (the channels of the
UHF operator are the RHF singlet and triplet operators) and S2 (the
curvature along a triplet mode is $4\lambda_T$).

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

**ECPs.** $V_{\rm ECP}$ is part of $h$ and does not enter $E_{xc}$, so it adds the same two terms to the RKS and UKS Hessians
as to the RHF and UHF ones: the skeleton $\sum_{\mu\nu} D_{\mu\nu}\,\partial^2 V^{\rm ECP}_{\mu\nu}/\partial x\partial y$ (`hessian_ecp.rs`,
a stencil of the first-derivative blocks) and $\partial V^{\rm ECP}/\partial x$ in every first-order Fock matrix $F^x$ (`D` the total
density in the skeleton; the same $\partial V^{\rm ECP}/\partial x$ for both spins). The XC terms above are unchanged.

| Equation | Rust | Numerical test |
|---|---|---|
| $g$, $\rho_x$, $\mathbf g_x$ | `gradient_densities`, `rho_derivative`, `grad_rho_derivative` | `ks_fixed_grid_hessian.rs::gradient_matches_central_difference_of_the_fixed_grid_energy` |
| $F\rho_{xy}+\mathbf G\cdot\mathbf g_{xy}$, $\delta F$, $\delta G$ | `add_same_centre`, `add_cross_centre`, `weights_response`, `kernel_part` | `xc_pieces_match_frozen_density_differences`, `gga_xc_pieces_match_frozen_density_differences` |
| $\partial V/\partial x$ | `first_order_fock`, `add_ao_derivative_fock` | same two tests (FD of $V_{xc}$ at displaced nuclei) |
| $V^{(1)}[D_1]$ | `XcResponse::apply` | `xc_kernel_response_matches_difference_of_vxc_in_the_density`, `gga_kernel_response_…` |
| polarized $F^\sigma,\mathbf G^\sigma,\delta F^\sigma,\delta\mathbf G^\sigma$, skeleton, $\partial V^\sigma/\partial x$, $V^{(1)\sigma}$ | `ks_fixed_grid/polarized.rs` (`PolarizedKs`, `pol_response`, `PolarizedXcResponse`) | `uks_xc_pieces.rs` (closed-shell limit, frozen-density and frozen-grid FD, kernel FD with $D_1^\alpha\neq D_1^\beta$, mutants); full Hessian `uks_moving_grid_hessian.rs` |
| $T G_h T^{\mathsf T}$, moving $\partial V/\partial x$ | `skeleton_hessian_moving`, `translation_fill`, `first_order_fock_moving`, `weight_terms` | `ks_moving_grid_hessian.rs::moving_xc_pieces_match_frozen_density_differences`, `moving_grid_mutants_each_fail_a_frozen_density_anchor` |
| ECP skeleton and $\partial V^{\rm ECP}/\partial x$ in the KS Hessian | `extra_skeleton`, `add_extra_first_order` (`hessian.rs`), called from `hessian/ks.rs` and `hessian/uks.rs` | `ks_ecp_hessian.rs` (RKS HI, UKS H-I-H vs FD of the production gradient, both terms as negative controls); `validation_ks_fixed_grid_hessian.rs::fixed_grid_pbe_hessian_with_an_ecp_matches_pyscf_on_the_same_grid` |

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

## Peak memory of the fitted MP2 gradient

An MP2-family gradient on an RI-JK reference keeps the reference's fitted
J and K for the Z-vector and differentiates the fitted two-electron energy
of the relaxed density (`response_jk.rs`, `fitted_bilinear_gradient`).
`ferric_mp2::gradient_memory` declares its peak from the shapes before any
integral and refuses a budget below it. With $n$ basis functions, $N_J$
JK-fitting and $N_R$ correlation auxiliary functions, $o$ correlated
occupied and $v$ virtual orbitals, $w$ rayon workers, $s_J$ the widest
JK-fitting shell and 8-byte elements, the declared peak is

$$ M = M_0 + R + \max_k T_k, $$

where $M_0$ is the process's resident set at entry and the terms resident
from the J/K route set-up to the end of the gradient are

$$ R = (ov)^2 + N_R\,ov + N_R^2 + t(N_J) + N_J\,\tfrac{n(n+1)}{2} + 2N_J^2 , $$

the amplitudes, $B^P_{ia}$, the RI metric, the RI-J raw tensor $(P|\mu\nu)$
in the tier it takes under the budget, the packed RI-K dressed band and the
JK metrics. $t(N) = Nn^2$ in core, $Nn(n+1)/2 + b$ packed, $b$ spilled, with
$b$ one $(N_b, n, n)$ block of at most a quarter of the budget and 512 MiB
(plus one shell). The stage transients, of which only the largest is live
at a time, are

$$
\begin{aligned}
T_{\mathrm{MP2}} &= t(N_R) + (ov)^2 + w\,o v^2, \\
T_{\mathrm{RI\text{-}K}} &= t(N_J), \\
T_{\partial B} &= 2N_J \textstyle\sum_s m_s^2 + 5N_J^2 + 2b + w\,s_J n^2, \\
T_{\mathrm{RI}} &= 3N_R\,ov + N_R^2 + w\,v\max(ov, n^2):
\end{aligned}
$$

the MP2 intermediates (raw RI tensor, $(ia|jb)$, the per-worker $g_i$
panel), the second raw JK tensor while RI-K dresses it, the fitted
derivative's exchange intermediates $L^s$ and $\Gamma^s$ for every channel,
its metric matrices, 3c block and per-worker weight slab, and the RI
response's $x$, $y$, $c$, $\Gamma_{2c}$ and per-worker panels. The channel
ranks are not known before the Z-vector; the bound uses the split
$B(d_1, d_2) = Q(u) - Q(v)$. A positive-semidefinite density is one channel of
rank $m \le n$; an indefinite one is the three channels $P$, $N$, $P + N$
with $m_P + m_N \le n$, so $m_P^2 + m_N^2 + m_{P+N}^2 \le 2n^2$. With both
$u$ and $v$ indefinite,

$$ \textstyle\sum_s m_s^2 \le 4n^2 . $$

The four-centre route holds none of the JK terms, which is why the fitted
gradient's peak is a multiple of the four-centre one at the same size.

## Factored RI-K in the open-shell CPHF

The RI-K exchange of the CPHF kernel (`TwoElectronBuilder::uhf_fitted_exchanges`
in `src/hessian.rs`, `DfK::build_factored_many` in `src/df_k/response.rs`) is
$K[D] = \sum_P B^P D B^P$ with every $B^P$ symmetric. For any density of the
form $D = XC_o^{\mathsf T} + C_oX^{\mathsf T}$,
$$K[D] = G + G^{\mathsf T},\qquad G = \sum_P (B^PX)(B^PC_o)^{\mathsf T},$$
since $B^P X C_o^{\mathsf T} B^P = (B^P X)(B^P C_o)^{\mathsf T}$ and the
$C_oX^{\mathsf T}$ half is its transpose.
$L^P = B^PC_o$ is built once per $C_o$ and reused for every $X$:
$4\,n_{\rm aux}n^2 o$ flops per density against $4\,n_{\rm aux}n^3$ for the
density route. $D$ need not be definite or idempotent; only its form matters.

**UHF and UKS.** Each spin's CPHF density has this form with that spin's
occupied block, $C_o = C_{\sigma,o}$: the rotation response
$D_1(U_\sigma) = C_{\sigma,v}U_\sigma C_{\sigma,o}^{\mathsf T} + {\rm h.c.}$ has
$X_\sigma = C_{\sigma,v}U_\sigma$, and the occupied block
$D^y_{\sigma,oo} = -C_{\sigma,o}S^y_{\sigma,oo}C_{\sigma,o}^{\mathsf T}$ has
$X_\sigma = -\tfrac12 C_{\sigma,o}S^y_{\sigma,oo}$ ($S_{oo}$ symmetric). The
kernel $v_\sigma = J(D_{1\alpha} + D_{1\beta}) - a\,K(D_{1\sigma})$ never couples
the spins in $K$, so the factored pass runs once per spin, with $L^P_\sigma =
B^PC_{\sigma,o}$, over every coordinate of a lockstep block.

**ROHF.** One orbital set $C$ (closed | open | virtual) and occupation vectors
$n_\sigma$ whose first $n_\sigma$ entries are one. Every density change of the
ROHF response is $C(Un_\sigma + n_\sigma U^{\mathsf T})C^{\mathsf T}$ for a
generator $U$: the rotation part $(n_{\sigma q} - n_{\sigma p})\kappa_{pq}$ is
$U = \kappa$ ($\kappa$ antisymmetric), the overlap part
$-\tfrac12 S^y_{pq}(n_{\sigma p} + n_{\sigma q})$ is $U = -\tfrac12 S^y_{\rm MO}$
(symmetric), and the total $D^y_\sigma$ is $U = \kappa - \tfrac12 S^y_{\rm MO}$.
With $(Un_\sigma)_{pq} = U_{pq}$ for $q < n_\sigma$ and zero otherwise,
$$C\,U n_\sigma\,C^{\mathsf T} = (CU)_{:,\,<n_\sigma}\,C_{:,\,<n_\sigma}^{\mathsf T},$$
so $C_{\sigma,o} = C_{:,\,<n_\sigma}$ and $X_\sigma = (CU)_{:,\,<n_\sigma}$
(`Frame::spin_factors`). The ROHF response solves one coordinate at a time, so
each factored pass carries one density per spin.

**Closed-shell limit.** With $C_\alpha = C_\beta$ and $U_\alpha = U_\beta$,
$X_{\rm RHF} = 2X_\sigma$, so $K_{\rm RHF} = 2K(D_{1\sigma})$ and
$v_\sigma = J(2D_{1\sigma}) - K(D_{1\sigma}) = J(D_{\rm RHF}) - \tfrac12K(D_{\rm RHF})$,
the RHF $G$ of the closed-shell factored route.

Proof: `scripts/hessian_factored_k_sympy.py` (F1, F3 symbolic; F2, F4, F5
against the explicit four-index exchange).

## COSMO reaction field

`cosmo.rs` implements the conductor-like screening model (Klamt and
Schüürmann, J. Chem. Soc. Perkin Trans. 2, 799 (1993)) as a density-dependent
term folded into the Fock matrix every iteration (`driver::solvent_terms`).
The equations below are the ones `cosmo_reaction_field` evaluates.

**Segment potential.** For a cavity of $N$ segments at $\mathbf{s}_k$ and the
total AO density $\mathbf{P} = \sum_\sigma \mathbf{D}_\sigma$,

$$v_k = \sum_{A} \frac{Z^{\rm eff}_A}{|\mathbf{s}_k - \mathbf{R}_A|}
 + \sum_{\mu\nu} P_{\mu\nu} M_{k,\mu\nu},
\qquad
M_{k,\mu\nu} = \Big\langle \mu \Big| \frac{-1}{|\mathbf{r} - \mathbf{s}_k|} \Big| \nu \Big\rangle ,$$

where $M_k$ is the nuclear-attraction engine block with a $+1$ probe at
$\mathbf{s}_k$, $Z^{\rm eff}_A$ is `Atom::effective_z` (valence charge on an
ECP atom) and ghost atoms are skipped. The probe is a point: the potential is
not smeared over the segment's Gaussian.

**Charges.** With the symmetric segment matrix $\mathbf{S}$ and the
dielectric factor

$$f(\varepsilon) = \frac{\varepsilon - 1}{\varepsilon + \tfrac12}
 = \frac{2(\varepsilon-1)}{2\varepsilon+1},$$

the code solves $\mathbf{S}\mathbf{q} = -f\,\mathbf{v}$ by one LU solve
(`CosmoConfig::f_epsilon`). $f$ is Kirkwood's $k_1$: $f(1) = 0$ (vacuum) and
$f \to 1$ as $\varepsilon \to \infty$, where the equation becomes the
conductor condition $\mathbf{S}\mathbf{q} = -\mathbf{v}$ (the $f = 1$ limit of
IEF-PCM, `crates/ferric-pcm/theory.md` section 6). C-PCM's $(\varepsilon-1)/\varepsilon$
is not used.

**Energy and Fock term.** With $\mathbf{Q} = -f\mathbf{S}^{-1}$,

$$E_{\rm cosmo} = \tfrac12\,\mathbf{q}\cdot\mathbf{v}
 = -\tfrac{f}{2}\,\mathbf{v}^{\mathsf T}\mathbf{S}^{-1}\mathbf{v},
\qquad
V^{\rm cosmo}_{\mu\nu} = \frac{\partial E_{\rm cosmo}}{\partial P_{\mu\nu}}
 = -f\,(\mathbf{S}^{-1}\mathbf{v})\cdot\frac{\partial \mathbf{v}}{\partial P_{\mu\nu}}
 = \sum_k q_k M_{k,\mu\nu}.$$

The middle step uses $\mathbf{S} = \mathbf{S}^{\mathsf T}$; for a
non-symmetric matrix the derivative would be
$-\tfrac f2[(\mathbf{S}^{-1} + \mathbf{S}^{-\mathsf T})\mathbf{v}]\cdot\partial\mathbf{v}/\partial P$
and $\sum_k q_kM_k$ would not be it. There is no factor $\tfrac12$ in
$V^{\rm cosmo}$: the $\tfrac12$ of the energy cancels against the two factors
of $\mathbf{v}$. $E_{\rm cosmo}$ depends on the spins only through
$\mathbf{P}$, so the same $V^{\rm cosmo}$ is added to every spin Fock matrix,
and $E_{\rm cosmo}$ is added to the SCF energy outside the
$\tfrac12\mathrm{Tr}[\mathbf{P}(\mathbf{h}+\mathbf{F})]$ trace (the trace is
taken before $V^{\rm cosmo}$ enters $\mathbf{F}$). $E_{\rm cosmo}$ includes the
nuclear part of $\mathbf{v}$, so it is the full electrostatic solvation term
of the solute. For a positive definite $\mathbf{S}$, $E_{\rm cosmo} \le 0$.

**Segment matrix.** `SMatrixKind::GaussianSmeared` (the default, PySCF's
`pcm.py` form) treats segment $k$ as the unit Gaussian
$(\xi_k^2/\pi)^{3/2}e^{-\xi_k^2|\mathbf{r}-\mathbf{s}_k|^2}$ with
$\xi_k = \Xi_{N_{\rm Leb}}/(R_A\sqrt{4\pi w_k})$ (`gaussian_xi_table`, $w_k$ the
ferric Lebedev weight, which sums to 1):

$$S_{kl} = \frac{\operatorname{erf}(\xi_{kl} r_{kl})}{r_{kl}},
\quad \xi_{kl} = \frac{\xi_k\xi_l}{\sqrt{\xi_k^2+\xi_l^2}},
\qquad
S_{kk} = \frac{\xi_k\sqrt{2/\pi}}{\mathrm{swf}_k}.$$

The off-diagonal element is the Coulomb energy of the two Gaussians
($1/\xi_{kl}^2 = 1/\xi_k^2 + 1/\xi_l^2$), and $\xi_k\sqrt{2/\pi}$ is a Gaussian's
self-energy, the $r\to0$ limit of $\operatorname{erf}(\xi r/\sqrt2)/r$. The
self term is divided by the segment's switching weight $\mathrm{swf}_k$, the
product of the smooth switching functions $h$ of every other atom's sphere
(`ferric_pcm::cavity::switch_h`), so a segment fading out of the surface
costs more to charge. `SMatrixKind::PointCharge` has $S_{kl} = 1/r_{kl}$,
the $\xi\to\infty$ limit of the Gaussian form, and the Klamt–Schüürmann
diagonal $S_{kk} = 3.8\sqrt{4\pi/a_k}$ with the switched area
$a_k = 4\pi R_A^2 w_k\,\mathrm{swf}_k$; the constant 3.8 is a model constant,
not derived here. Segments with $w_k\,\mathrm{swf}_k \le 10^{-16}$ are
dropped; the radii are Bondi radii times `radius_scale` (default 1.17).

**No gradient.** There is no COSMO nuclear gradient.
`gradient::refuse_cosmo_gradient` stops every gradient dispatcher, geometry
optimization, saddle search, IRC and finite-difference-of-gradient frequency
run on a COSMO SCF. The derivative of $E_{\rm cosmo}$ with respect to the
nuclei would need $\partial\mathbf{S}/\partial\mathbf{R}$ (segments move with
their atoms, and $\mathrm{swf}_k$ and $\xi_k$ change), the derivative of
$M_k$ with respect to both basis centres and segment positions, and the
nuclear part of $\mathbf{v}$; none of it is implemented.

Proof: `scripts/cosmo_sympy.py` (C1 the Fock term as $\partial E/\partial P$
in symbols; C2 the same against a finite difference of $E(\mathbf{P})$ on a
Gaussian $\mathbf{S}$, with $V/2$ and a non-symmetric $\mathbf{S}$ as negative
controls; C3 $f = k_1$ and its limits; C4 the Gaussian $S_{kl}$ and $S_{kk}$
against a Fourier-space Coulomb quadrature, and the point-charge limit).

## External point charges and fields

`ferric_core::external_potential::ExternalPotential` holds fixed point charges
$\{q_c, \mathbf{r}_c\}$, Gaussian-smeared charges $\{q_s, \mathbf{r}_s, w_s\}$
and an optional uniform field $\mathbf{E}$. It enters every SCF variant once,
before the iterations: `oneelectron::hcore_with_external` adds the one-electron
operator to $\mathbf{h}$, and `driver.rs` adds the classical terms to the
nuclear repulsion $V_{nn}$. Nothing depends on the density, so the term needs
no Fock update; `ext = None` and an empty potential take the plain
`hcore` path. Charges interact with the nuclei through
$Z^{\rm eff}_A$ (`Atom::effective_z`, the valence charge on an ECP atom);
ghost atoms are skipped. Charge–charge interactions inside the environment are
not computed (they are a constant).

**Operators and energies.** For an electron at $\mathbf{r}$,

$$h^{\rm ext}_{\mu\nu} = -\sum_c q_c\Big\langle\mu\Big|\frac{1}{|\mathbf{r}-\mathbf{r}_c|}\Big|\nu\Big\rangle
 - \sum_s q_s\Big\langle\mu\Big|\frac{\operatorname{erf}(|\mathbf{r}-\mathbf{r}_s|/w_s)}{|\mathbf{r}-\mathbf{r}_s|}\Big|\nu\Big\rangle
 + \mathbf{E}\cdot\langle\mu|\mathbf{r}|\nu\rangle,$$

$$V^{\rm ext}_{nn} = \sum_A Z^{\rm eff}_A\Big[\sum_c \frac{q_c}{R_{Ac}}
 + \sum_s q_s\frac{\operatorname{erf}(R_{As}/w_s)}{R_{As}}\Big]
 - \mathbf{E}\cdot\sum_A Z^{\rm eff}_A\mathbf{R}_A .$$

The point charges are appended to the nuclear-attraction engine
(`nuclear_with_external`). The smeared operator is built as
$-q_s(\mu\nu|g_s)/\!\int g_s$ from three-centre integrals against a normalized
$s$ Gaussian of exponent $\zeta_s = 1/w_s^2$ (`smeared_attraction`). The field
term uses the dipole integrals $\langle\mu|\mathbf{r}-\mathbf{O}|\nu\rangle$ at
$\mathbf{O} = 0$ (`field_hcore_term`); the electron's charge $-1$ is folded in,
so a charge $q$ at $\mathbf{R}$ has energy $-q\,\mathbf{E}\cdot\mathbf{R}$.

**Smeared charges.** $\operatorname{erf}(r/w)/r$ is the potential of the unit
Gaussian $(\zeta/\pi)^{3/2}e^{-\zeta r^2}$, $\zeta = 1/w^2$: it satisfies
$-\nabla^2\phi = 4\pi\rho$, is finite at the centre ($2/(\sqrt\pi\,w)$), and
tends to $1/r$ as $w \to 0$. $w = 0$ is not a valid smeared charge; the point
charge is the separate representation. Only the environment charge is smeared:
the nuclei stay points, so $V^{\rm ext}_{nn}$ has the form above. With
$f(R) = \operatorname{erf}(R/w)/R$ and $g = \operatorname{erf}(R/w)$,

$$f' = \frac{g'}{R} - \frac{g}{R^2},\qquad
f'' = -g'\Big(\frac{2}{w^2} + \frac{2}{R^2}\Big) + \frac{2g}{R^3},\qquad
g' = \frac{2e^{-R^2/w^2}}{\sqrt\pi\,w}$$

(`smeared_radial_derivs`). With $\mathbf{d} = \mathbf{R}_A - \mathbf{r}_s$, the
classical gradient is $Z_Aq_s f'(R)\,\mathbf{d}/R$ on the atom and its negative on
the site (`smeared_charge_nuclear_site_gradient`), and the Hessian block on
atom $A$'s diagonal is

$$Z_Aq_s\Big[f''\frac{\mathbf{d}\mathbf{d}^{\mathsf T}}{R^2}
 + \frac{f'}{R}\Big(\mathbf{1} - \frac{\mathbf{d}\mathbf{d}^{\mathsf T}}{R^2}\Big)\Big],$$

whose $w\to0$ limit is the point block $Z_Aq_c(3\mathbf{d}\mathbf{d}^{\mathsf T}/R^5 - \mathbf{1}/R^3)$.
The fixed charges do not move with the atoms, so the classical Hessian is block
diagonal; the field is linear in $\mathbf{R}$ and contributes none.

**Field origin.** At an origin $\mathbf{O}$ the electronic term is
$\mathbf{E}\cdot\mathrm{Tr}[\mathbf{P}\langle\mathbf{r}\rangle] - (\mathbf{E}\cdot\mathbf{O})\mathrm{Tr}[\mathbf{P}\mathbf{S}]$
and the nuclear term gains $+(\mathbf{E}\cdot\mathbf{O})\sum_AZ^{\rm eff}_A$, so

$$E(\mathbf{O}) - E(0) = (\mathbf{E}\cdot\mathbf{O})\Big(\sum_A Z^{\rm eff}_A - N\Big)
 = (\mathbf{E}\cdot\mathbf{O})\,Q .$$

For a neutral molecule the energy does not depend on the origin. For a charged
one it shifts by a constant that changes neither the orbitals (the shift adds
$-(\mathbf{E}\cdot\mathbf{O})\mathbf{S}$ to $\mathbf{h}$, which moves every
orbital energy by the same amount) nor the gradient; the code's
$\mathbf{O} = 0$ fixes the constant.

**Gradient** (`gradient::oneelectron_gradient` with `ext`). Three kinds of
term:

1. *Point charges, electronic.* The nuclear-attraction derivative engine is
   built with the external charges appended (`set_point_charges_extra`,
   `compute_1e_deriv_block_n`, $3(2 + N_{\rm at} + N_c)$ blocks). The two
   shell-centre blocks and the $N_{\rm at}$ real-nucleus blocks are contracted
   with $\mathbf{P}$; the $N_c$ charge-centre blocks are computed and
   discarded, because the charges are fixed. Their contraction is minus the
   force on each charge, which QM/MM takes from `electric_field_at_points`
   instead. Analytic.
2. *Smeared charges, electronic.* $\sum_{\mu\nu}P_{\mu\nu}\,\partial h^{\rm ext}_{\mu\nu}/\partial\mathbf{R}_A$
   from the three-centre derivative engine, blocks 1 and 2 (the two orbital
   centres; `smeared_charge_qm_gradient`). The site derivative is
   $-(\text{block 1} + \text{block 2})$ by translational invariance
   (`smeared_site_forces`), plus the classical site term above. Analytic.
3. *Classical.* `charge_nuclear_gradient` ($-Z_Aq_c\mathbf{d}/R^3$ and the
   smeared form) and `field_nuclear_gradient` ($-Z^{\rm eff}_A\mathbf{E}$).
   Analytic.

The fourth term, the field–density derivative
$\partial/\partial\mathbf{R}_A\,\mathrm{Tr}[\mathbf{P}\,\mathbf{E}\cdot\langle\mathbf{r}\rangle]$
at fixed $\mathbf{P}$, is a **central finite difference** of
`field_hcore_term` rebuilt at displaced basis centres, step $h = 10^{-4}$ Bohr
(`field_density_gradient_fd`), because the shim has no derivative engine for
`emultipole1`. Its truncation error is $O(h^2)$, about $10^{-10}$ on the
model of the proof. Translating every basis centre by $\mathbf{t}$ changes
$\langle\mu|\mathbf{r}|\nu\rangle$ by $\mathbf{t}S_{\mu\nu}$, so the atom sum of
this term is $\mathbf{E}\,\mathrm{Tr}[\mathbf{P}\mathbf{S}] = N\mathbf{E}$;
with the field–nuclear rows the net force on the molecule is $Q\mathbf{E}$.
The exact derivative obeys both; the finite difference obeys them to its
$O(h^2)$ error. The Pulay and two-electron terms are those of
the plain gradient at the embedded density.

Proof: `scripts/external_potential_sympy.py` (E1 the smeared potential and its
point limit; E2 $f'$ and $f''$; E3 the classical gradient, site gradient and
Hessian blocks; E4 the origin shift; E5 the finite-difference field–density
term against analytic dipole-integral derivatives of an $s$-Gaussian model,
with the sum rule and net force).

## QM/MM boundary and the full gradient

`qmmm.rs` partitions one structure into a QM molecule and an
`ExternalPotential` of MM charges (section "External point charges and
fields"), caps cut bonds with link hydrogens, optionally moves the charge of
each cut bond's MM host, and maps the gradient of the embedded SCF back onto
every atom of the structure. MM charges on atoms selected into the QM region
are dropped (they would be counted twice).

**Link atoms** (`QmmmSystem::with_link_atoms`). A bond $(Q, M)$ with exactly
one end in the QM region gets a hydrogen at

$$\mathbf{R}_L = \mathbf{R}_Q + g(\mathbf{R}_M - \mathbf{R}_Q) = (1-g)\mathbf{R}_Q + g\mathbf{R}_M,
\qquad 0 < g < 1,$$

with $g = 1.09/1.53$ by default (`DEFAULT_LINK_SCALE`). $\mathbf{R}_L$ is not a
variable: it moves with its two hosts, so

$$\frac{\partial E}{\partial\mathbf{R}_Q} \mathrel{+}= (1-g)\frac{\partial E}{\partial\mathbf{R}_L},
\qquad
\frac{\partial E}{\partial\mathbf{R}_M} \mathrel{+}= g\,\frac{\partial E}{\partial\mathbf{R}_L},$$

and the two contributions sum to $\partial E/\partial\mathbf{R}_L$, so a rigid
translation still has zero net gradient. The link hydrogen is a real QM atom
($Z = 1$, basis functions) in the SCF and in the classical charge–nuclear
energy.

**Boundary charges** (`QmmmSystem::with_boundary_charges`). For each MM host
$M_1$ of a cut bond with charge $q$ and its $n$ MM neighbours $M_2^{(j)}$
(bonded MM atoms that are not themselves hosts; a host bonded to another host
is an error, as is a host with no MM neighbour under RC/RCD):

| Scheme | $q(M_1)$ | $q(M_2^{(j)})$ | added at $\mathbf{m}_j = \tfrac12(\mathbf{R}_{M_1} + \mathbf{R}_{M_2^{(j)}})$ |
|---|---|---|---|
| `Keep` | $q$ | unchanged | nothing |
| `DeleteHost` (Z1) | $0$ | unchanged | nothing |
| `RedistributedCharge` (RC) | $0$ | unchanged | $q/n$ |
| `RedistributedChargeDipole` (RCD) | $0$ | $-q/n$ added | $2q/n$ |

Z1 changes the total charge by $-q$. RC and RCD conserve it. The dipole of
the group $\{M_1, M_2^{(j)}\}$ about any origin changes under RC by

$$\Delta\mathbf{d}_{\rm RC} = \sum_j \frac qn\,\mathbf{m}_j - q\mathbf{R}_{M_1}
 = \frac q2\Big(\frac1n\sum_j\mathbf{R}_{M_2^{(j)}} - \mathbf{R}_{M_1}\Big),$$

which is nonzero unless $M_1$ sits at the centroid of its neighbours, and
under RCD by

$$\Delta\mathbf{d}_{\rm RCD} = \sum_j\Big[\frac{2q}{n}\,\mathbf{m}_j - \frac qn\mathbf{R}_{M_2^{(j)}}\Big] - q\mathbf{R}_{M_1}
 = \sum_j \frac qn\mathbf{R}_{M_1} - q\mathbf{R}_{M_1} = 0$$

for every $n$. The charges are constants; only the midpoint positions depend on
the geometry. The embedding charges are ordered as the MM atoms with a nonzero
effective charge (ascending index), then the midpoint charges
(`active_charges`); `mm_charge_positions`, `mm_forces` and `full_gradient` all
use this order.

**Forces on the charges** (`mm_forces`). For a point charge the force is
$q_c\,\mathbf{E}^{\rm QM}(\mathbf{r}_c)$, the field of the QM nuclei and density at
the charge (`electric_field_at_points`). For a smeared charge it is minus the
site derivative of the embedded energy (`gradient::smeared_site_forces`,
section "External point charges and fields"), which is not $q_s\mathbf{E}(\mathbf{r}_s)$.

**Full gradient** (`full_gradient`). Given the QM gradient (real QM rows, then
link rows) and the forces $\mathbf{F}_c$:

1. real QM rows go to their atoms unchanged;
2. each link row is split $(1-g, g)$ onto $(Q, M)$;
3. an atom-centred charge adds $-\mathbf{F}_c$ to its atom;
4. a midpoint charge adds $-\tfrac12\mathbf{F}_c$ to each of $M_1$ and $M_2$,
   since $\partial\mathbf{m}/\partial\mathbf{R}_{M_1} = \partial\mathbf{m}/\partial\mathbf{R}_{M_2} = \tfrac12\mathbf{1}$.

An MM atom whose charge is zero and that hosts nothing has a zero row. The
result is $dE/d\mathbf{R}$ of the embedded SCF energy with the partition
rebuilt at each geometry (`QmmmSystem::with_coordinates` does that rebuild in
`optimize_qmmm`). Its rows sum to zero for an energy that depends on relative
positions only. It contains no force-field term and no polarizable term:
`qmmm_mm_terms`/`full_gradient_with_mm` add the force field and
`full_gradient_with_polarizable` adds the Thole terms of section "Thole
polarizable embedding".

Proof: `scripts/qmmm_boundary_sympy.py` (Q1 the link chain rule; Q2 charge and
dipole bookkeeping of Z1, RC and RCD for 1 to 5 neighbours; Q3
`full_gradient` against a central difference of a model embedded energy over
the full structure, for all four schemes, with the swapped link split and an
unsplit midpoint as negative controls; Q4 translational invariance).

## Thole polarizable embedding

`polarizable.rs` adds induced point dipoles $\boldsymbol\mu_i$ on MM sites of
isotropic polarizability $\alpha_i$ (Applequist with Thole damping),
re-solved from the current density every SCF iteration
(`driver::solvent_terms`).

**Induction.** The field at site $i$ is $\mathbf{E}_i = \mathbf{E}^{\rm QM}_i(\mathbf{P}) + \mathbf{E}^{\rm perm}_i$:
the field of the QM nuclei and total density (`qmmm::electric_field_at_points`,
point-charge nuclear-attraction derivative blocks) plus that of the
environment's permanent charges. The dipoles solve the dense $3N\times3N$ system
(`solve_induction`)

$$\big(\boldsymbol\alpha^{-1} - \mathbf{T}\big)\boldsymbol\mu = \mathbf{E},
\qquad
\mathbf{T}_{ij} = \frac{3\lambda_5\,\hat{\mathbf{r}}\hat{\mathbf{r}}^{\mathsf T} - \lambda_3\mathbf{1}}{r^3}
\quad(i\ne j,\ \mathbf{r} = \mathbf{R}_i - \mathbf{R}_j),$$

with $\mathbf{T}_{ii} = 0$ and $\mathbf{T}_{ij} = 0$ for excluded pairs, and the
exponential Thole screening (`thole_screening`, $a = 2.1304$ by default)

$$u = \frac{r}{(\alpha_i\alpha_j)^{1/6}},\quad v = au,\qquad
\lambda_3 = 1 - \Big(1 + v + \frac{v^2}{2}\Big)e^{-v},\qquad
\lambda_5 = \lambda_3 - \frac{v^3}{6}e^{-v}.$$

$\lambda_5 = \lambda_3 - \tfrac u3\,d\lambda_3/du$, which makes
$T_{ab} = \partial_a\partial_b\phi(r)$ for $\phi'(r) = -\lambda_3/r^2$: the damped
tensor is the Hessian of a radial function, hence symmetric in $a, b$ and
even in $\mathbf{r}$, $\mathbf{T}_{ij} = \mathbf{T}_{ji}$, and
$\boldsymbol\alpha^{-1} - \mathbf{T}$ is symmetric. $a\to\infty$ (or
`thole_a: None`) gives $\lambda_3 = \lambda_5 = 1$, the bare dipole tensor.
$\mathbf{E}^{\rm perm}_i$ skips permanent charges colocated (within $10^{-8}$
Bohr) with site $i$ or with a site excluded from $i$. For a Gaussian-smeared
permanent charge it uses the point-charge field $q\,\mathbf{r}/r^3$, not the
field of the Gaussian (`permanent_field_at_sites`); the gradients below use the
same point form, so the gradient is that of the energy computed.

**Energy and the stationary functional.** The polarization energy is
$E_{\rm pol} = -\tfrac12\boldsymbol\mu\cdot\mathbf{E}$, added outside the
$\tfrac12\mathrm{Tr}[\mathbf{P}(\mathbf{h}+\mathbf{F})]$ trace. It is the value at
$\boldsymbol\mu^*$ of

$$W[\boldsymbol\mu] = -\boldsymbol\mu\cdot\mathbf{E}
 + \tfrac12\boldsymbol\mu\cdot\boldsymbol\alpha^{-1}\boldsymbol\mu
 - \tfrac12\boldsymbol\mu\cdot\mathbf{T}\boldsymbol\mu,$$

which is stationary there: $\partial W/\partial\boldsymbol\mu = 0$ is the
induction equation (the symmetry of $\mathbf{T}$ is used), and
$W[\boldsymbol\mu^*] = -\boldsymbol\mu^*\cdot\mathbf{E} + \tfrac12\boldsymbol\mu^*\cdot\mathbf{E}
= -\tfrac12\boldsymbol\mu^*\cdot\mathbf{E}$. Every derivative of $E_{\rm pol}$
is therefore a derivative of $W$ at fixed $\boldsymbol\mu = \boldsymbol\mu^*$.
Differentiating $-\tfrac12\boldsymbol\mu\cdot\mathbf{E}$ at fixed
$\boldsymbol\mu$ instead is wrong: that expression is not stationary in
$\boldsymbol\mu$, and it misses both a factor 2 on the field term and the
whole $\mathbf{T}$ term.

**Fock term.** $\mathbf{E}^{\rm QM}_i$ is linear in $\mathbf{P}$,
$\mathbf{E}^{\rm QM}_i = \mathbf{E}^{\rm nuc}_i + \sum_{\mu\nu}P_{\mu\nu}\mathbf{f}_{i,\mu\nu}$, so

$$V^{\rm ind}_{\mu\nu} = \frac{\partial W}{\partial P_{\mu\nu}}\Big|_{\boldsymbol\mu^*}
 = -\sum_i \boldsymbol\mu_i\cdot\mathbf{f}_{i,\mu\nu},$$

with no factor $\tfrac12$, added to every spin Fock matrix
(`build_v_induced`). The field integral $\mathbf{f}_{i,\mu\nu}$ is built from
three-centre integrals against a pure $p$ shell of exponent $\zeta$ = `dipole_zeta`
(default $10^4$ Bohr$^{-2}$) at the site,
$f_{i,c} = \mathcal{N}_p(\zeta)\,(\mu\nu|p_{(c+2)\bmod 3})$ with
$\mathcal{N}_p(\zeta) = 2^{1/4}\zeta^{5/4}/(2\pi^{3/4})$ (`norm_int_p_shell`) and
the overall sign carried by `build_v_induced`. $\mathcal{N}_p$ is the reciprocal
of the dipole moment $\int x\,p_x$ of the unit-normalized
$p_x = \mathcal{N}x\,e^{-\zeta r^2}$, so $(\mu\nu|p_x)\,\mathcal{N}_p$ is the
potential integral of a unit point dipole as $\zeta\to\infty$; and
$(c+2)\bmod 3$ maps field axis $(x, y, z)$ to the real solid harmonics of a pure
$l = 1$ shell, ordered $m = -1, 0, +1 = (y, z, x)$. At finite $\zeta$ the
Fock and gradient integrals are those of a Gaussian dipole of width
$\zeta^{-1/2} \approx 0.01$ Bohr, while $\mathbf{E}^{\rm QM}$ in the induction
uses the point field, so $V^{\rm ind}$ equals $\partial W/\partial\mathbf{P}$ up
to that finite-width error.

**Gradients** (all at fixed $\boldsymbol\mu^*$, from $W$):

$$\frac{dE}{d\mathbf{R}_A} \mathrel{+}= -\sum_i\boldsymbol\mu_i\cdot\frac{\partial\mathbf{E}^{\rm QM}_i}{\partial\mathbf{R}_A},
\qquad
\frac{dE}{d\mathbf{R}_i} = -\boldsymbol\mu_i\cdot\frac{\partial\mathbf{E}_i}{\partial\mathbf{R}_i}
 - \sum_{j\ne i}\boldsymbol\mu_i\cdot\frac{\partial\mathbf{T}_{ij}}{\partial\mathbf{R}_i}\boldsymbol\mu_j,
\qquad
\frac{dE}{d\mathbf{r}_c} = -\sum_i\boldsymbol\mu_i\cdot\frac{\partial\mathbf{E}^{\rm perm}_i}{\partial\mathbf{r}_c}.$$

- QM atoms (`qm_gradient_contribution`): the basis-centre derivative of
  $V^{\rm ind}$ contracted with $\mathbf{P}$ (three-centre derivative blocks 1
  and 2), plus the nuclear part $-\boldsymbol\mu_i\cdot\partial\mathbf{E}^{\rm nuc}_i/\partial\mathbf{R}_A$
  in closed form.
- Sites (`site_gradient`): the electronic part by translational invariance,
  $-(\text{block 1} + \text{block 2})$; the nuclear and permanent-charge parts
  in closed form; and the $\mathbf{T}$ term. $W$ holds
  $-\tfrac12\boldsymbol\mu_i\cdot\mathbf{T}_{ij}\boldsymbol\mu_j$ and
  $-\tfrac12\boldsymbol\mu_j\cdot\mathbf{T}_{ji}\boldsymbol\mu_i$, equal by the
  symmetry above, so the loop over $j$ for fixed $i$ carries coefficient 1.
  $\partial\mathbf{T}_{ij}/\partial\mathbf{R}_i$ is a central difference of
  `thole_tensor`, step $h = 10^{-6}$ Bohr; its rounding error is about
  $\epsilon|\mathbf{T}|/h$ ($\sim10^{-10}$ relative), its truncation error
  $O(h^2)$ is far below that. The analytic form is
  $\partial_k T_{ab} = \partial_a\partial_b\partial_k\phi$ with the $\phi$ above.
- Permanent charges (`charge_gradient_contribution`): the second formula's
  charge half; `qmmm::full_gradient_with_polarizable` adds the site and charge
  rows to `full_gradient`.

With $\alpha_i\to0$ the dipoles vanish and every term above is zero; an empty
site list returns before any solve and is bit-identical to no embedding.

Proof: `scripts/thole_polarizable_sympy.py` (T1 the $\lambda_5$ identity, $\mathbf{T}$
as a Hessian, its symmetry and the undamped limit; T2 stationarity of $W$;
T3 site and charge gradients against finite differences of $E_{\rm pol}$ with
$\boldsymbol\mu$ re-solved, with the coefficient-$\tfrac12$ $\mathbf{T}$ term and
the naive fixed-$\boldsymbol\mu$ derivative as negative controls, and the
$h = 10^{-6}$ difference of $\mathbf{T}$ against its analytic derivative; T4
the Fock term; T5 the $\alpha\to0$ limit; T6 $\mathcal{N}_p$ and the $p$-shell
order).

## Atomic charges

`properties.rs` and `properties/hirshfeld_grid.rs` compute six charge models.
All use $Z^{\rm eff}_A$ = `Atom::effective_z` (the valence charge on an ECP atom)
and the total spin-summed density $\mathbf{P}$; nothing in them is closed-shell
specific. $N$ = `Molecule::nelec` $= \sum_A Z^{\rm eff}_A - Q$.

**Population partitions.**

$$q^{\rm Mulliken}_A = Z^{\rm eff}_A - \sum_{\mu\in A}(\mathbf{P}\mathbf{S})_{\mu\mu},
\qquad
q^{\rm L\ddot owdin}_A = Z^{\rm eff}_A - \sum_{\mu\in A}(\mathbf{S}^{1/2}\mathbf{P}\mathbf{S}^{1/2})_{\mu\mu}.$$

$\mathbf{S}^{1/2}$ comes from the eigendecomposition of $\mathbf{S}$; a
non-positive eigenvalue is an error. Both sum to $\sum_AZ^{\rm eff}_A -
\mathrm{Tr}[\mathbf{P}\mathbf{S}]$, since
$\mathrm{Tr}[\mathbf{S}^{1/2}\mathbf{P}\mathbf{S}^{1/2}] = \mathrm{Tr}[\mathbf{P}\mathbf{S}]$,
which is $Q$ for an SCF density. There is no rescaling. The Löwdin populations
are diagonal elements of a positive semidefinite matrix and so nonnegative.

**Grid partitions.** On the Becke–Lebedev grid (`build_atomic_grid`, default
config) with weights $w_g$ that already contain the Becke partition of the
point's home atom, and $\rho_g = \sum_{\mu\nu}P_{\mu\nu}\chi_\mu(g)\chi_\nu(g)$,

$$n^{\rm Becke}_A = \sum_{g:\,{\rm home}(g)=A} w_g\rho_g,
\qquad
n^{\rm Hirsh}_A = \sum_g w_g\rho_g\,\frac{\rho^0_A(g)}{\sum_B\rho^0_B(g) + 10^{-12}},$$

$$q_A = Z^{\rm eff}_A - s\,n_A,\qquad s = \frac{N}{\sum_B n_B}.$$

The rescale $s$ absorbs the quadrature error of $\int\rho$ (and the
$10^{-12}$ floor, which makes the Hirshfeld weights sum to slightly less than one
where every proatom density is tiny), so $\sum_Aq_A = Q$ exactly. The
Hirshfeld proatom $\rho^0_A$ is the provider's spherically averaged free-atom
SCF density, or the single Slater function $Z_A\xi^3/\pi\,e^{-2\xi r}$
(`slater_xi_for_z`) for an atom without one; both are built for the full
nuclear charge $Z_A$.

**CHELPG** (`solve_chelpg_normal_equations`). On the grid points $\mathbf{r}_g$
kept by `chelpg_in_shell` (outside every Bondi sphere scaled by
`vdw_scale`, and within `outer_cutoff` of at least one scaled sphere's
surface) with the molecular potential
$V_g = \sum_BZ^{\rm eff}_B/|\mathbf{r}_g-\mathbf{R}_B| - \sum_{\mu\nu}P_{\mu\nu}\langle\mu|1/|\mathbf{r}-\mathbf{r}_g||\nu\rangle$,
the charges minimize $\tfrac12\sum_g(V_g - \sum_Aq_A/r_{gA})^2$ subject to
$\sum_Aq_A = Q$. Stationarity of
$\tfrac12\chi^2 + \lambda(\sum_Aq_A - Q)$ is the bordered system

$$\begin{pmatrix}\mathbf{A} & \mathbf{1}\\ \mathbf{1}^{\mathsf T} & 0\end{pmatrix}
\begin{pmatrix}\mathbf{q}\\ \lambda\end{pmatrix}
= \begin{pmatrix}\mathbf{b}\\ Q\end{pmatrix},
\qquad A_{AB} = \sum_g\frac{1}{r_{gA}r_{gB}},\quad b_A = \sum_g\frac{V_g}{r_{gA}},$$

solved once by LU. The points carry no weights.

**RESP** (`solve_resp_restrained`). The same grid and system, with the
hyperbolic restraint on every atom with $Z \ne 1$. The code iterates

$$\Big(\mathbf{A} + \mathrm{diag}_{A\,\in\,\rm heavy}\frac{a}{\sqrt{(q^{(k)}_A)^2 + b^2}}\Big)\mathbf{q}^{(k+1)}
 + \lambda\mathbf{1} = \mathbf{b},\qquad \sum_Aq^{(k+1)}_A = Q,$$

from $\mathbf{q}^{(0)} = 0$ (so the first solve already carries the slope $a/b$)
until $\max_A|q^{(k+1)}_A - q^{(k)}_A| < 10^{-8}$ or 50 iterations, with
$a = 0.0005$ and $b = 0.1$ by default. This is a fixed-point (iteratively
reweighted least squares) iteration, not Newton's method: the restraint's
gradient $aq/\sqrt{q^2+b^2}$ is written as $[a/\sqrt{q^2+b^2}]\,q$ with the
bracket frozen at the previous iterate. At its fixed point,
$(\mathbf{A}\mathbf{q} - \mathbf{b}) + a\,q_A/\sqrt{q_A^2+b^2} + \lambda = 0$, which
is the stationarity condition of

$$\tfrac12\sum_g\Big(V_g - \sum_A\frac{q_A}{r_{gA}}\Big)^2
 + a\sum_{A\,\in\,\rm heavy}\Big(\sqrt{q_A^2+b^2} - b\Big)
 \quad\text{subject to}\quad \sum_Aq_A = Q .$$

The $\tfrac12$ on the least-squares term matters for the meaning of $a$: the
objective without it has a different minimizer, equal to the code's with $a/2$.
$a \to 0$ recovers CHELPG. The iteration returns its last iterate without an
error if it reaches 50 iterations.

Proof: `scripts/atomic_charges_sympy.py` (A1 Mulliken and Löwdin sum rules and
Löwdin nonnegativity; A2 the grid rescale; A3 CHELPG's bordered system against
a null-space constrained least-squares solve; A4 the RESP fixed point as the
constrained minimizer of the objective above, the objective without the
$\tfrac12$ as a negative control, and the $a\to0$ limit).

## Density-fitted J/K nuclear gradient

`df_gradient.rs` differentiates the fitted energies the SCF reports, not the
four-centre ones. The orbital response is carried by the energy-weighted
density of the Hartree–Fock gradient, because the SCF made the fitted energy
stationary; what remains is the derivative at fixed density of the raw
three-index integrals $(P|\mu\nu)$, which act on all three centres including the
auxiliary one, and of the two-index metric $V_{PQ} = (P|Q)$ of the same
operator, on both auxiliary centres.

**RI-J.** With the Cholesky-solved metric (no truncation, as `DfJ` builds it),
$d_P = \sum_{\mu\nu}D_{\mu\nu}(P|\mu\nu)$ and $c = V^{-1}d$:

$$E_J = \tfrac12 d^{\mathsf T}V^{-1}d,\qquad
\partial E_J = \sum_P c_P\sum_{\mu\nu}D_{\mu\nu}\,\partial(P|\mu\nu) - \tfrac12 c^{\mathsf T}(\partial V)c,$$

using $\partial V^{-1} = -V^{-1}(\partial V)V^{-1}$.

**RI-K.** One channel per density $D_s = X_sX_s^{\mathsf T}$ with weight
$\alpha_s$ ($-c_K/4$ for a closed-shell total density, $-c_K/2$ per spin for
UHF and ROHF, $c_K$ the exact-exchange fraction; `exchange_channels`), with
$L^s_{P,ij} = \sum_{\mu\nu}X^s_{\mu i}(P|\mu\nu)X^s_{\nu j}$ and $V^{+}$ the
metric inverse the exchange builder applies:

$$E_K = \sum_s\alpha_s\sum_{ij}L^{s\,\mathsf T}_{ij}V^{+}L^s_{ij},\qquad
\partial E_K = \sum_s 2\alpha_s\sum_{P,ij}\Gamma^s_{P,ij}\,\partial L^s_{P,ij} + \sum_{PQ}Y_{PQ}\,\partial V^{+}_{PQ},$$

$\Gamma^s = V^{+}L^s$, $Y = \sum_s\alpha_s\sum_{ij}L^s_{ij}L^{s\,\mathsf T}_{ij}$, and the
factor 2 from the two $L$ factors and the symmetry of $V^{+}$. Both gradients are
two contractions with weights built once (`df_jk_gradient_core`):

$$\partial E_{\rm DF} = \sum_{P,\mu\nu}G_{P,\mu\nu}\,\partial(P|\mu\nu) + \sum_{PQ}H_{PQ}\,\partial V_{PQ},\qquad
G_{P,\mu\nu} = c_PD_{\mu\nu} + \sum_s2\alpha_s\big(X^s\Gamma^s_PX^{s\,\mathsf T}\big)_{\mu\nu},\qquad
H = -\tfrac12cc^{\mathsf T} + H_K .$$

**The truncated exchange metric.** `DfK` forms $V^{+} = f(V)$ from
$V = U\Lambda U^{\mathsf T}$ with $f(\lambda) = 1/\lambda$ above the cut
(`df_k::metric_lindep_cut`) and $0$ below. The derivative of a spectral
function is the Daleckii–Krein formula,

$$\partial V^{+} = U\big(\mathsf F\circ U^{\mathsf T}\partial V\,U\big)U^{\mathsf T},\qquad
\mathsf F_{kl} = \frac{f(\lambda_k)-f(\lambda_l)}{\lambda_k-\lambda_l},\quad \mathsf F_{kk} = f'(\lambda_k),$$

so $H_K = U\big(\mathsf F\circ U^{\mathsf T}YU\big)U^{\mathsf T}$. Case by case
(`truncated_inverse_divided_difference`): kept/kept $-1/(\lambda_k\lambda_l)$,
kept $k$/dropped $l$ $1/(\lambda_k(\lambda_k-\lambda_l))$, dropped $k$/kept $l$
$1/(\lambda_l(\lambda_l-\lambda_k))$, dropped/dropped $0$. With nothing dropped
this is $-V^{+}(\partial V)V^{+}$; the kept/dropped entries are the rotation of
kept modes into dropped ones, which that formula misses. The formula is exact
for a fixed partition of the spectrum, which holds when the cut lies in a gap.

The gradient additionally treats modes below `DFK_GRADIENT_NOISE_FLOOR`
($10^{-7}$) as dropped even where the energy kept them, because
$\Gamma_k \propto 1/\lambda_k$ amplifies integral noise there; it warns when
the energy held by that band exceeds `NOISE_BAND_ENERGY_WARN` ($10^{-6}$ Ha).
The gradient is then that of the energy with this band removed, and differs
from the reported energy's gradient by the band's derivative. With no metric
eigenvalue below $10^{-7}$, the usual case for Coulomb and erfc metrics, the
floor changes nothing.

Proof: `scripts/df_jk_gradient_sympy.py` (D1 RI-J against the exact sympy
derivative in rationals; D2 RI-K with every mode kept, likewise; D3 the
case-by-case kernel against a fourth-order finite difference of $E_K$ with the
cut in a gap, with $-V^{+}(\partial V)V^{+}$ as the negative control).

## Geometry searches

Three searches use the analytic gradient: a minimizer (`optimize.rs`), a
saddle search (`saddle.rs`) and a reaction-path walk (`irc.rs`).

**Minimization** (`optimize_coordinates`, Cartesian default). BFGS on the
inverse Hessian $H$, started from the inverse of the Lindh model Hessian
(`initial_hessian = "lindh"`, the default; next section) or from the identity
(`"simple"`).
There is no line search. Each step is $p = -Hg$ scaled uniformly onto the
trust radius $\Delta$ when longer,

$$s = \alpha p,\qquad \alpha = \min\!\Big(1, \frac{\Delta}{\lVert p\rVert}\Big).$$

With `trust_update = "fixed"` $\Delta = r_{\rm trust}$ and the new point is
always accepted. With `"ratio"` (the default; `ratio_trust_update`; Nocedal and Wright,
*Numerical Optimization*, Algorithm 4.1 with $\eta = 0$) the quadratic model
$m(s) = g^{\mathsf T}s + \tfrac12 s^{\mathsf T}Bs$, $B = H^{-1}$, predicts
the change. Because $Bs = -\alpha BHg = -\alpha g$,

$$\Delta E_{\rm pred} = \alpha\, g^{\mathsf T}p\,\big(1 - \tfrac{\alpha}{2}\big) < 0,$$

with no linear solve. With $r = \Delta E/\Delta E_{\rm pred}$ the radius becomes
$\tfrac14\lVert s\rVert$ when $r < \tfrac14$, $2\Delta$ when $r > \tfrac34$ and
$\lVert s\rVert \ge 0.99\,\Delta$, and stays otherwise, clamped to
$[0.01, 10]\,r_{\rm trust}$. The step is kept when $\Delta E < 10^{-8}$ Ha
(the energy fell, or rose by less than the noise of an SCF energy); a rejected
step costs its evaluation, the search stays at the old point, and its
$(s, y)$ pair still updates $H$. When both $\lvert\Delta E\rvert$ and
$\lvert\Delta E_{\rm pred}\rvert$ are below $10^{-8}$ the ratio is noise and
$\Delta$ is kept. With $y = g_{k+1} - g_k$ and
$\rho = 1/(y^{\mathsf T}s)$,

$$H' = H + \rho^2\big(y^{\mathsf T}s + y^{\mathsf T}Hy\big)ss^{\mathsf T} - \rho\big(Hys^{\mathsf T} + sy^{\mathsf T}H\big).$$

This is the inverse of the direct BFGS update
$B' = B - \frac{Bss^{\mathsf T}B}{s^{\mathsf T}Bs} + \frac{yy^{\mathsf T}}{y^{\mathsf T}s}$
of $B = H^{-1}$ (Sherman–Morrison–Woodbury), it satisfies the secant condition
$H'y = s$, and it keeps $H$ positive definite when $y^{\mathsf T}s > 0$. The
converse also holds for any update that satisfies the secant condition:
$y^{\mathsf T}H'y = y^{\mathsf T}s$, so $y^{\mathsf T}s \le 0$ leaves no
positive-definite $H'$, and a later step $-H'g$ can point uphill. Under
`"fixed"` every step is accepted, so nothing downstream would catch that. The update is therefore
guarded (`guarded_inverse_bfgs_update`):

$$H' = \begin{cases} H & \lvert y^{\mathsf T}s\rvert \le 10^{-12}:\ \text{reset to the seed instead},\\ H & y^{\mathsf T}s \le \varepsilon\,\lVert y\rVert\,\lVert s\rVert,\ \ \varepsilon = 10^{-8}:\ \text{skipped},\\ \text{the update above} & \text{otherwise.} \end{cases}$$

A skipped update leaves $H$ positive definite by induction from the identity,
so every step is a descent direction. Above the threshold the guard changes
nothing, bit for bit. Skipping was chosen over Powell damping, which replaces
$y$ by $\theta y + (1-\theta)Bs$ so that $y^{\mathsf T}s$ becomes
$0.2\,s^{\mathsf T}Bs$. On the triplet-water optimizations where
$y^{\mathsf T}s < 0$ occurs, damping converged in the same or slightly more
steps (58 vs 58 UMP2, 91 vs 88 UHF/STO-3G). It also needs $Bs = H^{-1}s$, a
linear solve with the carried inverse. The saddle search does not use this
update. It carries an indefinite $B$ with the Bofill update below, and the IRC
walk carries no Hessian. Convergence needs at least one step and
$\lvert\Delta E\rvert < e_{\rm conv}$, $\max\lvert g\rvert < g_{\max}$ and
${\rm rms}(g) < g_{\rm rms}$ together. Redundant internals
(`CoordSystem::Internal`) run the same update on $g_q = G^{-}Bg_x$ from an
empirical diagonal Hessian, clip bonds against $r_{\rm trust}$ and angles
against 0.3 rad separately (`clip_internal_step`), update with the achieved
internal displacement under the same guard, and on a degenerate pair reset to the empirical diagonal.
Under `"ratio"` the internal search scales both radii by one factor $f \in
[0.01, 4]$, predicts $g_q^{\mathsf T}s + \tfrac12 s^{\mathsf T}H^{-1}s$ on the
achieved internal step $s$ (a linear solve, since a redundant step is not
$\alpha p$), and measures the step as
$\max(\lVert s_{\rm dist}\rVert/r_{\rm trust}, \lVert s_{\rm ang}\rVert/0.3)$
in the units of $f$. Two properties of this internal rule are known and not
changed. (1) The prediction is on the achieved step, so it can be $\ge 0$;
a rise predicted as a rise then gives $r > \tfrac34$ and the radius grows on
a rejected step (pregabalin at HF/STO-3G: 37 of 51 steps rejected, 22 in a
row at $f = 4$). (2) The step is clipped with separate factors for the
distance and the angular block, so $-g_q^{\mathsf T}DHg_q$, the first-order
change, is not sign-definite and a short step need not descend. Shrinking on
every rejection (a fix of 1 alone, tried and not kept) stalls alanine
dipeptide at $f = 0.01$: the same step, +1.3e-7 Ha uphill, is rejected for
the remaining 283 steps, which (2) would explain (not isolated further). Both
are reasons the coordinate default is Cartesian (see
*Measured defaults* below).

**Saddle points** (`find_saddle`). In mass-weighted Cartesians, with the
projected Hessian's eigenvalues $b_i$ (rigid modes shifted to $+10^3$) and
gradient components $g_i$, partitioned rational-function optimization solves a
separate augmented-Hessian problem per partition. For the followed mode $k$ the
step maximizes along $\left[\begin{smallmatrix}b_k & g_k\\ g_k & 0\end{smallmatrix}\right]$,
whose secular equation is $\lambda^2 - b_k\lambda - g_k^2 = 0$; the larger root

$$\lambda_p = \tfrac12\big(b_k + \sqrt{b_k^2 + 4g_k^2}\big) > \max(b_k, 0),\qquad \Delta q_k = -\frac{g_k}{b_k - \lambda_p},$$

so the denominator is negative and the step climbs whatever the sign of $b_k$.
For the other modes the shift is the root of
$\lambda_n = \sum_{i\ne k}g_i^2/(\lambda_n - b_i)$ below $\min_{i\ne k}b_i$, the
lowest eigenvalue of their augmented Hessian, and
$\Delta q_i = -g_i/(b_i - \lambda_n)$ descends (`prfo_step`, Newton iteration
from $\min(b_{\min}-1, 0)$, falling back to $b_{\min} - 10^{-3}$ if it leaves
the bracket). The Cartesian step is scaled uniformly onto the trust radius.
The Hessian is carried by the Bofill update (`bofill_update`), with
$e = \Delta g - B\Delta x$ and $\varphi = (e\cdot\Delta x)^2/(e\cdot e\,\Delta x\cdot\Delta x)$,

$$B' = B + \varphi\frac{ee^{\mathsf T}}{e\cdot\Delta x} + (1-\varphi)\Big[\frac{e\Delta x^{\mathsf T} + \Delta xe^{\mathsf T}}{\Delta x\cdot\Delta x} - \frac{(e\cdot\Delta x)\Delta x\Delta x^{\mathsf T}}{(\Delta x\cdot\Delta x)^2}\Big],$$

a mixture of the Murtagh–Sargent and Powell symmetric-Broyden corrections, each
of which satisfies $\Delta B\,\Delta x = e$, so $B'\Delta x = \Delta g$ for every
$\varphi$; unlike BFGS it does not force $B$ positive definite.

**Reaction paths** (`irc::walk`). With $q_i = \sqrt{m_i}x_i$ the chain rule gives
$\partial E/\partial q_i = g_i/\sqrt{m_i}$. Each step moves a length
$\ell = \min(\ell_0, 2\lVert g_q\rVert)$ down the mass-weighted gradient,

$$\Delta q = -\ell\frac{g_q}{\lVert g_q\rVert},\qquad \Delta x_i = \frac{\Delta q_i}{\sqrt{m_i}} = -\ell\frac{g_i}{m_i\lVert g_q\rVert},$$

which is steepest descent in $q$ and differs from Cartesian steepest descent
unless all masses are equal. The cap $2\lVert g_q\rVert$ shrinks the step near a
minimum, where a fixed length would orbit.

Proof: `scripts/geometry_search_steps_sympy.py` (B1 the secant condition in
symbols; B2 the inverse update against the inverted direct update, symmetry and
positive definiteness; B3 the curvature guard: $y^{\mathsf T}H'y = y^{\mathsf T}s$
in symbols, the guarded update positive definite for every pair, the unguarded
one indefinite for every $y^{\mathsf T}s < 0$ pair, and bitwise unchanged above
the threshold; P1 the followed-mode root and step in symbols; P2
`prfo_step` against the augmented-Hessian eigenvectors; F1 the two Bofill
corrections' secant conditions; F2 the update's secant condition, symmetry and
retained negative eigenvalue; I1 the mass-weighted step).

## Lindh model Hessian

`ferric_core::model_hessian` (R. Lindh, A. Bernhardsson, G. Karlström and
P.-Å. Malmqvist, Chem. Phys. Lett. 241, 423 (1995)). For atoms $i, j$ in
periods $p_i, p_j$ (H–He, Li–Ne, Na and heavier),

$$\rho_{ij} = \exp\!\big[\alpha_{p_ip_j}\big(r_{{\rm ref},p_ip_j}^2 - r_{ij}^2\big)\big],$$

with $\alpha$ = (1.0000, 0.3949, 0.3949; 0.3949, 0.2800, 0.2800; 0.3949,
0.2800, 0.2800) Bohr$^{-2}$ and $r_{\rm ref}$ = (1.35, 2.10, 2.53; 2.10, 2.87,
3.40; 2.53, 3.40, 3.40) Bohr (Table 1 of the paper). The force constants are

$$k_{ij} = k_r\rho_{ij},\qquad k_{ijk} = k_\varphi\rho_{ij}\rho_{jk},\qquad k_{ijkl} = k_\tau\rho_{ij}\rho_{jk}\rho_{kl},$$

$k_r = 0.45$ Ha/Bohr$^2$, $k_\varphi = 0.15$, $k_\tau = 0.005$ Ha/rad$^2$, so
$\rho = 1$ and $k = k_r$ at the reference distance, and
$\partial\ln\rho_{ij}/\partial r_{ij} = -2\alpha r_{ij}$. The Cartesian model is
a sum over every stretch, bend and torsion of the molecule, bonded or not,

$$H^{\rm L} = \sum_q k_q\, b_q b_q^{\mathsf T},\qquad b_q = \partial q/\partial x,$$

with the Wilson vectors of `primitive_derivatives`. Each term is a rank-one
positive semi-definite matrix, so $H^{\rm L}$ is symmetric and positive
semi-definite. Each primitive is invariant under rigid motions $t$, so
$b_q^{\mathsf T}t = 0$ and $H^{\rm L}t = 0$ term by term: the three
translations and three rotations are null vectors. Bends within 5° of linear
and torsions containing one are dropped (their $b_q$ diverge), and so are terms
with $\rho$ product below $10^{-8}$. Dropping a positive semi-definite term
keeps the sum positive semi-definite.

The optimizer seeds BFGS with an inverse. With $T$ the orthonormal rigid-body
basis and $Q$ its orthonormal complement, $Q^{\mathsf T}H^{\rm L}Q = VwV^{\mathsf T}$,

$$H_0 = QV\,{\rm diag}\big(1/\max(w_i, 10^{-3})\big)V^{\mathsf T}Q^{\mathsf T} + TT^{\mathsf T}.$$

The floor covers directions with no model curvature (a linear molecule's
bends); $10^{-3}$ is below the curvature of one bonded torsion, about
$k_\tau\lVert b\rVert^2 \approx 3\times10^{-3}$. The identity block on $T$
matches the simple start, so a gradient with a rigid-body component (an
external field) is treated as before. In redundant internals the seed is the
diagonal $1/\max(k_q, 10^{-3})$ over the bonded primitives. The
coordinate-only search `optimize_coordinates` (periodic cells) has no
molecule to build the model from: it starts from the identity, refuses an
explicit `lindh`, and uses the trust rule it is given.

**QM/MM** (`optimize_qmmm`). The search moves the free atoms $f$ (the QM real
atoms, plus the MM atoms `move_mm` frees) and holds the others fixed, so its
Hessian is the free-free block $H^{\rm L}_{ff}$ of the model of the whole
structure: QM real atoms and MM atoms, but not link atoms, whose positions
follow from their frontier and host atoms. A QM-only model would drop the
boundary bonds that tie a frontier atom to its fixed host and, under
`move_mm`, every MM curvature. $H^{\rm L}_{ff}$ only needs the terms that
contain a free atom. Each factor of a term is at most
$\rho_{\max} = \max(1, \max_{i<j}\rho_{ij})$, so every link of a term
above the cutoff has $\rho \ge 10^{-8}/\rho_{\max}^2$, and every atom of
such a term is within three such links of a free atom. The block is taken
from the model of that environment (`lindh_environment`), which gives the
same block to rounding without the $(3N)^2$ matrix of a large MM region.
Fixed atoms make the rigid motions of the free atoms physical, but the model
gives them curvature only through bonds to fixed atoms, so the seed keeps the
identity on them:

$$H_0 = Q_fV\,{\rm diag}\big(1/\max(w_i, 10^{-3})\big)V^{\mathsf T}Q_f^{\mathsf T} + T_fT_f^{\mathsf T},$$

with $T_f$ the rigid-body basis of the free atoms and $Q_f^{\mathsf T}H^{\rm L}_{ff}Q_f = VwV^{\mathsf T}$.
With every atom free this is the molecular seed, bit for bit. The
search is Cartesian; `initial_hessian = "simple"` restores the identity start.

Proof: `scripts/lindh_hessian_sympy.py` (L1 the force constants and their
distance derivatives in symbols, and the constants and tables read from the
Rust against Table 1; L2 the Wilson vectors against sympy gradients of the
primitive values; L3 symmetry, positive semi-definiteness and the six null
vectors; L4 a brute-force numpy sum over every term, which the Rust test
`lindh_hessian_matches_the_numpy_reference` matches through
`testdata/reference/lindh_model_hessian.json`).

**Measured defaults.** Steps to convergence at HF/STO-3G from RDKit/MMFF
starting geometries, default thresholds (`g_max` 4.5e-4, `g_rms` 3e-4,
`e_conv` 1e-6), "old" = `simple` + `fixed`, "new" = `lindh` + `ratio`,
geomeTRIC 1.1 in TRIC coordinates on ferric gradients (`ferric.ase`) as an
outside reference:

| molecule (atoms) | Cartesian old | Cartesian new | internal old | internal new | geomeTRIC |
|---|---|---|---|---|---|
| water (3) | 5 | 3 | 5 | 3 | 4 |
| ethanol (9) | 28 | 4 | 16 | 15 | 5 |
| butane (14) | 19 | 5 | 16 | 14 | 8 |
| paracetamol (20) | 60 | 33 | 30 | 22 | 29 |
| alanine dipeptide (22) | 118 | 39 | 40 | 31 | 22 |
| aspirin (21) | 58 | 16 | 32 | 15 | 21 |
| ibuprofen (33) | 80 | 19 | 67 | 30 | 16 |
| sulfamethoxazole (28) | 170 | 92 | 46 | 26 | 25 |
| total | 538 | 211 | 252 | 156 | 130 |
| pregabalin (28) | | 21 | | 51 | 17 |
| procaine (37) | | 13 | | 28 | 18 |
| lidocaine (39) | | 78 | | 49 | 42 |
| metoprolol (44) | | 43 | | 52 | 30 |
| total, 12 molecules | | 366 | | 336 | 237 |

`lindh` alone (Cartesian, `fixed`) gives 4, 5, 49, 30, 26 steps on ethanol,
butane, alanine dipeptide, aspirin and ibuprofen; `ratio` alone reproduces the
old trajectories exactly, since with the identity seed no step reaches the
trust boundary. The old Cartesian search stops 0.01 to 0.10 mHa above the
minimum the others reach (0.03 to 0.21 Å RMSD): with the identity seed a soft
mode meets the gradient thresholds before it is relaxed. The new searches end
within 3e-3 mHa of each other, except Cartesian new on sulfamethoxazole: its
long early steps turn the sulfonamide C–C–N–S torsion from 13° to −64°
(internals and geomeTRIC stop at +60°) and it ends in a rotamer 3.4 mHa lower.
Both end points are minima: re-optimizing either one with the other search
moves the energy by less than 0.02 mHa. 

**Default coordinates.** The default was chosen on three criteria, in order:
the search stays in the starting basin, it does not fail, and then the step
count. The four flexible molecules in the lower rows (RDKit ETKDG seed 7 +
MMFF starts) were added for this choice. The basin test is the largest
change of any heavy-atom dihedral between start and end point. Before the
runs the artifact hypothesis was written down: a broken internal
back-transformation would show as many rejected steps and back-transformation
warnings, as dihedrals that stay near the start only because steps are
suppressed, and as an end point that differs from geomeTRIC's and moves when
a Cartesian search restarts from it. A real coordinate effect gives the same
end point as geomeTRIC. The two cases are told apart by the end-point
comparison, not by the step count.

| molecule | largest dihedral change: Cartesian / internal / geomeTRIC | rejected steps: Cartesian / internal |
|---|---|---|
| paracetamol | 22° / 22° (opposite signs, same energy) / – | |
| sulfamethoxazole | 77° (C–C–N–S, +13° → −64°) / 46° (→ +59°) / – | |
| pregabalin | 15.6° / 14.6° / 16.1° | 1 / 37 |
| procaine | 4.7° / 4.7° / 4.7° | 0 / 14 |
| lidocaine | 80° / 80° / 80° (N–C–C–N, −107° → +172°) | 5 / 17 |
| metoprolol | 22.7° / 24.7° / 24.5° | 2 / 31 |

The other molecules change no dihedral by more than 17° in either search.
On the four new molecules both searches end at geomeTRIC's end point (largest
dihedral difference 1.9°, RMSD at most 0.027 Å), with no back-transformation
warnings, so the internal rejections are a cost, not a wrong result. Lidocaine
turns by 80° in all three optimizers: that is a downhill relaxation from the
MMFF start, not a barrier crossing. Only Cartesian sulfamethoxazole leaves the
starting basin. On step counts internals lead in total (336 against 366) but
take more steps on six of the twelve molecules (one tie), and they reject 14 to 37
steps per new molecule against 0 to 5. The data do not separate the two on
the first criterion beyond the one sulfamethoxazole case, and on the second
they favour Cartesians, so the default stays Cartesian; `coordinates =
"internal"` is the choice when a soft torsion near a barrier matters.

Pregabalin at PBE/def2-SVP (RI-J/RI-K, f64), same start: Cartesian 34 steps
(2 rejected), internal 61 (24 rejected), no back-transformation warnings.
Both turn the C–C–C–O torsion from +100° to +128/129° and end 1.2° and
0.013 Å RMSD apart; the internal end point is 0.17 mHa higher, within the
loose default thresholds.

Sulfamethoxazole at PBE/def2-SVP (RI-J/RI-K, f64), same start: Cartesian
old does not converge in 150 steps; Cartesian new converges in 47 steps,
internal new in 37, internal old in 35. The new searches end within 0.04 mHa
of each other. A further 150 internal steps from each end point at `g_max`
1.5e-5 (none reaches that threshold) move them by less
than 0.05 mHa, and they stay 0.02 mHa apart. The analytic Hessian at the
internal end point has one mode at 22.7i cm⁻¹ (NH2 and methyl hydrogens);
re-optimizing from ±0.15 Å along it lowers the energy by at most 0.05 mHa, so
it is a soft region of the same basin at the loose default thresholds, not a
saddle above a lower one.
The unconverged old search ends 1.39 mHa lower and keeps falling in the same
re-optimization, to 1.50 mHa below the new end points: its 150 short steps carry it out of
the starting basin into a lower one. geomeTRIC on the same surface (checked:
its calculator and `run_optimize` agree to 0.01 mHa at one geometry) takes 53
steps and ends 2.3 mHa below the new end points, in a third basin. At this
level the molecule has several rotamers within a few mHa, and which one a local
search reaches depends on its early steps. An end-point gap of this size between the
old Cartesian and internal searches is a different basin, not a convergence
criterion.

## Kohn-Sham Z-vector for post-SCF corrections

`zvector_ks.rs` supplies the orbital-relaxation term of the nuclear gradient
of a quantity $Q(R, D)$ computed after a converged Kohn-Sham SCF (its
production caller is the MBD@rsSCS gradient). With
$V = \partial Q/\partial D_{\rm total}$ (AO, symmetric), the total derivative is

$$ \frac{dQ}{dx} = \underbrace{\frac{\partial Q}{\partial x}\Big|_D + \operatorname{tr}\big[V\,\delta D^{\rm fix}\big]}_{\text{caller}} + \sum Z\cdot\frac{\partial F_{\rm vo}}{\partial x}\Big|_{\rm fix} . $$

The caller's part holds the occupied orbitals on the fixed path, on which they
follow their basis functions and are Löwdin re-orthonormalized among
themselves: $\delta C_o = -\tfrac12 C_oS^x_{oo}$, and the virtual space follows
as the $S(x)$-orthogonal complement. The module supplies the second term from
one linear solve. Orbitals rotate as $C \leftarrow C\,U(\kappa)$ with
$\kappa_{ai}$ stored (book notation §6), and the Brillouin condition holds on
every geometry, so $\partial_xF_{ai}|_{\rm fix} + (H\kappa^x)_{ai} = 0$ with
$H_{ai,bj} = \partial F_{ai}/\partial\kappa_{bj}$; with $dQ = c\,\kappa\cdot V_{vo}$
($c$ the occupation factor below) and $H$ symmetric,
$c\,V\cdot\kappa^x = -c\,V\cdot H^{-1}\partial_xF = Z\cdot\partial_xF$ with
$HZ = -c\,V_{vo}$.

**Closed shell (RKS).** $D = 2C_oC_o^{\mathsf T}$, a rotation changes it by
$\delta D = 2(C_v\kappa C_o^{\mathsf T} + C_o\kappa^{\mathsf T}C_v^{\mathsf T})$, so
$dE = 4\sum\kappa_{ai}F_{ai}$, $dQ = 4\sum\kappa_{ai}V_{ai}$, and

$$ (Hx)_{ai} = (F_{vv}x - xF_{oo})_{ai} + \big(C_v^{\mathsf T}\,\delta F[\delta D(x)]\,C_o\big)_{ai}, \qquad HZ = -4V_{vo},\quad V_{vo} = C_v^{\mathsf T}VC_o, $$

$\delta F[\delta D] = \delta J - \tfrac12(c_{\rm sr}\delta K_{\rm sr} + c_{\rm lr}\delta K_{\rm lr}) + \delta V_{xc}$,
with $\delta V_{xc}$ the spin kernel applied to $\delta D_\alpha = \delta D_\beta = \tfrac12\delta D$.
The fixed path gives $\delta D^{\rm fix} = -\tfrac12 DS^xD$.

**The contraction without derivative integrals.** $F_{ai} = \tfrac14\partial E/\partial\kappa_{ai}$,
and $\partial E/\partial x$ along the fixed path is the SCF gradient expression
$g(D, W)$ with $W = \tfrac12DF[D]D$ (any projector $D$:
$\operatorname{tr}(F\,\delta D^{\rm fix}) = -\operatorname{tr}(S^xW)$). Exchanging the order
of the two derivatives,

$$ \sum_{ai}Z_{ai}\frac{\partial F_{ai}}{\partial x} = \frac14\frac{d}{dt}\,g\big(D(t),\,W + t\dot W\big)\Big|_{t=0},\qquad \dot W = \tfrac12\big(\delta D_ZFD + DF\delta D_Z + D\,\delta F[\delta D_Z]\,D\big), $$

$\delta D_Z = \delta D(Z)$. For $t \ne 0$ the fixed path of the rotated
orbitals differs from the rotated fixed path by an occupied-virtual rotation
of order $t$, whose energy gradient is itself of
order $t$, so the $t$-derivative is unchanged. The code evaluates
$\tfrac14[g(+t) - g(-t)]/(2t)$ with `ks_gradient_closed_for_density`, the same
routine, $J/K$ route and grid response as the SCF gradient, at

$$ D(t) = 2\,C_o(t)C_o(t)^{\mathsf T},\qquad C_o(t) = (C_o + tC_vZ)\,(\mathbf 1 + t^2Z^{\mathsf T}Z)^{-1/2}, $$

an exact projector (positive semidefinite, which the fitted-exchange gradient
needs; $D \pm t\,\delta D_Z$ is not), with $\dot D(0) = \delta D_Z$, and
$t = 10^{-4}/\max\lvert\delta D_Z\rvert$ (`T_SCALE`). $g$ is linear in $W$, so
$W + t\dot W$ is exact to the order needed. The central difference is not
exact for the one- and two-electron parts: $D(t)$ has an odd $t^3$ term,
$-t^3(C_vZZ^{\mathsf T}ZC_o^{\mathsf T} + {\rm h.c.})$ from the
$(\mathbf 1 + t^2Z^{\mathsf T}Z)^{-1/2}$ factor, and its even $t^2$ term enters
the bilinear two-electron part, so every part of the contraction carries an
$O(t^2)$ error, of relative size $\sim 10^{-8}$ at this step.

**Unrestricted (UKS).** Unit occupations: $D_\sigma = C_{\sigma o}C_{\sigma o}^{\mathsf T}$,
$\delta D_\sigma = C_{\sigma v}x^\sigma C_{\sigma o}^{\mathsf T} + {\rm h.c.}$, so
$dE = 2\sum_\sigma\kappa^\sigma\cdot F^\sigma_{vo}$ and, for $Q(D_\alpha + D_\beta)$,

$$ (Hx)^\sigma = F^\sigma_{vv}x^\sigma - x^\sigma F^\sigma_{oo} + C_{\sigma v}^{\mathsf T}\,\delta F_\sigma[\delta D_\alpha, \delta D_\beta]\,C_{\sigma o},\qquad HZ^\sigma = -2V^\sigma_{vo}, $$

$\delta F_\sigma = \delta J[\delta D_\alpha + \delta D_\beta] - (c_{\rm sr}\delta K_{\rm sr} + c_{\rm lr}\delta K_{\rm lr})[\delta D_\sigma] + \delta V^\sigma_{xc}$;
$\delta D^{\rm fix}_\sigma = -D_\sigma S^xD_\sigma$, $W = \sum_\sigma D_\sigma F_\sigma D_\sigma$,
$\dot W = \sum_\sigma(\delta D_\sigma F_\sigma D_\sigma + D_\sigma F_\sigma\delta D_\sigma + D_\sigma\delta F_\sigma D_\sigma)$,
and the contraction is $\tfrac12\,d/dt$ of `ks_gradient_uks_for_density` at the
rotated projectors of each spin (occupation 1). For $\alpha = \beta$ the spin
blocks are half the closed-shell $Z$ and the relaxation is the RKS one.

**Restricted open shell (ROKS).** One set $C = (C_c|C_o|C_v)$ (closed, open,
virtual), $P_x = C_xC_x^{\mathsf T}$, $D_\alpha = P_c + P_o$, $D_\beta = P_c$. The
density-changing rotations are $\kappa_{vc}$ (both spins), $\kappa_{vo}$
($\alpha$ only) and $\kappa_{oc}$ ($\beta$ only), with
$dE = 2(\kappa_{vc}\cdot g_{vc} + \kappa_{vo}\cdot g_{vo} + \kappa_{oc}\cdot g_{oc})$,
$g_{vc} = (F_\alpha + F_\beta)_{vc}$, $g_{vo} = F_{\alpha,vo}$, $g_{oc} = F_{\beta,oc}$
(MO spin Focks). The Hessian, written with the stationarity conditions so
that it is symmetric ($F^+ = F_\alpha + F_\beta$, $G_{vc} = \tfrac12(F_\beta - F_\alpha)_{vc}$):

$$ \begin{aligned}
(Hx)_{vc} &= F^+_{vv}x_{vc} - x_{vc}F^+_{cc} + F_{\beta,vo}x_{oc} - x_{vo}F_{\alpha,oc} + C_v^{\mathsf T}(\delta F_\alpha + \delta F_\beta)C_c,\\
(Hx)_{vo} &= F_{\alpha,vv}x_{vo} - x_{vo}F_{\alpha,oo} + G_{vc}x_{oc}^{\mathsf T} - x_{vc}F_{\alpha,oc}^{\mathsf T} + C_v^{\mathsf T}\delta F_\alpha C_o,\\
(Hx)_{oc} &= F_{\beta,oo}x_{oc} - x_{oc}F_{\beta,cc} + F_{\beta,vo}^{\mathsf T}x_{vc} + x_{vo}^{\mathsf T}G_{vc} + C_o^{\mathsf T}\delta F_\beta C_c,
\end{aligned} $$

with $\delta D_\alpha = C_vx_{vc}C_c^{\mathsf T} + C_vx_{vo}C_o^{\mathsf T} + {\rm h.c.}$ and
$\delta D_\beta = C_vx_{vc}C_c^{\mathsf T} + C_ox_{oc}C_c^{\mathsf T} + {\rm h.c.}$
(`RoksBlocks::{densities,product}`). Since $\kappa_{vc}$ moves both spins of
$D_{\rm total}$,

$$ HZ = -(4V_{vc},\ 2V_{vo},\ 2V_{oc}),\qquad V_{xy} = C_x^{\mathsf T}VC_y . $$

The fixed path re-orthonormalizes closed and open orbitals as one set, so
$\delta D^{\rm fix}_\alpha = -D_\alpha S^xD_\alpha$ and
$\delta D^{\rm fix}_\beta = -P_cS^xP_c - \tfrac12(P_oS^xP_c + P_cS^xP_o)$; the
closed-open cross term is a real part of the caller's term because $V_{co} \ne 0$.
For the energy the same path gives

$$ W = D_\alpha F_\alpha D_\alpha + P_cF_\beta P_c + \tfrac12(P_cF_\beta P_o + P_oF_\beta P_c), $$

whose cross term vanishes at convergence ($P_cF_\beta P_o = C_cF_{\beta,co}C_o^{\mathsf T}$, $F_{\beta,oc} = 0$)
but whose derivative does not: $\dot W$ differentiates every term with
$\delta P_c = \delta D_\beta$, $\delta P_o = \delta D_\alpha - \delta D_\beta$
(`roks_energy_weighted_density_derivative`). The contraction is
$\tfrac12[g(+t) - g(-t)]/(2t)$ of `ks_gradient_roks_for_density` at the orbitals
$C\,(\mathbf 1 - t\kappa/2)^{-1}(\mathbf 1 + t\kappa/2)$, $\kappa$ the antisymmetric
generator of the three $Z$ blocks.

**The solve.** Preconditioned conjugate gradient with the diagonal gaps
($F_{aa} - F_{ii}$ per spin; for ROKS the diagonals $F^+_{vv} - F^+_{cc}$,
$F_{\alpha,vv} - F_{\alpha,oo}$ and $F_{\beta,oo} - F_{\beta,cc}$ of the three blocks), started from
$\mathrm{rhs}/\mathrm{gap}$. It stops when $\max\lvert r\rvert \le 10^{-10}\max\lvert\mathrm{rhs}\rvert$
and fails after 200 iterations. A step with
$p\cdot Hp \le 10^{-12}\,p\cdot Mp$ is accepted as the rounding floor if
$\max\lvert r\rvert \le 10^{-8}\max\lvert\mathrm{rhs}\rvert$ and is an error
otherwise (negative curvature: not a minimum; zero: stalled). A non-positive
gap is refused. For the UKS and ROKS solves of a linear molecule in a state
that breaks the cylindrical symmetry, the rotation about the axis is an
occupied-virtual null mode $u$ of $H$: when its gap-metric Rayleigh quotient is
below $0.05$ and $\lvert\mathrm{rhs}\cdot u\rvert/\lVert\mathrm{rhs}\rVert \le 10^{-3}$,
$u$ is projected out of the right-hand side and every iterate ($Z_u = 0$); a
larger overlap is refused. The closed-shell solve has no such projection.
The Hessian uses exact four-centre $J$ and $K$ even when the SCF fitted them;
the contraction uses the SCF's own route.

Proof: `scripts/ks_zvector_sympy.py` (Z0 the occupation factors in symbols and
the anchor $Q = E_{\rm SCF}$, where $V_{vo} = F_{vo} = 0$ gives $Z = 0$ and the
caller's part alone equals the finite difference of the energy; Z1 the RKS
relaxation, with $\partial_xF_{ai}$ taken along the explicit fixed path, plus the
caller's part against a Richardson difference of $Q$ with the SCF re-solved, with
and without an XC kernel; Z2 the coded central-difference contraction against
the same $\sum Z\cdot\partial_xF$, and its $O(t^2)$ error with and without XC;
Z3 the UKS solve and contraction against the finite difference, and the
$\alpha = \beta$ limit; Z4 the ROKS three-block solve, the transcribed product's
symmetry and the contraction against the finite difference, with the
closed-open cross term of the fixed path as a negative control).

## Density-fitted gradient (RI-J) on a pair-screened tensor

**Energy.** With auxiliary metric $V_{PQ}=(P|Q)$ and three-index integrals $(P|\mu\nu)$, RI-J is
$$E_J=\tfrac12\,d^{\mathsf T}V^{-1}d,\qquad d_P=\sum_{\mu\nu}(P|\mu\nu)\,D_{\mu\nu}.$$
A J-only SCF stores $(P|\mu\nu)$ only on a set $K$ of AO pairs that a Cauchy-Schwarz test keeps
($Q_{\mu\nu}\,q_{\max}>\tau$, $Q_{\mu\nu}=\sqrt{(\mu\nu|\mu\nu)}$, $q_{\max}=\max_P\sqrt{(P|P)}$,
$\tau=$ `integral_thresh`). Its energy is
$$\tilde E_J=\tfrac12\,\tilde d^{\mathsf T}V^{-1}\tilde d,\qquad
\tilde d_P=\sum_{(\mu\nu)\in K}(P|\mu\nu)\,D_{\mu\nu}=d_P(\tilde D),$$
where $\tilde D_{\mu\nu}=D_{\mu\nu}$ for $(\mu\nu)\in K$ and $0$ otherwise (both triangles). The last
equality is the whole point: $\tilde d$ is the unscreened $d$ evaluated at the masked density.

**Gradient.** Treat $K$ as fixed (a step function of the geometry, constant almost everywhere) and
write $c=V^{-1}d$ for the fit coefficients. For a nuclear coordinate $x$, with $D$ held fixed
(the SCF is variational, so the orbital-response terms are collected in the energy-weighted overlap
term, which does not involve $E_J$'s tensor),
$$\frac{\partial\tilde E_J}{\partial x}
=\sum_P\tilde c_P\sum_{(\mu\nu)\in K}\frac{\partial(P|\mu\nu)}{\partial x}D_{\mu\nu}
-\tfrac12\,\tilde c^{\mathsf T}\frac{\partial V}{\partial x}\tilde c,\qquad \tilde c=V^{-1}\tilde d,$$
using $\partial V^{-1}=-V^{-1}(\partial V)V^{-1}$. The first sum runs over $K$ only. Because the
three-centre derivative pass is linear in the density it contracts, restricting its pair sum to $K$
is the same as contracting $\tilde D$ over all pairs. Hence
$$\boxed{\;\nabla\tilde E_J=g_J(\tilde D)\;}$$
where $g_J(\cdot)$ is the ordinary (unscreened) RI-J gradient as a function of the density it is
handed, with $d$, $c$ and the derivative contraction all formed from $\tilde D$. The implementation
therefore (i) forms $\tilde d$ from the stored columns, an off-diagonal column standing for
$(\mu\nu)$ and $(\nu\mu)$ with weight $D_{\mu\nu}+D_{\nu\mu}$, and (ii) hands the derivative pass
$\tilde D$. Dropped pairs contribute zero to both, consistent with the screened energy.
Nothing else changes: the metric terms use $V$ and $\tilde c$ only.

**Difference from the unscreened gradient.** $g_J$ is a quadratic form $G(D,D)$, so
$$g_J(\tilde D)-g_J(D)=-G(D_{\mathrm{drop}},D)-G(D,D_{\mathrm{drop}})+G(D_{\mathrm{drop}},D_{\mathrm{drop}}),
\quad D_{\mathrm{drop}}=D-\tilde D,$$
which is $O(\|D_{\mathrm{drop}}\|/\|D\|)\,\|g_J\|$. On the fitted density the Schwarz bound gives,
for every $P$,
$$|d_P-\tilde d_P|\le\sum_{\text{dropped}}|(P|\mu\nu)|\,|D_{\mu\nu}|
\le\tau\sum_{\text{dropped}}|D_{\mu\nu}|\,\frac{Q_P}{q_{\max}}\le\tau\sum_{\text{dropped}}|D_{\mu\nu}|,$$
and $\tilde c-c=-V^{-1}(d-\tilde d)$. At $\tau\le0$ nothing is dropped, $\tilde D=D$ and the two
gradients are the same bits.

**Cross-check.** Independent of the derivation above, `df_gradient_screened_j_handoff.rs` checks (a)
$g_{\text{reuse}}-g_{\text{recompute}}=g_J(\tilde D)-g_J(D)$ with the right side from the public
recomputing path and an independently built pair screen, (b) the Schwarz bound on $d_P$, and (c) a
central finite difference of the SCF energy run with the same screening: the screened-tensor
gradient is the derivative of that energy; the recompute gradient is not.
