# Random Phase Approximation (RPA) Theory

This module implements the direct Random Phase Approximation (dRPA) correlation energy using the Projective Dielectric Eigenpotential (PDEP) approach.

## RPA Correlation Energy

The RPA correlation energy is commonly evaluated using the Adiabatic-Connection Fluctuation-Dissipation (ACFD) theorem via an imaginary frequency integration:

$$ E_c^{\text{RPA}} = \frac{1}{2\pi} \int_0^\infty \text{Tr}\left[ \ln(\mathbf{1} - \mathbf{\chi}_0(i\omega) \mathbf{v}) + \mathbf{\chi}_0(i\omega) \mathbf{v} \right] d\omega $$

where $\mathbf{\chi}_0(i\omega)$ is the independent-particle polarizability and $\mathbf{v}$ is the Coulomb interaction. The term $\mathbf{\epsilon}(i\omega) = \mathbf{1} - \mathbf{\chi}_0(i\omega) \mathbf{v}$ is the interacting dielectric matrix.

## The PDEP Eigenpotential Approach

Instead of explicitly constructing and diagonalizing the large dense dielectric matrix $\mathbf{\epsilon}(i\omega)$ at each frequency point, the PDEP approach uses the rapid decay of the dielectric operator's spectrum.

1. **Static Dielectric Eigenpotentials:** The static ($\omega=0$) dielectric operator $\mathbf{\epsilon}(0)$ is diagonalized iteratively (block Lanczos by default, `Eigensolver::Lanczos`; Davidson is selectable) to find the dominant eigenpotentials (often called plasmon density eigenpotentials) $|V_\alpha\rangle$:
   $$ \mathbf{\epsilon}(0) |V_\alpha\rangle = \lambda_\alpha(0) |V_\alpha\rangle $$
   Because the eigenvalues decay steeply, only a small number of eigenpotentials with eigenvalues $\lambda_\alpha(0) > \epsilon_{\text{trunc}}$ need to be retained. They form a compressed, system-adapted basis.

2. **Sternheimer Linear Response:** The action of the dielectric matrix on a trial potential is written in the Sternheimer form, which avoids an explicit sum over virtual states in a plane-wave or real-space implementation. The Sternheimer equation for each occupied orbital $i$ is the Sternheimer linear response equation for each occupied orbital $i$:
   $$ (\hat{H}_0 - \epsilon_i \mathbf{1} + i\omega \mathbf{1}) |\Delta \Psi_i(i\omega, V)\rangle = - P_{\text{virt}} \hat{V} |\Psi_i\rangle $$
   where $P_{\text{virt}}$ is the projector onto the virtual space.

   ferric's RI-MO implementation does sum over every occupied-virtual pair: `sternheimer::chi_from_trial_potential` (`crates/ferric-rpa/src/sternheimer.rs:330`) evaluates $\chi_0$ as $-2\sum_{ia} e_{ia}/(\omega^2+e_{ia}^2)\,|\sum_P V_P B^P_{ia}|^2$. The Sternheimer form is the formulation; PDEP compresses the result of the sum.

3. **Frequency-Dependent Eigenvalues:** Assuming the static eigenpotential basis $|V_\alpha\rangle$ is adequate for all frequencies, the frequency-dependent eigenvalues $\lambda_\alpha(i\omega)$ are evaluated by projecting the frequency-dependent dielectric operator onto the fixed static PDEP basis:
   $$ \lambda_\alpha(i\omega) = \langle V_\alpha | \mathbf{\epsilon}(i\omega) | V_\alpha \rangle $$

4. **Energy Integration:** The correlation energy trace is then evaluated via numerical quadrature (e.g., MiniMax or mapped Gauss-Legendre) over the imaginary frequency grid:
   $$ E_c^{\text{RPA}} = \frac{1}{2\pi} \sum_k w_k \sum_\alpha \left[ \ln(\lambda_\alpha(i\omega_k)) + (1 - \lambda_\alpha(i\omega_k)) \right] $$

This formulation avoids the $\mathcal{O}(N^6)$ cost and $\mathcal{O}(N^4)$ memory associated with forming and manipulating $\mathbf{\chi}_0(i\omega)$ across all frequencies.

## Open-Shell (Spin-Summed) Dielectric

`run_u_pdep_rpa` (`crates/ferric-rpa/src/lib.rs`) takes a UHF reference, or a ROHF reference semi-canonicalized per spin. Each spin $\sigma$ has its own RI tensor $B^P_{i_\sigma a_\sigma}$ and gaps $e_{ia}^\sigma = \epsilon_a^\sigma - \epsilon_i^\sigma$ in one shared aux basis, and the symmetrized dielectric is

$$ \tilde{\mathbf{\epsilon}}_U(i\omega) = \mathbf{1} + \mathbf{\Pi}_\alpha(i\omega) + \mathbf{\Pi}_\beta(i\omega), \qquad \Pi^\sigma_{PQ}(i\omega) = 2 \sum_{ia} \frac{e^\sigma_{ia}}{\omega^2 + (e^\sigma_{ia})^2} B^P_{i_\sigma a_\sigma} B^Q_{i_\sigma a_\sigma}. $$

The closed-shell $\mathbf{\Pi}$ carries the prefactor 4; with identical spin channels $\mathbf{\Pi}_\alpha + \mathbf{\Pi}_\beta$ equals it, so a UHF singlet reproduces `run_pdep_rpa`. The PDEP eigensolve, truncation, frequency projection and trace-log integration are those of the closed-shell path applied to $\tilde{\mathbf{\epsilon}}_U$; the matvec $\tilde{\mathbf{\epsilon}}_U \mathbf{V} = \mathbf{V} + \mathbf{\Pi}_\alpha \mathbf{V} + \mathbf{\Pi}_\beta \mathbf{V}$ is two closed-shell-shaped applications (`sternheimer::dielectric_apply_unrestricted`). The Laplace $\chi_0$ backend builds one Laplace quadrature per spin on that spin's gap range. `run_diagnostics` assembles $\tilde{\mathbf{\epsilon}}_U(i\omega_k)$ densely at every quadrature node (`diagnostics::u_ri_drpa_energy`).

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
