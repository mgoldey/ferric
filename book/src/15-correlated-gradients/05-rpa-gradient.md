# RI-dRPA gradient {#rpa-gradient}

<!-- status: ready; registry: rpa-gradient -->

The direct RPA correlation energy sums ring diagrams to all orders: an
electron–hole pair is created, propagates, and creates the next pair through
the Coulomb interaction, any number of times. Like MP2 it is evaluated on
Hartree–Fock orbitals that do not minimize it, so its gradient needs the
same orbital response. Unlike MP2 it is an integral over imaginary frequency
of a log-determinant, so the densities that feed the response have to be
collected frequency by frequency.

The analytic dRPA gradient differentiates the energy of section @sec:pdep at full
rank (`trunc_thresh = 0`) on a fixed $n$-point imaginary-frequency grid
$(u_k, w_k)$ of `crates/ferric-rpa/src/quadrature.rs`. Every scheme there
(Gauss-Legendre, MiniMax, ChebyshevTan) has nodes and weights that depend
only on $(n, u_0)$, never on the molecule, so the grid has no geometry
derivative, and the result is the exact derivative of the $n$-point sum, not
of the converged integral. At $n = 6$ it must match a finite difference of
the 6-point energy, which differs from the 40-point one by
$3\times10^{-4}$ Ha/Bohr on distorted H2O. The route is the
frequency-integrated relaxed density of Burow, Bates, Furche and Eshuis. The
ring-CCD Lagrangian of Rekkedal et al. gives the same gradient for the
direct-ring energy and is not used. The frequency loop produces a three-index density $\Gamma$ and
occupied and virtual densities $P$; from there the RI-MP2 machinery of
section @sec:rimp2-gradient is reused unchanged.

| Symbol | Meaning | Code name |
|---|---|---|
| $u_k, w_k$ | frequency nodes and weights | `omega`, `wk` |
| $g_k(e)$ | $4e/(u_k^2 + e^2)$ | `g` |
| $\Pi_k$ | $B\,g_k(\Delta)\,B^{\mathsf T}$ | `pi` |
| $Q_k$ | $(\mathbf 1 + \Pi_k)^{-1} - \mathbf 1$ | `q` |
| $\Gamma^P_{ia}$ | three-index density | stored as $x = \Gamma/4$, `x_ov` |
| $P_{ij}, P_{ab}$ | orbital-energy densities | stored as $P/2$, `p_oo`, `p_vv` |
| $g^{[1]}(x, y)$ | divided difference of $g$ | `g_divided_difference` |

| Entry point | Location |
|---|---|
| Rust | `ferric-rpa/src/gradient_analytic.rs::{rpa_analytic_gradient,rpa_analytic_gradient_warm,rpa_analytic_gradient_on}` |
| CLI | `method.kind = "pdep-rpa"`, `task = "optimize"`, `[rpa] gradient = "analytic"` |
| Python | `run_optimize(..., method="pdep-rpa", rpa_gradient="analytic")`; `run_frequencies(..., method="pdep-rpa")` |

## The densities {#rpa-gradient-densities}

The energy on the grid is

$$ E_c = \frac{1}{2\pi}\sum_k w_k\,S(\Pi_k), \qquad S(\Pi) = \ln\det(\mathbf 1 + \Pi) - \operatorname{tr}\Pi . $$

$\Pi_k$ is the density response in the auxiliary basis at frequency $iu_k$:
a sum over occupied–virtual pairs $ia$ of the outer product of the fitted
transition density $B^P_{ia}$ with itself, weighted by $g_k(e_{ia})$, the
closed-shell response of one transition of energy
$e_{ia} = \varepsilon_a - \varepsilon_i$.

### The energy as a function of $B$ and $F$ {#rpa-gradient-energy-b-f}

The fitted factor is $B^P_{ia} = \sum_QM_{PQ}(Q|ia)$ with
$M^{\mathsf T}M = V^{-1}$, $V_{PQ} = (P|Q)$; the code's $M$ is the Cholesky
$L^{-1}$ of the Coulomb metric. $\Pi = \tilde\epsilon - \mathbf 1$ is the
closed-shell dielectric of section @sec:pdep minus the identity. At full rank
$S$ is basis-invariant, so the eigenvalue and log-det routes of
`crates/ferric-rpa/src/energy.rs` both evaluate this expression.

Two invariances decide what the gradient has to differentiate. First,
$\Delta$ is the occupied-virtual orbital-energy-difference operator

$$ \Delta_{ia,jb} = F_{ab}\delta_{ij} - F_{ij}\delta_{ab}, $$

which is $\operatorname{diag}(e_{ia})$ in canonical orbitals. Write $E_c$
with the Fock blocks instead of the orbital energies, $E_c(B, F_{oo}, F_{vv})$.
A rotation $O$ among the occupied orbitals sends $B \to O^{\mathsf T}B$ (on
the occupied index) and $F_{oo} \to O^{\mathsf T}F_{oo}O$, and leaves $\Pi$
unchanged; the same holds among the virtual orbitals. So the
orbital-energy derivative is taken with respect to the full symmetric Fock
blocks, and the occupied-occupied and virtual-virtual rotations need no
response equation. Non-canonical dependence is handled by never leaving this
invariant form.

Second, $E_c$ depends on $B$ only through the fitted integrals
$B^{\mathsf T}B$. By Sylvester's determinant identity,

$$ \ln\det\big(\mathbf 1 + BgB^{\mathsf T}\big) = \ln\det\big(\mathbf 1 + g^{1/2}B^{\mathsf T}Bg^{1/2}\big), $$

and $\operatorname{tr}(BgB^{\mathsf T}) = \operatorname{tr}(g^{1/2}B^{\mathsf T}Bg^{1/2})$.
So any factor $M$ of $V^{-1}$ gives the same energy and the same gradient.

Differentiate $S$. The first variation of a log-determinant is
$\delta\ln\det(\mathbf 1 + \Pi) = \operatorname{tr}[(\mathbf 1 + \Pi)^{-1}\delta\Pi]$,
and $\delta\operatorname{tr}\Pi = \operatorname{tr}\delta\Pi$. Subtract:

$$ \delta S = \operatorname{tr}\big[Q_k\,\delta\Pi_k\big], \qquad Q_k = (\mathbf 1 + \Pi_k)^{-1} - \mathbf 1 . $$

$\Pi_k = B\,g_k\,B^{\mathsf T}$ moves when $B$ moves and when the orbital
energies in $g_k$ move. Take the two in turn.

**Through $B$.** $\delta\Pi = \delta B\,g\,B^{\mathsf T} + B\,g\,\delta B^{\mathsf T}$.
$Q$ is symmetric, so both terms give the same trace, and
$\partial S/\partial B^P_{ia} = 2\,g(e_{ia})\,(QB)^P_{ia}$. Multiply by the
prefactor $w_k/2\pi$ and sum over nodes:

$$ \Gamma^P_{ia} = \frac{\partial E_c}{\partial B^P_{ia}} = \frac1\pi\sum_kw_k\,g_k(e_{ia})\,(Q_kB)^P_{ia} . \label{eq:rpa-grad-densities} $$

**Through the orbital energies.** With $M_k = B^{\mathsf T}Q_kB$:

$$ P_{ab} = \frac{1}{2\pi}\sum_kw_k\sum_i(M_k)_{ia,ib}\,g_k^{[1]}(e_{ia}, e_{ib}), \qquad P_{ij} = -\frac{1}{2\pi}\sum_kw_k\sum_a(M_k)_{ia,ja}\,g_k^{[1]}(e_{ia}, e_{ja}), \label{eq:rpa-gradient-2} $$

$$ g^{[1]}(x, y) = \frac{g(x) - g(y)}{x - y} = \frac{4(u^2 - xy)}{(u^2 + x^2)(u^2 + y^2)} . $$

The diagonal is the familiar chain rule: $\delta\Pi = B\,\delta g\,B^{\mathsf T}$
with $\delta g_{ia} = g'(e_{ia})\,\delta e_{ia}$ gives
$\partial E_c/\partial\varepsilon_a = P_{aa}$ and
$\partial E_c/\partial\varepsilon_i = P_{ii}$, the minus sign of $P_{ij}$
coming from $\partial e_{ia}/\partial\varepsilon_i = -1$. The off-diagonal
elements are derivatives with respect to the off-diagonal Fock elements:
$P$ is the derivative with respect to the full Fock blocks $F_{ab}$ and
$F_{ij}$, not only the orbital energies. In a non-canonical basis $g$ becomes
a function of the Fock blocks, and the first derivative of a matrix
function, written in its eigenbasis, is the divided difference of the scalar
function (the Daleckii–Krein formula). This is why the
occupied–occupied and virtual–virtual rotations need no response equation.
At $x = y$ the divided difference is $g'(x) = 4(u^2 - x^2)/(u^2 + x^2)^2$,
and the closed form above has no cancellation there, so degenerate orbitals
need no special case.

**Second-order limit.** Expand $S$ in the eigenvalues $\lambda$ of $\Pi$:
$\ln(1 + \lambda) - \lambda = -\tfrac12\lambda^2 + O(\lambda^3)$. Replacing $S$
by $-\tfrac12\operatorname{tr}\Pi^2$ makes $Q_k = -\Pi_k$. As $n \to \infty$
the frequency sum becomes an integral, and

$$ \int_0^\infty\frac{xy}{(u^2 + x^2)(u^2 + y^2)}\,du = \frac{\pi}{2(x + y)} $$

turns the product of two $g$ factors into an energy denominator. The result
is the direct-ring MP2 energy

$$ E^{(2)}_{\mathrm{dr}} = -2\sum_{iajb}\frac{(ia|jb)^2}{\Delta_{ij}^{ab}}, $$

twice the opposite-spin MP2 energy, which is SCS-MP2 at
$(c_{\mathrm{OS}}, c_{\mathrm{SS}}) = (2, 0)$. With
$t_{ij}^{ab} = -(ia|jb)/\Delta_{ij}^{ab}$ the densities become

$$ \Gamma^P_{ia} \to 8\sum_{jb}t_{ij}^{ab}B^P_{jb}, \qquad P_{ij} \to -4\sum_{kab}t_{ik}^{ab}t_{jk}^{ab}, \qquad P_{ab} \to 4\sum_{ijc}t_{ij}^{ac}t_{ij}^{bc}, $$

the $(2, 0)$-scaled SCS-MP2 densities of section @sec:rimp2-gradient in this
section's normalization, $\Gamma = 4x$ and
$P = P^{\mathrm{MP2}} + P^{\mathrm{MP2}\,\mathsf T}$. That limit is the
exactness anchor against independent code.

### The frequency loop

`ferric-rpa/src/gradient_analytic.rs::frequency_response` diagonalizes
$\Pi_k$ once per node and reads both $S$ and $Q$ off the eigenvalues. For an
eigenvalue $\lambda$ of $\Pi_k$, $S$ contributes $\ln(1 + \lambda) - \lambda$
and $Q$ has the eigenvalue $(1 + \lambda)^{-1} - 1 = -\lambda/(1 + \lambda)$:

<!-- excerpt: ferric-rpa/src/gradient_analytic.rs::frequency_response -->
```rust
let s = lam.iter().map(|&l| trace_log_summand_term(1.0 + l)).sum();
let d = lam.mapv(|l| -l / (1.0 + l));
```

| Symbol | Code |
|---|---|
| eigenvalues of $\Pi_k$ | `lam` |
| $S(\Pi_k)$ | `s` |
| eigenvalues of $Q_k$, $(1 + \lambda)^{-1} - 1$ | `d` |

An eigenvalue of $\mathbf 1 + \Pi_k$ at or below zero is an error: the
log-determinant is then undefined.
`ferric-rpa/src/gradient_analytic.rs::rpa_ri_densities` accumulates the
results with the scale factors that convert to the RI-MP2 conventions. This
is its node loop with the $P_{oo}$ call elided and the $P_{vv}$ call joined
onto one line:

```rust
let g = gaps.mapv(|e| 4.0 * e / (omega * omega + e * e));   // g_k(e_ia)
let (s_k, q_b) = frequency_response(b_ov, &g, order)?;        // S(Π_k), Q_k B
out.e_c += wk * s_k / (2.0 * std::f64::consts::PI);           // E_c += w_k S / 2π
let xs = wk / (4.0 * std::f64::consts::PI);                   // (1/π)·(1/4): x = Γ/4
out.x_ov.zip_mut_with(&(&q_b * &g), |x, v| *x += xs * v);
let ps = wk / (4.0 * std::f64::consts::PI);                   // (1/2π)·(1/2): P/2
accumulate_p_vv(&mut out.p_vv, b_ov.view(), q_b.view(), &gaps, omega, ps, no, nv);
// ... (accumulate_p_oo with -ps)
```

> **Verified equations @eq:rpa-grad-densities and @eq:rpa-gradient-2** · the densities and the gradient
>
> On a toy closed-shell system with $x$-dependent $S$, $h$ and exactly
> factorized integrals, $\Gamma$ and $P$ are the derivatives of
> $E_c(B, F_{oo}, F_{vv})$ with off-diagonal Fock blocks; the Lagrangian
> gradient with $\Gamma = 4x$ and $P = 2p$ from the transcribed
> `rpa_ri_densities` equals a Richardson difference of
> $E_{\mathrm{RHF}} + E_c$ with the RHF re-solved, with and without a frozen
> core.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_gradient_sympy.py`, checks `gamma-vs-fd`, `p-vs-fd`, `gradient-vs-fd`, `gradient-vs-fd-frozen-core` |
> | Negative control | the runner's mutations of the $\Gamma$ weight $1/(4\pi) \to 1/(2\pi)$, the sign in $g^{[1]}$, the $W_{vo}$ weight and the frozen-core denominator each turn a check red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/gradient_analytic.rs::{rpa_ri_densities,frequency_response,g_divided_difference}` |
> | Code test | `h2o_matches_fd5_of_the_same_quadrature` in `crates/ferric-rpa/tests/rpa_analytic_gradient.rs` |
>
> <!-- verified: id=rpa.grad.densities; script=scripts/rpa_gradient_sympy.py; claim=gradient-vs-fd; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/gradient_analytic.rs::{rpa_ri_densities,frequency_response,g_divided_difference}; rust_test=ferric-rpa/tests/rpa_analytic_gradient.rs::h2o_matches_fd5_of_the_same_quadrature; registry=rpa-gradient -->

## Lagrangian, Z-vector and frozen core {#rpa-gradient-lagrangian-z-vector-frozen}

The orbital derivative of $E_c$ is collected as $\sum_{pq}Y_{pq}U_{pq}$ with
$Y_{pi} = \sum_{Pa}\Gamma^P_{ia}B^P_{pa}$, $Y_{pa} = \sum_{Pi}\Gamma^P_{ia}B^P_{ip}$
and $2P_{pq}\varepsilon_p$ on the $oo$ and $vv$ blocks. These are the
RPA counterparts of the Lagrangian matrix $I$ of (@eq:rimp2-gradient-2). The Z-vector
right-hand side has the same shape as (@eq:mp2-grad-assembly):

$$ L_{ai} = Y_{ai} - Y_{ia} + 4\big(\mathbf C_v^{\mathsf T}G[P^{\mathrm{AO}}]\mathbf C_o\big)_{ai}, \label{eq:rpa-gradient-3} $$

and the solve is the closed-shell equation (@eq:zvec-hessian-product). With core orbitals $I$
frozen, $\Gamma$ and $P_{ij}$ run over the active occupied orbitals and the
canonical condition $F_{Ij} = 0$ adds the core-active block
$P_{Ij} = -Y_{Ij}/\big(2(\varepsilon_I - \varepsilon_j)\big)$ to $P_{oo}$.

Why can the RI-MP2 code take over from here? It reads the amplitudes only
through $x$ and the unrelaxed densities $p$, and it treats them as the
derivatives of the energy: $\partial E/\partial B = 4x$ (consistent with
Euler's theorem, since $E^{(2)} = \sum B\cdot x$ is of degree 4 in $B$) and
$\partial E/\partial\varepsilon_k = 2p_{kk}$ (section @sec:rimp2-gradient-densities-right-hand-side). Give it
$x = \Gamma/4$ and $p = P/2$ and every later step differentiates $E_c$
instead of $E^{(2)}$. So the Lagrangian matrix, the frozen-core block, the
Z-vector, the relaxed and energy-weighted densities and the RI response are
the RI-MP2 functions; the RPA intermediates enter through
`Mp2Intermediates::from_ri_densities`.

### The orbital response {#rpa-gradient-orbital-response}

The orbitals are canonical RHF orbitals: $F_{ai} = 0$, orthonormal in the
moving AO basis. Parametrize their change as $\mathbf C(x) = \mathbf C\,U(x)$.
Orthonormality in the moving basis fixes the symmetric part,

$$ U + U^{\mathsf T} = -S^{(x)}, \qquad S^{(x)} = \mathbf C^{\mathsf T}\,\partial_xS\,\mathbf C . $$

By the invariance of section @sec:rpa-gradient-energy-b-f, the occupied-occupied and
virtual-virtual parts can take the symmetric choice
$U_{oo} = -\tfrac12S^{(x)}_{oo}$, $U_{vv} = -\tfrac12S^{(x)}_{vv}$ exactly.
Only $U_{ai}$ needs a response equation; $U_{ia} = -S^{(x)}_{ia} - U_{ai}$
follows.

Collect the orbital derivative of $E_c$ as $\sum_{pq}Y_{pq}U_{pq}$. The
$B$ dependence gives

$$ Y_{pi} = \sum_{Pa}\Gamma^P_{ia}B^P_{pa}, \qquad Y_{pa} = \sum_{Pi}\Gamma^P_{ia}B^P_{ip} . $$

The Fock blocks move with the orbitals: in canonical orbitals
$\delta F_{pq} \ni U_{qp}\varepsilon_q + U_{pq}\varepsilon_p$, and $P$ is
symmetric, so $\sum_{pq}P_{pq}\,\delta F_{pq}$ adds
$Y_{pq} \mathrel{+}= 2P_{pq}\varepsilon_p$ on the $oo$ and $vv$ blocks.

The Fock blocks also move through the density. With $G[X] = J[X] - \tfrac12K[X]$
and $G$ symmetric,

$$ \sum_{pq}P_{pq}\big(\mathbf C^{\mathsf T}G[D^{(1)}]\mathbf C\big)_{pq} = \operatorname{tr}\big(G[P^{\mathrm{AO}}]\,D^{(1)}\big), \qquad P^{\mathrm{AO}} = \mathbf C_oP_{oo}\mathbf C_o^{\mathsf T} + \mathbf C_vP_{vv}\mathbf C_v^{\mathsf T}, $$

with $D^{(1)}$ the first-order change of $D = 2\mathbf C_o\mathbf C_o^{\mathsf T}$.
Insert $\delta\mathbf C_o = \mathbf C_oU_{oo} + \mathbf C_vU_{vo}$ and
$U_{oo} + U_{oo}^{\mathsf T} = -S^{(x)}_{oo}$:

$$ D^{(1)} = 2\big(\mathbf C_vU_{vo}\mathbf C_o^{\mathsf T} + \text{h.c.}\big) - 2\mathbf C_oS^{(x)}_{oo}\mathbf C_o^{\mathsf T} . $$

The coefficient of $U_{ai}$ in all of this is $L_{ai}$ of (@eq:rpa-gradient-3).

The reference supplies the equation for $U_{ai}$, the CPHF equation
$A\,U^{(x)} = R^{(x)}$ with

$$ (AU)_{ai} = (\varepsilon_a - \varepsilon_i)U_{ai} + \Big(\mathbf C_v^{\mathsf T}G\big[2(\mathbf C_vU\mathbf C_o^{\mathsf T} + \text{h.c.})\big]\mathbf C_o\Big)_{ai}, $$

$$ R^{(x)}_{ai} = -F^{[x]}_{ai} + \varepsilon_iS^{(x)}_{ai} + \Big(\mathbf C_v^{\mathsf T}G\big[2\mathbf C_oS^{(x)}_{oo}\mathbf C_o^{\mathsf T}\big]\mathbf C_o\Big)_{ai}, $$

where $F^{[x]} = \mathbf C^{\mathsf T}\big(\partial_xh + G^{[x]}[D^{\mathrm{HF}}]\big)\mathbf C$
is the skeleton Fock derivative at fixed density. $A$ is symmetric, so the
$3N$ responses are replaced by one solve $Az = L$, and
$\sum_{ai}L_{ai}U^{(x)}_{ai} = \sum_{ai}z_{ai}R^{(x)}_{ai}$.

### The gradient {#rpa-gradient-assembly}

With the relaxed correlation density

$$ D_2 = P^{\mathrm{AO}} - \tfrac12\big(\mathbf C_vz\mathbf C_o^{\mathsf T} + \text{h.c.}\big) $$

the gradient is

$$ \frac{dE}{dx} = \frac{dE_{\mathrm{HF}}}{dx} + \operatorname{tr}\Big[D_2\big(\partial_xh + G^{[x]}[D^{\mathrm{HF}}]\big)\Big] + \sum_{pq}W_{pq}S^{(x)}_{pq} + \sum_{Q\mu\nu}\tilde\Gamma^Q_{\mu\nu}\,\partial_x(Q|\mu\nu) - \tfrac12\sum_{PQ}\gamma_{PQ}\,\partial_xV_{PQ}, $$

with the energy-weighted density

$$ W_{oo} = -\tfrac12Y_{oo} - 2\mathbf C_o^{\mathsf T}G[D_2]\mathbf C_o, \qquad W_{vv} = -\tfrac12Y_{vv}, \qquad W_{ia} = -Y_{ia}, \qquad W_{ai} = z_{ai}\varepsilon_i . $$

$\tilde\Gamma = M^{\mathsf T}\Gamma$, that is
$\tilde\Gamma^Q_{ia} = \sum_PM_{PQ}\Gamma^P_{ia}$, back-transformed to AOs,
and $\gamma_{PQ} = \sum_{ia}\tilde\Gamma^P_{ia}c^Q_{ia}$ with
$c = V^{-1}(Q|ia)$. Because $U_{oo}$ is fixed at $-\tfrac12S^{(x)}_{oo}$,
the $G[D^{(1)}]$ overlap term of $P^{\mathrm{AO}}$ and the one of the
Z-vector combine into the single $G[D_2]$ term of $W_{oo}$.

### Frozen core {#rpa-gradient-frozen-core}

With core orbitals $I$ frozen, $\Gamma$ and $P_{ij}$ run over the active
occupied orbitals only, and core-active rotations are no longer a symmetry
of $E_c$. The canonical condition $F_{Ij} = 0$ fixes them. Without the
orbital terms it reads

$$ U_{Ij}(\varepsilon_I - \varepsilon_j) = -F^{(x)}_{Ij} + \varepsilon_jS^{(x)}_{Ij}, $$

and it adds the symmetric core-active block

$$ P_{Ij} = P_{jI} = -\frac{Y_{Ij} - Y_{jI}}{2(\varepsilon_I - \varepsilon_j)} = -\frac{Y_{Ij}}{2(\varepsilon_I - \varepsilon_j)} $$

to $P_{oo}$. $Y_{jI} = 0$ because no core column enters $B$. The factor
$\tfrac12$ appears because $\sum_{ij}P_{ij}\,\delta F_{ij}$ visits the pair
twice. From there $P_{Ij}$ is an ordinary part of $P$: it enters the
$2P\varepsilon$ term of $Y$, $P^{\mathrm{AO}}$, $L$, $D_2$ and $W$.

This is `ferric-mp2/src/zvector.rs::frozen_core_block`, which forms
$D_{kI} = I_{kI}/(\varepsilon_k - \varepsilon_I)$, in the code's naming with
$k$ the frozen core orbital and $I$ the active one; its
$I_{qi} = -2\sum_{Pa}x^P_{ia}B^P_{qa}$ is $-\tfrac12Y_{qi}$ at $x = \Gamma/4$.
The block needs a gap between the core and the active occupied orbitals,
which every frozen-core choice of `ferric_core` has. A gap below $10^{-8}$
is an error ("the frozen-core gradient is undefined").

### What is reused from ferric-mp2 {#rpa-gradient-reuse}

The gradient is the RI-MP2 relaxed-density gradient of
section @sec:rimp2-gradient with a different $(\Gamma, P_{oo}, P_{vv})$. Unchanged:

| Step | Function |
|---|---|
| Lagrangian matrix $I$ from the three-index density, at $x = \Gamma/4$ | `ferric-mp2/src/zvector.rs::build_imat_ri` |
| frozen core-active block | `ferric-mp2/src/zvector.rs::frozen_core_block` |
| Z-vector solve on the reference $J/K$ route | `ferric-mp2/src/zvector.rs::solve_zvector_timed` |
| relaxed and energy-weighted densities, their one-electron, overlap and two-electron derivative terms | `ferric-mp2/src/gradient.rs::mp2_relaxed_lagrangian_gradient` |
| 3-centre/2-centre RI response | `ferric-mp2/src/gradient.rs::integral_response_gradient_3c2c_timed` |

The RPA-specific code is the frequency loop that builds $\Gamma$, $P_{oo}$
and $P_{vv}$ from $B$, the orbital energies and the grid. The MP2 pipeline
reads the pair amplitudes only through $x$, so the RPA intermediates carry
$x$ directly and no $t_2$.

## The algorithm {#rpa-gradient-algorithm}

| Step | Quantity | Function |
|---|---|---|
| 1 | RHF, density convergence $10^{-9}$ | `solve_rhf_warm` |
| 2 | scope check | `check_analytic_scope` |
| 3 | per node: $\Pi_k$, its eigendecomposition, $S$, $Q_kB$; accumulate $E_c$, $x$, $P_{oo}$, $P_{vv}$ | `rpa_ri_densities`, `frequency_response`, `accumulate_p_vv`, `accumulate_p_oo` |
| 4 | Lagrangian matrix, Z-vector, assembly, RI response | the RI-MP2 functions of section @sec:rimp2-gradient |

Per node the cost is one $n_{\mathrm{aux}} \times n_{\mathrm{aux}}$
eigendecomposition, $W_k = Q_kB$ ($n_{\mathrm{aux}}^2 n_on_v$), and the $P$
blocks. Only the blocks $(M_k)_{ia,ib}$ and $(M_k)_{ia,ja}$ of
$M_k = B^{\mathsf T}Q_kB$ enter (@eq:rpa-gradient-2), so the $P$ blocks cost one
$(n_v \times n_{\mathrm{aux}})(n_{\mathrm{aux}} \times n_v)$ product per
occupied orbital and one $(n_o \times n_{\mathrm{aux}})(n_{\mathrm{aux}} \times n_o)$
product per virtual: the RI-MP2 density cost times $n$.

## Options and scope {#rpa-gradient-options-scope}

| Option | Values | Default | Effect |
|---|---|---|---|
| `[rpa] gradient` (Python `rpa_gradient`) | `"fd"`, `"analytic"` | this one where `check_analytic_scope` accepts the configuration, else `"fd"` | the finite-difference gradient (3-point, $6N_{\mathrm{at}}$ energies) or this one |
| `[rpa] n_quad`, `u0`, `quadrature` | | as in section @sec:quadratures | the grid that is differentiated |
| `[rpa] frozen_core` | integer | as for the energy | the core-active block |

An explicit `"analytic"` raises, and an unset gradient falls back to `"fd"`
and records the reason in the certificate, when `check_analytic_scope`
refuses: `trunc_thresh` $\ne 0$, the Laplace $\chi_0$ backend, a non-dense
`chi0_sparsity`, and a non-Coulomb operator. Each of these changes the
energy away from the full-rank dense sum that (@eq:rpa-grad-densities)–(@eq:rpa-gradient-2) differentiate.
`[rpa] xc` (a KS reference) is refused for any pdep-rpa gradient.

## Numerical anchors {#rpa-gradient-numerical-anchors}

From `crates/ferric-rpa/tests/rpa_analytic_gradient.rs` (distorted H2O and
NH3 / cc-pVDZ, cc-pvdz-ri, every Cartesian component nonzero). The FD
reference is a step-converged 5-point difference of the PySCF energy surface.

| Quantity | Measured | Bar |
|---|---|---|
| H2O, $n = 40$ | $2.40\times10^{-9}$ Ha/Bohr | $2\times10^{-8}$ |
| H2O, $n = 6$ | $2.34\times10^{-9}$ | $2\times10^{-8}$ |
| H2O, $n = 40$, frozen core 1 | $2.44\times10^{-9}$ | $2\times10^{-8}$ |
| NH3, $n = 40$ | $8.31\times10^{-10}$ | $2\times10^{-8}$ |
| $n = 6$ gradient vs the $n = 40$ FD | misses by $3.05\times10^{-4}$ | must miss |
| second-order term ($n = 200$) vs `scs_mp2_gradient_analytical` at $(2, 0)$ | $2.6\times10^{-15}$ | $10^{-7}$ |

The $n = 6$ row is the artifact control. If the code differentiated the
converged integral instead of the $n$-point sum, its $n = 6$ gradient would
match the $n = 40$ FD and miss the $n = 6$ one. It does the opposite.

`site/src/methods/rpa-gw.md` reports one timing: on n-hexane / cc-pVDZ at
$n = 20$ the analytic gradient took 38.4 s against 1556.6 s for the
finite-difference gradient, and the two differ by at most
$2.25\times10^{-8}$ Ha/Bohr (one run each, one BLAS thread, loaded box).
The finite-difference gradient needs $6N_{\mathrm{at}}$ energies; the
analytic one needs one energy pass and one Z-vector.

### What is proven and what is not {#rpa-gradient-proven}

`scripts/rpa_gradient_sympy.py` checks the gradient on a toy closed-shell
system with $x$-dependent $S$ and $h$ and exactly factorized integrals
$(\mu\nu|\lambda\sigma) = \sum_PL_{P,\mu\nu}L_{P,\lambda\sigma}$. The metric is
the unit matrix, so the $\gamma$ term of the gradient vanishes there. The
Lagrangian expression of section @sec:rpa-gradient-assembly, with
$\Gamma = 4x$ and $P = 2p$ from the transcribed `rpa_ri_densities`, equals a
Richardson central difference of $E_{\mathrm{RHF}} + E_c$ with the RHF
re-solved at $x \pm h$ to $2\times10^{-12}$, and to $3\times10^{-11}$ with
one frozen core orbital. The script also checks $\Gamma$ and $P$ directly as
derivatives of $E_c(B, F_{oo}, F_{vv})$ with off-diagonal Fock blocks, the
divided difference $g^{[1]}$ and the second-order integral. The $\gamma$
term, with a non-unit metric, is covered by the Rust tests of the table
above, not by the script.

## Sources {#rpa-gradient-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-rpa/theory.md`, "Analytic nuclear gradient (closed-shell RI-dRPA, HF reference)" and its subsections | (@eq:rpa-grad-densities)–(@eq:rpa-gradient-3), $W$, the frozen-core block |
| `scripts/rpa_gradient_sympy.py` | the proofs |
| Burow, Bates, Furche and Eshuis, JCTC 10, 180 (2014) | the frequency-integrated relaxed-density route (cited in `theory.md`) |
| Rekkedal et al., JCP 139, 081101 (2013) | the ring-CCD Lagrangian, which gives the same gradient for the direct-ring energy; not used |

**Known limits (2026-11-15).** Closed-shell RHF reference, full rank, dense
$\chi_0$, Coulomb operator. The CLI and Python `run_optimize` default to this
gradient where `check_analytic_scope` accepts the configuration, and to the
finite-difference one elsewhere.
