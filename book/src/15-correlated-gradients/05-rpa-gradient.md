# RI-dRPA gradient {#rpa-gradient}

<!-- status: ready; registry: rpa-gradient;
     theory: crates/ferric-rpa/theory.md#"Analytic nuclear gradient (closed-shell RI-dRPA, HF reference)", "Lagrangian and Z-vector", "Gradient", "Frozen core" -->

The direct RPA correlation energy sums ring diagrams to all orders: an
electron–hole pair is created, propagates, and creates the next pair through
the Coulomb interaction, any number of times. Like MP2 it is evaluated on
Hartree–Fock orbitals that do not minimize it, so its gradient needs the
same orbital response. Unlike MP2 it is an integral over imaginary frequency
of a log-determinant, so the densities that feed the response have to be
collected frequency by frequency.

The analytic dRPA gradient differentiates the energy of section @sec:pdep at full
rank on a fixed $n$-point imaginary-frequency grid. The nodes and weights
depend only on $(n, u_0)$, so the grid has no geometry derivative, and the
result is the exact derivative of the $n$-point sum, not of the converged
integral. The frequency loop produces a three-index density $\Gamma$ and
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
this gives twice the opposite-spin MP2 energy, which is SCS-MP2 at
$(c_{\mathrm{OS}}, c_{\mathrm{SS}}) = (2, 0)$. The densities then become the
SCS-MP2 densities of section @sec:rimp2-gradient with $\Gamma = 4x$ and
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

## The algorithm {#rpa-gradient-algorithm}

| Step | Quantity | Function |
|---|---|---|
| 1 | RHF, density convergence $10^{-9}$ | `solve_rhf_warm` |
| 2 | scope check | `check_analytic_scope` |
| 3 | per node: $\Pi_k$, its eigendecomposition, $S$, $Q_kB$; accumulate $E_c$, $x$, $P_{oo}$, $P_{vv}$ | `rpa_ri_densities`, `frequency_response`, `accumulate_p_vv`, `accumulate_p_oo` |
| 4 | Lagrangian matrix, Z-vector, assembly, RI response | the RI-MP2 functions of section @sec:rimp2-gradient |

Per node the cost is one $n_{\mathrm{aux}} \times n_{\mathrm{aux}}$
eigendecomposition, $Q_kB$ ($n_{\mathrm{aux}}^2 n_on_v$), and the RI-MP2
density cost for the $P$ blocks.

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

## Sources {#rpa-gradient-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-rpa/theory.md`, "Analytic nuclear gradient (closed-shell RI-dRPA, HF reference)" and its subsections | (@eq:rpa-grad-densities)–(@eq:rpa-gradient-3), $W$, the frozen-core block |
| `scripts/rpa_gradient_sympy.py` | the proofs |
| Burow, Bates, Furche and Eshuis, JCTC 10, 180 (2014) | the frequency-integrated relaxed-density route (cited in `theory.md`) |

**Known limits (2026-11-15).** Closed-shell RHF reference, full rank, dense
$\chi_0$, Coulomb operator. The CLI and Python `run_optimize` default to this
gradient where `check_analytic_scope` accepts the configuration, and to the
finite-difference one elsewhere.
