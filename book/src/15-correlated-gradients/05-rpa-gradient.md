# 15.5 RI-dRPA gradient

<!-- section: 15.5; status: ready; registry: rpa-gradient;
     theory: crates/ferric-rpa/theory.md#"Analytic nuclear gradient (closed-shell RI-dRPA, HF reference)", "Lagrangian and Z-vector", "Gradient", "Frozen core" -->

The analytic dRPA gradient differentiates the energy of section 18.3 at full
rank on a fixed $n$-point imaginary-frequency grid. The nodes and weights
depend only on $(n, u_0)$, so the grid has no geometry derivative, and the
result is the exact derivative of the $n$-point sum, not of the converged
integral. The frequency loop produces a three-index density $\Gamma$ and
occupied and virtual densities $P$; from there the RI-MP2 machinery of
section 15.2 is reused unchanged.

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
| Python | none; `run_optimize` has no `pdep-rpa` method |

## 15.5.1 The densities

$$ E_c = \frac{1}{2\pi}\sum_k w_k\,S(\Pi_k), \qquad S(\Pi) = \ln\det(\mathbf 1 + \Pi) - \operatorname{tr}\Pi . $$

With $\delta S = \operatorname{tr}[Q_k\,\delta\Pi_k]$ and $M_k = B^{\mathsf T}Q_kB$:

$$ \Gamma^P_{ia} = \frac1\pi\sum_kw_k\,g_k(e_{ia})\,(Q_kB)^P_{ia}, \tag{15.12} $$

$$ P_{ab} = \frac{1}{2\pi}\sum_kw_k\sum_i(M_k)_{ia,ib}\,g_k^{[1]}(e_{ia}, e_{ib}), \qquad P_{ij} = -\frac{1}{2\pi}\sum_kw_k\sum_a(M_k)_{ia,ja}\,g_k^{[1]}(e_{ia}, e_{ja}), \tag{15.13} $$

$$ g^{[1]}(x, y) = \frac{g(x) - g(y)}{x - y} = \frac{4(u^2 - xy)}{(u^2 + x^2)(u^2 + y^2)} . $$

$P$ is the derivative with respect to the full Fock blocks, not only the
orbital energies, so the occupied-occupied and virtual-virtual rotations need
no response equation. The divided difference is smooth at $x = y$, where it
is $g'(x)$, so degenerate orbitals need no special case.

**Second-order limit.** Replacing $S$ by $-\tfrac12\operatorname{tr}\Pi^2$
($Q_k = -\Pi_k$) gives, as $n \to \infty$, twice the opposite-spin MP2
energy, which is SCS-MP2 at $(c_{\mathrm{OS}}, c_{\mathrm{SS}}) = (2, 0)$.
The densities then become the SCS-MP2 densities of section 15.2 with
$\Gamma = 4x$ and $P = P^{\mathrm{MP2}} + P^{\mathrm{MP2}\,\mathsf T}$. That
limit is the exactness anchor against independent code.

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

An eigenvalue of $\mathbf 1 + \Pi_k$ at or below zero is an error.

> **Verified equations 15.12 and 15.13** · the densities and the gradient
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

## 15.5.2 Lagrangian, Z-vector and frozen core

The orbital derivative of $E_c$ is collected as $\sum_{pq}Y_{pq}U_{pq}$ with
$Y_{pi} = \sum_{Pa}\Gamma^P_{ia}B^P_{pa}$, $Y_{pa} = \sum_{Pi}\Gamma^P_{ia}B^P_{ip}$
and $2P_{pq}\varepsilon_p$ on the $oo$ and $vv$ blocks. The Z-vector
right-hand side is

$$ L_{ai} = Y_{ai} - Y_{ia} + 4\big(\mathbf C_v^{\mathsf T}G[P^{\mathrm{AO}}]\mathbf C_o\big)_{ai}, \tag{15.14} $$

and the solve is the closed-shell equation (15.3). With core orbitals $I$
frozen, $\Gamma$ and $P_{ij}$ run over the active occupied orbitals and the
canonical condition $F_{Ij} = 0$ adds the core-active block
$P_{Ij} = -Y_{Ij}/\big(2(\varepsilon_I - \varepsilon_j)\big)$ to $P_{oo}$.

The RI-MP2 pipeline reads the amplitudes only through $x$, so the RPA
intermediates carry $x = \Gamma/4$ and $P/2$ directly, and the Lagrangian
matrix, the frozen-core block, the Z-vector, the relaxed and energy-weighted
densities and the RI response are the RI-MP2 functions.

## 15.5.3 The algorithm

| Step | Quantity | Function |
|---|---|---|
| 1 | RHF, density convergence $10^{-9}$ | `solve_rhf_warm` |
| 2 | scope check | `check_analytic_scope` |
| 3 | per node: $\Pi_k$, its eigendecomposition, $S$, $Q_kB$; accumulate $E_c$, $x$, $P_{oo}$, $P_{vv}$ | `rpa_ri_densities`, `frequency_response`, `accumulate_p_vv`, `accumulate_p_oo` |
| 4 | Lagrangian matrix, Z-vector, assembly, RI response | the RI-MP2 functions of section 15.2 |

Per node the cost is one $n_{\mathrm{aux}} \times n_{\mathrm{aux}}$
eigendecomposition, $Q_kB$ ($n_{\mathrm{aux}}^2 n_on_v$), and the RI-MP2
density cost for the $P$ blocks.

## 15.5.4 Options and scope

| Option | Values | Default | Effect |
|---|---|---|---|
| `[rpa] gradient` | `"fd"`, `"analytic"` | `"fd"` | the finite-difference gradient (3-point, $6N_{\mathrm{at}}$ energies) or this one |
| `[rpa] n_quad`, `u0`, `quadrature` | | as in section 18.2 | the grid that is differentiated |
| `[rpa] frozen_core` | integer | as for the energy | the core-active block |

`check_analytic_scope` refuses, and points to `gradient = "fd"`:
`trunc_thresh` $\ne 0$, the Laplace $\chi_0$ backend, a non-dense
`chi0_sparsity`, and a non-Coulomb operator. `[rpa] xc` (a KS reference) is
refused for any pdep-rpa gradient.

## 15.5.5 Numerical anchors

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

The $n = 6$ row is the artifact control: a gradient of the converged integral
instead of the $n$-point sum would match the $n = 40$ FD and miss the $n = 6$
one.

`site/src/methods/rpa-gw.md` reports one timing: on n-hexane / cc-pVDZ at
$n = 20$ the analytic gradient took 38.4 s against 1556.6 s for the
finite-difference gradient, and the two differ by at most
$2.25\times10^{-8}$ Ha/Bohr (one run each, one BLAS thread, loaded box).

## 15.5.6 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-rpa/theory.md`, "Analytic nuclear gradient (closed-shell RI-dRPA, HF reference)" and its subsections | (15.12)–(15.14), $W$, the frozen-core block |
| `scripts/rpa_gradient_sympy.py` | the proofs |
| Burow, Bates, Furche and Eshuis, JCTC 10, 180 (2014) | the frequency-integrated relaxed-density route (cited in `theory.md`) |

**Known limits (2026-11-15).** Closed-shell RHF reference, full rank, dense
$\chi_0$, Coulomb operator. CLI only. The default CLI gradient is still the
finite-difference one.
