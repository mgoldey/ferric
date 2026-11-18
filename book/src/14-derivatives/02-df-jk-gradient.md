# Gradients of density-fitted J and K {#df-jk-gradient}

<!-- status: edit; registry: hf-gradient -->

A density-fitted SCF reports a different energy from the four-centre one,
so it has a different gradient. The fitted energy depends on the raw
three-index integrals, on where the auxiliary functions sit (they are
centred on atoms and move with them), and on the metric that inverts them.
A gradient that pairs the fitted energy with four-centre derivative
integrals is not the slope of the energy the SCF reports, and an optimizer
driven by it stops at a point that is stationary for neither surface.

The plan: differentiate the fitted Coulomb and exchange energies at fixed
density, collect both into two weight arrays, deal with the truncated
exchange metric, then follow the passes that contract the weights with
derivative integrals.

| Symbol | Meaning | Code name |
|---|---|---|
| $(P\vert\mu\nu)$ | raw three-index integral | `ThreeIndexSource` blocks |
| $V_{PQ}$ | two-index metric $(P\vert Q)$ of the same operator | `coulomb_metric_2c` |
| $d_P$ | fitted Coulomb density, $\sum_{\mu\nu}D_{\mu\nu}(P\vert\mu\nu)$ | `d_p` |
| $c$ | RI-J coefficients $V^{-1}d$ | `c_j` |
| $X_s$, $\alpha_s$ | factor of one exchange density, $D_s = X_sX_s^{\mathsf T}$, and its energy weight | `DfKChannel`, `density_factor` |
| $L^s_{P,ij}$ | fitted exchange integrals $\sum X_{\mu i}(P\vert\mu\nu)X_{\nu j}$ | `ls` |
| $V^{+}$ | the metric inverse the exchange energy applies: $1/\lambda$ above its cut, $0$ below | `metric_lindep_cut` in `ferric-scf/src/df_k.rs` |
| $\Gamma^s$ | exchange fit coefficients $V^{+}L^s$ | `gammas` |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/df_gradient.rs::df_jk_gradient`, `ferric-scf/src/df_gradient.rs::df_jk_gradient_at_precision`, `ferric-scf/src/df_gradient.rs::df_jk_gradient_with_raw` |
| Routing | `ferric-scf/src/df_gradient.rs::routed_two_electron_gradient` reads the builders the SCF recorded in `ScfResult::df_jk` and sends each fitted term here and each exact term to the four-centre loops |
| CLI, Python | no separate entry: every `task = "optimize"` and `with_gradient=True` run of a fitted SCF uses it |

## The energies that are differentiated {#df-jk-gradient-energies-differentiated}

The SCF made the fitted energy stationary in the orbitals, so the argument
of section @sec:hf-gradient goes through unchanged: the orbital response is carried by
the energy-weighted density $W$ in the Pulay term. What remains is the
derivative of the fitted two-electron energy at fixed density. Section @sec:ri-j-ri-k-split-ri-j
defines the fitted J and K; here we differentiate them.

**RI-J.** The Coulomb builder `DfJ` solves with the Cholesky factor of the
Coulomb metric, with no truncation, so the energy is

$$ E_J = \tfrac12\, d^{\mathsf T}V^{-1}d, \qquad d_P = \sum_{\mu\nu}D_{\mu\nu}(P\vert\mu\nu) . $$

At fixed $D$, $\partial d_P = \sum_{\mu\nu}D_{\mu\nu}\,\partial(P\vert\mu\nu)$.
Apply the product rule; the two $d$ factors give equal terms because
$V^{-1}$ is symmetric:

$$ \partial E_J = d^{\mathsf T}V^{-1}\,\partial d + \tfrac12\, d^{\mathsf T}\,\partial(V^{-1})\,d . $$

Differentiate $VV^{-1} = 1$ to get $\partial V^{-1} = -V^{-1}(\partial V)V^{-1}$,
substitute, and write $c = V^{-1}d$:

$$ E_J = \tfrac12\, d^{\mathsf T}V^{-1}d, \qquad c = V^{-1}d, \qquad
\partial E_J = \sum_P c_P \sum_{\mu\nu} D_{\mu\nu}\,\partial(P\vert\mu\nu) \;-\; \tfrac12\, c^{\mathsf T}(\partial V)\,c . \label{eq:df-jk-gradient-1} $$

Check it with one auxiliary function: $E_J = d^2/(2V)$, and the quotient rule
gives $(d/V)\,\partial d - \tfrac12 (d/V)^2\,\partial V$, which is (@eq:df-jk-gradient-1) with
$c = d/V$.

**RI-K.** Exchange is fitted one channel at a time. Factor each density as
$D_s = X_sX_s^{\mathsf T}$ and give it a weight $\alpha_s$. The weight is
$-c_K/4$ for a closed-shell total density and $-c_K/2$ per spin for UHF and
ROHF, where $c_K$ is the exact-exchange fraction
(`ferric-scf/src/df_gradient.rs::exchange_channels`). These are the exchange
weights of the four-centre $\Gamma$ in section @sec:hf-gradient, $-\tfrac14 D_{\mu\lambda}D_{\nu\sigma}$
and $-\tfrac12 D^\tau_{\mu\lambda}D^\tau_{\nu\sigma}$, scaled by $c_K$. The fitted energy is a
sum of quadratic forms in the fitted integrals
$L^s_{P,ij} = \sum_{\mu\nu}X^s_{\mu i}(P\vert\mu\nu)X^s_{\nu j}$. Differentiate
it with the product rule. The two $L$ factors give equal terms because
$V^{+}$ is symmetric, which is the factor 2; the metric gives the rest:

$$ E_K = \sum_s \alpha_s \sum_{ij} L^{s\,\mathsf T}_{ij}\,V^{+}\,L^s_{ij}, \qquad
\partial E_K = \sum_s 2\alpha_s \sum_{P,ij}\Gamma^s_{P,ij}\,\partial L^s_{P,ij} \;+\; \sum_{PQ} Y_{PQ}\,\partial V^{+}_{PQ}, \label{eq:df-jk-gradient-2} $$

with $\Gamma^s = V^{+}L^s$, $Y = \sum_s\alpha_s\sum_{ij}L^s_{ij}L^{s\,\mathsf T}_{ij}$, and
$\partial L^s_{P,ij} = \sum X_{\mu i}\,\partial(P\vert\mu\nu)\,X_{\nu j}$ at fixed
$X_s$.

**Two weight arrays.** Both gradients are contractions of the same two
derivative integrals. Move the $X$ factors of $\partial L^s$ onto $\Gamma^s$,
$\sum_{ij}\Gamma^s_{P,ij}X^s_{\mu i}X^s_{\nu j} = (X^s\Gamma^s_PX^{s\,\mathsf T})_{\mu\nu}$,
and collect:

$$ \partial E_{\mathrm{DF}} = \sum_{P,\mu\nu} G_{P,\mu\nu}\,\partial(P\vert\mu\nu) + \sum_{PQ} H_{PQ}\,\partial V_{PQ}, \label{eq:df-grad-jk} $$

$$ G_{P,\mu\nu} = c_P D_{\mu\nu} + \sum_s 2\alpha_s \sum_{ij} X^s_{\mu i}\,\Gamma^s_{P,ij}\,X^s_{\nu j}, \qquad
H = -\tfrac12\,c\,c^{\mathsf T} + H_K . $$

$H_K$ is $Y$ pushed through the derivative of $V^{+}$; section @sec:df-jk-gradient-truncated-exchange-metric builds
it. The weights are built once, before any derivative integral is
computed.

What makes (@eq:df-grad-jk) more than the four-centre gradient in new notation is
where the derivatives act. $\partial(P\vert\mu\nu)$ acts on all three
centres, including the auxiliary function's, and $\partial V_{PQ}$ on both
auxiliary centres. The auxiliary functions sit on atoms, so they exert
forces of their own, in the same way the orbital basis gives the Pulay
term. These are the standard DF-SCF gradient expressions (Weigend, Häser, Patzelt and
Ahlrichs, 1998, for RI-J; Weigend, Kattannek and Ahlrichs, 2009, for RI-K).

How large is the difference? On the cc-pVDZ test systems of section @sec:df-jk-gradient-numerical-anchors,
ferric's fitted gradient agrees with PySCF `df.grad` to about $10^{-10}$ Ha/Bohr, while the
four-centre derivative paired with the fitted energy differs from PySCF by
more than $5\times10^{-6}$ Ha/Bohr (section @sec:df-jk-gradient-numerical-anchors). An optimizer with a
$10^{-6}$ Ha/Bohr convergence bar would stop at the wrong place.

> **Verified equation @eq:df-grad-jk** · the fitted J and K gradient
>
> On a model whose three-index integrals and metric depend on one parameter, at a
> fixed density: the RI-J gradient of (@eq:df-jk-gradient-1) equals the exact sympy derivative of
> $\tfrac12d^{\mathsf T}V^{-1}d$ in rational arithmetic (claim D1); the RI-K
> gradient of (@eq:df-jk-gradient-2), with the weights $G$ and $H$ of (@eq:df-grad-jk) and every metric mode
> kept, equals the exact derivative of $E_K$ (claim D2); and with a dropped mode
> the kernel of equation @eq:df-jk-gradient-4, case by case as the code writes it, equals a
> fourth-order finite difference of $E_K(x)$ with the spectral cut in a gap, while
> the plain $-V^{+}(\partial V)V^{+}$ misses (claim D3).
>
> | | |
> |---|---|
> | Proof | `scripts/df_jk_gradient_sympy.py`, claims D1, D2 and D3 |
> | Negative control | the runner drops the $\tfrac12$ of the RI-J metric term (claim D1 fails) and zeroes the kept-dropped kernel entry (claim D3 fails); the plain formula is the script's own control |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/df_gradient.rs::{df_jk_gradient_core,truncated_inverse_divided_difference}` |
> | Code test | `hf_ri_jk_gradient_matches_fd_of_ri_energy` in `crates/ferric-scf/tests/df_jk_gradient.rs` |
>
> <!-- verified: id=df.grad.jk; script=scripts/df_jk_gradient_sympy.py; claim=D1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/df_gradient.rs::{df_jk_gradient_core,truncated_inverse_divided_difference}; rust_test=ferric-scf/tests/df_jk_gradient.rs::hf_ri_jk_gradient_matches_fd_of_ri_energy; registry=hf-gradient -->

The check on water in cc-pVDZ with def2-universal-jkfit, from the `589b50832`
build. Every fitted run stores the RI-J tensor in f64; energy-only runs
default to f32 (section @sec:ri-j-ri-k-split-ri-j), and an f32 energy in the
finite difference would put the rounding error into the comparison. The fitted
gradient matches the central difference of the fitted energy to 1.3e-7
Ha/Bohr; the four-centre gradient misses it by 2.5e-6:

```python
import numpy as np
import ferric

sym = ["O", "H", "H"]
x0 = np.array([[0.0, 0.0, 0.25], [0.0, 1.50, -0.85], [0.0, -1.35, -0.95]])
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = "def2-universal-jkfit"


def run(x, grad=False, fitted=True):
    mol = ferric.Molecule.from_arrays(sym, x, units="bohr")
    # f64 RI-J storage on every run: energy-only runs default to f32
    kw = dict(df_j_aux=aux, df_k_aux=aux, ri_j_precision="f64") if fitted else {}
    return ferric.run_rhf(mol, bs, energy_conv=1e-11, density_conv=1e-9, with_gradient=grad, **kw)


g_fit = np.asarray(run(x0, grad=True).gradient())
g_exact = np.asarray(run(x0, grad=True, fitted=False).gradient())
print("max |g_fit - g_exact| (fitting defect):", np.abs(g_fit - g_exact).max())
print("sum over atoms of g_fit:", g_fit.sum(axis=0))
h = 1e-3
atom, axis = 1, 1
xp = x0.copy(); xp[atom, axis] += h
xm = x0.copy(); xm[atom, axis] -= h
fd = (run(xp).energy - run(xm).energy) / (2 * h)
print(f"fitted: analytic {g_fit[atom, axis]:.10f}  FD of fitted energy {fd:.10f}  diff {abs(fd - g_fit[atom, axis]):.1e}")
print(f"exact four-centre gradient at the same coordinate {g_exact[atom, axis]:.10f}  miss vs FD {abs(fd - g_exact[atom, axis]):.1e}")
```

```text
max |g_fit - g_exact| (fitting defect): 1.7993087324086332e-05
sum over atoms of g_fit: [ 1.61085397e-27 -3.35842465e-15 -1.28785871e-14]
fitted: analytic 0.0289765390  FD of fitted energy 0.0289764043  diff 1.3e-07
exact four-centre gradient at the same coordinate 0.0289788597  miss vs FD 2.5e-06
```

## The truncated exchange metric {#df-jk-gradient-truncated-exchange-metric}

If $V^{+}$ were $V^{-1}$, $H_K$ would follow from the same identity as RI-J.
It is not. The exchange builder `DfK` forms $V^{+}$ from an
eigendecomposition $V = U\Lambda U^{\mathsf T}$ and drops the modes below a
cut (`ferric-scf/src/df_k.rs::metric_lindep_cut`, nominally $10^{-10}$,
moved into a spectral gap when an eigenvalue sits within rounding of it).
The long-range erf metric with a JK-fit basis always has such modes; the
module documentation of `df_gradient.rs` counts 17 of 113 for water /
def2-universal-jkfit at $\omega = 0.3$.

So $V^{+} = f(V)$ is a spectral function, with $f(\lambda) = 1/\lambda$ above
the cut and $0$ below. The derivative of a spectral function is the
Daleckii–Krein formula. Rotate $\partial V$ into the eigenbasis, multiply
element by element with the divided differences of $f$, and rotate back:

$$ \partial V^{+} = U\big(\mathsf F \circ U^{\mathsf T}\,\partial V\,U\big)U^{\mathsf T}, \qquad
\mathsf F_{kl} = \frac{f(\lambda_k)-f(\lambda_l)}{\lambda_k-\lambda_l},\quad \mathsf F_{kk} = f'(\lambda_k). \label{eq:df-jk-gradient-4} $$

Put the two values of $f$ into the divided difference and you get the four
cases the code writes
(`ferric-scf/src/df_gradient.rs::truncated_inverse_divided_difference`).
For two kept modes, $(1/\lambda_k - 1/\lambda_l)/(\lambda_k - \lambda_l) = -1/(\lambda_k\lambda_l)$,
and the diagonal $f'(\lambda_k) = -1/\lambda_k^2$ is the same expression.
For a kept $k$ and a dropped $l$, $f(\lambda_l) = 0$ leaves
$1/\big(\lambda_k(\lambda_k-\lambda_l)\big)$.

| $k$ | $l$ | $\mathsf F_{kl}$ |
|---|---|---|
| kept | kept | $-1/(\lambda_k\lambda_l)$ |
| kept | dropped | $1/\big(\lambda_k(\lambda_k-\lambda_l)\big)$ |
| dropped | kept | $1/\big(\lambda_l(\lambda_l-\lambda_k)\big)$ |
| dropped | dropped | $0$ |

<!-- excerpt: ferric-scf/src/df_gradient.rs::truncated_inverse_divided_difference -->
```rust
match (kk, kl) {
    (true, true) => -1.0 / (lk * ll),
    (true, false) => 1.0 / (lk * (lk - ll)),
    (false, true) => 1.0 / (ll * (ll - lk)),
    (false, false) => 0.0,
}
```

Check the limit with nothing dropped: every entry is $-1/(\lambda_k\lambda_l)$,
and (@eq:df-jk-gradient-4) becomes the familiar $\partial V^{+} = -V^{+}(\partial V)V^{+}$.
The kept-dropped entries are the rotation of kept modes into dropped ones:
as the nuclei move, the eigenvectors turn, and a kept direction picks up a
component along a direction the energy ignores. The familiar formula misses
that. Contracting (@eq:df-jk-gradient-4) with $Y$ gives
$H_K = U\big(\mathsf F\circ\tilde Y\big)U^{\mathsf T}$, $\tilde Y = U^{\mathsf T}YU$,
the two-centre weight of (@eq:df-grad-jk). The test
`divided_difference_reduces_to_minus_m_dv_m_when_nothing_dropped` in
`crates/ferric-scf/src/df_gradient.rs` pins the kept-kept and kept-dropped
entries.

The formula is exact for a fixed partition of the spectrum into kept and
dropped modes, which holds when the cut lies in a gap. The module
documentation records a prototype on water/6-31G at fixed density with the
cut in a clean gap: the Daleckii–Krein form matches finite differences to
$2\times10^{-8}$ Ha/Bohr, and the plain $-V^{+}(\partial V)V^{+}$ is
$5\times10^{-5}$ off.

**The gradient's noise floor.** Modes whose eigenvalue sits a little above the
energy's cut are noise-dominated in the derivative. Their fit coefficient
$\Gamma_k = (U^{\mathsf T}L)_k/\lambda_k$ carries the integrals' absolute
error divided by $\lambda_k$, while the modes carry almost no energy. The
gradient therefore treats modes below
`ferric-scf/src/df_gradient.rs::DFK_GRADIENT_NOISE_FLOOR` ($10^{-7}$) as
dropped even where the energy kept them, and prints a warning when the
energy held by that band exceeds `NOISE_BAND_ENERGY_WARN` ($10^{-6}$ Ha).
The gradient is then that of the energy with this band removed, and the
residual between gradient and reported energy is exactly the derivative of
that band. With no metric eigenvalue below $10^{-7}$, the usual case, the
floor changes nothing: the Coulomb and erfc metrics of ordinary systems
have no eigenvalue below about $10^{-5}$, and for them the formula is the
plain one.

The code builds $\Gamma^s$ in the eigenbasis with that floor applied. This
is the body of the channel loop in `df_jk_gradient_core`:

<!-- excerpt: ferric-scf/src/df_gradient.rs::df_jk_gradient_core -->
```rust
// Ỹ += α Lt Ltᵀ
par_gemm(*alpha, &lt.view(), &lt.t(), 1.0, &mut y_t);
// Γ = U diag(w) Lt, w_k = 1/λ_k on gradient-kept modes, else 0.
let mut wlt = lt;
for k in 0..naux {
    let w = if keep_g[k] { 1.0 / lam[k] } else { 0.0 };
    wlt.row_mut(k).mapv_inplace(|v| v * w);
}
let mut gam = Array2::<f64>::zeros(wlt.dim());
par_gemm(1.0, &u.view(), &wlt.view(), 0.0, &mut gam);
```

| Code | Equation |
|---|---|
| `lt` | $U^{\mathsf T}L^s$, the fitted integrals in the metric eigenbasis |
| `y_t` | $\tilde Y = U^{\mathsf T}YU$, accumulated over channels with weight $\alpha_s$ |
| `keep_g[k]` | $\lambda_k$ at or above the larger of the energy's cut and the noise floor |
| `gam` | $\Gamma^s = V^{+}L^s = U\,\mathrm{diag}(1/\lambda_k)\,U^{\mathsf T}L^s$ on kept modes |

After the loop, every element of `y_t` is multiplied by
`truncated_inverse_divided_difference` of its two modes and rotated back,
which is $H_K = U(\mathsf F\circ\tilde Y)U^{\mathsf T}$. The RI-J part of $H$ is
one line before it, `h2[(p, q)] -= 0.5 * c[p] * c[q]`, the
$-\tfrac12cc^{\mathsf T}$ of (@eq:df-grad-jk).

## The algorithm {#df-jk-gradient-algorithm}

The driver is `ferric-scf/src/df_gradient.rs::df_jk_gradient_core`. Its stages carry timer labels
that `FERRIC_TIMING` prints.

| Stage | Work | Timer label |
|---|---|---|
| 1 | $V_{PQ} = (P\vert Q)$ | `dfg:metric2c` |
| 2 | one pass over the raw $(P\vert\mu\nu)$ in aux-shell blocks: $d_P$ for RI-J, $L^s_{P,ij}$ per exchange channel. The tensor is never stored whole | `dfg:3c-pass` |
| 3 | $c = V^{-1}d$ by the Cholesky factor, as `DfJ`; the metric eigendecomposition (divide and conquer, the solver `DfK` used); $\Gamma^s$; $H = -\tfrac12 cc^{\mathsf T} + H_K$ | `dfg:metric-weights`, `dfg:metric-eigh`, `dfg:metric-gemms` |
| 4 | $\sum G_{P,\mu\nu}\,\partial(P\vert\mu\nu)$, one auxiliary shell per rayon task | `dfg:3c-deriv-pass` |
| 5 | $\sum H_{PQ}\,\partial V_{PQ}$ | `dfg:2c-deriv-pass` |

Why does stage 2 read the raw integrals at all, when the SCF already did?
The weights need $d_P$ and $L^s$ at the converged density, and the
derivative passes need $G$ and $H$ before they start. Stage 2 can read the
raw tensor the SCF kept
(`ferric-scf/src/df_gradient.rs::df_jk_gradient_with_raw`), after checking
that it was built for the same auxiliary basis, operator, molecule and
orbital basis; otherwise it recomputes. The two routes agree to rounding.
Stage 3 can reuse the SCF's own metric eigendecomposition when its metric is
bit-identical to the one rebuilt here.

The three-centre pass forms, for each auxiliary shell, the weights
$G_{P,\mu\nu}+G_{P,\nu\mu}$ for $s_1\neq s_2$ and $G_{P,\mu\nu}$ for
$s_1=s_2$, once per block in the order libint returns its nine components
$[\partial_P,\partial_1,\partial_2]$, and contracts them with nine dot
products (`ferric-scf/src/df_gradient/deriv3c.rs::three_center_deriv_pass`).
The nine are three Cartesian directions for each of the three centres
(section @sec:integrals-derivatives counts them).

## RI-J on a pair-screened tensor {#df-jk-gradient-pair-screened-ri-j}

A J-only SCF (a pure functional with RI-J and no exact exchange) can store
its raw tensor compressed: most AO pairs on a large molecule are two
functions on distant atoms whose product is numerically zero, and their
rows are left out (section @sec:three-index-sources). The SCF energy is then the energy of
the kept pairs only. Its gradient has to be the derivative of that energy,
or a search walks one surface while reporting another.

**The screen.** Cauchy–Schwarz bounds every element of a pair's row,
$\lvert(P\vert\mu\nu)\rvert \le Q_{\mu\nu}Q_P$ with
$Q_{\mu\nu} = \sqrt{(\mu\nu\vert\mu\nu)}$ and $Q_P = \sqrt{(P\vert P)}$.
`ferric-integrals/src/three_index_source/pair_screen.rs::PairScreen` keeps a
shell pair $(s_1, s_2)$ when $Q_{s_1s_2}\,q_{\max} > \tau$, where
$Q_{s_1s_2}$ is the operator's shell-pair Schwarz value (the largest
$Q_{\mu\nu}$ in the pair), $q_{\max} = \max_P\sqrt{(P\vert P)}$ of the same
operator's auxiliary metric, and $\tau$ is `integral_thresh`. Write $K$ for
the set of function pairs it keeps. On danuglipron (71 atoms, def2-SVP,
def2-universal-jkfit, 724 basis functions, 3635 auxiliary functions) at
$\tau = 10^{-12}$, 56.0% of the 56,616 shell pairs and 52.0% of the 262,450
packed function pairs survive, and the stored tensor drops from 7.63 GB to
3.97 GB (module documentation of `pair_screen.rs`).

**The energy.** RI-J is $E_J = \tfrac12 d^{\mathsf T}V^{-1}d$ with
$d_P = \sum_{\mu\nu}(P\vert\mu\nu)D_{\mu\nu}$ (@eq:df-jk-gradient-1). The screened SCF forms $d$
from the kept pairs only:

$$ \tilde E_J = \tfrac12\,\tilde d^{\mathsf T}V^{-1}\tilde d, \qquad
\tilde d_P = \sum_{(\mu\nu)\in K}(P\vert\mu\nu)\,D_{\mu\nu} = d_P(\tilde D), $$

with $\tilde D_{\mu\nu} = D_{\mu\nu}$ for $(\mu\nu)\in K$ and $0$ otherwise,
in both triangles. The last equality is what the gradient uses: $\tilde d$
is the unscreened $d$ evaluated at the masked density $\tilde D$.

**The gradient.** Hold $K$ fixed. It is a step function of the geometry,
constant almost everywhere. Write $\tilde c = V^{-1}\tilde d$. The SCF is
variational, so at fixed $D$ the orbital response goes into the
energy-weighted overlap term, which does not involve this tensor. Repeat the
steps that gave (@eq:df-jk-gradient-1), with $\tilde d$ in place of $d$:

$$ \frac{\partial\tilde E_J}{\partial x}
= \sum_P\tilde c_P\sum_{(\mu\nu)\in K}\frac{\partial(P\vert\mu\nu)}{\partial x}D_{\mu\nu}
- \tfrac12\,\tilde c^{\mathsf T}\frac{\partial V}{\partial x}\tilde c . $$

The first sum runs over $K$ only. The three-centre derivative pass is linear
in the density it contracts, so restricting its pair sum to $K$ is the same
as contracting $\tilde D$ over all pairs. Hence

$$ \nabla\tilde E_J = g_J(\tilde D), $$

where $g_J(\cdot)$ is the ordinary RI-J gradient as a function of the
density it is handed, with $d$, $c$ and the derivative contraction all formed
from $\tilde D$. The metric term uses $V$ and $\tilde c$ only, so nothing
else changes.

**The code.** `ferric-scf/src/df_gradient.rs::df_jk_gradient_core` does two
things when the kept tensor it was handed is pair-screened. It forms
$\tilde d$ from the stored columns
(`ferric-scf/src/df_gradient.rs::screened_d_p`). A stored column is one
function pair $(\mu, \nu)$ with $\nu \le \mu$, so an off-diagonal column
stands for $(\mu\nu)$ and $(\nu\mu)$ and carries the weight
$D_{\mu\nu} + D_{\nu\mu}$:

<!-- excerpt: ferric-scf/src/df_gradient.rs::screened_d_p -->
```rust
.map(|&(mu, nu)| {
    let (mu, nu) = (mu as usize, nu as usize);
    if mu == nu {
        d[(mu, mu)]
    } else {
        d[(mu, nu)] + d[(nu, mu)]
    }
})
```

And it hands the derivative pass $\tilde D$
(`ferric-scf/src/df_gradient.rs::density_on_kept_pairs`), which copies
$D_{\mu\nu}$ and $D_{\nu\mu}$ for every kept column into a zero matrix. A
dropped pair contributes zero to both, as it did to the SCF energy.

Only an f64 RI-J pass with no exchange channel reads a pair-screened
tensor. Any other pass drops it and recomputes the integrals, with the note
`kept pair-screened tensor not used (not a J-only f64 pass): recomputing`.
Inside the three-centre pass the same two conditions are hard errors:
"DF gradient: a pair-screened kept tensor serves an f64 J-only pass" and
"DF gradient: a pair-screened kept tensor cannot feed an exchange pass".

**How far it moves from the unscreened gradient.** $g_J$ is a quadratic form
$G(D, D)$ in the density. Put $D_{\mathrm{drop}} = D - \tilde D$ and expand
$G(D - D_{\mathrm{drop}}, D - D_{\mathrm{drop}})$:

$$ g_J(\tilde D) - g_J(D) = -G(D_{\mathrm{drop}}, D) - G(D, D_{\mathrm{drop}}) + G(D_{\mathrm{drop}}, D_{\mathrm{drop}}), $$

which is $O(\lVert D_{\mathrm{drop}}\rVert/\lVert D\rVert)\,\lVert g_J\rVert$.
On the fitted density the Schwarz bound gives, for every $P$, since a dropped
pair has $Q_{\mu\nu} \le \tau/q_{\max}$,

$$ \lvert d_P - \tilde d_P\rvert \le \sum_{\text{dropped}}\lvert(P\vert\mu\nu)\rvert\,\lvert D_{\mu\nu}\rvert
\le \tau\sum_{\text{dropped}}\lvert D_{\mu\nu}\rvert\,\frac{Q_P}{q_{\max}} \le \tau\sum_{\text{dropped}}\lvert D_{\mu\nu}\rvert, $$

and the coefficients move by $\tilde c - c = -V^{-1}(d - \tilde d)$. At
$\tau \le 0$ nothing is dropped, $\tilde D = D$, and the two gradients are
the same bits.

### What is proven and what is not {#df-jk-gradient-proven}

The tests in `crates/ferric-scf/tests/df_gradient_screened_j_handoff.rs` run
PBE with RI-J only (def2-SVP, def2-universal-jkfit, `energy_conv` $10^{-11}$,
`density_conv` $10^{-9}$) and are independent of the derivation above:

| Test | Check | Bar |
|---|---|---|
| `without_screening_reuse_is_bitwise_the_recompute` | butane at $\tau = 0$: reading the kept tensor and recomputing give the same bits | bitwise; the gradient is not near zero ($> 10^{-3}$) |
| `screened_reuse_matches_the_density_masked_recompute_on_water`, `..._on_butane` | $g_{\text{reuse}} - g_{\text{recompute}} = g_J(\tilde D) - g_J(D)$, the right side from the public recomputing `df_jk_gradient` and an independently built pair screen (`ri_j_pair_screen_for_test`). Water uses the first threshold of the ladder $10^{-3}, 3\times10^{-3}, \ldots, 3\times10^{-1}$ at which a pair drops; butane uses $10^{-5}$ | identity residual $< 10^{-11}$ Ha/Bohr; the difference itself above $10^{-10}$ (not vacuous) and below $4\cdot2\,r\max\lvert g_J(D)\rvert$, $r = \lVert D - \tilde D\rVert_F/\lVert D\rVert_F$ (a scaling bar with a safety factor of 4, not a theorem) |
| `dropped_pairs_move_d_p_by_less_than_thresh_times_their_weight` | butane, $\tau = 10^{-5}$: $\max_P\lvert d_P - \tilde d_P\rvert$ from the full tensor against $\tau\sum_{\text{dropped}}\lvert D_{\mu\nu}\rvert$ | the Schwarz bound, rigorous |
| `screened_gradient_is_the_derivative_of_the_screened_energy` | water: central difference ($h = 2\times10^{-3}$ Bohr, three coordinates) of the SCF energy run with the same screening | the screened route's error at most the recompute route's error plus twice the FD step floor $\lvert\mathrm{fd}(h)-\mathrm{fd}(h/2)\rvert$ plus $10^{-7}$ |

The test file records its artifact hypothesis: a stale or unscreened tensor
would make reuse equal recompute, which the identity test rules out because
the difference is nonzero by construction when pairs drop; zeroing the wrong
pairs would miss the identity by $O(r)$. The gradient bound of the expansion
above is a scaling statement; only the bound on $d_P$ is proved.

## Reproducibility and memory {#df-jk-gradient-reproducibility-memory}

- Both derivative passes map over auxiliary shells, collect in shell order and
  fold serially, so the gradient is bit-identical across thread counts.
- Resident beside one three-index block (at most
  `ferric-scf/src/df_gradient.rs::GRADIENT_3C_BLOCK_CAP_BYTES` and a quarter
  of the budget): $2\,N_{\mathrm{aux}}\,m_s^2$ doubles per exchange channel
  ($m_s$ the density rank, the occupied count), five
  $N_{\mathrm{aux}}\times N_{\mathrm{aux}}$ metric matrices, and one
  $(n_P, N_{\mathrm{bf}}, N_{\mathrm{bf}})$ weight slab per rayon worker. All are charged
  to the global memory pool before allocation.
- Under MPI the SCF stripes the auxiliary band across ranks, but this
  gradient does not: every rank computes the same full gradient, so it is
  correct and its memory does not shrink with rank count.

## Options that change the equation {#df-jk-gradient-options-change-equation}

| Option | Effect |
|---|---|
| RI-J precision | with f32 RI-J the fitted density $d_P$ is formed from the same rounded integrals the SCF stored, so $c$ is the SCF's own coefficient vector and the gradient is that of the rounded functional the SCF minimized. Derivative integrals and the metric derivative stay f64 |
| RI-K storage precision | RI-K is always differentiated from freshly computed f64 integrals, so an f32 SCF yields the gradient of the f64 functional at the f32 SCF's density |
| separate J and K auxiliary bases | two passes, one per basis; with a shared basis one pass handles both |
| range-separated exchange | two fitted channels with $\mathrm{erfc}(\omega r)$ and $\mathrm{erf}(\omega r)$ in both integrals and metric, weights $c_{\mathrm{SR}}$ and $c_{\mathrm{LR}}$, with the SCF's own auxiliary basis and $\omega$ |
| exact J or K | an exact term goes through the four-centre loops, so mixed routes (RI-J with exact K, the reverse) are differentiated term by term |

## Numerical anchors {#df-jk-gradient-numerical-anchors}

| Check | Reference | Agreement | Test |
|---|---|---|---|
| RI-J, RI-JK, UHF, ROHF, KS (PBE, B3LYP, UKS B3LYP, wB97X) gradients: water, OH, HO2 / cc-pVDZ, 6-31G, def2-universal-jkfit | central finite differences of ferric's own fitted energy, $h = 10^{-3}$ Bohr | bar $10^{-6}$ Ha/Bohr; each case also asserts that the unfitted four-centre derivative misses by at least half the smallest predicted defect | `crates/ferric-scf/tests/df_jk_gradient.rs` |
| RI-J and RI-JK RHF, UHF: cc-pVDZ | PySCF `df.grad` with the same auxiliary basis and auxiliary-basis response; the DF energies compared first to $10^{-7}$ Ha | about $10^{-10}$ Ha/Bohr, bar $10^{-6}$; the old pairing differs from PySCF by more than $5\times10^{-6}$ | `hf_ri_jk_gradient_matches_pyscf_df`, `uhf_ri_jk_gradient_matches_pyscf_df` in `crates/ferric-scf/tests/df_jk_gradient.rs` |

<!-- figures: site/src/reference/validation.md "RHF/UHF/ROHF/KS gradients, including density-fitted J/K" -->

KS cases carry a grid-integration residual that is not a density-fitting
effect, so they are checked differentially: the residual of the fitted run
must equal the residual of the same functional run with exact J/K. The grid
residual cancels; the density-fitting defect does not.

**Known limit.** The gradient is validated on small molecules in the bases
above. The noise-floor treatment of section @sec:df-jk-gradient-truncated-exchange-metric is a measured choice on
water, not a proven bound; the warning exists so a system where the band is
not negligible cannot pass silently.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/src/df_gradient.rs`, module documentation and `df_jk_gradient_core` | the stage list, the erf-metric mode count and the prototype measurements |
| `crates/ferric-scf/src/df_gradient/deriv3c.rs` | the three-centre contraction layout |
| `scripts/df_jk_gradient_sympy.py`, claims D1 to D3 | verified equation @eq:df-grad-jk |
| `crates/ferric-scf/tests/df_jk_gradient.rs` | thresholds and the negative controls |
| `crates/ferric-integrals/src/three_index_source/pair_screen.rs`, module documentation; `crates/ferric-scf/tests/df_gradient_screened_j_handoff.rs` | the pair screen, its danuglipron numbers, and the checks of section @sec:df-jk-gradient-pair-screened-ri-j |
| `site/src/reference/validation.md` | the gradient row |
