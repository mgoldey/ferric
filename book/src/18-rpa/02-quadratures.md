# Imaginary-frequency quadratures {#quadratures}

<!-- status: ready; registry: pdep-rpa -->

The frequency form of the RPA energy (@eq:drpa-three-ways-4) is an integral over the whole
half-line $u \in [0, \infty)$. The integrand is smooth, largest near
$u = 0$, changes on the scale of the orbital-energy gaps, and falls as
$u^{-4}$ at large $u$ (section @sec:drpa-three-ways-imaginary-frequency). Each node costs one dielectric
matrix and one factorization, so the grid has to capture the integral with
a few dozen points. Every frequency-route RPA energy in ferric (sections
@sec:drpa-three-ways, @sec:pdep, @sec:open-shell-dielectric), the GW self-energy and the dynamic polarizabilities use
the grids of this section.

The plan: the three grids (@sec:quadratures-rule), their defaults and how the order of the
exact dRPA route was chosen (@sec:quadratures-defaults), then anchors and sources.

| Symbol | Meaning | Code name |
|---|---|---|
| $u$ | imaginary frequency, Hartree | `omega` in the code |
| $u_k, w_k$ | nodes and weights; $w_k$ includes the Jacobian | `freqs`, `weights` |
| $u_0$ | scale of the map, Hartree | `u0` |
| $n$ | number of nodes | `n_points` |

The code names the imaginary frequency `omega`. The book writes $u$, so that
$\omega$ stays the range-separation parameter (chapter @sec:notation, section @sec:notation-attenuated-coulomb-operators).

| Entry point | Location |
|---|---|
| Rust | `ferric-rpa/src/quadrature.rs::{build_quadrature,gauss_legendre_nodes,chebyshev_tan_nodes}` |
| CLI | `[rpa] quadrature`, `n_quad`, `u0` |
| Python | `quadrature=`, `n_quad=`, `u0=` of `run_pdep_rpa`, `run_gw` and the other frequency-route functions |

## The rule {#quadratures-rule}

$$ \int_0^\infty f(u)\,du \approx \sum_{k=1}^{n} w_k\,f(u_k), \qquad E_c = \frac{1}{2\pi}\sum_k w_k\,S(u_k) . \label{eq:rpa-quad-mapped-weights} $$

A Gaussian rule lives on a finite interval. All three grids map a rule on
$x \in (-1, 1)$ to $u \in (0, \infty)$ and multiply each weight by the
Jacobian $du/dx$ at its node. The weights returned by `build_quadrature`
already carry that Jacobian. `build_quadrature` picks the grid with a `match`
on `QuadratureConfig.scheme`, a `QuadratureScheme`; the configuration string
is parsed by `ferric-rpa/src/config.rs::QuadratureScheme` (its
`parse_config_str`), which accepts the forms in the first column below,
case-insensitive, and returns an error for any other string instead of
falling back to Gauss–Legendre.

| Scheme (`[rpa] quadrature`) | Nodes | Weights | Function |
|---|---|---|---|
| Gauss–Legendre (`gauss-legendre`, `gl`) | $u_k = u_0\,\dfrac{1 + x_k}{1 - x_k}$, $x_k$ Gauss–Legendre on $(-1, 1)$ | $w_k^{\mathrm{GL}}\,\dfrac{2u_0}{(1 - x_k)^2}$ | `gauss_legendre_nodes` |
| MiniMax (`minimax`, `mm`) | Gauss–Legendre map with $u_0 = 0.3$ ($n \le 8$), 0.4 ($9 \le n \le 16$), 0.5 ($n > 16$); the user $u_0$ is ignored | as Gauss–Legendre | `build_quadrature`, `optimized_u0` |
| Chebyshev–Tan (`chebyshev-tan`, `chebyshev`, `ct`) | $u_k = u_0\tan\big(\pi(1 + x_k)/4\big)$, $x_k = \cos\theta_k$, $\theta_k = \pi k/(n+1)$ | $\dfrac{\pi}{n+1}\sin\theta_k\,\dfrac{u_0\pi}{4}\sec^2\!\Big(\dfrac{\pi(1 + x_k)}{4}\Big)$ | `chebyshev_tan_nodes` |

**The Gauss–Legendre map.** Differentiate $u = u_0(1 + x)/(1 - x)$ by the
quotient rule: $du/dx = u_0\big[(1 - x) + (1 + x)\big]/(1 - x)^2 = 2u_0/(1 - x)^2$.
The midpoint $x = 0$ maps to $u = u_0$.
The rule on $(-1, 1)$ comes from `ferric-rpa/src/quadrature.rs::gl_nodes_weights`:
Newton's method on the three-term recurrence of the Legendre polynomial
$P_n$, started from $x^{(0)} = \cos\big(\pi(4i + 3)/(4n + 2)\big)$ for the
$i$-th root, with weight $w^{\mathrm{GL}} = 2/\big((1 - x^2)P_n'(x)^2\big)$ at
each root. The Gauss–Legendre nodes are
symmetric about $x = 0$, so half of the nodes lie below $u_0$ and half
above. $u_0$ is the frequency that splits the grid in two.

The mapped rule integrates exactly every $f$ whose pull-back
$f(u(x))\,du/dx$ is a polynomial in $x$ of degree $\le 2n - 1$. An example
is $f(u) = x(u)^m\,2u_0/(u + u_0)^2$ with $x(u) = (u - u_0)/(u + u_0)$: since
$1 - x = 2u_0/(u + u_0)$, $du/dx = (u + u_0)^2/(2u_0)$ and the pull-back is
$x^m$, integrated exactly for $m \le 2n - 1$. Check the
tail. Near $x = 1$, $u \approx 2u_0/(1 - x)$, and the pull-back of an
integrand falling as $u^{-4}$ is proportional to $(1 - x)^2$. It goes to
zero smoothly at the end of the interval, so the mapped rule meets no
singularity at $x = 1$.

**MiniMax** is not a minimax grid. It is the Gauss–Legendre map with $u_0$
taken from a three-entry table, `ferric-rpa/src/quadrature.rs::optimized_u0`
(the values documented as Furche, J. Chem. Phys. 122, 164106 (2005)). The
name is kept because it is the configuration string. The configured $u_0$ is
not read: `QuadratureScheme::honours_u0` is false for MiniMax, and callers
use it to warn instead of dropping a user setting silently.

**Chebyshev–Tan** (Eshuis, Yarkony and Furche, J. Chem. Phys. 132, 234114 (2010)) starts from Chebyshev
nodes of the second kind. Its weight is the Chebyshev-2 weight
$\frac{\pi}{n+1}\sin^2\theta_k$, times $1/\sqrt{1 - x_k^2} = 1/\sin\theta_k$,
times the Jacobian $\frac{u_0\pi}{4}\sec^2\frac{\pi(1 + x_k)}{4}$.
Substitute $x = \cos\theta$ and the rule is the trapezoid rule in $\theta$
with step $\pi/(n + 1)$; the endpoints carry zero weight. So its error on a
smooth integrand falls as $n^{-2}$. For the Lorentzian $f = u_0/(u_0^2 + u^2)$ the
Jacobian cancels $f$ exactly: with $u = u_0\tan\phi$, $f = 1/(u_0\sec^2\phi)$,
so $f\,du/dx = \pi/4$. The rule is then $\frac{\pi^2}{4(n+1)}\sum_k\sin\theta_k$, and
the sum of sines gives the closed form the tests use,

$$ \sum_k w_k f(u_k) = \frac{\pi}{4}\,\frac{\pi}{n+1}\cot\frac{\pi}{2(n+1)} \quad\text{at every } n . $$

Its largest node is at $k = 1$. Rewrite the tangent as a cotangent of the
complementary angle,
$\tan\frac{\pi(1 + \cos\theta)}{4} = \cot\frac{\pi(1 - \cos\theta)}{4} = \cot\big(\tfrac\pi2\sin^2\tfrac\theta2\big)$,
which gives $u_0\cot\big(\tfrac\pi2\sin^2\tfrac{\pi}{2(n+1)}\big)$. That is
about 0.4 of the Gauss–Legendre map's largest node at the same $(n, u_0)$: 57
against 145 Hartree at $n = 20$, $u_0 = 0.5$. Both grow as $n^2$.

The grid depends only on $(n, u_0)$ and the scheme, not on the molecule's
orbital energies.

The Gauss–Legendre map in the code, unchanged:

<!-- excerpt: ferric-rpa/src/quadrature.rs::gauss_legendre_nodes -->
```rust
let (x, w) = gl_nodes_weights(n);
let freqs: Vec<f64> = x.iter().map(|&xi| u0 * (1.0 + xi) / (1.0 - xi)).collect();
let weights: Vec<f64> = x
    .iter()
    .zip(w.iter())
    .map(|(&xi, &wi)| wi * 2.0 * u0 / (1.0 - xi).powi(2))
    .collect();
```

| Symbol | Code |
|---|---|
| $x_k, w_k^{\mathrm{GL}}$ | `x`, `w` |
| $u_k$ | `freqs` |
| $w_k = w_k^{\mathrm{GL}}\,du/dx$ | `weights` |

> **Verified equation @eq:rpa-quad-mapped-weights** · the mapped weights
>
> The Gauss–Legendre and Chebyshev–Tan weights are the rule's weights times
> the Jacobian $du/dx$ of the map; Chebyshev–Tan is the trapezoid rule in
> $\theta$. The mapped Gauss–Legendre rule integrates exactly every $f$ whose
> pull-back $f(u(x))\,du/dx$ is a polynomial of degree $\le 2n - 1$, and
> MiniMax is the Gauss–Legendre map at the tabulated $u_0$.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_quadrature_sympy.py`, claims Q1 (Jacobians), Q2 (trapezoid form), Q3 (exactness anchor), Q4 (Lorentzian closed form, $n^{-2}$), Q5 (closed-form RPA integrands), Q6 (MiniMax) |
> | Negative control | the pull-back of degree $2n$ must miss; the runner drops the factor 2 of the Gauss–Legendre Jacobian, squares $\sin\theta$ in the Chebyshev–Tan weight, doubles its $u_0\pi/4$, and changes the MiniMax table |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/quadrature.rs::{gauss_legendre_nodes,chebyshev_tan_nodes,build_quadrature}` |
> | Code test | `chebyshev_tan_lorentzian_at_a_equals_u0_matches_closed_form` in `crates/ferric-rpa/src/quadrature.rs` |
>
> <!-- verified: id=rpa.quad.mapped-weights; script=scripts/rpa_quadrature_sympy.py; claim=Q3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/quadrature.rs::{gauss_legendre_nodes,chebyshev_tan_nodes,build_quadrature}; rust_test=ferric-rpa/src/quadrature.rs::chebyshev_tan_lorentzian_at_a_equals_u0_matches_closed_form; registry=pdep-rpa -->

The proof does not cover the half-below-$u_0$ split, the tail pull-back or
the largest-node formula written out above; the last is pinned by the test
`chebyshev_tan_largest_node_matches_closed_form`.

## Defaults {#quadratures-defaults}

| Entry point | Scheme | $n$ | $u_0$ |
|---|---|---|---|
| Library `QuadratureConfig::default()`, CLI `[rpa]` unset, Python `run_pdep_rpa`, `run_gw` | Gauss–Legendre | 20 | 0.5 |
| CLI `task = "optimize"` with `pdep-rpa` | Gauss–Legendre | 16 | 0.5 |
| Exact dRPA (`exact_drpa_config`; CLI `drpa`) | Gauss–Legendre | 48 | 0.5 |

The table is the doc comment of `QuadratureConfig` in
`crates/ferric-rpa/src/config.rs`, pinned by its test
`quadrature_defaults_by_entry_point`.

**Why 48 for exact dRPA.** The order was read off a convergence table against
independent plasmon references, Gauss–Legendre at $u_0 = 0.5$ with 24 to 200
nodes (`drpa_frequency_quadrature_convergence_table` in
`crates/ferric-rpa/tests/validation_drpa_frequency.rs`). With every orbital
correlated, 32 nodes miss by up to $1.3\times10^{-8}$ Ha (water / cc-pVDZ) and
40 by up to $9.4\times10^{-10}$. At 48 every row is within
$2.7\times10^{-10}$. From 64 on, rounding grows past the quadrature error, so
more nodes stop helping. The test asserts the chosen order.

**$u_0$.** Only $u_0 = 0.5$ has been calibrated on the production (dense
$\chi_0$) path; there the scheme changes only the quadrature error. The
Laplace $\chi_0$ backend (section @sec:laplace-ao-chi0) falls back to the dense kernel at any
node with $u\,t_{\max} > \pi/2$. A smaller $u_0$ moves nodes to smaller $u$,
so it keeps more of them on the Laplace path.
`CHEBYSHEV_TAN_RECOMMENDED_U0 = 0.2` is calibrated for that backend only, and
costs a coarser high-frequency tail.

At equal $(n, u_0)$, Chebyshev–Tan and Gauss–Legendre send about the same
number of nodes to the fallback, and which sends more depends on $t_{\max}$.
At $n = 20$:

| $t_{\max}$ | $u_0$ | Nodes on the Laplace path, Chebyshev–Tan | Gauss–Legendre |
|---|---|---|---|
| $\approx 1.5$ (H2O / cc-pVDZ) | 0.2 | 16 | 15 |
| $\approx 1.5$ (H2O / cc-pVDZ) | 0.5 | 13 | 12 |

At $t_{\max} = 2\pi$ the order reverses: Chebyshev–Tan sends one node more to
the fallback, 13 against 12 at $u_0 = 0.5$. Lowering $u_0$ moves more nodes
onto the Laplace path than the choice of scheme does. The CLI and the Python
bindings always use `Chi0Backend::Dense`; `Chi0Backend::Laplace`
(`crates/ferric-rpa/src/laplace_chi0.rs`) is a library option.

## Numerical anchors {#quadratures-numerical-anchors}

| Quantity | Value | Source |
|---|---|---|
| Chebyshev–Tan Lorentzian closed form, $u_0 \in \{0.2, 0.5, 1.3\}$, $n \in \{1, 4, 14, 20, 64\}$ | $10^{-13}$ | `chebyshev_tan_lorentzian_at_a_equals_u0_matches_closed_form` in `crates/ferric-rpa/src/quadrature.rs` |
| Chebyshev–Tan error per doubling of $n$: $\int_0^\infty a/(a^2 + u^2)\,du = \pi/2$ at $u_0 = 0.5$, $a \in \{0.3, 3\}$, $n = 20 \to 40 \to 80$ | ratio 3.5 to 4.5 | `chebyshev_tan_lorentzian_error_falls_as_n_to_the_minus_two` |
| Chebyshev–Tan largest node against its closed form; ratio to the Gauss–Legendre largest node | $10^{-12}$ relative; ratio in 0.3 to 0.5 | `chebyshev_tan_largest_node_matches_closed_form` |
| RPA correlation energy, H2O / cc-pVDZ: Chebyshev–Tan ($n = 20$, $u_0 = 0.5$) with the Laplace $\chi_0$ backend against Gauss–Legendre with the dense backend | within $5\times10^{-4}$ Ha | `closed_shell_rpa_laplace_chebyshev_matches_dense_h2o` in `crates/ferric-rpa/tests/u_pdep_rpa.rs` |
| Exact dRPA, 48 Gauss–Legendre nodes | $\le 2.7\times10^{-10}$ Ha | `crates/ferric-rpa/tests/validation_drpa_frequency.rs` |

A ratio of 4 per doubling is the $n^{-2}$ law; the test allows 3.5 to 4.5.

Claim Q5 of `scripts/rpa_quadrature_sympy.py` integrates two closed forms
with every scheme: $\int_0^\infty a/(a^2 + u^2)\,du = \pi/2$ and the one-mode
dRPA integral $\tfrac12\big(\sqrt{\Delta(\Delta + 2b)} - \Delta - b\big)$, the
latter reproduced to rounding by Gauss–Legendre at 48 nodes. Claim Q6 checks
that MiniMax returns the Gauss–Legendre map at the tabulated $u_0$ whatever
$u_0$ is configured.

## Sources {#quadratures-sources}

| Source | What it gives |
|---|---|
| `scripts/rpa_quadrature_sympy.py`, claims Q1 to Q6 | the proof |
| `crates/ferric-rpa/src/config.rs` | the defaults table |

**Known limits (2026-11-15).** No grid adapts to the orbital-energy range of
the system. The same $(n, u_0)$ is used for a small-gap and a large-gap
molecule, so the 48-node choice holds for the systems in the convergence
table and is not a bound elsewhere.
