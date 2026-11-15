# 18.2 Imaginary-frequency quadratures

<!-- section: 18.2; status: edit; registry: none;
     theory: crates/ferric-quadrature/theory.md#"2. Imaginary-Frequency Grids for RPA" -->

Every frequency-route RPA energy in ferric (sections 18.1, 18.3, 18.4), the
GW self-energy and the dynamic polarizabilities integrate a function of the
imaginary frequency $u$ over $[0, \infty)$. This section describes the three
grids, their defaults, and how the default order of the exact dRPA route was
chosen.

| Symbol | Meaning | Code name |
|---|---|---|
| $u$ | imaginary frequency, Hartree | `omega` in the code |
| $u_k, w_k$ | nodes and weights; $w_k$ includes the Jacobian | `freqs`, `weights` |
| $u_0$ | scale of the map, Hartree | `u0` |
| $n$ | number of nodes | `n_points` |

The code names the imaginary frequency `omega`. The book writes $u$, so that
$\omega$ stays the range-separation parameter (chapter 0, section 8).

| Entry point | Location |
|---|---|
| Rust | `ferric-rpa/src/quadrature.rs::{build_quadrature,gauss_legendre_nodes,chebyshev_tan_nodes}` |
| CLI | `[rpa] quadrature`, `n_quad`, `u0` |
| Python | `quadrature=`, `n_quad=`, `u0=` of `run_pdep_rpa`, `run_gw` and the other frequency-route functions |

## 18.2.1 The rule

$$ \int_0^\infty f(u)\,du \approx \sum_{k=1}^{n} w_k\,f(u_k), \qquad E_c = \frac{1}{2\pi}\sum_k w_k\,S(u_k) . \tag{18.6} $$

All three grids map a rule on a finite interval to $[0, \infty)$, and the
weights returned by `build_quadrature` already carry the Jacobian.

| Scheme (`[rpa] quadrature`) | Nodes | Weights | Function |
|---|---|---|---|
| Gauss–Legendre (`gauss-legendre`, `gl`) | $u_k = u_0\,\dfrac{1 + x_k}{1 - x_k}$, $x_k$ Gauss–Legendre on $(-1, 1)$ | $w_k^{\mathrm{GL}}\,\dfrac{2u_0}{(1 - x_k)^2}$ | `gauss_legendre_nodes` |
| MiniMax (`minimax`, `mm`) | Gauss–Legendre map with $u_0 = 0.3$ ($n \le 8$), 0.4 ($n \le 16$), 0.5 (above); the user $u_0$ is ignored | as Gauss–Legendre | `build_quadrature` |
| Chebyshev–Tan (`chebyshev-tan`, `chebyshev`, `ct`) | $u_k = u_0\tan\big(\pi(1 + x_k)/4\big)$, $x_k = \cos\theta_k$, $\theta_k = \pi k/(n+1)$ | $\dfrac{\pi}{n+1}\sin\theta_k\,\dfrac{u_0\pi}{4}\sec^2\!\Big(\dfrac{\pi(1 + x_k)}{4}\Big)$ | `chebyshev_tan_nodes` |

The scheme named MiniMax is not a minimax grid. It is the Gauss–Legendre map
with $u_0$ taken from a three-entry table (Furche 2005), and the name is kept
because it is the configuration string.

Chebyshev–Tan (Eshuis, Yarkony and Furche 2010) is the trapezoid rule in
$\theta$, so its error on a smooth integrand falls as $n^{-2}$. Its largest
node is $u_0\cot\big(\tfrac\pi2\sin^2\tfrac{\pi}{2(n+1)}\big)$, about 0.4 of the
Gauss–Legendre map's largest node at the same $(n, u_0)$: 57 against 145
Hartree at $n = 20$, $u_0 = 0.5$. Both grow as $n^2$.

The grid depends only on $(n, u_0)$ and the scheme, not on the molecule's
orbital energies.

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
| $w_k$ | `weights` |

> **Unverified equation 18.6** · the mapped weights
>
> The Jacobians of the Gauss–Legendre and Chebyshev–Tan maps in the table
> above.
>
> | | |
> |---|---|
> | Proof | none in a script. The Rust unit tests in `crates/ferric-rpa/src/quadrature.rs` check the Chebyshev–Tan rule against the closed form for a Lorentzian whose scale equals $u_0$ (to $10^{-13}$), its $n^{-2}$ error, and its largest node; check `gl48-vs-definition` of `scripts/rpa_drpa_pdep_sympy.py` checks the 48-point Gauss–Legendre grid only through a dRPA energy. |
> | Status | tested in Rust, no derivation script |
>
> <!-- unverified: id=rpa.quad.mapped-weights; todo=add a check to scripts/rpa_drpa_pdep_sympy.py that integrates closed-form test functions with both maps -->

## 18.2.2 Defaults

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
40 by up to $9.4\times10^{-10}$; at 48 every row is within
$2.7\times10^{-10}$; from 64 on, rounding grows past the quadrature error.
The test asserts the chosen order.

**$u_0$.** Only $u_0 = 0.5$ has been calibrated on the production (dense
$\chi_0$) path; there the scheme changes only the quadrature error. The
Laplace $\chi_0$ backend (section 18.5) falls back to the dense kernel at any
node with $u\,t_{\max} > \pi/2$, so a smaller $u_0$ keeps more nodes on the
Laplace path; `CHEBYSHEV_TAN_RECOMMENDED_U0 = 0.2` is calibrated for that
backend only.

## 18.2.3 Numerical anchors

| Quantity | Value | Source |
|---|---|---|
| Chebyshev–Tan Lorentzian closed form | $10^{-13}$ | `chebyshev_tan_lorentzian_at_a_equals_u0_matches_closed_form` in `crates/ferric-rpa/src/quadrature.rs` |
| Chebyshev–Tan error per doubling of $n$ | ratio 3.5 to 4.5 | `chebyshev_tan_lorentzian_error_falls_as_n_to_the_minus_two` |
| Exact dRPA, 48 Gauss–Legendre nodes | $\le 2.7\times10^{-10}$ Ha | `crates/ferric-rpa/tests/validation_drpa_frequency.rs` |

## 18.2.4 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-quadrature/theory.md`, "2. Imaginary-Frequency Grids for RPA" | the three maps, the largest-node formula, the Laplace fallback counts |
| `crates/ferric-rpa/src/config.rs` | the defaults table |

**Known limits (2026-11-15).** No grid adapts to the orbital-energy range of
the system. The same $(n, u_0)$ is used for a small-gap and a large-gap
molecule, so the 48-node choice holds for the systems in the convergence
table and is not a bound elsewhere.
