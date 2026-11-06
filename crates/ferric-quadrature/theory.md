# ferric-quadrature Mathematical Foundations

This module provides numerical quadrature schemes for integration over real and imaginary frequency grids.

---

## 1. Minimax Laplace Quadrature

The canonical MP2 and RPA correlation calculations involve denominators of the form $1/x$ (where $x = \epsilon_a + \epsilon_b - \epsilon_i - \epsilon_j$ represents orbital energy differences). Minimax approximation maps this to a separable sum of exponentials:
$$ \frac{1}{x} \approx \sum_{l=1}^{n_q} w_l e^{-t_l x} \quad \text{for } x \in [1, R] $$
where $R$ is the ratio between the maximum and minimum orbital energy gaps.

### Point Selection & Scaling
* The quadrature points $t_l$ and weights $w_l$ are pre-computed minimax coefficients.
* `ferric` retrieves these parameters dynamically from literature tables (Takatsuka/Ten-no/Hackbusch 2008).
* In imaginary-time RPA, this decomposes the independent-particle polarizability $\chi_0(i\omega)$ into occupied and virtual products. The formulation reaches $\mathcal{O}(N^3)$ only with AO sparsity; `crates/ferric-rpa/src/ao_rpa.rs` implements the MO-basis version, which has the dense $\mathcal{O}(N^4)$ scaling.

---

## 2. Imaginary-Frequency Grids for RPA

The grids are not in this crate. They are implemented in `ferric_rpa::quadrature` (`crates/ferric-rpa/src/quadrature.rs`) and selected by `build_quadrature` from `QuadratureConfig.scheme`. Each returns frequencies $\omega_k$ and weights $w_k$ that already include the mapping Jacobian, for
$$ E_c = \frac{1}{2\pi} \int_0^\infty f(\omega)\, d\omega \approx \frac{1}{2\pi} \sum_k w_k f(\omega_k). $$

| Scheme (`[rpa] quadrature`) | Mapping | Code |
|---|---|---|
| Gauss-Legendre (`gauss-legendre`, `gl`) | $\omega = u_0 \dfrac{1+x}{1-x}$, $x_k$ Gauss-Legendre nodes on $(-1,1)$; $w_k = w_k^{GL}\, 2u_0/(1-x_k)^2$ | `gauss_legendre_nodes` |
| MiniMax (`minimax`, `mm`) | Gauss-Legendre map with $u_0$ taken from a table in $n$ (0.3 for $n\le 8$, 0.4 for $n\le 16$, 0.5 above; Furche, JCP 122, 164106, 2005); the user $u_0$ is ignored | `optimized_u0` |
| Chebyshev-Tan (`chebyshev-tan`, `chebyshev`, `ct`) | Eshuis-Yarkony-Furche (JCP 132, 234114, 2010), below | `chebyshev_tan_nodes` |

Scheme strings are parsed by `QuadratureScheme::parse_config_str` (`crates/ferric-rpa/src/config.rs`); unknown strings are an error.

### Chebyshev-Tan

Chebyshev-2 nodes
$$ x_k = \cos\frac{\pi k}{n+1}, \qquad k = 1,\dots,n, $$
are mapped to $\omega \in (0, \infty)$ by
$$ \omega_k = u_0 \tan\!\left(\frac{\pi(1+x_k)}{4}\right). $$
The weight combines the Chebyshev-2 weight $\frac{\pi}{n+1}\sin^2\theta_k$ (with $\theta_k = \pi k/(n+1)$), the factor $1/\sqrt{1-x_k^2} = 1/\sin\theta_k$ and the Jacobian $\frac{u_0\pi}{4}\sec^2\!\left(\frac{\pi(1+x_k)}{4}\right)$:
$$ w_k = \frac{\pi}{n+1}\, \sin\theta_k\, \frac{u_0 \pi}{4}\, \sec^2\!\left(\frac{\pi(1+x_k)}{4}\right). $$
In $\theta$ this is the trapezoid rule on an integrand that vanishes like $\sin\theta$ at both ends, so the error falls as $O(n^{-2})$.

The largest node is at $k = 1$:
$$ \omega_{\max} = u_0 \cot\!\left(\frac{\pi}{2}\sin^2\frac{\pi}{2(n+1)}\right). $$
It is finite at every $n$ and grows as $n^2$, as the Gauss-Legendre map's largest node does. At equal $(n, u_0)$ it is about 0.4 of the Gauss-Legendre value: 57 vs 145 Hartree at $n = 20$, $u_0 = 0.5$.

The Laplace $\chi_0$ backend (`crates/ferric-rpa/src/laplace_chi0.rs`) uses the dense kernel at any node with $\omega\, t_{\max} > \pi/2$. Chebyshev-Tan sends fewer nodes to that fallback than Gauss-Legendre. That backend is a library option (`Chi0Backend::Laplace`); the CLI and the Python bindings always use `Chi0Backend::Dense`, where the scheme choice changes only quadrature error. The recommended $u_0$ with that backend is `CHEBYSHEV_TAN_RECOMMENDED_U0` = 0.2.

Tests in `crates/ferric-rpa/src/quadrature.rs`:

| Test | Checks |
|---|---|
| `chebyshev_tan_lorentzian_at_a_equals_u0_matches_closed_form` | For $f = u_0/(u_0^2+\omega^2)$ the Jacobian cancels $f$, so the rule equals $\frac{\pi}{4}\frac{\pi}{n+1}\cot\frac{\pi}{2(n+1)}$; asserted to $10^{-13}$ for $u_0 \in \{0.2, 0.5, 1.3\}$, $n \in \{1, 4, 14, 20, 64\}$ |
| `chebyshev_tan_lorentzian_error_falls_as_n_to_the_minus_two` | $\int_0^\infty a/(a^2+\omega^2)\,d\omega = \pi/2$ at $u_0 = 0.5$, $a \in \{0.3, 3\}$: error ratio 3.5 to 4.5 on each doubling $n = 20 \to 40 \to 80$ |
| `chebyshev_tan_largest_node_matches_closed_form` | $\omega_{\max}$ against the closed form to $10^{-12}$ relative, and its ratio to the Gauss-Legendre largest node in 0.3 to 0.5 |
| `closed_shell_rpa_laplace_chebyshev_matches_dense_h2o` (`crates/ferric-rpa/tests/u_pdep_rpa.rs`) | RPA correlation energy, H2O / cc-pVDZ: Chebyshev-Tan ($n = 20$, $u_0 = 0.5$) with the Laplace $\chi_0$ backend against Gauss-Legendre with the dense backend, within $5\times10^{-4}$ Hartree |

### Defaults

$u_0 = 0.5$ Hartree and $n = 20$ points, except where an entry point overrides them. The default scheme is Gauss-Legendre on the CLI, the Python bindings and the Rust `QuadratureConfig::default()`; only the CLI optimize task (16 points) differs. The table is in the `QuadratureConfig` doc comment and is pinned by the test `quadrature_defaults_by_entry_point`.

---

## 3. Lebedev Angular Grids

`ferric_quadrature::lebedev` (`src/lebedev.rs`) holds the Lebedev angular quadrature rules used for DFT grids and the PCM cavity. `ferric_dft::lebedev` re-exports it.
