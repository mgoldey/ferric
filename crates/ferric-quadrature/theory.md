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

The grids are not in this crate. They are implemented in `ferric_rpa::quadrature` (`crates/ferric-rpa/src/quadrature.rs`) and selected by `build_quadrature` (`quadrature.rs:34`) from `QuadratureConfig.scheme`. Each returns frequencies $\omega_k$ and weights $w_k$ that already include the mapping Jacobian, for
$$ E_c = \frac{1}{2\pi} \int_0^\infty f(\omega)\, d\omega \approx \frac{1}{2\pi} \sum_k w_k f(\omega_k). $$

| Scheme (`[rpa] quadrature`) | Mapping | Code |
|---|---|---|
| Gauss-Legendre (`gauss-legendre`, `gl`) | $\omega = u_0 \dfrac{1+x}{1-x}$, $x_k$ Gauss-Legendre nodes on $(-1,1)$; $w_k = w_k^{GL}\, 2u_0/(1-x_k)^2$ | `gauss_legendre_nodes`, `quadrature.rs:89` |
| MiniMax (`minimax`, `mm`) | Gauss-Legendre map with $u_0$ taken from a table in $n$ (0.3 for $n\le 8$, 0.4 for $n\le 16$, 0.5 above; Furche, JCP 122, 164106, 2005); the user $u_0$ is ignored | `optimized_u0`, `quadrature.rs:101` |
| Chebyshev-Tan (`chebyshev-tan`, `chebyshev`, `ct`) | Eshuis-Yarkony-Furche (JCP 132, 234114, 2010), below | `chebyshev_tan_nodes`, `quadrature.rs:63` |

Scheme strings are parsed by `QuadratureScheme::parse_config_str` (`crates/ferric-rpa/src/config.rs:445`); unknown strings are an error.

### Chebyshev-Tan

Chebyshev-2 nodes
$$ x_k = \cos\frac{\pi k}{n+1}, \qquad k = 1,\dots,n, $$
are mapped to the bounded range $\omega \in (0,\, u_0\tan(\pi/2))$ by
$$ \omega_k = u_0 \tan\!\left(\frac{\pi(1+x_k)}{4}\right). $$
The weight combines the Chebyshev-2 weight $\frac{\pi}{n+1}\sin^2\theta_k$ (with $\theta_k = \pi k/(n+1)$), the factor $1/\sqrt{1-x_k^2} = 1/\sin\theta_k$ and the Jacobian $\frac{u_0\pi}{4}\sec^2\!\left(\frac{\pi(1+x_k)}{4}\right)$:
$$ w_k = \frac{\pi}{n+1}\, \sin\theta_k\, \frac{u_0 \pi}{4}\, \sec^2\!\left(\frac{\pi(1+x_k)}{4}\right). $$
The finite maximum $\omega$ keeps $\omega\, t_{\max}$ in the safe range of the cosine-modulated Laplace $\chi_0$ backend. The recommended $u_0$ with that backend is `CHEBYSHEV_TAN_RECOMMENDED_U0` = 0.2.

### Defaults

$u_0 = 0.5$ Hartree and $n = 20$ points, except where an entry point overrides them. The default scheme depends on the entry point: the CLI and the Python bindings use Gauss-Legendre, the Rust `QuadratureConfig::default()` uses MiniMax. The table is in the `QuadratureConfig` doc comment (`config.rs:393`) and is pinned by the test `quadrature_defaults_by_entry_point`.

---

## 3. Lebedev Angular Grids

`ferric_quadrature::lebedev` (`src/lebedev.rs`) holds the Lebedev angular quadrature rules used for DFT grids and the PCM cavity. `ferric_dft::lebedev` re-exports it.
