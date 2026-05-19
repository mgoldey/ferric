# ferric-quadrature Mathematical Foundations

This module provides high-accuracy numerical quadrature schemes for integration over real and imaginary frequency grids.

---

## 1. Minimax Laplace Quadrature

The canonical MP2 and RPA correlation calculations involve denominators of the form $1/x$ (where $x = \epsilon_a + \epsilon_b - \epsilon_i - \epsilon_j$ represents orbital energy differences). Minimax approximation maps this to a separable sum of exponentials:
$$ \frac{1}{x} \approx \sum_{l=1}^{n_q} w_l e^{-t_l x} \quad \text{for } x \in [1, R] $$
where $R$ is the ratio between the maximum and minimum orbital energy gaps.

### Point Selection & Scaling
* The quadrature points $t_l$ and weights $w_l$ are pre-computed minimax coefficients.
* `ferric` retrieves these parameters dynamically from literature tables (Takatsuka/Ten-no/Hackbusch 2008).
* In PDEP-RPA, this allows the independent-particle polarizability matrix $\chi_0(i\omega)$ to be decomposed into occupied and virtual products, reducing the computational scaling from $\mathcal{O}(N^4)$ to $\mathcal{O}(N^3)$ (see `docs/pdep-boys-laplace-scaling.md`).

---

## 2. Chebyshev-Tan (Eshuis) Frequency Mappings

For imaginary frequency integrations over the infinite domain $[0, \infty)$ in RPA:
$$ E_c = \frac{1}{2\pi} \int_0^\infty f(\omega) d\omega $$
`ferric` implements the Chebyshev-Tan coordinate transformation:
$$ \omega = u_0 \tan\left( \frac{\pi}{2} y \right) $$
which maps $y \in [0, 1)$ to $\omega \in [0, \infty)$. Gauss-Legendre or Chebyshev quadrature is then applied over the mapped domain. The parameter $u_0$ acts as a scaling factor, typically optimized to match the typical valence excitation gap of the molecular system (default: `0.5` Hartree).
