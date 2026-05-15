# Density Functional Theory (DFT)

This module implements the Kohn-Sham DFT method with numerical integration on a grid.

## Kohn-Sham Equations
The DFT energy is:
$$E_{DFT} = T_s + V_{ext} + J + E_{xc}$$
where $E_{xc}$ is the exchange-correlation functional.

## Numerical Integration
The exchange-correlation energy and potential are computed on a grid of points $\{\mathbf{r}_k, w_k\}$:
$$E_{xc} = \int \rho(\mathbf{r}) \epsilon_{xc}(\rho, \nabla \rho, \dots) d\mathbf{r} \approx \sum_k w_k \rho(\mathbf{r}_k) \epsilon_{xc}(\rho_k, \gamma_k, \dots)$$
where $\gamma = |\nabla \rho|^2$.

## Functionals
We support LDA and GGA functionals via an interface to `libxc`.
- **LDA**: $\epsilon_{xc} = \epsilon_{xc}(\rho)$
- **GGA**: $\epsilon_{xc} = \epsilon_{xc}(\rho, \gamma)$

## Integration Grid
The grid is typically a combination of a radial grid (e.g., Mura-Knowles or Lebedev) and an angular grid (Lebedev-Laikov), with pruning to reduce the number of points in low-density regions.
