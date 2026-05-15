# Molecular Integrals Theory

This module provides the core integral engine for `ferric`, utilizing `libint2` for high-performance evaluation of Gaussian-type orbital (GTO) integrals.

## Basis Sets
Orbitals are expanded in a basis of Gaussian-type functions:
$$\chi_\mu(\mathbf{r}) = \sum_i d_i e^{-\alpha_i |\mathbf{r} - \mathbf{R}|^2}$$

## One-Electron Integrals
- **Overlap**: $S_{\mu\nu} = \int \chi_\mu^*(\mathbf{r}) \chi_\nu(\mathbf{r}) d\mathbf{r}$
- **Kinetic Energy**: $T_{\mu\nu} = \int \chi_\mu^*(\mathbf{r}) \left( -\frac{1}{2} \nabla^2 \right) \chi_\nu(\mathbf{r}) d\mathbf{r}$
- **Nuclear Attraction**: $V_{\mu\nu} = \int \chi_\mu^*(\mathbf{r}) \left( -\sum_A \frac{Z_A}{|\mathbf{r} - \mathbf{R}_A|} \right) \chi_\nu(\mathbf{r}) d\mathbf{r}$

## Two-Electron Repulsion Integrals (ERI)
The 4-center ERIs in Mulliken notation:
$$(\mu\nu|\lambda\sigma) = \iint \frac{\chi_\mu^*(\mathbf{r}_1) \chi_\nu(\mathbf{r}_1) \chi_\lambda^*(\mathbf{r}_2) \chi_\sigma(\mathbf{r}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{r}_1 d\mathbf{r}_2$$

## Three-Center Integrals
Used in Resolution of Identity (RI) methods:
$$(P|\mu\nu) = \iint \frac{\chi_P^*(\mathbf{r}_1) \chi_\mu(\mathbf{r}_2) \chi_\nu(\mathbf{r}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{r}_1 d\mathbf{r}_2$$
where $P$ is an auxiliary basis function.

## Two-Center Metric
$$(P|Q) = \iint \frac{\chi_P^*(\mathbf{r}_1) \chi_Q(\mathbf{r}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{r}_1 d\mathbf{r}_2$$
This matrix is used to define the RI transformation $B^P_{\mu\nu}$.
