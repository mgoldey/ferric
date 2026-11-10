# Molecular Integrals Theory

This module provides the core integral engine for `ferric`, using `libint2` to evaluate Gaussian-type orbital (GTO) integrals.

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

## Effective Core Potentials
`ecp.rs` evaluates the semi-local ECP operator on centre $C$,

$$\hat U = U_L(r) + \sum_{l=0}^{L-1}\sum_{m=-l}^{l} |lm\rangle\,\bigl[U_l(r) - U_L(r)\bigr]\,\langle lm|,\qquad U_l(r) = \sum_t d_t\, r^{n_t-2} e^{-\zeta_t r^2},$$

with $r = |\mathbf r - \mathbf C|$, through one of two backends (`FERRIC_ECP_BACKEND`):
`quadrature`, the default (`ecp_quad/`, pure Rust), or `libecpint` (C++ via
`shim/ecp_shim.cc`, kept as a cross-check). Both return Cartesian blocks that
`ecp.rs` transforms to spherical harmonics.

**Angular projection.** A Gaussian on centre $A$, with $\mathbf a = \mathbf A - \mathbf C$,
expands about $C$ through the plane-wave (Bessel) identity

$$e^{-\alpha|\mathbf r-\mathbf a|^2} = 4\pi\, e^{-\alpha(r^2+a^2)} \sum_{\lambda}\, i_\lambda(2\alpha a r)\sum_{\mu} Y_{\lambda\mu}(\hat{\mathbf r})\,Y_{\lambda\mu}(\hat{\mathbf a}),$$

where $i_\lambda$ is the modified spherical Bessel function of the first kind
and $Y_{\lambda\mu}$ are real spherical harmonics. The semi-local (type-2) term
needs $F_{lm}(r) = \int Y_{lm}(\Omega)\,\chi(\mathbf C + r\Omega)\,d\Omega$; its angular
factor does not depend on $r$ and is built once per shell pair, centre and
channel (`ecp_quad/type2.rs`). The local (type-1) term expands the bra-ket
product Gaussian the same way (`ecp_quad/type1.rs`).

**Radial integrals.** The code uses the exponentially scaled
$\tilde k_\lambda(z) = e^{-z} i_\lambda(z)$ (`ecp_quad/bessel.rs`) and combines the
Gaussian exponents into $e^{-p(r-r_0)^2 - K}$ with $p = \alpha+\beta+\zeta$,
$r_0 = (\alpha a + \beta b)/p$ and $K = \alpha a^2 + \beta b^2 - p r_0^2 \ge 0$, so no
factor overflows at long range. Each (primitive pair, ECP term) radial
integral runs on its own window: Gauss-Hermite away from $r = 0$, adaptive
Gauss-Legendre otherwise (`ecp_quad/radial.rs`).

**Derivatives.** Shell derivatives use raised and lowered shells,
$\partial\chi/\partial A_x = 2\alpha\,\chi(i+1) - i\,\chi(i-1)$, and the ECP-centre
derivative follows from translation invariance as $-(\partial_A + \partial_B)$.

The formulation follows McMurchie and Davidson, *J. Comput. Phys.* **44**, 289
(1981); libecpint is Shaw and Hill, *J. Chem. Phys.* **147**, 074108 (2017).
Tests: `tests/ecp_quadrature.rs` (quadrature against libecpint, and negative controls: broken
constructions that must fail), `tests/ecp_matrix.rs`
(spherical blocks against PySCF `ECPscalar`), `tests/ecp_matrix_deriv.rs`,
`tests/ecp_deriv_block.rs`.
