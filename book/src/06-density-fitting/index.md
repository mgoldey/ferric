# Density fitting {#density-fitting}

A four-index integral is the Coulomb energy between two orbital-product
densities. Density fitting expands each product density in an auxiliary
basis and turns every four-index integral into a product of two three-index
ones. This chapter defines the fit once for the rest of the book.

- @sec:factorization-and-metric derives the fit and the factorization (@eq:factorization-and-metric-1), and explains the two
  factors of the inverse metric the code uses: Cholesky by default, a
  truncated eigendecomposition for erf, and a signed pair of Cholesky factors
  for terf.
- @sec:three-index-sources covers where the three-index tensor lives (in core, packed, spilled to
  disk or recomputed) and which results are bit-identical across those tiers.
- @sec:robust-fit derives the error of the fit and Dunlap's robust form for a fitting
  kernel that differs from the physical one.
- @sec:cabs builds the complementary auxiliary basis (CABS) that F12 methods need.
  CABS resolves a one-particle identity rather than fitting product
  densities; it sits here because it is the other construction built from
  an auxiliary basis.
