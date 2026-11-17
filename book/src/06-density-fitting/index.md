# 6. Density fitting

A four-index integral is the Coulomb energy between two orbital-product
densities. Density fitting expands each product density in an auxiliary
basis and turns every four-index integral into a product of two three-index
ones. This chapter defines the fit once for the rest of the book.

- 6.1 derives the fit and the factorization (6.1), and explains the two
  factors of the inverse metric the code uses: Cholesky by default, a
  truncated eigendecomposition for erf and terf.
- 6.2 covers where the three-index tensor lives (in core, packed, spilled to
  disk or recomputed) and which results are bit-identical across those tiers.
- 6.3 derives the error of the fit and Dunlap's robust form for a fitting
  kernel that differs from the physical one.
- 6.4 builds the complementary auxiliary basis (CABS) that F12 methods need.
