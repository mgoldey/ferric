# 3. Molecular integrals

Every method in the book is built from a few kinds of Gaussian integrals: the
one-electron matrices, the two-electron integrals over a kernel, their
derivatives, and basis functions evaluated on a grid. This chapter states
what each integral is, which engine computes it, and the conventions a caller
has to respect: block layout, derivative buffer sizes, effective charges.

| Section | Integral | Engine |
|---|---|---|
| 3.1 | $S$, $T$, $V$, multipoles | libint2 |
| 3.2 | $(\mu\nu\vert\lambda\sigma)$, $(P\vert\mu\nu)$, $(P\vert Q)$ | libint2; table engine for terf and terfc |
| 3.3 | first and second derivatives | libint2 |
| 3.4 | $\int \chi_\mu\chi_\nu/\vert\mathbf r - \mathbf r_g\vert$ on a grid | McMurchie–Davidson, in Rust |
| 3.5 | effective core potentials | quadrature, in Rust; libecpint as a cross-check |
| 3.6 | pVp and the X2C-1e core Hamiltonian | libint2 second derivatives |
| 3.7 | $\chi_\mu$, $\nabla\chi_\mu$, $\nabla\nabla\chi_\mu$ on points | Rust |

The derivations live in `crates/ferric-integrals/theory.md`; each section
includes the part it needs.
