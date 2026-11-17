# Molecular integrals {#integrals}

A Hartree–Fock or correlated calculation never touches a wavefunction on a
grid of points in space. It works with matrices: how much two basis functions
overlap, how fast an electron in them moves, how strongly it is pulled toward
the nuclei, and how strongly two charge clouds repel. Every one of those
numbers is an integral over Gaussian functions, and every method in this book
is built from a few kinds of them: the one-electron matrices, the
two-electron integrals over a kernel, their derivatives, and basis functions
evaluated on a grid.

Gaussians are used because these integrals have closed forms. The product
of two Gaussians on different atoms is one Gaussian on a point between them
(section @sec:one-electron derives it), and the Coulomb potential of a Gaussian reduces to
one special function, the Boys function (section @sec:md-three-center). Higher angular
momentum follows from recurrence relations.

This chapter states what each integral is, which engine computes it, and the
conventions a caller has to respect: block layout, derivative buffer sizes,
effective charges.

| Section | Integral | Engine |
|---|---|---|
| 3.1 | $S$, $T$, $V$, multipoles | libint2 |
| 3.2 | $(\mu\nu\vert\lambda\sigma)$, $(P\vert\mu\nu)$, $(P\vert Q)$ | libint2; table engine for terf and terfc |
| 3.3 | first and second derivatives | libint2 |
| 3.4 | $\int \chi_\mu\chi_\nu/\vert\mathbf r - \mathbf r_g\vert$ on a grid | McMurchie–Davidson, in Rust |
| 3.5 | effective core potentials | quadrature, in Rust; libecpint as a cross-check |
| 3.6 | pVp and the X2C-1e core Hamiltonian | libint2 second derivatives |
| 3.7 | $\chi_\mu$, $\nabla\chi_\mu$, $\nabla\nabla\chi_\mu$ on points | Rust |

Most integral classes go through libint2 behind a C shim
(`crates/ferric-integrals/shim/shim.cc`). The terf and terfc kernels, the
three-center one-electron batch kernel of section @sec:md-three-center and the ECPs have their
own implementations. The derivations live in
`crates/ferric-integrals/theory.md`; each section derives the part it needs.
