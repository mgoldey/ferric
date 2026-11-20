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
one special function, the Boys function (section @sec:two-electron-boys). Higher angular
momentum follows from recurrence relations.

This chapter states what each integral is, which engine computes it, and the
conventions a caller has to respect: block layout, derivative buffer sizes,
effective charges.

| Index | Ranges over |
|---|---|
| $\mu, \nu, \lambda, \sigma$ | AO basis functions |
| $P, Q$ | auxiliary basis functions |
| $s_1, \dots, s_4$ | shells |
| $\mathbf A, \mathbf B, \mathbf C$ | centres of basis functions, nuclei, point charges or grid points |
| $a, b$; $p = a + b$ | the exponents of two primitives and of their product (section @sec:one-electron) |

A shell, its exponents $\alpha$ and its contraction coefficients $c$ are
defined in section @sec:basis-sets-contracted-shell. This chapter writes the
primitive index as $k$ ($\alpha_k$, $c_k$), because $p$ is the exponent of a
product. $Z_C$ is always the
effective charge $Z^{\mathrm{eff}}_C$ of (@eq:molecule-1), and
$n^{\mathrm{ECP}}_C$ the number of core electrons an ECP replaces.

| Section | Integral | Engine |
|---|---|---|
| @sec:one-electron | $S$, $T$, $V$, multipoles | libint2 |
| @sec:two-electron | $(\mu\nu\vert\lambda\sigma)$, $(P\vert\mu\nu)$, $(P\vert Q)$ | libint2; table engine for terf and terfc |
| @sec:integrals-derivatives | first and second derivatives | libint2 |
| @sec:md-three-center | $\int \chi_\mu\chi_\nu/\vert\mathbf r - \mathbf r_g\vert$ on a grid | McMurchie–Davidson, in Rust |
| @sec:ecp | effective core potentials | quadrature, in Rust; libecpint as a cross-check |
| @sec:pvp-x2c | pVp and the X2C-1e core Hamiltonian | libint2 second derivatives |
| @sec:ao-grid | $\chi_\mu$, $\nabla\chi_\mu$, $\nabla\nabla\chi_\mu$ on points | Rust |

Most integral classes go through libint2 behind a C shim
(`crates/ferric-integrals/shim/shim.cc`). The terf and terfc kernels, the
three-center one-electron batch kernel of section @sec:md-three-center and the ECPs have their
own implementations. Each section derives the part it needs.
