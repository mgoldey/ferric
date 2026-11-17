# 9. Kohn–Sham density functional theory

Hartree–Fock gives every electron the average field of the others and then
stops: it has exchange but no correlation. Kohn–Sham theory keeps the
one-determinant machinery of chapter 7 and adds one more term to the
energy, an exchange-correlation functional of the density. The SCF loop
does not change. What changes is that part of the energy is now an integral
over space, evaluated on a grid.

- 9.1 writes the Kohn–Sham energy, derives its Fock matrix and lists the
  functional families ferric evaluates through libxc.
- 9.2 builds the atom-centred quadrature grid: a radial rule, a Lebedev
  sphere and Becke's partition of space between the atoms.
- 9.3 derives the exchange-correlation matrix for LDA, GGA and meta-GGA
  functionals, and the batched loop that builds it.
- 9.4 adds exact exchange, globally or split by electron–electron distance,
  and tunes the split parameter $\omega$.
- 9.5 adds VV10 nonlocal correlation, a double integral over the density.
- 9.6 differentiates the potential once more: the response kernels the
  orbital Hessian, the Z-vector and TDDFT use.
- 9.7 constrains the charge on a fragment with a Lagrange multiplier and
  computes the electron-transfer coupling between two constrained states.

The XC nuclear gradient is section 14.3, the KS Z-vector section 15.4 and
TDDFT chapter 20. Each section derives its equations in full;
`crates/ferric-dft/theory.md` keeps the terse reference version, and the
verified-equation boxes tie both to the sympy proofs and the code.
