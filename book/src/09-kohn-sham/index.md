# Kohn–Sham density functional theory {#kohn-sham}

Hartree–Fock gives every electron the average field of the others and then
stops: it has exchange but no correlation. Kohn–Sham theory keeps the
one-determinant machinery of chapter @sec:hartree-fock and adds one more term to the
energy, an exchange-correlation functional of the density. The SCF loop
does not change. What changes is that part of the energy is now an integral
over space, evaluated on a grid.

- @sec:ks-equations writes the Kohn–Sham energy, derives its Fock matrix and lists the
  functional families ferric evaluates through libxc.
- @sec:grids builds the atom-centred quadrature grid: a radial rule, a Lebedev
  sphere and Becke's partition of space between the atoms.
- @sec:vxc derives the exchange-correlation matrix for LDA, GGA and meta-GGA
  functionals, and the batched loop that builds it.
- @sec:hybrids adds exact exchange, globally or split by electron–electron distance,
  and tunes the split parameter $\omega$.
- @sec:vv10 adds VV10 nonlocal correlation, a double integral over the density.
- @sec:fxc differentiates the potential once more: the response kernels the
  orbital Hessian, the Z-vector and TDDFT use.
- @sec:cdft constrains the charge on a fragment with a Lagrange multiplier and
  computes the electron-transfer coupling between two constrained states.

The XC nuclear gradient is section @sec:xc-gradient, the KS Z-vector section @sec:ks-zvector and
TDDFT chapter @sec:tddft. Each section derives its equations in full, and
the verified-equation boxes tie them to the sympy proofs and the code.
