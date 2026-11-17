# 7. Hartree–Fock

Hartree–Fock puts each electron in an orbital and lets it feel the average
charge cloud of all the others. That average field depends on the orbitals,
and the orbitals depend on the field, so the equations are solved by
iteration: guess a density, build the field, find new orbitals, repeat. The
field is the Fock matrix. Its two-electron part has a Coulomb term $J$, the
classical repulsion of the charge cloud, and an exchange term $K$, which
comes from the antisymmetry of the wavefunction: it cancels the repulsion of
each electron with its own charge and lowers the repulsion between electrons
of the same spin.

This chapter covers the RHF, UHF and ROHF energy functionals and how the Fock
matrix is built: orthogonalization, starting guesses, direct and
density-fitted Coulomb and exchange, and the hook for user-supplied Fock
terms. Chapter 8 covers what drives the iteration to convergence.

Sections:

- [7.1 RHF, UHF and ROHF energy functionals](01-energy-functionals.md):
  the three energies, their Fock matrices as derivatives, and the
  stationarity conditions.
- [7.2 Orthogonalization and starting guesses](02-orthogonalization-and-guesses.md):
  the orthonormal basis each iteration diagonalizes in, which functions it
  drops, and the first density.
- [7.3 Direct and incremental J/K builds](03-direct-and-incremental.md):
  the exact four-centre build, its screen, and building from the density
  change.
- [7.4 RI-J, RI-K and Split-RI-J](04-ri-j-ri-k-split-ri-j.md):
  the density-fitted builds of chapter 6 applied to $J$ and $K$.
- [7.5 LinK, seminumerical exchange (COSX) and CFMM](05-link-cosx-cfmm.md):
  three builders that give up some exactness or some generality for large
  molecules.
- [7.6 Fock hooks](06-fock-hooks.md): a caller-supplied energy term and its
  potential, and what the solver refuses with one.
