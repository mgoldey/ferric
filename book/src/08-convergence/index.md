# Convergence {#convergence}

The SCF equations have no closed-form solution. You guess orbitals, build the
Fock matrix from their density, diagonalize it, and repeat until the density
stops changing. That loop can fail in three ways. It can oscillate between
two densities or diverge. It can settle on a solution of the SCF equations
that is a saddle point of the energy, not a minimum. And it can stop because
the density stopped moving while the orbitals are still not stationary. This
chapter covers the tools that drive the loop to a stationary point and the
analysis that checks which kind of stationary point it found.

- [@sec:diis DIIS and its variants](01-diis.md): the error vector, why it is the
  orbital gradient, and the constrained least-squares problem that
  extrapolates the Fock matrix.
- [@sec:level-shift-mom-smearing Level shift, MOM and smearing](02-level-shift-mom-smearing.md): a
  shift that damps the step, a rule that keeps the occupied set continuous,
  and fractional occupations for a degenerate frontier.
- [@sec:second-order Second-order steps: Newton, augmented Hessian, trust region](03-second-order.md):
  steps from the orbital Hessian, and the shift that keeps them downhill.
- [@sec:stability Stability analysis](04-stability.md): the lowest eigenvalue of the
  orbital Hessian at the converged orbitals, and the descent off a saddle.
