# Molecular dynamics {#md}

A geometry optimization asks where the nuclei want to sit. Molecular dynamics
asks how they move. The nuclei are heavy enough to follow classical
mechanics on the potential-energy surface $E(\mathbf R)$, so a trajectory
is the solution of Newton's equations with the force
$-\nabla E$ that every gradient chapter of this book already computes.
Averaging along the trajectory gives temperatures, vibrational spectra and
reaction paths that a single stationary point cannot give.

ferric adds no new electronic-structure method for this. It adds a time
integrator that takes any force source, a gradient from the SCF drivers
(ab-initio MD) or from the molecular-mechanics force field (force-field MD),
and advances positions and velocities. This chapter has one section:

- @sec:molecular-dynamics, integrators and thermostats: velocity Verlet for
  constant-energy dynamics, Berendsen weak coupling and Langevin dynamics with
  the BAOAB splitting for constant temperature, the conserved quantity that
  tells you whether a run is trustworthy, and the initial velocities.

The force sources are @sec:hf-gradient, @sec:xc-gradient and
@sec:molecular-mechanics. Hybrid QM/MM dynamics, Nosé–Hoover chains, constraint
algorithms and periodic boundary conditions are not implemented.
