# 22. Implicit solvation

A molecule in water polarizes the water around it, and the polarized water
pulls on the molecule's electrons in return. An implicit-solvent model
replaces the solvent molecules by a polarizable continuum: a dielectric with
constant $\varepsilon$ that fills all space outside a cavity wrapped around
the solute. The solute's electric field polarizes the dielectric. The
polarization is represented by a charge spread over the cavity surface, and
that charge acts back on the solute's electrons as one more one-electron
potential.

Section 22.1 builds the cavity and the boundary-element matrices. Section
22.2 solves the integral equation for the apparent surface charges
(IEF-PCM) and checks it against the analytic sphere. Section 22.3 puts the
charges into the Fock matrix, 22.4 differentiates the energy with respect to
the nuclei, and 22.5 covers the conductor limit that COSMO uses.

Sections 22.1 to 22.4 are the `ferric-pcm` crate. Its terse reference is
`crates/ferric-pcm/theory.md`, which each section header names. COSMO (22.5)
lives in `ferric-scf` and shares the cavity switching function and the
Gaussian-width table with `ferric-pcm`.
