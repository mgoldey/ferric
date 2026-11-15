# 22. Implicit solvation

A solvent treated as a dielectric continuum around a molecular cavity. Section
22.1 builds the cavity and the boundary-element matrices, 22.2 solves the
integral equation for the apparent surface charges (IEF-PCM) and checks it
against the analytic sphere, 22.3 puts the charges into the Fock matrix, 22.4
differentiates the energy with respect to the nuclei, and 22.5 covers the
conductor limit that COSMO uses.

Sections 22.1 to 22.4 are the `ferric-pcm` crate, whose derivations are in
`crates/ferric-pcm/theory.md`; the sections include them by heading. COSMO
(22.5) lives in `ferric-scf` and shares the cavity switching function and
the Gaussian-width table with `ferric-pcm`.
