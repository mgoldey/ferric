# 20. Time-dependent density functional theory

Shine light of frequency $\omega$ on a molecule and its electron density
oscillates at that frequency. For most $\omega$ the oscillation is small.
When $\hbar\omega$ matches the energy of an excited state, the response
grows without bound: the excitation energies are the poles of the density
response. Linear-response TDDFT finds those poles from the converged
ground-state orbitals, without ever writing down an excited-state
wavefunction.

The chapter builds the response problem for a converged SCF reference.
Section 20.1 derives the response matrices $\mathbf A$ and $\mathbf B$, the
Tamm–Dancoff approximation and the Casida equations, and the dense and
matrix-free solvers. Section 20.2 turns the amplitudes into transition
dipoles and oscillator strengths. Section 20.3 covers the
exchange-correlation kernel and which functionals are accepted, section
20.4 range-separated hybrids, and section 20.5 triplets. Section 20.6
solves the spin-conserving problem of UHF and UKS references and gives
$\langle S^2\rangle$ for each root.

The kernel itself is derived in section 9.6, and the matrix
$\mathbf A + \mathbf B$ is the orbital Hessian of sections 8.3 and 8.4. Every equation of
the chapter is also written, in terse form, in
`crates/ferric-tddft/theory.md`.
