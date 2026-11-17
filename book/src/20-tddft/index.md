# Time-dependent density functional theory {#tddft}

Shine light of frequency $\omega$ on a molecule and its electron density
oscillates at that frequency. For most $\omega$ the oscillation is small.
When $\hbar\omega$ matches the energy of an excited state, the response
grows without bound: the excitation energies are the poles of the density
response. Linear-response TDDFT finds those poles from the converged
ground-state orbitals, without ever writing down an excited-state
wavefunction.

The chapter builds the response problem for a converged SCF reference.
Section @sec:response derives the response matrices $\mathbf A$ and $\mathbf B$, the
Tamm–Dancoff approximation and the Casida equations, and the dense and
matrix-free solvers. Section @sec:oscillator-strengths turns the amplitudes into transition
dipoles and oscillator strengths. Section @sec:kernel covers the
exchange-correlation kernel and which functionals are accepted, section
@sec:rsh range-separated hybrids, and section @sec:triplets triplets. Section @sec:unrestricted
solves the spin-conserving problem of UHF and UKS references and gives
$\langle S^2\rangle$ for each root.

The kernel itself is derived in section @sec:fxc, and the matrix
$\mathbf A + \mathbf B$ is the orbital Hessian of sections @sec:second-order and @sec:stability. Every equation of
the chapter is also written, in terse form, in
`crates/ferric-tddft/theory.md`.
