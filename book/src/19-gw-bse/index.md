# 19. GW and the Bethe–Salpeter equation

Quasiparticle energies from the GW approximation, on the screened interaction
that chapter 18 builds as a by-product of the random phase approximation, and
two uses of the same screened interaction at a static frequency: Bethe–Salpeter
excitation energies in the Tamm–Dancoff approximation and a statically
screened RPAx polarizability.

The chapter follows the data. Section 19.1 builds the three-index tensor over
every pair of active orbitals and the exchange self-energy. Section 19.2 turns
the dielectric eigenbasis of section 18.3 into a screened interaction.
Sections 19.3 to 19.6 evaluate the correlation self-energy, its static limit,
the analytic continuation and the quasiparticle equation, and the eigenvalue
self-consistent variants. Section 19.7 gives the open-shell form, and 19.8 and
19.9 use the static screened interaction as a kernel.

The imaginary frequency is $u$, as in chapter 18; the range-separation
parameter $\omega$ does not appear. PDEP modes are $\xi, \eta$. Orbital
energies on the mean-field reference are $\epsilon_p$; quasiparticle energies
are $\epsilon_p^{\rm QP}$.

Where `theory.md` carries the derivation, the section includes it, and the
section adds the algorithm and the code. The derivations live in
`crates/ferric-gw/theory.md`.
