# GW and the Bethe–Salpeter equation {#gw-bse}

Hartree–Fock orbital energies are often read as ionization energies and
electron affinities (Koopmans' theorem). They are not. When an electron
leaves a molecule, the electrons that stay behind rearrange around the hole
and lower its energy; when one is added, the others move away from it. An
orbital energy computed in the frozen field of the neutral molecule misses
both responses. GW adds them: the self-energy $\Sigma = iGW$ lets the added
or removed electron interact with the others through the *screened*
interaction $W$, the bare Coulomb interaction weakened by the polarization
it induces. The result is a quasiparticle energy, the energy of an electron
or hole dressed by that response.

$W$ is the object chapter @sec:rpa already builds: the RPA dielectric matrix,
inverted. This chapter reuses it three times. Sections @sec:ri-factors-and-exchange to @sec:open-shell-gw compute
quasiparticle energies with it. Section @sec:bse-tda uses its static value as the
attraction between an excited electron and the hole it left, which turns
quasiparticle energy differences into excitation energies (the
Bethe–Salpeter equation). Section @sec:rpax-static-polarizability puts the same static kernel into the
full linear-response problem and computes a polarizability from it.

The chapter follows the data:

1. Section @sec:ri-factors-and-exchange builds the three-index tensor over every pair of active
   orbitals, and the exchange self-energy from it.
2. Section @sec:screened-interaction turns the dielectric eigenbasis of section @sec:pdep into a
   screened interaction.
3. Sections @sec:correlation-self-energy to @sec:eigenvalue-self-consistency evaluate the correlation self-energy, its static
   limit, the analytic continuation and the quasiparticle equation, and the
   eigenvalue self-consistent variants.
4. Section @sec:open-shell-gw gives the open-shell form, and sections @sec:bse-tda and @sec:rpax-static-polarizability use
   the static screened interaction as a kernel.

The imaginary frequency is $u$, as in chapter @sec:rpa. The range-separation
parameter $\omega$ does not appear. PDEP modes are $\xi, \eta$. Orbital
energies on the mean-field reference are $\epsilon_p$, and quasiparticle
energies are $\epsilon_p^{\rm QP}$. Units are Hartree atomic units.

The sections of this chapter derive the equations step by step, and the
verified-equation boxes tie them to the sympy proofs and the code.
