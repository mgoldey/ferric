# 19. GW and the Bethe–Salpeter equation

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

$W$ is the object chapter 18 already builds: the RPA dielectric matrix,
inverted. This chapter reuses it three times. Sections 19.1 to 19.7 compute
quasiparticle energies with it. Section 19.8 uses its static value as the
attraction between an excited electron and the hole it left, which turns
quasiparticle energy differences into excitation energies (the
Bethe–Salpeter equation). Section 19.9 puts the same static kernel into the
full linear-response problem and computes a polarizability from it.

The chapter follows the data:

1. Section 19.1 builds the three-index tensor over every pair of active
   orbitals, and the exchange self-energy from it.
2. Section 19.2 turns the dielectric eigenbasis of section 18.3 into a
   screened interaction.
3. Sections 19.3 to 19.6 evaluate the correlation self-energy, its static
   limit, the analytic continuation and the quasiparticle equation, and the
   eigenvalue self-consistent variants.
4. Section 19.7 gives the open-shell form, and sections 19.8 and 19.9 use
   the static screened interaction as a kernel.

The imaginary frequency is $u$, as in chapter 18. The range-separation
parameter $\omega$ does not appear. PDEP modes are $\xi, \eta$. Orbital
energies on the mean-field reference are $\epsilon_p$, and quasiparticle
energies are $\epsilon_p^{\rm QP}$. Units are Hartree atomic units.

The terse reference for every equation is `crates/ferric-gw/theory.md`. The
sections here derive the same equations step by step, and the
verified-equation boxes tie both to the sympy proofs and the code.
