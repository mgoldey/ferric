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
inverted. The chapter follows the data:

1. Section @sec:ri-factors-and-exchange builds the three-index tensor over every pair of active
   orbitals, and the exchange self-energy from it.
2. Section @sec:screened-interaction turns the dielectric eigenbasis of section @sec:pdep into a
   screened interaction.
3. Sections @sec:correlation-self-energy to @sec:eigenvalue-self-consistency evaluate the correlation self-energy, its static
   limit, the analytic continuation and the quasiparticle equation, and the
   eigenvalue self-consistent variants.
4. Section @sec:open-shell-gw gives the open-shell form.
5. Section @sec:bse-tda uses the static $W$ as the attraction between an
   excited electron and its hole (the Bethe–Salpeter equation), and section
   @sec:rpax-static-polarizability puts the same kernel into the full
   response problem for a polarizability. Both use the response matrices,
   the singlet spin adaptation and the oscillator strengths of chapter
   @sec:tddft.

| CLI kind | Python | Sections | Grade |
|---|---|---|---|
| `gw` (closed shell; open shell when the multiplicity is above 1) | `run_gw`, `run_u_gw` | @sec:ri-factors-and-exchange to @sec:open-shell-gw | Proven (narrow) |
| `bse-tda` | `run_bse_tda` | @sec:bse-tda | Smoke |
| `tdhf-static-polarizability` | `run_tdhf_static_polarizability` | @sec:rpax-static-polarizability | Smoke |

The grades are those of `site/src/reference/validation.md`.

The imaginary frequency is $u$, as in chapter @sec:rpa. $\omega$ is the
range-separation parameter of a hybrid reference in section
@sec:continuation-and-qp-equation-hybrid-references and nowhere else; a real
or complex driving frequency is $\nu$. PDEP modes are $\xi, \eta$. Orbital
energies on the mean-field reference are $\epsilon_p$, and quasiparticle
energies are $\epsilon_p^{\rm QP}$. This chapter writes $\epsilon_p$, not
the $\varepsilon_p$ of the notation page, because
$\tilde{\boldsymbol\varepsilon}$ is the dielectric matrix. Units are
Hartree atomic units.
