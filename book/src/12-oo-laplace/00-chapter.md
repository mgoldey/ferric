# Orbital optimization and Laplace forms {#oo-laplace}

<!-- status: verify; registry: oo-rimp2, u-oo-rimp2, laplace-mp2, laplace-sos-mp2, laplace-sos-mp2-local -->

Chapter @sec:mp2 computed the MP2 energy with the Hartree–Fock orbitals held
fixed and the energy denominator $1/\Delta_{ij}^{ab}$ evaluated term by
term. This chapter changes each of those two choices in turn. The pair loop
of chapter @sec:mp2 stays the same in both.

- **Orbital optimization** (sections @sec:oo-rimp2 and @sec:unrestricted-oo-rimp2). Hartree–Fock orbitals
  are the best orbitals for a single determinant. Once MP2 correlation is
  added, they are no longer the best orbitals for the total energy. OO-MP2
  lets the orbitals relax in the field of the correlation energy, until the
  Hartree–Fock plus MP2 energy is stationary under orbital rotations. The
  MP2 energy is then taken in orbitals that are no longer canonical.
- **The Laplace forms** (sections @sec:laplace-mp2 and @sec:local-laplace-sos). The denominator
  $\Delta_{ij}^{ab}$ ties the occupied–virtual pair $(i,a)$ to the pair
  $(j,b)$. Writing $1/\Delta$ as a short sum of exponentials separates the
  two pairs in every term. The energy becomes a few RI-MP2-like
  contractions, one per quadrature point. Two factorizations are exact,
  one in the MO basis and one in the AO basis. Two more truncate by
  distance: one in the AO basis, one in localized orbitals.

| Section | Method | Registry id |
|---|---|---|
| @sec:oo-rimp2 | Closed-shell OO-RI-MP2 | `oo-rimp2` |
| @sec:unrestricted-oo-rimp2 | Unrestricted OO-RI-MP2 | `u-oo-rimp2` |
| @sec:laplace-mp2 | Laplace RI-MP2 and Laplace SOS-MP2, `mo` and `ao` | `laplace-mp2`, `laplace-sos-mp2` |
| @sec:local-laplace-sos | Laplace SOS-MP2, `ao-sparse` and `local` | `laplace-sos-mp2-local` (not graded) |

**Symbols used throughout.**

- $t \ge 0$ is the **Laplace variable**, the real parameter of the
  transform $1/x = \int_0^\infty e^{-tx}\,dt$. It is not a time and not an
  imaginary frequency; the imaginary frequency is $u$ (notation, section @sec:notation-attenuated-coulomb-operators).
- $\kappa$ is an orbital rotation generator (notation, section @sec:notation-orbital-rotations).
- $\mu$ is the level shift of a rotation step.
- $M$ is the dressing matrix with $M^{\mathsf T}M = V^{-1}$, as in
  section @sec:closed-shell-ri-mp2.
- $\Delta_{ij}^{ab} = \varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j > 0$.
  A minus sign in an energy expression is always written out.
- Section @sec:unrestricted-oo-rimp2 marks $\beta$ orbitals with an overbar.

Equations are numbered (@eq:oo-rimp2-1) and up across the chapter.
