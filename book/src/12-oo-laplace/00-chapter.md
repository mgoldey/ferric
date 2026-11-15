# 12 Orbital optimization and Laplace forms

<!-- section: 12.0; status: verify; registry: oo-rimp2, u-oo-rimp2, laplace-mp2, laplace-sos-mp2;
     theory: crates/ferric-mp2/theory.md#"7. Orbital-optimized RI-MP2", "8. Laplace-transform MP2", "14. Unrestricted OO-RI-MP2 orbital gradient" -->

This chapter covers two changes to the RI-MP2 energy of chapter 10. Neither
change alters the pair loop.

- Orbital optimization (sections 12.1 and 12.2) varies the orbitals to make
  the Hartree–Fock plus MP2 energy stationary. After that, the MP2 energy is
  taken in a frame that is no longer canonical.
- The Laplace forms (sections 12.3 and 12.4) replace the energy denominator
  by a short sum of exponentials. This removes the coupling between the
  $(i,a)$ and $(j,b)$ halves of every term. One factorization is exact in the
  MO and AO bases; the other, with localized orbitals, is truncated.

| Section | Method | Registry id | Status |
|---|---|---|---|
| 12.1 | Closed-shell OO-RI-MP2 | `oo-rimp2` | ready |
| 12.2 | Unrestricted OO-RI-MP2 | `u-oo-rimp2` (gap) | verify |
| 12.3 | Laplace RI-MP2 and Laplace SOS-MP2, `mo` and `ao` | `laplace-mp2`, `laplace-sos-mp2` | ready |
| 12.4 | Laplace SOS-MP2, `ao-sparse` and `local` | (part of `laplace-sos-mp2`, not graded) | gap |

**Symbols used throughout.**

- $t \ge 0$ is the **Laplace variable**. It is a real parameter of the
  transform $1/x = \int_0^\infty e^{-tx}\,dt$. It is not a time or an
  imaginary frequency; the imaginary frequency is $u$ (notation, section 8).
- $\kappa$ is an orbital rotation generator (notation, section 6).
- $\mu$ is the level shift of a rotation step.
- $M$ is the dressing matrix with $M^{\mathsf T}M = V^{-1}$, as in
  section 10.1.
- $\Delta_{ij}^{ab} = \varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j > 0$.
  A minus sign in an energy expression is always written out.
- Section 12.2 marks $\beta$ orbitals with an overbar.

Equations are numbered (12.1) and up across the chapter.
