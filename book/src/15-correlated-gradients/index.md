# Correlated gradients and the Z-vector {#correlated-gradients}

A Hartree–Fock gradient needs no orbital response, because the energy is
stationary in the orbitals. A correlation energy computed on top of those
orbitals is not, so its gradient has to account for how the orbitals move
when a nucleus moves. This chapter shows how one linear equation, the
Z-vector equation, supplies that response for every nuclear coordinate at
once, and how the resulting relaxed densities give nuclear gradients and
static response properties of the correlated methods.

## Conventions across the chapter {#correlated-gradients-conventions}

The sections follow the code they describe, and the codes were written
against different references, so four symbols change meaning from section
to section. The table lists them; each section restates its own choice.

| Section | $\Delta$ | $P$ (unrelaxed density) | Z-vector unknown | $W$ | core / active occupied |
|---|---|---|---|---|---|
| @sec:lagrangian-zvector | $\Delta_i^a = \varepsilon_a - \varepsilon_i > 0$ | $p$, one-sided blocks | $z$, solves $Az = -X$; the multiplier of $F_{ai}$ is $2z$ | orthonormality multiplier | |
| @sec:rimp2-gradient | $\Delta_{ij}^{ab} = \varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b < 0$ | $P$, one-sided | $z$ of section @sec:lagrangian-zvector | energy-weighted density | $k$ / $I$ |
| @sec:oo-rimp2-gradient | | $D^{\mathrm{MO}}$, the correlation density | none: $z = 0$ at convergence | energy-weighted density | |
| @sec:ks-zvector | | | $Z$, the full multiplier, $HZ = -4V$ | $\tfrac12DFD$ and its derivative | |
| @sec:rpa-gradient | $\Delta_{ij}^{ab} = \varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j > 0$ | $P = 2p = p + p^{\mathsf T}$ | $\bar z = -2z$, solves $A\bar z = L$ with $L = 2X$ | energy-weighted density | $k$ / $I$ |
| @sec:mp2-polarizability | positive | $p$, one-sided | $z$ of section @sec:lagrangian-zvector | | |
| @sec:u-rimp2-gradient | positive | $P^\sigma = \tfrac12(p^\sigma + p^{\sigma\mathsf T})$, as the code stores it | $z^\sigma$, coupled over spins | energy-weighted density per spin | |

In every section the relaxed density is $D^{\mathrm{HF}}$ plus the
correlation density plus the Z-vector density $D^z$ of
(@eq:zvec-hessian-product); the normalizations above only change how each
piece is written.
