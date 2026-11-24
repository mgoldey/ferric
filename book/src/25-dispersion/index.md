# Dispersion {#dispersion}

Two neutral atoms far apart still attract. The electron cloud of each
fluctuates, the momentary dipole of one polarizes the other, and the
correlated motion of the two dipoles lowers the energy as $-C_6/R^6$. A
semilocal functional sees only the density and its gradient at each point,
so it has no term that couples two separated fluctuations and misses this
attraction. This chapter adds it back in three ways, from the cheapest to
the most physical:

- @sec:d3bj, D3(BJ): a damped $-C_6/R^6 - C_8/R^8$ sum over atom pairs, with
  $C_6$ interpolated from the bonding environment and the damping fitted per
  functional.
- @sec:casimir-polder-c6, Casimir–Polder $C_6$: the pair coefficients computed from dynamic
  polarizabilities, by integrating their product over imaginary frequency.
- @sec:mbd, many-body dispersion (MBD@rsSCS): the atoms become coupled
  oscillators, so the screening of one atom's fluctuation by its neighbours
  is part of the energy instead of being summed pair by pair.

The three side by side on one molecule, water (`testdata/molecules/water.xyz`),
with the numbers of the runnable examples in each section:

| Quantity | D3(BJ) | Casimir–Polder $C_6$ | MBD@rsSCS |
|---|---|---|---|
| dispersion energy, PBE | $-3.5948\times10^{-4}$ Ha (two-body, section @sec:d3bj-three-body-term) | no energy (exported coefficients only) | $-2.4229\times10^{-4}$ Ha at PBE/cc-pVDZ volume ratios (section @sec:mbd-coupled-oscillator-energy) |
| molecular $C_6$, RHF/cc-pVDZ (section @sec:casimir-polder-c6-three-sources) | tabulated per pair, not a molecular output | `ts` 45.38 a.u., `pdep` 11.51 a.u. | `mbd` source (full-range screening, section @sec:mbd-full-range-screening) 41.25 a.u. |

D3 needs only the geometry. MBD needs the SCF's Hirshfeld volumes, and the
`pdep` $C_6$ a direct-RPA response.

The imaginary frequency is $u$ (chapter @sec:notation, section @sec:notation-attenuated-coulomb-operators). An atom's oscillator
frequency carries an atom subscript, $\omega_A$, as in the code and in
section @sec:polarizability-pdep-c6; it is never the range-separation parameter, which does not
appear in this chapter.
