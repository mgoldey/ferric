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

The imaginary frequency is $u$ (chapter @sec:notation, section @sec:notation-attenuated-coulomb-operators). An atom's oscillator
frequency carries an atom subscript, $\omega_A$, as in the code and in
section @sec:polarizability-pdep-c6; it is never the range-separation parameter, which does not
appear in this chapter.
