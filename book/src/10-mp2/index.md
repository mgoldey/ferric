# MP2 {#mp2}

Hartree–Fock lets each electron feel the average charge cloud of the
others. Real electrons also avoid each other instant by instant, and that
lowers the energy further. MP2 is the cheapest wavefunction method that
captures this correlation: second-order perturbation theory on top of
Hartree–Fock. It describes hydrogen bonds well, overestimates dispersion
between stacked rings, and in a finite basis adds basis set superposition
error on top.

The chapter has three parts.

The MP2 energy and its spin forms:

- @sec:closed-shell-ri-mp2: the closed-shell energy, its opposite- and
  same-spin parts, density fitting, and the loop that runs.
- @sec:unrestricted-ri-mp2: three spin blocks for an open-shell reference.
- @sec:scs-and-kappa: spin-component scaling and κ-regularization, two
  one-line changes to the pair loop.

Attenuation, which cuts the correlation off at a fitted length and removes
much of the overbinding:

- @sec:attenuated-mp2: erfc and terfc attenuation, short-range correlation
  with a full Hartree–Fock energy.
- @sec:scs-mp2-2terfc: two attenuation lengths, one for each spin
  component.
- @sec:attenuation-basis-limit: a summary of the literature on how the
  fitted length moves with the basis; no code of its own.
- @sec:range-split-cross-term: the cross term between short- and long-range
  correlation, and the two ways ferric adds long-range RPA to attenuated
  MP2.
- @sec:mp2-v: attenuated MP2 with a VV10 dispersion tail.

Methods that reuse the amplitudes:

- @sec:mp3: the third-order energy, closed and open shell.
- @sec:double-hybrids: a DFT method, a scaled MP2 term on Kohn–Sham
  orbitals.
- @sec:opmp3: orbital-partitioned Padé MP3, a size-consistent resummation
  of MP2 and MP3.
