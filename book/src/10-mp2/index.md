# MP2 {#mp2}

Hartree–Fock lets each electron feel the average charge cloud of the
others. Real electrons also avoid each other instant by instant, and that
lowers the energy further. MP2 is the cheapest wavefunction method that
captures this correlation: second-order perturbation theory on top of
Hartree–Fock. It describes hydrogen bonds well, overestimates dispersion
between stacked rings, and in a finite basis adds basis set superposition
error on top.

This chapter starts from the closed-shell RI-MP2 energy and builds the
methods that reuse its pair loop. The open-shell form splits by spin.
Spin-component scaling weights the opposite- and same-spin parts
differently, and κ-regularization damps small denominators. Attenuated MP2
cuts the correlation off at a fitted length, which removes much of the
overbinding. Later sections cover the range-split cross term, a VV10
dispersion tail, MP3 and the double hybrids.

Sections:

- [@sec:closed-shell-ri-mp2 Closed-shell RI-MP2 energy](01-closed-shell-ri-mp2.md): the
  energy, its opposite- and same-spin parts, density fitting, and the loop
  that runs.
- [@sec:unrestricted-ri-mp2 Unrestricted RI-MP2](02-unrestricted-ri-mp2.md): three spin blocks
  for an open-shell reference.
- [@sec:scs-and-kappa Spin-component scaling and κ-regularization](03-scs-and-kappa.md):
  two one-line changes to the pair loop.
- [@sec:attenuated-mp2 Attenuated MP2: erfc and terfc](04-attenuated-mp2.md): short-range
  correlation with a full Hartree–Fock energy.
- [@sec:scs-mp2-2terfc SCS-MP2(2terfc)](05-scs-mp2-2terfc.md): two attenuation lengths,
  one for each spin component.
- [@sec:attenuation-basis-limit Attenuation and the basis-set limit](06-attenuation-basis-limit.md):
  how the fitted length moves with the basis.
- [@sec:range-split-cross-term The range-split cross term](07-range-split-cross-term.md)
- [@sec:mp2-v MP2-V: attenuated MP2 with a VV10 tail](08-mp2-v.md)
- [@sec:mp3 MP3](09-mp3.md)
- [@sec:double-hybrids MP2-based double hybrids](10-double-hybrids.md)
- [@sec:opmp3 Orbital-partitioned Padé MP3](11-opmp3.md)
