# Where the methods come from

The three method families in `ferric` are not an arbitrary selection. Each one
attacks the response function from a different direction.

## Attenuated MP2 — fixing a response error

MP2 correlation is built from an **uncoupled** polarizability. Uncoupled means
the density fluctuation does not feel the field it creates: there is no
self-consistency in the response. The result over-polarizes, which shows up as:

- \\( C_6 \\) dispersion coefficients that are too large
- overestimated π-stacking energies
- basis-set superposition error that partly cancels the overestimate, disguising
  the problem in small basis sets

Attenuating the correlation operator — replacing \\( 1/r \\) with
\\( \mathrm{erfc}(\omega r)/r \\) or a `terfc` form — damps the long-range part
where the uncoupled approximation is worst, with a single tunable parameter.

This is Goldey & Head-Gordon (JPCL 2012); the dual-attenuated SCS variant is
Goldey, Dutoi & Head-Gordon (PCCP 2013). See [The MP2 family](../methods/mp2.md).

## PDEP-RPA and GW — compressing the response

The dielectric matrix **is** the density–density response function. Conventional
RPA and GW evaluate it through an explicit sum over empty orbital states, which
is expensive and converges slowly with basis size.

PDEP — projective dielectric eigenpotentials — builds a low-rank basis from the
dominant eigenmodes of the dielectric matrix instead. Because the spectrum
decays quickly, a modest number of modes captures the physics, and the sum over
empty states disappears.

This is the part of the locality-and-low-rank claim that is **actually
demonstrated** in this codebase. See [RPA and GW](../methods/rpa-gw.md).

## Constrained DFT — reading the response

A cDFT constraint couples a Lagrange multiplier \\( \lambda \\) to a
fragment-weighted density operator. The derivative \\( \partial N / \partial
\lambda \\) — how much charge moves per unit constraint potential — *is* a
susceptibility.

That makes cDFT a direct probe of the same object, and it yields
charge-localized diabatic states whose electron-transfer couplings
\\( H_{ab} \\) follow from non-orthogonal determinant overlaps.

See [Constrained DFT](../methods/cdft.md).

## What this buys

Three methods, one object. An error in the polarizability shows up as an error
in dispersion, in screening, and in charge-transfer coupling — so a fix
validated in one place has predictable consequences in the others.

That is the design bet. Whether it pays off in *cost* is still open (see
[Electronic response](./response.md) for the measured negatives); that it pays
off in *diagnosis* is already clear.
