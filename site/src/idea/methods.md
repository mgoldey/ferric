# Where the methods come from

The method families in `ferric` are not an arbitrary selection. Each one works
on the response function from a different direction. The definitions and
parameters live on the method pages; this page is the map.

## Attenuated MP2: removing the long-range part

MP2's dispersion comes from an **uncoupled** response, which over-polarizes for
systems with low-lying, highly polarizable excitations (π-stacked aromatics
are the classic case); it is not a general property of polarizable molecules. In small basis sets the
resulting overbinding is partly cancelled by basis-set superposition error,
which disguises it.

Attenuated MP2 replaces \\( 1/r \\) in the correlation energy with a
short-range operator (erfc or terfc) and fits its range to interaction
energies in a chosen basis. It removes the long-range correlation entirely
rather than correcting it, so it has no asymptotic \\( C_6 \\), and its
parameters are specific to the basis and protocol they were fitted in. MP2-V
restores long-range dispersion with VV10; RS-MP2 + LR-RPA restores it with
long-range RPA.

Goldey & Head-Gordon (JPCL 2012) introduced it in aug-cc-pVDZ; Goldey, Dutoi &
Head-Gordon (PCCP 2013) introduced terfc in aug-cc-pVTZ; the dual-attenuated
SCS variant is Goldey & Head-Gordon (JPCB 2014). Operators, parameters and
fitting protocol: [The MP2 family](../methods/mp2.md#attenuated-mp2).

## PDEP-RPA and GW: compressing the response

The dielectric matrix is built from the density–density response function.
ferric forms the independent-particle response in the RI auxiliary basis by
summing over occupied–virtual pairs, then works in the eigenbasis of the
static dielectric matrix, dropping eigenpotentials that carry almost no
screening. PDEP (projective dielectric eigenpotentials) is that compression.
It is the demonstrated part of the low-rank premise; the empty-state sum that
feeds it is still there. See [RPA, GW and excited states](../methods/rpa-gw.md).

## Constrained DFT: reading the response

A cDFT constraint couples a Lagrange multiplier \\( \lambda \\) to a
fragment-weighted density operator. The derivative \\( \partial N / \partial
\lambda \\), how much charge moves per unit constraint potential, *is* a
susceptibility.

That makes cDFT a direct probe of the same object. It also yields
charge-localized diabatic states whose electron-transfer couplings
\\( H_{ab} \\) follow from non-orthogonal determinant overlaps. See
[Constrained DFT](../methods/cdft.md).

## What this buys

Three families, one object. An error in the polarizability shows up as an error
in dispersion, in screening, and in charge-transfer coupling, so a fix
validated in one place has predictable consequences in the others.

That is the design bet. Whether it pays off in *cost* is still open; see
[Electronic response](./response.md) for what has and has not been measured.
