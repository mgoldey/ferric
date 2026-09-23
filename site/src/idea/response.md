# Electronic response

`ferric` is organized around **electronic response**: how the electron density
reacts to a perturbation. Standard quantum-chemistry codes are usually organized
around a hierarchy of *wavefunction ansätze* (HF → MP2 → CCSD → CCSD(T)). That
is a perfectly good organizing principle. It is not the one used here.

The central object appears under several names depending on which coupling you
look at:

| Object | Definition | Where it appears in ferric |
|---|---|---|
| Density response \\( \chi \\) | \\( \chi = \delta\rho / \delta v_{\text{ext}} \\); \\( \chi_0 \\) is its independent-particle form | RPA, GW, MP2's dispersion |
| Dielectric matrix \\( \varepsilon \\) | \\( \varepsilon = 1 - v\chi_0 \\) (RPA), with \\( v \\) the Coulomb kernel | PDEP-RPA and GW screening |
| Polarizability \\( \alpha \\) | response of the dipole to a uniform field; \\( \alpha(i\omega) \\) gives \\( C_6 \\) | polarizabilities, dispersion coefficients |
| \\( \partial N / \partial \lambda \\) | charge moved per unit constraint potential | constrained DFT |

These are the same physics viewed through different couplings. A code that
computes one well should be able to compute the others, and errors in one should
be diagnosable as errors in the others.

## The claim

The premise behind the architecture is that response is:

1. **Local in real space**: a density fluctuation here does not much affect the
   density far away, so the response should be sparse in a localized basis.
2. **Low-rank in its eigenspectrum**: the dielectric matrix has a small number
   of dominant eigenmodes, so it can be compressed without losing the physics.

If both hold, organizing the computation around response should make it
cheaper: *attenuate the operator, keep the dominant dielectric modes.*

## What is actually demonstrated

**Low rank: demonstrated, and narrower than it sounds.** PDEP compresses the
dielectric matrix to its dominant eigenpotentials in the RI auxiliary space,
and that works in production paths; see [RPA and GW](../methods/rpa-gw.md).
It does **not** remove the sum over empty states: ferric builds
\\( \chi_0 \\) by summing over every occupied–virtual pair, and PDEP compresses
what comes out of that sum.

**Locality: one positive result, still without a speedup.**

- **AO-sparse Laplace SOS-MP2.** Restricting each localized orbital's
  pseudo-density to an AO domain works: the radius needed does not grow with
  the molecule (exact for octane at the radius that is exact for butane;
  within 0.05% at 4 Bohr on a 71-atom drug molecule). The tensor algebra is
  still dense, so **no timing gain is claimed**. This result is specific to
  that formulation and does not carry over to the other locality lanes below.
  An earlier version of this page reported it as a measured negative; that was
  an indexing bug, now fixed and recorded in the test
  `sos_ao_sparse_truncation_radius_is_transferable_across_sizes`.
- **Local MP2 (amplitude threshold)** has localized virtuals and per-pair
  domain-local RI fits, but its J build is still dense (from RI), so **no
  scaling claim is made**.
- **RI-Laplace MP2** is dense; it is the correctness reference for the AO
  formulation, not a reduced-scaling path.

The locality premise is therefore partly supported and not yet cashed in as
cost. Measured limits are kept on
[What is validated](../reference/validation.md#known-limits-and-negatives).

## Why this framing is useful anyway

Even where the scaling payoff has not arrived, the response framing makes the
*error* in one method diagnosable through another.

MP2's dispersion is the clearest case. Its dispersion energy is built from an
**uncoupled** (uncoupled Hartree–Fock) response, in which the density
fluctuation does not feel the field it creates. For systems with low-lying,
highly polarizable excitations, such as π-stacked and other π systems, that
uncoupled response over-polarizes, and MP2 overbinds. This is documented in the
literature that replaces MP2's uncoupled dispersion with a coupled one
(Cybulski & Lytle 2007; Heßelmann 2008; Pitoňák & Heßelmann 2010). The size of
that coupling correction varies from system to system, and ferric has no
coupled-dispersion (MP2C-style) implementation of its own, so read this as the
literature's diagnosis, not a ferric measurement.

Attenuated MP2 takes a blunter route: it removes the long-range correlation
altogether and relies on a fitted short-range operator plus the basis-set
error it is fitted in. That is why it has zero asymptotic \\( C_6 \\), why its
parameters belong to one basis, and why MP2-V adds long-range dispersion back
through VV10. RS-MP2 + LR-RPA puts the long-range part back through response
instead. See [The MP2 family](../methods/mp2.md).
