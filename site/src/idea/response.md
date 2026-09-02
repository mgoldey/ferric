# Electronic response

`ferric` is organized around **electronic response** — how the electron density
reacts to a perturbation. Standard quantum-chemistry codes are usually organized
around a hierarchy of *wavefunction ansätze* (HF → MP2 → CCSD → CCSD(T)). That
is a perfectly good organizing principle. It is not the one used here.

The object of interest appears under several names depending on which
perturbation you apply:

| Perturbation | Response object |
|---|---|
| Uniform electric field | Polarizability \\( \alpha \\) |
| Density fluctuation | Susceptibility \\( \chi \\) |
| Screened Coulomb interaction | Dielectric function \\( \varepsilon \\) |
| Constraint potential | \\( \partial N / \partial \lambda \\) |

These are the same physics viewed through different couplings. A code that
computes one well should be able to compute the others, and errors in one should
be diagnosable as errors in the others.

## The claim

The premise motivating the architecture is that response is:

1. **Local in real space** — a density fluctuation here does not much affect the
   density far away, so the response matrix should be sparse in a localized
   basis.
2. **Low-rank in its eigenspectrum** — the dielectric matrix has a small number
   of dominant eigenmodes, so it can be compressed without losing the physics.

If both hold, then organizing the computation around response should make it
cheaper: *attenuate the operator, keep the dominant dielectric modes.*

## What is actually demonstrated

This is where honesty matters more than the pitch.

**The low-rank half is demonstrated.** PDEP's compression of the dielectric
matrix works and is used in production paths — see [RPA and GW](../methods/rpa-gw.md).
Keeping only the dominant eigenmodes removes the explicit sum over empty states
that conventional RPA and GW require.

**The real-space locality half remains a design premise, not a measured
result.** Several attempts to exploit it are implemented and **measured
negative**:

- The **AO-sparse Laplace SOS-MP2** variant's truncation radius tracks the
  molecular diameter instead of saturating — so it is not a reduced-scaling
  path.
- **Local MP2 (amplitude-threshold)** is implemented with localized virtuals and
  per-pair domain-local RI fits, but the J build is still dense-from-RI, so **no
  scaling claim is made**.
- **RI-Laplace MP2** is dense; it serves as a correctness reference for the AO
  formulation, not as an O(N) path.

Those are reported as negative results rather than quietly omitted, because a
locality claim that has not survived measurement is not a feature.

## Why this framing is useful anyway

Even where the scaling payoff has not materialized, the response framing buys
something concrete: it makes the *error* in one method diagnosable through
another.

MP2's dispersion error is the clearest case. MP2 builds dispersion from an
**uncoupled** polarizability, which over-polarizes — giving \\( C_6 \\)
coefficients that are too large and overestimated π-stacking. That is not a
mysterious failure of a wavefunction ansatz; it is a specific, identifiable
defect in a response function, and it suggests a specific fix: attenuate the
correlation operator so the over-polarizing long-range part is damped.

That is [attenuated MP2](../methods/mp2.md), and it works for a reason the
response picture predicts.
