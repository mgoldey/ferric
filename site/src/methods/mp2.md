# The MP2 family

The largest method family here, and the one most directly tied to the
[response](../idea/response.md) framing.

## Why attenuation

MP2 builds dispersion from an **uncoupled** polarizability — the density
fluctuation does not feel the field it creates. That over-polarizes, producing:

- \\( C_6 \\) coefficients that are too large
- overestimated π-stacking
- an error partly masked by BSSE in small basis sets, which is why the problem
  is easy to miss

Attenuating the correlation operator damps the long-range part where the
uncoupled approximation is worst. One parameter, and the error it targets is
identifiable rather than empirical.

## Variants

**RI-MP2** — density-fitted via 3-center/2-center integrals. **Canonical MP2**
is also implemented, for cross-validation rather than production.

**Attenuated RI-MP2** — \\( \mathrm{erfc}(\omega r)/r \\) and `terfc` operators
(Goldey & Head-Gordon, JPCL 2012).

**SCS-MP2** — Grimme spin-component scaling (JCP 2003), and
**SCS-MP2(2terfc)**, dual-attenuated (Goldey, Dutoi & Head-Gordon, PCCP 2013).

**OO-RI-MP2** — orbital-optimized, with level-shifted Newton, orbital DIIS,
Cayley rotations and backtracking.

**MP3** — spin-orbital third-order Møller–Plesset via the `einsum!` framework.

**MP2-V** — attenuated MP2 combined with VV10 nonlocal correlation.

## Laplace formulations

**RI-Laplace MP2** — AO-Laplace via pseudo-density matrices. The implementation
is **dense**; it is the correctness reference for that formulation, not a
reduced-scaling path. No O(N) has been measured.

**Laplace SOS-MP2** — opposite-spin-only with a Laplace-factorized denominator
(`c_os` scaling, minimax quadrature). The MO and AO formulations agree to
machine precision.

The **AO-sparse variant is measured negative**: its truncation radius tracks the
molecular diameter instead of saturating, so it does not deliver reduced
scaling. That is reported rather than omitted.

## Local MP2

**Amplitude-threshold LMP2** — WSHG23 single-threshold, with localized virtuals
and per-pair domain-local RI fits.

**Counters only — no scaling claim is made.** The J build is still
dense-from-RI, so while the amplitude machinery is in place and anchored, the
end-to-end cost is not reduced. The assembly step, not the solve, is the
measured wall.

## Size-extensivity

RI-MP2's total energy is size-extensive to **2e-12 Ha** for a well-separated
dimer versus twice the monomer — pinned by a test, not asserted. That is five
orders inside the test's tolerance, so it passes on physics rather than on a
loose bound.

## Robust fitting

When the RI metric differs from the physical kernel — as it does for attenuated
operators — **robust (Dunlap) density fitting is required**, not optional.
Domain-local \\( V^{-1} \\) plus robust fitting gives size-extensive µHa-level
MP2 error; the non-robust form collapses in a way driven by the metric, not by
the domain size.
