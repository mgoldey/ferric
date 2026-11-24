# SCF gradients, Hessians and geometry {#derivatives}

A force on a nucleus is the slope of the energy along that nucleus's
coordinate, with the sign flipped. A geometry optimization follows the
forces downhill, and a frequency calculation reads the curvature at the
bottom. This chapter computes both for a self-consistent-field energy: the
Hartree–Fock gradient, the gradient of density-fitted Coulomb and exchange,
the exchange-correlation gradient, the Hartree–Fock and Kohn–Sham Hessians
with their coupled-perturbed equations, harmonic frequencies, the gradient
of an SCF that carries a caller-supplied energy term, and the searches that
use the gradient (minimum, saddle point, reaction path).

The first sections build on chapter @sec:integrals (derivative integrals, section @sec:integrals-derivatives),
chapter @sec:hartree-fock (the Hartree–Fock energy and its fitted J and K) and chapter @sec:kohn-sham
(the XC energy on a Becke grid). Correlated gradients (MP2, OO-MP2, dRPA)
and the Z-vector are chapter @sec:correlated-gradients. They reuse the energy-weighted density and
the integral-derivative loops described here.

| Section | Topic | Registry id | Status |
|---|---|---|---|
| @sec:hf-gradient | Hartree–Fock nuclear gradient | `hf-gradient`, `cosx-gradient` | ready |
| @sec:df-jk-gradient | Gradients of density-fitted J and K | `hf-gradient` | edit |
| @sec:xc-gradient | XC gradient: AO-derivative and grid-response terms | `ks-gradient`, `vv10-gradient` | ready |
| @sec:hf-hessian | Hartree–Fock Hessian and coupled-perturbed equations | `hf-hessian` | ready |
| @sec:ks-hessian | Kohn–Sham Hessian on a fixed and a moving grid | `ks-hessian` | ready |
| @sec:frequencies | Harmonic frequencies | `harmonic-frequencies`, `rrho-thermo`, `ir-intensities` | ready |
| @sec:fock-hook-gradient | Fock-hook gradients | `fock-hook-gradient` | edit |
| @sec:geometry-search | Geometry optimization, saddle points and reaction paths | `geometry-search` | edit |

Units throughout: Hartree and Bohr. Atomic masses are in u, except in
section @sec:frequencies, which converts them to electron masses before mass
weighting; the saddle search of section @sec:geometry-search keeps them in u. Gradients are `(natoms, 3)` arrays in
Hartree/Bohr.
