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
| 14.1 | Hartree–Fock nuclear gradient | `hf-gradient` | ready |
| 14.2 | Gradients of density-fitted J and K | none | gap |
| 14.3 | XC gradient: AO-derivative and grid-response terms | `ks-gradient` | ready |
| 14.4 | Hartree–Fock Hessian and coupled-perturbed equations | `hf-hessian` | ready |
| 14.5 | Kohn–Sham Hessian on a fixed and a moving grid | `ks-hessian` | ready |
| 14.6 | Harmonic frequencies | `harmonic-frequencies` | ready |
| 14.7 | Fock-hook gradients | none | edit |
| 14.8 | Geometry optimization, saddle points and reaction paths | none | gap |

Units throughout: Hartree, Bohr, atomic masses in u until a section says
otherwise. Gradients are `(natoms, 3)` arrays in Hartree/Bohr.
