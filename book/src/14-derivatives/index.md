# 14. SCF gradients, Hessians and geometry

Nuclear derivatives of the self-consistent-field energy and what is built on
them: the Hartree–Fock gradient, the gradient of density-fitted Coulomb and
exchange, the exchange-correlation gradient, the Hartree–Fock and
Kohn–Sham Hessians with their coupled-perturbed equations, harmonic
frequencies, the gradient of an SCF that carries a caller-supplied energy
term, and the searches that use the gradient (minimum, saddle point,
reaction path).

Correlated gradients (MP2, OO-MP2, dRPA) and the Z-vector are chapter 15.
They reuse the energy-weighted density and the integral-derivative loops
described here.

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
