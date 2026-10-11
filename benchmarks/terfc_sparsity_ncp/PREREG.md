# Pre-registration: does the non-CP MP2-V operator give better amplitude sparsity? (before any number)

Operator under test (NCP optimum, stage-1 refit, in-sample, 8 A24 systems, aDZ):
r0 = 0.991 A, s = r0*omega = 1.354, b = 10.95. omega = s/r0 (verified against
a24-mp2v-refit/harness.py `omega_from`: r0 in A, omega = r0omega/r0 in A^-1):
omega = 1.3663 A^-1 = 0.72301 Bohr^-1 (r0 = 1.8727 Bohr). Harness spec "terfc:0.991:1.354".
Comparators: Coulomb, linked terfc (1.0, 0.7071), erfc w=0.42, secondary CP optimum
(1.408, 1.164). Protocol identical to benchmarks/terfc_locality_map (trivial maps,
6-31G / cc-pvdz-ri, frozen core = N carbons, relerr to same-operator eps=0 ri_mp2).

Physics hypothesis (H1, from the earlier map): locality at matched relative error is
set by kernel range R(0.1) (rho 0.92). NCP has R(0.1) slightly shorter than linked
(r0 smaller, seam sharper): predict keep% ~0.85-0.95x of linked at C8-C12, pairs and
domains unchanged within noise (saturated), E_att/E_Coul ~1.0-1.2, per-quartet cost
~ linked +15% (~25-28x Coulomb), no Schwarz bound.
Artifact hypothesis (H2): any gain is the energy-scale effect (E_att/E_Coul) or
interpolation bias; predict gain vanishes against a same-range erfc, or tracks
E_att/E_Coul. H1 and H2 differ in whether the gain survives (a) relative-error
matching (built into the criterion), (b) range-matched erfc comparison (Stage C).
Resolvable effect: confirmed-at-eps* differences < ~8% are NOT resolved
(interpolation bias 3-6%).
Claim gate for "improved sparsity": NCP/linked keep ratio below 0.92 at relerr 1e-3
AND 1e-2, same direction at C8, C10, C12 and in the tail (>=C20 if feasible), anchor
passing. A wall-time claim additionally requires wall time at matched relative error.
A null/negative result is a valid answer.
