# terfc locality map: pre-registration (written before any production number)

Branch `research/terfc-locality-map`, 2026-10-10. Harness:
`crates/ferric-mp2/tests/terfc_locality_map.rs` (ignored research test).

## Question

Does relaxing the curvature link (omega = 1/(r0*sqrt2), r0*omega = 0.7071) of the
terfc attenuator make amplitude-threshold local MP2 more local? Operator:
terfc(r; r0, omega)/r (`Operator::terfc_with_omega`), 6-31G / cc-pvdz-ri,
frozen core = number of carbons, alkanes, SCF reference full Coulomb.

## Hypotheses (stated first, per the Experimental Protocol)

PHYSICS (H1). Amplitude sparsity at FIXED RELATIVE ACCURACY is set by the kernel's
range and tail: r0, and how fast the kernel dies beyond r0. omega only changes the
tail through the seam transition width (~1/omega). So any locality gain should come
from a SMALLER r0 and/or a SOFTER seam (smaller r0*omega), not a sharper one.
If H1 is real I expect, at matched relative error: keep%, mean domain and pair
count fall monotonically with r0 at fixed r0*omega; they are flat or slightly WORSE
for r0*omega = 2, 4 than for 0.7071 (sharp seam = truncated Coulomb, whose
k-space kernel oscillates as cos(q r0)); and softer seams (0.35) are sparser than
linked at equal r0.

ARTIFACT (H2). An apparent gain at fixed ABSOLUTE eps is produced by the
shrinking correlation-energy scale of a shorter-range operator (the same eps
removes a larger fraction of a smaller total; MP2 energy is quadratic in the
kernel, amplitude magnitudes linear), and by the single-threshold mask interacting
with that energy scale. If H2 is the explanation I expect: gains at fixed absolute
eps that VANISH (or shrink to the second digit) once eps is re-chosen so the
relative error to the SAME operator's eps=0 energy is the same; and keep% at the
matched-relative eps tracking E_att(eps=0)/E_Coulomb rather than r0.
X != Y holds: H1 predicts a gain surviving the matching, H2 predicts it does not.

Further artifact candidates, each with a control:
 * mask bookkeeping at eps=0: ANCHOR, direct eps=0 (trivial maps) vs ri_mp2 of the
   SAME operator <= 1e-9 Ha, asserted per operator before its sweep.
 * scale-invariance null control: a terfc with r0 -> huge (r0 = 10 A, r0*omega = 4)
   must reproduce the Coulomb row (energy ratio -> 1, same keep/dom/pairs at the
   same relative error). If it does not, the table seam / far field is the artifact.
 * sharp terfc energies are ~2x Coulomb (k-space overshoot, measured earlier). A
   larger energy scale makes a fixed absolute eps look TIGHTER; relative matching
   removes it, so the sharp-seam rows are where H2 bites hardest.
 * integral correctness for softer seams (r0*omega < 0.707) is UNVERIFIED (tables
   checked only for 0.707-16). Hypothesis: if the softer-seam integrals are wrong,
   I expect a mismatch vs the independent Fourier generator at the 1e-6 level or
   worse, a non-PD metric, or an energy outside the Coulomb-bounded range. The
   eps=0 anchor is BLIND to engine errors (shared engine), so it cannot clear
   them. Softer seams are excluded unless the Fourier check passes AND the metric
   is PD AND the energy is within a sane range.

## Criterion

For each operator: sweep eps in {1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5}
(extended if the target is not bracketed). Define relerr(eps) =
|E(eps) - E(0)| / |E(0)| with E(0) the SAME operator's eps=0 value (ri_mp2). Find
eps*(tol) for tol in {1e-3, 1e-2} by linear interpolation of log(relerr) vs
log(eps); report at eps*: keep fraction (%), mean per-LMO domain, retained pair
count, E_att(0)/E_Coulomb(0), plus per-quartet cost relative to Coulomb (eri3 wall
on the same shells) and wall of the local run. Interpolated rows get a confirming
run at the bracket nearest eps* when it becomes load-bearing.
Maps: trivial (aux_radius 1e6, no virt radius, ao_tail 0, schwarz_skip 0) so the
only truncation is the amplitude mask under test; production maps (CLI defaults
aux 10, virt 12, ao_tail 1e-3, batch_merge 4) used as a secondary check on the
best operators only.

## Grid

C8 (map): r0 in {0.5, 0.7, 1.0, 1.35} A x r0*omega in {0.35 (if validated), 0.7071,
2, 4}; controls: Coulomb, erfc(w = 1/r0) at matched tail length for r0 = 0.7, 1.0,
and the production erfc w = 0.42 A^-1; null control terfc r0=10 A.
C12: best 3-4 operators + Coulomb + linked r0=1.0. C10: size check. Cheapest-first.
Size trend = 3 points (C8/C10/C12): provisional, no scaling claim; per the lane's
rule, no negative is declared below the onset (~C20+).

## What would change my mind

H1 supported only if a gain at matched relative error exceeds the run-to-run
interpolation noise and tracks r0 (not E_att/E_Coulomb). If everything at matched
relative error collapses onto one curve versus E_att/E_Coulomb, H2.
