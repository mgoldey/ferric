# Pre-registered hypotheses — ROHF unconstrained state selection

**Registered 2026-09-17, BEFORE any ROHF guess × system number in this lane was
measured.** Branch `fix/rohf-guess-selection`, based on `26b64d60`
(`fix/scf-unconstrained-state-selection`, the UHF half).

Everything below is written down first so that the measurement cannot be
retro-fitted to whichever answer came out. The only ROHF numbers that exist at
registration time are the four printed by `rohf_still_uses_hcore_measure_whether_it_matters`
(commit `8fe1e9df`, quoted in §4 as PRIOR DATA), which were produced by the UHF
lane and are the reason this lane exists.

---

## 1. The defect in source

`crates/ferric-scf/src/rohf.rs` references neither `RhfConfig::init_guess_density`
nor `RhfConfig::use_sad_guess`. Its only contact with `guess.rs` is

```rust
line  30: use crate::guess::hcore_guess;
line 279: let _ = hcore_guess(&s, &h, nocc_a.max(1))?;
```

`let _ =` — the guess density is computed and immediately dropped; the code then
diagonalizes bare `h`. `rhf.rs` has honoured both fields for a long time and
defaults `use_sad_guess = true` (the MINAO projection). `uhf.rs` carried the
identical pattern and was fixed at `b687c394`.

This is a statement about the SOURCE, verifiable by reading it, and is not in
question. What IS in question is everything below.

---

## 2. Hypotheses

### H-GUESS (physics)
ROHF's σ/π (or equivalent) states are separate SCF basins and the initial guess
selects one. The hcore guess falls into a higher basin on some systems; a better
density guess (MINAO) reaches the lower one.

**Observable:** E is MULTI-VALUED across guesses on the affected systems, and at
least one guess reaches the PySCF ROHF reference. Unaffected systems are
SINGLE-valued (all guesses agree).

### H-DEGEN (physics, alternative)
A degeneracy-ordering defect in the ROHF effective-Fock diagonalization fixes the
occupation at iteration 1 and nothing re-examines it. The UHF agent found no such
bug in UHF, but ROHF's effective Fock is a DIFFERENT operator (Roothaan coupling
of closed/open/virtual blocks), so it does not inherit that finding.

**Observable:** E is SINGLE-VALUED at the wrong state across ALL guesses,
**including a guess built from ferric's own converged-correct density** (from the
MINAO run, or from a converged UHF/RHF density). A guess that is already AT the
answer and still walks away from it cannot be a basin-selection effect.

### H-ARTIFACT (implementation, the one that must be excluded)
The guess rows are not actually different inputs — e.g. `use_sad_guess` is
threaded but never reaches the diagonalization, or the MINAO path errors and
silently falls back to hcore, so "both guesses" is one guess measured twice.

**Observable:** BIT-IDENTICAL energies across guess rows — `|ΔE|` exactly `0.0`,
not merely below a tolerance — on systems where H-GUESS predicts a spread.

### H-NOTAPPLICABLE (the honest null)
ROHF is materially LESS affected than UHF, or unaffected on most systems, because
the ROHF ansatz's restriction removes the symmetry-breaking freedom that the
lower state uses.

**Observable:** the guess fix changes few or no systems, and the prior OH /
HeNe⁺ deltas either do not reproduce or have a different cause.

### Distinguishability

| | E multi-valued across guesses? | Some guess reaches ref? | Bit-identical rows? |
|---|---|---|---|
| H-GUESS | yes | yes | no |
| H-DEGEN | no | no | no (near-identical, not bit) |
| H-ARTIFACT | no | n/a | **yes, exactly 0.0** |
| H-NOTAPPLICABLE | no | already there | no |

H-GUESS and H-DEGEN predict **opposite** spreads. H-DEGEN and H-ARTIFACT are
separated by bit-identity vs near-identity — this is why the test prints the raw
`ΔE` in Ha to full precision and compares against `0.0` exactly, never against a
tolerance. H-NOTAPPLICABLE is separated from H-DEGEN by whether the single value
is AT or ABOVE the reference.

**The observables are distinguishable.** No pair of hypotheses above predicts the
same row of that table.

---

## 3. The artifact hypothesis stated next to the physics hypothesis

*If the ROHF guess defect is real (physics), I expect:* at least one system where
the hcore row and the MINAO row differ by ≫ 1e-9 Ha, the MINAO row lands on an
independently-computed PySCF ROHF energy, and systems that were already right
stay right.

*If my implementation of the measurement is broken, I expect:* every row identical
(the config field never reaches the solver), OR every row moving by the same
amount (I changed something global, not the guess), OR the "MINAO" row matching
the reference on one system and nothing else (a single-input comparison, which is
unfalsifiable — measured in this lane's sibling: an anchor matched PySCF to 3e-13
Ha with ferric's solver DELETED).

`X == Y` for none of these. The experiment can distinguish them, so it may run.

**The defence against the single-input failure mode is structural, not
tolerance-based:** the fixed path is asserted against ROHF references for
**multiple distinct systems at two bases**. No constant can satisfy several
ROHF energies at once, so a fabricating or short-circuited implementation cannot
pass by returning a number.

---

## 4. PRIOR DATA (not measured by this lane; the reason it exists)

From `8fe1e9df`, ferric ROHF default path vs PySCF 2.13.0 ROHF:

```
OH/6-31G         -75.2037249529  vs  -75.3618462925   +4.3027 eV  <== ABOVE
HeNe+/def2-SVP  -130.4965141210  vs -130.5013033958   +0.1303 eV  <== ABOVE
N2+/6-31G       -108.2805841257  vs -108.2805842569   +0.0000 eV
O2/6-31G        -149.5279966531  vs -149.5279966339   -0.0000 eV
```

This lane must REPRODUCE these before trusting anything downstream of them, and
must treat the 2-of-4 count as a FLOOR, not a census — the UHF sweep's own OH row
surfaced only because an unrelated `ferric-dft` test failed, not from deliberate
search.

---

## 5. Scope decided in advance: is a stability descent available for ROHF?

**No.** Determined by reading the source before measuring, so it is not a
post-hoc excuse:

* `crates/ferric-scf/src/stability.rs` implements exactly two operators —
  `uhf_internal_stability` and `rhf_internal_stability`.
* `StabilitySkip::Rohf` exists specifically to record that the Roothaan
  open-shell Hessian is a **third** operator (`rohf_newton::hessian_matvec`, one
  MO set with closed/open/virtual blocks and Roothaan coupling), not a special
  case of either implemented one.
* `rohf.rs` at its convergence exit already prints that skip reason when
  `check_stability` is set, and returns `stability: None`.

Therefore the UHF fix's SECOND half (`scf_stability_descent`) is **unavailable**
to ROHF, and this lane will not fabricate one on the wrong operator. If the guess
fix alone leaves a system above its reference, that residue will be REPORTED, and
the existing skip-warning is the detectability mechanism — not silently left as a
converged wrong answer.

Pre-registered consequence: a residual is an EXPECTED possible outcome of this
lane, not a failure of it. Saying so in advance is what stops it being explained
away later.

---

## 6. What would REFUTE the fix

* MINAO rows that do not move any system → H-GUESS refuted for ROHF; report
  H-NOTAPPLICABLE and re-examine the prior OH/HeNe⁺ numbers rather than
  generalizing them.
* A system that was AT its reference before and is ABOVE it after → the fix costs
  more than it buys; must be reported, not hidden behind an average.
* Bit-identical guess rows → H-ARTIFACT; the measurement is broken and no
  physics claim may be made from it.
* Any system reaching the reference while a deliberately-broken solver ALSO
  reaches it (mutation test) → the assertion measures nothing.
