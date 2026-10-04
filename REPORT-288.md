# REPORT — issue #288: ω tuning has no control over the cation state

## Part 1: hypotheses, written and committed BEFORE any measurement


Date: 2026-10-03. Branch `fix/omega-tuning-cation-state` off `origin/main@0b4fa7dc`.
Nothing in this file has been measured yet. It exists so the conclusion cannot
be fitted to the data after the fact.

## What is being changed

`omega_tuning::eval_j` solves the doublet cation from the default guess (MINAO
projection, `uhf_guess_mos`) at every ω, and checks only `converged`.
`tune_omega`'s golden-section search visits ω out of order. So the cation state
is whatever the symmetric guess happens to reach at each ω independently, and
two J points in one curve can sit on two different branches with no warning.

Planned deliverable: continuation seeding (seed each SCF from the converged
orbitals of the NEAREST already-evaluated ω) plus a branch-consistency check
built from quantities available without a stability operator.

## Scope: item 3 is out

Issue task 3 (`uhf_internal_stability` on each cation) depends on the separate
open issue "Range-separated hybrids: orbital-Hessian matvec ...":
`stability::ks_reference_is_analysable` refuses ω ≠ 0 with
`StabilitySkip::RangeSeparated`. No ω≠0 orbital Hessian is written here.

## Physics hypotheses (P)

* **P1 — the branch exists and continuation holds it.** For N2/def2-SVP
  ωB97X-V the symmetric D∞h ²Σg⁺ N2⁺ stops being a minimum between ω = 0.53
  and 0.56 (PySCF `symmetric_cation_probe`: λ_min +1.13e-4 at 0.53, −2.71e-3
  at 0.56). If a lower, symmetry-broken (hole-localized) state exists past the
  onset and ferric's UKS can reach it, then seeding ω = 0.60 from a converged
  ω = 0.56-or-above broken cation gives E(0.60) **below** the symmetric
  E(0.60), with a visibly non-zero per-atom spin-population asymmetry
  |p_N1 − p_N2| on the two symmetry-equivalent nitrogens.
* **P2 — J gains a kink at the onset.** If branches are mixed, J(ω) sampled
  densely across 0.50–0.60 shows a slope change or a non-monotone step at the
  onset; continuation on one branch removes it from that branch's curve.
* **P3 — ⟨S²⟩ is a WEAK discriminator here.** PySCF's own probe gives
  ⟨S²⟩ = 0.7559 (ω 0.53) and 0.7563 (ω 0.56) — a 3e-4 change across the onset,
  which is the same size as the smooth ω-drift of ⟨S²⟩ over 0.2→0.5
  (0.7532→0.7556). So a ⟨S²⟩ threshold alone cannot separate "hole localized"
  from "ω moved": the spatial spin-population asymmetry is the discriminator
  that can, because the symmetric state has p_N1 = p_N2 by symmetry and the
  broken one does not. Prediction: the asymmetry separates the branches by
  orders of magnitude while ⟨S²⟩ separates them by <1e-3.
* **P4 — the default guess is symmetric, so today's code stays symmetric.**
  The MINAO projection guess for N2 respects D∞h; the β HOMO/LUMO mixing in
  `solve_uhf_fockmod` only applies when nocc_a == nocc_b, which a doublet is
  not. So today's `eval_j` at ω = 0.60 should land on the SYMMETRIC saddle,
  p_N1 ≈ p_N2 to grid noise, and its J should lie on the symmetric branch
  (continuing the 0.2…0.5 trend to J ≈ 0 near 0.56), NOT on the broken branch.

## Artifact hypotheses (A) — what a broken implementation would show

* **A1 — seeding does nothing (silent no-op).** If `solve_uhf_with_guess` is
  handed MOs that are then discarded (wrong shape silently ignored, wrong
  argument position, the seed overwritten by `init_guess_density`), then
  continuation-on vs continuation-off give **bit-identical** energies at every
  ω, including past the onset. P1 predicts a difference past the onset;
  A1 predicts zero difference everywhere. **These are distinguishable**, so
  the experiment is valid — but only if the no-difference case is reported as
  a no-op suspicion, not as "the branch does not exist".
  *Discriminator:* the seed must also be shown to MATTER somewhere, i.e. a
  deliberately wrong seed must change the result. If no seed ever changes
  anything, the plumbing is dead, not the physics.
* **A2 — continuation only ever seeds from the previous call.** Golden section
  visits ω out of order, so "previous eval" ≠ "nearest ω". If I accidentally
  implement previous-eval seeding, the N2 ω-ordered sweep (monotone ω) cannot
  tell the two apart — both are the same there. **The monotone sweep CANNOT
  distinguish A2 from the intended design and must not be used as its
  evidence.** The nearest-ω selector therefore needs a direct unit test on a
  synthetic out-of-order ω list, not an SCF test.
* **A3 — the asymmetry metric is an artifact of the AO partition.** A
  Mulliken/Löwdin spin population on symmetry-equivalent atoms could be
  non-zero for a symmetric density if the per-atom AO assignment were wrong
  (shell→atom mapping off by one). *Discriminator:* a symmetric state must
  give |p_N1 − p_N2| at the 1e-10 level, i.e. the metric's own trivial limit.
  If the symmetric state shows a "large" asymmetry, the metric is broken, not
  the state.
* **A4 — the flag fires on everything (vacuous alarm).** A branch threshold set
  below the ω-drift of the metric flags every adjacent pair, which looks like
  "the check works". *Discriminator:* H2O and NH3 across 0.2–0.6 must produce
  ZERO flags; the threshold must be derived from the measured both-sides
  separation, never guessed.

## Exactness anchors (written before the sweep, must pass before it runs)

1. `continuation disabled` ⇒ every J, ε_HOMO, IP is **bit-identical** to the
   pre-change code (default struct field is off ⇒ the old code path).
2. The asymmetry metric on a converged SYMMETRIC N2⁺ is ≤ 1e-8 (trivial limit
   of "hole localization": no localization).
3. The nearest-ω selector returns the only candidate when exactly one exists,
   and `None` on the first evaluation (no neighbour) ⇒ default guess ⇒ anchor 1
   covers the first eval for free.

## Pass-condition reachability

The N2 test's GO condition is "the tool flags the 0.50–0.60 window". It is
reachable only if ferric's branch metric actually separates adjacent ω by more
than its ω-drift. That separation is measured FIRST (the table in the report);
the threshold is then placed between the two measured sides. If the measured
separation is not there, the test is NOT written to pass anyway — the negative
is reported, with the dependency on the ω≠0 Hessian named as the reason a
stability verdict would have been the right instrument.
