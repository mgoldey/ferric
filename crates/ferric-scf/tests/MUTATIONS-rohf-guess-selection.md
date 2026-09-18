# Mutation ledger — fix/rohf-guess-selection

Every mutation run in the FOREGROUND, `git status` clean after each cycle,
and every one committed BEFORE mutating (one earlier `git checkout` in this
session did lose an uncommitted test — see M-restore note).

| # | Mutation | Observed | Verdict |
|---|---|---|---|
| 1 | Revert the fix: force `rohf_guess_mos` to return `None` (restore `let _ = hcore_guess`) | **6 of 8 FAILED**: `the_fixed_rohf_path_reaches_every_reference`, `cn_is_the_system_the_guess_fix_costs`, `ferric_finds_lower_rohf_states_than_pyscfs_standard_guesses`, `the_guess_fix_repairs_two_non_convergences`, `an_explicit_init_guess_density_reaches_the_rohf_solver`, `a_misshaped_init_guess_density_is_rejected`. Survivors: `hcore_config_reproduces_the_pre_fix_answer_bit_identically` (correct — it pins the OLD path) and `rohf_guess_by_system_table` (correct — a measurement, asserts only H-ARTIFACT) | KILLED |
| 2 | Occupation spin split → UHF's `D/2` | **ALL 8 PASSED.** Energies identical to 1–2 ulp (HeNe⁺/6-31G −130.60332290752973 vs …967) | **SURVIVED → docstring claim RETRACTED** (`cfc3072f`) |
| 3 | `roothaan_fock(...)` → bare `F_α` | **ALL 8 PASSED**, same states throughout | **SURVIVED → docstring claim RETRACTED** (`cfc3072f`) |
| 4 | Misshape guard: `Err(...)` → `Ok(None)` (silent fallback) | `a_misshaped_init_guess_density_is_rejected` FAILED: "a misshaped init_guess_density must be an error: -75.20372495290783" | KILLED |
| 5 | Ignore `use_sad_guess == false` (always MINAO) | **5 FAILED**, incl. `hcore_config_reproduces_the_pre_fix_answer_bit_identically` ("no longer reproduces the pre-fix hcore answer (-75.2037249529); got -75.3618462891") and the H-ARTIFACT check in the table | KILLED |
| 6 | Perturb `F2P_PYSCF_HCORE` by 1e-7 | `ferrics_pre_fix_state_matches_pyscfs_hcore_state_on_two_of_four_systems` FAILED naming both values | KILLED |
| 7 | Second assert in that test: compare hcore against itself (make it a tautology) | FAILED: "the MINAO run must be LOWER than this shared hcore state, otherwise there was nothing to corroborate" | KILLED — **guard proven REACHABLE, not a tautology** |
| 8 | H-DEGEN test: source CN's "correct" density from the MINAO run (i.e. feed back a WRONG density) | FAILED: "the run used to SOURCE the correct density is not at the reference (-92.1186235242 vs -92.1397655314) — this test's premise is broken" | KILLED — premise check is live |
| 9 | H-DEGEN test: drop `init_guess_density` | FAILED: "ROHF did NOT stay at a density that was already correct (-75.3618462891 -> -75.2037249529)" | KILLED — **proves the test WOULD detect an ordering defect if one existed** |

## The two survivors are the most valuable result

M2 and M3 refuted two design justifications I had written into `rohf_guess_mos`'s
docstring before testing them. Both choices were KEPT (they are the internally
consistent ones — they match this file's own occupation convention and Fock
combination) but the docstring now states plainly that **no accuracy claim rests
on either**, that changing either is a refactor rather than a correctness fix,
and that a system where they differ belongs in the sweep if one is found.

A docstring rationale that has never been mutation-tested is an assumption, and
this one would have been cited by a later session as settled design.

## Two additional refutations found outside the mutation table

* **"MINAO converges more slowly on the DF path"** — the plausible explanation
  for raising one test's `max_iter`. A regression test asserting it FAILED
  (48 vs 154 iterations). The ordering reverses on a 6e-5 Å geometry change and
  again with convergence thresholds; it is DIIS-path chaos, not a guess property
  (`e2b821ce`).
* **"ferric's pre-fix answers are bit-equal to PySCF's hcore answers"** — true on
  F₂⁺ (7.3e-12) and HeNe⁺ (3.1e-10), FALSE on OH (1.6e-1) and NH₂ (6.6e-2),
  where PySCF's hcore reaches the CORRECT state and ferric's did not. Half the
  repaired set is explained by the guess; the other half is repaired empirically
  with the mechanism unexplained (`af6987c9`).

## M-restore note

After M7 I ran `git checkout` on the test file to revert the mutation, which
also destroyed the (uncommitted) test that mutation was checking. Re-added from
the transcript and committed immediately. The lesson the task stated is real:
**commit before mutating**, and prefer a targeted revert of the mutated lines
over `git checkout <file>`.

## ADDENDUM 2026-09-18 — row 8's quoted MINAO energy is now historical

Row 8 quotes CN/6-31G MINAO at `-92.1186235242`, the +0.575 eV penalised
value that held when this ledger was recorded. PR #83 (`a268fc3c`,
spherically symmetrizing the MINAO atomic block) removed that penalty:
MINAO now reaches `-92.1397654895`, the same state as hcore, +0.0000 eV
against the reference. CN/cc-pVDZ's +0.386 eV penalty is gone the same way.

The mutation itself still stands — feeding back a wrong density still trips
the premise check. Only the specific number it produced has moved, because
the defect that produced it was fixed. The table above is left as recorded;
`cn_is_the_system_the_guess_fix_costs` carries the current figures and now
asserts the REPAIRED behaviour, so a regression re-introducing the penalty
fails there.
