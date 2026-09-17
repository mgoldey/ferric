# Hypotheses — UNCONSTRAINED SCF state SELECTION (written BEFORE any measurement)

Date: 2026-09-16. Branch `fix/scf-unconstrained-state-selection`, off
`fix/cdft-state-selection` (8def6ef6).

## The established defect (not under test here)

HeNe⁺/def2-SVP UHF at R = 2.0 Å, **no constraint at all**:

| code | E (Ha) | state | verdict |
|---|---|---|---|
| NWChem 7.2.2 | −130.505340538816 | σ hole (Ne 2p_z) | — |
| ORCA 6.1.1 | −130.505340538350 | σ hole | `STABPerform` → STABLE |
| PySCF 2.13.0 | −130.5053405386 | σ hole | `stability()` → stable |
| ferric | −130.50034664 | ²Π hole (Ne 2pπ) | UNSTABLE, 0.136 eV higher |

ORCA reproduced BOTH energies and independently called ferric's ²Π unstable
(λ_min = −0.00486901 Eh). ferric's own analysis gives λ_min = −4.8706726160e-3
Ha, matching PySCF's `gen_g_hop_uhf` to 3.8e-10. Integrals, basis and
thresholds are RULED OUT: ferric computes the right energy for the state it
finds; it finds the wrong state.

## A STRUCTURAL OBSERVATION MADE BEFORE MEASURING (source-read, not a result)

`RhfConfig` carries two guess knobs: `init_guess_density: Option<Array2<f64>>`
and `use_sad_guess: bool` (default **true**, routing to
`guess::minao_projection_guess`). `rhf.rs` honours both (rhf.rs ~line 682).
**`uhf.rs` and `rohf.rs` reference NEITHER field** (grep: zero hits). UHF's
guess block (uhf.rs ~line 223) calls `hcore_guess(...)` and DISCARDS the result
(`let _ = ...; // sanity check it succeeds`), then diagonalizes bare `h`.

So the UNCONSTRAINED open-shell path always starts from hcore, silently
ignoring a guess the caller explicitly requested. That is a defect in its own
right regardless of what it does to HeNe⁺, but whether it CAUSES the σ/π
mis-selection is exactly what Part 1 must measure, not assume.

## THE THREE COMPETING HYPOTHESES (pre-registered, and DISTINGUISHABLE)

**H-GUESS (physics: the hcore guess lands in the wrong basin, a better guess
does not).**
  The σ and ²Π states are separate SCF basins. hcore's density is so far from
  either that it falls into ²Π; a guess closer to the true density (MINAO/SAD,
  or a PySCF-imported converged density) falls into σ.
  OBSERVABLE: E is MULTI-VALUED across guesses. At least one guess reaches
  −130.5053405 (σ) and at least one reaches −130.5003466 (²Π), with the
  hcore run landing on ²Π. Each run is individually converged (ΔE, ΔP at
  threshold) and the two energies are separated by ≫ 1e-6 Ha, so the spread is
  NOT a convergence artifact. Within a basin, different guesses agree to
  ≲ 1e-8 Ha.

**H-DEGEN (physics: a degeneracy-ordering defect in the first occupation).**
  Ne 2p is 3-fold degenerate in the free atom. If ferric's first-iteration
  aufbau fill picks π over σ from a degenerate/near-degenerate eigenvalue
  cluster — and DIIS/MOM then entrenches it — then EVERY guess that presents
  the same near-degenerate cluster lands on ²Π.
  OBSERVABLE: E is essentially SINGLE-VALUED at −130.5003466 across ALL guesses
  including MINAO, SAD and the PySCF-imported density. Additionally, the guess
  Fock's frontier eigenvalues show a near-degenerate pair/triple (gap ≲ 1e-3
  Ha) whose ordering decides the hole, and the ferric hole orbital is π while
  the σ orbital is within that cluster.
  DISTINGUISHING NOTE: importing PySCF's CONVERGED σ density and still landing
  on ²Π would be near-decisive for H-DEGEN over H-GUESS, since that guess is
  already AT the σ answer — nothing about basin distance can explain it.

**H-ARTIFACT (the harness is measuring something other than what it says).**
  The runner mislabels states, compares energies at different geometries/bases,
  or the "guesses" are not actually distinct inputs (e.g. UHF ignoring
  `init_guess_density` means every row is secretly the same hcore run).
  OBSERVABLE: all rows identical to the last bit (|ΔE| = 0.0e0 exactly, not
  ~1e-10), i.e. bit-identical rather than merely close; or a row's reported
  nbf/nelec/E_nuc differs between rows that are supposed to share a system.
  MITIGATION, pre-committed: every row prints nbf, n_α, n_β, E_nuc and the
  guess's own tr(DS); and the table is only trusted once two rows differ at
  SOME digit. **Bit-identity across genuinely different guesses is a STOP
  condition, not a result** — it is the signature of the ignored-field bug
  swallowing the input, and would mean the table measures nothing.

H-GUESS and H-DEGEN predict OPPOSITE observables on the same statistic (spread
across guesses: large vs zero), so they are distinguishable. H-ARTIFACT is
distinguished from H-DEGEN by bit-identity vs near-identity: a real degeneracy
defect still produces runs that differ in the last digits (different guesses,
different DIIS paths, same basin), whereas an ignored input produces runs that
agree to every bit.

BOTH H-GUESS and H-DEGEN may hold on different systems. That is a legitimate
third outcome and will be reported as such.

## THE LOAD-BEARING QUESTION: HeNe⁺-specific or systemic?

Measured on ≥ 3 open-shell systems with near-degenerate frontier orbitals, each
against PySCF 2.13.0 run in the SAME basis and geometry:
HeNe⁺ (the known case), O₂ triplet, NO doublet, and a first-row diatomic cation
(N₂⁺ or CO⁺). For each: ferric's E from each guess vs PySCF's E, and PySCF's own
`stability()` verdict on its solution.

  * ferric wrong ONLY on HeNe⁺ ⇒ narrow bug; the fix is a safety net.
  * ferric wrong on ≥ 2 of 4 ⇒ SYSTEMIC guess defect; report loudly.

This split is pre-committed so a "it's just HeNe⁺" conclusion cannot be reached
by only ever running HeNe⁺.

## EXACTNESS ANCHOR (must pass before any verdict)

A guess that is ALREADY the converged answer must reproduce that answer: feed
ferric its OWN converged σ MOs via `solve_uhf_with_guess` and require E to come
back at the same state to ≤ 1e-9 Ha. This anchors the guess-injection harness —
if injecting the answer does not return the answer, no other row means anything.

## THE ANCHOR'S KNOWN BLIND SPOTS (stated so validation cannot hide behind them)

1. **A trivial-limit anchor is blind to defects proportional to the quantity it
   zeroes.** The self-guess anchor injects an already-converged density, so the
   SCF does ~zero work; any defect in how a FAR guess is processed (projection,
   normalization, orthogonalization) is invisible to it. Mitigation: the table
   includes guesses that are far from the answer (hcore, MINAO) AND reports
   iteration counts, so a row that "converged" in 1 iteration from a far guess
   is visibly suspect.

2. **A reference comparison at a SINGLE input is unfalsifiable regardless of
   tolerance.** Measured in this lane today: a ΔIP anchor matched PySCF to
   3e-13 Ha with ferric's solver DELETED. Mitigation, pre-committed: the final
   validation must respond to an input the fabricator does not read. Concretely
   the fix is asserted at **two bases** (def2-SVP and 6-31G) on HeNe⁺ and at a
   **second system**, so a constant-returning stub fails: the reference numbers
   differ between them and are not available to a hardcoded value.

## MUTATION PLAN (every new guard must be seen to FAIL)

Every guard added in Part 2 is mutation-tested IN THE FOREGROUND against a
deliberately broken version, and the observed failure output is recorded in the
commit message. A test never seen to fail is an assumption. Specifically
planned mutations:
  * the "strictly lower" acceptance predicate → `true` (accept a HIGHER state);
  * the UNSTABLE-only descent gate → descend on STABLE too;
  * the guess plumbing → ignore the supplied density again (must resurrect ²Π).

## TOO CLEAN IS A STOP CONDITION

If every guess agrees to machine precision on every system, that is the
H-ARTIFACT observable, not a physics result, and the task stops to audit the
harness. Real multi-solution landscapes have basins with edges.

---

# OUTCOME (appended 2026-09-17, after the work — the text above is unchanged)

## Verdicts on the pre-registered hypotheses

**H-GUESS: CONFIRMED.** On the UNCONSTRAINED HeNe⁺ solve,
E is multi-valued across guesses (spread 4.99e-3 Ha, not bit-identical), MINAO
and SAD reach −130.5053405386 (the NWChem/ORCA/PySCF σ state) and report STABLE,
while hcore reaches −130.5003466413 and reports UNSTABLE.

**H-DEGEN: REFUTED.** Its observable was "E single-valued at ²Π across ALL
guesses, including one built from the converged σ density". Not observed. Given
a decent guess ferric's aufbau fill picks σ correctly, so there is no
degeneracy-ordering bug. Claiming one would have been the lane's seventh
retraction; it was not claimed.

**H-ARTIFACT: REFUTED.** The rows are not bit-identical and the guesses
demonstrably reach different basins.

## Is it HeNe⁺-specific or systemic? — SYSTEMIC

Pre-committed split: wrong on ≥2 systems ⇒ systemic. Measured **4 of 7** wrong
(HeNe⁺ at def2-SVP and 6-31G, OH/6-31G, N₂⁺/6-31G), every one flagged UNSTABLE
by ferric's own check, with errors from 0.126 eV to 10.37 eV. OH was found by
`ferric-dft`'s own f_xc FD test failing, not by looking — so the sweep in
`scf_state_selection.rs` was an UNDER-count, not an over-count.

## The anchor's blind spots, as handled

1. *Trivial-limit blindness.* The self-guess anchor does ~zero SCF work. Handled
   by printing iteration counts: the "self" row's 2 iters against 14–16 for
   every far guess is what that blind spot looks like when made visible.
2. *Single-input unfalsifiability.* Handled as pre-committed:
   `the_fixed_path_reaches_every_reference` asserts against SIX references at
   TWO bases. No constant satisfies −130.5053405386, −130.6043266127,
   −149.5455745334 and −108.3186843228 at once.

## Where the pre-registration was WRONG, recorded rather than quietly fixed

The hypotheses above assumed the σ/π hole could be read off the α SOMO. It
cannot: for HeNe⁺ the α SOMO is He-1s/2s dominated in BOTH states, so that
classifier labelled the ²Π state "sigma" and carried no information. The hole is
the β LUMO. Caught by `the_hole_classifier_separates_the_two_hene_states`, which
pins the classifier against both states so a non-discriminating label fails
rather than decorating a table.

## MUTATION LEDGER — every new guard, seen to fail

| # | mutation | guard | observed |
|---|---|---|---|
| 1 | `use_sad_guess` branch disabled (`false &&`) | `hene_reaches_sigma_and_is_stable_at_the_default` | FAILED — ²Π resurrected at −130.5003466413, UNSTABLE |
| 2 | descend on STABLE points too (`false &&` on the verdict gate) | `descent_off_is_bit_identical_to_no_descent` | FAILED — energies print identically at 12 digits yet differ in the last bits, which is why bit-identity and not a tolerance is asserted |
| 3 | `accepts_candidate` → `true` (accept a HIGHER state) | `descent_never_accepts_a_higher_state` | FAILED — and every OTHER test stayed green, confirming the guard is unreachable through the SCF path and must be unit-tested |
| 4 | σ/σ counterexample threshold 1e-4 → 1e-30 | `a_sigma_sigma_hole_pair_still_has_tiny_overlap` | FAILED — "the sigma/sigma counterexample is GONE" |
| 5 | `S_ab` product over α only | `s_ab_is_carried_by_a_single_beta_singular_value` | FAILED — 4.146e-7 vs 9.804e-1, rel 2.36e6 |
| 6 | He₂⁺ `S_ab` forced non-monotone | `he2_plus_s_ab_is_monotone_and_unchanged` | FAILED — reproduces the exact 0.0135/0.0038/0.0076 signature a real basin artifact would show |
| 7 | `FD_FLOOR` 1e-7 → 1e-30 (order checks never skip) | `gga_fxc_matches_finite_difference_of_vxc` | FAILED — proves the new gate is live, not a no-op |
| 8 | descent fabricates an `Unstable` verdict when none exists | `the_descent_skips_a_reference_it_cannot_analyse` | **SURVIVED first**, then FAILED after the test was strengthened — see below |

**Mutation 8 is the ledger's most useful entry.** The test originally asserted
only `stability.is_none()`, which the mutation leaves true, so it passed a
descent that fabricates verdicts. The assertion was changed to a BEHAVIOURAL one
— turning the descent on must leave the energy bit-identical on a reference
whose stability was never computed — and the mutation then failed. A test that
checks a precondition is not a test of the behaviour that reads it.

## TOO CLEAN was triggered once, and audited rather than reported

The f_xc FD residual improved from 6.1e-4 to 1.1e-9 — six orders, which is the
stop condition. Audited rather than written up: it traced to OH/6-31G having
been solved at a saddle 4.22 eV above the minimum, so the "improvement" is a
finite-difference check finally being evaluated at a correct density. That audit
is what found the fourth wrong system.

## PROCESS FINDING: a truncated suite reports nothing, not "no failures"

`cargo test -p ferric-scf --tests` builds 91 test binaries, several of the
`cosx_*` ones taking 8–10 minutes each on a loaded box. Under a 7000 s timeout
it completed **10 of 91** and printed `0 failed` for those ten — which reads
exactly like a green suite.

Five more premise-moved tests (3 in `scf_stability.rs`, 2 in
`scf_stability_wiring.rs`, including
`both_verdicts_are_reachable_through_the_config_flag`, whose entire purpose is
that both verdicts BE reachable) were sitting in the unreached 81 and were found
only by running the suites individually.

**The rule this lane adds**: when a change touches a core path, sweep the test
binaries ONE AT A TIME and count how many produced a result. A run that stops
early is missing data, not passing. `grep -c "test result"` against the number
of test files is the check; anything less than the file count means the sweep is
incomplete and its silence is not evidence.
