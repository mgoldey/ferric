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
