# BSE-TDA oscillator-strength pilot (Phase 1, 2026-07-19)

**What this is:** a 3-molecule pipeline-proof pilot for BSE-TDA[G0W0@HF]
excitation energies + the NEW oscillator-strength computation
(`ferric_gw::bse::tda_oscillator_strengths`, see `crates/ferric-gw/src/bse.rs`
and the PySCF cross-check in `crates/ferric-gw/tests/bse_oscillator_strength.rs`).

**What this is NOT:** the full Thiel-set-anchored, multi-molecule statistical
benchmark that `docs/VALIDATION.md`'s BSE-TDA row scoped as "a 1-2 week
effort." That remains open — see `docs/bse-tda-benchmark-plan.md` for what a
real Phase 2 needs (more molecules, literature reference compilation,
multi-root state matching, proper MAE/statistics). Do not cite this README's
3-molecule table as if it were that benchmark.

## Setup

- Method: `BSE-TDA[G0W0@HF]`, closed-shell, `frozen_core = 0`, `trunc_thresh
  = 0.0` (full-rank PDEP screening, no truncation), `n_quad = 16`
  Gauss-Legendre, `eigensolver_conv_thresh = 1e-7` — same knobs as the
  existing `examples/water-bse-tda.toml` gate test.
- Basis: cc-pVDZ orbital / cc-pvdz-ri auxiliary throughout. **No diffuse
  functions** — see the water caveat below, this matters.
- Run via `benchmarks/bse-tda-pilot/run_pilot.sh` (invokes the release CLI
  binary on `examples/{water,c2h4,h2co}-bse-tda.toml`); raw output in
  `results.txt` in this directory.
- Molecules: water (`testdata/molecules/water.xyz`, pre-existing), ethylene
  (`testdata/molecules/c2h4.xyz`, pre-existing), formaldehyde
  (`testdata/molecules/h2co.xyz`, new — standard experimental-derived
  geometry, r(C=O)=1.200 Å, r(C-H)=1.105 Å, H-C-H=116.3°, C2v). Picked
  because: (a) all three are canonical small-molecule photochemistry
  textbook cases with well-documented literature vertical excitations, (b)
  cheap enough to run BSE-TDA (G0W0 on every MO + full PDEP screening) in
  seconds on this basis, (c) between them they exercise a bright
  Rydberg-like state (water), a strong valence π→π* (ethylene), and a
  symmetry-forbidden dark n→π* state (formaldehyde) — three qualitatively
  different oscillator-strength regimes in one pass.

## Results

| Molecule | Lowest state character | ferric Ω₁ (eV) | ferric f₁ | Literature Ω (eV) | Literature f | Source |
|---|---|---|---|---|---|---|
| H₂O | ¹B₁ (n→3s Rydberg) | 8.457 | 0.0269 | 7.62 (exFCI, aug-cc-pVTZ+diffuse) | not compiled here | Loos-group exFCI/CC benchmark literature (see caveat below) |
| C₂H₄ | ¹B₁ᵤ (π→π* valence) | 8.973 | 0.635 | 7.90–7.92 (CC3/CCSDT/FCI-CBS, aug-cc-pVTZ) | 0.34–0.35 | Schreiber/Silva-Junior/Sauer/Thiel JCP 2008 (TBE); cross-confirmed by later CCSDT/FCI-CBS work |
| H₂CO | ¹A₂ (n→π*, symmetry-forbidden) | 4.598 | 0.0000 | 3.97 (exFCI/AVTZ-corrected TBE) | 0 (forbidden by symmetry, both sides) | Loos et al., "Mountaineering Strategy," JCTC 2021 (arXiv:2011.08509) |

Errors vs the literature TBE/exFCI energies (signed, ferric − reference):

| Molecule | ΔΩ₁ (eV) | Comment |
|---|---|---|
| H₂O | +0.84 | **Not a like-for-like comparison** — see caveat below |
| C₂H₄ | +1.05 to +1.07 | Consistent with the known G0W0@HF-BSE overshoot direction |
| H₂CO | +0.63 | Same direction, forbidden state correctly reproduced (f=0 both sides) |

MAE over the 2 basis-consistent points (ethylene, formaldehyde; water
excluded, see below): **~0.85 eV**.

## Honest assessment

**This is a 3-molecule pilot, not a statistically meaningful benchmark.** A
MAE computed from 2-3 points carries essentially no statistical power — it
is reported here only to show the sign and rough magnitude of the expected
systematic bias, not as a validated accuracy claim.

**Direction and magnitude are physically sane and consistent with what's
already documented for this exact pipeline.** `docs/VALIDATION.md` and
`docs/bse-tda-water-gap-investigation.md` already establish that ferric's
BSE-TDA excitation energies are GW-gap-limited: G0W0@HF systematically
overshoots the true (experimental/high-level-correlated) gap because HF is
a poor starting point (no correlation, too-large HOMO-LUMO gap carried
through to the QP energies). The +0.6 to +1.1 eV overshoot seen here across
all three molecules is the same well-understood bias, not a new bug. This
pilot does not fix that bias (fixing it — e.g. moving to a G0W0@PBE or
self-consistent starting point — is out of scope for this task, which is
about oscillator strengths, not the GW starting-point question).

**Formaldehyde's symmetry-forbidden state is the cleanest validation point
here.** Both ferric and the literature TBE agree the lowest H₂CO singlet is
dark (f≈0, forbidden by C2v symmetry: ground state A₁, excited state A₂).
Getting f=0.0000 right on a real, non-water molecule (not just the
oscillator-strength formula's own PySCF cross-check) is a genuine, if small,
independent physicality check on the new oscillator-strength code beyond
the numerical validation in `bse_oscillator_strength.rs`.

**Ethylene's oscillator strength (0.635) is roughly 2x the CC3/CCSDT
literature value (~0.34-0.35).** This is a real, flagged discrepancy, not
swept under the rug. Two candidate explanations, neither investigated
further here (out of scope for this task):
1. TDA/CIS-level oscillator strengths on a G0W0@HF QP spectrum are known in
   the broader literature to be less reliable than the excitation energies
   themselves (the TDA is known to violate the TRK sum rule — see the
   `docs/VALIDATION.md` BSE-TDA row's existing GW-gap caveat, which applies
   to energies; nothing in this codebase has validated BSE-TDA oscillator
   strengths against literature on multiply-bonded/π systems before now).
2. cc-pVDZ (no diffuse/polarization-rich functions) is a modest basis for a
   π→π* valence state's transition dipole magnitude; the literature TBE
   values were computed at aug-cc-pVTZ or larger.
   A real Phase 2 basis-convergence check would help distinguish these.

**Water is excluded from the MAE because it is not a like-for-like
comparison.** The literature value (7.62 eV) is for a **Rydberg** state
(n→3s) computed with diffuse basis functions (aug-cc-pVTZ or larger).
cc-pVDZ has **no diffuse functions at all** — it structurally cannot
represent a 3s Rydberg orbital, so ferric's 8.457 eV number here is
whatever a Rydberg-starved cc-pVDZ virtual space produces, not a
basis-converged answer to the same physical question. Comparing it directly
to the diffuse-basis literature value would be scientifically dishonest.
This is exactly the kind of geometry/basis mismatch a real Phase 2 sweep
needs to control for (see `docs/bse-tda-benchmark-plan.md`).

## Reproduce

```
bash benchmarks/bse-tda-pilot/run_pilot.sh
```

Takes well under a minute total on this basis (BSE-TDA at cc-pVDZ scale for
3-8 atom molecules with GW+PDEP is cheap — this is NOT representative of
Thiel-set-scale timing, see docs/bse-tda-benchmark-plan.md's cost estimate
for what larger molecules cost).
