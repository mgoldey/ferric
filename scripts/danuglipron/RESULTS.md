# Danuglipron toxicity-reduction campaign — measurements

All numbers GFN2-xTB unless stated. Raw JSON in `out/`. Design in `PLAN.md`.

Measurement and interpretation are kept separate below, per CLAUDE.md's
Experimental Protocol: tables survive re-interpretation, verdicts calcify.

---

## M1. Toxicity baseline (2026-08-29, offline provider)

Source: RDKit `FilterCatalog` — Brenk, PAINS, NIH, ChEMBL/Glaxo/Dundee/BMS —
plus Lipinski/Veber. ADMETlab 3.0's documented `POST /api/admet` returned
**HTTP 404** on every path variant tried; ProTox-3.0 is reachable but exposes no
JSON API and is not screen-scraped by design. So the web tier contributed
nothing and the offline tier carried the measurement.

| endpoint | danuglipron |
|---|---|
| Brenk / PAINS / NIH / Glaxo / Dundee / BMS alerts | **0 hits each** |
| Lipinski violations | 1 of 4 (**MW 555.6 > 500**) |
| Veber violations | 0 of 2 (TPSA 113.5, rotB 9) |
| cLogP | 4.89 (limit 5) |

**Measured:** danuglipron trips zero structural alerts in six published
catalogs. Its only developability flag is molecular weight, with cLogP at the
rule-of-5 boundary.

**Interpretation (provisional):** the clinical failure was not a
structural-toxicophore problem, which is consistent with the public record —
Pfizer's April 2025 discontinuation followed a *single asymptomatic, reversible*
DILI case, with liver-enzyme elevations otherwise "in line with approved agents
in the class" across >1400 participants, while the program-defining problem was
dose-dependent GI intolerability. That reframes the tractable lever as
**exposure** (lower efficacious dose), not toxicophore deletion.

## M2. Conformer strain, free-solution scan (Arm A)

20/20 committed conformers relaxed in vacuum, 209 s total.

| quantity | value |
|---|---|
| free minimum | `conf_02_rdkit`, −117.32863093 Ha |
| ensemble spread | **10.87 kcal/mol** |
| bound cryo-EM pose (`conf_00_cryo_em`) | **+2.28 kcal/mol**, rank **3 of 20** |

**Measured:** the experimentally bound conformer sits 2.28 kcal/mol above the
global free minimum found, third-lowest of twenty.

**Interpretation (provisional):** danuglipron is not paying a large
conformational strain penalty to bind, so *relieving strain is not an available
dose-reduction lever for this molecule*. This is a negative result for the
strain route specifically, not for H1 as a whole (the size/lipophilicity route
is untouched by it).

Sanity signs that this is a real landscape rather than an artifact: the spread
is broad (0–10.9 kcal/mol) with genuine scatter, and the cryo-EM pose is **not**
rank 1 — an implementation that collapsed every conformer onto one minimum, or
that trivially favoured the input pose, would not produce either feature.

## M3. Pocket electrostatic fit (Arms A/B) — **v1 REFUTED BY ITS OWN GATE**

### v1 (`out/fit_and_rank_BIASED_MIN_v1.json`) — do not cite these numbers

The metric gate FAILED. Both pharmacophore-breaking negative controls scored
*better* than the parent:

| candidate | v1 fit (kcal/mol) | poses scored |
|---|---|---|
| parent (cryo-EM pose) | −22.86 | **1** |
| NC1-methyl-ester (acid anchor deleted) | **−47.11** | 6 |
| NC2-decyano (Trp33 terminus deleted) | **−38.11** | 6 |
| all nine real candidates | −25.9 … −49.4 | 6 each |

**Diagnosed cause — a selection bias, not chemistry.** The parent was scored at
its single committed cryo-EM pose (RMSD 0.00 Å) while every other candidate was
scored at the **minimum over 6** re-embedded, rigidly re-aligned poses (RMSD
1.9–3.5 Å). Per-pose fits within a single analogue span up to **64 kcal/mol**
(H3b-acylsulfonamide: −45.6 to +18.6). A minimum over 6 noisy samples versus 1
sample is therefore biased by tens of kcal/mol, in the favourable direction, for
everything except the parent.

This is precisely the failure the negative controls were included to detect, and
they detected it. Nine apparently-improved analogues were an artifact of the
comparison protocol.

**Fixes applied for v2:** (a) every candidate, parent included, goes through the
identical embed → align → score path; (b) the ranking axis is the **mean** over
a fixed pose count, not the min — a min is a biased estimator whose bias grows
with sample count and is not comparable across differing pose counts; (c) the
cryo-EM pose is still scored but reported separately as an experimental
reference, not as the parent's entry in the comparison; (d) any candidate whose
pose-to-pose spread exceeds 20 kcal/mol is flagged as imprecisely posed.

### v2 (`out/fit_and_rank_NEUTRAL_v2.json`) — bias removed, but wrong species

Identical treatment plus a mean estimator removed the protocol artifact. But the
gate still failed, and diagnosing *why* found two further errors — one in the
science, one in the gate itself.

**Error 1 (science): wrong ionization state.** Every candidate was modelled as a
**neutral** molecule. Danuglipron's carboxylic acid has pKa ~4, so it is
**anionic at pH 7.4**, and that anion is the salt bridge which *is* the potency
anchor. Scored in the 7LCJ pocket at the cryo-EM geometry:

| species, same geometry | fit (kcal/mol) |
|---|---|
| neutral acid | −22.86 |
| **anion** | **−165.81** |

Modelling everything neutral omitted **~143 kcal/mol of exactly the interaction
under study**, which is why the methyl-ester control looked equivalent to the
parent — the acid never had its charge in the first place. Two independent
sensitivity checks confirmed the metric is not blind: zeroing the 59 pocket
charges within 6 Å of the carboxylate moves fit by **+11.3 kcal/mol**.

**Error 2 (statistics): the gate used the wrong noise measure.** The precision
check compared the pose-to-pose **range** against the signal. That is invalid: a
range *grows* with sample count as extremes accumulate, whereas the precision of
a mean *falls* as 1/√n. A convergence probe (`out/convergence.log`) confirmed
clean 1/√n behaviour — SEM 5.66 → 3.33 → 1.99 → 1.35 kcal/mol at n = 5, 10, 20,
40 — so ranges of 118–253 kcal/mol coexisted with SEMs of 5–10 kcal/mol. The
range-based test declared a metric unusable when its means were good to ~7
kcal/mol.

Both errors are fixed in the tooling: `Analogue.smiles_ionized`/`net_charge`
carry the pH-7.4 species, `rank.noise_exceeds_signal` uses the **SEM** with a
2σ resolution limit, and `rank.significant_difference` provides the pairwise
test (the aggregate check licenses the set's spread, not every pair in it).

### v3 (`out/fit_and_rank.json`) — anions, n=40 poses, SEM statistics

Precision now **PASSES**: candidate range 142.7 kcal/mol against an 18.4
kcal/mol resolution limit (2σ on a 6.5 kcal/mol standard error).

| candidate | q | fit (kcal/mol) | SEM | vs parent | distinguishable? |
|---|---|---|---|---|---|
| H1c-azetidine | −1 | −165.63 | 4.68 | −45.81 | **YES** |
| NC2-decyano *(control)* | −1 | −137.66 | 7.97 | −17.84 | no — within noise |
| H2a-difluoro-benzylic | −1 | −132.41 | 5.92 | −12.59 | no |
| H3c-oxadiazolone | −1 | −132.13 | 7.75 | −12.31 | no |
| H2b-gem-dimethyl-oxetane | −1 | −127.46 | 8.02 | −7.64 | no |
| H1a-defluoro | −1 | −123.06 | 7.47 | −3.24 | no |
| H1b-des-oxetane-methyl | −1 | −121.20 | 5.86 | −1.38 | no |
| **parent** | −1 | **−119.82** | 7.61 | — | — |
| H3a-tetrazole | −1 | −117.78 | 6.52 | +2.04 | no |
| H3b-acylsulfonamide | −1 | −107.33 | 8.13 | +12.49 | no |
| **NC1-methyl-ester** *(control)* | **0** | **−22.90** | 1.65 | **+96.92** | **YES** |
| *cryo-EM reference pose, anion* | *−1* | *−165.81* | *n/a* | *−45.99* | *—* |

**What the controls now say.** NC1 — which removes the ionizable anchor — is
penalized by **+96.9 kcal/mol**, decisively and in the correct direction. So the
metric *does* resolve the pharmacophore feature it was designed around. NC2 —
which deletes the nitrile and fluorine but **keeps the carboxylate** — is
**−17.8 kcal/mol, within noise**, and is correctly reported as indistinguishable
rather than as an improvement.

**The metric gate still FAILS, and that verdict is correct.** A rigid
electrostatic overlay resolves a formal charge; it does not resolve deleting a
nitrile from an aryl ring 10+ Å away. The gate is doing its job by refusing to
license a ranking on a metric that cannot separate a known-inactive control.

### What the fit arm supports, and what it does not

**Supported:** exactly one candidate is distinguishable from the parent —
**H1c-azetidine-for-piperidine**, at −165.63 ± 4.68 vs −119.82 ± 7.61 kcal/mol,
a 45.8 kcal/mol improvement in pocket electrostatic complementarity. It is also
28 Da lighter with one fewer rotatable bond, i.e. it moves in the H1 direction
while *improving* the electrostatic term. Notably, its fit lands within noise of
the experimental cryo-EM reference pose (−165.81).

**Not supported:** any ordering among the other eight candidates — all sit within
2σ of the parent. In particular the H3 acid-bioisostere arm shows **no**
electrostatic advantage (tetrazole +2.0, acylsulfonamide +12.5, oxadiazolone
−12.3, none significant), so H3 is neither confirmed nor refuted by this arm.
And no candidate is *recommended*, because the gate that would license
recommendations did not pass.

---

## Standing limitations (apply to every fit number here)

- **Rigid scaffold overlay, not docking.** Analogues are placed by superimposing
  their MCS scaffold on the bound pose. An analogue that would genuinely rebind
  in a different orientation is scored pessimistically. No docking engine is
  used and none is available in this repo.
- **One term of a binding free energy.** The fit number is the interaction of
  the ligand density with fixed classical pocket charges. It omits desolvation
  of both partners, pocket reorganization and flexibility, and entropy. It is
  not an affinity and not a potency prediction.
- **Pocket is rigid and classical.** PDB2PQR/AMBER point charges from 7LCJ, held
  fixed for every candidate.
- **Toxicity is external and coarse.** Structural-alert densities and
  physicochemical rules are literature liability *flags*, not predicted
  probabilities of harm. Zero alerts does not mean safe.
- **GFN2-xTB, not DFT.** Adequate for ranking conformers and electrostatic
  interactions of this size; not a benchmark energy.

## M4. Power analysis — what sampling can and cannot fix (2026-08-29)

Computed from the n=40 anion data, before spending compute on a bigger run.
For two independent means, resolving a gap `g` at 2σ needs

    n >= 4 * (sd_a^2 + sd_b^2) / g^2

| pair | gap (kcal/mol) | per-pose sd | poses needed |
|---|---|---|---|
| parent vs **NC2-decyano** (the control the gate fails on) | 17.84 | 46.3 / 50.4 | **n ≥ 59** |
| parent vs **H1b** (finest real candidate gap) | 1.38 | 46.3 / 37.1 | **n ≥ 7350** |

**Measured:** the failing control is ~60 poses away from being resolvable; the
finest candidate-to-candidate distinction is ~7000 poses away.

**Interpretation (provisional):** these are qualitatively different problems and
were previously conflated. More sampling **can** rescue the metric *gate* — a
control is supposed to be grossly separated, and 59 poses is an afternoon. More
sampling **cannot** rescue a fine-grained candidate *ranking*: at ~46 kcal/mol
per-pose scatter, resolving 1.4 kcal/mol by averaging is not a compute problem,
it is the wrong instrument.

That is the quantitative statement of the earlier qualitative verdict ("the gap
is pose determination, not scoring"). Reducing the per-pose sd — by letting
poses settle, or by a real pose search — is the only route to a ranking. Driving
n up is the route to a defensible gate, and nothing more.

---

## Summary of what this campaign established

| # | Finding | Status |
|---|---|---|
| 1 | Danuglipron trips **zero** structural alerts across six published catalogs; its only flag is MW 556 > 500 with cLogP 4.89. The liability is **exposure/dose**, not a toxicophore. | Measured |
| 2 | The bound cryo-EM conformer is only **2.28 kcal/mol** above the free minimum (rank 3 of 20, spread 10.9). **Strain relief is not an available dose-reduction lever.** | Measured, negative |
| 3 | The pocket-fit metric resolves the **ionized carboxylate anchor** decisively (+96.9 kcal/mol on the ester control) but **cannot** resolve deleting the distal nitrile (−17.8, within noise). | Measured |
| 4 | Exactly one analogue is statistically distinguishable from the parent: **H1c-azetidine-for-piperidine**, −45.8 kcal/mol better fit, 28 Da lighter, one fewer rotatable bond. | Measured, single hypothesis |
| 5 | The **H3 acid-bioisostere arm shows no electrostatic advantage** — all three within 2σ of the parent. Neither confirmed nor refuted. | Inconclusive |
| 6 | The metric gate **FAILS** on the NC2 control, so **no candidate ranking is licensed** and none is reported. | Gate held |

### Three self-inflicted errors this campaign caught, and how

Recorded because the catching mechanism is the transferable part:

1. **Selection bias** — parent scored at 1 pose, analogues at the best of 6.
   Caught by the **negative controls** (both scored better than the parent).
2. **Wrong ionization state** — everything modelled neutral when the potency
   anchor is an anion. Caught by **following up the control failure** with a
   direct sensitivity probe instead of accepting the gate's guess.
3. **Wrong noise statistic** — precision judged by a pose *range*, which grows
   with n, instead of the SEM, which shrinks as 1/√n. Caught by a
   **convergence probe** that measured whether the estimator converged at all.

Error 2 is the one worth dwelling on: the gate's original failure message
*asserted* the cause was "tracking molecular size". That was measured and found
**false** (r(MW, fit) = +0.132). The gate now reports the observation and hands
over a prioritized checklist rather than naming an unverified cause — a wrong
diagnosis in a failure message is worse than none, because it misdirects the
next reader.

### Recommended next step, if this were continued

The single blocking gap is **pose determination**, not scoring. `H1c`'s result
is interesting but rests on a rigid scaffold overlay; and NC2 is unresolvable
for the same reason. That needs a real pose search (docking) — not a better
Hamiltonian, since a formal charge is already resolved cleanly at GFN2. Until
then, `H1c-azetidine-for-piperidine` is a **hypothesis worth testing**, not a
recommendation.
