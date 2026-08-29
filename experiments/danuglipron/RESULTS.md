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

## M5. n=100 run — the power analysis held, and the metric is refuted (2026-08-29)

Raised to 100 poses/candidate per M4. `out/fit_and_rank.json`; n=40 data kept at
`out/fit_and_rank_n40.json`.

**The power analysis was correct.** Per-pose sd stayed flat (parent 46.3 → 46.3,
NC2 50.4 → 46.7) while the SEM fell as 1/√n (7.6 → 4.9 and 8.0 → 4.7). The
parent-vs-NC2 pair is now **resolved at 2σ**: gap 18.51 against a 13.46 bar,
where at n=40 it was 17.84 against 22.04. Precision check **PASSES** (4.3
kcal/mol typical SEM, 12.2 kcal/mol resolution limit).

**RETRACTION.** M3 said the metric "cannot resolve deleting the distal nitrile".
That was a statement about precision at n=40 and it is now **wrong**. At n=100
NC2 is resolved — and it resolves on the **wrong side**:

| candidate | fit (kcal/mol) | gap vs parent | 2σ bar | verdict |
|---|---|---|---|---|
| H1c-azetidine | −159.81 | −33.74 | 11.81 | **BETTER** |
| **NC2-decyano** *(control)* | −144.57 | **−18.51** | 13.46 | **BETTER** ← refutation |
| H2a-difluoro-benzylic | −135.30 | −9.23 | 12.39 | within noise |
| H1a-defluoro | −131.48 | −5.41 | 13.68 | within noise |
| H3c-oxadiazolone | −130.63 | −4.56 | 14.02 | within noise |
| H1b-des-oxetane-methyl | −126.31 | −0.25 | 13.19 | within noise |
| **parent** | **−126.07** | — | — | reference |
| H2b-gem-dimethyl-oxetane | −124.56 | +1.51 | 13.74 | within noise |
| H3b-acylsulfonamide | −121.12 | +4.95 | 14.70 | within noise |
| H3a-tetrazole | −118.36 | +7.71 | 13.71 | within noise |
| **NC1-methyl-ester** *(control)* | −22.33 | **+103.74** | 9.91 | **WORSE** ✓ |

A known-inactive control scoring **significantly better** than the parent is a
refutation of the metric, not a precision problem. More sampling made the
verdict *stronger*, not weaker.

### What the metric actually measures

| comparison | magnitude |
|---|---|
| anion (q=−1) vs neutral (q=0) | **−109.5 kcal/mol** |
| full spread among the 10 anions | 41.4 kcal/mol |
| r(MW, fit) among anions only | **+0.490** |

The metric is dominated by **formal charge**, which is why NC1 (the only neutral)
separates so cleanly — that is charge detection, not pharmacophore recognition.
With charge held constant, a size correlation appears (r = +0.490 across the ten
anions), which was *absent* in the mixed-charge n=40 set (r = +0.132). The
earlier "size is ruled out" statement was made on a set where a ±109 kcal/mol
charge term swamped it; controlling for charge reverses it.

**Verdict (dated, provisional): the rigid-overlay electrostatic fit metric is
REFUTED for ranking these analogues** — not merely imprecise. It resolves formal
charge and molecular size, and it ranks a pharmacophore-deleted control above the
parent. No candidate ranking is reported, and `H1c`'s apparent −33.7 kcal/mol
advantage must be read in that light: it is the largest anion-subset effect in a
metric that correlates with size at r = +0.49.

**This closes the "more sampling" route.** M4 predicted sampling could rescue the
gate; it could not, because the gate's failure was never a precision failure.
M4's other prediction stands and is now the operative one: the per-pose scatter
must be *reduced* (real pose determination), not averaged down.

## M6. In-field pose relaxation — helps, but nowhere near enough (2026-08-29)

M5 closed the "more sampling" route, leaving one option: *reduce* the per-pose
scatter by letting each overlaid pose settle in the pocket field before scoring.
`run_pose_relax_probe.py`. **PARTIAL RUN** — the parent completed (9 of 12 poses;
3 dropped on rescoring), NC2 was interrupted part-way. Numbers below are the
parent's.

| | mean (kcal/mol) | sd | range | geometric spread |
|---|---|---|---|---|
| rigid overlay | −107.31 | 34.23 | 99.0 | 3.98 Å |
| relaxed in field | −159.98 | 29.07 | 80.2 | 3.84 Å |

**The artifact check passed.** The hypothesis stated before measuring was that a
sd reduction would be meaningless if it came from every pose collapsing onto one
geometry. It did not: the mean pairwise all-atom RMSD moved only 3.98 → 3.84 Å
(4%), so the poses stay geometrically distinct. The **15% sd reduction is a real
energetic tightening**, not a collapse.

**And it is nowhere near enough.** At sd = 29.1, the SEM at n=100 would be 2.9
kcal/mol, while the parent-vs-H1b gap that a ranking must resolve is 0.25
kcal/mol (n=100 data). That needs

    n >= 4 * 2 * 29.07^2 / 0.25^2  ≈  108,000 poses per candidate

each of which is now a full in-field geometry optimization rather than a single
point. Relaxation buys roughly a 15% sd reduction against a requirement that is
three orders of magnitude away.

**Verdict (dated, provisional): in-field relaxation is a real but marginal
effect and does NOT rescue the metric.** Combined with M5, both routes that do
not require new capability are now closed:

| route | status |
|---|---|
| more poses (M4/M5) | **closed** — the failure was never precision; the metric is refuted |
| relax poses in field (M6) | **closed** — real 15% effect, ~3 orders of magnitude short |
| real pose search (docking) | **untested** — no docking engine available in this repo |

### Known limitation in this probe

3 of 12 parent poses were dropped when rescoring reported failure. Re-running
two of them (poses 05, 07) in isolation **succeeded**, so the failures are not
properties of those geometries — they are transient, and libxtb is not
thread-safe, so concurrent load is the likely cause. The probe discards
`FitResult.error` in its progress line, which is why the cause could not be read
off the log; that is a defect in the probe's reporting, not in the measurement.
Fix before re-running: print `fr2.error`, and do not run the probe alongside
other xtb work.

## M7. The real root cause: CANDIDATE GENERATION, not scoring (2026-08-29)

Prompted by the observation that the same molecule spans ~5x more energy across
its own poses (228 kcal/mol) than the entire designed candidate set spans between
molecules (41 kcal/mol). That ratio is diagnostic: it says the pose ensemble, not
the chemistry, is what the metric is responding to.

Tracing back past the fit stage:

**The alignment code is correct.** Aligning the bound pose onto itself gives
**RMSD = 0.0000 Å** over all 41 heavy atoms. So the 2–4 Å scaffold fits reported
throughout M3–M6 are not an alignment defect — they are real conformational
mismatch.

**Every generated conformer misses the bound pose:**

| conformer | scaffold RMSD vs bound pose |
|---|---|
| `conf_00_cryo_em` (the reference itself) | 0.00 Å |
| `conf_02_rdkit` — **best generated** | **2.23 Å** |
| `conf_14_rdkit` | 2.70 Å |
| `conf_01_pubchem` | 3.05 Å |
| remaining 16 RDKit conformers | 3.14 – 3.70 Å |

The conventional bar for a successful docking pose is **RMSD < 2.0 Å**. **Not one
of the 20 committed conformers clears it**, and freshly embedded ensembles behave
the same (2.3–4.1 Å across 12 poses of the parent).

### Why this was inevitable, not bad luck

Danuglipron has **9 rotatable bonds** over 41 heavy atoms. ETKDG samples the
*free-solution* torsional space; the bound conformer is one specific point in
that space, selected by the receptor. The chance that unbiased conformer
generation lands within 2 Å of it is small, and it does not improve with more
conformers in any practical number — the 20-member ensemble's best is 2.23 Å and
100 freshly embedded poses did no better.

So the campaign was **scoring the wrong geometries from the start**. Every
downstream measurement inherits it:

- the ~46 kcal/mol per-pose scatter (M4) is the spread over *non-bound* poses;
- the charge/size domination (M5) is what a scoring function reports when the
  specific contacts are absent — with no salt bridge in place, only the
  monopole and molecular volume remain;
- the 15% relaxation effect (M6) is small precisely because relaxation cannot
  fix a torsional mismatch — it settles bond lengths and angles, not a 3 Å
  scaffold displacement.

**This supersedes the M5/M6 framing.** Those measurements are still valid as
measurements, but the verdict "the metric is refuted" is too narrow: the metric
was never given a fair test. The correct statement is that **candidate pose
generation failed**, and no scoring function — ferric's or anyone's — can rank
poses that are 3 Å from the binding mode.

### What this changes about the next step

Previously recorded as "needs docking". That is still true but now for a sharper
reason: the missing capability is not pose *refinement*, it is pose *search* —
something that biases conformer generation toward the receptor rather than
sampling free-solution torsions and hoping. A constrained embed against the bound
scaffold (RDKit `ConstrainedEmbed` / core-constrained ETKDG) is the cheap version
of that and is available here; full docking is the proper version.

**That is the single highest-value next experiment**, and it is testable with the
existing pipeline: constrain each analogue's shared scaffold to the bound pose's
coordinates, generate only the modified region, and re-run. If pose generation is
really the root cause, the per-pose scatter should collapse and the controls
should separate correctly.

---

## Summary of what this campaign established

| # | Finding | Status |
|---|---|---|
| 1 | Danuglipron trips **zero** structural alerts across six published catalogs; its only flag is MW 556 > 500 with cLogP 4.89. The liability is **exposure/dose**, not a toxicophore. | Measured |
| 2 | The bound cryo-EM conformer is only **2.28 kcal/mol** above the free minimum (rank 3 of 20, spread 10.9). **Strain relief is not an available dose-reduction lever.** This result is independent of #3 — it uses the committed bound geometry directly and never relies on generated poses. | Measured, negative |
| 3 | **ROOT CAUSE — candidate pose generation failed.** No generated conformer reaches the bound pose: best is **2.23 Å**, against the conventional **< 2.0 Å** docking-success bar; the other 19 are 2.7–3.7 Å. The alignment code is correct (self-alignment = 0.0000 Å), so this is real conformational mismatch. With 9 rotatable bonds, unbiased ETKDG was never likely to hit the receptor-selected torsion set. | Measured (M7) |
| 4 | Everything downstream inherits #3. The pocket-fit metric being **charge-dominated** (−109.5 vs a 41.4 kcal/mol anion spread) and size-correlated (r = +0.490) is what a scoring function reports when the specific contacts are simply **absent from the geometry**. | Measured (M5), reinterpreted |
| 5 | At n=100 the **NC2 control scores significantly BETTER than the parent** (−18.5, 2σ bar 13.5) — consistent with #3: poses 3 Å off the binding mode cannot express a pharmacophore difference. | Measured (M5) |
| 6 | The metric gate **FAILS**, so **no candidate ranking is licensed** and none is reported. `H1c`'s apparent advantage is confounded with both size and pose error. | Gate held |

### Four self-inflicted errors this campaign caught, and how

Recorded because the catching mechanism is the transferable part:

1. **Selection bias** — parent scored at 1 pose, analogues at the best of 6.
   Caught by the **negative controls** (both scored better than the parent).
2. **Wrong ionization state** — everything modelled neutral when the potency
   anchor is an anion. Caught by **following up the control failure** with a
   direct sensitivity probe instead of accepting the gate's guess.
3. **Wrong noise statistic** — precision judged by a pose *range*, which grows
   with n, instead of the SEM, which shrinks as 1/√n. Caught by a
   **convergence probe** that measured whether the estimator converged at all.
4. **Never validated the input geometries** (M7) — the most expensive error, and
   the one that invalidated the most work. Four measurement rounds were spent
   characterising a metric that was being fed poses 2–4 Å from the binding mode.
   The check that would have caught it on day one costs one line: align the
   generated conformers onto the known bound pose and look at the RMSD. It was
   never run because the bound pose was treated as a *scoring reference* rather
   than as *ground truth for pose generation* — the campaign had it in hand the
   whole time and never used it that way.

Error 2 is the one worth dwelling on: the gate's original failure message
*asserted* the cause was "tracking molecular size". That was measured and found
**false** (r(MW, fit) = +0.132). The gate now reports the observation and hands
over a prioritized checklist rather than naming an unverified cause — a wrong
diagnosis in a failure message is worse than none, because it misdirects the
next reader.

### Recommended next step, if this were continued

**Both routes reachable without new capability are now closed** (M5, M6). More
sampling cannot help — the metric is refuted, not imprecise. In-field relaxation
is a real effect but ~3 orders of magnitude too small.

What remains is a **real pose search (docking)**, which this repo has no engine
for — checked 2026-08-29: no vina, smina, gnina, obabel, rdock, meeko. Adding one
is a dependency decision, not a coding task, and it is the only thing that would
make a fit ranking meaningful here.

A second, independent objection stands regardless of pose quality: the metric is
**charge-dominated** (−109.5 kcal/mol anion-vs-neutral vs a 41.4 kcal/mol spread
among anions) and, with charge controlled, size-correlated (r = +0.490). Better
poses would sharpen a quantity that is still measuring the wrong things. A
scoring function with desolvation and a flexible pocket is the other half of the
gap.

`H1c-azetidine-for-piperidine` remains a **hypothesis**, and a weaker one after
M5 than before it: its −33.7 kcal/mol advantage is the largest anion-subset
effect in a metric that correlates with molecular size. It is not a
recommendation and this campaign does not make one.
