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
| real pose search (docking) | ~~**untested** — no docking engine available in this repo~~ **STALE. Closed by M12: 1%, 32.5x short.** Docking landed in M9 (August); this line was written before it and misled a later probe into re-testing on a false premise. |
| select one pose (M13) | **closed** — 7-10x WORSE than averaging |
| a different scorer (M14) | **closed** — none available is less pose-sensitive |

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

## M8. Simulated annealing as a pose search — closed (2026-08-29)

M7 showed the error is TORSIONAL, and relaxation cannot fix a torsional error
(0.04-0.16 A). MD at elevated temperature can cross torsional barriers, so it
was the one mechanism available here that could. Tested on the parent, where the
bound pose is known exactly. 4 ps at 500 K, 20 frames, 3 independent starts, run
BOTH in the pocket field and in vacuum as the control.

| start | start RMSD | in-field best | vacuum best |
|---|---|---|---|
| 0 | 3.29 | 3.17 | **2.41** |
| 1 | 2.70 | 2.70 | **2.43** |
| 2 | 4.08 | **2.98** | 3.20 |
| **mean of best** | 3.36 | 2.95 | **2.68** |

**Best frame anywhere: 2.41 A. The 2.0 A bar is not reached from any start, in
either condition.**

Against the three hypotheses stated before measuring:

- **"annealing works"** — NO. Mean improvement +0.41 A. Better than relaxation's
  0.04-0.16 A, and still ~0.4 A short.
- **"thermal noise"** — largely YES. The MEAN over frames got *worse* than the
  starting pose in the in-field runs (3.42 vs a 3.36 mean start), i.e. a typical
  frame is worse than where it began; only the best frame improves, which is
  what sampling noise looks like.
- **"the field does nothing"** — worse than nothing: **vacuum BEAT in-field on 2
  of 3 starts** (2.68 vs 2.95 A mean best). But the 0.27 A difference is only
  **25% of the 1.07 A within-run spread**, so at n=3 this does NOT support "the
  field hurts". The supportable claim is that the pocket field provides **no
  detectable guidance at this timescale**.

### Why 4 ps was never going to be enough — and why that is my error

At 500 K, kT = 0.99 kcal/mol. A 5 kcal/mol benzylic torsional barrier has a
Boltzmann factor of 6.5e-3; reorganizing **9 rotatable bonds** requires crossing
several such barriers, which is a **nanosecond** process. 4 ps is ~3 orders of
magnitude short.

So this run was underpowered by construction, in the same way the campaign's
original pose generation was — and I nearly filed it as a clean method failure.
The measured cost of doing it properly (141 s/ps on this system):

| timescale | per anneal | 11 candidates x 3 starts |
|---|---|---|
| 4 ps (run) | 9 min | — |
| 20 ps | 47 min | 26 h serial / ~2 h on 12 cores |
| 100 ps | 3.9 h | 129 h serial / ~11 h on 12 cores |

**Verdict (dated, provisional): MD-as-pose-search is CLOSED as a practical
route** — not because it was disproven at an adequate timescale, but because
reaching an adequate timescale means using a 141 s/ps method for a job a
purpose-built search does in seconds. Metadynamics (`xtb --metadyn`) would reach
in tens of ps what plain MD needs ns for, but that is still working around a
missing dependency rather than fixing the gap.

### What we built vs what docking is

| | this pipeline | docking |
|---|---|---|
| search | **none** — Kabsch is a least-squares FIT of a fixed internal geometry | MC + BFGS over 6 rigid-body DOF + 9 torsions |
| poses evaluated | ~10^2 | 10^5 - 10^7 |
| per-pose cost | ~0.5 s (GFN2 SP) | ~10 us (empirical sum) |
| pocket | rigid classical charges | rigid, but pose-aware scoring |
| score | electrostatics/polarization only | empirical, calibrated to binding data |

Nothing in this pipeline ever proposes a pose and asks whether the pocket likes
it. Conformers are generated in FREE SOLUTION, ignoring the receptor, then
rigidly superimposed. That is a **rescoring** pipeline — a legitimate and
standard technique, but one that presupposes poses from elsewhere. We built the
second half of "dock, then rescore" and never had the first.

Where this pipeline is genuinely better: GIVEN a correct pose, GFN2 in a real
point-charge field captures polarization no empirical function does — which is
why the anion/neutral distinction showed up as a clean 143 kcal/mol effect (M5).

### The unblocking step

**AutoDock Vina 1.2.7 + Meeko 0.8.0**, both pure-pip with no compilation
(checked 2026-08-29); Apache-2.0 on the engine, LGPL-2.1 on the prep layer.
gnina is not on PyPI (needs a CUDA source build).

First test is self-validating: **redock danuglipron into 7LCJ and measure RMSD
against the known bound pose.** Vina's published redocking success is ~60-80%
under 2 A; if it cannot reproduce a pose we already hold, that is known in
minutes rather than after committing to the analogue set.

## M9. Docking closes the pose gap — 0.95 A (2026-08-29)

Added tier 1 (AutoDock Vina 1.2.7 + Meeko 0.8.0, pure pip, Apache-2.0 engine)
and ran the self-validating test: **redock danuglipron into 7LCJ and measure
RMSD against the crystal pose we already hold.** The ligand was re-embedded from
SMILES with a fresh seed, so the search starts from a geometry carrying no
information about the answer.

| metric | value |
|---|---|
| **best-of-20 RMSD** | **0.95 A** |
| top-ranked pose RMSD | **0.95 A** (rank 0) |
| poses within 5 A of the known site | 20/20 |
| verdict (pass < 2.0 A) | **PASS** |

For scale: every method tried before this topped out at **2.41 A** after 62
minutes of GFN2 MD. Docking reached 0.95 A in ~2 minutes.

**Pose generation is solved for this target.** M7's root cause is closed.

> **Refined by M11 (2026-09-02).** This was ONE seed at `exhaustiveness=32`,
> `n_poses=20`. Repeating across 3 seeds and 4 effort levels puts 0.95 A
> comfortably inside the observed 0.75-1.24 A band, so the headline conclusion
> holds -- but the 32 is not load-bearing and was never the reason it worked.
> Measured, `exhaustiveness=4` reaches the same accuracy for a quarter of the
> cost, and 32 has the WORST mean of the four levels tried. The variable that
> actually moves this number is the ETKDG SEED, not the search effort.
> Do not cite the 32 as a validated requirement.

### The cheap score finds the pose but cannot rank it

Measured on the same run: r(vina_score, RMSD) = **+0.461**, and only **4 of 20**
poses are under 2.0 A. Vina put the right pose first here, but the correlation
is weak enough that it did so partly by luck.

That is the empirical justification for the whole hierarchy: **the cheap tier
generates the right answer among its candidates and cannot reliably pick it
out.** If it could, no rescoring would be needed and tiers 2-4 would be
decoration.

### Two bugs this exposed, both silent

1. **AutoDock types are not element symbols.** PDBQT's last column is a docking
   type -- `OA` is an H-bond-accepting oxygen, `NA` an accepting nitrogen -- and
   `.capitalize()` turns `OA` into the non-existent element "Oa" (and `NA` into
   sodium). The first run reported **NO ALIGNABLE POSE** on 20 perfectly good
   poses. Fixed with an explicit type->element table.
2. **PDBQT is united-atom.** Nonpolar hydrogens are merged into their carbons,
   so a docked danuglipron has 41 atoms where the RDKit mol has 70. The
   alignment's identity check demanded a full-formula match and rejected every
   pose as "not this molecule". Now compares HEAVY-atom formula, which still
   catches a wrong molecule -- the thing the guard is actually for.

Both produced a confident, wrong, *negative* verdict on a working pipeline. Both
were found by reading the reported error instead of the headline.

---

## Systematizing the methods: the cost hierarchy

`tools/campaign/hierarchy.py` (types and rules) +
`experiments/danuglipron/hierarchy.py` (this system's measured costs).

| tier | method | s/pose (measured) | poses | job | validated |
|---|---|---|---|---|---|
| 1 | AutoDock Vina | 1e-5 | 10^5-10^6 | **search** pose space | yes -- 0.95 A redock |
| 2 | MMFF94 | 1e-3 | 10^2-10^3 | relax, declash | yes -- adequate to declash, NOT to rank |
| 3 | GFN2-xTB | 5e-1 | 10-10^2 | rank survivors | yes -- 143 kcal/mol anion/neutral split |
| 4 | ferric DFT + dispersion | 6e+2 | 1-10 | final energetics | **NO -- never used** |

Each tier exists to DISCARD, cheaply, what the next cannot afford to examine. A
tier is chosen for cost and discrimination, not for being "best": the best
method on the wrong candidates is waste, and a cheap method asked for a fine
distinction is noise.

**The campaign's central failure was a hierarchy failure, not a physics one.**
It ran tier 3 alone, fed by tier-2 output that had never seen the receptor.

### Rules (encoded in `hierarchy.py`, tested in `tests/test_hierarchy.py`)

1. **Validate each tier against ground truth before trusting it.** For poses
   that means redocking a known complex. It costs minutes and is the difference
   between a pipeline and a guess.
2. **A tier's output is only as good as its input.** No tier-4 rigour rescues a
   tier-1 failure. **Diagnose downward** -- check the cheapest tier first,
   because that is where the candidate population is set.
3. **Never ask a tier for a distinction finer than its noise.** Quantify with
   the standard error, never a sample range (which grows with n).
4. **Retire a tier when a cheaper one matches it, or when its job belongs to
   someone else.** MD-as-pose-search was retired on the second ground.
5. **Keep each tier's failure visible.** A tier that cannot answer returns
   None/UNEVALUATED, never a neutral-looking number.

**Tier 4 is the standing gap: ferric's own DFT has never been used in this
campaign.** Every energy reported anywhere above is GFN2. That is now the next
step, and for the first time it has geometries worth spending it on.

## M10. The isomer pipeline runs end to end; ~~tier 4 does not fit~~ (2026-08-30)

> **TITLE RETRACTED 2026-09-02, and re-confirmed 2026-09-19.** Tier 4 DOES
> fit (612 s, converged) and now runs end to end in
> `tools/pipeline/tests/test_golden_path_smoke.py`. The title is kept in
> strikethrough because it was cited as a live blocker for weeks after the
> retraction directly below it.

> **STATUS as of 2026-09-02: the title of this section is WRONG and kept for
> the record.** Tier 4 DOES fit -- **612.4 s (10.2 min), 18 iterations,
> converged** for the 71-atom neutral acid at STO-3G/PBE. The ">57 min, did not
> finish" recorded below was **memory contention**, not DFT cost: that run
> auto-resolved a 7.26 GB budget from live MemAvailable while needing ~9.5 GB
> and paged until the kernel OOM-killed it.
>
> This section is left in chronological order, with the wrong turns intact,
> because the sequence of corrections is the useful part. Read
> **"RESOLVED: tier 4 costs 10.2 min"** below for the current verdict, and
> treat every ">57 min" above it as superseded.

Built `tools/isomers` (enumeration) + `tools/pipeline` (the funnel) and ran the
full four-tier stack on danuglipron. Plan:
`experiments/danuglipron/plans/2026-08-30-isomer-pipeline.md`.

### Enumeration replaces hand-writing

61 candidates generated from ONE SMILES -> 60 after dedup (54 substitutional,
5 structural), versus the 11 analogues previously written by hand. It
independently rediscovers three of those hand designs -- tetrazole,
acylsulfonamide, piperidine->azetidine -- which is the closest thing to a
correctness check this stage has.

### The funnel works

| tier | stage | in | out | failed |
|---|---|---|---|---|
| 1 | Vina dock | 60 | 24 | 2 |
| 2 | MMFF94 | 24 | 12 | 0 |
| 3 | GFN2-xTB | 12 | 5 | 0 |
| 4 | ferric DFT | 5 | **0** | **5** |

55 minutes for tiers 1-3. The per-tier bookkeeping is what made the tier-4
failure visible at all: the driver still printed "tier 4 reordered tier 3 ->
DFT is load-bearing", which was **meaningless**, because nothing had been
computed. A funnel that reported only survivors would have shown an empty list
and no reason.

### Two bugs the run exposed, both mine

**1. Ionization state (physics).** Every tier-4 candidate failed with
`inconsistent charge/multiplicity: 325 electrons with multiplicity 1 implies
n_alpha = 325/2`. The driver set `net_charge=-1` on **neutral** structures.
Removing H+ takes a bare proton and leaves its electrons behind, so an anion has
the **same** electron count as its acid; declaring -1 on the neutral SMILES asks
for an electron that does not exist, and makes an even count odd. ferric was
right to refuse. Fixed by deprotonating the STRUCTURE
(`Isomer.deprotonated()`), with tests pinning that electron count is CONSERVED.

**2. Disconnected fragments.** A ring-contraction transform can sever a ring
rather than shrink it. Two candidates reached tier 1 as fragment pairs and died
in Meeko. Now rejected at enumeration with a readable reason.

### Tier 4 is not affordable at this size — measured, not estimated

After the ionization fix, a single tier-4 point on the 70-atom anion:

| | |
|---|---|
| basis | STO-3G (~234 basis functions) |
| runtime | **>57 min, did not finish** |
| peak RSS | **9.5 GB** |

For comparison, the same code does def2-SVP on a 32-atom alkane in **96 s**
(re-measured 2026-09-02: **99.0 s**, so that figure reproduces).

**CORRECTION (2026-09-02).** The comparison as originally written -- "234 basis
functions taking >57 min while 450 takes 96 s is not a size effect" -- was wrong
in two ways, and the second one matters.

1. The alkane's def2-SVP basis is ~330 functions, not 450. Minor.
2. **Basis-function count is the wrong cost axis for the part that dominates.**
   ferric's KS-DFT grid is a flat 75x110 Becke-Lebedev grid **per atom**
   (`ferric-dft/src/grid.rs`), so the XC grid scales with ATOM COUNT and is
   completely independent of the basis:

   | | atoms | grid points | nbf | AO cache (4*nbf*npts*8) |
   |---|---|---|---|---|
   | alkane_10 / def2-SVP | 32 | 264,000 | ~330 | 2.79 GB |
   | danuglipron / STO-3G | 70 | **577,500** | ~234 | **4.32 GB** |

   Shrinking the basis to STO-3G cut nbf but RAISED the resident AO cache,
   because 2.19x the grid points outweighs 0.71x the basis functions.

So the honest statement of the anomaly is: the XC grid work grew **1.55x**
(nbf x npts) while runtime grew **>35x**. The disproportion is real -- it just
is not the basis-size paradox the original text claimed. **The cause is still
not diagnosed.**

### What was ruled out, and what was not (2026-09-02)

- **Ruled out -- the 96 s anchor being bogus.** It had NO recorded run behind it
  anywhere in the repo; every mention was a citation of the same number. Re-run
  from scratch: **99.0 s**, converged. The anchor is sound; its provenance was
  not, and now is.
- **Not established -- the anion/diffuse-HOMO hypothesis.** The neutral-vs-anion
  control at identical size and basis is the experiment that separates "charge"
  from "size", and it did NOT complete: another user's `llama-server` job started
  mid-run, the box went to `/proc/pressure/memory full avg300=89` with swap
  fully consumed, and any timing taken under that contention is worthless. I
  killed my own job rather than compete for the memory. **Unmeasured, still open.**
- **Batching is RULED OUT (2026-09-02).** Measured on the neutral acid under
  the 12 GB cap: RSS plateaus at **6.28 GB** and stays there. That matches the
  predicted full AO cache (nbf=235, npts=585,750 -> 4.40 GB) plus SCF matrices,
  and sits well under the ~9.6 GB budget (`0.8 x` the cap), so `check_grid_budget`
  returned `Ok(true)` and `GridCache::Full` was used. **The grid was never
  batched**, so the batching cliff cannot explain tier 4's cost. Superseded
  reasoning below, kept for the record:
- ~~**Batching is a candidate, not a finding.**~~ `ks.rs` falls back to walking the
  grid in point-batches (recomputing AO values per batch) when the cache exceeds
  the resolved budget. Under `ferric-limited`'s 12 GB cap the budget is
  ~9.6 GB and the 4.32 GB cache should fit -- consistent with the observed
  9.5 GB peak being a FULL cache plus SCF matrices, i.e. batching probably never
  engaged. Testing this needs an uncontended box.

### RESOLVED: tier 4 costs 10.2 min, not >57. M10's number was contention.

**Measured 2026-09-02 on a box verified quiet throughout** (`pressure avg10=0.00`
and zero swap traffic for the whole run, budget pinned at 11 GB, cap 14 GB):

| | |
|---|---|
| molecule | danuglipron neutral acid, 71 atoms, 292 e- |
| basis / functional | STO-3G / PBE |
| **wall** | **612.4 s = 10.2 min** |
| **iterations** | **18** |
| exit | **Converged** |

**M10 recorded ">57 min, did not finish" for this system. The real cost is
10.2 minutes.** That run auto-resolved a **7.26 GB** budget from live
MemAvailable while needing ~9.5-9.9 GB, and spent its time paging until the
kernel OOM-killed it (`anon-rss 9,945,236 kB`, `global_oom`). **It measured
memory contention, not DFT.** Three further attempts today died the same way
before one finally got a clean box.

**Tier 4 is therefore AFFORDABLE for the handful of candidates the funnel
delivers** -- ~10 min each at STO-3G, so the 5 survivors of a run are under an
hour. The fix was the budget pin plus headroom, not a cheaper method.

### Why 10.2 min and not the predicted 6.8

The cost model (work-scaled from alkane_20) predicted 6.8 min. The gap is
**iteration count, not per-iteration cost**:

| | alkanes (17-62 atoms) | danuglipron |
|---|---|---|
| iterations | **10** (all three) | **18** |

Correcting the work-scaled 408 s by the real ratio 18/10 gives 734 s against a
measured 612.4 s -- **within 20%, erring conservative**. Work-scaling alone is
off by 50%.

**RETRACTED:** I attributed that gap to N/O/F having sharper core densities
than carbon, making the radial grid costlier per basis function. That was a
guess made before the run finished, and it is wrong -- the extra time is
mostly extra ITERATIONS. The per-iteration model was fine.

So a wall-time estimate needs BOTH factors:

    wall ~ xc_fock_work x (s per iteration) x (iterations)

and the iteration count is NOT transferable across chemistries. Encoded in
`tools/pipeline/cost.py`, which now documents `predicted_seconds` as a lower
bound whenever the target may converge more slowly than the reference.

### The cost term is IDENTIFIED, and ">57 min" is probably not a cost result

`vxc.rs` assembles V_xc with `buf.dot(&chi.t())` -- an `(nbf, npts) x (npts, nbf)`
GEMM, i.e. **O(nbf^2 x npts)**, executed **every SCF iteration**. Since npts is
proportional to atom count, that is cubic in molecular size and paid per
iteration, which is exactly the shape the constant-iteration measurement
demanded.

Calibrated on the alkane runs, predicting from **alkane_5 alone**:

| atoms | predicted | actual |
|---|---|---|
| 32 | 17.8 s | 19.6 s |
| 62 | 134.3 s | 130.2 s |

Within 10% across a **54x** span of cost. Encoded as
`tools/pipeline/cost.py::xc_fock_work` / `predicted_seconds()`.

**Applied to danuglipron (nbf=235, npts=585,750) the model predicts 6.8 min** --
not the >57 min recorded in M10.

**What the re-run actually showed.** With the budget PINNED at 9 GB and the cap
raised to 11 GB, the neutral acid ran to **7:26** and was still going. But at
that point its CPU had fallen from 250-600% to **77%**, `/proc/pressure/memory`
read `some avg10=49`, and **swap was 100% consumed (1.9/1.9 GiB) with active
si/so traffic**. The job was THRASHING, so its wall time was I/O-bound, not
compute-bound. I killed it: past that point the clock measures paging, not
chemistry.

**Revised reading of M10's ">57 min", stated as a hypothesis and not a
conclusion:** that run auto-resolved a **7.26 GB** budget (live MemAvailable,
depressed by other jobs), needed ~9.5-9.9 GB, and spent its time paging before
being OOM-killed. The cost model says the *compute* is ~7 min. If that is
right, tier 4's headline number was never a measurement of DFT cost -- it was a
measurement of memory contention, and the fix is the budget pin plus enough
headroom, not a cheaper method.

**This is NOT yet established.** Confirming it needs one clean run of the
neutral acid on a box with >=12 GB genuinely free, reporting `iterations` and
finishing without swap traffic. Every attempt so far has been interrupted by
competing jobs (three separate occasions on 2026-09-02). Until then M10's
">57 min" stands as recorded, with this caveat attached.

> **CONFIRMED later the same day.** The fourth attempt got a clean box:
> **612.4 s, 18 iterations, converged.** The hypothesis above was right. See
> "RESOLVED: tier 4 costs 10.2 min" above.

### Iteration count is CONSTANT; the N^3 lives INSIDE each iteration (2026-09-02)

Built the `iterations` / `exit_reason` getters on `PyDftResult` and re-ran the
alkane series with the memory budget PINNED (`FERRIC_MEM_BUDGET_GB=9`) so the
AO-cache path could not drift with box load:

| atoms | wall | iterations | exit | s/iter |
|---|---|---|---|---|
| 17 | 2.5 s | **10** | Converged | 0.25 |
| 32 | 19.6 s | **10** | Converged | 1.96 |
| 62 | 130.2 s | **10** | Converged | 13.02 |

**The iteration count is identical -- 10 -- across a 3.6x size range.** So none
of the alkane scaling comes from convergence behaviour; ALL of it is
per-iteration cost, scaling at N^3.26 then N^2.86.

**This refutes my own leading hypothesis.** I had attributed the ~N^3 term to
one-time grid construction (`becke_weights_all`, O(natoms^2) per point x O(natoms)
points). But a ONE-TIME cost cannot produce N^3 scaling *per iteration*. The
cubic term is inside the SCF loop, not in setup. The grid-setup reasoning
recorded above stands as an accurate description of that function's complexity
and a WRONG explanation of where the time goes.

### The size hypothesis is REFUTED (2026-09-02, measured on a quiet box)

Homologous alkane series, STO-3G/PBE, all converged, each under
`scripts/ferric-limited`:

| molecule | atoms | nelec | time |
|---|---|---|---|
| alkane_5 | 17 | 42 | 2.5 s |
| alkane_10 | 32 | 82 | 20.1 s |
| alkane_15 | 47 | 122 | 60.6 s |
| **alkane_20** | **62** | **162** | **123.5 s** |

Pairwise exponents **3.30 -> 2.87 -> 2.57** (global log-log fit **N^3.03**).
The exponent DECLINES with size, consistent with the O(natoms^3) grid
construction being amortized as the linear-scaling parts grow. Per the
protocol's "fit the TAIL, not the whole series", the tail exponent 2.57 is the
one to extrapolate with.

**alkane_20 has 62 atoms -- within 12% of danuglipron's 70 -- and finishes in
just over two minutes.** Extrapolating on the composition-aware axis (XC work
= nbf x npts, ratio 1.86x) gives **3.8 min** if cost is linear in that work and
**10.2 min** at the tail exponent.

**Observed for danuglipron at the time: >57 min, did not finish.** At least a
**6x** gap that size does not explain.

> **SUPERSEDED.** That ">57 min" was memory contention, not compute; the real
> figure is **612.4 s (10.2 min)**. So the 6x gap was an artefact and there is
> no size anomaly to explain. Note the tail-exponent extrapolation just above
> predicted **10.2 min** -- it was correct, and only the number it was being
> compared against was wrong. The reasoning in this section about atom count vs
> composition still stands; the verdict it reaches does not.

So the anomaly is real, and it is NOT:

- **basis size** -- STO-3G is the smallest bundled set, and the alkane series
  used the same one;
- **atom count** -- a 62-atom molecule of the same class runs in 123.5 s;
- **composition / heavy-atom content** -- accounted for by scaling on
  nbf x npts (danuglipron is 1.86x alkane_20's XC work, nowhere near 6x).

Composition deserves an explicit note because atom count HIDES it: alkane_20 is
hydrogen-padded (20 C + 42 H = 142 STO-3G functions) while danuglipron is
heavy-atom rich (41 heavy + 29 H = 234). A 1.13x atom ratio conceals a 1.86x
work ratio. `tools/pipeline/cost.py::sto3g_basis_functions` exists so the
comparison is made on the right axis -- I nearly used alkane_20 as a
"size-matched control" that was matched on the wrong variable.

**Charge is NOT the explanation either (partial, 2026-09-02).** The
neutral-vs-anion control was started on a quiet box. The **neutral** acid
(71 atoms, q=0) reached **9.4 GB RSS within 2 minutes** -- the same memory
signature previously seen on the anion, and far past the ~2 min in which the
62-atom alkane finishes ENTIRELY. I stopped it there: another user's
`llama-server` started and free memory fell to 260 MB, so continuing would have
risked the box for a number I could already tell was going to be large.

The run did not finish, so there is no timing to quote -- but the memory
trajectory alone rules charge out as the driver, since the neutral species
behaves the same way. **Do not re-run the anion first**: the neutral is the
cheaper falsification and it already fired.

**A second, separate anomaly: the memory does not add up (2026-09-02).** The
neutral acid was OOM-killed at **9.48 GB** (`anon-rss 9,945,236 kB`, kernel
`global_oom` — the SYSTEM ran out while other workloads competed, not the
cgroup; the cap did its job and my process died in its own scope rather than
taking the box down). Breaking that down:

| term | size |
|---|---|
| full AO cache (chi + grad-chi, nbf=235, npts=585,750) | 4.40 GB |
| RI three-index `(P\|mn)` tensor | **1.61 GB** |
| RI metric `(P\|Q)` | 0.11 GB |
| **subtotal** | **6.12 GB** |
| observed peak | ~9.75 GB |
| unaccounted | 3.63 GB |

**Corrected 2026-09-02:** I first estimated the RI tensor at 0.31-0.42 GB by
guessing naux ~700-950. Counted from the basis JSON it is **3,635 auxiliary
functions** -- `def2-universal-jkfit` is built for large orbital bases and
`run_dft` always enables it, so at STO-3G the aux basis is **15x** the orbital
basis (3,635 vs 235). That single correction moved 1.2 GB from "unexplained"
into "accounted for".

**But it is NOT the actionable win I first wrote it up as.** I claimed "a
JK-fitting basis matched to STO-3G would cut ~1.6 GB", then checked: 
`def2-universal-jkfit` is already the **smallest** JK-fitting set ferric
bundles -- 75 aux functions per carbon, against 79 for cc-pVTZ-JKFIT and 106
for cc-pVQZ-JKFIT. The 15x ratio comes from STO-3G being tiny, not from this
aux basis being large, and acting on it would mean ADDING a small-basis JK set
to the repo, not selecting a different one. Recorded as accounting, not advice.

The remaining 3.63 GB is plausibly transient copies during the RI build (form
the tensor, then contract it) plus Fock/DIIS/grid working set, but that is NOT
measured. DIIS history is ruled out: `diis.rs` uses a FIXED-capacity ring
buffer, so it cannot grow with iteration count, and at 0.44 MB per matrix it is
~9 MB regardless.

Note also that the budget auto-resolved to **7.26 GB**, not the ~9.6 GB expected
from `0.8 x` the 12 GB cap, because auto-detect reads *live* MemAvailable and
other jobs had depressed it. ferric warned that usage exceeded it. This is
precisely the nondeterminism `tier4_dft`'s `mem_budget_gb` pin exists to remove.

**What remains, by elimination:** SCF convergence behaviour -- iteration count
and level-shift ladder rungs -- rather than the cost of any one iteration.
Something about this molecule's electronic structure (not its size, basis,
composition, or charge) makes the SCF expensive. The next probe should report
ITERATIONS and RUNGS, which `PyDftResult` does not currently expose; that is a
small addition to `crates/ferric-python/src/lib.rs` (`ScfResult` already
carries `iterations` and a typed `exit`, and `LadderResult.rung_outcomes`
carries the per-rung counts). Exact patch, written and then reverted UNBUILT
because the box hit load 59 with 255 MB free and a cold libint2 shim compile
needs ~8 GB:

```rust
// in `impl PyDftResult`, alongside `gradient`:
#[getter]
fn iterations(&self) -> usize { self.scf_data.iterations }

#[getter]
fn exit_reason(&self) -> String { format!("{:?}", self.scf_data.exit) }
```

`ScfExit` derives `Debug`, so `exit_reason` yields the variant name
(`"Converged"`, `"Plateau"`, `"Stalled"`, `"Diverged"`, `"MaxIter"`) — strictly
more informative than the `converged` bool, which collapses every failure mode
into `false`. Build with `cargo build --release -p ferric-python` and re-run
the control.

**Operational note:** `pkill` on the wrapper does NOT kill a ferric process
running inside a `scripts/ferric-limited` systemd scope -- the child survives
its parent. Kill the PID inside the scope directly (`kill -TERM <pid>`, then
`-KILL`). This bit twice today; the 9.4 GB process outlived two wrapper kills.

### Leads from reading the code (2026-09-02, NOT yet confirmed by timing)

Two structural facts about ferric's KS grid, both read from source rather than
inferred from a benchmark:

1. **Grid setup is O(natoms^3).** `becke_weights_all` (becke.rs:115) is
   O(natoms^2) per point via its nested a/b loop, and the number of grid points
   is itself O(natoms). Predicted setup ratio danuglipron:alkane_10 = **10.5x**,
   against **1.55x** for the XC pass. This is the only term found so far that
   scales anywhere near the observed runtime gap.
   - **But it is a ONE-TIME cost**, paid in `KsXc::new`, not per SCF iteration,
     and it is parallelized (order-preserving `into_par_iter`). So it cannot by
     itself explain a large per-ITERATION cost.
   - The energy path uses `becke_weights_all` (O(natoms^2)/point); the
     O(natoms^3)/point comment at grid.rs:247 belongs to the GRADIENT variant
     `becke_weights_and_grad`, which an energy-only run does not call. Do not
     quote grid.rs:247 as if it applied here.

2. **The AO cache is built once and reused** (`GridCache::Full`, ks.rs:372), so
   AO values are NOT re-evaluated per iteration unless batching engaged.

**What would settle it:** a run that separates one-time setup from per-iteration
cost, e.g. t(max_iter=1) vs t(max_iter=3), giving per_iter = (t3-t1)/2 and
setup = t1 - per_iter. Written as `scratchpad/split.py`; **not yet run to
completion on an uncontended box.**

**Measurement hygiene note.** Several timings attempted this session are
discarded, not reported: load average reached 19.65 on 12 cores with 2.8 GB
available while other users' `llama-server` jobs and my own probes overlapped.
An earlier reading of "alkane_10 at STO-3G did not finish in 1800 s" was also
WRONG -- it converges in 3 iterations / 23.5 s; I had misread an empty output
file (the ladder was stuck on a later, larger case) as a result for the first
entry. Timings taken under contention are not evidence, and an empty file is
not a measurement.

**Method note:** two of my probe scripts failed on ferric's API (`charge` belongs
to `Molecule.from_xyz`, not `run_dft`), and one of those failures printed a
`291 electrons` error that looked like a physics bug in the geometry. It was
not -- both geometry files check out (neutral 71 atoms/292 e-, anion 70
atoms/291 nuclear charge, 292 e- at q=-1, electron count conserved exactly as
the M10 fix intends). A broken probe imitating a physics failure is the same
trap as M9's two silent bugs: read the error, don't trust the headline.

**Verdict (dated, provisional): tier 4 as configured cannot process even ONE
70-atom candidate in a usable time, so the four-tier stack is validated only
through tier 3.** Candidate next steps, in order of cheapness: check whether the
SCF is converging at all (an anion with a diffuse HOMO may be cycling), try
`level_shift`, and profile where the 9.5 GB goes before assuming the basis is
the problem.

### A shared-box near-miss worth recording

The first tier-4 probe ran **unbounded** and reached 9.5 GB with 1 GB free while
another user's `mel-flow-tts` training job was on the same machine. This repo
has documented that exact collateral OOM three times. Re-running under
`scripts/ferric-limited` (12 GB cgroup cap) contained it — measured
`memory.current` 9.57 GB against `memory.max` 12 GB, and the other job survived.
**Any ferric job on a drug-sized system should start under that launcher, not
be moved there after watching RSS climb.**

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

---

## M11. Screening efficiency: the funnel spent 2.6x more than it needed to (2026-09-02)

The pipeline docks at `exhaustiveness=16`, while the setting VALIDATED by
redocking (M9, 0.95 A) was **32**. Neither number was measured against cost.
Tier 1 is also the only tier that touches every candidate, so it sets the
screen's price. `experiments/danuglipron/probes/exhaustiveness_sweep.py`
measures both axes on the one system with ground truth.

### Exhaustiveness is not resolvable on this target

Redock danuglipron into 7LCJ, 3 ETKDG seeds per level, best-of-10 RMSD against
the crystal pose. The ligand is re-embedded each time, so the search starts
from a geometry carrying no information about the answer.

| exhaustiveness | seed 61453 | 61454 | 61455 | mean | SEM | s/dock | cost |
|---|---|---|---|---|---|---|---|
| **4** | 1.11 | 1.14 | 0.75 | **1.00** | 0.13 | 26.4 | 1.0x |
| 8 | 1.11 | 1.14 | 0.75 | **1.00** | 0.13 | 34.5 | 1.3x |
| 16 | 1.20 | 0.92 | 0.75 | **0.96** | 0.13 | 67.6 | 2.6x |
| 32 | 1.24 | 1.15 | 0.77 | **1.05** | 0.14 | 102.8 | 3.9x |

**Every level passes the 2.0 A bar on every seed**, and the spread of the means
across an **8x range of search effort is 0.097 A -- SMALLER than the 0.131 A
within-level SEM**. The differences are not measurements of anything. Quoted as
SEM and not as a range, because a range grows with n (M6).

**No seed improves monotonically with effort**, and the most expensive setting
has the WORST mean (1.05 A at 32 vs 1.00 A at 4, for 3.9x the price):

| seed | ex=4 -> 8 -> 16 -> 32 |
|---|---|
| 61453 | 1.11 -> 1.11 -> 1.20 -> 1.24 (monotonically WORSE) |
| 61454 | 1.14 -> 1.14 -> 0.92 -> 1.15 (non-monotone) |
| 61455 | 0.75 -> 0.75 -> 0.75 -> 0.77 (flat) |

That is what search noise looks like. Anything read as a trend here would be
reading noise as signal.

**Seed matters more than search effort.** The 0.75-1.20 A spread tracks the
starting conformer, not exhaustiveness -- at seed 61455 every level returns
0.75 A to two decimals. For a screen this inverts the usual instinct: **more
seeds buy more than more exhaustiveness, at lower cost.**

`dock_ligand`'s docstring justified 32 on the grounds that "the failure being
fixed is a SEARCH failure, and under-searching would reproduce it in a new
form." Reasonable prior, now measured: it does not, at least for this pocket.
The claim was never wrong about M7's root cause -- free-solution conformers
really were the problem -- it just does not follow that MORE search helps once
docking is doing the searching at all.

### "Too clean" nearly hid the answer

Levels 4 and 8 returned RMSD **identical to two decimals on all three seeds**.
Per the protocol that is a stop condition, not a result, so I tested the
artifact hypothesis directly rather than writing it up: does exhaustiveness
change Vina's output at all? The control, at FIXED seed and geometry, `cpu=1`:

| exhaustiveness | wall | top score | poses returned |
|---|---|---|---|
| 1 | 32.2 s | -11.7930 | 2 |
| 4 | 109.0 s | -11.8370 | 5 |
| 32 | **738.0 s** | -11.8420 | 5 |

**The parameter is live** -- it finds more poses and marginally better scores,
and level 16 moved RMSD in BOTH directions (0.92 and 1.20 vs 1.14 and 1.11).
So this is a real property of the pocket, not a plumbing bug.

But it also quantifies the futility precisely: **ex=4 -> 32 costs 6.8x for a
0.005 kcal/mol score gain.** Vina's scoring function has a published RMSE
around 2.5 kcal/mol, so the purchased improvement is ~500x smaller than the
score's own error bar. And per M9, r(vina_score, RMSD) = +0.461 -- the score
only weakly tracks pose correctness anyway, so 0.005 kcal/mol of score is not
even 0.005 kcal/mol of pose accuracy.

**Buying score precision below a method's error bar is the general trap here**,
and it is what "exhaustiveness=32 for safety" amounts to on this target.

### The bigger lever was CPU allocation, and my first reading of it was wrong

Vina's `cpu=0` -- its default, and what this repo used -- means **"take every
core on the box"**. It also warns `At low exhaustiveness, it may be impossible
to utilize all CPUs`, which I read as "sequential docking leaves the machine
idle". Measured, that reading is too strong:

| ex=4, per dock | wall |
|---|---|
| `cpu=0` (12 cores) | 26.4 s |
| `cpu=1` (1 core) | 109.0 s |

Giving up 11 of 12 cores costs only **4.1x**, i.e. Vina's internal parallelism
runs at **34% efficiency** -- poor, but not idle. Fan-out across ligands
therefore only wins above ~4 workers (`w > t_one/t_all = 4.1`), which is a
narrower claim than "the machine sits idle".

### What a 60-ligand screen costs

| strategy | projected |
|---|---|
| today: ex=16, sequential, cpu=0 | **67.5 min** |
| ex=4, sequential, cpu=0 | 26.4 min |
| ex=4, 10 workers x cpu=1 | **10.9 min** |

**6.2x**, from two independent measured changes. This also explains M10's
missing time: the 55-minute run was tier 1 almost end to end, not the
"15-30 s/ligand" its docstring claimed.

Landed as `Stage(workers=k)` (process fan-out, results re-ordered to input
order so survivors are bit-identical to a serial run -- tested and
mutation-tested) and `dock_ligand(cpu=...)`, with tier 1 defaulting to `cpu=1`
because it runs inside that fan-out and two levels of parallelism
oversubscribe the box.

**Reproducibility caveat:** Vina's threads race to fill the pose buffer, so its
result depends on `cpu` as well as `seed`. A screen that varies core count
between runs is not reproducible even with identical seeds. Both are now
pinned.

---

## M12. Docked poses do NOT tighten the xtb scatter — and why that is consistent (2026-09-19)

`run_docked_pose_scatter.py`, `out/m7_docked_scatter.json`. Parent only,
15 Vina poses (ex=32, seed 0xF00D, cpu=1), rescored with the SAME
`tools.campaign.fit.pose_fit` M6 used.

| ensemble | pose_fit sd (kcal/mol) | mean pairwise RMSD |
|---|---|---|
| M6 rigid overlay | 34.23 | 3.98 A |
| M6 relaxed in field | 29.07 | 3.84 A |
| **M12 docked (this run)** | **28.75** | **5.81 A** |

**A 1% change.** Docking does not reduce the scatter that M5 and M6 closed the
other two routes against. The bar, stated in the probe BEFORE running, was
sd <= 0.88 (what a 0.25 kcal/mol ranking gap needs at n=100); this is short by
**32.5x**, essentially the same factor M6 was short by.

The artifact check passed in the other direction from M6's: mean pairwise RMSD
went UP (3.84 -> 5.81 A), so docked poses are geometrically MORE diverse. The
sd is not being held up by a collapsed ensemble.

### The number that actually locates the problem

Vina's own score on those same 15 poses has **sd = 0.83 kcal/mol**. Rescoring
the identical geometries with xtb gives **sd = 28.75**. The two scores disagree
by a factor of ~35 on how much the poses differ.

That is M9's `r(vina_score, RMSD) = +0.461` seen from the energy side rather
than the geometry side, and it sharpens the hierarchy's justification: the
cheap tier does not merely rank imperfectly, it reports a nearly FLAT energy
landscape across poses that xtb sees as spanning 103 kcal/mol. Tier 1 is a pose
GENERATOR whose score carries almost no ranking information; that is not a
defect to fix but the reason tiers 2-4 exist.

### Consistency with M9, which this does not contradict

M9 showed docking SOLVES pose generation (0.95 A redock, 20/20 poses on-site).
M12 shows it does not solve pose SCORING. Both are true because they are about
different stages: the right geometry is now reliably IN the candidate set
(M9), and the ensemble of candidates still spans ~100 kcal/mol under the
scoring metric (M12). Averaging over that ensemble is still the wrong
instrument for a 0.25 kcal/mol question.

**Route table, updated:**

| route | status |
|---|---|
| more poses (M4/M5) | closed -- sd flat in n |
| relax poses in field (M6) | closed -- real 15%, ~3 orders short |
| real pose search (M12) | **closed -- 1%, 32.5x short** |

All three routes to rescuing a per-pose-averaged ranking are now closed with
numbers. What M9 leaves open is different and is the live path: do not average
over poses at all -- **select** the pose (docking gets within 1 A) and score
that one, accepting that the selection is the assumption.

### A methodology error this probe made, recorded because it nearly shipped

The first run compared Vina's `vina_score` sd (0.83) against M6's 29.07 and
reported a **"35x reduction, bar cleared"**. That is a category error: the two
are different quantities on different scales (~-11 vs ~-160 kcal/mol), and
`vina_dock.py`'s own docstring calls `vina_score` "empirical -- a ranking
heuristic only". The tell was tidiness -- 0.83 landing just under a 0.88 bar
stated minutes earlier is the "too clean is a stop condition" signature. The
corrected probe rescores with the same instrument, and the answer inverts from
"closes the gap" to "1% change". Same data, opposite verdict, entirely because
of which scale was read.

---

## M13. "Select a pose" is WORSE than averaging — retracting M12's recommendation (2026-09-19)

M12 closed the three averaging routes and concluded: *"do not average over
poses at all -- SELECT the pose (docking gets within 1 A) and score that."*
**That recommendation is wrong, and this retracts it.** Same 15-pose M12 data,
analysed for the question M12 did not ask.

### The measurement

Vina returns poses rank-ordered, so index 0 IS the selected pose.

| estimator | pose_fit (kcal/mol) | bias vs mean |
|---|---|---|
| **selected** (vina rank 0) | −83.46 | **+27.43** |
| mean over 15 | −110.89 | 0 (by definition) |
| min over 15 | −160.78 | **−49.89** <- the v1 estimator |

And **no association between the two axes is detectable in this sample**:

    Spearman(vina rank, pose_fit) = -0.261   p = 0.35   n = 15
    Pearson                       = -0.328   p = 0.23

**That is not a demonstration of independence, and this section originally
claimed it was.** At n = 15 the smallest correlation this test could have
detected at p < 0.05 is **|ρ| = 0.514**, and the 95% CI on ρ runs
**[−0.682, +0.290]** — it does not exclude a *strong* negative correlation.
The honest statement is "no detectable association in 15 poses", which is a
statement about the probe's power, not about Vina.

### Why that kills the recommendation ANYWAY

The recommendation dies on PRECISION, and — this is the part that matters —
**the verdict does not depend on the independence claim that was wrong.**

If the axes were genuinely uncorrelated, "pick rank 0" is equivalent to
drawing one sample at random from a distribution with sd 28.75. For a ddE
between two analogues (two draws):

| protocol | noise on ddE | vs the 0.25 kcal/mol gap |
|---|---|---|
| select one pose | sd·√2 = **40.66** | **163x** |
| average n = 100 | SEM·√2 = **4.07** | **16x** |

That is a factor √n = 10x, and it is **conditional on ρ = 0**.

So test the conclusion at the edge of what the data allow. At **ρ = −0.682**,
the most favourable value the CI permits, a rank-0 pick still carries residual
sd `28.75·√(1−ρ²) = 21.08` about its conditional mean:

| ρ | selection noise on ddE | vs average n=100 |
|---|---|---|
| 0 (the original claim) | 40.66 | 10.0x worse |
| −0.682 (best the CI allows) | 29.81 | **7.3x worse** |

**The ranking of the two protocols is robust across the whole interval.** The
exact 10x is not — quote it as "roughly an order of magnitude, 7-10x depending
on a correlation this probe could not resolve". Both protocols remain far short
of the 0.25 kcal/mol gap either way. **Averaging is the better of two
inadequate options, not the worse one.**

### A latent bug found while fixing the wording

`analyze_selection_bias.py` set `informative = ps < 0.05` and then branched on
`if informative is False:`. scipy returns a **`numpy.bool_`**, and
`np.bool_(False) is False` evaluates to **False** — so that branch never fired
and **the script printed no verdict at all**, for its entire life. Nobody
noticed because a missing line looks like nothing; the numbers above it printed
correctly every time, and I read the verdict out of the numbers myself.

Fixed with an explicit `bool(...)`. The general form is worth keeping: `is
True` / `is False` against anything that has passed through numpy or pandas is
an identity check that silently fails. Use truthiness, or coerce at the
boundary.

Note what a nonzero ρ would and would not buy. Correlation with `pose_fit`
changes the BIAS of rank-0 selection, not the per-draw variance; variance only
falls insofar as the selector tracks the quantity being estimated. Neither
mechanism closes a 16x shortfall.

### What I mis-read, and it was already written down

M12's case for selection was "M9 redocks danuglipron to 0.95 A". M9 itself says
why that does not license selection, two paragraphs below its own headline:

> r(vina_score, RMSD) = **+0.461**, and only **4 of 20** poses are under 2.0 A.
> Vina put the right pose first here, but the correlation is weak enough that
> it did so **partly by luck**.

0.95 A is one draw from a weak selector, not a property of the protocol. I
quoted the headline and not the caveat directly under it.

### The distinction that survives, and matters

**Geometric selection and energetic selection are different claims.** Docking
reliably puts a near-native pose SOMEWHERE in its candidate set (M9: 20/20
within 5 A of the site). It does not reliably put it FIRST, and its ranking
axis carries no information about the xtb energy (this section). So:

* for *"where does this ligand sit?"* — docking is the right tool, use it
* for *"which analogue binds better by 1 kcal/mol?"* — no protocol built on
  this scoring metric works, selected or averaged

### Status of the pose problem: OPEN, and all four routes closed

| route | status |
|---|---|
| more poses (M4/M5) | closed — sd flat in n |
| relax in field (M6) | closed — real 15%, ~3 orders short |
| real pose search (M12) | closed — 1%, 32.5x short |
| **select one pose (M13)** | **closed — 7-10x WORSE than averaging** |

The remaining honest options are not protocol changes: reduce the per-pose sd
at its source (a scoring function less sensitive to pose than the current
point-charge interaction energy), or accept that this metric answers "does it
bind here" and not "which analogue is better", and rank on something else.

### Method note

This cost nothing to run — it is a re-analysis of M12's existing JSON, asking a
question M12 did not. Worth stating because the expensive part (15 docked poses
+ 15 xtb rescores, ~9 min) was already paid, and the finding that inverts the
recommendation came from four lines of arithmetic on data already on disk.

---

## M14. No available scorer is less pose-sensitive — the last lever is closed (2026-09-19)

`run_scorer_pose_sensitivity.py`, `out/m14_scorer_sensitivity.json`. M13 left
one route open: *"the lever is a scoring metric less pose-sensitive than a
point-charge interaction energy."* This tests it on the cheapest possible
experiment — score the SAME docked geometries with every scorer in the repo.

19 poses, all three scorers, all 19 scored by all three.

| scorer | mean | sd | **CV** | Spearman vs pose_fit |
|---|---|---|---|---|
| vina_score | −11.06 | 0.784 | **0.071** | −0.202 (p=0.41) |
| pose_fit (xtb) | −104.97 | 33.06 | **0.315** | — (reference) |
| prescreen (classical) | −0.0055 | 0.0163 | **2.955** | +0.353 (p=0.14) |

**The figure of merit is the coefficient of variation, sd/|mean|**, because a
raw sd is only meaningful on its own scale — comparing sds across scorers is
exactly the category error M12 made.

**Neither alternative is a candidate:**

* **prescreen is 9.4x WORSE** than pose_fit, not better. The classical field
  score is *more* pose-sensitive, which is the opposite of the hypothesis.
* **Vina looks 4.4x smoother and is not measuring the same thing.** Spearman
  −0.202, p=0.41 — indistinguishable from zero. It is smooth because it is
  insensitive, which is the artifact hypothesis this probe wrote down before
  running, and it is the same finding as M13's from the other direction.

The artifact hypothesis was stated in advance precisely so "low CV" could not
be reported as a win on its own, and it earned its keep: without the
correlation column, Vina's 0.071 reads as a 4x improvement.

### A near-miss worth recording

The first run scored **0 of 19** poses. `embed_ligand_from_coords` refused with
*"263 electrons with multiplicity 1 implies n_alpha = 263/2"*.

Cause: **PDBQT is united-atom.** Nonpolar hydrogens are merged into their
carbons, so a docked danuglipron has 42 atoms and 263 electrons where the real
molecule has 71 and 292. 263 is odd, so a singlet is arithmetically impossible.

What makes this worth writing down is that **`pose_fit` accepted the same
structure without complaint** — xtb will happily run on a molecule missing 29
hydrogens. Two scorers on identical input, one refusing and one silently
scoring an incomplete species. Every pose_fit number in M12/M13 was computed on
the united-atom structure; those conclusions are about SCATTER and survive
(the same systematic omission is in every pose), but no absolute pose_fit
energy from a docked pose should be quoted as this molecule's interaction
energy.

Fixed in this probe by rebuilding the full-hydrogen topology from SMILES,
pinning the docked heavy atoms, and MMFF-relaxing only the hydrogens — so the
pose being scored is still the pose that was docked. M9 hit the mirror image of
this bug in its alignment check.

### And a guard the probe needed against itself

The first verdict reported `ddE_noise_at_n100 = 0.0023 kcal/mol`, which would
have been a spectacular result. It was prescreen's sd (0.0163) divided by √100
— a scale artifact, since prescreen's mean is −0.0055. The verdict now takes
the lowest-CV scorer **that still tracks the reference**, giving **4.68
kcal/mol**, consistent with M5's 4.07.

### Status: all five routes closed

| route | status |
|---|---|
| more poses (M4/M5) | closed — sd flat in n |
| relax in field (M6) | closed — real 15%, ~3 orders short |
| real pose search (M12) | closed — 1%, 32.5x short |
| select one pose (M13) | closed — 7-10x worse than averaging |
| **a different scorer (M14)** | **closed — none available is less pose-sensitive** |

Nothing in this repo ranks analogues at 1-2 kcal/mol against a pose ensemble.
That is now measured from five directions rather than assumed. A sixth route
exists and is outside this campaign: a scorer that is pose-averaged *by
construction* (free-energy perturbation, or an ML affinity model trained on
ensembles) rather than a single-pose energy.

> **AMENDED 2026-09-19 (M17).** All five routes attack the per-pose **sd**, and
> all five accept the estimator. A seventh route attacks the **estimator**: every
> one of them computed `ddE` over INDEPENDENTLY embedded ensembles, which is an
> UNPAIRED design over noise that is largely common to the two molecules. See
> M17 — this closure is provisional pending a pocket-based test of the paired
> construction.

---

## M15. The cheap gate is SITE-BLIND by construction (2026-09-19)

The pipeline's stated unit is the **(substituent, SITE) pair** — measured
within/between ratio 0.94–0.95, i.e. *where* a group goes matters as much as
*which*. This checks whether the cheap descriptor gate, which is what survives
after M4–M14 closed every pose route, can actually express that unit.

It cannot.

    substituent   sites   distinct descriptor tuples
    CF3             9       1    all (67.997, 1.0188, 0.0)
    CN              9       1    all (25.010, -0.1283, 23.79)
    F               9       1    all (17.990, 0.1391, 0.0)

Nine sites, **one** value each.

### It is inherent, not a defect

MW, cLogP (Crippen) and TPSA are whole-molecule sums over atoms and fragment
types. Two constitutional isomers have the same atoms and the same fragment
types, so all three are identical BY CONSTRUCTION. Verified on the simplest
possible case — ortho/meta/para fluorobenzoic acid:

    Fc1ccccc1C(=O)O      MW=140.113  cLogP=1.524  TPSA=37.30
    O=C(O)c1cccc(F)c1    MW=140.113  cLogP=1.524  TPSA=37.30
    O=C(O)c1ccc(F)cc1    MW=140.113  cLogP=1.524  TPSA=37.30

Bit-identical. The gate is CORRECT; the descriptors simply do not carry
positional information, and no fix to `relative_descriptors` changes that.

### What this means for the pipeline

Combining M15 with M4–M14 gives the pipeline's actual, honest scope:

| question | answerable? | by what |
|---|---|---|
| which SUBSTITUENT is more promising? | **yes** | relative descriptors — discriminating and chemically sensible (CN lowers cLogP, CF3 is worst) |
| which SITE should it go on? | **no** | cheap gate is site-blind (M15); pose-based ranking is noise-limited (M4–M14) |
| does an analogue bind here at all? | **yes** | docking, 0.95 Å redock (M9) |
| how much better does it bind? | **no** | ddE noise 4.07 kcal/mol vs effects of 1–2 |

**Both halves of the pipeline's stated unit are now blocked, for different
reasons.** The substituent axis is answerable and the site axis is not — the
cheap tier cannot see position and the expensive tier cannot resolve it. A
campaign should therefore rank SUBSTITUENTS cheaply, and treat placement as a
question for chemistry knowledge or an experiment rather than for this pipeline.

That is a narrower claim than "the pipeline proposes viable substitutions", and
it is the one the measurements support.

### What WOULD see a site

Descriptors that carry positional information exist and none is wired here:
3-D shape/electrostatic similarity, per-atom partial charges at the substitution
point, or a QM property evaluated at the site (e.g. local electrostatic
potential). Any of those is a real addition rather than a fix, and should be
costed before being built.

---

## M16. xTB does NOT rank conformers the way DFT does at ~3 kcal/mol (2026-09-19)

`run_xtb_vs_dft_tracking.py`, `out/m16_n20.json`. The golden path costs tier 3
(~0.5 s/pose) and says it "ranks survivors", but the only accuracy claim
attached to it was a **143 kcal/mol** anion/neutral split — a gap so large that
resolving it says nothing about ordering conformers a few kcal/mol apart, which
is what ranking survivors means.

Measured directly: 20 conformers, xtb-relaxed, then **the DFT single point taken
at the xtb-RELAXED geometry** so both tiers score the same structures.

| | |
|---|---|
| conformers scored by both | 20 |
| xtb span / DFT span | 2.67 / 2.81 kcal/mol |
| MAE on relative energies | **0.825 kcal/mol** |
| **Spearman rho** | **0.011** (p = 0.96) |

**No detectable correlation.** On a set spanning ~3 kcal/mol, xtb's ordering
carries no information about DFT's *that 20 conformers can resolve* — the 95%
CI on rho is [−0.434, +0.451], so a moderate association is not excluded. What
IS established is the operational point: this probe cannot find an ordering
signal to rely on, so tier 3's output may not be used as a fine ranking.

### The n=8 run was underpowered and would have been a false positive

The first run used 8 conformers and gave rho = 0.643, p = 0.086 — "positive but
not significant", which reads like a near-miss worth more sampling. It was not:

    n=8  can only detect rho >= 0.707 at p<0.05
    n=20 can detect rho >= 0.444

So 0.643 at n=8 was inside the noise floor of the test itself — its 95% CI was
[−0.113, +0.927], which spans everything from mildly negative to nearly
perfect. At n=20, where 0.643 WOULD have been significant, the value came out
at 0.011.

**The precise claim: the n=8 estimate was too uncertain to support a ranking
conclusion, and it did not replicate.** That is not the same as proving it was
sampling noise — a single non-replication cannot establish which of the two
runs was the fluke. It does not need to: an estimate whose CI spans
[−0.113, +0.927] licenses nothing regardless of what the next run shows, and
quoting it would have been a power failure dressed as a finding.

### What this means for the funnel

Tier 3 is a **coarse gate, not a ranker** — at least for conformer selection at
this energy scale. It is still the right tool for what M-series measured it on:
a 143 kcal/mol anion/neutral split is resolved trivially. The failure is
specific to fine ordering.

Combined with M14 (no detectable association between Vina's score and xtb's)
the pattern is
consistent across the whole funnel: **each tier reliably separates things that
are grossly different and does not reliably order things that are close.** That
is the hierarchy working as designed, and it bounds what any single tier's
output may be used for.

### Scope

One molecule (a paracetamol-like scaffold with one flexible tail), one
conformer set, STO-3G. This measures whether the tiers AGREE, not whether
either is right — neither is validated against experiment here. A larger basis
or a system with bigger conformer gaps could give a different answer, and the
probe takes `--smiles` / `--basis` / `--n-conformers` so that is testable.


## M17. The five closed routes were all UNPAIRED (2026-09-19)

The pose problem is recorded closed from five directions (M4-M14). **Every one
of them attacks the per-pose sd and accepts the estimator.** This probe attacks
the estimator instead.

### The observation

All five computed

    ddE = mean(E_B over ensemble_B) - mean(E_A over ensemble_A)

with the two ensembles embedded INDEPENDENTLY. That discards the structure of
the problem. The ~28.75 kcal/mol per-pose scatter is pose-conformational -- a
property of the scaffold sitting in the pocket -- while a substitution changes a
handful of atoms and leaves ~68 where they were. Variance common to both
molecules cancels in a paired difference:

    var(ddE_paired) = sd_A^2 + sd_B^2 - 2*rho*sd_A*sd_B
                    = 2*sd^2*(1 - rho)   WHEN the two spreads are equal
    var(ddE_unpaired) = sd_A^2 + sd_B^2

A is the PARENT and B the analogue, so a positive ddE means the analogue sits
higher -- the sign `paired_ddE` computes and the sign of every value in the
table below. The spreads are equal here to a few percent, and the last column
is MEASURED as the ratio of two SEMs on the same data rather than derived from
the formula, so the reported gain does not rest on that.

**A shared random seed is not a pairing.** `embed_analogue` uses
`random_seed=0xF00D` for every candidate, but ETKDG with the same seed on two
different molecular graphs gives uncorrelated conformers: the seed indexes a
random stream, not a geometry. The pairing has to be geometric.

### The measurement

`tools/morph/paired.py`. Paracetamol-like parent (34 atoms), 24 ETKDG poses,
MMFF energies. Pose k of the analogue is BUILT FROM pose k of the parent,
sharing the MCS scaffold. Two constructions for the B side:

| arm | case | ddE | sd | SEM | rho | SEM ratio |
|---|---|---:|---:|---:|---:|---:|
| hard | **SELF** | **+13.841** | 2.597 | 0.530 | 0.579 | 1.37x |
| hard | Cl-for-H | 31.023 | 7.946 | 1.622 | 0.070 | 1.01x |
| relaxed | **SELF** | **+0.004** | 0.005 | 0.001 | 1.000 | 422x |
| relaxed | F-for-H | 8.987 | 1.084 | 0.221 | 0.860 | 2.42x |
| relaxed | Cl-for-H | 13.064 | 3.011 | 0.615 | 0.399 | 1.21x |
| relaxed | N-methyl | 24.135 | 0.034 | 0.007 | 1.000 | 65.5x |

The last column compares the paired SEM against the unpaired SEM **on the same
data**, so it isolates the pairing. **It is a STANDARD-ERROR ratio, not a
variance ratio** -- the corresponding variance reduction is its SQUARE (2.42x
on the SEM is ~5.9x on the variance). It was labelled `var.red` until review
caught it; that name understated the variance effect while overstating what had
been measured. `rho` is Pearson between the two paired energy
series; `var = 2*sd^2*(1-rho)` assumes the two sds are EQUAL, which they are
here to within a few percent, but the reported SEM ratio is measured rather than
derived from that formula and does not depend on the assumption.

**A latent index bug was found in review and fixed; these numbers are
UNCHANGED by it** (re-measured after the fix, identical to 3 decimals). The
first pass mapped stripped-molecule indices back to the original by POSITION
among the non-hydrogens, and `RemoveHs(sanitize=False)` RETAINS degree-zero
hydrogens -- RDKit even warns "not removing hydrogen atom without neighbors".
A stray atom would shift every index after it and pair the wrong atoms
SILENTLY. This parent has no such atom, so the measurement was unaffected, but
a perceived-from-XYZ molecule easily does.

The self-anchor could not have caught it: it measures the same `pairs` used to
build `coord_map`, so a wrong correspondence gets pinned to the parent's
coordinates and measures as ZERO drift -- the check is downstream of the defect.
Fixed by stamping the original index on each atom before stripping, plus an
element-identity check that does NOT share that failure mode.

### The exactness anchor did the work, twice

`SELF` is the parent paired with itself: no substitution, so ddE must be **0**.
Pinning the scaffold hard charges **+13.841 kcal/mol for doing nothing** -- the
substituent is forced into whatever room the parent pose left. The strain is
substituent-DEPENDENT (F +9.5, Cl +17.2, N-methyl +31.1 above the self value),
so subtracting it does not fix it. Cross-checked against freely relaxed
geometries: +12.2 (F) and +17.0 (Cl) kcal/mol of real strain.

Relaxing the substituent against a spring-restrained scaffold passes the anchor
at **+0.004** while the scaffold still holds to 0.011-0.128 A.

### The too-clean check, which it passed

`rho = 1.000` on two relaxed rows is a stop condition, not a result. MMFF
minimisation collapsing 24 poses onto one geometry would drive sd to 0 and rho
to 1 by destroying the ensemble -- the failure M6 checked for. MEASURED:

| case | mean pairwise RMSD before | after | retained |
|---|---:|---:|---:|
| SELF | 4.350 | 4.350 | 100.0% |
| F-for-H | 4.359 | 4.359 | 100.0% |
| N-methyl | 4.447 | 4.460 | 100.3% |

No collapse. The poses stay 4.35 A apart; only embedding strain is removed
(energy sd 3.17 -> 1.59). The correlation is real.

### An inert guard, found and replaced

The module first flagged a disagreement between `ddE_paired` and `ddE_unpaired`.
**That guard could never fire**: `mean(E_B - E_A)` and `mean(E_B) - mean(E_A)`
are the same number. It printed identical values in all four rows and I read
that as agreement rather than as an identity. Pairing changes the estimator's
VARIANCE, never its value. The live guard is the self-anchor.

### Status: PROVISIONAL, and what it does not license

MEASURED on one molecule, one force field, gas phase, **no pocket**. The relaxed
SEM (0.221-0.615 kcal/mol) is below the 1-2 kcal/mol effect size where 4.07 was
well above it. **That is a reason to run the pocket experiment, not a ranking.**

Not established: that this survives at xtb or DFT (not smooth force fields), or
IN A POCKET, where a substituent may change the binding mode and break the
pairing outright.

**The Cl row is the warning**: rho 0.399, SEM ratio only 1.21x. Pairing helps where the
substitution is LOCAL and degrades smoothly to the unpaired case where it is
not -- so it must be reported per-candidate, never as one campaign-wide floor.
