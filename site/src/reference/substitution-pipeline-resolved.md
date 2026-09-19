# How we make a pipeline for proposing viable active-site substitutions

**Resolved 2026-09-19.** The question has been open across several sessions
because one design decision was genuinely undecided: whether to average over a
pose ensemble or to select a single pose. M12 closes it with a measurement, so
the pipeline shape below is now determined rather than chosen.

Everything here is MEASURED on danuglipron against the GLP-1R pocket (7LCJ)
unless labelled otherwise. Sources: `experiments/danuglipron/RESULTS.md`
(M4-M12), `wiki/golden-path-pipeline-2026-09-18.md`.

---

## The decision that was blocking: select, do not average

Per-pose energy scatter is **sd ~29 kcal/mol** against substituent effects of
**1-2 kcal/mol**. Three routes to averaging that away have now been tried, and
all three are closed with numbers:

| route | result | verdict |
|---|---|---|
| more poses (M4/M5) | sd flat in n; SEM falls as 1/sqrt(n) but sd does not move | closed -- resolving a 0.25 kcal/mol gap needs ~7350 poses |
| relax poses in field (M6) | 34.23 -> 29.07, a real 15% tightening, poses stay distinct | closed -- ~3 orders of magnitude short |
| real pose search (M12) | 29.07 -> 28.75, **1%**, poses MORE diverse (RMSD 3.84 -> 5.81 A) | closed -- 32.5x short |

So the answer is not a better ensemble. **The pipeline must select one pose per
analogue and score that**, stating the selection as an assumption.

That is defensible because pose GENERATION is solved for this target: M9
redocks danuglipron into 7LCJ at **0.95 A** (best-of-20 and top-ranked both),
20/20 poses within 5 A of the known site. Before docking, the best of 20 RDKit
conformers was 2.23 A and nothing cleared the conventional 2.0 A bar.

### The number that makes the decision concrete

On the SAME 15 docked geometries:

```
Vina's own score      sd =  0.83 kcal/mol
xtb (pose_fit)        sd = 28.75 kcal/mol
```

The cheap tier sees a nearly flat landscape where xtb sees one spanning
103 kcal/mol. This is M9's `r(vina_score, RMSD) = +0.461` viewed from the
energy side, and it is the empirical case for the whole hierarchy: **tier 1
generates the right answer among its candidates and cannot pick it out.** If it
could, tiers 2-4 would be decoration.

---

## The pipeline

```
P0  parent + site SMARTS + pocket PDB
P1  ENUMERATE      propose_substitutions          -> SMILES per (substituent, SITE)
P2  DESCRIPTOR     relative_descriptors           -> gate vs the PARENT, not absolutes
P3  EMBED          embed_proposals (ETKDG, seeded) -> symbols + coords
P4  DOCK           dock_ligand per analogue        -> SELECT one pose  <-- the decision
P5  PRESCREEN      prescreen_pose (classical)      -> cheap triage, no SCF
P6  RANK           xtb pose_fit                    -> ddE vs the parent
P7  CONFIRM        ferric DFT + D3(BJ)             -> survivors only
```

Steps P1-P3 and P5 exist and are tested (#96, #102). P4 exists
(`tools/docking/vina_dock.py`) and is validated by M9's redock. P6's engine
exists (`tools/campaign/xtb_engine.py`). P7 needs #99's D3(BJ) to land, without
which a halogen/CF3 scan is missing its dominant attractive term.

### The four rules the pipeline must follow, each from a measurement

1. **ddE, never absolute.** The parent is carried through every stage for this
   reason. An absolute binding energy carries the full method error; the
   difference cancels most of it. Same argument as P2's relative gate.

2. **The unit is the (substituent, SITE) pair, not the molecule.** MEASURED
   within/between ratio **0.94-0.95** on two independent constructions: WHERE a
   group goes matters as much as WHICH group. A pipeline keyed on substituent
   alone averages over the larger effect.

3. **One row per molecule is now CORRECT.** `funnel.py`'s one-row-per-candidate
   keying was previously written up as a blocker ("cannot express an
   ensemble"). Given P4 selects a pose, there is no ensemble to express, and
   the existing shape is right. This is the concrete thing M12 changed.

4. **Ionization state is part of the measurement.** Danuglipron's carboxylic
   acid is deprotonated at pH 7.4; the anion/neutral split is 143 kcal/mol
   (M-series). Net charge is an explicit input at every scoring stage, never
   defaulted to 0.

### Cost, from the measured hierarchy

| tier | method | s/pose | population | job |
|---|---|---|---|---|
| 1 | AutoDock Vina | 1e-5 | 1e5-1e6 | **search** pose space |
| 2 | MMFF94 | 1e-3 | 1e2-1e3 | relax, declash |
| 3 | GFN2-xTB | 5e-1 | 10-1e2 | rank survivors |
| 4 | ferric DFT + D3(BJ) | 6e+2 | 1-10 | final energetics |

Tier 4 is **not yet validated in this pipeline** -- M10 recorded that it does
not fit the funnel as configured. Do not present tier-4 numbers as the
pipeline's output until that is resolved.

---

## What is still NOT settled, stated plainly

* **The selection is an assumption, not a proof.** Docking gets within ~1 A on
  THIS target, where a crystal pose exists to check against. On a novel target
  there is no such check, and the pipeline's output inherits that uncertainty.
  Report it; do not launder it.
* **No ranking has been validated end to end.** M12 closes the averaging
  routes; it does not demonstrate that selected-pose ddE ranks analogues
  correctly. That needs a held-out set with known relative affinities, which
  this campaign does not have.
* **Tier 4 unvalidated** (above), and dispersion (#99) not yet merged.
* **The scan's prescreen is classical.** It triages; it does not rank.

## Honest status line

> The pipeline's SHAPE is now determined: enumerate relative to the parent,
> dock to select a pose, score that pose, report ddE per (substituent, site).
> Every stage exists in code and four of them are validated. What is not
> established is that the resulting ranking is correct -- only that the three
> obvious ways of getting it wrong by averaging have been measured and closed.
