# How we make a pipeline for proposing viable active-site substitutions

**2026-09-19, revised the same day.** The pipeline's STAGES are settled and
every one of them exists in code. What is NOT settled -- and what this note now
says plainly, having briefly said the opposite -- is that its ranking can be
trusted at the 1-2 kcal/mol resolution a substitution campaign needs. All four
pose protocols have been measured and all four fail.

Everything here is MEASURED on danuglipron against the GLP-1R pocket (7LCJ)
unless labelled otherwise. Sources: `experiments/danuglipron/RESULTS.md`
(M4-M13), `wiki/golden-path-pipeline-2026-09-18.md`.

---

## RETRACTION, same day (M13)

An earlier version of this note recommended **"select one pose, do not
average"**. That was wrong and is retracted below. The correction does not
change the pipeline's STAGES -- P1-P7 stand -- but it changes what the output
may be used for, which is the more important half.

Short version: selecting the top-docked pose is **10x worse** than averaging,
because Vina's ranking axis is statistically independent of the xtb scoring
axis (Spearman -0.261, p=0.35), so "pick rank 0" is a single random draw from a
distribution with sd 28.75 kcal/mol. Averaging at n=100 gives ddE noise 4.07;
selecting gives 40.66. Both miss the 0.25 kcal/mol gap, by 16x and 163x.

I had justified selection with "M9 redocks to 0.95 A". M9 says two paragraphs
under its own headline that it did so **"partly by luck"** (r = +0.461, only
4/20 poses under 2.0 A). I quoted the headline and not the caveat beneath it.

**What this means for the pipeline:** the stages below are right, and the
ranking they produce is NOT trustworthy at 1-2 kcal/mol resolution by any pose
protocol currently available. Use it to answer *"does this analogue bind in
this site at all"*, not *"which of these two is better"*. See
"What the pipeline may and may not claim" at the end.

## The decision that WAS thought to be blocking: select vs average (SUPERSEDED)

Per-pose energy scatter is **sd ~29 kcal/mol** against substituent effects of
**1-2 kcal/mol**. Three routes to averaging that away have now been tried, and
all three are closed with numbers:

| route | result | verdict |
|---|---|---|
| more poses (M4/M5) | sd flat in n; SEM falls as 1/sqrt(n) but sd does not move | closed -- resolving a 0.25 kcal/mol gap needs ~7350 poses |
| relax poses in field (M6) | 34.23 -> 29.07, a real 15% tightening, poses stay distinct | closed -- ~3 orders of magnitude short |
| real pose search (M12) | 29.07 -> 28.75, **1%**, poses MORE diverse (RMSD 3.84 -> 5.81 A) | closed -- 32.5x short |

So the answer is not a better ensemble. It is also **not** selection -- M13
measured that and it is worse (see the retraction above). A fourth row belongs
in that table:

| select one pose (M13) | ddE noise 40.66 vs averaging's 4.07 | closed -- 10x WORSE |

Pose GENERATION is nonetheless solved for this target: M9 redocks danuglipron
into 7LCJ at **0.95 A**, 20/20 poses within 5 A of the known site, where the
best of 20 RDKit conformers was 2.23 A. That licenses *"the near-native pose is
in the candidate set"*. It does not license *"the first one is it"* -- M9's own
r(vina_score, RMSD) = +0.461 says otherwise.

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
P4  DOCK           dock_ligand per analogue        -> pose ENSEMBLE (do NOT take rank 0)
P5  PRESCREEN      prescreen_pose (classical)      -> cheap triage, no SCF
P6  RANK           xtb pose_fit, MEAN over the ensemble -> ddE vs the parent
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

2. **The unit is the (substituent, SITE) pair -- and NEITHER TIER CAN
   EXPRESS IT.** MEASURED within/between ratio **0.94-0.95** on two independent
   constructions: WHERE a group goes matters as much as WHICH group. But:

   - the CHEAP gate is **site-blind by construction** (M15). MW, cLogP and
     TPSA are whole-molecule sums, so 9 sites of one substituent give ONE
     descriptor tuple. Verified on ortho/meta/para fluorobenzoic acid: bit
     identical. No fix to `relative_descriptors` changes this.
   - the EXPENSIVE tier is **noise-limited** (M4-M14): ddE noise 4.07 kcal/mol
     against effects of 1-2.

   So the substituent axis is answerable and the site axis is not. Rank
   SUBSTITUENTS cheaply; treat placement as a question for chemistry knowledge
   or an experiment, not for this pipeline. Seeing a site would need a
   POSITIONAL descriptor (3-D shape, per-atom charge, a QM property at the
   site) -- an addition to cost, not a fix to apply.

3. **One row per molecule is the WRONG shape, after all.** An earlier version
   of this note said the opposite, on the strength of the now-retracted
   selection recommendation. With averaging restored as the least-bad
   estimator, `funnel.py` does need to express an ensemble -- it keys one row
   per candidate (`funnel.py:162`). This is a real, open gap, and it was
   briefly recorded as closed.

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

## What the pipeline may and may not claim

**May:** "this analogue docks into this site, and here is where." Pose
GENERATION is validated (M9: 0.95 A redock, 20/20 within 5 A). The enumeration,
the relative descriptor gate and the classical prescreen all work and are
tested.

**May not, part 1:** "put this group at THIS position." The cheap gate cannot
see position at all (M15, inherent), and the expensive tier cannot resolve the
difference. Both halves of the stated unit are blocked, for different reasons.

**May not, part 2:** "analogue A binds better than analogue B by 1-2 kcal/mol."
No pose protocol available today supports that. The five measured options:

| protocol | ddE noise (kcal/mol) | vs the 0.25 gap |
|---|---|---|
| select one pose (M13) | 40.66 | 163x |
| average n = 100 (M5) | 4.07 | 16x |
| a different scorer (M14) | 4.68 (best tracking) | 19x |
| relax then average (M6) | ~4.1 | ~16x |
| dock then average (M12) | ~4.1 | ~16x |

Averaging is the least-bad option and is still 16x short. Reporting a ranking
from this would be reporting noise.

## What is still NOT settled

* ~~**The lever left is a scorer less sensitive to pose.**~~ **CLOSED by M14
  (2026-09-19).** Scored the SAME 19 docked poses with every scorer in the
  repo, comparing coefficient of variation (dimensionless, so comparable
  across scales):

  | scorer | CV | Spearman vs pose_fit |
  |---|---|---|
  | vina_score | 0.071 | -0.202 (p=0.41) |
  | pose_fit (xtb) | 0.315 | reference |
  | prescreen (classical) | 2.955 | +0.353 (p=0.14) |

  prescreen is **9.4x worse**. Vina looks 4.4x smoother and does not track
  pose_fit at all -- smooth because it is insensitive, not because it is
  better. **No scorer in this repo is less pose-sensitive**, so the remaining
  route is one that is pose-averaged BY CONSTRUCTION (FEP, or an ML affinity
  model trained on ensembles), which is outside this campaign.
* **`funnel.py` cannot express an ensemble** (`funnel.py:162`, one row per
  candidate). Whether that is worth fixing depends on the resolution you want,
  and the answer is measured (sd = 33.06 from M14's 19 docked poses):

  | poses averaged | ddE noise | vs the 0.25 kcal/mol gap |
  |---|---|---|
  | 1 | 46.75 | 187x |
  | 15 | 12.07 | 48x |
  | 100 | 4.68 | 19x |
  | 1000 | 1.48 | 6x |

  A 2-sigma resolution of 0.25 kcal/mol needs **~140,000 poses per candidate**.

  So ensemble support is NOT the blocker it was listed as for lead
  optimisation -- no reachable n gets there. It IS worth building if the
  question is coarse: separating a 15 kcal/mol control from the parent needs
  only a handful of poses, and that is the gate the campaign actually failed
  at n=1 (v1's selection-bias artifact). **Build it for gates, not for
  rankings**, and size n from this table rather than from intuition.
* ~~**Tier 4 unvalidated -- M10 recorded it does not fit.**~~ **STALE, and it
  cites a title M10 itself retracted on 2026-09-02.** Checked against the code
  2026-09-19:

  - **Cost**: RESOLVED. 612.4 s (10.2 min), 18 iterations, converged, for the
    71-atom neutral acid at STO-3G/PBE. The ">57 min, did not finish" was
    MEMORY CONTENTION -- a 7.26 GB auto-budget against a ~9.5 GB need, paging
    until the OOM killer fired -- not DFT cost.
  - **The 0-of-5 survivor failure**: both causes FIXED and verified present.
    (a) the driver declared `net_charge=-1` on NEUTRAL structures, which asks
    for an electron that does not exist; now deprotonates the STRUCTURE
    (`Isomer.deprotonated`, `tools/isomers/model.py:44`) with
    `test_deprotonation_conserves_electron_count` pinning it. (b) ring
    contractions that SEVER a ring produced fragment pairs; now rejected at
    enumeration (`enumerate.py:82`) and again at tier 1 (`tiers.py:125`).

  What remains is narrower and worth stating as itself: tier 4 has never
  produced a survivor list end to end SINCE those fixes, so "it works" rests on
  the per-bug tests rather than on a completed funnel run. Dispersion (#99) is
  also still unmerged, so a halogen/CF3 scan would be missing its dominant
  attractive term.
* **No ranking validated end to end.** VERIFIED 2026-09-19 that the repo
  contains no experimental affinity data at all (grepped `experiments/` and
  `testdata/` for IC50/Ki/Kd/pChEMBL: zero hits), so this is blocked on DATA
  ACQUISITION, not on analysis. No further measurement inside this campaign can
  move it, which is why the question "how do we make this pipeline" is answered
  and "is its ranking right" is not.

  What it would take: a congeneric series with measured relative affinities on
  this target, ~10+ compounds spanning >2 kcal/mol. Until then the pipeline's
  output is a HYPOTHESIS GENERATOR, and the measured noise floors above say how
  far to trust it.

## Honest status line

> The pipeline's SHAPE is settled and every stage exists: enumerate relative to
> the parent, dock, prescreen, score, report ddE per (substituent, site). Its
> RANKING is not trustworthy at the resolution a substitution campaign needs,
> and four pose protocols have now been measured to establish that rather than
> assumed. Use it to triage what binds; do not use it to order candidates.
