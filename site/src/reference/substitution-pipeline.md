# Proposing viable substitutions for a drug active site — resolved on danuglipron

Prototyped 2026-09-19 against the real parent (`DANUGLIPRON_SMILES`, PubChem
CID 134611040) and the real receptor (7LCJ, GLP-1R). Every number below was RUN,
not estimated.

## The answer: ~90% exists, and the missing piece is not what I expected

I expected the gap to be a connector between enumeration and pocket scoring.
Its enumeration half now exists: `propose_substitutions` and
`embed_proposals` in `tools/pipeline/substitution.py` turn a parent SMILES and
a site into proposals with 3-D geometries (Å). Placing a proposal in the
pocket still needs docking; see "Order of work" below. But prototyping showed a more basic problem first: **both
cheap gates are useless on this target, for different reasons.**

## What ran

```
S1 enumerate      : 54 analogues   (9 aromatic CH sites x 6 substituents)
S2 pharmacophore  : 54 kept, 0 rejected
S3 liability      :  0 clean, 54 flagged
```

### S2 rejects nothing — and that is CORRECT, not inert

`GLP1R_PHARMACOPHORE.check()` passed all 54. Before concluding the filter
works, I tested whether it CAN reject:

| case | all satisfied | broken features |
|---|---|---|
| PARENT (control) | True | — |
| acid -> methyl ester | False | `acid_or_bioisostere` |
| benzene | False | all four |
| ethanol | False | all four |

So the gate is reachable and discriminating. 0/54 is a true negative:
**substituting an aromatic CH cannot break an acid, a fused diazole, a basic
amine or a nitrile terminus.** The pharmacophore gate belongs on SCAFFOLD moves
(`bioisostere_swaps`, `ring_contractions`), where those features are at risk —
not on a substituent scan.

### S3 rejects everything — because the PARENT already violates it

```
PARENT   MW 555.6   cLogP 4.89   TPSA 113.5   violations: ['MW 556 > 500']
```

Danuglipron is a Phase-2 clinical compound that breaks Lipinski on MW. Every
substitution inherits that violation, so an absolute rule-of-5 gate rejects
54/54 and ranks nothing. The rule of 5 is a hit-finding filter; applied to an
optimized clinical molecule it is a constant, not a discriminator.

## The gate that works: RELATIVE to the parent

Same 54 analogues, scored as the CHANGE each substitution makes:

| subst | dMW | dcLogP | dTPSA | verdict |
|---|---|---|---|---|
| **CN** | +25.0 | **-0.13** | +23.8 | score it |
| OMe | +30.0 | +0.01 | +9.2 | marginal |
| F | +18.0 | +0.14 | 0.0 | deprioritize |
| Me | +14.0 | +0.31 | 0.0 | deprioritize |
| Cl | +34.4 | +0.65 | 0.0 | deprioritize |
| CF3 | +68.0 | +1.02 | 0.0 | deprioritize |

Chemically sensible: nitrile is the only substituent that LOWERS lipophilicity,
and CF3 is the worst offender — against a parent already at cLogP 4.89. That is
a ranked, actionable answer where the absolute gates gave none.

This generalises: **for lead OPTIMIZATION, every cheap descriptor gate must be
relative.** The same logic is why the QM tier must report ddE against the parent
rather than an absolute binding energy — error cancellation and gate
discrimination are the same argument applied at different cost tiers.

## The pipeline

```
S0  Site selection   SMARTS chosen from the POCKET CONTACT MAP, not "every
                     aromatic CH". 9 sites x 6 substituents = 54 is already
                     more than the QM tier can afford; site choice is the real
                     budget control and it is human judgment.
S1  Enumerate        substituent_scan (+ bioisostere_swaps for scaffold moves)   ms
S2  Pharmacophore    GLP1R_PHARMACOPHORE.check -- gate SCAFFOLD moves only       ms
S3  Liability        RELATIVE to parent, never absolute                          ms
S4  Dock             into 7LCJ; harvest the pose (see PR #93)              26.4 s/lig
S5  Prescreen        batch_prescreen -- classical pocket field, no SCF        ms/pose
S6  xtb in field     context["point_charges"]                            0.05-0.152 s
S7  QM/MM ddE        compute_binding_energy on parent AND analogue              minutes
```

## Blockers, in order

1. **`context["geometry"]` is never written** (PR #93 fixes it). Until then S4's
   pose is discarded and S6/S7 score a gas-phase conformer.
2. **No dispersion correction in ferric.** For a halogen or CF3 scan this is the
   dominant attractive term. Three places label tier 4 "DFT + dispersion" and
   it is not computed. Disqualifying for exactly the substituents this scan
   enumerates.
3. **Pose noise swamps the signal.** RESULTS.md MEASURED per-pose sd
   29.07 kcal/mol; substituent effects are 1-2 kcal/mol. One pose per analogue
   reports noise. `funnel.py` keys one row per MOLECULE and cannot express an
   ensemble — decide the pose treatment BEFORE writing the adapter.
4. **No connector** between `tools.isomers` and `tools.active_site` (verified
   by grep, both directions).

## The connector, specified against the REAL signatures (2026-09-19)

PR #96 landed the enumeration + relative-scoring half
(`tools/pipeline/substitution.py`). What remains is carrying a proposal into
the pocket. The interfaces it must bridge, read from source rather than
assumed:

```
tools/pipeline/substitution.py
    propose_substitutions(parent_smiles, substituents, site_smarts,
                          require_smarts) -> list[SubstitutionProposal]
        SubstitutionProposal{ smiles, label, is_parent, d_mw, d_clogp, d_tpsa }

tools/active_site/ligand_embedding.py:91
    embed_ligand_from_coords(symbols, coords_angstrom, pocket=None,
                             basis="def2-svp", charge=0, multiplicity=1,
                             overlap_cutoff_angstrom=1.5) -> EmbeddedLigand

tools/active_site/prescreen.py:128
    batch_prescreen(pocket, ligand_xyz_paths, charge_source,
                    basis="def2-svp", overlap_cutoff_angstrom=1.5)

tools/active_site/binding_energy.py:64
    compute_binding_energy(ligand_xyz, pocket_pdb, basis="def2-svp",
                           method="rhf", xc=None, ff="AMBER",
                           min_available_gb=2.0) -> BindingEnergyResult
```

**`embed_ligand_from_coords` is the seam.** It takes in-memory symbols and
coordinates, carries explicit charge/multiplicity, and does the pocket-overlap
filtering itself -- so the connector needs no temp files and no second
embedding path.

### The connector, and the step still missing

`SubstitutionProposal` carries SMILES; `embed_ligand_from_coords` needs 3-D
coordinates in the pocket's frame. So the connector is:

```
proposal.smiles --(ETKDG, seeded: embed_proposals)--> symbols + coords
                --(docking: dock_ligand)--> pose in the receptor frame
                --> embed_ligand_from_coords(..., pocket=pocket)
                --> batch_prescreen / compute_binding_energy
                --> report ddE against the PARENT proposal
```

The ETKDG hop exists and is tested: `embed_proposals` delegates to
`tools.structure.from_smiles` (seeded ETKDG + MMFF) and returns `(symbols,
coords)` in Angstrom. Those coordinates are centred on the origin, 226 A from
the 7LCJ pocket, so they cannot go into the pocket directly. What is missing is
the placement: docking each proposal (`dock_ligand` returns
`DockedPose.coords_angstrom` in the receptor frame, and
`funnel._harvest_geometry` carries it into `context["geometry"]`) and the
pocket-side tier callables that consume that pose.

### VERIFIED on the REAL GLP-1R pocket (2026-09-19)

The full chain, run against `testdata/molecules/c9_systems/danuglipron/7LCJ_pocket.pdb`:

```
pocket derived : 6458 point charges in 2.7 s (pdb2pqr30 3.7.1)
proposals      : embedded via embed_proposals
ligand         : 15 QM atoms handed to embed_ligand_from_coords
```

SMILES -> 3-D -> real pocket -> QM-ready. Every hop is code that exists today.

**The first version of this check was VACUOUS, and the tell was a suspiciously
round number.** It reported "6458 charges, 0 filtered" -- which looked like a
clean pass but meant the overlap filter had removed nothing. The pocket spans
x = 185.8..284.8 Bohr; my probe ligand sat near the ORIGIN, ~200 Bohr away, so
there was nothing to filter. Re-run with the ligand translated to the pocket
centroid (124.3, 148.2, 116.9 A):

| ligand position | charges kept | filtered |
|---|---|---|
| origin (outside the pocket) | 6458 / 6458 | 0 |
| pocket centroid | 6449 / 6458 | **9** |

Both rows together are the measurement: the filter fires when the ligand is
inside and not when it is outside. Either row alone proves nothing.

Same failure mode as the x=0.0 fixture in the structure readers -- a geometry
that cannot exercise the thing under test produces a green that means nothing.

### The prescreen tier RUNS -- and its first result changes the pipeline design

Full chain executed: propose -> embed -> real 7LCJ pocket -> `prescreen_pose`
with cheap Gasteiger charges (no QM). 13 poses scored. It works.

The ranking it produced is the finding:

| substituent | n sites | mean (kcal/mol) | spread across sites |
|---|---|---|---|
| CN | 3 | -4.77 | 11.81 |
| F | 3 | -4.32 | 10.26 |
| CF3 | 3 | -2.21 | **16.49** |
| Cl | 3 | -1.97 | 9.71 |
| parent | 1 | +5.46 | -- |

```
BETWEEN substituents (full range):  17.30 kcal/mol
WITHIN one substituent (same group, different ring position): 16.49
ratio: 0.95
```

**WHERE you put the group matters as much as WHICH group it is.** The same Cl
lands at ranks 5, 6 and 11 depending on ring position. A pipeline that reports
one number per SUBSTITUENT -- which is what the cheap descriptor gate does, and
what `funnel.py` is shaped for (one row per `iso.canonical`) -- is averaging
over a variable as large as the one it is trying to measure.

This is the same shape as the pose-noise problem below, arriving one tier
earlier and for a different reason: there it is conformational, here it is
positional, and the positional one is NOT fixable by ensembles because the
sites are genuinely different molecules.

**Design consequence:** the unit of the pipeline must be the (substituent,
SITE) pair, not the substituent. `propose_substitutions` already enumerates per
site and `SubstitutionProposal` already carries the product SMILES, so the data
is there -- what must change is any downstream aggregation that collapses to a
per-label mean. The `agg` in the earlier descriptor demo does exactly that and
was fine ONLY because dMW/dcLogP are site-independent by construction. Nothing
downstream of the pocket is.

CAVEAT on scope: benzoic acid at the pocket centroid, Gasteiger charges, no
docking. This measures the METHOD's sensitivity to placement, not a real
affinity ranking -- a docked pose would place each analogue properly rather
than translating a fixed conformer. The 0.95 ratio is the transferable part.

### Two constraints the connector must respect, or it reports noise

1. **ddE, never absolute.** The parent proposal is already in the output for
   exactly this reason. An absolute binding energy carries the full method
   error; the difference cancels most of it. Same argument as the relative
   descriptor gate.
2. **The pose problem is RESOLVED as a design decision, 2026-09-19 (M12).**
   All three routes to averaging the scatter away are now closed with
   numbers: more poses (M5, sd flat in n), relax in field (M6, real 15%,
   ~3 orders short), and real docking (M12, **1%**, 32.5x short). Docked
   poses score at sd 28.75 vs M6's 29.07 while being geometrically MORE
   diverse (pairwise RMSD 3.84 -> 5.81 A), so the scatter is not an
   ensemble-quality problem that better poses fix.

   The sharpest number: Vina's own score on those same 15 geometries has
   sd 0.83, xtb on the identical geometries 28.75. The cheap tier sees a
   nearly flat landscape where xtb sees 103 kcal/mol.

   **The connector must AVERAGE over poses, not select one.** Measured on the
   same 15 poses: Vina's ranking showed NO DETECTABLE relation to the xtb
   score (Spearman -0.261, p=0.35 -- non-significant, i.e. no correlation
   demonstrated, not independence proven), so picking rank 0 behaves as one
   draw from an sd-28.75 distribution. ddE noise is 40.66 selected
   vs 4.07 averaged at n=100 -- selection is 10x WORSE. M9's 0.95 A redock
   licenses "the near-native pose is in the set", not "it is first"; M9
   itself says Vina got it first "partly by luck" (r = +0.461, 4/20 under
   2.0 A).

   So averaging stands as the least-bad estimator, and `funnel.py`'s
   one-row-per-MOLECULE keying IS still a blocker. All four pose
   protocols are now closed; the remaining lever is a scoring metric less
   pose-sensitive than a point-charge interaction energy.

   **The pose treatment is decided (average across poses); the QM tier stays
   blocked until the funnel can carry an ensemble.** MEASURED per-pose sd is
   29.07 kcal/mol (RESULTS.md M5/M6) against substituent effects of
   1-2 kcal/mol, so one pose per analogue reports noise, and `funnel.py` keys
   one row per MOLECULE (`funnel.py:162`), so it cannot express an ensemble.
   **Do not wire the QM tier until ensemble support exists** -- the prescreen
   tier is cheap enough to run per-pose and is the right place to start.

### Order of work

1. Pocket placement: dock each embedded proposal (`embed_proposals` already
   gives the `(symbols, coords)` from its SMILES).
2. Connector to `batch_prescreen` -- CHEAP (classical field, no SCF), so it can
   afford an ensemble and sidesteps constraint 2 entirely.
3. QM tier (`compute_binding_energy`, ddE) only after the funnel can carry a
   pose ensemble (the treatment, averaging, is decided). Dispersion is available: D3(BJ) (#99, merged), without which a
   halogen/CF3 scan would miss its dominant attractive term.

## What to build

`tools/pipeline/substitution.py` — an adapter, not new chemistry: parent SMILES
+ site SMARTS + pocket PDB in, funnel-shaped tier callables out, always
reporting ddE against the parent. It holds the enumeration half today
(`propose_substitutions`, `embed_proposals`); the pocket-side tier callables
are still to be written.

The anchor test is in place (`test_substitution.py`): **an empty substituent
set must return exactly the parent.**
