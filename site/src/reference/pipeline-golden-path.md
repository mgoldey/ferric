# Golden path: input formats -> docking -> xtb -> DFT/QM-MM (2026-09-18)

Status of each claim is marked MEASURED (with source) or ESTIMATED (with
reasoning). Two companion docs: `golden-path-iteration-1.md` (the survey that
scoped this) and `golden-path-qmmm-pipeline.md` (the full QM/MM tier design).

---

## STATUS AS OF 2026-09-19 — what has landed on main since this was written

**C3 (transition-state search) is MERGED (#106) and now DEMONSTRATED under QM/MM
embedding, not merely reachable.** The TS cost model was also corrected from
`2*6N + n_steps` to `2*(6N+1) + (n_steps+1)` (#110) -- a constant +3, so no
conclusion moves, but the tables below are each 3 low.

This document was written against a main where several of its blockers were
live. Ten PRs have merged since. VERIFIED against `origin/main` just now:

| blocker as written | status |
|---|---|
| `context["geometry"]` never written -> docked pose discarded | **FIXED** (#93). `_harvest_geometry` present in funnel.py |
| No structure readers -- xyz only | **FIXED** (#91). `tools/structure` reads PDB/mmCIF/PQR/SDF/mol2/SMILES |
| `normal_modes` not exposed to Python -> C4 uncompletable | **FIXED** (#97). Present in the bindings |
| No substitution enumerator wired to the pocket | **FIXED** (#96). `propose_substitutions` + `embed_proposals` |
| Gradient memory 3.033 GB peak | **FIXED** (#92). 7.4x lower |
| MPI job flakes ~7% | **FIXED** (#98). Launch retried once |
| No dispersion at all | **IN REVIEW** (#99, draft). Native D3(BJ), verified 2e-16 Ha |
| Correlated energies never reach a machine-readable log | **IN REVIEW** (#100, draft) |
| `build-and-test` is a 63-min critical path | **IN REVIEW** (#101). 4-way shard, 2466 tests, 8.3% spread |
| danuglipron flow exists only as scratch scripts | **IN REVIEW** (#102) |

STILL OPEN, and these are the real remaining gaps:

- ~~**No saddle search.**~~ **CLOSED AND MERGED 2026-09-19** (#106):
  `ferric_scf::saddle::find_saddle`, P-RFO with a Bofill update. C0-C5 is
  complete, and it is now WIRED TO AND DEMONSTRATED ON the QM/MM evaluator --
  NH3 umbrella inversion under point-charge embedding converges in 5 steps to
  exactly one imaginary mode (`qmmm_saddle_converges.rs`). The "still to do:
  wire it to the QM/MM evaluator" this line used to carry is done.
  STILL MISSING: there is no IRC, so which two minima a saddle connects remains
  unverified -- the one-imaginary-mode check is necessary, not sufficient.
  Section 4 has the full scope.
- **No analytic dispersion gradients** even once #99 lands. The consequence is
  no longer a silently wrong surface: #99 now REFUSES `task="optimize"` and
  `with_gradient=True` when dispersion is configured, rather than optimising
  the uncorrected surface while reporting corrected energies. Dispersion-
  corrected geometry optimization is unavailable, and says so.
- **QM/MM dispersion.** D3/D4/XDM/VV10 are all QM-atom-pairwise; MM point
  charges carry none. Needs LJ terms in `ferric-mm`.
- **Pose noise.** MEASURED per-pose sd 29.07 kcal/mol against 1-2 kcal/mol
  substituent effects, and `funnel.py` keys one row per MOLECULE so it cannot
  express an ensemble. This gates the QM tier, not the cheap tiers.
- **Site dependence.** MEASURED within/between ratio 0.94-0.95 on two
  independent constructions: WHERE a group goes matters as much as WHICH group.
  The pipeline's unit must be the (substituent, SITE) pair.

## 0. The headline defect: the docked pose is thrown away

VERIFIED by grep, both directions, on 2026-09-18:

```
$ grep -rnE '\["geometry"\]|\.get\("geometry"|"geometry":' tools/ experiments/ --include=*.py | grep -v test
tools/pipeline/tiers.py:84:    cached = context.get("geometry", {}).get(iso.canonical)
```

**One read. Zero writes.** Nothing in the repository ever populates
`context["geometry"]`.

The consequence is concrete. `tier1_dock` DOES return the pose
(`tiers.py:181-185`, payload carries `symbols` and `coords_angstrom`), and
`_embedded` DOES look for a cached geometry (`tiers.py:84`) -- but no code
connects the two. So `_embedded` falls through to `tier2_forcefield`, which
re-embeds from SMILES with ETKDG **in free solution**.

Tier 1 costs ~2 min/ligand (MEASURED, `tiers.py:11`). That entire spend is
discarded, and tiers 3 and 4 then score a gas-phase conformer that has never
seen the pocket. This violates the funnel's own stated premise -- `_embedded`'s
docstring says the cache exists so "tiers 3 and 4 would not be scoring
DIFFERENT geometries of the same candidate".

**This is independent of QM/MM.** It is the single highest-value fix in the
pipeline and should land before any new tier is added, because every
downstream number is currently computed on the wrong geometry.

### VERIFIED end to end, with and without the fix (2026-09-19)

Ran the REAL `run_funnel` with a geometry-producing tier 1 and a tier-4 stand-in
that calls `tiers._embedded` -- the actual accessor tiers 3 and 4 use, not a
proxy for it. The docked pose was set to an unmistakable `[(9.9,9.9,9.9),
(8.8,8.8,8.8)]`.

| funnel | what tier 4 received |
|---|---|
| `origin/main` (no fix) | **15 ETKDG coordinates** -- a freshly embedded, free-solution geometry |
| with PR #93 | exactly `[(9.9,9.9,9.9), (8.8,8.8,8.8)]` -- the docked pose |

The pre-fix arm is the important half: it is not that the pose arrives
slightly perturbed, it is that a DIFFERENT MOLECULE GEOMETRY arrives, re-embedded
from SMILES with a different atom count. Anything computed downstream of that
describes a conformer the pocket never saw.

This is the A/B that makes the defect concrete rather than inferred from a grep,
and it also confirms the fix is not vacuous -- the no-fix arm genuinely fails.

### Where the fix goes (not inside a tier)

`_run_stage` (`funnel.py:111-115`) dispatches through a `ProcessPoolExecutor`,
so a tier mutating `context` in a worker mutates a COPY -- the change never
returns. Any fix that writes `context["geometry"]` from inside `tier1_dock`
will silently do nothing under the parallel path while appearing to work
serially.

The results DO return to the driver (`funnel.py:158`), so the harvest belongs
in `run_funnel`'s stage loop, between stages:

```python
results = _run_stage(stage, population, context)
# harvest any geometry a tier produced, so later tiers reuse it
for r in results:
    if r.ok and r.payload and "coords" in r.payload:
        context.setdefault("geometry", {})[r.candidate_id] = {
            "symbols": r.payload["symbols"], "coords": r.payload["coords"],
        }
```

Anchor test before changing anything: assert that with tier 1 in the stack,
tier 3 receives the DOCKED coordinates, not ETKDG's. The trivial-limit anchor
is a funnel with no docking stage, where behaviour must be byte-identical to
today.

---

## 0b. QUICKSTART -- the pipeline in code you can paste

Everything below this section describes the pipeline and costs it. This one
RUNS it. Every snippet was executed on 2026-09-19 and its real output is shown;
none is illustrative.

```python
# A. any input format -> a ferric Molecule (Angstrom, seeded ETKDG for SMILES)
from tools.structure import from_smiles, read_structure
mol = from_smiles("CC(=O)Oc1ccccc1C(=O)O", seed=0xF00D)   # aspirin: 21 atoms
# read_structure("x.pdb" | "x.sdf" | "x.mol2" | "x.xyz")  -- same entry point
#   NOTE a crystal PDB is REJECTED ("no hydrogens"). A receptor goes through
#   derive_pocket_charges (which protonates), not read_structure.

# B. enumerate analogues at every matching site
from tools.pipeline.substitution import propose_substitutions, relative_descriptors
props = propose_substitutions("c1ccccc1C(=O)O", {"F": "F", "Cl": "Cl", "Me": "C"})
#   -> 10 proposals, and props[0] is the PARENT. That is the anchor, not a bug:
#      the parent rides through every stage so scores can be reported as ddE.

# C. gate RELATIVE to the parent, never on absolutes
relative_descriptors(props[0].smiles, "c1ccccc1C(=O)O")
#   -> (-1.42e-14, 0.0, 0.0)   the parent against itself -- NOT exactly zero,
#      and this doc used to print "(-0.0, 0.0, 0.0) ... exactly zero, by
#      construction". The distinction is real and worth knowing:
#        relative_descriptors(s, s)                     -> exactly (0.0,0.0,0.0)
#        relative_descriptors(props[0].smiles, s)       -> -1.42e-14 in dMW
#      because props[0].smiles is the CANONICAL form ("O=C(O)c1ccccc1") of the
#      input spelling ("c1ccccc1C(=O)O"). Same molecule, different atom order,
#      so the MW float sum lands one ulp apart. Compare ddE against a
#      tolerance, never `== 0.0`, whenever either side has been round-tripped
#      through a canonicaliser.

# D. liability flags (published alert sets -- NOT a probability of harm)
from tools.tox.alerts import RdkitAlertsProvider
provider = RdkitAlertsProvider()          # build ONCE: 47 ms vs 9.4 ms/molecule
eps = provider.fetch("CC(=O)Oc1ccccc1C(=O)O")
#   -> 13 endpoints, e.g. alert_brenk = 0.333
```

For the pocket half (docking, xtb, DFT) the entry points are
`tools.docking.vina_dock.dock_ligand`, `tools.campaign.fit.pose_fit` and
`tools.active_site.binding_energy.compute_binding_energy`; the composition is
`tools.pipeline.run_funnel`, and
`tools/pipeline/tests/test_golden_path_smoke.py` runs two real tiers through it
end to end in about 7 seconds.

**Before you rank anything with the output, read "WHICH METHOD FOR WHICH
QUESTION" below.** The pipeline will happily produce a ddE ordering that the
measurements do not support -- five pose protocols have been tried and the best
available noise is 4.07 kcal/mol against substituent effects of 1-2.

---

## 1. Input formats: what can enter

Rust `Molecule` has only `load_xyz` / `load_xyz_with_charge` / `parse_xyz`
(`crates/ferric-core/src/mol.rs`). Everything else is Python-side.

| format | reader | layer | notes |
|---|---|---|---|
| xyz | `Molecule::load_xyz`, `parse_xyz` | Rust | the only native format |
| xyz (multi-frame) | `ConformerEnsemble.from_multi_xyz` | Rust | `from_xyz` reads ONLY frame 1 |
| PDB / mmCIF | `tools/structure` (gemmi) | Python | gemmi is a CORE dep |
| PQR | `tools/active_site/pqr_parser`, `tools/structure` | Python | charges are MM, not a QM charge state |
| SDF / mol / mol2 | `tools/structure` (rdkit) | Python | `ferric[docking]` extra |
| SMILES | `tools/structure.from_smiles` (rdkit ETKDG) | Python | geometry is tier-2 grade, NOT optimized |
| AMBER prmtop | via OpenMM only | Python | no direct reader |
| OpenMM | `active_site/mm_topology.topology_from_openmm` | Python | |

`tools/structure` (added 2026-09-18, PR #91) funnels every format through
`Molecule.from_xyz_string`, so a PDB and an XYZ of the same geometry give
bit-identical Bohr coordinates rather than drifting through two parsers.

### What each entry path COSTS (MEASURED 2026-09-19)

The format table above says what can enter; these say what it costs. Min of
1-3 reps, single-threaded, on this box:

| entry path | cost | notes |
|---|---|---|
| xyz -> `Molecule` (71 atoms) | **0.3 ms** | the native path; free |
| SMILES -> 3-D (`from_smiles`, ETKDG+MMFF) | **8.5 ms** | per molecule |
| PDB -> `PocketCharges` (`derive_pocket_charges`, 7LCJ pocket) | **2.66 s** | 6458 charges, pdb2pqr30 |

Four orders of magnitude separate them, and the ordering is the point: **the
PDB path is ~300x the SMILES path and ~9000x an xyz read.** It is also a
ONE-OFF per target -- `PocketCharges` is derived once and reused across the
whole ensemble (that is the reason the type exists), so 2.66 s amortises to
nothing over a 1000-analogue campaign and is a real cost for a one-molecule
run.

Note `from_smiles` at 8.5 ms here vs 214 ms/proposal for `embed_proposals` in
the cheap-stage table: same ETKDG machinery, ~25x apart, because a drug-sized
analogue is far harder to embed than the small test molecule timed here. Quote
the 214 ms for campaign planning.

**A refusal worth knowing about before you hit it.** `read_structure` REJECTS a
crystal PDB with `StructureError: ... no hydrogens`. That is correct -- a PDB
from the PDB has no hydrogens, and silently treating it as a QM molecule would
hand the solver a species that does not exist. A receptor goes through
`derive_pocket_charges` (which runs pdb2pqr and protonates), not through
`read_structure`. The error names the problem, but the two paths are easy to
confuse on first use.

**Unit hazard, worth stating once:** Python geometry entry is Angstrom;
`point_charges` and `QmmmSystem.point_charges()` are BOHR; `PocketCharges`
holds Bohr. Mixing them is a silent 1.89x error, not a crash.

**Charge and multiplicity are never inferred.** No common structure format
records multiplicity (2S+1) at all. ferric's parity check catches an
odd-electron species left at the default singlet, but CANNOT catch an
even-electron triplet (O2, many carbenes) -- that converges cleanly to a state
that does not exist.

---

## 1b. Dependency status, CHECKED rather than assumed (2026-09-19)

Every stage below names a tool. Whether those tools are actually present is a
separate question from whether the code is written, and it is the one that
decides if a stage runs today.

| dependency | status | used by |
|---|---|---|
| `pdb2pqr30` 3.7.1 | on PATH | `active_site/pdb2pqr_runner` (PDB -> pocket charges) |
| `openmm` | importable | `active_site/mm_topology` |
| `vina` | importable | `tools/docking` tier 1 |
| `meeko` | importable | ligand prep for Vina |
| `rdkit` | importable | SMILES/SDF/mol2, ETKDG, `tools/isomers` |
| `gemmi` | importable (core dep) | PDB/mmCIF via `tools/structure` |
| `xtb` | see [[xtb-rollup]] | tier 3 |

ONE FALSE ALARM WORTH RECORDING, because the same check will mislead the next
person: `import pdb2pqr` FAILS (ModuleNotFoundError) while `pdb2pqr30` is on
PATH. That is not a broken install -- `pdb2pqr_runner.py` is a deliberate
SUBPROCESS wrapper around the CLI ("Subprocess wrapper around the PDB2PQR CLI",
line 1), so the module never needs to import. Checking importability would have
reported a working stage as broken.

So the docking branch's tooling is complete. The blockers listed elsewhere in
this document are about PHYSICS and DATA FLOW (missing dispersion, pose noise),
not about missing software.

## 2. Cost estimates

### WHICH METHOD FOR WHICH QUESTION (the table this note was missing)

Everything below this heading costs methods. This one says which to REACH FOR,
and -- more usefully -- what resolution each can actually deliver on this
campaign. A method that is cheap and cannot answer your question is not a
bargain.

| the question | method | cost | resolution it delivers | verdict |
|---|---|---|---|---|
| where does this ligand sit? | Vina dock | ~2 min/ligand @ ex=32, ~30 s @ ex=4 | **0.95 A** redock (M9), 20/20 poses on-site | **use it** |
| which pose is best? | Vina score | free (comes with the dock) | r(score, RMSD) = +0.461; only 4/20 under 2.0 A | **do not trust** -- generates, cannot rank |
| is this geometry sane? | MMFF94 | 2-22 ms/pose (9-34 atoms), ~73 ms @ 71 | adequate to declash | **use it**, for declashing only |
| how strained is this conformer? | GFN2-xTB | ~0.5 s/pose | 143 kcal/mol anion/neutral split resolved | **use it** for coarse separation |
| which of these conformers is lowest? | GFN2-xTB | ~0.5 s/pose | **Spearman 0.011 vs DFT** over a 3 kcal/mol span (M16, n=20) | **do not trust** -- a gate, not a ranker |
| which analogue binds better by 1-2 kcal/mol? | any of the above + ddE | -- | ddE noise **4.07 kcal/mol** at best (M4-M13) | **NO METHOD QUALIFIES** |
| what is the SCF energy here? | ferric RHF / KS-DFT | 96 s @ 32 atoms, 612 s @ 71 | 1e-8 Ha vs PySCF (RHF), 2e-8 (PBE/B3LYP) | **use it** |
| does the pocket field change it? | + `external_potential` | **~1.0x** the gas-phase SP (MEASURED) | embedding is essentially free | **use it** -- no reason not to |
| where is the transition state? | `saddle::find_saddle` | 2*(6N+1) + (n_steps+1) gradients | converges on a known saddle; refuses a minimum's basin | **use it** |
| is this really a TS? | `harmonic_frequencies` | 6N+1 gradients | exactly-one-imaginary check, from Rust AND Python | **use it** |
| which two minima does it connect? | -- | -- | no IRC in ferric | **not available** |
| is this molecule a liability? | `tools/tox` alerts | 9.4 ms/molecule | published alert sets, NOT a probability of harm | **use it as a FLAG** |

**The row that matters most is the one with no method.** Four pose protocols
have been measured (RESULTS.md M4-M13) and the best available ddE noise is
4.07 kcal/mol against substituent effects of 1-2. Every OTHER row in this table
is a green light; that one is not, and no amount of tier-4 DFT fixes it,
because the error is in the pose ensemble and not the electronic structure.

**Two rows are worth reading together.** "Where does this ligand sit?" is a
green light and "which pose is best?" is a red one, from the SAME tool. Docking
generates the right answer among its candidates and cannot pick it out. That is
not a defect to fix -- it is the empirical reason tiers 2-4 exist.

### MEASURED (quoted with source)

| tier | method | cost | source |
|---|---|---|---|
| 1 | Vina, **exhaustiveness 4** (the RECOMMENDED setting) | **26.4 s/ligand** @ cpu=0 (12 cores); 109.0 s @ cpu=1 | RESULTS.md M11 |
| 1 | ~~Vina, exhaustiveness 32, ~2 min/ligand~~ | superseded by the ex=4 row above; the figure was never measured and its `tiers.py:11` citation pointed at the module doc comment | |
| 1 | Vina, per pose | ~10 us | `tools/docking/vina_dock.py:7` |

**Budget from the ex=4 row, not the ex=32 one.** M11 measured that across an
8x range of exhaustiveness the mean redock RMSD moved 0.097 A -- SMALLER than
the 0.131 A between-seed SEM -- and ex=32 had the WORST mean of the four levels
tried. So ex=32 costs 6.8x for no accuracy, and the ~2 min row is kept only
because `tiers.py:11` still documents it.

Note the parallel efficiency while you are here: giving up 11 of 12 cores costs
4.1x, so Vina's internal parallelism runs at **34%**. Fan-out across ligands
only wins above ~4 workers, which is a narrower claim than "the machine sits
idle at low exhaustiveness".
| 2 | MMFF94 (`tier2_forcefield` = embed + optimize) | **2.2 ms @ 9 atoms, 8.2 @ 19, 21.6 @ 34; ~73 ms projected @ 71** | MEASURED 2026-09-19 |
| 2 | ~~MMFF94 ~1 ms/pose~~ | superseded: that figure cited `tiers.py:12`, which is the same doc comment -- a circular citation, never a measurement, and it described a single point rather than embed+optimize | |
| 3 | GFN2-xTB via `tier3_gfn2` | **0.05-0.15 s** @ 9-19 atoms | MEASURED 2026-09-19 |
| 4 | ferric DFT via `tier4_dft` | **0.66 s @ 9, 8.7 s @ 19** (STO-3G); 96.1 s @ 32 (**def2-SVP**, ~450 bf) | MEASURED 2026-09-19 / RESULTS.md |
| 4 | ferric DFT | 612.4 s @ 71 atoms, **STO-3G**/PBE (~234 bf), 18 iters, converged | RESULTS.md |

**DO NOT DERIVE A SCALING LAW FROM THOSE TWO ROWS.** They differ in BASIS as
well as size, and in the unhelpful direction: the BIGGER system used the
SMALLER basis. Fitting them gives p = 2.32, which UNDERSTATES pure N-scaling
because part of the size increase was paid for by a cheaper basis.

**Measured directly, and the confound turns out to be small.** A DFT N-sweep at
FIXED basis (PBE/STO-3G, alkanes C2-C4, 2026-09-19) gives a tail exponent of
**2.32** -- exactly what the confounded pair gives. Two independent routes to
the same number:

| route | exponent | fixed basis? |
|---|---|---|
| the two rows above | 2.32 | NO (def2-SVP vs STO-3G) |
| PBE/STO-3G N-sweep (this) | **2.32** | yes |
| RHF/STO-3G N-sweep | 2.58 | yes |

So p ~ 2.3-2.6 is the honest band, and the basis confound moves the answer less
than the RHF-vs-DFT difference does. **Still quote the fixed-basis numbers**,
because the agreement is what makes the confounded pair usable rather than the
other way round -- and note all three are 2-3 point tail fits, which is
indicative, not decisive.

One measurement artifact worth naming: C1 came in at 5.30 s against C2's 0.37 s
-- a first-call warm-up (basis parse, grid construction), not chemistry.
Tail-fitting excludes it automatically, which is the reason the repo's protocol
says to fit the tail rather than the whole series.

### ESTIMATED (reasoning stated, do not quote as measured)

No QM/MM wall-time measurement exists anywhere in the repo -- the 9 Rust and
29 Python qmmm tests are PySCF CORRECTNESS tests on H2/ethane/water with no
timings. So:

- ~~**Tier 3.5 (in a pocket field): ~2x a gas-phase single point.**~~
  **RETRACTED 2026-09-18 -- MEASURED at ~1.0x, so the estimate was wrong by
  about 2x.** See the measurement immediately below. I had reasoned "MM charges
  add a one-body term" and then guessed 2x anyway; the same reasoning actually
  PREDICTS a small overhead, because a one-body term is cheap. The guess did
  not follow from the argument I gave for it.
- **Tier 5 (QM/MM DFT optimization): ~5.8 h**, was "1.5-17 h".
  The 10x width came entirely from an unmeasured BFGS step count. MEASURED
  2026-09-18: ethane with a link atom across the C-C cut (the catalyst shape)
  converges in **34 steps** -- and in 34 at BOTH STO-3G and 6-31G, so the step
  count is set by the GEOMETRY, not the basis, as BFGS theory predicts.
  34 x the 612 s single point = ~5.8 h. Still ESTIMATED, because the
  MULTIPLICAND is a 71-atom DFT single point measured on a different system;
  only the multiplier is now measured.
  CAVEAT, and it is the load-bearing one: 34 steps is a near-rigid CH3 relaxing
  a few bond lengths. A floppy ligand in a pocket has far more soft degrees of
  freedom and will take more. Treat 34 as a FLOOR for the step count, not a
  typical value -- one geometry is not a distribution.

### Every `tiers.py:NN` citation in this table pointed at a DOC COMMENT

Three of the cost rows cited `tiers.py:11/13/14` or `vina_dock.py:7`. Every one
of those lines is the module header's own cost table -- **the citations pointed
at prose, not at a measurement**, and the golden path and the docstring were
quoting each other.

A `file.py:NN` reference survives the check "is this sourced?", which is what
let three unmeasured figures sit in a MEASURED table. They also rot: correcting
the tier-2 line shifted the numbering, so `tiers.py:13` and `:14` came to point
at the wrong rows entirely.

All three are now measured THROUGH THE TIER FUNCTIONS (not through the
underlying library, which is a different cost):

| tier | 9 atoms | 19 atoms | 34 atoms |
|---|---|---|---|
| 2 `tier2_forcefield` | 2.2 ms | 8.2 ms | 21.6 ms |
| 3 `tier3_gfn2` | 0.152 s | 0.050 s | -- |
| 4 `tier4_dft` (STO-3G) | 0.66 s | 8.7 s | -- |

Tier 3's old `~0.5 s` was the right order. Tier 2's `~1 ms` was 20x low. Both
were unmeasured; the difference is luck, not diligence.

**When a cost in this table cites a file and line, open it.** If the line is a
doc comment, the number is unsourced.

### Tier 2 is 20x costlier than claimed, and it changes nothing (MEASURED 2026-09-19)

The `~1 ms/pose` figure cited `tiers.py:12` -- which is the same doc comment.
A circular citation, never a measurement, and it described a single MMFF point
while `tier2_forcefield` does embed + optimize.

| molecule | atoms | tier 2 |
|---|---:|---:|
| ethanol | 9 | 2.2 ms |
| acetanilide | 19 | 8.2 ms |
| drug-like | 34 | 21.6 ms |

Tail exponent 1.66, projecting to **~73 ms at danuglipron's 71 atoms** -- about
20x the claimed figure.

**And the conclusion is unchanged.** Tier 2 runs on the ~10% that survive
docking, so at N=1000 it is **0.4%** of the campaign against docking's 78%.
Recorded because the next person to notice the discrepancy should not have to
re-measure it to find out it does not matter: a wrong number that changes no
decision is still worth fixing once, and worth marking as decision-neutral so
it is not fixed twice.

### The xtb -> DFT ratio is ~200x at drug scale, not 1000x (MEASURED 2026-09-19)

"DFT costs ~1000x" appeared twice as a STOP-HERE decision criterion. Measured
on the SAME molecules (the only comparison that means anything -- the table's
0.5 s and 96-612 s rows came from different systems), GFN2-xTB relax vs
PBE/STO-3G single point:

| molecule | N | xtb | DFT | ratio |
|---|---:|---:|---:|---:|
| ethane | 8 | 0.024 s | 0.397 s | **16x** |
| butane | 14 | 0.044 s | 1.374 s | **31x** |
| hexane | 20 | 0.077 s | 3.619 s | **47x** |

The ratio GROWS with size, as `N^1.17` on the last two points, because DFT
scales ~N^2.3 while xtb is much flatter. Projecting:

| N | projected ratio |
|---|---|
| 32 | 81x |
| 71 (danuglipron) | **206x** |
| 100 | 307x |
| ~274 | 1000x |

So the ~1000x figure is right only above ~270 atoms -- roughly 4x larger than
anything this pipeline runs. At danuglipron's 71 atoms it overstates by ~5x.

**The decision does not change**: 200x is still a decisive reason to stop at
tier 3 for a coarse sort. What changes is that the number is now measured, and
anyone budgeting a campaign from it is out by 5x rather than in the right
place by luck.

### The CHEAP stages, measured at last (2026-09-19)

The cost table covered the quantum tiers and said nothing about the four stages
that run before them -- enumeration, descriptors, embedding, toxicology -- even
though those are what a substitution campaign spends its first hour on. All
MEASURED on danuglipron (73 proposals from 8 substituent groups), min of 3-5
reps, single-threaded:

| stage | cost | notes |
|---|---|---|
| enumerate (`propose_substitutions`) | **2.8 ms** / proposal | 207 ms for all 73 |
| relative descriptors | **1.7 ms** / proposal | the P2 gate |
| embed (`embed_proposals`, ETKDG+MMFF) | **214 ms** / proposal | ~75x the other two combined |
| toxicology alerts (`RdkitAlertsProvider.fetch`) | **9.4 ms** / molecule | 13 endpoints |
| tox catalog construction | **47 ms**, ONE-OFF | build the provider once |

**What this changes about where to worry.** For 1000 analogues the whole cheap
half is `1000 x (2.8 + 1.7 + 214 + 9.4) ms` = **~3.8 minutes**, of which
embedding is 94%. Everything upstream of docking is free at campaign scale; the
only cheap-tier stage worth optimizing is the ETKDG embed, and only if the
campaign is much larger than 1000.

**A docstring correction.** `RdkitAlertsProvider`'s own comment says
per-molecule construction "dominates the runtime of a batch". MEASURED the
ratio is **5x** (47 ms build vs 9.4 ms/molecule), so constructing per molecule
would cost 6x a batch, not orders of magnitude. Building it once is still
right; the stated reason overstates the effect.

### End to end: DOCKING dominates a substitution campaign, not DFT (2026-09-19)

Composing the measured per-stage numbers over a 10x-per-tier funnel
(all -> 10% docked -> 10% xtb -> 1% DFT), single-threaded:

| N analogues | cheap half | dock | xtb | DFT | total |
|---|---|---|---|---|---|
| 100 | 0.4 min | 0.6 h | 0.0 h | 0.2 h | **0.8 h** |
| 1000 | 3.8 min | 5.6 h | 0.3 h | 1.7 h | **7.6 h** |

Shares are scale-invariant at this funnel ratio. Recomputed 2026-09-19 with
M11's MEASURED 26.4 s/ligand for tier 1 rather than the ~20 s estimate used
first:

| | cheap | dock | xtb | DFT | total (N=1000) |
|---|---|---|---|---|---|
| with the ~20 s estimate | 0.8% | 73% | 4% | 22% | 7.6 h |
| with MEASURED 26.4 s | **0.7%** | **79%** | **3%** | **18%** | **9.3 h** |

The correction STRENGTHENS the conclusion rather than softening it: docking is
79% of the campaign, DFT 18%. Quote the measured row.

**That inverts the intuition this pipeline was designed around.** DFT is the
most expensive thing PER CALL by five orders of magnitude (6e+2 s vs 1e-5 s),
and it is still only **18%** of the campaign, because the funnel has already cut
the population 100x by the time it runs. Docking is **79%** -- it is cheap per
pose and runs on EVERYTHING, 20 poses each.

This is the same conclusion M11 reached from the other direction ("the funnel
spent 2.6x more than it needed to", and the fix was tier-1 effort and fan-out,
not tier 4). Two independent routes to "tier 1 is the budget" is worth more
than either alone.

**Practical consequence.** The lever is `exhaustiveness` and worker fan-out at
tier 1, both already measured (M11: ex=4 matches ex=32's accuracy at a quarter
the cost; 10 workers x cpu=1 gives 6.2x). Optimizing the DFT tier -- the
instinctive target -- can win at most **18%**.

ESTIMATED, with the inputs labelled: the per-stage costs are MEASURED (above,
and the hierarchy table), the funnel RATIOS are a design choice, and the DFT
600 s is a 71-atom single point from a different system. Change the ratios and
the shares move; the ordering is robust to anything reasonable.

### Transition-state search costs 2 Hessians + n_steps (MEASURED, 2026-09-19)

New entry: until `ferric_scf::saddle` landed there was no saddle search to
cost. `crates/ferric-scf/tests/saddle_cost.rs` counts the actual calls rather
than timing them, because a call count is a property of the algorithm while a
wall time is a property of this box.

    total = n_hessian * (6N + 1)  +  (n_steps + 1)   (gradient evaluations)

**CORRECTED 2026-09-19 (was `2*6N + n_steps`).** `harmonic_frequencies` takes
one energy-and-gradient at the UNDISPLACED geometry before its displacement
loop, and `n_gradient_evaluations` is zeroed AFTER that call -- so the reported
count is 6N while the true cost is 6N+1. `find_saddle` likewise takes one
gradient before its first step. A constant +3 for a default search, so every
conclusion below is unchanged and the tables are each 3 low.

Worth recording WHY it survived review, because the failure is reusable: the
cost test supplies its own ANALYTIC Hessian closure and therefore never calls
`harmonic_frequencies` at all. The documented model described the
finite-difference path while every assertion measured a synthetic one. The test
looked like it pinned the cost model and pinned something else. Fixed by
`a_finite_difference_hessian_costs_6n_plus_one_gradients`, which runs the real
FD Hessian.

MEASURED on an analytic surface whose saddle is known in closed form
(`hessian_recalc_every = 0`, the default):

| quantity | value | why |
|---|---|---|
| Hessians per search | **2** | one at the start (also the "is there anything to climb?" check), one at the end for the character check. None in between -- Bofill carries it. |
| gradients per step | **1** | |
| one Hessian | **6N** | central difference of the analytic gradient; H2 = 12, water = 18 |

So a search on **N = 20** atoms costs **242 + n_steps + 1** gradient evaluations,
and **the two Hessians dominate until n_steps exceeds ~240**. That is the whole
reason `hessian_recalc_every` defaults to 0; MEASURED, setting it to 1 doubles
the Hessian work (2 -> 4 on this surface, i.e. 240 -> 480 gradient-equivalents
at N=20).

**The multiplicand, measured at three sizes (2026-09-19).** The TS cost model
`2*(6N+1) + (n_steps+1)` had a measured MULTIPLIER and an unmeasured MULTIPLICAND -- one
gradient, taken from a 71-atom DFT single point on a different system. Measured
directly on linear alkanes, RHF/STO-3G, single-threaded:

| N atoms | one single point |
|---|---|
| 5 (methane) | 22 ms |
| 8 (ethane) | 29 ms |
| 11 (propane) | 66 ms |

Last-two exponent **p = 2.58**. THREE POINTS IS NOT A SCALING MEASUREMENT --
quote it as indicative, not as an exponent, and note the tail was fitted rather
than the whole series (a global fit averages in the flat N=5->8 start).

Projecting `t(N) = t(11)*(N/11)^2.58`:

| QM region | steps | gradients | projected (STO-3G) |
|---|---|---|---|
| N = 11 | 30 | 162 | 0.2 min |
| N = 20 | 30 | 270 | **1.4 min** |
| N = 20 | 100 | 340 | 1.8 min |
| N = 40 | 30 | 510 | **15.7 min** |

**The step count barely matters and the BASIS dominates.** Going 30 -> 100
steps at N=20 moves the total 26%; going N=20 -> 40 moves it 11x. And every row
above is STO-3G, the cheapest basis there is -- a real catalyst at def2-SVP or
better is orders above these. Treat the table as the N-SCALING SHAPE, not as
wall times.

That is the practical guidance the cost model was missing: **size the QM region
first** (golden path C0 already says it "sets the cost" -- this is by how much),
and do not spend effort shaving P-RFO steps.

**Putting a number on a catalyst TS.** Using the same 612 s DFT single point
the tier-5 estimate uses, and treating a gradient as ~1 single point:

    N = 20 QM atoms, n_steps = 30    ->  270 gradients  ->  ~46 h
    N = 20 QM atoms, n_steps = 100   ->  340 gradients  ->  ~58 h

ESTIMATED, and the multiplicand is the load-bearing weakness -- it is a 71-atom
DFT single point measured on a different system. What is MEASURED is the
multiplier (2 Hessians + n_steps) and the 6N Hessian cost. Note the step count
matters much less than it does for a minimization: going from 30 to 100 steps
moves the total by 26%, because the fixed 240-gradient Hessian cost swamps it.

**The floor caveat, same as the BFGS one below.** The step count above comes
from an analytic two-atom surface. A real catalyst TS has soft degrees of
freedom it does not. Treat any step count from this test as a FLOOR.

### An EMBEDDED saddle search, demonstrated end to end (2026-09-19)

`examples/qmmm_saddle.rs` originally showed only a REFUSAL -- H2 in an MM field
has no saddle, so `find_saddle` declines. That is worth showing, but it does not
demonstrate the workflow WORKS: code that rejected everything would print the
same thing. `crates/ferric-scf/tests/qmmm_saddle_converges.rs` now pins the
positive half.

NH3 umbrella inversion under point-charge embedding: **converges in 5 steps,
exactly 1 imaginary mode, `is_transition_state() = true`, z spread 0.0075 Bohr**
(i.e. planar, which is the physically right answer).

Two findings came out of building it, both of which cost real time:

**1. An MM field that BREAKS THE SYMMETRY DEFINING THE SADDLE does not make the
search harder -- it removes the target.**

| field | max\|g_z\| at the planar geometry | outcome |
|---|---|---|
| gas phase | ~1e-16 | converges, 5 steps, 1 imaginary |
| symmetric (-0.2, -0.2) | ~1e-17 | converges, 5 steps, 1 imaginary; E shifted 5.0e-4 Ha |
| **antisymmetric (-0.2, +0.2)** | **3.8e-3** | runs to max_steps, `converged = false` |

An antisymmetric pair puts a CONSTANT force along z, so planar NH3 stops being
a stationary point at all. The search is right not to converge -- there is
nothing there -- but it reads exactly like a solver bug. The tell that it is
not step starvation: raising `max_steps` 60 -> 200 moved the energy by 2e-8.

**Before debugging an embedded saddle search that will not converge, evaluate
the gradient AT the symmetric geometry under the field.** If it does not
vanish, the field removed the saddle.

**2. Relax every coordinate EXCEPT the one under study first.** A textbook
1.01 A N-H left a bond-stretch gradient of 5.2e-3 that the convergence test
(g_max 3e-4) rightly refuses; the STO-3G planar optimum is 1.006 A (g_max
6.1e-4). The search looked broken in a coordinate with nothing to do with the
umbrella.

**TS candidates REJECTED**, recorded so they are not retried: linear H3+ and
linear H2O are both SECOND-order saddles (2 imaginary, -1068 and -2328.7 cm^-1,
each doubly degenerate). The bend of a linear molecule comes in a perpendicular
pair, so "the linear form of a bent molecule" is almost never a transition
state. NH3 is the smallest unambiguous closed-shell TS that is not already the
starting geometry.

The negative half of that test is MUTATION-VERIFIED and is what makes the pair
meaningful: setting `external_potential: None` passes the positive case --
the gas-phase saddle is right there -- and FAILS the antisymmetric one. Without
it, a `find_saddle` that ignored the embedding entirely would look correct.

### Point-charge embedding is essentially FREE (MEASURED, 2026-09-18)

Vacuum vs point-charge-embedded RHF on the SAME molecule (water), 8 MM charges
of +/-0.5 e on a 6-Bohr shell, `OPENBLAS_NUM_THREADS=1`, min of 9 reps on an
idle box (load ~1.0):

| basis | vacuum | embedded | ratio | iterations |
|---|---|---|---|---|
| STO-3G | 10.6 ms | 11.2 ms | 1.06x | 8 -> 9 |
| cc-pVDZ | 162.0 ms | 156.1 ms | 0.96x | 11 -> 11 |
| def2-SVP | 35.1 ms | 32.9 ms | 0.94x | 11 -> 11 |

**0.94-1.06x across three bases -- indistinguishable from free.** The iteration
count moved once (8 -> 9 at STO-3G) and not at all in the other two, so the
"weak assumption" the retracted estimate worried about does not bite here.

The physics is unsurprising in hindsight: `hcore_with_external` folds the
point-charge term into hcore ONCE before the SCF loop (see CLAUDE.md's
external_potential notes), so embedding adds a fixed one-body cost and nothing
per-iteration. Sub-1.0 ratios are timing noise, not speedups -- at 3 reps the
STO-3G row read 0.74x, which is what prompted going to 9.

**Consequence for planning:** the MM environment is not what costs you. The QM
region is, and specifically its GRADIENT peak -- see the next section. Do not
shrink a QM region to "afford the embedding"; there is nothing to afford.

CAVEAT on scope, now RESOLVED by the sweep below: 8 charges on water measures
the MECHANISM, not a protein-scale run.

### ...and it stays linear out to pocket scale (MEASURED, 2026-09-18)

Same QM region (water/cc-pVDZ), sweeping the MM charge count on an 8-Bohr
Fibonacci shell. Min of 5 reps, `OPENBLAS_NUM_THREADS=1`, idle box.

| charges | wall | ratio | SCF iters | ms per charge |
|---|---|---|---|---|
| 0 (vacuum) | 95.8 ms | 1.00x | 11 | -- |
| 8 | 91.1 ms | 0.95x | 11 | (noise) |
| 64 | 96.6 ms | 1.01x | 11 | 0.012 |
| 256 | 115.4 ms | 1.20x | 11 | 0.077 |
| 1024 | 176.0 ms | 1.84x | 11 | 0.078 |
| 4096 | 403.7 ms | 4.21x | 11 | 0.075 |

**Per-charge cost converges to 0.075-0.078 ms and stays there across three
decades.** A log-log fit over the last three points gives slope **0.993** --
linear to within 0.7%, which is what a one-body hcore term must be.

**The SCF iteration count is 11 at EVERY charge count, vacuum included.** The
embedding field does not make the SCF harder to converge; that was the "weak
assumption" behind the retracted 2x estimate, and it simply does not occur.

Extrapolating at 0.078 ms/charge (the 1000-charge row predicts 174 ms against a
measured 176 ms, so the fit holds):

| pocket size | predicted | vs vacuum |
|---|---|---|
| 1,000 charges | 174 ms | 1.8x |
| 10,000 charges | 876 ms | 9.1x |

So embedding is free at a few hundred charges and becomes a real but modest
cost at whole-protein scale -- and it is LINEAR, so it never overtakes the
QM region's own N^3-N^4 scaling. Choose the QM region on its gradient peak
(next section); choose the MM cutoff on physics, not on cost.

### The DFT GRADIENT peak, not the SCF, is what sizes a QM/MM job (MEASURED)

New measurement, 2026-09-18, serial on an idle box
(`gradient_pool_peak_measurement.rs`, 4 passed):

| system | basis | nbf | SCF peak | gradient peak | ratio |
|---|---|---|---|---|---|
| water | cc-pVDZ | 24 | 0.000 GB | 0.103 GB | -- |
| benzene | 6-31G | 66 | 0.012 GB | 1.099 GB | 92x |
| benzene | cc-pVDZ | 114 | 0.035 GB | 1.860 GB | 53x |
| alkane_10 | 6-31G | 134 | 0.162 GB | 5.930 GB | 37x |
| alkane_16 | 6-31G | 212 | 0.503 GB | 13.879 GB | **27x** |

Two things follow for anyone sizing a QM/MM job.

**1. Budget for the GRADIENT, not the SCF.** The gradient peak is 27-92x the
SCF peak that precedes it. A QM region whose SCF fits comfortably can still
fail at the gradient step, which is the step every optimization and every one
of the 6N+1 frequency displacements needs. Sizing a QM region from a
single-point SCF is the wrong measurement.

**2. alkane_16 is already at the edge.** Its grid is TRUNCATED -- 405,038 of
412,500 points -- i.e. the unbatched path does not fit at 212 basis functions
on this box. That is well below drug scale (danuglipron is ~73 atoms), so this
is not a corner case.

Batching (PR #92) caps the gradient peak at the budget instead: benzene/cc-pVDZ
1.860 -> 0.250 GB (7.4x), benzene/6-31G 1.099 -> 0.250 GB (4.4x). The two water
rows are unchanged at 1.0x by design -- batching engages only when the working
set does not fit.

MEASUREMENT CAVEAT, learned the hard way: these tests carry
`#[ignore = "production-scale measurement; run serially on a quiet box"]`.
Run concurrently they FAIL, and the failure looks exactly like a code defect --
it is not. Satisfy the precondition (`--test-threads=1`, idle box) before
believing either the numbers or a failure.

### Frequencies / TS verification cost 6N+1 gradients (MEASURED)

`harmonic_frequencies` central-differences the ANALYTIC gradient, so a full
Hessian is **6N+1 gradient evaluations** (6N displaced plus one undisplaced,
which `n_gradient_evaluations` does not count), each requiring its own
converged SCF.
There is no analytic Hessian to fall back on -- see section 4.

(Heading and this paragraph corrected 2026-09-19: both said "ESTIMATED" and
"6N + 1" while the table two paragraphs below recorded the MEASURED 6N that
refuted exactly that. A stale summary above a correct measurement is the more
dangerous of the two, because it is what gets quoted.)

VERIFIED TWICE, and the second pass CORRECTED the first. Reading the loop gives
`for b in 0..n_coord` over 3N coordinates with TWO `energy_and_gradient` calls
inside (`+delta`, `-delta`), plus one at the undisplaced geometry before the
loop -- from which I concluded 6N+1. **That was wrong by one.** The live counter
`FrequencyResult.n_gradient_evaluations` reports exactly 6N:

| system | atoms | counted |
|---|---|---|
| H2 | 2 | 12 |
| water | 3 | 18 |

The undisplaced call supplies `result.energy` and is not counted as a gradient
evaluation; `frequencies.rs:194` documents the field as `6N`. Lesson worth
keeping: reading a loop is better than trusting a docstring, but an EXPOSED
COUNTER beats both -- it cannot drift from what the code did. `energy_and_gradient` takes
`(mol, basis_name, op, scf_config, reference)` -- no density argument, so
every one of those calls runs a complete SCF from the default guess.

Scaling the one MEASURED DFT point (612.4 s at 71 atoms, STO-3G/PBE):

- a 20-atom QM region -> 120 gradient calls
- a 40-atom QM region -> 240 gradient calls

ESTIMATED, and the multiplier is the honest part: each call is a fresh SCF, so
the cost is 6N x (one converged SCF + one gradient), not 6N x (one gradient).

VERIFIED: `frequencies.rs` contains no guess-reuse or restart machinery -- grep
for guess / initial_density / restart / previous finds only unrelated test
comments. Every displaced SCF therefore starts from the DEFAULT GUESS, not from
the converged undisplaced density, even though a delta-Bohr displacement
perturbs the density negligibly. Seeding each displacement from the
undisplaced converged density is a self-contained optimization that should cut
the iteration count substantially on every one of the 6N calls. Unmeasured, but
the mechanism is not in doubt: it is the same reason geometry optimizers carry
the density between steps.

Two practical consequences for a catalyst workflow. First, TS verification is
not a cheap afterthought appended to an optimization; on a realistic QM region
it can dominate the whole job. Second, this is the strongest argument for
wanting libint2 `deriv_order=2`: an analytic Hessian replaces 6N SCFs with one
CPKS solve. That is a real speedup of something that WORKS, not an unblock --
unlike the missing saddle search, which blocks the workflow outright.

### One structural result that IS solid

**QM/MM memory is set by the QM region alone.** MM point charges carry no grid
points, so embedding a ligand in a whole protein costs the same AO cache as the
ligand in vacuum. This follows directly from `cost.py`'s model (grid scales
with ATOM COUNT of the QM region) and is why QM/MM is affordable where a
larger QM region is not.

But read it with the gradient measurement below: the quantity that must fit is
the GRADIENT peak of the QM region, which is 27-92x its SCF peak, and a 212
basis-function region already truncates the grid unbatched. "QM/MM is
affordable" means the MM environment is nearly free -- it does NOT mean a
generous QM region is. The two statements are often conflated, and only the
first is true.

### Where the budget should NOT go (MEASURED, and counter-intuitive)

`tools/campaign/hierarchy.py` rule 6, from RESULTS.md M11: raising Vina's
`exhaustiveness` from 4 to 32 cost **6.8x** and improved the top score by
**0.005 kcal/mol** -- against a scoring function whose published RMSE is
~2.5 kcal/mol, a gain ~500x smaller than its own error bar. Redock RMSD across
an 8x effort range moved 0.097 A against a 0.131 A between-seed SEM, so it was
not resolvable either.

The sensitive variable was the STARTING CONFORMER (0.75-1.24 A across ETKDG
seeds), so **three seeds at the cheap setting beat one seed at the expensive
one** -- cheaper AND better sampled.

Two consequences for the cost table above. First, the "~2 min/ligand" tier-1
row is exhaustiveness 32; at exhaustiveness 4 with three seeds you get a better
answer for less. Second, and more general: the measured tier costs tell you
what a run WILL cost, not where to spend. Find the sensitive variable by
measuring, then spend there.

### Why the funnel exists at all (MEASURED)

Also from `hierarchy.py`: Vina reproduced the crystal pose at **0.95 A in ~2
minutes**; the best any tier-3 method managed in **62 minutes** was 2.41 A.
But `r(vina_score, pose RMSD) = +0.461`, and only 4 of 20 poses were under
2.0 A -- so the cheap score FINDS the right pose and cannot reliably RANK it.
That asymmetry is the empirical justification for tiers 2-4 existing: if tier 1
could pick its own best pose, no rescoring would be needed.

This is also the cautionary tale for the geometry defect in section 0. The
danuglipron campaign ran tier 3 alone on free-solution conformers -- no tier 1,
so it never searched -- and spent four rounds of increasingly careful
statistics on a metric fed geometries 2.2-4.1 A from the binding mode. The
unwritten `context["geometry"]` reproduces exactly that failure inside a
pipeline that LOOKS like it has a docking tier.

### Which dispersion model, once one exists (researched 2026-09-19)

D3(BJ) is being added (see the open work). The comparison behind that choice,
because "add dispersion" has four plausible answers and they are not equivalent:

| model | needs from the SCF | cost | status in ferric |
|---|---|---|---|
| D3(BJ) | geometry + Z only, not even a density | negligible | being added |
| D4 | geometry, Z, EEQ charges | negligible | none; reuses nothing ferric has |
| XDM | rho, grad-rho, tau, grad^2-rho on a grid + Hirshfeld weights | negligible vs the SCF | ~80% present, see below |
| VV10 | rho, grad-rho INSIDE the SCF | O(N_pts^2) pair sum | implemented, inside specific functionals |

**Accuracy is not the discriminator.** Published S66 MAE (kcal/mol): BLYP-D3
0.19 vs BLYP-XDM 0.19; B3LYP-D3 0.28 vs B3LYP-XDM 0.22. The 2026 GMTKN55 paper
states D3(BJ), XDM(BJ) and XDM(Z) "perform similarly". Anyone claiming a clear
winner on dimer binding energies is overreading the data.

**Name collision worth knowing.** D3(BJ) uses Becke-Johnson DAMPING but
Grimme's C6 table. XDM is Becke & Johnson's own MODEL, deriving C6/C8/C10 from
the exchange-hole dipole moment. They are different things that share names.

**ferric is unusually close to XDM.** Each ingredient VERIFIED present:

| XDM needs | ferric has |
|---|---|
| tau (Eq. 48) | `eval_tau_closed` / `eval_tau_uks` |
| AO Hessians -> grad^2-rho | `eval_basis_grad_hess_on_points` |
| **V_A = int r^3 w_A rho -- literally XDM Eq. 45** | `atomic_effective_volumes_becke` (properties.rs:478) |
| free-atom alpha, Z=1-54 | `ts_free_atom` |
| grad^2-rho assembled | **ABSENT** -- the one real gap |

So XDM Step 3 is already implemented and tested here. The gap is assembling
grad^2-rho from Hessians that exist, plus a scalar Newton solve per grid point.
The reference implementation, `postg` (Otero-de-la-Roza & Johnson), is GPL-3.0
and a PROGRAM not a library -- no permissively-licensed XDM library exists in
any language, and it is in neither PySCF nor Psi4. That argues for a native
implementation rather than against one.

**A dispersion correction does NOT fix QM/MM.** D3/D4/XDM/VV10 are all
QM-atom-pairwise. ferric feeds MM atoms in as point charges carrying no
dispersion at all, so QM-MM dispersion is a SEPARATE gap -- probably LJ terms
across the boundary in `ferric-mm`. Easy to assume away, so stated explicitly.

### The label "DFT + dispersion" -- FIXED 2026-09-19 (not yet merged)

`crates/ferric-d3` implements two-body D3(BJ) natively, Z=1..103, with ZERO new
dependencies. Tier 4 now defaults to `dispersion="d3bj"`, so the label three
places carry is finally true.

INDEPENDENTLY VERIFIED by me, not taken from the implementer's report: same
water geometry through ferric and through live simple-dftd3 1.6.0.

| functional | simple-dftd3 1.6.0 | ferric | delta |
|---|---|---|---|
| PBE | -3.594687655702e-4 | -3.594687655704e-4 | 2e-16 Ha |
| B3LYP | -5.738758352593e-4 | -5.738758352595e-4 | 2e-16 Ha |

Machine precision. 21 Rust tests pass. The reference tables are GENERATED by
`generate_tables.py` from upstream s-dftd3 Fortran rather than hand-copied, so
they are reproducible and auditable; CONTRIBUTING.md gained a separate "Derived
data (not linked)" section, because the crate links nothing but its tables
still derive from LGPL-3.0 source -- an obligation the existing linking table
could not express.

WHAT IS STILL MISSING, stated because a dispersion correction invites the
assumption that everything dispersive is now handled:
- **No analytic gradients.** `task="optimize"` silently optimizes the
  UNCORRECTED surface. That is a sharp edge, not a rounding error, for any
  geometry work on a dispersion-bound complex.
- **No ATM three-body term.** Absent rather than approximated -- there is no
  knob that does nothing. MEASURED contribution rises with system size:
  water 0.0001% -> benzene 0.1003% of the two-body energy. Extrapolating that
  trend to drug-scale ligands is explicitly NOT done.
- **QM/MM dispersion is still unfixed.** D3 is QM-atom-pairwise; the
  QM-to-MM-point-charge gap needs LJ terms in `ferric-mm`.

### The label "DFT + dispersion" was wrong before that

`hierarchy.py:15` and `vina_dock.py:8` both label tier 4 "DFT + dispersion".
Grep for D3 / D3(BJ) / Grimme / dftd3 across ferric-dft, ferric-scf and
ferric-python finds NO dispersion correction. VV10 exists inside specific
functionals, not as an add-on for PBE. `ferric_d3.py` wraps the external
`dftd3` package but is a loose helper, not wired into the tier.

Dispersion is the DOMINANT attractive term in ligand binding, so this is not a
cosmetic mislabel: tier 4 as run is missing the physics its own label claims.

---

## 3. The golden path, as an ordered list

Decision points are marked. Steps 1-6 are available today; step 7 is blocked
(see section 4).

1. **Ligand in.** SMILES -> `tools.structure.from_smiles`; a file ->
   `tools.structure.read`. State charge and multiplicity explicitly.
2. **Receptor in.** PDB -> `pdb2pqr` (`active_site/pdb2pqr_runner`) to add
   hydrogens and assign MM charges. Do NOT feed a crystallographic PDB
   directly; `tools.structure` refuses one with no hydrogens for this reason.
3. **Tier 1, dock.** ~2 min/ligand. The pose is a HYPOTHESIS -- run
   `redock_rmsd` against a known bound pose on your target first, or you do
   not know the search works.
4. **Harvest the pose into `context["geometry"]`** -- see section 0. Without
   this, everything below scores a gas-phase conformer.
5. **Tier 3, xtb.** ~0.5 s. **DECISION: stop here?** If you are rank-ordering
   many ligands and only need a coarse sort, GFN2 is often enough. Going to
   DFT costs ~200x per candidate at this scale (MEASURED; ~1000x only
   above ~270 atoms).
6. **Tier 3.5/4, DFT.** **DECISION: gas phase or embedded?** Gas-phase DFT on
   a docked pose ignores the pocket electrostatics entirely. Use the pocket
   field (`context["point_charges"]`, already consumed at `tiers.py:196` and
   `tiers.py:249`) when the pocket is charged or polar.
   **DECISION: do you trust the number?** See the pose-noise caveat below.
7. **Catalyst / barrier work.** BLOCKED -- see section 4.

### The pose-noise caveat, which decides whether step 6 is worth running

`funnel.py:162` keys results on `iso.canonical` -- ONE row per MOLECULE. But
QM/MM binding energy is a property of a POSE. RESULTS.md M5/M6 MEASURED a
per-pose sd of 29.07 kcal/mol and a within-molecule/between-molecule spread
ratio of 228 vs 41 kcal/mol; M6 computes that resolving a 0.25 kcal/mol gap
needs ~108,000 poses per candidate.

A one-pose-per-ligand QM/MM tier therefore reports a number whose noise is
orders of magnitude above the distinction being asked of it. **Decide the pose
treatment (ensemble? Boltzmann weight? best-N?) BEFORE writing the adapter**,
because the data structure the funnel uses cannot currently express it.

---

## 3b. Decision procedure: what a QM/MM workflow should DO

The two workflows are NOT the same problem and must not share a recipe.

### (a) Docking / binding affinity

The question is a RELATIVE energy between ligands, so error cancellation does
most of the work and the QM region can be modest.

```
G0. POSE-QUALITY GATE -- before scoring anything.
    tools.campaign.align.pose_quality_gate(aligned, threshold=2.0 A)
    Redock a KNOWN complex; if no pose clears the bar, STOP. Nothing
    downstream can recover, and this is precisely the check whose absence
    cost the danuglipron campaign four measurement rounds.
G1. Dock (tier 1), then HARVEST the pose into context["geometry"] (section 0).
G2. Rank with GFN2-xTB in the pocket field (tier 3 + point_charges).
    DECISION: if you only need a coarse sort, STOP HERE. DFT costs ~200x at
    danuglipron scale (MEASURED, see below -- NOT the ~1000x this line used
    to claim).
G3. QM region = the ligand. Pocket = MM point charges. No link atoms needed
    when the cut does not cross a covalent bond -- which for a non-covalent
    ligand it does not. This is the case ferric handles cleanly today.
G4. Report a DIFFERENCE (dG_bind between ligands, or vs a reference ligand),
    never an absolute. The absolute carries the full method error; the
    difference is what error cancellation protects.
```

**That blocking question is now ANSWERED, and the answer is "none of them"**
(2026-09-19, RESULTS.md M4-M14). It used to read: decide the pose treatment --
ensemble, Boltzmann weight, best-N -- before writing the adapter. All five
candidate treatments have since been measured:

| treatment | ddE noise | vs a 0.25 kcal/mol gap |
|---|---|---|
| more poses, averaged (M4/M5) | 4.07 | 16x |
| relax in field then average (M6) | ~4.1 | ~16x |
| dock then average (M12) | ~4.1 | ~16x |
| select the top-docked pose (M13) | 40.66 | 163x |
| a different scorer (M14) | 4.68 best tracking | 19x |

So `funnel.py:162`'s one-row-per-MOLECULE keying is not the blocker it was
written up as -- no pose treatment the data structure could express resolves a
1-2 kcal/mol substituent effect. **G2 is the last step whose output is
trustworthy.** Its own "STOP HERE if you only need a coarse sort" is now the
recommendation rather than an option, and G4's dG_bind difference is reportable
only when the gap is large (>~5 kcal/mol, i.e. outside the measured noise), not
for lead optimisation.

This does NOT weaken G0-G3: the pose is found reliably (M9, 0.95 A redock) and
the coarse sort works. It bounds what G4 may claim.

#### G1-G4 VERIFIED END TO END on merged main (2026-09-18)

Not a plan -- run against `origin/main` after PR #91 landed the readers:

```
PDB -> Molecule : 3 atoms, 10 electrons, ['O', 'H', 'H']
vacuum RHF      : -74.72406147 Ha, converged=True, 14 it
embedded RHF    : -74.74153533 Ha, converged=True, 12 it
G4 difference   : dE = -0.017474 Ha (-10.97 kcal/mol)
```

Every hop the docking branch claims is real today: `tools.structure.read` takes
a PDB (explicit charge/multiplicity, no guessing), hands a `Molecule` to
`run_rhf`, the same QM region runs again with `point_charges=`, and the answer
reported is a DIFFERENCE rather than an absolute. Both SCFs converged.

The script is `/tmp` scratch, not committed -- it is four calls and is
reproduced above in full effect. What matters is that it was RUN, so the (a)
branch below is a description of working code rather than an intention. The (b)
branch is not, and cannot be until a saddle search exists.

### (b) Catalyst optimization

The question is a BARRIER, i.e. a SADDLE POINT. Error cancellation does not
save you, and the QM region must contain the reacting bonds.

```
C0. QM region MUST contain every bond that breaks or forms, plus any residue
    donating/accepting a proton or coordinating the metal. This is bigger
    than a ligand-only region and sets the cost.
    AND: check the MM field does not break a symmetry that DEFINES your
    saddle. MEASURED -- an antisymmetric charge pair makes planar NH3
    non-stationary (max|g_z| 3.8e-3 vs ~1e-16 in gas phase), so the search
    fails for want of a target and looks like a solver bug.
C1. The cut WILL cross covalent bonds, so link atoms are mandatory:
    .with_link_atoms(bonds, DEFAULT_LINK_SCALE) and a boundary-charge scheme
    (Z1/RC/RCD). ferric has all of these, PySCF-validated.
C2. Optimize the reactant and product complexes -- ferric CAN do this
    (optimize_qmmm is a minimizer).
C3. FIND THE TRANSITION STATE. AVAILABLE AND MERGED (#106, 2026-09-19) from
    BOTH languages: ferric_scf::saddle::find_saddle (Rust) and
    ferric.run_saddle(mol, basis, ...) (Python). The Python binding
    matters because the whole tools/ pipeline is driven from Python --
    without it C3 existed in a language the pipeline does not speak.
    See section 4.
C4. Verify the TS: harmonic_frequencies -> n_imaginary() == 1, AND the
    imaginary mode must point along the reaction coordinate (one imaginary
    frequency is necessary, not sufficient -- a methyl rotor gives one too).
    COMPLETE since #97: `PyFrequencyResult.normal_modes` is a real
    #[pyo3(get)] accessor on main (VERIFIED against origin/main
    2026-09-19), so both halves are reachable from Python. The "MODE
    VECTORS are Rust-only" caveat this line used to carry is STALE.
C5. Barrier = E(TS) - E(reactant), with ZPE from the same frequency run.
```

#### C0-C5 VERIFIED reachable from PYTHON, end to end (2026-09-19)

The steps landed one at a time across several PRs, and the failure mode is a
procedure that READS as complete while one step lives only in Rust. That
happened twice: C3 until `run_saddle` was bound, and C4's mode vectors until
#97 -- after which this document carried a stale "MODE VECTORS are Rust-only"
caveat for a day.

So it is now asserted by execution rather than by reading:

```
C0/C1 QM region + link atoms     ferric.QmmmSystem
C2    optimize reactant/product  ferric.run_optimize_qmmm
C3    FIND the transition state  ferric.run_saddle
C4    verify                     ferric.run_frequencies
C5    barrier                    arithmetic on C2/C3 energies

C4a count   : 1 mode, 0 imaginary (H2 at equilibrium IS a minimum)
C4b vectors : normal_modes present, 1x6 -- 3N components per mode
```

Pinned by `crates/ferric-python/tests/test_saddle.py`, and mutation-tested:
renaming a checked attribute fails the test, so it is not a tautology over
`hasattr`. It asserts REACHABILITY only -- each step has its own correctness
tests; what this catches is a step quietly leaving the language `tools/` is
written in.

#### ...and C3 now RUNS under embedding, not just reachable (2026-09-19)

Reachability was the weaker claim, and the QM/MM example made it weaker still:
it demonstrated only a REFUSAL (H2 in an MM field has no saddle, so
`find_saddle` declines). A refusal alone does not show the procedure works --
code that rejected everything would print the same thing.

`crates/ferric-scf/tests/qmmm_saddle_converges.rs` closes that. NH3 umbrella
inversion under point-charge embedding **converges in 5 steps to exactly one
imaginary mode, `is_transition_state() = true`, z spread 0.0075 Bohr** -- i.e.
planar, the physically right answer. C3 is now demonstrated on the embedded
surface, which is the surface a catalyst question is actually asked on.

**The trap it surfaced, which belongs in C0.** An MM field that BREAKS THE
SYMMETRY DEFINING THE SADDLE removes the target rather than making the search
harder: with an antisymmetric charge pair, max|g_z| at the planar geometry is
3.8e-3 against ~1e-16 in gas phase, so planar NH3 is not a stationary point at
all. `find_saddle` correctly fails and it reads like a solver bug. See the
transition-state cost section for the table and the diagnostic.

#### C1 and C4 VERIFIED to work (2026-09-18)

Run against the merged extension, so these are not claims:

```
C1 bare cut       : QM has 4 atoms            (ethane, QM = one CH3)
C1 with link atom : QM has 5 atoms  -> ADDED  symbols ['C','H','H','H','H']
C4 H2 at minimum  : 1 mode, n_imaginary = 0   frequencies [5018.8] cm-1
```

C1 (`QmmmSystem(...).with_link_atoms([(0, 4)])`) caps a cut C-C bond with an H,
exactly as the catalyst path requires -- the cut across a covalent bond is the
step that distinguishes a catalyst QM region from a ligand one. C4's
`n_imaginary` returns 0 at a MINIMUM, which is the verifier's negative control:
a verifier that cannot report "this is not a saddle" cannot report "this is"
either.

So C0-C2 and C4-C5 are working code. C3 was the ONLY gap, established by grep
(no dimer / NEB / P-RFO / eigenvector-following anywhere) rather than by this
script -- a negative cannot be demonstrated by running something.

**C3 CLOSED AND MERGED 2026-09-19** (`ferric_scf::saddle`, #106), and
demonstrated under QM/MM embedding rather than only in the gas phase. The chain
C0-C5 is complete. What that does and does not mean is in the rewritten
section 4 -- in particular there is still no IRC.

API INCONSISTENCY worth knowing before writing a workflow: `run_rhf` takes a
`BasisSet` OBJECT (`ferric.BasisSet.bundled("sto-3g")`), while
`run_frequencies` takes a basis-name STRING. Passing the wrong one is a clean
TypeError, not a silent failure, but it costs a round trip. The frequency entry
point is `run_frequencies`, NOT `harmonic_frequencies` -- the latter is the
Rust name and is not what pyo3 exports.

#### C4 now COMPLETABLE from Python (2026-09-19, PR #97)

`FrequencyResult.normal_modes` is bound, so a Python workflow can finally do
what C4 requires -- count imaginary modes AND inspect what one displaces.
Demonstrated on water/STO-3G:

```
3 modes, n_imaginary = 0  -> MINIMUM
softest mode (2049.4 cm-1) displacements:
    O0  0.0016
    H1  0.0125
    H2  0.0125
```

The bend moves both hydrogens symmetrically while the oxygen barely moves --
physically right, and the kind of check that distinguishes "one imaginary mode"
from "one imaginary mode ALONG THE REACTION COORDINATE". A methyl rotor also
gives exactly one imaginary frequency; only the vector tells them apart.

So the catalyst path's verification half is complete: **C0-C2 and C4-C5 all
work from Python.** C3 -- FINDING the saddle -- remains the single blocker, and
it is a missing capability (no dimer/NEB/P-RFO anywhere), not a missing
binding.

~~**API caveat, VERIFIED 2026-09-18:** `FrequencyResult.normal_modes` is NOT
exposed in the pyo3 bindings...~~ **SUPERSEDED 2026-09-19.** #97 exposed it.
`PyFrequencyResult.normal_modes` is a real `#[pyo3(get)]` accessor
(`crates/ferric-python/src/lib.rs:1768` on origin/main, re-verified
2026-09-19), returning `Vec<Vec<f64>>` in the documented (mode, 3N) layout. A
Python workflow can now both COUNT imaginary modes and INSPECT them, so C4 is
complete from Python.

If a local check disagrees, check the loaded extension before the source: the
`.so` symlinked into `.venv` points at the MAIN checkout's `target/release`, so
a worktree can be testing a stale build. That is what made this caveat look
current when I first re-read it today.

~~**C3 does not exist.**~~ **SUPERSEDED 2026-09-19** by
`ferric_scf::saddle::find_saddle` (P-RFO), and wired to QM/MM in
`examples/qmmm_saddle.rs`. C0-C5 is complete in principle; section 4 states
what that does and does not mean. A catalyst workflow no longer has to import
its transition state from another code -- though doing so and using ferric to
verify (C4) and compute the barrier at a better level (C5) remains a perfectly
good option, and is the cheaper one when a TS is already in hand.

## 4. Catalyst optimization: the search gap is CLOSED, the cost one is not

**UPDATED 2026-09-19.** This section previously read "BLOCKED". The blocker was
C3 -- no saddle search anywhere in the tree. That is now implemented.

### What landed

`ferric_scf::saddle::find_saddle` (MERGED, #106):
partitioned rational function optimization. It partitions the Hessian
eigenspace and solves a separate RFO step in each -- maximize along one
followed mode, minimize in the orthogonal complement (Banerjee/Adams/Simons/
Shepard, JPC 89, 52 (1985)). Between steps the Hessian is carried by a Bofill
update, chosen because it does NOT preserve positive definiteness; BFGS would
drive out the negative eigenvalue the whole search depends on.

Why it could not be a flag on `optimize.rs`: that is a MINIMIZER, and its
quasi-Newton update is kept positive definite on purpose. No step size turns a
minimizer into a saddle finder.

### CORRECTION to the previous draft of this section

It said P-RFO "reuses the Hessian machinery already in `hessian.rs`". **Wrong.**
`hessian.rs::rhf_hessian` is a documented stub that ALWAYS returns `Err` (this
same section says so four paragraphs below, which should have caught it). The
working Hessian is `frequencies.rs`'s central-differenced analytic gradient,
and that is what `saddle.rs` calls.

### The cost, which is now the binding constraint

The Hessian is **6N+1 gradient evaluations** -- the counter reports 6N and
omits the undisplaced call (MEASURED via the gradient counter:
H2 = 12, water = 18 -- exactly 6N). For a 20-atom QM region that is 120
gradients for ONE Hessian. Rebuilding it every step is not a search, it is a
Hessian benchmark, which is why `hessian_recalc_every` defaults to 0 (build
once, then Bofill). A realistic catalyst run is therefore:

    1 Hessian (6N+1 gradients) + ~20-60 P-RFO steps (1 gradient each)

so the Hessian dominates at small N and the steps dominate past roughly N = 10.
That ratio, not the algorithm, is what sizes a catalyst job now.

### What is still NOT available

- **No analytic Hessian.** libint2 `deriv_order=2` is absent from this build
  (VERIFIED from `~/.local/include/libint2/config.h`: `INCLUDE_ERI 1`,
  `INCLUDE_ONEBODY 1`). Raising it means re-running libint's generation stage,
  a once-off out-of-band build, not a cmake flag. Every Hessian here is finite
  difference.
- **No IRC**, so "this saddle connects THESE two minima" is still unverified.
  P-RFO finds a first-order saddle; it does not prove which reaction it belongs
  to.
- **No reaction-coordinate constraint / relaxed scan.** `MoveMm` freezes whole
  MM atoms and QM atoms are always free (`free_atom_indices`, `qmmm.rs:1815`),
  so there is still no constrained-scan route to a starting guess.
- ~~**Not wired to QM/MM.**~~ **DONE 2026-09-19**, and it needed less than
  expected. `crates/ferric-scf/examples/qmmm_saddle.rs` runs the whole chain on
  an embedded system: SCF -> analytic embedded gradient -> finite-difference
  embedded Hessian -> projection -> saddle step.

  The part I expected to block it did not: **a QM/MM Hessian needs no new
  machinery.** `frequencies::harmonic_frequencies` already threads
  `config.external_potential` into the same `rhf_gradient(.., ext)` /
  `ks_gradient_closed(.., ext)` calls (`frequencies.rs:649`), so an embedded
  Hessian is one ordinary call.

  Three limitations remain, and they are why this ships as an EXAMPLE rather
  than a library entry point:
  - **MM charges are FIXED** (`to_external_potential()` evaluated once). A
    barrier computed this way omits MM relaxation along the reaction
    coordinate. `optimize_qmmm` rebuilds the field per step
    (`qmmm.rs:2066`) precisely because that matters when MM atoms are free.
  - **Link atoms do not track the frontier** as the QM region distorts.
    `optimize_qmmm` shares this.
  - Making it a library function means extracting `optimize_qmmm`'s 172-line
    inline evaluator closure -- a refactor with its own risk, and its own PR.

  **Reachable from Python since 2026-09-19**: `ferric.run_saddle(mol, basis,
  xc=, multiplicity=, max_steps=, trust_radius=, follow_mode=, delta=)` ->
  `SaddleResult`, with `is_transition_state()` as a method so `converged`
  alone cannot be read as a TS. The refusal crosses the FFI boundary with its
  reason intact -- VERIFIED on H2 at 0.74 A, which returns "the projected
  Hessian at the starting geometry has NO negative eigenvalue (lowest =
  9.612869e-1)" rather than an opaque failure.
- **Cartesian only.** Internal-coordinate P-RFO converges in fewer steps on
  floppy systems.
- **The mode is not checked for being the RIGHT one.** Exactly one imaginary
  frequency means first-order saddle, not "saddle for the reaction you meant" --
  a methyl rotor gives one too. `SaddleResult::imaginary_mode` returns the
  vector so a caller can check; the module does not pretend to.

### What it refuses to do, on purpose

- Starting with no negative projected eigenvalue is a HARD ERROR naming the
  lowest eigenvalue. P-RFO from a minimum's basin has nothing to climb and
  would otherwise return a minimum labelled as a transition state.
- `is_transition_state()` requires gradient convergence AND exactly one
  imaginary mode. Convergence alone is satisfied by every stationary point.

### Honest status line

> ferric can now SEARCH for a transition state and CONFIRM one. It still has no
> analytic Hessian, no IRC, and no QM/MM wiring for the search -- so a catalyst
> workflow knows what to do, and the remaining work is integration and cost,
> not a missing capability.

---

## 5. Ordered next actions

**UPDATED 2026-09-19.** Item 0 is new and is now the top of the list, because
the capability it wires up did not exist when this list was written.

0. ~~**Wire `saddle::find_saddle` to the QM/MM evaluator.**~~ **DONE
   2026-09-19** -- `examples/qmmm_saddle.rs`, verified running end to end on an
   embedded system. The catalyst workflow can now run. What remains is
   promoting it from an example to a library entry point (needs
   `optimize_qmmm`'s evaluator extracted) and lifting the fixed-MM-field
   approximation; see section 4.

1. **Harvest the docked pose** into `context["geometry"]` in `run_funnel`'s
   stage loop (section 0). Highest value, smallest change, independent of
   QM/MM. Needs the parallel-path anchor test.
2. **Fix the "DFT + dispersion" label** -- either wire `ferric_d3.py` into
   tier 4 or correct the two docstrings. Do not leave the label claiming
   physics the tier does not compute.
3. **Decide the pose treatment** (section 3) before any QM/MM adapter.
4. **Add the tier 3.5 producer**: `derive_pocket_charges` -> `context`. Both
   quantum tiers ALREADY consume `context["point_charges"]`, so this is a
   producer, not a new capability.
5. **Fix `relax_pose_in_pocket`'s movable pocket**: it builds its QmmmSystem
   from `(q,x,y,z)` with symbol `"X"` (`pose_relaxation.py:188`), but
   `move_mm != "none"` needs an `MmTopology` in the same atom order, and
   `PocketCharges` carries no element symbols. `move_mm="within"/"all"`
   type-checks and is unreachable in practice.
6. **Fix the drifted docstring**: `qmmm.rs:26-34` says "no Lennard-Jones QM-MM
   term", but `qmmm_mm_terms` (`qmmm.rs:1578`) computes one at
   `qmmm.rs:1674-1704`. The code is right; the comment is stale.
7. **Expose `FrequencyResult.normal_modes` to Python.** [still open; and now
   MORE valuable -- it is what lets a Python caller check `find_saddle`'s
   imaginary mode points along the reaction coordinate, which is step C4's
   second half.] VERIFIED absent from
   the bindings. Without it a Python workflow can count imaginary modes but
   not check one points along the reaction coordinate, so it cannot complete
   TS verification (step C4). Small, self-contained, and a prerequisite for
   any Python-driven catalyst work.
8. **Seed each finite-difference displacement from the undisplaced converged
   density.** VERIFIED that `frequencies.rs` has no restart machinery, so all
   6N displaced SCFs start cold from the default guess despite being a
   delta-Bohr perturbation apart. Self-contained, and it pays off on every
   frequency run -- which is every TS verification.
9. **Surface `tools/` in `site/src/SUMMARY.md`.** None of this pipeline is in
   the published docs.
