# Golden path / pipeline audit — iteration 1 (2026-09-18)

Findings from reading the source. MEASURED vs ESTIMATED labelled throughout.
A companion design doc (`golden-path-qmmm-pipeline.md`) is being written by a
research agent; this file is the survey that scoped it.

## 1. The pipeline already exists. It is invisible.

`tools/pipeline/` implements the full SMILES -> dock -> xtb -> DFT funnel:

| module | role |
|---|---|
| `tiers.py` | four uniform tier adapters over four very different methods |
| `funnel.py` | the narrowing loop + auditable bookkeeping (entered/survived/FAILED) |
| `cost.py`  | DFT cost model (grid scales with ATOMS, not basis) |
| `tools/campaign/hierarchy.py` | the discard rules |
| `tools/docking/vina_dock.py` | tier 1, with an honest-limits section |

MEASURED costs, quoted from `tools/pipeline/tiers.py:9-16` (70-atom anion,
def2-SVP/PBE at tier 4):

    tier 1  Vina        ~2 min/ligand at exhaustiveness 32
    tier 2  MMFF94      ~1 ms/pose
    tier 3  GFN2-xTB    ~0.5 s single point
    tier 4  ferric DFT  96.1 s at 32 atoms -> 17-37 min at 70 (N^3-N^4)

The `vina_dock.py` header carries a second, order-of-magnitude table
(~10 us/pose at tier 1, 10^5-10^6 poses). The two tables are consistent:
one is per-pose, the other per-ligand at production exhaustiveness.

**Gap: `grep -rn 'tools/pipeline|funnel|vina' site/src/` returns NOTHING.**
None of this is in the published mdBook. Same failure mode as the QM/MM page
that PR #89 just fixed: a working, documented-in-source capability with no
route to it from the docs.

## 2. Two complete stacks, zero cross-imports

- `tools/pipeline/` — ligand side, SMILES-driven.
- `tools/active_site/` — protein side: `pdb2pqr_runner`, `pqr_parser`,
  `pocket_charges`, `ligand_embedding`, `mm_topology`, `pose_relaxation`
  (calls `ferric.run_optimize_qmmm`), `binding_energy`.

VERIFIED: `grep -rn 'active_site' tools/pipeline/` -> empty, and
`grep -rn 'tools.pipeline' tools/active_site/` -> empty. `tiers.py` imports
only `tools.isomers.model`. **There is no tier that carries a docked pose into
a QM/MM calculation**, even though both ends of that bridge are built.

## 3. Catalyst optimization is BLOCKED, not merely unimplemented

VERIFIED by grep across `crates/`: no dimer method, no NEB, no
eigenvector-following, no P-RFO, no saddle search of any kind.
`crates/ferric-scf/src/optimize.rs` exposes exactly four entry points --
`optimize_geometry`, `_uhf`, `_rohf`, `optimize_coordinates` -- all MINIMIZERS.

What DOES exist is the verifier: `frequencies.rs` gives `harmonic_frequencies`
and `FrequencyResult::n_imaginary()`, documented as "Zero at a minimum, one at
a first-order saddle point" (frequencies.rs:215-216).

So the asymmetry is sharp:
- ferric can **confirm** you are standing on a transition state.
- ferric cannot **find** one.

A catalyst workflow therefore cannot be completed inside ferric today. It must
either import a TS geometry from elsewhere and use ferric to verify/refine, or
a saddle-point search must be implemented. This is the single largest gap
between the current code and "a QM/MM workflow knows what to do for catalyst
optimization".

Docking/binding is NOT blocked in the same way -- it is a minimization problem
end to end, and every piece exists; only the connecting tier is missing.

## 4. Input formats

Rust `Molecule` (`crates/ferric-core/src/mol.rs`) has ONLY `load_xyz`,
`load_xyz_with_charge`, `parse_xyz`. Every other format is handled Python-side:

| format | reader | layer |
|---|---|---|
| xyz | `Molecule::load_xyz` / `parse_xyz` | Rust |
| SMILES | rdkit ETKDG via `tiers.py` tier2 | Python |
| PDB | `tools/active_site/pdb2pqr_runner.py` | Python (external pdb2pqr) |
| PQR | `tools/active_site/pqr_parser.py` | Python |
| AMBER prmtop / OpenMM | `mm_topology.topology_from_openmm` | Python |

No Rust-side PDB/SDF/mol2/SMILES reader exists (VERIFIED by grep for
`fn .*(read|parse|load)_(pdb|sdf|mol2|smiles)` across `crates/` -> empty).

## 5. Next actions (iteration 1 verdict)

1. Surface `tools/` in `site/src/SUMMARY.md` -- the pipeline is the single most
   useful undocumented thing in the repo.
2. Add the missing QM/MM tier adapter so `funnel.py` can reach
   `active_site.relax_pose_in_pocket`.
3. Decide on the catalyst lane: import-and-verify (cheap, honest, available
   now) vs implement a saddle search (large). Do NOT document a catalyst
   workflow as supported until one of these lands.
