# Exhaustive plan: Python integration touchpoints (Rich Hickey lens)

Simplicity over ease: prefer plain data (dicts/arrays) over classes; immutable results; composable functions.

Goal: make ferric easy to use from Python for method development, matching CLI capabilities.

## Touchpoints
1. Packaging: py.typed, ferric.pyi in package (done).
2. Build freshness: maturin develop workflow documented.
3. Binding coverage: run_* for all CLI method kinds.
4. Result objects: expose all Rust result fields (done for PDEP-RPA extras).
5. Serialization: RESULT_SCHEMA kept in sync (done for new fields).
6. Docstrings: populate from ferric.pyi (done).
7. Geometry optimization from Python (run_optimize with xc/reference) (done).
8. Fock hooks derivatives (done).
9. X2C driver option (done: rimp2, scs-mp2, ccsd, ccsd(t)).
10. QM/MM integrations (binding exists; docs done).
11. Tests: pytest coverage for each binding (done).

## Data transformation & ownership (Raymond Hettinger lens)

- Prefer transforming data with comprehensions/pipelines over mutating objects.
- Each binding owns its inputs; results are immutable snapshots.
- Clear ownership: Rust owns computation; Python owns orchestration.

## QM/MM integration ease (cheminformatics practitioner lens)

- Accept RDKit/ASE/OpenMM inputs directly; convert at boundary.
- Keep MM parameters explicit and validated.
- Provide one-call QM/MM energy/gradient with sensible defaults.
- Export results to standard formats (XYZ, PDB, NPZ).

## Plan gaps for Python dev (Tim Berkelbach lens)

- Match PySCF's ergonomics: `run_*` returns objects with `.kernel()`-like convenience.
- Ensure basis sets and AO ordering map to PySCF for validation.
- Provide finite-difference gradient checks and reference comparisons.
- Keep performance competitive for small molecules (prototyping).

## Stat mech / Chandler lens

- Add molecular dynamics (NVE/NVT/NPT) bindings with thermostats/barostats.
- Expose time-correlation functions and transport properties.
- Ensure energy/force consistency and conserved quantities.
- Support periodic boundary conditions and long-range electrostatics.

## Execution order
- Phase 1: packaging/type info (done).
- Phase 2: expose missing result fields.
- Phase 3: add missing run_* wrappers.
- Phase 4: geometry/X2C/Fock hooks (done).
- Phase 5: docs + tests (done).
