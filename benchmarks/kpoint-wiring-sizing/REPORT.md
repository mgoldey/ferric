# Sizing: CLI and Python wiring of the k-point forces and stress (#236 follow-up)

Sizing written before the implementation; the implementation followed it (M, 3-5 days).

## What was wired before

| | Gamma | k-mesh |
|---|---|---|
| CLI energy | yes | yes |
| CLI forces | only inside `task = "optimize"` (fixed lattice, Cartesian BFGS, final gradient printed) | refused: "k-point forces are not wired" |
| CLI stress | none, even at Gamma | none |
| Python forces/stress | `with_gradient` / `with_stress` on the `run_*_gamma` functions | none (`run_rks_kpts` accepted the flags and raised) |
| Cell optimisation | none anywhere | none |

The obstacle: `solve_krhf` and `solve_kuhf` dropped the hcore and J/K object they built, which the force and stress entry points need.

## Sizing: M

- Library retained-state API: about 0.5 d.
- Python: about 1 d.
- CLI optimize: about 1.5 d.
- Tests and docs: about 1.5 d.

## Out of scope, with reasons

- CLI stress output: it does not exist at Gamma either, so adding it is a separate feature for both.
- Cell optimisation: absent everywhere.
- `range_split` with forces or stress: the Python bindings refuse this at Gamma, and the CLI does too under `task = "optimize"`; the k-point wiring mirrors that.

## Risks

- The retained J/K objects stay alive through the force, so peak memory is the energy peak plus the force's own budget.
- Meshes with every n_i <= 2 cannot detect phase errors, so some anchors need a mesh with n_i >= 3.

## Decomposition and anchors

- Library: `solve_krhf_keep` / `solve_kuhf_keep` return a `KRetained` (hcore and J/K object); `solve_krhf` / `solve_kuhf` call them, so energies are bit-identical by construction. Anchors: keep vs plain energies bitwise equal; force through the retained objects vs central FD of the energy; a stale-hcore mutant refused by the Fock-consistency check.
- Python: `with_gradient` / `with_stress` on `run_rhf_kpts` and `run_uhf_kpts`; units identical to the Gamma API (Hartree/Bohr per cell; Hartree/Bohr^3). Anchors: force vs FD, stress trace vs isotropic scaling, 1x1x1 mesh vs the Gamma bindings, 1x1x3 mesh vs one copy of the 3x Gamma supercell, UHF with staged Ewald, RS-GDF, refusals; mutation: an Angstrom/Bohr factor error and a wrong exxdiv to the stress.
- CLI: `kmesh` with `task = "optimize"` for `rhf` and `uhf` (dense or RS-GDF J/K, fixed lattice); k-point KS, ROHF and MP2/dRPA stay refused, as do `range_split` and `gdf_omega = "auto"`. Anchors: the printed final gradient vs the library force at the printed geometry (bar 5e-7: the printed geometry has 8 decimals), a real optimisation of stretched H2, a mesh mutant.
- Guards: `input_reference_documents_exactly_the_accepted_keys`, `all_shipped_examples_parse`, `examples_index_lists_every_example`, `units_literal_guard`, `proof_links_are_live`, and the pyi tests against a fresh `.so`.
