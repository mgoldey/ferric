# The Python surface, cube files and provenance {#python-cube-provenance}

<!-- status: edit; registry: none -->

The NPZ file is one way to get features out of ferric. A dataset builder
written in Python usually calls the bindings instead, either for a whole
result or for the building blocks, and keeps the arrays in memory. Whatever
the route, each sample needs a record of what produced it: two samples with
the same keys can come from different builds, references or grids, and the
arrays do not show it. This section maps the Python entry points onto the
quantities of sections @sec:npz-bundle to @sec:polarizability-pdep-c6 (@sec:python-cube-provenance-results-property-functions-building), covers cube files (@sec:python-cube-provenance-cube-files), and then the
certificate, which is the provenance record a dataset should keep with every
sample (@sec:python-cube-provenance-certificate-provenance-label).

## Results, property functions and building blocks {#python-cube-provenance-results-property-functions-building}

The bindings come in layers, from a finished result down to the integrals.
Pick the highest layer that computes what you need; the lower ones exist for
features ferric does not compute for you.

| Layer | What it returns | Use it for |
|---|---|---|
| `run_*` results (`RhfResult`, `DftResult`, `PdepRpaResult`, …) | energy, MO coefficients and energies, density, gradient where requested, `.certificate` | labels and the reference wavefunction |
| `run_pdep_rpa(..., properties=True)` | every NPZ array as an attribute of the same name, `None` when not requested or failed | the full feature bundle without a file |
| property functions: `esp_at_atoms`, `esp_at_points`, `mulliken_charges`, `lowdin_charges`, `hirshfeld_charges`, `chelpg_charges`, `resp_charges`, `hirshfeld_polarizability`, `orbital_moments`, `density_second_moment` | one property from an `RhfResult` or `DftResult` | features from an SCF you already ran, including KS references |
| `AoIntegrals(mol, basis)` | overlap, kinetic, nuclear, core Hamiltonian, dipole integrals, ERIs (4-, 3-, 2-index), J/K, their nuclear derivatives, `pvp`, `hcore_x2c` | rebuilding $S$ and the AO metadata the NPZ omits; custom features |
| `DfJk`, `XcFunctional`, `Cphf` | density-fitted J/K, XC on a grid, coupled-perturbed solves | building a new response property |
| `response_modes` | leading dielectric modes at one imaginary frequency | PDEP modes outside the NPZ path |

The property functions differ from the NPZ path in one way that matters:
they take the reference you give them, so they work on a Kohn–Sham density.
The NPZ path always uses its own RHF reference. All of them are signed in
`crates/ferric-python/python/ferric/ferric.pyi`.

`AoIntegrals` reports AOs in ferric's shell order. `shell_offsets()`,
`shell_atoms()` and `ao_atoms()` give the map from AO index to shell and to
atom. An AO-resolved model, one that consumes `density_matrix` or
`mo_coeffs`, needs that map to group AOs by atom.

The ASE calculator and the QCSchema runner wrap the same `run_*` calls. Both
pass the certificate through, as `calc.results["certificate"]` and
`extras["ferric"]["certificate"]`. Their options are in
`site/src/using/ase.md` and `site/src/using/qcschema.md`.

## Cube files {#python-cube-provenance-cube-files}

A voxel model, such as a 3-D convolutional network, wants a field on a
regular grid. Cube files supply that. Section @sec:export-schemas-gaussian-cube-grid describes the grid and
the file layout; this table gives the writers a consumer calls.

| Writer | Contents | Defaults |
|---|---|---|
| `write_cube(path, mol, basis, result, what="density")` | electron density, $e$/Bohr³ | spacing 0.3 Bohr, padding 4.0 Bohr |
| `write_cube(..., what="mo", mo_index=p)` | one restricted MO | the same |
| `[rpa] export_eigpot_prefix` / `export_eigpot_prefix=` | the leading PDEP mode densities, one file each | spacing 0.2 Bohr, margin 4.0 Bohr, 10 modes |

The grid is axis-aligned. Unless you give them, its origin and shape come
from the atoms' bounding box plus the padding, so two molecules of different
size get grids of different shape. `write_cube` returns the origin and shape
it used. Pass them back in (`origin=`, `shape=`) to put a series of molecules
or conformers on one grid, which a voxel model needs.

The eigenpotential cubes hold the auxiliary-basis **charge density**
$\rho_\xi(\mathbf r) = \sum_P c_{P\xi}\chi_P(\mathbf r)$ of each mode, not
the potential it generates; the eigenpotential is its Coulomb potential. A
model trained on these files is trained on densities.

## The certificate as a provenance label {#python-cube-provenance-certificate-provenance-label}

Why keep provenance per sample at all? A label is safe to train on only if
you know what it is: which code version computed it, with which
approximations, and how well that calculation has been checked. Without
that, a dataset that mixes builds or settings cannot be split or filtered
afterwards, and a bad sample cannot be traced. ferric's answer is the
certificate.

Every result carries one, and the NPZ stores it as JSON bytes. Its fields,
from `crates/ferric-core/src/certificate.rs`:

| Field | Content | Use in a dataset |
|---|---|---|
| `capability` | the CLI `method.kind` or library capability | method label |
| `build` | `version`, `commit`, `dirty` | which code produced the sample |
| `grade`, `grade_source` | the validation grade of the capability's energy and where it came from | trust filter (section @sec:reproducibility-and-grades) |
| `measured_error` | `quantity`, `value`, `units`, `reference`, `evidence`, present only when the run is inside its anchor's tested settings | error bar, when present |
| `approximations` | `(name, detail)` pairs in effect: RI and its auxiliary basis, frozen core, local cutoffs, grids, Hessian construction, inherited SCF ones prefixed `scf ` | to split or filter a dataset by setting |
| `precision` | `F64`, or `Mixed` with one record per kernel allowed to run mixed, when attached | to keep mixed-precision samples apart |
| `warnings` | warnings raised during the run, duplicates dropped | to drop suspect samples |
| `blocks`, `routes` | building-block records and exact routes, when used | composed or unusual paths |

The grade has two spellings. The JSON form (the NPZ `certificate` key, and
the `certificate` dict of a Python result) holds the serialized enum in
snake case; the printed certificate and appendix @sec:appendix-c use the
validation page's word (`Grade::as_str` in `crates/ferric-core/src/certificate.rs`).
Filter a dataset on the JSON value:

| JSON value | Printed word |
|---|---|
| `proven` | Proven |
| `proven_narrow` | Proven (narrow) |
| `smoke` | Smoke |
| `spike` | Spike |
| `stub` | Stub |
| `ungraded` | not graded |

For `pdep-rpa` the grade is capped by the reference SCF's grade, and the
reference's settings appear among the approximations with the `scf ` prefix
(`PdepRpaResult.certificate` in `ferric.pyi`).

Three cautions apply when you use the certificate as a trust label.

1. **The grade is the energy's.** A `pdep-rpa` certificate grades the
   PDEP-RPA energy. It says nothing about `hirshfeld_charges` or `c6_iso`.
   The trust level of an exported property is that of its own validation row
   (section @sec:reproducibility-and-grades).
2. **`measured_error` is often absent.** The `pdep-rpa` anchor covers frozen
   core 0, `trunc_thresh` 0 or below, and Gauss–Legendre with 6 or 40 points
   at $u_0 = 0.5$ (`site/src/reference/validation.md`, "Certificates"). The
   default 20-point grid, or any truncation, is outside that scope and gives
   no `measured_error`. Absence means "no anchor for these settings", not
   "no error".
3. **Composed results are ungraded.** `certificate_from_blocks` records the
   `AoIntegrals.provenance()` dicts a custom result was built from and always
   sets `grade` to `ungraded` (printed "not graded"), because no validation row covers the caller's
   code.

Keep the whole certificate with each sample. It is small, and it is the only
record of settings, such as the reference's J/K route, that the arrays do not
reveal.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-python/python/ferric/ferric.pyi` | the Python names, defaults and units; the `pdep-rpa` grade cap |
| `crates/ferric-python/src/blocks.rs` | the building-block classes |
| `crates/ferric-python/python/ferric/ase.py`, `crates/ferric-python/python/ferric/qcschema.py` | where the certificate is passed through |
| `crates/ferric-export/src/cube.rs`, `crates/ferric-rpa/src/property_bundle.rs` | cube contents and defaults |
| `crates/ferric-core/src/certificate.rs` | the certificate fields and grade rules |
| `site/src/reference/validation.md`, "Certificates" | the anchor scope of `pdep-rpa` |

Changes from the sources: section @sec:export-schemas-gaussian-cube-grid calls the
eigenpotential cube the potential $V_\alpha$; the cube holds the mode's
charge density (@sec:python-cube-provenance-cube-files).
