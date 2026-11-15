# 21.3 Export schemas

<!-- section: 21.3; status: edit; registry: none;
     theory: crates/ferric-export/theory.md#"Enabling keys", "2. Gaussian Cube Real-Space Grid Exporter" -->

Sections 21.1 and 21.2 define quantities. This section says how they leave the
program: which array each quantity becomes, what the writer refuses to write,
and the grid format for volumetric fields. The per-array tables, with shapes
and units, are the schema of record in `crates/ferric-export/theory.md`; the
consumer's reading of each array, including its behaviour under rotation, is
in sections 27.1 to 27.4. This section does not repeat them.

| Entry point | Location |
|---|---|
| Rust (NPZ writer) | `ferric-export/src/ml.rs::export_npz`, `ferric-export/src/ml.rs::NpzBundle` |
| Rust (cube writers) | `ferric-export/src/cube.rs::{export_cube,export_mo_cubes}` |
| CLI | `method.kind = "pdep-rpa"`, `[rpa] export_npz = "<path>"`, `[rpa] export_eigpot_prefix` |
| Python | `run_pdep_rpa(..., properties=True)` or `export_npz=`; `write_cube` (`crates/ferric-python/ferric.pyi`) |

## 21.3.1 From quantity to key

The writer is a plain serializer. It computes nothing: every array is built by
the caller, in `ferric-rpa/src/property_bundle.rs::compute_pdep_properties`,
and `export_npz` only checks that groups of arrays are consistent before
writing them. The keys fall into the groups below, by the derivation behind
each.

| Group | Keys | Defined in |
|---|---|---|
| Geometry and wavefunction | `coords`, `atomic_numbers`, `mo_coeffs`, `orbital_energies`, `density_matrix`, `boys_coeffs` | the SCF reference (chapter 7) |
| Orbital moments | `orbital_centers`, `orbital_spreads`, `density_second_moment`, `dipole` | section 27.2.3 |
| Electrostatics | `esp_atoms`, `esp_surface` with `esp_points`, `electric_field` | section 27.2.1 and 27.2.2 |
| Atomic charges | `hirshfeld_charges`, `lowdin_charges`, `mulliken_charges`, `chelpg_charges`, `resp_charges` | section 21.2 |
| Molecular response | `pdep_eigenvectors`, `alpha_tensor` | section 21.1 |
| Per-atom response | `alpha_atomic`, `alpha_ct`, `alpha_atomic_dynamic`, `alpha_ct_dynamic` | sections 21.1, 21.2.3 and 27.3 |
| Dispersion | `c6_partition`, `c6_source`, `c6_freqs`, `c6_weights`, `c6_iso`, `c6_aniso`, `c6_molecular_iso` | chapter 25 and section 27.3 |
| Provenance | `certificate` | section 27.4.3 |

Which keys a run contains depends on the `compute_*` knobs. The next block
is the theory file's table of knobs and conditions.

<!-- include: crates/ferric-export/theory.md#"Enabling keys" -->

## 21.3.2 What the writer enforces

`export_npz` refuses some combinations and encodes some types in a fixed way.
These are properties of the file format, not of the physics.

| Rule | Behaviour | Why |
|---|---|---|
| Surface ESP travels with its points | `esp_surface` without `esp_points`, or the reverse, or a length mismatch, is an error | values without their coordinates are unusable |
| The dispersion block is all or nothing | `c6_partition`, `c6_source`, the per-atom arrays and `c6_molecular_iso` are written from one optional group; a `alpha_ct_dynamic` whose length differs from `c6_freqs` is an error before any key is written | a per-atom $C_6$ with no partition attached cannot be interpreted |
| Strings are bytes | `c6_partition`, `c6_source` and `certificate` are `uint8` arrays of UTF-8; decode with `.tobytes().decode()` | the numpy writer in use supports numeric element types only |
| Dtypes | `atomic_numbers` is `int64`; every other numeric array is `float64` | one floating dtype for every real array |
| The molecular $C_6$ is a length-1 array | read `c6_molecular_iso[0]` | the format has no scalar |
| Remainder identity | `alpha_ct + alpha_atomic.sum(0) == alpha_tensor`; the caller supplies `alpha_ct` exactly when it supplies both operands, and the writer does not check the sum | the remainder is computed from the same response as both |

Two of these carry a warning that is easy to miss.

- **`c6_iso.sum()` is not the molecular $C_6$.** The per-atom pair tensors use
  atom-centred operators and exclude inter-atomic charge transfer, which the
  molecular response includes. The code comment on `C6Export` records
  differences of tens of percent on water, with a bounded regression test
  (`ferric-rpa/tests/s9_per_atom_c6_consistency.rs`). The observable is
  `c6_molecular_iso`.
- **`esp_surface` is not `esp_atoms` at the nuclei.** The surface values keep
  every nucleus' $Z_A/r$ term and so diverge at a nucleus. `esp_atoms` leaves
  out the nucleus it is evaluated at (section 27.2.1).

The charge arrays carry no partition tag, unlike the per-atom polarizability
arrays; the scheme is the key name, and the lattice parameters of CHELPG and
RESP are not recorded (section 21.2.5).

## 21.3.3 The Gaussian cube grid

Volumetric fields (the density, one molecular orbital, the PDEP eigenpotential
charge densities) are written as Gaussian cube files by
`ferric-export/src/cube.rs::export_cube`. The grid is an axis-aligned lattice
described by `GridSpec`. The text below is the theory file's description of
the grid and the value at each point.

<!-- include: crates/ferric-export/theory.md#"2. Gaussian Cube Real-Space Grid Exporter" -->

`export_mo_cubes` evaluates the atomic orbitals on the grid once and reuses
them for every requested orbital, so the cost is one
$O(N_{\rm bf}N_{\rm pts})$ evaluation plus one matrix-vector product per
orbital. Files are named `{prefix}_mo{i:03}.cube`. The Python writer, its
defaults and the use of a shared origin and shape across a dataset are in
section 27.4.2.

## 21.3.4 Reading a file back

A minimal reader needs only numpy.

```python
import json
import numpy as np

with np.load("water.npz") as f:
    q = f["lowdin_charges"]                      # (N_atoms,), e
    alpha = f["alpha_tensor"]                    # (3, 3), Bohr^3
    partition = f["c6_partition"].tobytes().decode()
    c6 = f["c6_molecular_iso"][0]                # the observable, a.u.
    cert = json.loads(f["certificate"].tobytes().decode())
```

Every key is optional, so a consumer tests membership (`"alpha_ct" in f`)
rather than assuming a layout.

## 21.3.5 Known limits (2026-11-15)

- The bundle is built from a closed-shell RHF reference only (section 27.1).
- `boys_coeffs` is a library key: the CLI never writes it.
- The writer does not check that the arrays of a bundle belong to the same
  molecule or the same run. Only the pairs named in 21.3.2 are cross-checked.
- The schema has no version field. The certificate records the build, and
  that is the only provenance in the file.

## 21.3.6 Sources

| Source | Used for |
|---|---|
| `crates/ferric-export/theory.md` | the key tables, the enabling knobs and the cube grid |
| `crates/ferric-export/src/ml.rs` | the writer, its refusals and its dtypes |
| `crates/ferric-export/src/cube.rs` | the cube layout |
| `crates/ferric-integrals/src/ao_grid.rs::GridSpec` | the grid struct and `bounding_box` |

Changes to `theory.md` made with this section: the `GridSpec` fields are the
origin, the counts and the step vectors (the old text listed `spacing` and
`margin` as fields); the per-entry-point spacing and padding defaults; the
cube file layout; and the `esp_surface` row now says it includes every
nucleus' $Z_A/r$ term.
