# Export schemas {#export-schemas}

<!-- status: edit; registry: none -->

Sections @sec:polarizabilities-from-dielectric-modes and @sec:atomic-partitions define quantities. A model that learns from them sees
none of the derivations, only arrays in a file. It needs four things from that
file: the arrays indexed consistently (row $A$ of every per-atom array is atom
$A$ of `coords`), fixed units and shapes, a tag for every convention that
changes the numbers (which partition, which $\alpha(iu)$ source), and a way to
tell a missing array from a zero one. This section says how ferric's writers
provide each: which array each quantity becomes, what the writer refuses to
write, and the grid format for volumetric fields.

The per-array tables, with shapes and units, are in section @sec:npz-bundle-keys.
The consumer's reading of each array,
including its behaviour under rotation, is in sections @sec:npz-bundle to @sec:python-cube-provenance. This
section does not repeat the shapes, units or readings; the knob table below
is the one that section @sec:npz-bundle points to.

| Entry point | Location |
|---|---|
| Rust (NPZ writer) | `ferric-export/src/ml.rs::export_npz`, `ferric-export/src/ml.rs::NpzBundle` |
| Rust (cube writers) | `ferric-export/src/cube.rs::{export_cube,export_mo_cubes}` |
| CLI | `method.kind = "pdep-rpa"`, `[rpa] export_npz = "<path>"`, `[rpa] export_eigpot_prefix` |
| Python | `run_pdep_rpa(..., properties=True)` or `export_npz=`; `write_cube` (`crates/ferric-python/python/ferric/ferric.pyi`) |

## From quantity to key {#export-schemas-quantity-key}

The writer is a plain serializer. It computes nothing: the caller,
`ferric-rpa/src/property_bundle.rs::compute_pdep_properties`, builds every
array, and `export_npz` only checks that groups of arrays are consistent
before writing them. The keys fall into the groups below, by the derivation
behind each.

| Group | Keys | Defined in |
|---|---|---|
| Geometry and wavefunction | `coords`, `atomic_numbers`, `mo_coeffs`, `orbital_energies`, `density_matrix`, `boys_coeffs` | the SCF reference (chapter @sec:hartree-fock) |
| Orbital moments | `orbital_centers`, `orbital_spreads`, `density_second_moment`, `dipole` | section @sec:electrostatics-and-charges-multipoles-orbital-moments |
| Electrostatics | `esp_atoms`, `esp_surface` with `esp_points`, `electric_field` | section @sec:electrostatics-and-charges-potential-field-nuclei and @sec:electrostatics-and-charges-potential-arbitrary-points-surface |
| Atomic charges | `hirshfeld_charges`, `lowdin_charges`, `mulliken_charges`, `chelpg_charges`, `resp_charges` | section @sec:atomic-partitions |
| Molecular response | `pdep_eigenvectors`, `alpha_tensor` | section @sec:polarizabilities-from-dielectric-modes |
| Per-atom response | `alpha_atomic`, `alpha_ct`, `alpha_atomic_dynamic`, `alpha_ct_dynamic` | sections @sec:polarizabilities-from-dielectric-modes, @sec:atomic-partitions-partition-unity-buys-polarizability and @sec:polarizability-pdep-c6 |
| Dispersion | `c6_partition`, `c6_source`, `c6_freqs`, `c6_weights`, `c6_iso`, `c6_aniso`, `c6_molecular_iso` | chapter @sec:dispersion and section @sec:polarizability-pdep-c6 |
| Provenance | `certificate` | section @sec:python-cube-provenance-certificate-provenance-label |

**Units and shapes.** Every number is in atomic units: coordinates, centroids
and spreads in Bohr, orbital energies in Hartree, charges in $e$, the dipole
in $e\,a_0$, the density second moment in Bohr², potentials in Hartree per $e$, polarizabilities in Bohr³,
frequencies, quadrature weights and $C_6$ in a.u. The shapes follow one
pattern. A per-atom array has the atom index first, in the order of `coords`:
$(N_{\rm atoms},)$ for a charge, $(N_{\rm atoms},3)$ for a field,
$(N_{\rm atoms},3,3)$ for `alpha_atomic`. A frequency-resolved array puts the
frequency index next, in the order of `c6_freqs`:
`alpha_atomic_dynamic` is $(N_{\rm atoms},N_u,3,3)$ and `alpha_ct_dynamic`
is $(N_u,3,3)$. A pair array has two atom indices: `c6_iso` is
$(N_{\rm atoms},N_{\rm atoms})$ and `c6_aniso` is
$(N_{\rm atoms},N_{\rm atoms},3,3)$. Cartesian $3\times3$ blocks are always
the last two axes.

**Which keys are present.** Every key is optional; an absent array means the
property was not requested or not computed. The CLI
(`kind = "pdep-rpa"`, `[rpa] export_npz = "<path>"`) turns the `compute_*`
knobs on by default. Each key is written under the knob or condition in this
table.

| Keys | TOML knob / condition |
| :--- | :--- |
| `coords`, `atomic_numbers`, `pdep_eigenvectors`, `certificate` | always (`certificate` is the PDEP-RPA certificate the CLI prints) |
| `mo_coeffs`, `orbital_energies`, `orbital_centers`, `orbital_spreads` | restricted reference only |
| `density_matrix`, `density_second_moment` | `compute_density_matrix` |
| `dipole` | `compute_dipole` |
| `esp_atoms` | `compute_esp` |
| `esp_surface`, `esp_points` | `compute_esp_surface` (default `false`); `esp_surface_vdw_scale`, `esp_surface_n_angular` |
| `electric_field` | `compute_electric_field` |
| `alpha_tensor` | `compute_polarizability` |
| `alpha_atomic` | `compute_alpha_atomic` |
| `alpha_ct` | `compute_polarizability` and `compute_alpha_atomic` |
| `hirshfeld_charges`, `lowdin_charges`, `mulliken_charges`, `chelpg_charges`, `resp_charges` | `compute_hirshfeld_charges`, `compute_lowdin_charges`, `compute_mulliken_charges`, `compute_chelpg_charges`, `compute_resp_charges` |
| `c6_partition`, `c6_source`, `c6_freqs`, `c6_weights`, `alpha_atomic_dynamic`, `c6_iso`, `c6_aniso`, `c6_molecular_iso` | `compute_c6`; all of them are written, or none. `c6_source` defaults to `ts`; `c6_partition` defaults to `hirshfeld` for `pdep`, `becke` for `ts`/`mbd` |
| `alpha_ct_dynamic` | `compute_c6` with `c6_source = "pdep"` |
| `boys_coeffs` | library only: the CLI never writes it |

## What the writer enforces {#export-schemas-writer-enforces}

`export_npz` refuses some combinations and encodes some types in a fixed way.
These rules come from the file format and from what a consumer can
interpret, not from the physics.

| Rule | Behaviour | Why |
|---|---|---|
| Surface ESP travels with its points | `esp_surface` without `esp_points`, or the reverse, or a length mismatch, is an error | values without their coordinates are unusable |
| The dispersion block is all or nothing | `c6_partition`, `c6_source`, the per-atom arrays and `c6_molecular_iso` are written from one optional group; a `alpha_ct_dynamic` whose length differs from `c6_freqs` is an error before any key is written | a per-atom $C_6$ with no partition attached cannot be interpreted |
| Strings are bytes | `c6_partition`, `c6_source` and `certificate` are `uint8` arrays of UTF-8; decode with `.tobytes().decode()` | the numpy writer in use supports numeric element types only |
| Dtypes | `atomic_numbers` is `int64`; every other numeric array is `float64` | one floating dtype for every real array |
| The molecular $C_6$ is a length-1 array | read `c6_molecular_iso[0]` | the format has no scalar |
| Remainder identity | `alpha_ct + alpha_atomic.sum(0) == alpha_tensor`; the caller supplies `alpha_ct` exactly when it supplies both operands, and the writer does not check the sum | the remainder is computed from the same response as both |

The dispersion rule is enforced by the type, not by a runtime test. The
per-atom $C_6$ arrays and their provenance (`C6Provenance`: partition and
source) sit in one struct, `C6Export`, behind one `Option` in
`DispersionBundle`. A caller cannot supply the arrays and leave the tag out,
so no file can hold an untagged per-atom $C_6$. The strings are bytes because
`ndarray-npy` writes only numeric and `bool` elements; an integer enum code
would have made the consumer carry ferric's discriminant table separately.

Two of these rules, and one pair of keys, carry a warning that is easy to
miss.

- **`c6_iso.sum()` is not the molecular $C_6$.** The per-atom pair tensors use
  atom-centred operators and exclude inter-atomic charge transfer, which the
  molecular response includes (the $\alpha_{\rm CT}$ of @sec:atomic-partitions-partition-unity-buys-polarizability). The code
  comment on `C6Export` records differences of tens of percent on water
  (water/aug-cc-pVDZ/HF: $-19.5\%$ Hirshfeld, $-46.7\%$ Becke), with a
  bounded regression test (`ferric-rpa/tests/s9_per_atom_c6_consistency.rs`).
  The observable is `c6_molecular_iso`.
- **`esp_surface` is not `esp_atoms` at the nuclei.** The surface values keep
  every nucleus' $Z_A/r$ term and so diverge at a nucleus. `esp_atoms` leaves
  out the nucleus it is evaluated at (section @sec:electrostatics-and-charges-potential-field-nuclei).
- **`alpha_atomic` and `alpha_atomic_dynamic` can use different partitions.**
  `alpha_atomic` and `alpha_ct` are always Becke. With
  `c6_partition = "hirshfeld"` (the `pdep` default) the dynamic arrays are
  Hirshfeld, so `alpha_atomic_dynamic` at the smallest frequency does not
  continue to `alpha_atomic`, and `alpha_ct` is not the $u\to0$ limit of
  `alpha_ct_dynamic`. The example below writes such a file.

The charge arrays carry no partition tag, and the tag the file does carry
covers only the dispersion block, not the static `alpha_atomic`: the scheme is
the key name, and the lattice parameters of CHELPG and
RESP are not recorded (section @sec:atomic-partitions-esp-fitted-charges).

## The Gaussian cube grid {#export-schemas-gaussian-cube-grid}

Volumetric fields (the density, one molecular orbital, the PDEP mode charge
densities) are written as Gaussian cube files by
`ferric-export/src/cube.rs::export_cube`. A cube file is a regular lattice of
values plus a header that places it in space, which every molecular viewer
reads.

**The grid.** `GridSpec` (`ferric-integrals/src/ao_grid.rs::GridSpec`) is an
axis-aligned uniform lattice stored as:

- `origin`: the coordinate $[x_0, y_0, z_0]$ (Bohr) of grid point $(0,0,0)$;
- `n_x`, `n_y`, `n_z`: the number of points along each axis;
- `step_x`, `step_y`, `step_z`: the step vectors (Bohr). The evaluators read
  only their diagonal entries, so the steps must be axis-aligned.

`GridSpec::bounding_box(mol, margin, spacing)` builds one from the atoms. The
origin is the lower corner of the atoms' bounding box moved out by `margin`.
Each axis length is the bounding-box extent plus twice `margin`, and each
count is that length divided by `spacing`, rounded up. The callers choose the
numbers, so there is no single default: `write_cube` takes a spacing of 0.3
Bohr and a padding of 4.0 Bohr, and the PDEP eigenpotential export takes 0.2
and 4.0 (`[rpa] cube_spacing`, `cube_margin`).

**The file.** `export_cube` writes two comment lines (the first is
`Ferric generated cube file`, the second the caller's comment), then the atom
count and the origin, then one line per axis with its point count and step
vector, then one line per atom (atomic number, the same number as a float,
and the position in Bohr). The values follow with the $x$ index outermost and
the $z$ index innermost, six per line, starting a new line after each
$(x, y)$ row.

**The value at a voxel.** A field expanded in a basis is evaluated at each
lattice point $\mathbf r = [x_0 + i h_x, y_0 + j h_y, z_0 + k h_z]$ by
summing the basis functions there. For a PDEP mode with auxiliary
coefficients $c_\xi^P$ the value is

$$ \rho_\xi(\mathbf r) = \sum_P c_\xi^P \chi_P(\mathbf r), $$

with $\chi_P$ the auxiliary basis function. As section @sec:polarizabilities-from-dielectric-modes-mode-form says, this is
the mode's charge density; the eigenpotential is its Coulomb potential, and
the cube holds the density (`[rpa] export_eigpot_prefix` writes
`<prefix>_eigpot_NNN.cube`, `export_eigpot_count` of them, default 10).

`export_mo_cubes` evaluates the atomic orbitals on the grid once and reuses
them for every requested orbital, so the cost is one
$O(N_{\rm bf}N_{\rm pts})$ evaluation plus one matrix-vector product per
orbital. Files are named `{prefix}_mo{i:03}.cube`. The Python writer, its
defaults and the use of a shared origin and shape across a dataset are in
section @sec:python-cube-provenance-cube-files.

## Reading a file back {#export-schemas-reading-file-back}

A minimal reader needs only numpy. The snippet below writes the bundle for
water / cc-pVDZ with PDEP $C_6$, then reads it back: every key with its shape
and dtype, the two string tags, the molecular $C_6$, the remainder identity
and the certificate's keys. Output from the `589b50832` build, with the
middle of the key list cut at `...` (30 keys in all):

```python
import json
import numpy as np
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

path = "ch21-water.npz"
ferric.run_pdep_rpa(mol, bs, aux, export_npz=path, compute_c6=True, c6_source="pdep")

with np.load(path) as f:
    for k in sorted(f.files):
        print(f"{k:22s} {str(f[k].shape):18s} {f[k].dtype}")
    print("c6_partition:", f["c6_partition"].tobytes().decode())
    print("c6_source   :", f["c6_source"].tobytes().decode())
    print("c6_molecular_iso:", f["c6_molecular_iso"][0])
    print("c6_iso.sum() (not the molecular C6):", f["c6_iso"].sum())
    if "alpha_ct" in f:
        d = f["alpha_ct"] + f["alpha_atomic"].sum(0) - f["alpha_tensor"]
        print("max |alpha_ct + alpha_atomic.sum(0) - alpha_tensor|:", np.abs(d).max())
    cert = json.loads(f["certificate"].tobytes().decode())
    print("certificate keys:", sorted(cert))
```

```text
alpha_atomic           (3, 3, 3)          float64
alpha_atomic_dynamic   (3, 20, 3, 3)      float64
alpha_ct               (3, 3)             float64
alpha_ct_dynamic       (20, 3, 3)         float64
alpha_tensor           (3, 3)             float64
...
orbital_centers        (24, 3)            float64
orbital_energies       (24,)              float64
orbital_spreads        (24,)              float64
pdep_eigenvectors      (84, 60)           float64
resp_charges           (3,)               float64
c6_partition: hirshfeld
c6_source   : pdep
c6_molecular_iso: 11.51382707982529
c6_iso.sum() (not the molecular C6): 5.940022214949585
max |alpha_ct + alpha_atomic.sum(0) - alpha_tensor|: 0.0
certificate keys: ['approximations', 'build', 'capability', 'grade', 'grade_source', 'measured_error', 'precision', 'routes', 'warnings']
```

Every key is optional, so a consumer tests membership (`"alpha_ct" in f`)
rather than assuming a layout.

## Known limits (2026-11-15) {#export-schemas-known-limits-2026-11}

- The bundle is built from a closed-shell RHF reference only (section @sec:npz-bundle).
- `boys_coeffs` is a library key: the CLI never writes it.
- The writer does not check that the arrays of a bundle belong to the same
  molecule or the same run. Only the pairs named in @sec:export-schemas-writer-enforces are cross-checked.
- The schema has no version field. The certificate records the build, and
  that is the only provenance in the file.

## Sources {#export-schemas-sources}

| Source | Used for |
|---|---|
| section @sec:npz-bundle-keys | the key tables |
| `crates/ferric-export/src/ml.rs` | the writer, its refusals and its dtypes |
| `crates/ferric-export/src/cube.rs` | the cube layout |
| `crates/ferric-integrals/src/ao_grid.rs::GridSpec` | the grid struct and `bounding_box` |
