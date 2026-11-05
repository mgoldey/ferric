# Export: cube files and NPZ bundles

ferric writes volumetric data as Gaussian cube files and per-molecule
property bundles as compressed NumPy `.npz` archives. The Python functions
evaluate basis functions, the density and orbitals at arbitrary points; the
CLI `[rpa]` section writes the NPZ bundle and PDEP eigenpotential cubes. The
array schema is in `crates/ferric-export/theory.md`.

## Units

| Quantity | Unit |
|---|---|
| Points, origins, spacings, padding, cube grids | Bohr |
| Density | e/Bohr³ |
| Orbitals and AO values | Bohr^(-3/2) |
| NPZ `coords` | Bohr |
| NPZ energies, ESP, field | Hartree, Hartree/e, a.u. |
| NPZ polarizabilities | Bohr³ |
| NPZ charges | e |

These differ from `Molecule.from_xyz_string`, which reads Å, and from the
periodic and QM/MM Python entry points, which take Å. The functions below
take Bohr throughout.

## Cube files from Python

`write_cube` writes the total density or one restricted MO on an axis-aligned
grid. Without `origin` and `shape` the grid is the atoms' bounding box plus
`padding`. It returns the `(origin, shape)` used. `what="mo"` needs
`mo_index` (0-based, energy order) and a restricted RHF or RKS result.
Orbital signs are the SCF's arbitrary phase.

<!-- doctest -->
```python
import ferric

water = ferric.Molecule.from_xyz_string("""3
water
O   0.000000   0.000000   0.117790
H   0.000000   0.755453  -0.471161
H   0.000000  -0.755453  -0.471161
""")
bs = ferric.BasisSet.bundled("sto-3g")
rhf = ferric.run_rhf(water, bs)

origin, shape = ferric.write_cube("density.cube", water, bs, rhf, spacing=0.3)
print(shape)
origin, shape = ferric.write_cube("homo.cube", water, bs, rhf,
                                  what="mo", mo_index=4, spacing=0.3)
print(shape)
```

```text
[28, 38, 32]
[28, 38, 32]
```

## Values at points

`eval_ao_on_points`, `eval_density_on_points` and `eval_mo_on_points` take an
`(N, 3)` array of points in Bohr.

| Function | Returns |
|---|---|
| `eval_ao_on_points(mol, bs, points)` | `(N, nbasis)` AO values, PySCF `eval_gto` layout |
| `eval_density_on_points(mol, bs, result, points)` | list of N total densities (α + β) |
| `eval_mo_on_points(mol, bs, result, mo_indices, points)` | `(N, len(mo_indices))` restricted MO values |

`result` is an `RhfResult` or `DftResult`. `eval_mo_on_points` refuses
open-shell results. All three check the allocation against the memory budget
before evaluating.

<!-- doctest -->
```python
import numpy as np

pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])    # Bohr
print(np.round(ferric.eval_density_on_points(water, bs, rhf, pts), 4))
print(ferric.eval_ao_on_points(water, bs, pts).shape)
print(ferric.eval_mo_on_points(water, bs, rhf, [4], pts).shape)
```

```text
[10.2392  0.8103]
(2, 7)
(2, 1)
```

## NPZ bundles from the CLI

A `pdep-rpa` run with `[rpa] export_npz = "<path>"` writes the bundle. The
`compute_*` keys then default to `true`; the full key list and defaults are
in [Input file (TOML)](../reference/input.md#rpa).

```toml
[molecule]
xyz = "testdata/molecules/water.xyz"

[basis]
name = "cc-pvdz"

[method]
kind = "pdep-rpa"
task = "energy"

[rpa]
auxbasis = "cc-pvdz-ri"
n_quad = 20
trunc_thresh = 1e-4
export_npz = "h2o_pdep_rpa.npz"
compute_esp = true
compute_polarizability = true
```

A requested property that cannot be computed fails the run unless
`allow_partial_npz = true`. Read the archive with `numpy.load`:

```python
import numpy as np
d = np.load("h2o_pdep_rpa.npz")
print(d.files)
```

Keys written for water/cc-pVDZ with the input above (N atoms, nao = 24):

| Keys | Shape |
|---|---|
| `coords`, `atomic_numbers` | (N, 3), (N,) |
| `mo_coeffs`, `density_matrix` | (nao, nao) |
| `orbital_energies` | (nao,) |
| `pdep_eigenvectors` | (naux, M), M retained eigenvalues |
| `orbital_centers`, `orbital_spreads` | (nmo, 3), (nmo,) |
| `density_second_moment`, `dipole` | (3, 3), (3,) |
| `esp_atoms`, `electric_field` | (N,), (N, 3) |
| `esp_surface`, `esp_points` | (npts,), (npts, 3); `compute_esp_surface = true` only |
| `alpha_tensor`, `alpha_ct` | (3, 3) |
| `alpha_atomic` | (N, 3, 3) |
| `hirshfeld_charges`, `lowdin_charges`, `mulliken_charges`, `chelpg_charges`, `resp_charges` | (N,) |
| `c6_freqs`, `c6_weights` | (n_quad,) |
| `alpha_atomic_dynamic` | (N, n_quad, 3, 3) |
| `alpha_ct_dynamic` | (n_quad, 3, 3); `c6_source = "pdep"` only |
| `c6_iso`, `c6_aniso` | (N, N), (N, N, 3, 3) |
| `c6_molecular_iso` | (1,) |
| `c6_partition`, `c6_source` | byte strings; decode with `.tobytes().decode()` |

`mo_coeffs`, `orbital_energies`, `orbital_centers` and `orbital_spreads` are
written for restricted references only. `c6_iso` and `c6_aniso` are per-atom pair tensors under a partition
convention. They do not sum to the molecular C6; read `c6_molecular_iso`.
`alpha_ct + alpha_atomic.sum(0) == alpha_tensor`. The definitions of
`alpha_atomic` and `alpha_ct` are in the `compute_alpha_atomic` row of the
input reference.

## PDEP eigenpotential cubes

Setting `export_eigpot_prefix` in `[rpa]` writes `<prefix>_eigpot_NNN.cube`
for the leading `export_eigpot_count` eigenpotentials (default 10; the count
is capped at the number retained). `cube_spacing` (default 0.2 Bohr) and
`cube_margin` (default 4.0 Bohr) set the grid.
`examples/benzene-pdep-rpa-export.toml` is a complete input.
