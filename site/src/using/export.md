# Export: cube files, Molden files and NPZ bundles

ferric writes volumetric data as Gaussian cube files, orbitals as Molden
files and per-molecule property bundles as compressed NumPy `.npz` archives. The Python functions
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

## Molden files

`ferric.molden.write(path, mol, bs, result)` writes the geometry, basis and
molecular orbitals of a `run_rhf`, `run_dft`/`run_ksdft`, `run_uhf` or
`run_rohf` result as a Molden file, which Avogadro, Jmol, IBOView, Multiwfn,
JANPA and PySCF (`pyscf.tools.molden.load`) read. `mol` and `bs` are the
molecule and orbital basis the SCF ran on.

<!-- doctest -->
```python
from ferric import molden

molden.write("water.molden", water, bs, rhf)
text = open("water.molden").read()
print(text.count("Spin= Alpha"), text.count("Occup= 2.0\n"))

oh = ferric.Molecule.from_xyz_string("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2)
uhf = ferric.run_uhf(oh, bs)
molden.write("oh.molden", oh, bs, uhf)
text = open("oh.molden").read()
print(text.count("Spin= Alpha"), text.count("Spin= Beta"))
```

```text
7 5
6 6
```

What the file holds:

- Coordinates in Bohr (`[Atoms] AU`). Every number is written so that it
  reads back exactly (17 significant digits).
- Every orbital with its energy (Hartree) and occupation. A closed-shell
  result has one set with occupations 2 and 0. An open-shell result has the
  α set, then the β set. ROHF writes its one MO set twice, with the α and the
  β occupations. Occupations are read off the result's density,
  `diag(Cᵀ S D S C)` (written as the integer when within 1e-8 of one), so a
  MOM state is written as it was converged.
- Shell types: `[5D7F]`, `[5D10F]` or `[7F]` for spherical d and/or f shells,
  `[9G]` for spherical g, nothing for Cartesian. Molden has one setting per
  angular momentum and stops at g, so a basis that mixes spherical and
  Cartesian shells of one l, or has h shells, raises `ValueError`. The
  6-31G family's Cartesian d shells are written as 6D.
- **ECPs.** Molden has no standard ECP block. The file has the valence
  basis and orbitals, the charge column of `[Atoms]` is `Z` minus the core
  electrons, and a `[core]` section lists the core electrons per atom (the
  Molden2AIM extension that PySCF and Multiwfn read). Other readers skip
  `[core]` and see a valence-only molecule.
- **Ghost atoms** are written as atoms of their element with charge 0, since
  their basis functions carry orbital amplitude. Viewers draw them as atoms,
  and PySCF, which ignores the charge column, loads them as real atoms; the
  orbitals and density are unaffected.

The writer converts ferric's AO conventions (libint2's component order, and
Cartesian components sharing the normalisation of x^l) to Molden's (m = 0,
+1, -1, ... for spherical shells, a fixed component list for Cartesian ones,
each component normalised). It is checked against PySCF's reader: for RHF,
KS, UHF and ROHF results over s to g shells, spherical and Cartesian, with
an ECP and with a ghost atom, the orbitals and density PySCF evaluates from
the file match ferric's to 1e-10 of their largest value
(`crates/ferric-python/tests/test_molden.py`).

`ferric.molden.orbitals(mol, bs, result)` returns the orbital sets the file
holds (coefficients in ferric's AO order, energies, occupations), and
`UhfResult` exposes them per spin as `mo_coefficients_alpha()`,
`mo_coefficients_beta()`, `orbital_energies_alpha()`/`_beta()` and
`occupations_alpha()`/`_beta()`.

Formatted checkpoints (`.fchk`) and AIM wavefunctions (`.wfx`) are not
written by ferric. [IOData](https://github.com/theochem/iodata) converts a
Molden file to either:

```python
from iodata import load_one, dump_one

dump_one(load_one("water.molden"), "water.fchk")
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
