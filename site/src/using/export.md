# Export: cube files, Molden files, FCIDUMP and NPZ bundles

ferric writes volumetric data as Gaussian cube files, orbitals as Molden
files, active-space Hamiltonians as FCIDUMP files and per-molecule property bundles as compressed NumPy `.npz` archives. The Python functions
evaluate basis functions, the density and orbitals at arbitrary points; the
CLI `[rpa]` section writes the NPZ bundle and PDEP eigenpotential cubes. The
array schema is in `book/src/27-ml/01-npz-bundle.md#npz-bundle-keys`.

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

### CASSCF natural orbitals

`molden.write` also takes a `run_casscf` result. The file holds one orbital
set: the doubly occupied orbitals (occupation 2), the active orbitals
rotated to natural orbitals (eigenvectors of the active 1-RDM `dm1`, written
in descending occupation, each with its natural occupation), then the
virtuals (occupation 0). The occupations sum to the electron count, so the
active ones sum to the active electron count. `ferric.molden.casscf_natural_orbitals(result)`
returns the rotated coefficients and occupations. The `Ene=` column is
`CasScfResult.mo_energy` when the run used `canonicalize=True` and 0
otherwise.

```python
cas = ferric.run_casscf(water, bs, 4, (2, 2), canonicalize=True)
molden.write("water-cas.molden", water, bs, cas)
```

The density PySCF evaluates from the file equals
`2 C_core C_coreᵀ + C_act dm1 C_actᵀ` evaluated with ferric's AOs, to 1e-10 of
its peak (`crates/ferric-python/tests/test_fcidump_molden_cas.py`).

## FCIDUMP files

`ferric.fcidump.write(path, mol, bs, mo_coeff, n_core, n_active, nelec, ms2)`
writes the Hamiltonian of an active space in the standard FCIDUMP format
that block2 (DMRG), Dice (SHCI), PySCF (`pyscf.tools.fcidump.read`) and most
CI codes read. Orbitals `n_core .. n_core + n_active` of `mo_coeff` are
active, the first `n_core` are doubly occupied and the rest are dropped.
`nelec` is the number of ACTIVE electrons and `ms2` is
`n_alpha - n_beta`.

```python
import ferric
from ferric import fcidump

water = ferric.Molecule.from_xyz_string(
    "3\n\nO 0 0 0.11779\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n"
)
bs = ferric.BasisSet.bundled("6-31g")
cas = ferric.run_casscf(water, bs, 4, (2, 2), canonicalize=True)
fcidump.write("water.fcidump", water, bs, cas.mo_coeff, cas.n_core, 4, 4, 0)
d = fcidump.read("water.fcidump")  # norb, nelec, ms2, h, eri, ecore
```

The file is built only from ferric's AO blocks (`AoIntegrals.hcore`, `veff`,
`ao2mo`):

- `h_pq = C_pᵀ (h + J[D_c] - K[D_c]/2) C_q`, with `D_c = 2 C_core C_coreᵀ`: the
  potential of the inactive electrons is in the one-electron integrals.
- `ECORE = E_nn + Tr[D_c h] + 1/2 Tr[D_c (J[D_c] - K[D_c]/2)]`: nuclear
  repulsion plus the inactive electrons' energy. It equals
  `CasCiResult.e_core` and `CasScfResult.e_core` for the same orbitals.
- `(pq|rs)` in chemist notation, one line per 8-fold-unique element
  (`p>=q`, `r>=s`, `pq>=rs`, 1-based), `h_pq` as `val p q 0 0` and `ECORE` as
  `val 0 0 0 0`. Numbers are written in Python's shortest round-trip form.
- `ORBSYM` is all 1 and `ISYM=1`: ferric writes no point-group labels.

`fcidump.read(path)` returns the integrals expanded by the 8-fold symmetry.
`read(write(x))` returns the written elements bitwise. Solving the file with
`pyscf.fci.direct_spin1` reproduces the `e_total` of `run_casci` and
`run_casscf` on water and N2 (`test_fcidump_molden_cas.py` states the bars).
`fcidump.integrals(mol, bs, mo_coeff, n_core, n_active)` returns
`(h, eri, ecore)` without writing a file. The two-electron tensor is held
densely (`n_active^4` doubles), so very large active spaces are limited by
memory.

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
