# Training data for ML potentials

`ferric.mlip` exports geometries, energies and forces as extended XYZ and as
an [ASE database](https://wiki.fysik.dtu.dk/ase/ase/db/db.html), the two
formats that machine-learned interatomic potential trainers read. It converts
results ferric already computed; it runs nothing itself.

ASE is optional. Building the frames needs only numpy; the writers import ASE
when called and raise `ImportError` ("pip install ase") without it. Install it
with `pip install "ferric[ase]"` or `pip install ase`.

```python
import ferric.qcschema as qcs
from ferric.mlip import write_asedb, write_extxyz

results = [
    qcs.run({
        "schema_name": "qcschema_input",
        "schema_version": 1,
        "molecule": {"symbols": ["H", "H"], "geometry": [0, 0, 0, 0, 0, z]},
        "driver": "gradient",
        "model": {"method": "hf", "basis": "6-31g"},
        "keywords": {},
    })
    for z in (1.2, 1.4, 1.6)
]
write_extxyz("train.xyz", results)    # 3 frames
write_asedb("train.db", results)      # appended to the database
```

## Inputs

One input or an iterable of them:

* A QCSchema `AtomicResult` dict from `ferric.qcschema.run` with
  `driver="gradient"`. A failed result, an energy-only result and a molecule
  with ghost atoms raise `ValueError`: none is a usable label.
* An ASE `Atoms` whose calculator is a `ferric.ase.FerricCalculator` with
  energy and forces computed. Any other calculator raises `ValueError`,
  because the level of theory could not be recorded.

`ferric.mlip.frames(results)` returns the normalised `Frame` list (and needs no
ASE for QCSchema inputs); `to_atoms(results)` returns ASE `Atoms` with the
energy and forces in a `SinglePointCalculator`.

## Units

The units are stated in the data, not inherited from the input.

| Quantity | Unit | From QCSchema |
|---|---|---|
| positions | Angstrom | Bohr times `BOHR_TO_ANGSTROM` |
| energy | eV | Hartree times `HARTREE_TO_EV` |
| forces | eV/Angstrom | minus the gradient, times `HARTREE_TO_EV / BOHR_TO_ANGSTROM` |

The factors are `ferric.units`' (27.211386245988 eV per Hartree, 0.52917721092
Angstrom per Bohr), the same numbers the Rust crates use; `ase.units` differs
from them by about 8e-9 relative (Hartree) and 7e-10 (Bohr). A gradient of 1 Hartree/Bohr is a force of
-51.42208 eV/Angstrom. Every frame carries `energy_unit`, `forces_unit` and
`positions_unit`.

## Metadata

Each frame carries flat keys on the extended-XYZ comment line and as
key-value pairs in the database, so `ase db train.db qc_method=hf` selects by
level of theory.

| Key | Meaning |
|---|---|
| `qc_method`, `qc_basis` | `model.method` and `model.basis` (for an `Atoms` input, the calculator's `method` and `basis`) |
| `qc_functional` | The XC functional, Kohn-Sham methods only |
| `qc_dispersion` | The `dispersion` keyword (or calculator option), when set |
| `qc_charge`, `qc_multiplicity` | Total charge and spin multiplicity |
| `ferric_version` | The version of the ferric that made the result |
| `ferric_git_sha`, `ferric_dirty` | The build stamp, `ferric.__build__` (from the result's `provenance.build` for a QCSchema result). Without a stamp, `ferric_git_sha` is `"unknown"` and `ferric_dirty` is absent |
| `energy_unit`, `forces_unit`, `positions_unit` | The units above |

For an `Atoms` input the version and stamp are those of the running ferric.
