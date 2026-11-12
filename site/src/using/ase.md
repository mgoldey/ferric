# ASE calculator

`ferric.ase.FerricCalculator` is an
[ASE](https://wiki.fysik.dtu.dk/ase/) calculator. Any ASE tool that takes a
calculator can run on ferric's surfaces with it: the optimizers in
`ase.optimize`, vibrations, NEB, molecular dynamics, and the programs that
accept an ASE calculator (geomeTRIC's `--engine ase`, Sella, i-PI through
`ase.calculators.socketio`).

ASE is optional. `pip install ferric` does not install it and `import ferric`
does not import it; `import ferric.ase` needs ASE (`pip install "ferric[ase]"`
or `pip install ase`).

```python
from ase.build import molecule
from ase.optimize import BFGS
from ferric.ase import FerricCalculator

atoms = molecule("H2O")
atoms.calc = FerricCalculator(method="rhf", basis="6-31g")
BFGS(atoms).run(fmax=0.01)
print(atoms.get_potential_energy())          # eV
print(atoms.calc.results["certificate"])     # ferric's validation certificate
```

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `method` | `"hf"` | See the table below |
| `basis` | `"sto-3g"` | A bundled basis name or a `ferric.BasisSet` |
| `functional` | `None` | XC functional, required by the Kohn-Sham methods |
| `auxbasis` | `None` | RI basis, required by `rimp2` / `oo-rimp2`; for a periodic cell it is passed as `auxbasis` (with `options={"jk": "rsgdf"}`) |
| `charge`, `multiplicity` | `0`, `1` | Electron count and spin |
| `kpts` | `None` | Periodic only: `None` is the Gamma point, `(n1, n2, n3)` a k-point mesh |
| `options` | `None` | A dict passed unchanged to the ferric driver as keyword arguments (`density_conv`, `solvent`, `point_charges`, `jk`, `omega`, ...) |

An unknown parameter raises `CalculatorSetupError`; driver keywords go in
`options`, where the driver's own checks apply.

| `method` | Molecule (`atoms.pbc` all False) | 3-D cell (`atoms.pbc` all True) |
|---|---|---|
| `hf` | `run_rhf` for a singlet, else `run_uhf` | `run_rhf_*` / `run_uhf_*` |
| `rhf`, `uhf`, `rohf` | `run_rhf`, `run_uhf`, `run_rohf` | `run_<method>_gamma` / `_kpts` |
| `dft`, `rks` | `run_dft` (closed shell) | `run_rks_*` (singlet) or `run_uks_*` |
| `uks`, `roks` | not available | `run_uks_*`, `run_roks_*` |
| `rimp2`, `oo-rimp2` | `run_rimp2`, `run_oo_rimp2` | not available |

A cell periodic in one or two directions raises `CalculatorSetupError`:
ferric's periodic code is 3-D only.

## Properties and units

`implemented_properties` are `energy`, `free_energy` (the same number),
`forces` and `stress`. Units are ASE's: Angstrom, eV, eV/Angstrom and
eV/Angstrom^3, converted from ferric's Hartree and Bohr with
`ase.units.Hartree` and `ase.units.Bohr`. The stress is ferric's
`sigma = (1/V) dE/d(strain)` (strain of the cell and the atoms), symmetrised
and in ASE's Voigt order (xx, yy, zz, yz, xz, xy). Stress on a molecule raises
`PropertyNotImplementedError`.

Forces and stress are the analytic derivatives of the reported energy (the
`with_gradient=True` / `with_stress=True` paths of the drivers). Their
refusals carry over: for example `run_rhf` refuses a gradient with
`symmetry=True`, `fock_hook`, `smearing_sigma` or `mom_after_iter`, and
`run_rimp2` refuses it for an open shell, a local model or a fitted reference
SCF. The docstrings in `ferric.pyi` list each driver's rules.

The molecular SCF and Kohn-Sham drivers run with `ri_j_precision="f64"` unless
`options` sets it, so energies and forces come from the same surface. The f32
RI-J default of energy-only `run_rhf` / `run_dft` calls gives an energy surface
that is jagged between nearby geometries.

`calc.results["certificate"]` holds the certificate of the run
(`result.certificate`), and `calc.ferric_result` the ferric result object
itself.

## Caching

The ASE base class caches results: asking again for a property of unchanged
atoms returns the stored value. Moving the atoms, or changing a parameter with
`calc.set(...)`, clears the results; `calc.reset()` clears them by hand.
A property is computed only when asked for. `get_forces()` returns the energy
as well, so an optimizer runs one SCF per geometry, but calling
`get_potential_energy()` first and `get_forces()` second runs two.

Each geometry runs a fresh SCF from the default guess; no density is carried
between steps.

## Accuracy

The force on each atom equals ferric's analytic gradient times
`-Hartree/Bohr` exactly, and matches a central finite difference of the
calculator's own energy to the bar derived in
`crates/ferric-python/tests/test_ase_calculator.py`; the periodic stress
matches ASE's numerical stress in the same file. For finite-difference work
tighten the SCF through `options` (`density_conv=1e-9`; `scf_density_conv`
for the MP2 drivers): the analytic gradient assumes a stationary SCF, so its
error follows the SCF's residual.

Molecular dynamics through ASE has not been validated: per-step SCF cost and
energy drift over long trajectories were not measured.
