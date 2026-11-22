# ferric Python Bindings API Guide

This directory contains the PyO3 Python bindings for the `ferric` quantum chemistry engine. The compiled library exposes the Rust kernels to Python, returning inputs and outputs as native Python types and `numpy` arrays.

---

## 1. Class Reference

### `ferric.Molecule`
Represents the molecular system, storing atoms and coordinates.

* **`from_xyz(path: str) -> Molecule`** *(Static Method)*:
  Loads geometry from a standard `.xyz` file. Coordinates are converted internally to Bohr.
* **`from_xyz_string(s: str, charge: int = 0, multiplicity: int = 1) -> Molecule`** *(Static Method)*:
  Parses geometry directly from a string buffer.
* **`nuclear_repulsion(self) -> float`**:
  Returns the nuclear-nuclear repulsion energy (Hartree).
* **`natoms(self) -> int`**:
  Returns the number of atoms in the molecule.
* **`nelec(self) -> int`**:
  Returns the total number of electrons.

---

### `ferric.BasisSet`
Represents basis sets definitions used in orbital and auxiliary fitting.

* **`bundled(name: str) -> BasisSet`** *(Static Method)*:
  Retrieves a bundled basis set from the repository assets. Examples: `"sto-3g"`, `"cc-pvdz"`, `"cc-pvdz-ri"`, `"def2-svp"`, `"def2-svp-rifit"`.

---

### `ferric.RhfResult`
Returned by Restricted Hartree-Fock calculations.

* **`energy`** *(float)*: Total converged energy (electronic + nuclear repulsion) in Hartree.
* **`converged`** *(bool)*: Whether SCF iterations converged.
* **`iterations`** *(int)*: Number of iterations performed.
* **`density(self) -> numpy.ndarray`**:
  Returns the converged AO-basis density matrix as a 2D float64 `numpy` array of shape `(nbasis, nbasis)`.
* **`orbital_energies(self) -> numpy.ndarray`**:
  Returns the orbital energies (Fock matrix eigenvalues) as a 1D float64 `numpy` array of shape `(nbasis,)`.

---

### `ferric.PdepRpaResult`
Returned by PDEP-RPA calculations.

* **`total_energy`** *(float)*: Converged SCF energy + RPA correlation energy (Hartree).
* **`rhf_energy`** *(float)*: Reference SCF energy (Hartree).
* **`e_rpa`** *(float)*: RPA correlation energy (Hartree, negative).
* **`n_eigenpotentials`** *(int)*: Number of eigenpotentials retained after static truncation.
* **`eigenvalues_static`** *(numpy.ndarray)*:
  Static dielectric eigenvalues $\lambda_\alpha(0)$ as a 1D float64 `numpy` array of shape `(M,)` (sorted descending).
* **`eigenpotentials`** *(numpy.ndarray)*:
  PDEP eigenpotential coefficients in the auxiliary basis (shape `(naux, M)`): column $\alpha$ holds $c_\alpha^P$ with $V_\alpha(r) = \sum_P c_\alpha^P \chi_P(r)$. The columns are orthonormal in the auxiliary Coulomb metric $V_{PQ} = (P|Q)$ ($c^\top V c = I$), not in the plain dot product, and $c^\top (V + \Pi) c = \mathrm{diag}(\lambda_\alpha(0))$ with $\Pi$ the static polarizability in the auxiliary basis.
* **`quad_freqs`** *(numpy.ndarray)*:
  Imaginary frequency points $\omega_k$ used in the integration.
* **`quad_weights`** *(numpy.ndarray)*:
  Quadrature weights $w_k$ used in the integration.
* **`eigenvalues_freq`** *(numpy.ndarray)*:
  Frequency-dependent eigenvalues $\lambda_\alpha(i\omega_k)$ as a 2D float64 `numpy` array of shape `(N_quad, M)`.
* **`save_scree_plot(self, path: str, title: str = None)`**:
  Saves a log-scale plot of the static dielectric eigenvalue deviations $|\lambda_\alpha(0) - 1|$ against $\alpha$ to the specified path. Requires `matplotlib` to be installed in the Python environment.

---

## 2. Solver Functions

The optional `k_builder` argument accepted by the RHF-based solvers takes
`"direct"` (default), `"link"` (LinK, exact) or `"cosx"` (seminumerical
exchange, Coulomb operator only; uses the CLI defaults for grid, overlap fit
and kernel; those knobs are not exposed in Python). All three are
honoured by the RHF, UHF and ROHF solvers. See the book's *Choosing how
exchange is built* for when each pays.

* **`run_rhf(mol: Molecule, basis_set: BasisSet, max_iter: int = None, energy_conv: float = None, k_builder: str = None) -> RhfResult`**:
  Runs a Restricted Hartree-Fock calculation.
* **`run_optimize(mol: Molecule, basis_name: str, max_steps: int = None, e_conv: float = None) -> OptimizeResult`**:
  Runs a BFGS geometry optimization using analytical RHF nuclear gradients.
* **`run_rimp2(mol: Molecule, basis_set: BasisSet, auxbasis: BasisSet, frozen_core: int = None, k_builder: str = None) -> RiMp2Result`**:
  Runs a Resolution-of-Identity MP2 calculation.
* **`run_laplace_mp2(mol: Molecule, basis_set: BasisSet, auxbasis: BasisSet, n_quad: int = None, frozen_core: int = None, k_builder: str = None) -> LaplaceMp2Result`**:
  Runs a Laplace-transformed RI-MP2 calculation.
* **`run_attenuated_rimp2(mol: Molecule, basis_set: BasisSet, auxbasis: BasisSet, omega: float = None, frozen_core: int = None, k_builder: str = None) -> AttenuatedMp2Result`**:
  Runs an erfc-attenuated RI-MP2 calculation. `omega` is the range-separation parameter in Å⁻¹ (default 0.420); the MP2 operator is erfc(ω·r)/r.
* **`run_scs_mp2(mol: Molecule, basis_set: BasisSet, auxbasis: BasisSet, c_os: float = None, c_ss: float = None, frozen_core: int = None, k_builder: str = None) -> ScsMp2Result`**:
  Runs Spin-Component Scaled MP2.
* **`run_ccsd_t(mol: Molecule, basis_set: BasisSet, auxbasis: BasisSet, frozen_core: int = None, k_builder: str = None) -> CcResult`**:
  Runs Coupled Cluster CCSD(T) correlation calculations.
* **`run_eom_ccsd(mol: Molecule, basis_set: BasisSet, auxbasis: BasisSet, n_roots: int = 5, conv_tol: float = 1e-6, max_iter: int = 100, max_space: int = None, frozen_core: int = None, k_builder: str = None, memory_budget_gb: float = None, ladder: str = "auto") -> EomCcsdResult`**:
  Runs EE-EOM-CCSD singlet excitation energies (right eigenvectors) on the spin-adapted closed-shell CCSD.
* **`run_pdep_rpa(mol: Molecule, basis_set: BasisSet, auxbasis: BasisSet, frozen_core: int = None, n_quad: int = None, quadrature: str = None, u0: float = None, trunc_thresh: float = None, eigensolver_conv_thresh: float = None, run_diagnostics: bool = False, k_builder: str = None) -> PdepRpaResult`**:
  Runs PDEP-RPA calculations.

---

## 3. Basic Example

```python
import ferric

# 1. Load molecule
mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")

# 2. Load orbital and auxiliary bases
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

# 3. Run PDEP-RPA
rpa = ferric.run_pdep_rpa(
    mol, bs, aux, 
    trunc_thresh=1e-4, 
    quadrature="minimax", 
    n_quad=30
)

print(f"Converged RPA total: {rpa.total_energy:.8f} Ha")
print(f"RPA correlation:     {rpa.e_rpa:.8f} Ha")
print(f"Kept eigenmodes:     {rpa.n_eigenpotentials}")

# 4. Generate scree plot diagnostic
rpa.save_scree_plot("water_rpa_scree.png")
```

---

## 4. Optional Python modules

Pure-Python helpers under `ferric.*` (not part of the compiled API; their optional dependencies are imported lazily):

* **`ferric.rdkit`**: `from_rdkit`, `from_smiles`, `to_rdkit` (RDKit `Mol` <-> `Molecule`; `pip install "ferric[rdkit]"`).
* **`ferric.units`**: `BOHR_TO_ANGSTROM`, `ANGSTROM_TO_BOHR`, `HARTREE_TO_EV`, `HARTREE_TO_KCAL_MOL`, `HARTREE_TO_KJ_MOL` (the Rust crates' values).
* **`ferric.serialize`**: `to_dict`, `to_json`, `molecule_from_dict`, `read_run_log` (plain-data views of molecules and results; the CLI JSON Lines run log).
* **`ferric.qcschema`**: `run`, `molecule_from_qcschema`, `molecule_to_qcschema`.
* **`ferric.mlip`**: `write_extxyz`, `write_asedb`, `frames`, `to_atoms` (QCSchema results or `FerricCalculator` atoms -> extended XYZ / ASE database, eV and eV/Angstrom, level-of-theory metadata; writing needs `pip install "ferric[ase]"`).
* **`ferric.ase`** (`pip install "ferric[ase]"`) and **`ferric.viz`** (py3Dmol): ASE calculator and 3-D views.

## 5. Errors

Every exception ferric raises for a failed computation derives from
`ferric.FerricError` **and** from the builtin class it raised before the
hierarchy existed, so `except RuntimeError` / `except ValueError` /
`except NotImplementedError` handlers keep working and messages are unchanged.

| class | also | meaning |
|---|---|---|
| `ConvergenceError` | `RuntimeError` | a typed non-convergence (`ScfConvergence`, `Convergence`): SCF, CC, a response solver, the orbital optimisation |
| `BasisError` | `InputError`, `RuntimeError` | an unknown, unparsable or non-covering basis set / ECP (also a `ValueError`: `BasisSet.bundled` raised that, the `run_*` drivers `RuntimeError`) |
| `NumericalError` | `RuntimeError` | a typed linear-algebra failure (`Lapack`) |
| `MemoryBudgetError` | `RuntimeError` | a typed memory refusal (`MemoryExhausted`) |
| `UnsupportedError` | `NotImplementedError` | a valid option combination the method does not implement (`Unsupported`) |
| `InputError` | `ValueError` | an invalid argument or option value |
| `ComputationError` | `RuntimeError` | any other failure that carries no type |

The classes follow the *typed* failure the Rust code returns, not the topic. A
failure still returned as untyped text is a `ComputationError` even when it
is "really" a convergence or memory problem (the pre-flight memory check in
`ferric-core` is one, an unknown functional another), so
`except ferric.ComputationError` is **not** "all runtime failures": catch
`RuntimeError` for that, or `FerricError` for everything ferric raises on
purpose.

```python
try:
    ferric.run_frequencies(mol, "def2-svp", scf_max_iter=20)
except ferric.ConvergenceError:
    ...  # retry with a level shift
except ferric.InputError:
    ...  # fix the call
```

Things to know when upgrading:

- `type(e).__name__` and the traceback's first token change
  (`ValueError: ...` is now `ferric.InputError: ...`). The message after it
  is the same text. Code that compares the class name or parses the
  traceback line must be updated; `isinstance` and `except` are unaffected.
- A failure that carries `FerricError::Unsupported` and was raised through a
  plain `RuntimeError` (the local OP-MP3 refusals in `run_opmp3`, for
  example) is now an `UnsupportedError`, so `except NotImplementedError`
  (including the handlers in `ferric.ase`) now catches it. It is still a
  `RuntimeError`.
- A bad auxiliary basis from a driver is a `BasisError`, which is also a
  `ValueError`: `except ValueError` now catches it.
- `ferric.qcschema.run` classifies by these classes (see its docstring): a
  driver basis failure is `input_error` (it was `execution_error`).

What stays plain (not in the hierarchy): `TypeError` from argument
conversion; `configure_gpu` / `configure_precision` and the GPU settings
helpers, `set_blas_threads` and the libint test hook (plain `ValueError` /
`RuntimeError`); `Molecule.from_xyz`'s `OSError` for a missing file and the MD
writers' `OSError`; `NotImplementedError` from the symmetry-frame accessors
(`density`, `mo_coefficients`, `gradient` of a `symmetry=True` result);
exceptions raised by pure-Python helpers (`ferric.ase`, `ferric.serialize`,
...); and Rust panics (`pyo3_runtime.PanicException`). The periodic drivers
map `General` and `MemoryExhausted` refusals to `InputError`, as they raised
`ValueError` before.

See `ARCHITECTURE.md` ("Errors") for the mapping from the Rust `FerricError`
variants and the rule for method authors.
