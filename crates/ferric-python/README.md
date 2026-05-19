# ferric Python Bindings API Guide

This directory contains the PyO3 Python bindings for the `ferric` quantum chemistry engine. The compiled library exposes high-performance Rust kernels to Python, returning inputs and outputs as native Python types and `numpy` arrays.

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
  Orthonormal PDEP eigenvectors in the auxiliary basis (shape `(naux, M)`).
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
* **`run_pdep_rpa(mol: Molecule, basis_set: BasisSet, auxbasis: BasisSet, frozen_core: int = None, n_quad: int = None, quadrature: str = None, u0: float = None, trunc_thresh: float = None, davidson_conv_thresh: float = None, run_diagnostics: bool = False, k_builder: str = None) -> PdepRpaResult`**:
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
