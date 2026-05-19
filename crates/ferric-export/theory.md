# ferric Export Schemas & Grid Theory

This module handles exporting calculations to standard files for visualization (Gaussian Cube) and machine learning (compressed NumPy `.npz` archives).

---

## 1. Machine Learning Dataset Schema (`.npz`)

When the RPA/export pipeline is executed (driven by the `export_npz` configuration in TOML), key calculation tensors are exported to a compressed `.npz` file. Downstream machine learning consumers (such as electrostatic potential prediction models or generation diffusion pipelines) read this schema.

| Key | Shape | Type | Units | Physical Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `coords` | $(N_{\text{atoms}}, 3)$ | `float64` | Bohr | Cartesian coordinates of the atomic nuclei. |
| `atomic_numbers` | $(N_{\text{atoms}},)$ | `int64` | - | Atomic numbers ($Z$) of the atoms. |
| `mo_coeffs` | $(N_{\text{AO}}, N_{\text{AO}})$ | `float64` | a.u. | Molecular Orbital (MO) coefficient matrix in the Atomic Orbital (AO) basis. |
| `orbital_energies` | $(N_{\text{AO}},)$ | `float64` | Hartree | Energies of the molecular orbitals. |
| `density_matrix` | $(N_{\text{AO}}, N_{\text{AO}})$ | `float64` | a.u. | AO-basis density matrix $D_{\mu\nu}$. Used for charges and density-based property derivation. |
| `boys_coeffs` | $(N_{\text{AO}}, N_{\text{occ}})$ | `float64` | a.u. | Coefficients of Boys-localized occupied molecular orbitals. |
| `pdep_eigenvectors` | $(N_{\text{aux}}, M)$ | `float64` | a.u. | PDEP eigenpotentials, where $M$ is the number of retained eigenvalues. |
| `esp_atoms` | $(N_{\text{atoms}},)$ | `float64` | Hartree / $e$ | Electrostatic potential $V(R_A)$ evaluated exactly at the coordinates of each atomic nucleus. |
| `electric_field` | $(N_{\text{atoms}}, 3)$ | `float64` | a.u. | Vector components of the electric field $\vec{E}(R_A)$ evaluated at each nucleus. |
| `alpha_tensor` | $(3, 3)$ | `float64` | Bohr$^3$ | Static molecular polarizability tensor $\alpha_{ij}$ at $\omega=0$. |
| `alpha_atomic` | $(N_{\text{atoms}}, 3, 3)$| `float64` | Bohr$^3$ | Per-atom Hirshfeld-decomposed polarizability contribution tensors (additive: $\sum_A \alpha^A = \alpha$). |
| `hirshfeld_charges` | $(N_{\text{atoms}},)$ | `float64` | $e$ | Atomic charges computed via Hirshfeld population analysis relative to a spherical proatom baseline. |
| `lowdin_charges` | $(N_{\text{atoms}},)$ | `float64` | $e$ | Atomic charges computed via Löwdin symmetric orthogonalization. recommended for CM5 charge models. |

---

## 2. Gaussian Cube Real-Space Grid Exporter

The Gaussian Cube format exports real-space representation fields (like molecular orbitals or PDEP eigenpotentials) evaluated on a regular 3D grid.

### Grid Specification (`GridSpec`)
The grid is defined via:
* **`origin`**: 3D coordinate $[x_0, y_0, z_0]$ (Bohr) representing the grid's bottom-left-front corner.
* **`n_x, n_y, n_z`**: Number of voxels along the $x, y, z$ dimensions.
* **`spacing`**: Distance between grid points $h$ (Bohr), default is `0.2`.
* **`margin`**: Pad distance surrounding the molecular bounding box to prevent boundary truncation, default is `4.0` Bohr.

### Voxel Evaluation
For a real-space function expanded in the basis set (such as a PDEP eigenpotential $V_\alpha(r) = \sum_P c_\alpha^P \chi_P(r)$), the value at a voxel coordinate $r = [x_0 + i h, y_0 + j h, z_0 + k h]$ is computed as:
$$ V_\alpha(r) = \sum_P c_\alpha^P \chi_P(r) $$
where $\chi_P(r)$ is the auxiliary basis function evaluated at $r$. The resulting 3D scalar grid is written row-by-row into the standard `.cube` file.
