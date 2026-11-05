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
| `orbital_centers` | $(N_{\text{MO}}, 3)$ | `float64` | Bohr | Orbital centroids $\langle\phi_p\vert \vec r\vert\phi_p\rangle$ about the origin. Restricted references only. |
| `orbital_spreads` | $(N_{\text{MO}},)$ | `float64` | Bohr | Orbital spreads $\sqrt{\langle r^2\rangle_p - \lvert\langle\vec r\rangle_p\rvert^2}$. Restricted references only. |
| `density_second_moment` | $(3, 3)$ | `float64` | Bohr$^2$ | Electronic density second moment $M_{xy} = \sum_{\mu\nu} D_{\mu\nu}\langle\mu\vert x\,y\vert\nu\rangle$ about the origin; trace is $\langle r^2\rangle$ of the density. Written with `density_matrix`. |
| `dipole` | $(3,)$ | `float64` | $e\,a_0$ | Molecular dipole moment (electronic + nuclear). |
| `pdep_eigenvectors` | $(N_{\text{aux}}, M)$ | `float64` | a.u. | PDEP eigenpotentials, where $M$ is the number of retained eigenvalues. |
| `esp_atoms` | $(N_{\text{atoms}},)$ | `float64` | Hartree / $e$ | Electrostatic potential $V(R_A)$ evaluated exactly at the coordinates of each atomic nucleus. |
| `esp_surface` | $(N_{\text{pts}},)$ | `float64` | Hartree / $e$ | ESP on the van der Waals shell, in the order of `esp_points`. Written only together with `esp_points`. |
| `esp_points` | $(N_{\text{pts}}, 3)$ | `float64` | Bohr | Shell points: per-atom Lebedev spheres at scaled vdW radii, buried points dropped. |
| `electric_field` | $(N_{\text{atoms}}, 3)$ | `float64` | a.u. | Vector components of the electric field $\vec{E}(R_A)$ evaluated at each nucleus. |
| `alpha_tensor` | $(3, 3)$ | `float64` | Bohr$^3$ | Static molecular polarizability tensor $\alpha_{ij}$ at $\omega=0$. |
| `alpha_atomic` | $(N_{\text{atoms}}, 3, 3)$| `float64` | Bohr$^3$ | Per-atom Becke intrinsic polarizability (Krishtal, Senet, Van Alsenoy, J. Chem. Phys. 125, 034312 (2006)): $\alpha^A_{dj} = \langle w_A (r - R_A)_d \,\vert\, R \,\vert\, \mu_j \rangle$, atom-centred dipole on the bra, molecular dipole on the field-side ket. Charge transfer is excluded, so $\sum_A \alpha^A \neq \alpha$. |
| `alpha_ct` | $(3, 3)$ | `float64` | Bohr$^3$ | Charge-transfer polarizability $\alpha_{\text{CT}} = \alpha - \sum_A \alpha^A$ (same response as both), so `alpha_ct + alpha_atomic.sum(0) == alpha_tensor`. Present when both `alpha_tensor` and `alpha_atomic` are. |
| `alpha_atomic_dynamic` | $(N_{\text{atoms}}, N_\omega, 3, 3)$ | `float64` | Bohr$^3$ | Per-atom $\alpha^A(i\omega_k)$ at the C6 quadrature nodes `c6_freqs`; intrinsic definition for `c6_source = "pdep"`, the TS/MBD model otherwise (`c6_source`, `c6_partition` record which). |
| `alpha_ct_dynamic` | $(N_\omega, 3, 3)$ | `float64` | Bohr$^3$ | Dynamic charge-transfer remainder $\alpha(i\omega_k) - \sum_A \alpha^A(i\omega_k)$; `c6_source = "pdep"` only. |
| `hirshfeld_charges` | $(N_{\text{atoms}},)$ | `float64` | $e$ | Atomic charges computed via Hirshfeld population analysis relative to a spherical proatom baseline. |
| `lowdin_charges` | $(N_{\text{atoms}},)$ | `float64` | $e$ | Atomic charges computed via Löwdin symmetric orthogonalization. Recommended as the CM5 charge-model baseline. |
| `mulliken_charges` | $(N_{\text{atoms}},)$ | `float64` | $e$ | Mulliken gross atomic charges. |
| `chelpg_charges` | $(N_{\text{atoms}},)$ | `float64` | $e$ | CHELPG electrostatic-potential-fit charges. |
| `resp_charges` | $(N_{\text{atoms}},)$ | `float64` | $e$ | RESP charges. |
| `c6_partition` | $(L,)$ | `uint8` | - | UTF-8 bytes of the atomic partition (`becke` or `hirshfeld`) behind the per-atom C6 arrays; decode with `.tobytes().decode()`. |
| `c6_source` | $(L,)$ | `uint8` | - | UTF-8 bytes of the $\alpha(i\omega)$ source (`ts`, `pdep` or `mbd`); decode as `c6_partition`. |
| `c6_freqs` | $(N_\omega,)$ | `float64` | a.u. | Imaginary-frequency quadrature nodes $\omega_k$. |
| `c6_weights` | $(N_\omega,)$ | `float64` | a.u. | Casimir-Polder quadrature weights $w_k$. |
| `c6_iso` | $(N_{\text{atoms}}, N_{\text{atoms}})$ | `float64` | a.u. | Per-atom-pair isotropic $C_6^{AB}$. A partition convention, not an observable; `c6_iso.sum()` is not the molecular $C_6$. |
| `c6_aniso` | $(N_{\text{atoms}}, N_{\text{atoms}}, 3, 3)$ | `float64` | a.u. | Per-atom-pair anisotropic $C_6^{AB}{}_{ij}$; same caveat as `c6_iso`. |
| `c6_molecular_iso` | $(1,)$ | `float64` | a.u. | Molecular isotropic $C_6$, the DOSD-comparable value. Read this, never the `c6_iso` sum. |

### Enabling keys

Every key is optional; an absent array means the property was not requested or not computed.
The CLI (`kind = "pdep-rpa"`, `[rpa] export_npz = "<path>"`) turns the `compute_*` knobs on by default.

| Keys | TOML knob / condition |
| :--- | :--- |
| `coords`, `atomic_numbers`, `pdep_eigenvectors` | always |
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
| `c6_partition`, `c6_source`, `c6_freqs`, `c6_weights`, `alpha_atomic_dynamic`, `c6_iso`, `c6_aniso`, `c6_molecular_iso` | `compute_c6`; written together or not at all. `c6_source` defaults to `ts`; `c6_partition` defaults to `hirshfeld` for `pdep`, `becke` for `ts`/`mbd` |
| `alpha_ct_dynamic` | `compute_c6` with `c6_source = "pdep"` |
| `boys_coeffs` | library only: the CLI never writes it |

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
