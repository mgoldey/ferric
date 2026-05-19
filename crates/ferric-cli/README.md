# ferric-cli Configuration Guide

This directory contains the CLI driver for `ferric`. Calculations are driven using TOML files. Below is a comprehensive reference of all blocks and parameters available in the TOML configuration.

---

## 1. `[molecule]` Block
Specifies the molecular geometry and chemical state.

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `xyz` | String | *Required* | Path to the input `.xyz` file. Coordinates in the file are parsed in Angstroms and converted internally to Bohr. |
| `charge` | Integer | `0` | Total electronic charge of the system. |
| `multiplicity` | Integer | `1` | Spin multiplicity ($2S + 1$). |

---

## 2. `[basis]` Block
Configures the primary orbital basis set.

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | String | *Optional* | Name of a bundled basis set (e.g., `"sto-3g"`, `"6-31g"`, `"cc-pvdz"`, `"def2-svp"`). |
| `path` | String | *Optional* | Path to a custom G94 or BSE-JSON format basis set file. |

---

## 3. `[method]` Block
Defines the method and task to run.

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `kind` | String | *Required* | The chemistry method. Supported values:<br>- `"rhf"`: Restricted Hartree-Fock<br>- `"dft"`: Density Functional Theory<br>- `"rimp2"`: Resolution-of-Identity MP2<br>- `"att-rimp2"`: Attenuated RI-MP2<br>- `"scs-mp2"`: Spin-Component Scaled MP2<br>- `"laplace-mp2"`: Laplace-transformed RI-MP2<br>- `"ccd"` / `"ccsd"` / `"ccsd(t)"`: Coupled Cluster methods<br>- `"pdep-rpa"`: PDEP-RPA correlation energy |
| `task` | String | `"energy"` | The task to perform: `"energy"` or `"gradient"`. |

---

## 4. `[scf]` Block
Configures the Self-Consistent Field (SCF) iterations.

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_iter` | Integer | `100` | Maximum number of SCF iterations. |
| `energy_conv` | Float | `1e-8` | Energy convergence threshold (Hartree). |
| `density_conv` | Float | `1e-7` | Density matrix convergence threshold. |
| `diis_size` | Integer | `8` | Subspace size for DIIS convergence acceleration. |
| `integral_thresh`| Float | `1e-12` | Schwarz screening threshold for two-electron integrals. |
| `k_builder` | String | *None* | Selects K-matrix builder (e.g., `"DirectK"`, `"LinkK"`). |
| `df_j_aux` | String | *None* | Auxiliary basis set for density-fitted Coulomb (J) build. |
| `df_k_aux` | String | *None* | Auxiliary basis set for density-fitted Exchange (K) build. |

---

## 5. `[mp2]` Block
Configures MP2 correlation and variant calculations.

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `auxbasis` | String | *None* | Name of auxiliary fitting basis set (e.g., `"cc-pvdz-ri"`). |
| `frozen_core` | Integer | `0` | Number of core orbitals to freeze (not correlated). |
| `omega` | Float | `0.420` | Range-separation parameter $\omega$ in Å⁻¹ (for `"att-rimp2"`); the operator is erfc($\omega r$)/$r$. |
| `c_os` | Float | *None* | Opposite-spin scaling factor (for `"scs-mp2"`). |
| `c_ss` | Float | *None* | Same-spin scaling factor (for `"scs-mp2"`). |
| `n_quad` | Integer | *None* | Number of minimax quadrature points (for `"laplace-mp2"`). |

---

## 6. `[optimize]` Block
Controls the BFGS geometry optimizer.

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_steps` | Integer | `100` | Maximum geometry optimization steps. |
| `g_max_thresh` | Float | `1e-4` | Maximum force component convergence threshold (Hartree/Bohr). |
| `g_rms_thresh` | Float | `1e-4` | Root-mean-square force convergence threshold (Hartree/Bohr). |
| `e_conv` | Float | `1e-6` | Energy convergence threshold between optimization steps. |
| `trust_radius` | Float | `0.1` | Initial trust radius (Bohr). |

---

## 7. `[rpa]` Block
Configures PDEP-RPA calculations, eigenvalues solvers, and exports.

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `auxbasis` | String | *None* | Auxiliary basis set for RPA fitting (e.g., `"cc-pvdz-ri"`). |
| `frozen_core` | Integer | `0` | Number of core orbitals to freeze. |
| `n_quad` | Integer | `40` | Number of quadrature points for imaginary frequency integration. |
| `quadrature` | String | `"gauss-legendre"`| Quadrature scheme (`"gauss-legendre"` or `"minimax"`). |
| `trunc_thresh` | Float | `1e-4` | Truncation threshold for static PDEP eigenvalues $|\lambda_\alpha(0) - 1|$. |
| `davidson_conv_thresh` | Float | `1e-6` | Davidson/Lanczos solver convergence tolerance. |
| `u0` | Float | `0.5` | Minimax frequency scaling parameter. |
| `run_diagnostics` | Boolean | `false` | Run dense reference RI-dRPA calculations for validation. |
| `export_eigpot_prefix` | String | *None* | File path prefix to save leading PDEP eigenpotentials as `.cube` files. |
| `export_eigpot_count` | Integer | `10` | Number of leading eigenpotentials to export. |
| `cube_spacing` | Float | `0.2` | Bounding box voxel spacing (Bohr) for Cube files. |
| `cube_margin` | Float | `4.0` | Margin around molecular box (Bohr) for Cube files. |
| `export_npz` | String | *None* | Output path to write compressed `.npz` feature bundle. |
| `compute_esp` | Boolean | `true` | Compute nuclear electrostatic potential (in `.npz`). |
| `compute_polarizability` | Boolean | `true` | Compute static polarizability tensor (in `.npz`). |
| `compute_alpha_atomic` | Boolean | `true` | Compute Hirshfeld-decomposed polarizabilities (in `.npz`). |
| `compute_electric_field` | Boolean | `true` | Compute nuclear electric field vectors (in `.npz`). |
| `compute_density_matrix` | Boolean | `true` | Include AO-basis density matrix (in `.npz`). |
| `compute_hirshfeld_charges`| Boolean | `true` | Include Hirshfeld atomic charges (in `.npz`). |
| `compute_lowdin_charges` | Boolean | `true` | Include Löwdin atomic charges (in `.npz`). |
