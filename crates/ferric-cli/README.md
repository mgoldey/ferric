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
| `k_builder` | String | *None* (= `"direct"`) | Exchange builder: `"direct"`, `"link"` (LinK, exact) or `"cosx"` (seminumerical COSX, Coulomb operator only, no gradients). Honoured by RHF, UHF and ROHF. Ignored with a warning when `df_k_aux` is set, when the functional uses no exact exchange, or for a range-separated functional. See the book's *Choosing how exchange is built*. |
| `cosx_grid` | Table | `{ radial = 50, angular = 110 }` | COSX grid. The default is the coarsest grid meeting a 0.1 kcal/mol reaction-energy bar; `angular` must be a tabulated Lebedev order. Only with `k_builder = "cosx"`. |
| `cosx_overlap_fit` | Boolean | `true` | Izsák–Neese overlap correction for COSX. Helps at the default grid; net-negative on coarser grids. Only with `k_builder = "cosx"`. |
| `cosx_backend` | String | `"md3c1e"` | COSX 3c1e integral kernel: `"md3c1e"` (batched McMurchie–Davidson) or `"cosx-a"` (per-point libint2, ~3× slower; the cross-check backend). Only with `k_builder = "cosx"`. |
| `cosx_screen_thresh` | Float | `1e-7` | COSX density-driven pair-screening threshold on bound(A) × max|F|. `0.0` = unscreened (bit-identical); negative/NaN, or a positive value with `cosx_backend = "cosx-a"`, are hard errors. Only with `k_builder = "cosx"`. |
| `df_j_aux` | String | *None* | Auxiliary basis set for density-fitted Coulomb (J) build. |
| `df_k_aux` | String | *None* | Auxiliary basis set for density-fitted Exchange (K) build. |

---

## 5. `[mp2]` Block
Configures MP2 correlation and variant calculations.

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `auxbasis` | String | *None* | Name of auxiliary fitting basis set (e.g., `"cc-pvdz-ri"`). |
| `frozen_core` | Integer or `"auto"` | `0` | Core orbitals to freeze (not correlated). `"auto"` derives the standard small-core count from the molecule — see [Frozen core](#frozen-core). Shared with the CC and double-hybrid methods. |
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
| `frozen_core` | Integer or `"auto"` | `0` | Core orbitals to freeze. `"auto"` derives the standard small-core count from the molecule — see [Frozen core](#frozen-core). |
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

---

## 8. Frozen core

`frozen_core` appears in `[mp2]` (shared by the whole MP2 family and by the CC
and double-hybrid methods), in `[rpa]`, and in `[gw]`. It takes either an
explicit orbital count or `"auto"`:

```toml
[mp2]
frozen_core = "auto"   # standard small-core count for THIS molecule
frozen_core = 3        # exactly three orbitals, whatever the molecule is
frozen_core = "none"   # correlate everything (the default, same as 0)
```

`true` and `false` are accepted as synonyms of `"auto"` and `"none"`.
Anything else — `"fc"`, `"core"`, a negative number — is a parse error, never
a silent fall-through to 0.

**The default is 0 (all-electron), not `"auto"`.** Freezing the core changes
the correlation energy, so it stays opt-in: an input file that does not mention
`frozen_core` computes today exactly what it computed before this key existed.

**What `"auto"` counts.** One orbital per core shell below the element's
valence shell:

| Elements | Core orbitals | Shells frozen |
|----------|---------------|---------------|
| H–He     | 0             | — |
| Li–Ne    | 1             | 1s |
| Na–Ar    | 5             | +2s2p |
| K–Zn     | 9             | +3s3p |
| Ga–Kr    | 14            | +3d |
| Rb–Cd    | 18            | +4s4p |
| In–Xe    | 23            | +4d |

This is the small-core convention Psi4, ORCA and Q-Chem apply. The (n−1)d
shell stays **correlated** across its own transition series and becomes core
only once the following p-block starts (3d is valence for Sc–Zn, core from Ga
on): the d electrons of a transition metal are chemically active and freezing
them is a large, uncontrolled error.

Two adjustments make the count match the calculation rather than the periodic
table:

* **Ghost atoms count 0.** A basis-only center has no electrons, so it has no
  core to freeze.
* **ECP atoms are counted on what is left.** An effective core potential has
  already removed core orbitals from the MO space, so only the remainder is
  frozen — for iodine under a def2-ECP (`n_core = 28`, i.e. 14 orbitals),
  23 − 14 = 9 orbitals (4s4p4d).

A run that resolves an `"auto"` prints the number it picked:

```
[ferric] frozen core: 3 orbital(s) frozen from [mp2] frozen_core = "auto"
```

A `frozen_core` that would freeze every occupied orbital of the minority spin
(leaving nothing to correlate) is an error before the SCF starts, naming the
section that set it.

**`[gw] frozen_core` is the one that may be left unset**, in which case it
follows `[rpa] frozen_core`; W and Σ must use the same frozen core for
self-consistency. Setting it to `"none"` is *not* the same as omitting it.
