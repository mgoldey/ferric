# The NPZ feature bundle {#npz-bundle}

<!-- status: edit; registry: pdep-rpa;
     theory: crates/ferric-export/theory.md#"1. Machine Learning Dataset Schema (`.npz`)" -->

A model that learns from a quantum-chemistry run never sees the run. It sees
arrays. For those arrays to be usable as features, every sample has to give
the same quantity under the same name, in the same units and shape, with the
atoms in the same order, and the model has to know how each array changes
when the molecule is moved. An invariant model can take a scalar per atom as
is; a dipole or a 3×3 tensor rotates with the molecule and needs an
equivariant model or a molecule-fixed frame.

ferric writes one NumPy `.npz` file per molecule. It holds the reference
wavefunction, the electrostatic and response properties and the run's
certificate. This section says what the bundle is computed from (@sec:npz-bundle-every-array-computed), how
the file is laid out (@sec:npz-bundle-layout), what every key means and how it behaves under
a rotation or translation (@sec:npz-bundle-keys), and how to switch keys on and off
(@sec:npz-bundle-turning-keys-off). Section @sec:export-schemas covers the writer itself: what it refuses and why.

| Entry point | Location |
|---|---|
| Rust (assembly) | `ferric-rpa/src/property_bundle.rs::compute_pdep_properties`, `ferric-rpa/src/property_bundle.rs::write_pdep_npz` |
| Rust (writer) | `ferric-export/src/ml.rs::export_npz`, `ferric-export/src/ml.rs::NpzBundle` |
| CLI | `method.kind = "pdep-rpa"`, `[rpa] export_npz = "<path>"` |
| Python | `run_pdep_rpa(..., properties=True)` or `export_npz="<path>"` (`crates/ferric-python/ferric.pyi`) |

The bundle is closed-shell only. It is built from a converged RHF reference,
after the PDEP-RPA correlation energy.

## What every array is computed from {#npz-bundle-every-array-computed}

Every property in the file is a property of the **reference SCF density**,
$D = 2\,C_o C_o^{\mathsf T}$, the matrix in `density_matrix`. None of them
uses an RPA-relaxed density. The polarizabilities are the response of that
same reference (section @sec:polarizability-pdep-c6). A model trained on these arrays therefore
learns Hartree–Fock electrostatics plus a direct-RPA response. It does not
learn correlated densities.

The CLI and Python reach the bundle with two different defaults:

| Setting | CLI `pdep-rpa` | Python `run_pdep_rpa` |
|---|---|---|
| Reference J/K | RI-JK on `def2-universal-jkfit` | exact four-centre J and K, unless `df_j_aux` / `df_k_aux` are set |
| `compute_*` knobs | on, except `compute_esp_surface` | the same, once `properties=True` or `export_npz=` is passed |

The first row matters for a dataset. The same molecule, basis and auxiliary
basis give arrays that differ by the density-fitting error of the reference,
depending on which front end produced them. Pin one front end for the whole
dataset, or pin `df_j_aux` / `df_k_aux`.

## Layout {#npz-bundle-layout}

Every array is `float64` in C order, except:

- `atomic_numbers` is `int64`;
- `c6_partition`, `c6_source` and `certificate` are `uint8` arrays holding
  UTF-8 bytes; decode with `f["certificate"].tobytes().decode()`;
- `c6_molecular_iso` is a length-1 array, not a scalar.

An absent key means the property was not requested or failed. A failed
property does not stop the file being written. The CLI then exits nonzero,
and Python raises `RuntimeError` after writing unless `allow_partial_npz` is
set; Python also lists the failures in `PdepRpaResult.property_gaps`. A
dataset builder should therefore check the exit status, and not only that
the file exists.

**AO order.** `mo_coeffs` and `density_matrix` use ferric's shell order, with
libint2's component order inside each shell. The file carries neither the
basis definition nor the overlap matrix. To use these arrays you rebuild both
with ferric for the same basis name (`AoIntegrals`, section @sec:python-cube-provenance). The
validation suite matched the order to PySCF through the overlap matrix. Do
not assume another code's order.

**What the file leaves out.** The total energy is not in the NPZ. Python
returns it as `PdepRpaResult.total_energy`. The CLI prints it, but its JSON
run log (`site/src/using/run-logs.md`) does not yet hold it for `pdep-rpa`:
the `run_end` record carries the SCF reference energy with
`energy_is = "scf_reference_only"`, and `pdep-rpa` writes a
`result_unlogged` record in place of a `result` record. The PDEP eigenvalues
are not in the NPZ either; Python returns them as
`PdepRpaResult.eigenvalues_static`, sorted descending. Neither the overlap
matrix nor the basis is stored.

## Keys {#npz-bundle-keys}

The table uses these sizes. $N$ is the number of atoms, $n$ the number of
AOs, and $n_{\mathrm{MO}} \le n$ the number of retained MOs (the
orthogonalizer drops near-linearly-dependent combinations).
$N_{\mathrm{aux}}$ is the auxiliary-basis size, $M$ the number of PDEP modes
kept, $N_u$ the number of $C_6$ quadrature nodes and $N_p$ the number of
surface points. The "Frame" column says what happens to the array under a
rigid rotation or translation of the input geometry.

| Key | Shape | Units | Definition (from the code) | Frame |
|---|---|---|---|---|
| `coords` | $(N,3)$ | Bohr | input geometry, not reoriented or recentred | rotates and translates with the input |
| `atomic_numbers` | $(N,)$ | — | bare $Z$ (not the ECP-reduced charge) | invariant |
| `mo_coeffs` | $(n, n_{\mathrm{MO}})$ | — | reference MOs $C$ | column signs and degenerate subspaces arbitrary; AO components mix under rotation |
| `orbital_energies` | $(n_{\mathrm{MO}},)$ | Hartree | reference $\varepsilon_p$ | invariant |
| `density_matrix` | $(n,n)$ | — | $D = 2C_oC_o^{\mathsf T}$ | no sign ambiguity; AO components mix under rotation |
| `orbital_centers` | $(n_{\mathrm{MO}},3)$ | Bohr | $\langle\phi_p\lvert\mathbf r\rvert\phi_p\rangle$ | vectors; translate with the molecule |
| `orbital_spreads` | $(n_{\mathrm{MO}},)$ | Bohr | $\sqrt{\max(0,\langle r^2\rangle_p - \lvert\langle\mathbf r\rangle_p\rvert^2)}$ | invariant |
| `density_second_moment` | $(3,3)$ | Bohr² | $M_{cd} = \sum_{\mu\nu} D_{\mu\nu}\langle\mu\lvert r_c r_d\rvert\nu\rangle$ about the origin, electronic only | rank-2 tensor about the origin; changes under translation |
| `dipole` | $(3,)$ | $e\,a_0$ | $\sum_A Z^{\mathrm{eff}}_A\mathbf R_A - \sum D_{\mu\nu}\langle\mu\lvert\mathbf r\rvert\nu\rangle$, origin at 0 | vector; origin-dependent for ions |
| `pdep_eigenvectors` | $(N_{\mathrm{aux}},M)$ | — | aux coefficients of the PDEP mode densities, $c^{\mathsf T}Vc = I$ (section @sec:polarizability-pdep-c6-pdep-eigenvectors) | signs and degenerate subspaces arbitrary |
| `esp_atoms` | $(N,)$ | Hartree/$e$ | eq. @eq:electrostatics-and-charges-1, own nucleus excluded | invariant |
| `esp_surface`, `esp_points` | $(N_p,)$, $(N_p,3)$ | Hartree/$e$, Bohr | eq. @eq:electrostatics-and-charges-3 on Lebedev spheres (section @sec:electrostatics-and-charges-potential-arbitrary-points-surface) | the point set is fixed in the lab frame, so it changes under rotation |
| `electric_field` | $(N,3)$ | Hartree/($e$ Bohr) | $-\nabla V$ at each nucleus, own nucleus excluded | vectors |
| `alpha_tensor` | $(3,3)$ | Bohr³ | static dRPA polarizability, eq. @eq:polarizability-pdep-c6-1 | rank-2 tensor; origin-independent |
| `alpha_atomic` | $(N,3,3)$ | Bohr³ | Krishtal intrinsic per-atom α, eq. @eq:ml-alpha-per-atom-kernel | rank-2 tensors, up to the Becke grid's lab-frame orientation |
| `alpha_ct` | $(3,3)$ | Bohr³ | `alpha_tensor - alpha_atomic.sum(0)` | as `alpha_atomic` |
| `lowdin_charges`, `mulliken_charges` | $(N,)$ | $e$ | eqs. @eq:electrostatics-and-charges-l-wdin, @eq:electrostatics-and-charges-mulliken | invariant |
| `hirshfeld_charges` | $(N,)$ | $e$ | eq. @eq:electrostatics-and-charges-hirshfeld on an atom-centred grid | invariant to grid error |
| `chelpg_charges`, `resp_charges` | $(N,)$ | $e$ | ESP fits on a lab-frame lattice | orientation-dependent (section @sec:electrostatics-and-charges-five-charge-schemes) |
| `c6_freqs`, `c6_weights` | $(N_u,)$ | a.u. | Gauss–Legendre nodes $u_k$ and weights $w_k$ | constants |
| `alpha_atomic_dynamic` | $(N,N_u,3,3)$ | Bohr³ | per-atom $\alpha^A(iu_k)$ from the `c6_source` model | as `alpha_atomic` |
| `alpha_ct_dynamic` | $(N_u,3,3)$ | Bohr³ | dynamic CT remainder; `c6_source = "pdep"` only | as `alpha_ct` |
| `c6_iso` | $(N,N)$ | Hartree Bohr⁶ | pair $C_6^{AB}$, eq. @eq:ml-c6-cp-quadrature | invariant; a partition convention |
| `c6_aniso` | $(N,N,3,3)$ | Hartree Bohr⁶ | element-wise $\alpha^A_{ij}\alpha^B_{ij}$ quadrature | **not** a tensor (section @sec:polarizability-pdep-c6-casimir-polder) |
| `c6_molecular_iso` | $(1,)$ | Hartree Bohr⁶ | molecular $C_6$ from the molecular $\alpha(iu)$ | invariant |
| `c6_source`, `c6_partition` | bytes | — | `ts`, `pdep` or `mbd`; `becke` or `hirshfeld` | labels |
| `certificate` | bytes | — | result certificate as JSON (section @sec:python-cube-provenance-certificate-provenance-label) | label |

`boys_coeffs` is defined by the writer, but neither the CLI nor Python ever
fills it.

Why does the dipole depend on the origin for an ion? Move the origin to
$\mathbf o$. Each nucleus contributes $Z^{\mathrm{eff}}_A(\mathbf R_A -
\mathbf o)$ and each electron $-(\mathbf r - \mathbf o)$, so the dipole
becomes $\boldsymbol\mu - Q\,\mathbf o$, with $Q$ the molecular charge. For
a neutral molecule $Q = 0$ and the origin drops out. The file puts the origin
at $(0,0,0)$ of the input coordinates.

The Frame column sorts the keys into three groups, and each group goes into a
model differently.

- **Scalars.** `esp_atoms`, the charges, `orbital_spreads`,
  `orbital_energies`, `c6_iso` and the isotropic parts of α are invariant.
  An invariant model can use them directly.
- **Vectors and tensors.** `electric_field`, `dipole` and the 3×3 tensors
  rotate with the molecule. Use them in an equivariant model, or project them
  onto a molecule-fixed frame first.
- **Gauge-dependent arrays.** `mo_coeffs`, `pdep_eigenvectors` and
  `density_matrix` carry arbitrary signs, arbitrary bases inside degenerate
  subspaces, or AO components that mix under rotation. They need sign-,
  degeneracy- and rotation-aware handling before they can be features.

## Turning keys on and off {#npz-bundle-turning-keys-off}

The CLI knobs live under `[rpa]` and are named `compute_<key>`. Section
@sec:export-schemas-quantity-key maps each key to its knob. The validation suite checks that turning
each knob off removes exactly its keys
(`crates/ferric-cli/tests/validation_npz.rs`).

```toml
[method]
kind = "pdep-rpa"

[rpa]
auxbasis = "cc-pvdz-ri"
export_npz = "water.npz"
compute_esp_surface = true     # the only knob that defaults to off
compute_c6 = false             # skip the dispersion block
```

Reading the file back needs only numpy. The last line checks the
charge-transfer identity of section @sec:polarizability-pdep-c6-per-atom-split-charge on the arrays themselves:

```python
import json, numpy as np
f = np.load("water.npz")
cert = json.loads(f["certificate"].tobytes().decode())
q = f["lowdin_charges"]               # (N,)
assert np.allclose(f["alpha_ct"] + f["alpha_atomic"].sum(0), f["alpha_tensor"])
```

## Known limits (2026-11-15)

- The NPZ carries no energy, no overlap matrix, no basis definition and no
  PDEP eigenvalues (section @sec:npz-bundle-layout), and the CLI run log has no PDEP-RPA total
  energy, so a CLI pipeline must take the total from stdout.
- With `c6_source = "pdep"` and `c6_partition = "hirshfeld"` on an open-shell
  reference, `ferric-rpa/src/dispersion.rs::pdep_dynamic_polarizability`
  falls back to the Becke partition, but `c6_partition` still reads
  `hirshfeld`. The NPZ path itself is closed-shell only, so this reaches only
  library callers today.
- `atomic_numbers` holds the bare $Z$, while the ESP, field, dipole and
  charge functions pair the density with the ECP-reduced $Z^{\mathrm{eff}}$
  (section @sec:electrostatics-and-charges-known-limits-2026-11). On an ECP atom the two differ by the number of core
  electrons the ECP replaces.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-export/src/ml.rs` | key names, dtypes, layout, the `esp_surface`/`esp_points` pairing |
| `crates/ferric-rpa/src/property_bundle.rs` | what each key is computed from, `coords`, defaults |
| `crates/ferric-export/theory.md`, "1. Machine Learning Dataset Schema (`.npz`)" | the key table this section extends |
| `crates/ferric-python/ferric.pyi`, `run_pdep_rpa` and `PdepRpaResult` | the Python defaults and the reference J/K difference |
| `site/src/reference/validation.md`, rows "NPZ export" and "NPZ export against PySCF" | the layout and knob checks |
| `crates/ferric-scf/src/properties.rs` | the effective nuclear charge in the ESP, field, dipole and charges |

Changes from the sources: the frame column and the origin shift of the
dipole are new. The theory table writes the per-atom α with $w_A$ on the bra
and the eigenpotential column index as $\alpha$; this section uses $\xi$ for
PDEP modes (notation).
