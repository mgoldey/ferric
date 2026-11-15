# 27.1 The NPZ feature bundle

<!-- section: 27.1; status: edit; registry: pdep-rpa;
     theory: crates/ferric-export/theory.md#"1. Machine Learning Dataset Schema (`.npz`)" -->

ferric writes one NumPy `.npz` file per molecule with the reference
wavefunction, the electrostatic and response properties and the run's
certificate. This section lists what is in the file, how each array is
defined in the code, and how it behaves when the molecule is moved or rotated.
That last point decides whether an array can be fed to an invariant or
equivariant model as is.

| Entry point | Location |
|---|---|
| Rust (assembly) | `ferric-rpa/src/property_bundle.rs::compute_pdep_properties`, `ferric-rpa/src/property_bundle.rs::write_pdep_npz` |
| Rust (writer) | `ferric-export/src/ml.rs::export_npz`, `ferric-export/src/ml.rs::NpzBundle` |
| CLI | `method.kind = "pdep-rpa"`, `[rpa] export_npz = "<path>"` |
| Python | `run_pdep_rpa(..., properties=True)` or `export_npz="<path>"` (`crates/ferric-python/ferric.pyi`) |

The bundle is closed-shell only. It is built from a converged RHF reference,
after the PDEP-RPA correlation energy.

## 27.1.1 What every array is computed from

Every property in the file is a property of the **reference SCF density**,
$D = 2\,C_o C_o^{\mathsf T}$, the matrix in `density_matrix`. None of them is
an RPA-relaxed density property. The polarizabilities are a response of that
same reference (section 27.3). A model trained on these arrays learns
Hartree–Fock electrostatics plus a direct-RPA response, not correlated
densities.

Two defaults differ between the CLI and Python:

| Setting | CLI `pdep-rpa` | Python `run_pdep_rpa` |
|---|---|---|
| Reference J/K | RI-JK on `def2-universal-jkfit` | exact four-centre J and K, unless `df_j_aux` / `df_k_aux` are set |
| `compute_*` knobs | on, except `compute_esp_surface` | the same, once `properties=True` or `export_npz=` is passed |

So the same molecule, basis and auxiliary basis give arrays that differ by the
density-fitting error of the reference, depending on the front end. Pin one
front end, or pin `df_j_aux` / `df_k_aux`, for a whole dataset.

## 27.1.2 Layout

Every array is `float64` in C order, except:

- `atomic_numbers` is `int64`;
- `c6_partition`, `c6_source` and `certificate` are `uint8` arrays holding
  UTF-8 bytes; decode with `f["certificate"].tobytes().decode()`;
- `c6_molecular_iso` is a length-1 array, not a scalar.

An absent key means the property was not requested or failed. A failed
property does not stop the file being written. The CLI then exits nonzero and
Python raises `RuntimeError` after writing, unless `allow_partial_npz` is set;
Python also lists the failures in `PdepRpaResult.property_gaps`. A dataset
builder should check the exit status, not only that the file exists.

**AO order.** `mo_coeffs` and `density_matrix` use ferric's shell order with
libint2's component order inside each shell. The file does not carry the basis
definition or the overlap matrix. Rebuild them with ferric for the same basis
name (`AoIntegrals`, section 27.4). The validation suite matched the order to
PySCF through the overlap matrix; do not assume another code's order.

**Not in the file.** The total energy is not in the NPZ. Python returns it as
`PdepRpaResult.total_energy`. The CLI prints it, but its JSON run log
(`site/src/using/run-logs.md`) does not yet hold it for `pdep-rpa`: the
`run_end` record carries the SCF reference energy with
`energy_is = "scf_reference_only"`, and `pdep-rpa` writes a `result_unlogged`
record instead of a `result` record. The PDEP
eigenvalues are not in the NPZ either; Python returns them as
`PdepRpaResult.eigenvalues_static`, sorted descending. Neither the overlap matrix nor the basis is stored.

## 27.1.3 Keys

$N$ is the number of atoms, $n$ the number of AOs, $n_{\mathrm{MO}} \le n$
the number of retained MOs (the orthogonalizer drops near-linear-dependent
combinations), $N_{\mathrm{aux}}$ the auxiliary-basis size, $M$ the number of
PDEP modes kept, $N_u$ the number of $C_6$ quadrature nodes and $N_p$ the
number of surface points. "Frame" says what happens to the array under a
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
| `pdep_eigenvectors` | $(N_{\mathrm{aux}},M)$ | — | aux coefficients of the PDEP mode densities, $c^{\mathsf T}Vc = I$ (section 27.3.5) | signs and degenerate subspaces arbitrary |
| `esp_atoms` | $(N,)$ | Hartree/$e$ | eq. 27.1, own nucleus excluded | invariant |
| `esp_surface`, `esp_points` | $(N_p,)$, $(N_p,3)$ | Hartree/$e$, Bohr | eq. 27.3 on Lebedev spheres (section 27.2.2) | the point set is fixed in the lab frame, so it changes under rotation |
| `electric_field` | $(N,3)$ | Hartree/($e$ Bohr) | $-\nabla V$ at each nucleus, own nucleus excluded | vectors |
| `alpha_tensor` | $(3,3)$ | Bohr³ | static dRPA polarizability, eq. 27.8 | rank-2 tensor; origin-independent |
| `alpha_atomic` | $(N,3,3)$ | Bohr³ | Krishtal intrinsic per-atom α, eq. 27.9 | rank-2 tensors, up to the Becke grid's lab-frame orientation |
| `alpha_ct` | $(3,3)$ | Bohr³ | `alpha_tensor - alpha_atomic.sum(0)` | as `alpha_atomic` |
| `lowdin_charges`, `mulliken_charges` | $(N,)$ | $e$ | eqs. 27.4, 27.5 | invariant |
| `hirshfeld_charges` | $(N,)$ | $e$ | eq. 27.6 on an atom-centred grid | invariant to grid error |
| `chelpg_charges`, `resp_charges` | $(N,)$ | $e$ | ESP fits on a lab-frame lattice | orientation-dependent (section 27.2.4) |
| `c6_freqs`, `c6_weights` | $(N_u,)$ | a.u. | Gauss–Legendre nodes $u_k$ and weights $w_k$ | constants |
| `alpha_atomic_dynamic` | $(N,N_u,3,3)$ | Bohr³ | per-atom $\alpha^A(iu_k)$ from the `c6_source` model | as `alpha_atomic` |
| `alpha_ct_dynamic` | $(N_u,3,3)$ | Bohr³ | dynamic CT remainder; `c6_source = "pdep"` only | as `alpha_ct` |
| `c6_iso` | $(N,N)$ | Hartree Bohr⁶ | pair $C_6^{AB}$, eq. 27.12 | invariant; a partition convention |
| `c6_aniso` | $(N,N,3,3)$ | Hartree Bohr⁶ | element-wise $\alpha^A_{ij}\alpha^B_{ij}$ quadrature | **not** a tensor (section 27.3.4) |
| `c6_molecular_iso` | $(1,)$ | Hartree Bohr⁶ | molecular $C_6$ from the molecular $\alpha(iu)$ | invariant |
| `c6_source`, `c6_partition` | bytes | — | `ts`, `pdep` or `mbd`; `becke` or `hirshfeld` | labels |
| `certificate` | bytes | — | result certificate as JSON (section 27.4.3) | label |

`boys_coeffs` is defined by the writer but the CLI and Python never fill it.

**Practical rule.** Use the scalars (`esp_atoms`, charges, spreads, orbital
energies, `c6_iso`, isotropic parts of α) directly in an invariant model. Use
`electric_field`, `dipole` and the 3×3 tensors only in an equivariant model,
or after projecting them onto a molecule-fixed frame. Treat `mo_coeffs`,
`pdep_eigenvectors` and `density_matrix` as gauge-dependent: they need
sign-, degeneracy- and rotation-aware handling before they can be features.

## 27.1.4 Turning keys on and off

The CLI knobs live under `[rpa]` and are named `compute_<key>`. The table in
`crates/ferric-export/theory.md` "Enabling keys" maps each key to its knob.
The validation suite checks that turning each knob off removes exactly its
keys (`crates/ferric-cli/tests/validation_npz.rs`).

```toml
[method]
kind = "pdep-rpa"

[rpa]
auxbasis = "cc-pvdz-ri"
export_npz = "water.npz"
compute_esp_surface = true     # the only knob that defaults to off
compute_c6 = false             # skip the dispersion block
```

```python
import json, numpy as np
f = np.load("water.npz")
cert = json.loads(f["certificate"].tobytes().decode())
q = f["lowdin_charges"]               # (N,)
assert np.allclose(f["alpha_ct"] + f["alpha_atomic"].sum(0), f["alpha_tensor"])
```

## Known limits (2026-11-15)

- The NPZ carries no energy, no overlap matrix, no basis definition and no
  PDEP eigenvalues (section 27.1.2), and the CLI run log has no PDEP-RPA total
  energy, so a CLI pipeline must take the total from stdout.
- With `c6_source = "pdep"` and `c6_partition = "hirshfeld"` on an open-shell
  reference, `ferric-rpa/src/dispersion.rs::pdep_dynamic_polarizability`
  falls back to the Becke partition, but `c6_partition` still reads
  `hirshfeld`. The NPZ path itself is closed-shell only, so this reaches only
  library callers today.
- The ESP, field and charge functions use the bare $Z$ (section 27.2.6).

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-export/src/ml.rs` | key names, dtypes, layout, the `esp_surface`/`esp_points` pairing |
| `crates/ferric-rpa/src/property_bundle.rs` | what each key is computed from, `coords`, defaults |
| `crates/ferric-export/theory.md`, "1. Machine Learning Dataset Schema (`.npz`)" | the key table this section extends |
| `crates/ferric-python/ferric.pyi`, `run_pdep_rpa` and `PdepRpaResult` | the Python defaults and the reference J/K difference |
| `site/src/reference/validation.md`, rows "NPZ export" and "NPZ export against PySCF" | the layout and knob checks |

Changes from the sources: the frame column is new. The theory table writes
the per-atom α with $w_A$ on the bra and the eigenpotential column index as
$\alpha$; this section uses $\xi$ for PDEP modes (notation).
