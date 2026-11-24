# QCSchema input and output

`ferric.qcschema.run` takes a [QCSchema](https://molssi-qc-schema.readthedocs.io/)
`AtomicInput` (schema version 1) as a dict and returns an `AtomicResult` dict.
Tools that speak QCSchema (QCFractal, QCEngine-style drivers, geomeTRIC
through a QCSchema engine) can pass jobs to ferric this way.

`ferric.qcschema` is plain Python on the public bindings and needs no extra
package. QCElemental is optional: `pip install "ferric[qcschema]"` installs it
for building inputs with its models and validating outputs with
`qcelemental.models.AtomicResult(**out)`.

```python
import ferric.qcschema as qcs

out = qcs.run({
    "schema_name": "qcschema_input",
    "schema_version": 1,
    "molecule": {
        "symbols": ["O", "H", "H"],
        "geometry": [0.0, 0.0, 0.2217, 0.0, 1.4309, -0.8867, 0.0, -1.4309, -0.8867],
    },
    "driver": "gradient",
    "model": {"method": "hf", "basis": "6-31g"},
    "keywords": {},
})
out["success"]         # True
out["return_result"]   # (natom, 3) gradient, Hartree/Bohr
```

## What it runs

Units are QCSchema's: geometry in Bohr, energy in Hartree, gradient in
Hartree/Bohr, Hessian in Hartree/Bohr^2. The geometry is used as given, so
the gradient and Hessian are in the input frame. `geometry` may be flat (3N)
or (N, 3).

| `driver` | Result |
|---|---|
| `energy` | the total energy, for every method below |
| `gradient` | the (natom, 3) analytic gradient; `hf`, `scf`, `mp2` and Kohn-Sham methods; `ccsd` and `ccsd(t)` give `unsupported_error` |
| `hessian` | the (3N, 3N) analytic Cartesian Hessian; `hf`, `scf` and Kohn-Sham methods; any other method gives `unsupported_error` |
| `properties` | a dict of properties in `return_result`; see [Properties](#properties-driver) |
| any other value | `input_error` |

| `model.method` | multiplicity 1 | multiplicity > 1 |
|---|---|---|
| `hf`, `scf` | `run_rhf` | `run_uhf` |
| `mp2` | `run_rimp2` | `run_rimp2` on a UHF reference |
| `ccsd`, `ccsd(t)` | `run_ccsd`, `run_ccsd_t` (energy only) | `unsupported_error` |
| any other name | `run_dft` with that functional (RKS) | `run_uks` with that functional (UKS) |

`model.basis` is a bundled ferric basis name. For `mp2`, `ccsd` and
`ccsd(t)` the auxiliary basis is `keywords["auxbasis"]`, or by default
`"<basis>-ri"` (for example `cc-pvdz` uses `cc-pvdz-ri`); a basis with no such
alias needs `auxbasis`. An unknown functional is an `input_error`. ROHF and
ROKS are not reachable through QCSchema.

Open-shell Kohn-Sham runs `run_uks` for the energy and the gradient.
`dispersion` is closed-shell only: with multiplicity > 1 it gives
`unsupported_error`.

The Hessian driver is analytic and never a finite difference. `hf` and `scf`
call `run_frequencies` with `hessian="analytic"` (RHF, or UHF for multiplicity
> 1). Kohn-Sham methods use `hessian="analytic-moving-grid"` (RKS, or UKS for
multiplicity > 1), which covers LDA and GGA functionals, pure or global
hybrid. A setting without an analytic Hessian (for example an ECP, a
range-separated or meta-GGA functional, VV10, or an external potential)
gives `unsupported_error`.

Molecule fields used: `symbols`, `geometry`, `molecular_charge` (default 0),
`molecular_multiplicity` (default 1 for an even electron count, else 2) and
`real` (`False` marks a ghost atom). The other fields (`masses`, `fragments`,
`name`, ...) do not change the electronic energy and are echoed unchanged.

## Keywords

`keywords` is a strict allowlist per method family; any other key is an
`input_error` that names it. Values go to the ferric driver unchanged, so the
driver's own checks apply.

| Family | Allowed keywords |
|---|---|
| `hf`, `scf` | `max_iter`, `energy_conv`, `density_conv`, `diis_size`, `integral_thresh`, `level_shift`, `k_builder`, `df_j_aux`, `df_k_aux`, `ri_j_precision`, `ri_k_precision`, `memory_budget_gb` |
| `mp2` | `auxbasis`, `frozen_core`, `memory_budget_gb` |
| DFT | the SCF keywords above except `diis_size` and `integral_thresh`, plus `grid_radial`, `grid_angular`, `grid_prune`, `dispersion` |
| `ccsd`, `ccsd(t)` | `auxbasis`, `frozen_core`, `memory_budget_gb`, `k_builder`, `ladder`, `scf_max_iter`, `scf_energy_conv`, `scf_density_conv` |
| `hessian` driver (`hf`, `scf`, DFT) | `max_iter`, `energy_conv`, `density_conv`, `df_j_aux`, `df_k_aux` |
| `properties` driver | the family's keywords above, plus `properties` |

A gradient run is a geometry task: an unset `ri_j_precision` is `"f64"` there,
while an energy run keeps the driver default. Set `ri_j_precision` explicitly
when an energy and a gradient (or finite differences of energies) must lie on
the same surface.

## Properties driver

`driver="properties"` runs the method's energy and returns a dict as
`return_result`, in atomic units. `keywords["properties"]` is a list of names;
without it ferric returns the defaults for the reference. A name ferric cannot
compute fails the whole job; nothing is skipped or estimated.

| Name | Value | Methods | Notes |
|---|---|---|---|
| `scf_dipole_moment` | (3,) e·Bohr | hf, scf, DFT | about the coordinate origin; also in `properties` |
| `scf_quadrupole_moment` | (3, 3) e·Bohr² | closed-shell hf, scf, DFT | raw Cartesian second moment about the origin, not traceless; also in `properties` |
| `orbital_energies_alpha` | Hartree | hf, scf, DFT | `orbital_energies_beta` only for an unrestricted reference |
| `homo_energy`, `lumo_energy` | Hartree | hf, scf, DFT | over both spins; an error when no occupied or virtual orbital exists |
| `mulliken_charges`, `lowdin_charges` | e per atom | closed-shell hf, scf, DFT | defaults for a closed shell |
| `hirshfeld_charges`, `chelpg_charges` | e per atom | closed-shell hf, scf, DFT | computed only when named |
| `scf_total_energy`, `mp2_total_energy`, ... | Hartree | as the method gives | any energy that `properties` carries |

The default set for an SCF or Kohn-Sham reference is the dipole, the
quadrupole (closed shell), the orbital energies, `homo_energy`, `lumo_energy`
and, for a closed shell, the Mulliken and Löwdin charges. For `mp2`, `ccsd` and
`ccsd(t)` the default is the energies the run produced. A density property with
those methods, including `mp2_dipole_moment` and `ccsd_dipole_moment`, gives
`unsupported_error`: their results expose no density. An unknown name is an
`input_error`.

QCSchema's `AtomicResultProperties` has no field for orbital energies, HOMO and
LUMO, or charges, so those appear in `return_result` only. The dipole and
quadrupole are copied into `properties` under their schema names.

## Output

`return_result` is the energy, the (natom, 3) gradient or the (3N, 3N)
Hessian. `properties` holds `return_energy`, `return_gradient` (gradient
driver), `return_hessian` (hessian driver), `scf_total_energy`,
`scf_iterations` (SCF and DFT), `scf_xc_energy` (RKS), `mp2_correlation_energy`
and `mp2_total_energy` (mp2), `ccsd_correlation_energy` and
`ccsd_total_energy` (ccsd and ccsd(t)), `ccsd_prt_pr_correlation_energy` and
`ccsd_prt_pr_total_energy` (ccsd(t)), `nuclear_repulsion_energy`,
`calcinfo_natom`, `calcinfo_nalpha`, `calcinfo_nbeta`, `calcinfo_nbasis` for
every method and driver (`BasisSet.n_functions`), and `calcinfo_nmo` for SCF,
DFT and CC (the MP2 result does not expose the MO count). For `ccsd` and `ccsd(t)` the return energy is the total energy of
the highest level run, and the reference energy is a separate `run_rhf` call
with the same SCF controls.

`provenance` names ferric (`creator`, `version`, `routine`, `wall_time`) and
carries `build`, the commit and dirty flag of the extension
(`ferric.build_info()`). `extras["ferric"]["certificate"]` is the validation
certificate of the run, as a plain dict; see
[Certificates](../reference/validation.md#certificates).

`protocols.wavefunction` other than `"none"` fails with `unsupported_error`.
A QCSchema v1 wavefunction has no field for the AO ordering or normalisation,
and ferric's (libint: spherical components m = -l..l, Cartesian components
sharing the norm of x^l) differs from other producers', so a consumer would read
the orbital coefficients wrongly without any error. `protocols.native_files`
other than `"none"` fails the same way. Orbital energies are available through
the properties driver.

## Failures

A job ferric cannot run returns a `FailedOperation` dict, never an exception:
`{"success": False, "error": {"error_type": ..., "error_message": ...},
"input_data": <the input>}`. The `error_type` values are `input_error`
(malformed input, unknown keyword, method, functional or basis, a driver's
refusal of a keyword), `unsupported_error` (a valid request ferric does not
provide), `convergence_error` (the SCF did not converge) and
`execution_error` (anything else raised while running).

## Accuracy

`run` adds no arithmetic: the energy and gradient equal the direct driver call
with the same settings bit for bit, and the gradient matches a central finite
difference of `run` energies to the bar derived in
`crates/ferric-python/tests/test_qcschema.py`. There is no QCEngine harness
integration yet.
