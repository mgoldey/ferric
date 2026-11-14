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
| `properties` | `unsupported_error` |
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

A gradient run is a geometry task: an unset `ri_j_precision` is `"f64"` there,
while an energy run keeps the driver default. Set `ri_j_precision` explicitly
when an energy and a gradient (or finite differences of energies) must lie on
the same surface.

## Output

`return_result` is the energy, the (natom, 3) gradient or the (3N, 3N)
Hessian. `properties` holds `return_energy`, `return_gradient` (gradient
driver), `return_hessian` (hessian driver), `scf_total_energy`,
`scf_iterations` (SCF and DFT), `scf_xc_energy` (RKS), `mp2_correlation_energy`
and `mp2_total_energy` (mp2), `ccsd_correlation_energy` and
`ccsd_total_energy` (ccsd and ccsd(t)), `ccsd_prt_pr_correlation_energy` and
`ccsd_prt_pr_total_energy` (ccsd(t)), `nuclear_repulsion_energy`,
`calcinfo_natom`, `calcinfo_nalpha`, `calcinfo_nbeta`, and for SCF and DFT
`calcinfo_nbasis` and `calcinfo_nmo` (the MP2 result does not expose the
basis size). For `ccsd` and `ccsd(t)` the return energy is the total energy of
the highest level run, and the reference energy is a separate `run_rhf` call
with the same SCF controls.

`provenance` names ferric (`creator`, `version`, `routine`, `wall_time`) and
carries `build`, the commit and dirty flag of the extension
(`ferric.build_info()`). `extras["ferric"]["certificate"]` is the validation
certificate of the run, as a plain dict; see
[Certificates](../reference/validation.md#certificates).

`protocols.wavefunction` other than `"none"` fails with `unsupported_error`:
no wavefunction is serialised.

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
