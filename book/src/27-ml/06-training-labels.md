# 27.6 Energies, gradients, Hessians and response as labels

<!-- section: 27.6; status: edit;
     registry: hf-gradient, ks-gradient, ri-mp2-gradient, rpa-gradient, hf-hessian, ks-hessian, harmonic-frequencies;
     theory: none -->

A machine-learned potential trains on energies and forces, sometimes on
Hessians, and a property model trains on dipoles and polarizabilities. For
each of these, a dataset builder needs to know whether ferric computes the
label analytically or by finite differences, and how much numerical noise it
carries. This section collects that, method by method. Part VI derives the
analytic derivatives; here only their scope and accuracy matter.

## 27.6.1 Energies

The energy label in Python is the result's energy attribute (`energy` on SCF results, `total_energy` on `PdepRpaResult`). In the CLI's JSON run log the
`run_end` record holds the total only for the SCF kinds; for a post-SCF kind
it holds the SCF reference (`extra.energy_is` says which), and the method's
total is in a separate `result` record. `pdep-rpa` and some other kinds do
not write that record yet (`result_unlogged`), so their total is only on
stdout. It is not in the NPZ. Its trust level is the grade on the
certificate (section 27.4.3); the grades per `method.kind` are in
`site/src/reference/validation.md`, "CLI `method.kind` matrix".

The SCF energy has a rounding floor that ferric estimates as
$\tfrac12\,\epsilon_{\mathrm{mach}}\lvert E\rvert\sqrt{n}$, plus a
screening term on the incremental direct-J/K path
(`ferric-scf/src/noise_floor.rs::de_noise_floor`). It is used to warn when an
`energy_conv` cannot be met. It is an estimate of the per-iteration $\lvert\Delta
E\rvert$ noise, not of the error of the energy, and not a gradient floor.

Two defaults move energy labels between nearby geometries by more than that
floor: f32 RI-J storage on single points (up to about 1e-6 Ha) and the CLI's
RI-JK reference (section 27.5.2). Fix both for a dataset of geometries.

## 27.6.2 Nuclear gradients (forces)

| Method | Gradient | Agreement | Grade of the gradient |
|---|---|---|---|
| RHF, UHF, ROHF, RKS, UKS, ROKS, exact or density-fitted J/K | analytic, D3(BJ) and MBD@rsSCS included with `[dft] dispersion` | 1e-7 to 3e-7 Ha/Bohr vs FD of the energy; about 1e-10 vs PySCF `df.grad` | `hf-gradient`, `ks-gradient` |
| KS on a custom or pruned grid | analytic on that grid | 0.9e-7 to 1.4e-7 Ha/Bohr vs FD on the same grid | — |
| RI-MP2, closed shell | analytic (Z-vector) | ≤ 2.1e-9 Ha/Bohr vs PySCF FD; ≤ 5.1e-8 vs ORCA | `ri-mp2-gradient` |
| SCS-MP2, OO-RI-MP2 | analytic | — | not graded |
| PDEP-RPA, closed-shell RHF reference | analytic SCF part plus a central finite difference of the correlation energy (`ferric-rpa/src/gradient.rs::total_rpa_gradient`) | ≤ 1.9e-9 Ha/Bohr vs PySCF at the same step; ≤ 5.7e-8 vs a converged 5-point FD | `rpa-gradient` |
| every other `method.kind` | none in the CLI | — | — |

The figures are from `site/src/reference/validation.md`, "Anchors". The
1e-7 to 3e-7 for the SCF gradients is the floor of the finite-difference
check, not the error of the analytic gradient; the PySCF comparison is the
tighter one.

Two cautions:

- A KS gradient is the derivative of the energy **on the SCF's grid**. The
  validation records that pairing a default-grid gradient with an SCF on a
  different grid misses by 2.8e-4 to 6.0e-4 Ha/Bohr. Do not mix grids between
  the energy and force labels of one sample.
- Molecules with ghost atoms have no gradient: `rhf_gradient` returns an error
  (`ferric-scf/src/gradient.rs::rhf_gradient`). Counterpoise-style datasets
  get energies only.

## 27.6.3 Hessians and frequencies

`run_frequencies` (and the CLI `task = "frequencies"`) chooses the Hessian
construction under `hessian = "auto"`:

| Reference | Default construction | Agreement |
|---|---|---|
| RHF, UHF, ROHF (exact or RI J/K, ECPs allowed) | analytic | RHF ≤ 1.6e-7 Ha/Bohr² vs PySCF analytic; frequencies ≤ 6.9e-4 cm⁻¹ |
| RKS, UKS: LDA, GGA, global hybrids, exact or RI J/K, no ECP, unpruned grid | analytic on the moving grid (grid response included) | see the validation row "Analytic closed-shell KS Hessian on the MOVING grid" |
| everything else (ROKS, meta-GGA, VV10, range-separated hybrids, dispersion) | finite differences of analytic gradients | same-step vs PySCF 2.4e-7 Ha/Bohr², 7.5e-4 cm⁻¹; HF FD vs analytic 2.75e-5 Ha/Bohr², 0.10 cm⁻¹ (the 5e-3 Bohr step) |

A KS Hessian from PySCF's analytic code differs from ferric's by 5–35 cm⁻¹
because that Hessian has no grid response; this is a difference of
definition, and matters if a dataset mixes the two codes.

The `FrequencyResult` exposes what a label pipeline should record:

| Attribute | Meaning |
|---|---|
| `hessian_source` | `analytic`, `analytic-moving-grid` or `finite-difference` |
| `cartesian_hessian` | $3N\times3N$, Hartree/Bohr², order $x_0,y_0,z_0,x_1,\dots$, before mass weighting |
| `asymmetry` | largest $\lvert H_{ij}-H_{ji}\rvert$ of the raw Hessian, a direct noise measure for FD Hessians |
| `n_gradient_evaluations` | cost, and a check that FD actually ran |
| `normal_modes` | mass-weighted eigenvectors divided by $\sqrt m$, **not normalised** |

The certificate records the Hessian construction, the FD step and gradient
count, and the asymmetry as approximations, and each imaginary frequency as a
warning. Frequency tasks resolve RI-J to f64 because an FD Hessian
differences the f32 jitter. With `scf_warm_start=True` the FD Hessian is
noisier at equal `scf_density_conv`; the docstring asks for a roughly 100×
tighter density threshold.

## 27.6.4 Response properties

| Label | Source | Analytic or FD | Status |
|---|---|---|---|
| dipole | reference density (`dipole`) | analytic expectation value | exact function of $D$ |
| static and dynamic α | dRPA on the RHF reference, eq. 27.8 | analytic linear response | `rpa-polarizability-c6` |
| per-atom α, $\alpha_{\mathrm{CT}}$, $C_6$ | eqs. 27.9–27.12 | analytic | `rpa-polarizability-c6` |
| relaxed RI-MP2 α | `ferric-mp2/src/cpks_polar.rs::mp2_polarizability_analytic`; finite-field `ferric-mp2/src/ff_polar.rs::mp2_polarizability_static` | both, library only | registry gap `mp2-polarizability` |
| statically screened TDHF α | `run_tdhf_static_polarizability` | analytic | Smoke; statically screened RPAx on an RKS reference, 46% below DOSD for water/cc-pVDZ at a 0.36 Ha scissor (`site/src/reference/validation.md`, "Known limits") |

The dRPA α has no exchange kernel. It is the response of the RHF reference
to a field at the direct-RPA level, not a CPHF, CPKS or correlated
polarizability. A dataset that mixes it with polarizabilities from another
code or method mixes definitions.

## Sources

| Source | Used for |
|---|---|
| `site/src/reference/validation.md`, "CLI `method.kind` matrix" and "Anchors" | which kinds have gradients and Hessians, and every figure here |
| `crates/ferric-python/ferric.pyi`, `run_frequencies` and `FrequencyResult` | the Hessian choice, attributes, warm-start advice |
| `crates/ferric-scf/src/noise_floor.rs` | the energy noise floor |
| `crates/ferric-scf/src/gradient.rs` | the ghost-atom refusal |
| `crates/ferric-rpa/src/gradient.rs` | the PDEP-RPA gradient route |

Changes from the sources: none.
