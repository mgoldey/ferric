# Energies, gradients, Hessians and response as labels {#training-labels}

<!-- status: edit;
     registry: hf-gradient, ks-gradient, ri-mp2-gradient, u-ri-mp2-gradient, scs-mp2-gradient, att-rimp2-gradient, u-att-rimp2-gradient, terfc-rimp2-gradient, scs-2terfc-gradient, oo-rimp2-gradient, rpa-gradient, hf-hessian, ks-hessian, harmonic-frequencies -->

A machine-learned potential trains on energies and forces, sometimes on
Hessians, and a property model trains on dipoles and polarizabilities. The
model fits whatever noise the labels carry. A force from a finite difference
carries a step-size error, a force from a different grid than its energy is
the derivative of a different surface, and an energy computed with a cheap
storage format jitters between nearby geometries. So for each label a
dataset builder needs to know how ferric computes it, analytically or by
finite differences, and how much numerical noise it carries.

This section collects that, method by method: energies (@sec:training-labels-energies), forces
(@sec:training-labels-nuclear-gradients-forces), Hessians (@sec:training-labels-hessians-frequencies) and response properties (@sec:training-labels-response-properties). Chapters
@sec:derivatives and @sec:correlated-gradients derive the analytic
derivatives; here only their scope and accuracy matter.

## Energies {#training-labels-energies}

In Python the energy label is the result's energy attribute: `energy` on SCF
results, `total_energy` on `PdepRpaResult`. In the CLI's JSON run log the
`run_end` record holds the total only for the SCF kinds. For a post-SCF kind
it holds the SCF reference (`extra.energy_is` says which), and the method's
total is in a separate `result` record. `pdep-rpa` and some other kinds do
not write that record yet (`result_unlogged`), so their total is only on
stdout. It is not in the NPZ. Its trust level is the grade on the
certificate (section @sec:python-cube-provenance-certificate-provenance-label); the grades per `method.kind` are in
`site/src/reference/validation.md`, "CLI `method.kind` matrix".

How precise is an SCF energy to begin with? ferric estimates its rounding
floor as $\tfrac12\,\epsilon_{\mathrm{mach}}\lvert E\rvert\sqrt{n}$, plus a
screening term on the incremental direct-J/K path
(`ferric-scf/src/noise_floor.rs::de_noise_floor`). The code uses it to warn
when an `energy_conv` cannot be met. It estimates the per-iteration
$\lvert\Delta E\rvert$ noise. It is not an estimate of the error of the
energy, and not a gradient floor.

Two defaults move energy labels between nearby geometries by more than that
floor: f32 RI-J storage on single points (up to about 1e-6 Ha) and the CLI's
RI-JK reference (section @sec:reproducibility-and-grades-precision-convergence-defaults-move). Fix both for a dataset of geometries.

## Nuclear gradients (forces) {#training-labels-nuclear-gradients-forces}

| Method | Gradient | Agreement | Derivation | Certificate grade of `task = "optimize"` |
|---|---|---|---|---|
| RHF, UHF, ROHF, RKS, UKS, ROKS, exact or density-fitted J/K | analytic, D3(BJ) and MBD@rsSCS included with `[dft] dispersion` | 1e-7 to 3e-7 Ha/Bohr vs FD of the energy; about 1e-10 vs PySCF `df.grad` | `hf-gradient`, `ks-gradient` | Proven |
| KS on a custom or pruned grid | analytic on that grid | 0.9e-7 to 1.4e-7 Ha/Bohr vs FD on the same grid | `ks-gradient` | Proven (`ksdft`) |
| RI-MP2, closed shell | analytic (Z-vector) | ≤ 2.1e-9 Ha/Bohr vs PySCF FD; ≤ 5.1e-8 vs ORCA | `ri-mp2-gradient` | Proven |
| RI-MP2, UHF reference | analytic (Z-vector) | ≤ 3.1e-9 Ha/Bohr vs Richardson FD of own energy (PySCF has no density-fitted UMP2 gradient) | `u-ri-mp2-gradient` | Proven (`rimp2`; the page's gradient cell says Proven (narrow)) |
| SCS-MP2 (closed shell, exact or RI-JK) | analytic | ≤ 2.5e-9 Ha/Bohr vs PySCF FD; ≤ 5.7e-8 vs own FD | `scs-mp2-gradient` | Proven (narrow) |
| attenuated RI-MP2 (`att-rimp2`, erfc and terfc; closed shell and UHF) | analytic (Z-vector) | closed-shell erfc ≤ 2.5e-9 Ha/Bohr vs Richardson FD of own energy; UHF erfc ≤ 4.0e-10 vs PySCF FD | `att-rimp2-gradient`, `u-att-rimp2-gradient`, `terfc-rimp2-gradient` | Proven (narrow) |
| SCS-MP2(2terfc) (`scs-mp2-2terfc`, closed shell and UHF) | analytic (one Z-vector) | ≤ 1.6e-9 Ha/Bohr vs Richardson FD of own energy | `scs-2terfc-gradient` | Proven (narrow) |
| OO-RI-MP2 (closed shell, frozen core 0) | analytic | ≤ 1.8e-9 Ha/Bohr vs own FD | `oo-rimp2-gradient` | Proven (narrow) |
| PDEP-RPA, closed-shell RHF reference | analytic by default (Z-vector relaxed RHF + RI-dRPA, `ferric-rpa/src/gradient_analytic.rs::rpa_analytic_gradient_warm`) for full-rank PDEP, dense χ₀ and the Coulomb operator; any other configuration, or `[rpa] gradient = "fd"`, uses a central finite difference of the total energy at $h = 5\times10^{-4}$ Bohr (`ferric-rpa/src/gradient.rs::total_rpa_gradient`) | analytic: vs a step-converged 5-point FD of the same quadrature energy (`ferric-rpa/tests/rpa_analytic_gradient.rs`); FD: ≤ 1.9e-9 Ha/Bohr vs PySCF at the same step, ≤ 5.7e-8 vs a converged 5-point FD | `rpa-gradient` | Proven |
| every other `method.kind` | none in the CLI | — | — | not graded or refused |

The figures are from `site/src/reference/validation.md`, "Anchors". The
Derivation column names the registry method whose book section and sympy
proof cover the gradient. The last column is the grade the certificate of an
optimization carries (`TASK_KIND_ROWS` and `TASK_KIND_GRADE_CAPS` in
`crates/ferric-core/src/certificate.rs`): the kind's energy grade, capped for
`scs-mp2` and `att-rimp2`. The
1e-7 to 3e-7 for the SCF gradients is the floor of the finite-difference
check, not the error of the analytic gradient; the PySCF comparison is the
tighter one. The PDEP-RPA certificate records which route ran and, when the
default fell back to finite differences, why.

Two cautions:

- A KS gradient is the derivative of the energy **on the SCF's grid**. The
  validation records that pairing a default-grid gradient with an SCF on a
  different grid misses by 2.8e-4 to 6.0e-4 Ha/Bohr. Do not mix grids
  between the energy and force labels of one sample.
- Molecules with ghost atoms have no gradient: `rhf_gradient` returns an
  error (`ferric-scf/src/gradient.rs::rhf_gradient`). Counterpoise-style
  datasets get energies only.

## Hessians and frequencies {#training-labels-hessians-frequencies}

`run_frequencies` (and the CLI `task = "frequencies"`) chooses the Hessian
construction under `hessian = "auto"`. An analytic Hessian is preferred where
one exists; otherwise ferric differences analytic gradients.

| Reference | Default construction | Agreement |
|---|---|---|
| RHF, UHF, ROHF (exact or RI J/K, ECPs allowed) | analytic | RHF ≤ 1.6e-7 Ha/Bohr² vs PySCF analytic; frequencies ≤ 6.9e-4 cm⁻¹ |
| RKS, UKS: LDA, GGA, global hybrids, exact or RI J/K, ECPs allowed, optionally D3(BJ) (its analytic Hessian is added), unpruned grid | analytic on the moving grid (grid response included) | see the validation row "Analytic closed-shell KS Hessian on the MOVING grid" |
| everything else (ROKS, meta-GGA, VV10, range-separated hybrids, MBD@rsSCS) | finite differences of analytic gradients | same-step vs PySCF 2.4e-7 Ha/Bohr², 7.5e-4 cm⁻¹; HF FD vs analytic 2.75e-5 Ha/Bohr², 0.10 cm⁻¹ (the 5e-3 Bohr step) |

A KS Hessian from PySCF's analytic code differs from ferric's by 5–35 cm⁻¹,
because that Hessian has no grid response. This is a difference of
definition. It matters when a dataset mixes the two codes.

The `FrequencyResult` exposes what a label pipeline should record:

| Attribute | Meaning |
|---|---|
| `hessian_source` | `analytic`, `analytic-moving-grid` or `finite-difference` |
| `cartesian_hessian` | $3N\times3N$, Hartree/Bohr², order $x_0,y_0,z_0,x_1,\dots$, before mass weighting |
| `asymmetry` | largest $\lvert H_{ij}-H_{ji}\rvert$ of the raw Hessian, a direct noise measure for FD Hessians |
| `n_gradient_evaluations` | cost, and a check that FD ran |
| `normal_modes` | mass-weighted eigenvectors divided by $\sqrt m$, **not normalised** |

Why is `asymmetry` a noise measure? The two mixed partials are equal in exact
arithmetic. A finite-difference Hessian takes $H_{ij}$ from gradients at
geometries displaced along one coordinate and $H_{ji}$ from gradients
displaced along the other, so their difference shows the step and
convergence error. ferric then symmetrizes, $H \leftarrow (H + H^{\mathsf T})/2$,
before mass weighting (`ferric-scf/src/frequencies.rs`).

The certificate records the Hessian construction, the FD step and gradient
count, and the asymmetry as approximations, and each imaginary frequency as a
warning. Frequency tasks resolve RI-J to f64, because an FD Hessian would
difference the f32 jitter. With `scf_warm_start=True` the FD Hessian is
noisier at equal `scf_density_conv`; the docstring asks for a roughly 100×
tighter density threshold.

## Response properties {#training-labels-response-properties}

| Label | Source | Analytic or FD | Status |
|---|---|---|---|
| dipole | reference density (`dipole`) | analytic expectation value | exact function of $D$ |
| static and dynamic α | dRPA on the RHF reference, (@eq:polarizability-pdep-c6-1) | analytic linear response | `rpa-polarizability-c6` |
| per-atom α, $\alpha_{\mathrm{CT}}$, $C_6$ | (@eq:ml-alpha-per-atom-kernel)–(@eq:ml-c6-cp-quadrature) | analytic | `rpa-polarizability-c6` |
| relaxed RI-MP2 α | `ferric-mp2/src/cpks_polar.rs::mp2_polarizability_analytic`; finite-field `ferric-mp2/src/ff_polar.rs::mp2_polarizability_static` | both, library only | `mp2-polarizability` (section @sec:mp2-polarizability) |
| statically screened TDHF α | `run_tdhf_static_polarizability` | analytic | Smoke; statically screened RPAx on an RKS reference, α set by basis and scissor: water/cc-pVDZ at a 0.36 Ha scissor is 3% below CP-PBE in the same basis (`site/src/reference/validation.md`, "Known limits") |

The dRPA α has no exchange kernel. It is the response of the RHF reference
to a field at the direct-RPA level, not a CPHF, CPKS or correlated
polarizability. A dataset that mixes it with polarizabilities from another
code or method mixes definitions.

## Trajectories as a label source {#training-labels-trajectories}

Where do the geometries of an energy-and-force dataset come from? A
molecular-dynamics trajectory (chapter @sec:md) samples the thermally
reachable region of the surface that the model will be asked about. Two
facts make a trajectory frame usable as a label directly.

- Each frame's force is the analytic SCF gradient of the table in section
  @sec:training-labels-nuclear-gradients-forces (`rhf`, `uhf`, `rohf` or
  `ksdft`), with the grade and agreement given there.
- The CLI `task = "md"` and the Python `run_aimd` mark their SCF as a
  geometry task, so an unset `ri_j_precision` resolves to f64
  (`ferric-cli/src/lib.rs::ri_j_task_for`). The f32 jitter of section
  @sec:reproducibility-and-grades-precision-convergence-defaults-move does
  not enter the trajectory's energies.

The trajectory itself is ungraded: the certificate of an MD run says so and
records the largest drift of the conserved quantity
(@sec:molecular-dynamics-proven). The drift checks the integrator and the
SCF convergence along the run, not the labels.

## Sources

| Source | Used for |
|---|---|
| `site/src/reference/validation.md`, "CLI `method.kind` matrix" and "Anchors" | which kinds have gradients and Hessians, and every figure here |
| `crates/ferric-python/ferric.pyi`, `run_frequencies` and `FrequencyResult` | the Hessian choice, attributes, warm-start advice |
| `crates/ferric-scf/src/noise_floor.rs` | the energy noise floor |
| `crates/ferric-scf/src/gradient.rs` | the ghost-atom refusal |
| `crates/ferric-rpa/src/gradient.rs`, `crates/ferric-rpa/src/gradient_analytic.rs` | the PDEP-RPA gradient routes and the default choice |
| `crates/ferric-core/src/certificate.rs` | the task grades of the gradient table |
| `crates/ferric-cli/src/lib.rs`, `crates/ferric-cli/src/md.rs` | the RI-J precision of `task = "md"` and the MD certificate |

Changes from the sources: none.
