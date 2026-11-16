# 14.6 Harmonic frequencies

<!-- section: 14.6; status: ready; registry: harmonic-frequencies, rrho-thermo, ir-intensities;
     theory: crates/ferric-scf/theory.md#"Harmonic frequencies"; crates/ferric-scf/theory.md#"RRHO thermochemistry"; crates/ferric-scf/theory.md#"Infrared intensities" -->

Harmonic frequencies need only the Cartesian Hessian and the masses. Whether
the Hessian came from section 14.4, from section 14.5 or from differences of
analytic gradients does not matter to what follows: one linear-algebra
routine turns it into wavenumbers. The only subtle step is removing the six
translations and rotations, which must be done in mass-weighted coordinates and
must not misclassify a small imaginary mode at a saddle point.

| Symbol | Meaning | Code name |
|---|---|---|
| $H$ | Cartesian Hessian, Hartree/Bohr², index $3A+\alpha$ | `cartesian_hessian` |
| $M$ | diagonal mass matrix, masses in electron masses | `mass_au` |
| $\tilde H = M^{-1/2}HM^{-1/2}$ | mass-weighted Hessian | `mw` |
| $t_k$ | orthonormal translation and rotation vectors, mass weighted | `tr_basis` |
| $P = 1-\sum_kt_kt_k^{\mathsf T}$ | projector off the rigid motions | `proj` |
| $\omega_k$, $\tilde\nu_k$ | angular frequency in atomic units, wavenumber in cm⁻¹ | `evals`, `to_cm` |

| Entry point | Location |
|---|---|
| Rust, Hessian to frequencies | `ferric-scf/src/frequencies.rs::frequencies_from_cartesian_hessian` |
| Rust, drivers | `ferric-scf/src/frequencies.rs::{harmonic_frequencies,harmonic_frequencies_analytic,harmonic_frequencies_from_gradient}` |
| CLI | `task = "frequencies"`, section `[frequencies]` |
| Python | `run_frequencies` |

## 14.6.1 From Hessian to wavenumbers

<!-- include: crates/ferric-scf/theory.md#"Harmonic frequencies" -->

> **Verified equation 14.15** · units and the diatomic limit
>
> The two conversion constants equal their CODATA 2018 forms
> ($\texttt{AU\_FREQ\_TO\_CM} = E_h/(hc)$ and the u-to-electron-mass ratio) to
> $10^{-10}$ relative; the mass-weighted diatomic spring has eigenvalues
> $\{0, k/\mu\}$, in symbols; and the pipeline's diatomic wavenumber equals
> the SI route $(2\pi c)^{-1}\sqrt{k/\mu}$ to $10^{-9}$ relative.
>
> | | |
> |---|---|
> | Proof | `scripts/hf_frequencies_sympy.py`, claims C1, C2, S1 and D1 |
> | Negative control | the runner drops the factor $2\pi$ from the wavenumber constant and replaces $M^{-1/2}$ by $M^{-1}$ (claim D1 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/frequencies.rs::frequencies_from_cartesian_hessian` |
> | Code test | `anchor_postprocessing_matches_pyscf` in `crates/ferric-scf/tests/validation_frequencies.rs` |
>
> <!-- verified: id=freq.units-diatomic; script=scripts/hf_frequencies_sympy.py; claim=D1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/frequencies.rs::frequencies_from_cartesian_hessian; rust_test=ferric-scf/tests/validation_frequencies.rs::anchor_postprocessing_matches_pyscf; registry=harmonic-frequencies -->

> **Verified equation 14.16** · the projector
>
> For a nonlinear molecule the translation and rotation basis has rank 6, is
> orthonormal, and $P$ is idempotent and annihilates every rigid mode, which
> preserve all interatomic distances (claim P1). A linear molecule has rank 5
> (claim P2). At a minimum of a spring model the projection leaves the
> vibrational eigenvalues unchanged (claim P3). Away from the minimum the
> rotations carry curvature, and the projection zeroes exactly six modes
> (claim P4).
>
> | | |
> |---|---|
> | Proof | `scripts/hf_frequencies_sympy.py`, claims P1 to P4 |
> | Negative control | the runner flips the sign of a rotation-vector component (claim P1 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/frequencies.rs::{translation_rotation_basis,detect_linear}` |
> | Code test | `anchor_postprocessing_matches_pyscf` in `crates/ferric-scf/tests/validation_frequencies.rs` |
>
> <!-- verified: id=freq.projector; script=scripts/hf_frequencies_sympy.py; claim=P1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/frequencies.rs::{translation_rotation_basis,detect_linear}; rust_test=ferric-scf/tests/validation_frequencies.rs::anchor_postprocessing_matches_pyscf; registry=harmonic-frequencies -->

**A negative eigenvalue is an imaginary frequency.** It is reported as a
negative wavenumber, $-\sqrt{\lvert\lambda\rvert}\,\mathtt{AU\_FREQ\_TO\_CM}$.
Eigenvectors are classified as rigid motion by their overlap with the
translation and rotation subspace (above $0.5$), not by the size of their
eigenvalue, because at a transition state a genuine imaginary mode can be
smaller in magnitude than the numerical zeros.

## 14.6.2 The algorithm

| Step | Quantity | Function |
|---|---|---|
| 1 | $H$: analytic (14.4, 14.5) or central differences of the analytic gradient, $H_{ab} = [g_a(x_b+\delta)-g_a(x_b-\delta)]/2\delta$, $\delta = 5\times10^{-3}$ Bohr, $6N$ gradients | `ferric-scf/src/frequencies.rs::fd_hessian_frequencies` |
| 2 | $H\leftarrow\tfrac12(H+H^{\mathsf T})$; the largest asymmetry is reported as a noise probe, `FrequencyResult::asymmetry` | same |
| 3 | $\tilde H_{ij} = H_{ij}/\sqrt{m_im_j}$ | `frequencies_from_cartesian_hessian` |
| 4 | $P\tilde HP$ by rank-one sweeps, one per rigid-body vector | same |
| 5 | `eigh_dc`, classify rigid modes by overlap, convert $\omega$ to cm⁻¹, back-transform modes $x = M^{-1/2}u$ | same |

The finite-difference route is general. It needs only an analytic gradient,
and there is deliberately no fallback that differences energies twice: a
method without an analytic gradient is an error naming the method. Differencing a
gradient once leaves $O(\delta^2)$ truncation plus $O(\epsilon_{\mathrm{SCF}}/\delta)$
noise, against $O(\epsilon_{\mathrm{SCF}}/\delta^2)$ for energies.

Steps 3 to 5 are the whole kernel. Mass weighting and one rank-one projection sweep:

<!-- excerpt: ferric-scf/src/frequencies.rs::frequencies_from_cartesian_hessian -->
```rust
mw[(i, j)] = cartesian_hessian[(i, j)] * inv_sqrt_m[i] * inv_sqrt_m[j];
```

<!-- excerpt: ferric-scf/src/frequencies.rs::frequencies_from_cartesian_hessian -->
```rust
proj[(i, j)] -= t[i] * ht[j] + ht[i] * t[j] - t[i] * t[j] * tht;
```

| Code | Equation |
|---|---|
| `inv_sqrt_m[i]` | $m_i^{-1/2}$ for Cartesian coordinate $i$, atom $i/3$ |
| `mw` | $\tilde H$ |
| `t` | one orthonormal rigid-motion vector $t_k$ |
| `ht` | $\tilde H\,t_k$ |
| `tht` | $t_k^{\mathsf T}\tilde Ht_k$ |
| the update | $\tilde H\leftarrow\tilde H - t(t^{\mathsf T}\tilde H) - (\tilde Ht)t^{\mathsf T} + t\,(t^{\mathsf T}\tilde Ht)\,t^{\mathsf T}$ |

The sweeps equal $P\tilde HP$ because the $t_k$ are orthonormal; the matrix is
re-symmetrized afterwards to remove accumulated round-off.

## 14.6.3 Reproducibility and refusals

- The rank check is a refusal: a translation and rotation subspace of rank
  other than 6, or 5 for a linear molecule, means a degenerate geometry
  (coincident atoms) and is an error.
- Masses are isotope averages (`ferric-scf/src/frequencies.rs::atom_masses`)
  for $Z = 1$ to $54$; a heavier element is an error naming it.
- Moments of inertia below `LINEARITY_THRESH` ($10^{-6}$ Bohr) classify a
  molecule as linear.
- With `scf_warm_start` off (the default), each displaced SCF starts cold. A
  warm start from the undisplaced orbitals is measurably noisier at equal
  density convergence, so it is opt-in.

## 14.6.4 Options that change the equation

| Option | Effect |
|---|---|
| `hessian = "auto"` (default) | analytic where the availability predicate of 14.4 or 14.5 accepts the system, finite differences otherwise; the choice is made before the SCF |
| `hessian = "analytic"` | the Hartree–Fock analytic Hessian only; a Kohn–Sham reference is refused before the SCF |
| `hessian = "analytic-moving-grid"` | the analytic RKS or UKS Hessian of 14.5 |
| `hessian = "fd"` | central differences of the analytic gradient |
| `delta` | FD displacement; unused by analytic Hessians |
| dispersion (D3(BJ), MBD@rsSCS) | finite differences of the gradient on the corrected surface (`harmonic_frequencies_with_scf_correction`); never the analytic Hessian, so the SCF and correction parts come from one construction |

## 14.6.5 Numerical anchors

| Check | Reference | Agreement | Test |
|---|---|---|---|
| Harmonic frequencies from FD of analytic gradients: H2O, NH3 × RHF, PBE, B3LYP / 6-31G, cc-pVDZ; UHF OH, CH3, HO2, UKS-PBE, ROHF CH3 / 6-31G | PySCF same-step FD of analytic gradients; PySCF analytic Hessians; `thermo.harmonic_analysis` with ferric's masses | same-step FD 2.4e-7 Ha/Bohr² and 7.5e-4 cm⁻¹, bars 2e-6 and 5e-3; HF against the analytic Hessian 2.75e-5 Ha/Bohr² and 0.10 cm⁻¹ (the 5e-3 Bohr step's truncation), bars 1e-4 and 0.5; KS against PySCF's analytic Hessian differs by 5 to 35 cm⁻¹ because that Hessian has no grid response | `crates/ferric-scf/tests/validation_frequencies.rs` |
| Analytic RHF Hessian frequencies: H2O, NH3, CH2O / cc-pVDZ | PySCF `hessian.rhf` | at most 6.9e-4 cm⁻¹, bar 1e-2 | `crates/ferric-scf/tests/validation_rhf_hessian.rs` |

<!-- figures: site/src/reference/validation.md "Harmonic frequencies (FD of analytic gradients; Cartesian Hessian and cm⁻¹)"; site/src/reference/validation.md "Analytic RHF Hessian (skeleton one- and two-electron, overlap/W and CPHF response terms) and its harmonic frequencies" -->

**Known limits.** The analytic Hessians cover the references and options listed in
sections 14.4 and 14.5. UHF states are not checked for internal stability, so
an unstable UHF solution gets the frequencies of that solution.

## 14.6.6 Thermochemistry and IR intensities

A frequency run ends with two quantities built on the modes: the ideal-gas
thermochemistry of the rigid rotor and harmonic oscillator, and the IR
intensity of each mode.

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/thermo.rs::rrho_thermochemistry`, `ferric-scf/src/infrared.rs::{ir_spectrum,dipole_derivatives_field,ir_intensities}` |
| CLI | `task = "frequencies"`, keys `temperature`, `pressure`, `symmetry_number`, `ir_intensities` |
| Python | `FrequencyResult.thermo`, `FrequencyResult.ir_intensities` |

<!-- include: crates/ferric-scf/theory.md#"RRHO thermochemistry" -->

> **Verified equation 14.17** · RRHO partition functions
>
> Each closed form follows from its partition function in symbols, with the
> classical and low-temperature limits of the oscillator; for water the
> transcribed formulas equal $\ln Q(T)$ summed over the vibrational levels and
> differentiated numerically to $10^{-10}$ relative.
>
> | | |
> |---|---|
> | Proof | `scripts/rrho_thermo_sympy.py`, claims S1 to S3, L1 and N1 |
> | Negative control | the runner drops the symmetry number from the nonlinear rotor (claim N1 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/thermo.rs::rrho_thermochemistry_with_masses` |
> | Code test | `thermo_matches_exact_in_the_trivial_limit` in `crates/ferric-scf/src/thermo.rs` |
>
> <!-- verified: id=freq.rrho-thermo; script=scripts/rrho_thermo_sympy.py; claim=N1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/thermo.rs::rrho_thermochemistry_with_masses; rust_test=ferric-scf/src/thermo.rs::thermo_matches_exact_in_the_trivial_limit; registry=rrho-thermo -->

<!-- include: crates/ferric-scf/theory.md#"Infrared intensities" -->

> **Verified equation 14.18** · the intensity and its units
>
> The constant equals $N_Ae^2/(12arepsilon_0c^2m_e)$ from CODATA 2018;
> the $0	o1$ band strength from the harmonic matrix element is the classical
> one; contracting the dipole derivative with the stored modes equals
> differencing the dipole along each mode of a model triatomic; and the field
> route recovers the dipole derivative to its $O(h^2)$ truncation.
>
> | | |
> |---|---|
> | Proof | `scripts/ir_intensity_sympy.py`, claims C1, S1, N1 to N3 |
> | Negative control | the runner drops the mass weighting of the modes (claim N2 fails) and flips the sign of the field route (claim N3 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/infrared.rs::ir_intensities` |
> | Code test | `ir_intensity_matches_exact_in_the_trivial_limit` in `crates/ferric-scf/src/infrared.rs` |
>
> <!-- verified: id=freq.ir-intensity; script=scripts/ir_intensity_sympy.py; claim=N1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/infrared.rs::ir_intensities; rust_test=ferric-scf/src/infrared.rs::ir_intensity_matches_exact_in_the_trivial_limit; registry=ir-intensities -->

**The symmetry number is an input.** ferric detects Abelian point groups
only, so it cannot count the proper rotations of NH3 or CH4; the default
$\sigma = 1$ overstates the entropy by $k\ln\sigma$.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/theory.md`, "Harmonic frequencies" | the equations and the constants |
| `scripts/hf_frequencies_sympy.py`, claims C1, C2, S1, D1, P1 to P4 | verified equations 14.15 and 14.16 |
| `crates/ferric-scf/src/frequencies.rs` module documentation | the pipeline and the Hessian options |
| `site/src/reference/validation.md` | the anchor rows |
| `crates/ferric-scf/theory.md`, "RRHO thermochemistry" and "Infrared intensities" | section 14.6.6 |
| `scripts/rrho_thermo_sympy.py`, `scripts/ir_intensity_sympy.py` | verified equations 14.17 and 14.18 |
