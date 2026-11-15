# 14.1 Hartree–Fock nuclear gradient

<!-- section: 14.1; status: ready; registry: hf-gradient;
     theory: crates/ferric-scf/theory.md#"Hartree-Fock nuclear gradient" -->

A converged SCF energy is stationary in the orbital coefficients, so its
nuclear gradient needs no derivative of the orbitals. It needs the
derivatives of the integrals at fixed coefficients, and one extra term that
keeps the orbitals orthonormal as the basis moves. That extra term is the
Pulay term, and it is the only place the orbitals' eigenvalues enter.

| Symbol | Meaning | Code name |
|---|---|---|
| $x$ | one nuclear Cartesian coordinate | column of the `(natoms, 3)` result |
| $D$ | total AO density, $D = 2\,C_o C_o^{\mathsf T}$ (closed shell) | `result.density_r()`, `d` |
| $W$ | energy-weighted density, the AO form of the orthonormality multiplier | `w`, `build_energy_weighted_density` |
| $\Gamma_{\mu\nu\lambda\sigma}$ | two-particle density, $\sum\Gamma(\mu\nu\vert\lambda\sigma) = E_{2e}$ | `gamma`, `gamma_uhf` |
| $h^x$, $S^x$, $(\mu\nu\vert\lambda\sigma)^x$ | integral derivatives at fixed basis coefficients | derivative engines of `ferric-integrals` |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/gradient.rs::{rhf_gradient,uhf_gradient,rohf_gradient}`; the Kohn–Sham routes of section 14.3 call the same pieces |
| CLI | `task = "optimize"` for `rhf`, `uhf`, `rohf`, `ksdft` (`examples/h2-lda-opt.toml`) |
| Python | `run_rhf(..., with_gradient=True)`, `run_optimize`, `run_frequencies` |

This section covers the exact four-centre route. When the SCF density-fitted
J or K, the two-electron term differentiates the fitted energy instead
(section 14.2). COSX exchange has its own gradient
(`ferric-scf/src/gradient.rs::rhf_gradient_cosx`) and is not derived here.

## 14.1.1 The gradient

<!-- include: crates/ferric-scf/theory.md#"Hartree-Fock nuclear gradient" -->

> **Verified equation 14.1** · RHF, UHF and ROHF gradients equal finite differences
>
> On a toy system whose integrals are smooth functions of two parameters,
> with a non-orthogonal overlap, the gradient assembled from the code's
> $\Gamma$ and $W$ equals a fourth-order finite difference of the
> re-solved SCF energy: RHF, UHF and ROHF each.
>
> | | |
> |---|---|
> | Proof | `scripts/hf_gradient_sympy.py`, claims G1 (RHF), G2 (UHF), G3 (ROHF) |
> | Negative control | the eigenvalue form $C\,\mathrm{diag}(2\varepsilon_c,\varepsilon_o)\,C^{\mathsf T}$ of the ROHF $W$ must miss the finite difference (claim G5); mutations in the runner halve the RHF $W$, change the ROHF $W$ term and the UHF exchange weight |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/gradient.rs::{build_energy_weighted_density,build_energy_weighted_density_uhf,rohf_energy_weighted_density,gamma,gamma_uhf}` |
> | Code test | `test_gradient_h2o_sto3g_vs_finite_diff` in `crates/ferric-scf/src/gradient.rs` |
>
> <!-- verified: id=scf.grad.hf; script=scripts/hf_gradient_sympy.py; claim=G1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/gradient.rs::{build_energy_weighted_density,build_energy_weighted_density_uhf,rohf_energy_weighted_density,gamma,gamma_uhf}; rust_test=ferric-scf/src/gradient.rs::test_gradient_h2o_sto3g_vs_finite_diff; registry=hf-gradient -->

> **Verified equation 14.2** · the Pulay term is $-\mathrm{tr}[W\,S^x]$
>
> For one doubly occupied orbital in a two-function basis, at fixed
> normalized direction, $\mathrm dE/\mathrm dx$ equals the skeleton term
> minus $\mathrm{tr}[W S^x]$ with $W = 2\varepsilon\,c\,c^{\mathsf T}$, in
> symbols.
>
> | | |
> |---|---|
> | Proof | `scripts/hf_gradient_sympy.py`, claim S1 |
> | Negative control | the runner flips the sign of the Pulay term (claim G1 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/gradient.rs::oneelectron_gradient` |
>
> <!-- verified: id=scf.grad.pulay; script=scripts/hf_gradient_sympy.py; claim=S1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/gradient.rs::oneelectron_gradient; registry=hf-gradient -->

**Why $W$ for ROHF is not the eigenvalue form.** One orbital set carries two
Fock matrices, so the orthonormality multiplier is the matrix
$\varepsilon_{ij} = \sum_\sigma n_{\sigma j}(f_\sigma)_{ij}$ over occupied
orbitals, not a diagonal. In the AO basis it is $D_\alpha F_\sigma D_\sigma$
summed over spin, and the code symmetrizes it
(`ferric-scf/src/gradient.rs::rohf_energy_weighted_density`). The
eigenvalue form drops the closed-open coupling. It agrees only when
symmetry decouples the open shell from the closed shells, as it does for the
$\pi$ SOMO of OH, so an OH-only test cannot tell the two apart.

## 14.1.2 The algorithm

| Step | Quantity | Function |
|---|---|---|
| 1 | $E_{nn}^x$, with `effective_z()` for ECP atoms | `ferric-scf/src/gradient.rs::oneelectron_gradient` |
| 2 | $\sum D\,h^x$: kinetic, nuclear attraction, and, when present, point-charge and field terms | same |
| 3 | $-\sum W\,S^x$ with $W$ from the converged orbitals and eigenvalues | `ferric-scf/src/gradient.rs::build_energy_weighted_density` |
| 4 | $\sum\Gamma\,(\mu\nu\vert\lambda\sigma)^x$ over Schwarz-screened shell quartets | `ferric-scf/src/gradient.rs::twoelectron_gradient`, `ferric-scf/src/gradient.rs::twoelectron_gradient_uhf` |
| 5 | $\sum D\,V_{\mathrm{ECP}}^x$, a no-op without ECPs | `ferric-scf/src/gradient.rs::ecp_gradient` |

Only converged-SCF quantities enter: no response equations are solved. The
gradient costs one pass of derivative integrals, comparable to one Fock build
over the same quartets.

The kernel of step 3 scales the occupied columns by their eigenvalues once
and forms $W$ with one matrix product.

<!-- excerpt: ferric-scf/src/gradient.rs::build_energy_weighted_density -->
```rust
let c_occ = c.slice(ndarray::s![.., ..nocc]);
let eps_occ = ndarray::ArrayView1::from(&eps[..nocc]);
let cw = &c_occ * &eps_occ;
c_occ.dot(&cw.t()) * 2.0
```

| Code | Equation |
|---|---|
| `c_occ` | $C_o$ |
| `eps_occ` | $\varepsilon_i$ over occupied orbitals |
| `cw` | $\varepsilon_i\,C_{\mu i}$, columns scaled |
| the returned matrix | $W = 2\,C_o\,\mathrm{diag}(\varepsilon_o)\,C_o^{\mathsf T}$ |

The closed-shell two-particle density is one line.

<!-- excerpt: ferric-scf/src/gradient.rs::gamma -->
```rust
0.5 * d[(mu, nu)] * d[(la, sg)] - 0.25 * d[(mu, la)] * d[(nu, sg)]
```

with `d` the total density $D$: Coulomb weight $\tfrac12 D_{\mu\nu}D_{\lambda\sigma}$
and exchange weight $-\tfrac14 D_{\mu\lambda}D_{\nu\sigma}$.

## 14.1.3 Reproducibility and refusals

- The gradient is bit-identical across rayon thread counts
  (`rhf_gradient_bit_identical_across_thread_counts` in
  `crates/ferric-scf/src/gradient.rs`).
- Fractional occupations: the UHF $W$ uses the occupations that built the
  stored density, recovered by projection and snapped to the aufbau or the
  fractional pattern. A density that matches no pattern is refused, because a
  Pulay term built on the wrong occupations is silently wrong.
- Ghost atoms are refused: derivatives with respect to a ghost centre need a
  different one-electron derivative structure.
- A relativistic (X2C) SCF is refused by
  `ferric-scf/src/x2c.rs::refuse_relativistic`; its gradient is not
  implemented.

## 14.1.4 Options that change the equation

| Option | Effect | Where |
|---|---|---|
| ECPs | adds $\sum D V^x_{\mathrm{ECP}}$ and uses $Z_{\mathrm{eff}}$ in $E_{nn}^x$ | this section |
| external point charges and field | folded into the one-electron term; the field-density term is a finite difference | `ferric-scf/src/gradient.rs::oneelectron_gradient` |
| density-fitted J and K | the two-electron term differentiates the fitted energy | section 14.2 |
| COSX exchange | exchange differentiated on the COSX grid | `ferric-scf/src/gradient.rs::rhf_gradient_cosx` |
| Kohn–Sham | adds the XC gradient and scales exchange by the hybrid fraction | section 14.3 |
| solvent (PCM) | adds the analytic IEF-PCM nuclear gradient | chapter 22 |

## 14.1.5 Numerical anchors

| Check | Reference | Agreement | Test |
|---|---|---|---|
| RHF, UHF, ROHF and KS gradients, including density-fitted J/K: water, OH, HO2 / cc-pVDZ, 6-31G | finite differences of the energy; PySCF `df.grad` | 1e-7 to 3e-7 Ha/Bohr (FD); about 1e-10 (PySCF), bar 1e-6 Ha/Bohr | `crates/ferric-scf/tests/df_jk_gradient.rs` |
| RHF and UHF with def2 ECPs: HI, CH3I, RbH, I atom / def2-SVP, def2-TZVP; SnH4 / def2-TZVP; HBr (all-electron control) | PySCF analytic gradient and ORCA, both fed ferric's basis and ECP | at most 2.3e-8 Ha/Bohr, bar 1e-7 Ha/Bohr | `crates/ferric-scf/tests/validation_ecp.rs` |
| Geometry optimization of H2O, NH3, CH2O from distorted starts, RHF and B3LYP | PySCF gradient driven to $10^{-6}$ Ha/Bohr | distances at most 2.6e-6 Bohr, bar 2e-5 Bohr | `crates/ferric-scf/tests/validation_geometry_optimization.rs` |

<!-- figures: site/src/reference/validation.md "RHF/UHF/ROHF/KS gradients, including density-fitted J/K"; site/src/reference/validation.md "RHF and UHF with def2 ECPs: energies and analytic gradients"; site/src/reference/validation.md "Geometry optimization, RHF and RKS" -->

The registry's `hf-gradient` entry is the toy-model proof above; the
finite-difference rows are the end-to-end check of the same assembly on real
integrals.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/theory.md`, "Hartree-Fock nuclear gradient" | the derivation, the $\Gamma$ and $W$ table, the toy-system check |
| `scripts/hf_gradient_sympy.py`, claims G1, G2, G3, G5, S1 | verified equations 14.1 and 14.2 |
| `crates/ferric-scf/src/gradient.rs` | the algorithm, refusals and options |
| `site/src/reference/validation.md` | the anchor rows |
