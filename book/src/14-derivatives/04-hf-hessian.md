# 14.4 Hartree–Fock Hessian and coupled-perturbed equations

<!-- section: 14.4; status: ready; registry: hf-hessian;
     theory: crates/ferric-scf/theory.md#"Hartree-Fock nuclear Hessian" -->

The Hessian is the derivative of the gradient of section 14.1, and the
gradient depends on the orbitals through $D$ and $W$. Differentiating it has
two parts. The skeleton is the second derivative of the energy at fixed $D$
and $W$, built from second derivatives of the integrals. The response is how
$D$ and $W$ move when a nucleus does, which needs the first-order change of
the orbitals. That change solves a linear system, the coupled-perturbed
Hartree–Fock (CPHF) equations, once per nuclear coordinate.

| Symbol | Meaning | Code name |
|---|---|---|
| $x$, $y$ | nuclear Cartesian coordinates, $x = (A,\alpha)$ | index $3A+\alpha$ |
| $U^y$ | first-order orbital rotation, $C^y = C\,U^y$ | `CphfSolution::u` |
| $S^y_{\mathrm{MO}}$ | $C^{\mathsf T}S^yC$ | `s1` |
| $F^y$ | first-order Fock matrix at fixed density, $C^{\mathsf T}(h^y+G^y(D))\,C_o$ | `h1` |
| $G(D)$ | two-electron response, $J(D)-\tfrac12K(D)$ (RHF) | `TwoElectronBuilder::g_rhf_many` |
| $\varepsilon^y_{ij}$ | first-order occupied multiplier | `CphfSolution::e1` |
| $\mathrm{occ}$ | occupation, 2 (RHF) or 1 per spin (UHF) | |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/hessian.rs::{rhf_hessian,uhf_hessian}`, `ferric-scf/src/hessian/rohf.rs::rohf_hessian` |
| CLI | `task = "frequencies"` with `[frequencies] hessian = "analytic"` or `"auto"` |
| Python | `run_frequencies(...)`; the Hessian is `FrequencyResult.cartesian_hessian` |

Kohn–Sham references add exchange-correlation terms (section 14.5).
Frequencies from the Hessian are section 14.6.

## 14.4.1 The equations

<!-- include: crates/ferric-scf/theory.md#"Hartree-Fock nuclear Hessian" -->

> **Verified equation 14.9** · the assembled Hessian equals the difference of the gradient
>
> On the toy system of section 14.1 (integrals quadratic in two parameters,
> non-orthogonal $S$): the analytic Hessian, skeleton plus CPHF response as
> coded, equals the fourth-order central difference of the analytic
> gradient with the SCF re-solved at every displaced point. RHF (claim H1),
> UHF with the coupled $\alpha$/$\beta$ CPHF (claim H3) and ROHF with the
> commutator-form response (claim H4).
>
> | | |
> |---|---|
> | Proof | `scripts/hf_hessian_sympy.py`, claims H1, H3, H4 |
> | Negative control | the runner changes $U_{oo}$ from $-\tfrac12S_{oo}$ to $-S_{oo}$ and drops the factor $2\,\mathrm{occ}$ of the response (H1 fails), drops the $D_\alpha F_\beta^yD_\beta$ term of $W^y$, and replaces the commutator $[f,\kappa]$ by its diagonal-gap truncation (H4 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/hessian.rs::{rhf_hessian,uhf_hessian,assemble_response}`, `ferric-scf/src/hessian/rohf.rs::rohf_hessian` |
> | Code test | `h2o_ccpvdz` in `crates/ferric-scf/tests/validation_rhf_hessian.rs` |
>
> <!-- verified: id=scf.hess.assembled; script=scripts/hf_hessian_sympy.py; claim=H1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/hessian.rs::{rhf_hessian,uhf_hessian,assemble_response},ferric-scf/src/hessian/rohf.rs::rohf_hessian; rust_test=ferric-scf/tests/validation_rhf_hessian.rs::h2o_ccpvdz; registry=hf-hessian -->

> **Verified equation 14.10** · the occupied-occupied rotation and the closed-shell limit
>
> $U = -\tfrac12S^y_{\mathrm{MO}} + \kappa$ satisfies the differentiated
> orthonormality $U + U^{\mathsf T} + S^y_{\mathrm{MO}} = 0$ for any
> antisymmetric $\kappa$, and an occupied-occupied antisymmetric $\kappa$ leaves
> $D^y$ unchanged (claims S1, S2, in symbols). A closed-shell UHF reproduces the
> RHF Hessian (claim H2).
>
> | | |
> |---|---|
> | Proof | `scripts/hf_hessian_sympy.py`, claims S1, S2 and H2 |
> | Negative control | $U_{oo} = -S_{oo}$ breaks H1 and H3 (see equation 14.9) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/hessian.rs::finish_cphf` |
> | Code test | `closed_shell_uhf_equals_rhf_term_by_term` in `crates/ferric-scf/tests/uhf_hessian_fd.rs` |
>
> <!-- verified: id=scf.hess.occ-occ-rotation; script=scripts/hf_hessian_sympy.py; claim=S1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/hessian.rs::finish_cphf; rust_test=ferric-scf/tests/uhf_hessian_fd.rs::closed_shell_uhf_equals_rhf_term_by_term; registry=hf-hessian -->

**Why the ROHF response keeps a commutator.** ROHF has one orbital set and two
Fock matrices, and its orbitals are not eigenvectors of either. Differentiating
the stationarity condition $g = 0$ along $y$ therefore brings in
$[f_\sigma,\kappa]$ with the full, non-diagonal $f_\sigma$. A truncation to the
diagonal gap is wrong, and the runner's mutation checks that the proof sees
it. ROHF uses its own $W$ (section 14.1), not the UHF eigenvalue form, so the
skeleton's $W$ term differs from UHF's at ROHF orbitals.

## 14.4.2 The algorithm

The driver is `ferric-scf/src/hessian.rs::rhf_hessian_parts_with`, which
returns the five terms separately for tests.

| Step | Quantity | Function |
|---|---|---|
| 1 | $\partial^2V_{nn}/\partial x\partial y$ | `ferric-scf/src/hessian.rs::hess_nuclear_repulsion` |
| 2 | $\sum D\,\partial^2(T+V)$, with $V$ differentiated at both basis centres and every nucleus | `ferric-scf/src/hessian.rs::skeleton_hess_1e` |
| 3 | $-\sum W\,\partial^2S$ | `ferric-scf/src/hessian.rs::skeleton_hess_overlap` |
| 4 | $\sum\Gamma\,\partial^2(\mu\nu\vert\lambda\sigma)$ over quartets | `ferric-scf/src/hessian.rs::skeleton_hess_2e` |
| 5 | $F^y$, $S^y$ in the MO basis for all $3N$ coordinates | `ferric-scf/src/hessian.rs::first_order_ao` |
| 6 | CPHF for every coordinate by preconditioned conjugate gradient | `ferric-scf/src/hessian.rs::solve_cphf_block` |
| 7 | $\varepsilon^y$, then the response $R_{xy}$, symmetrized | `ferric-scf/src/hessian.rs::finish_cphf`, `assemble_response` |

The CPHF operator is symmetric positive definite at a stable RHF minimum, so
the solver is conjugate gradient, preconditioned by the diagonal of the
operator (`pcg_to` in `ferric-scf/src/hessian.rs`). The coordinates run in lockstep blocks: each iteration builds
$G$ for every still-running coordinate in one four-centre quartet pass. The
block length is a function of the basis size, not the thread count
(`ferric-scf/src/hessian.rs::cphf_block_len`, a 32 MiB budget of stacked
response matrices). For UHF the $\alpha$ and $\beta$ rotations are one
coupled system, because the Coulomb term of $G_\sigma$ carries both spins.

## 14.4.3 Reproducibility and refusals

- The pre-SCF probes `analytic_hessian_available` and the `*_preflight`
  functions refuse what the analytic Hessian does not cover before any SCF
  runs: angular momentum beyond the second-derivative classes libint was built
  with, ECPs on the libecpint backend (its ECP second derivative is
  noise-dominated and biased), solvent and polarizable embedding, and
  configurations with no analytic route. Frequencies under `hessian = "auto"`
  fall back to finite differences where the analytic route refuses.
- After the SCF, `uhf_state_refusal` refuses a UHF state whose occupations are
  non-aufbau (a MOM state, or an occupied eigenvalue above a virtual one) and
  a non-stationary state whose occupied-virtual Fock block is above the
  stationarity bound.
- A semicanonicalized UHF result that came from ROHF is refused by
  `uhf_hessian`, because its orbitals are not a UHF stationary point. The
  ROHF Hessian maps it back first.
- UHF is not checked for internal stability or for being the lowest state. An
  excited or unstable UHF solution gets the Hessian of that solution.

## 14.4.4 Options that change the equation

| Option | Effect | Where |
|---|---|---|
| density-fitted J, K | fitted skeletons, fitted $J^y$ and $K^y$ in $F^y$, the SCF's own `DfJ`/`DfK` in the CPHF kernel; nothing else changes | `ferric-scf/src/hessian_df.rs`, `ferric-scf/src/hessian_df_k.rs` |
| ECPs | skeleton $\sum D\,\partial^2V_{\mathrm{ECP}}$ by a masked fourth-order stencil of first-derivative ECP blocks, and $\partial V_{\mathrm{ECP}}/\partial x$ in $F^y$ | `ferric-scf/src/hessian_ecp.rs` |
| external point charges, smeared charges | QM-QM block, charges fixed | `ferric-scf/src/hessian_smeared.rs` |
| uniform field | the field-nuclear energy is linear in $\mathbf R$, so only the electronic term enters; dipole integrals differentiated by a stencil | `ferric-scf/src/hessian_field.rs` |
| Kohn–Sham | XC skeleton, XC first-order Fock matrix and $f_{xc}$ kernel in the CPHF | section 14.5 |

## 14.4.5 Numerical anchors

| Check | Reference | Agreement | Test |
|---|---|---|---|
| RHF Hessian, skeleton and CPHF response: H2O, NH3, CH2O / cc-pVDZ; distorted H2O / def2-SVP | PySCF `hessian.rhf` analytic; PySCF finite differences of its gradient; ferric's gradient differenced | total at most 1.6e-7 Ha/Bohr² (at most 8.2e-8 at cc-pVDZ), bar 1e-6; skeleton at most 5.9e-10, bar 1e-8; frequencies at most 6.9e-4 cm⁻¹ | `crates/ferric-scf/tests/validation_rhf_hessian.rs`, `crates/ferric-scf/tests/rhf_hessian_fd.rs` |
| UHF Hessian: OH, NH2, CH2 / cc-pVDZ | PySCF `hessian.uhf` on the lowest internally stable UHF | total at most 3.8e-7 (OH; at most 7.7e-8 for NH2 and CH2), bar 2e-6; skeleton at most 1.4e-10; closed-shell water equals RHF term by term (response 2.8e-17) | `crates/ferric-scf/tests/validation_uhf_hessian.rs`, `crates/ferric-scf/tests/uhf_hessian_fd.rs` |
| ROHF Hessian: NH2, CH2, HO2 / cc-pVDZ; STO-3G, 6-31G | central FD of PySCF's ROHF gradient (PySCF has no ROHF Hessian); ferric's gradient differenced | vs ferric's differenced gradient at most 3.2e-10; vs PySCF FD 1.0e-6 or less, against PySCF's own h-vs-2h gap of 0.7e-6 to 2.6e-6; the Hessian without the response misses by 0.35 to 0.83 | `crates/ferric-scf/tests/validation_rohf_hessian.rs` |

<!-- figures: site/src/reference/validation.md "Analytic RHF Hessian (skeleton one- and two-electron, overlap/W and CPHF response terms)"; site/src/reference/validation.md "Analytic UHF Hessian (skeleton one- and two-electron"; site/src/reference/validation.md "Analytic ROHF Hessian (skeleton with the ROHF energy-weighted density" -->

Fitted-J/K, ECP, charge and field Hessians have their own rows on the
validation page; their agreement with finite differences of the matching
gradient is at the $10^{-10}$ to $10^{-7}$ Ha/Bohr² level quoted there.

**Known limits.** Closed-shell RHF, UHF and ROHF only, with basis functions up
to f; no analytic ROKS Hessian (it refuses before the SCF, and ROKS
frequencies difference the analytic gradient); no solvent and no polarizable
embedding.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/theory.md`, "Hartree-Fock nuclear Hessian" | equations and the toy-system check |
| `scripts/hf_hessian_sympy.py`, claims H1 to H4, S1, S2 | verified equations 14.9 and 14.10 |
| `crates/ferric-scf/src/hessian.rs` module documentation | the term table, the solver, the refusals |
| `site/src/reference/validation.md` | the anchor rows and the known limits |
