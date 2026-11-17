# 14.4 Hartree–Fock Hessian and coupled-perturbed equations

<!-- section: 14.4; status: ready; registry: hf-hessian;
     theory: crates/ferric-scf/theory.md#"Hartree-Fock nuclear Hessian" -->

The Hessian is the curvature of the energy surface: how the force on one
nucleus changes when another moves. It is the derivative of the gradient of
section 14.1, and that gradient contains the converged density $D$ and the
energy-weighted density $W$ explicitly. The gradient needed no orbital
derivative because the energy is stationary in the orbitals; $D$ and $W$
are not, so the Hessian needs to know how the orbitals relax when a nucleus
moves. That relaxation is the solution of a linear system, the
coupled-perturbed Hartree–Fock (CPHF) equations, one per nuclear coordinate.

The plan: split the Hessian into a skeleton at fixed $D$ and $W$ and a
response through the orbitals, derive the CPHF equations that give the
response, then follow the solver.

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

**The skeleton.** Differentiate the gradient of section 14.1 once more along
$y$, holding $D$, $W$ and $\Gamma$ fixed. Every integral gets a second
derivative (section 3.3 gives their layout):

$$H^{\mathrm{skel}}_{xy} = E_{nn}^{xy} + \sum D h^{xy} + \sum \Gamma\,(\mu\nu|\lambda\sigma)^{xy} - \sum W S^{xy},$$

with the $D$, $\Gamma$ and $W$ of the gradient table; ROHF uses its own $W$.

**The orbital rotation.** What remains is the change of $D$ and $W$. Write
the first-order orbitals as a rotation of the converged ones,
$C^y = C\,U^y$. The orbitals stay orthonormal in the moving basis:
differentiate $C^{\mathsf T}SC = 1$ along $y$ and substitute $C^y = CU$:

$$ U + U^{\mathsf T} + S^y_{\mathrm{MO}} = 0, \qquad S^y_{\mathrm{MO}} = C^{\mathsf T} S^y C . $$

The symmetric part of $U$ is fixed by the overlap; only the antisymmetric
part is free. A rotation among occupied orbitals does not change $D$, so the
code fixes the occupied-occupied block to its symmetric part,
$U^y_{ij} = -\tfrac12 S^y_{ij}$. The unknowns are the virtual-occupied
rotations $U_{ai}$. With $\mathrm{occ} = 2$ (RHF) or $1$ (each UHF spin), the
first-order density splits into the part they generate and the part the
overlap forces:

$$D_1(U) = \mathrm{occ}\,(C_v U C_o^{\mathsf T} + C_o U^{\mathsf T} C_v^{\mathsf T}),\qquad
D^y_{oo} = -\mathrm{occ}\;C_o S^y_{oo} C_o^{\mathsf T}, \qquad D^y = D_1(U^y) + D^y_{oo} . $$

**The CPHF equations.** The SCF condition at every geometry is that the
virtual-occupied block of the Fock matrix vanishes, $F_{ai} = 0$, with
$F_{\mathrm{MO}} = \mathrm{diag}(\varepsilon)$ at canonical orbitals.
Differentiate $(C^{\mathsf T}FC)_{ai}$ along $y$. The rotation hits both $C$
factors, giving $U_{ia}\varepsilon_i + \varepsilon_aU_{ai}$. The AO Fock matrix
changes through the integrals at fixed density, $F^y$, and through the
density, $G(D^y)$. Use the orthonormality condition to replace $U_{ia}$ by
$-U_{ai} - S^y_{ai}$, and set the derivative to zero:

$$(\varepsilon_a - \varepsilon_i)\,U^y_{ai} + G(D_1(U^y))_{ai}
= -\big(F^y_{ai} - \varepsilon_i S^y_{ai} + G(D^y_{oo})_{ai}\big) .$$

`ferric-scf/src/hessian.rs::solve_cphf_block` solves this system, with
right-hand side `ferric-scf/src/hessian.rs::cphf_rhs`. For RHF
$G = J - \tfrac12 K$. For UHF it is spin-coupled,
$G_\sigma(D_\alpha, D_\beta) = J(D_\alpha + D_\beta) - K(D_\sigma)$, and
$(U_\alpha, U_\beta)$ form one linear system.

Check the uncoupled limit. Drop the two-electron response $G$ on both sides
and the solution is
$U_{ai} = -(F^y_{ai} - \varepsilon_iS^y_{ai})/(\varepsilon_a - \varepsilon_i)$:
first-order perturbation theory, a coupling divided by an orbital-energy
gap. The $G$ terms add the self-consistency: the relaxed density changes
the Fock matrix, which changes the relaxation. A small HOMO-LUMO gap makes
the response large, as it makes the MP2 amplitudes large (section 10.1).

The operator on the left is the real singlet orbital Hessian of section
8.4.1 with $c_x = 1$. Expand $C_v^{\mathsf T}G(D_1(U))C_o$ in MO integrals: the
Coulomb part gives $4(ai|bj)U_{bj}$ and the exchange part
$-\big[(ab|ij) + (aj|bi)\big]U_{bj}$, the RHF internal stability operator. It
is symmetric, and positive definite exactly when the RHF solution is
internally stable.

**The occupied multiplier.** The response of $W$ needs the first-order
multiplier $\varepsilon^y_{ij}$ (`ferric-scf/src/hessian.rs::finish_cphf`):

$$\varepsilon^y_{ij} = F^y_{ij} + [C_o^{\mathsf T} G(D^y_{oo} + D_1(U^y)) C_o]_{ij} - \tfrac12 S^y_{ij}(\varepsilon_i + \varepsilon_j).$$

**The response term.** $R_{xy}$ is $\mathrm d/\mathrm dy$ of
$\sum D F^x - \sum W S^x$, with $D^y$ and $W^y$ written through $U^y$ and
$\varepsilon^y$ (`ferric-scf/src/hessian.rs::assemble_response`, PySCF
`hess_elec`):

$$R_{xy} = \sum_\sigma \Big[ 2\,\mathrm{occ} \sum_{p,i} (F^x - S^x\varepsilon)_{pi}\,U^y_{pi}
- \mathrm{occ} \sum_{ij} S^x_{ij}\,\varepsilon^y_{ij} \Big],$$

with $p$ over all orbitals; the $oo$ rows carry $U_{oo} = -\tfrac12 S_{oo}$.
For RHF this is
$4\sum (F^x - S^x\varepsilon) U^y - 2\sum S^x_{oo}\varepsilon^y$. The total is
$H_{xy} = H^{\mathrm{skel}}_{xy} + R_{xy}$, and a closed-shell UHF reproduces
the RHF Hessian term by term.

How much does the response matter? For the ROHF test systems of section
14.4.5 (NH2, CH2, HO2 / cc-pVDZ), the Hessian without it misses PySCF's
finite differences by 0.35 to 0.83 Ha/Bohr², and its frequencies by
1.3e3 to 2.5e3 cm⁻¹. With the response the miss is $1.0\times10^{-6}$ or less.

**Two checks you can do in your head.** The Hessian is symmetric,
$H_{xy} = H_{yx}$, because mixed partial derivatives commute; the response
term is the only one the code computes in non-symmetric form, which is why
step 7 below symmetrizes it and the frequency result reports the asymmetry.
And it is translationally invariant: the gradient does not change when the
whole molecule moves, so for every row $(A,\alpha)$ and axis $\beta$,
$\sum_B H_{(A,\alpha),(B,\beta)} = 0$.

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

On the gradient's toy system the analytic Hessian matches a fourth-order
difference of the analytic gradient, with the SCF re-solved, to
$3\times10^{-11}$ (RHF), $1\times10^{-9}$ (UHF) and $1\times10^{-9}$ (ROHF). The
closed-shell UHF Hessian equals the RHF one to $4\times10^{-16}$.

### ROHF: why the response keeps a commutator

ROHF has one orbital set and two Fock matrices, and its orbitals are
eigenvectors of neither. The derivation above used
$F_{\mathrm{MO}} = \mathrm{diag}(\varepsilon)$, so it does not carry over.
`ferric-scf/src/hessian/rohf.rs` keeps the rotation $\kappa$ (blocks $vc$, $vo$,
$oc$) separate from the overlap part, $U = \kappa - \tfrac12 S^y_{\mathrm{MO}}$,
and splits the first-order spin densities in the MO basis:

$$(D^y_\sigma)_{\mathrm{MO}} = U n_\sigma + n_\sigma U^{\mathsf T},\quad
(D^S_\sigma)_{pq} = -\tfrac12 S^y_{pq}(n_{\sigma p} + n_{\sigma q}),\quad
(D^\kappa_\sigma)_{pq} = (n_{\sigma q} - n_{\sigma p})\kappa_{pq}.$$

The spin Fock matrix in the MO basis, $f_\sigma$, changes along $y$ by the
overlap, by the rotation, and by the AO Fock derivative:

$$ \frac{\mathrm d f_\sigma}{\mathrm dy} = -\tfrac12(S f_\sigma + f_\sigma S) + [f_\sigma, \kappa] + C^{\mathsf T}(F^y_\sigma + G_\sigma(D^y))C . $$

Differentiate the ROHF stationarity condition $g = 0$ with it
(`Response::apply`, `Response::rhs`):

$$\Phi_\sigma = C^{\mathsf T}F^y_\sigma C + C^{\mathsf T}G_\sigma(D^S)C - \tfrac12(S f_\sigma + f_\sigma S)
+ C^{\mathsf T}G_\sigma(D^\kappa)C + [f_\sigma, \kappa],$$

$$(\Phi_\alpha + \Phi_\beta)_{vc} = 0,\qquad (\Phi_\alpha)_{vo} = 0,\qquad (\Phi_\beta)_{oc} = 0 .$$

This is a linear system $A\kappa = -B$. Its operator keeps the full
commutator $[f_\sigma, \kappa]$, because the $f_\sigma$ are not diagonal. A
truncation to the diagonal gap, which is exact for RHF and UHF, is wrong
here, and the runner's mutation checks that the proof sees it. The response
column (`Response::column`, `w_response`) is

$$R_{xy} = \sum_\sigma \mathrm{tr}[D^y_\sigma F^x_\sigma] - \mathrm{tr}[W^y S^x],$$

$$W^y = D^y_\alpha F_\alpha D_\alpha + D_\alpha F^y_\alpha D_\alpha + D_\alpha F_\alpha D^y_\alpha
+ D^y_\alpha F_\beta D_\beta + D_\alpha F^y_\beta D_\beta + D_\alpha F_\beta D^y_\beta,$$

the product-rule derivative of the ROHF $W$ of section 14.1, with
$F^y_\sigma$ the total derivative $h^y + G^y_\sigma(D) + G_\sigma(D^y)$. The
skeleton also uses that ROHF $W$, so its $W$ term differs from UHF's at ROHF
orbitals.

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

The CPHF right-hand side is the bracket of the CPHF equation with its sign
moved, $S^y_{ai}\varepsilon_i - F^y_{ai} - G(D^y_{oo})_{ai}$:

<!-- excerpt: ferric-scf/src/hessian.rs::cphf_rhs -->
```rust
let mut rhs = &s1.slice(s![nocc.., ..]) * &frame.eps_occ - h1.slice(s![nocc.., ..]);
rhs -= &frame.vo(g_oo);
rhs
```

| Code | Equation |
|---|---|
| `s1.slice(s![nocc.., ..]) * &frame.eps_occ` | $S^y_{ai}\varepsilon_i$, each column scaled by its occupied energy |
| `h1.slice(s![nocc.., ..])` | $F^y_{ai}$ |
| `frame.vo(g_oo)` | $G(D^y_{oo})_{ai}$, the $vo$ block of the overlap-forced response |

The operator is applied without forming it. Inside the block solver each
trial rotation $u$ is mapped to

<!-- excerpt: ferric-scf/src/hessian.rs::solve_cphf_blocks_of -->
```rust
.map(|(gu, u)| &frame.denom * *u + frame.vo(gu))
```

where `frame.denom` holds $\varepsilon_a - \varepsilon_i$ and `gu` is
$G(D_1(u))$, built for all coordinates of the block in one pass. The same
`frame.denom` is the preconditioner: conjugate gradient is preconditioned by
the orbital-energy gap, the uncoupled part of the operator, not by the full
diagonal.

The CPHF operator is symmetric positive definite at a stable RHF minimum, so
the solver is conjugate gradient
(`ferric-scf/src/hessian.rs::pcg_to`), to a max-abs residual of
`ferric-scf/src/hessian.rs::CPHF_TOL` ($10^{-10}$) within
`CPHF_MAX_ITER` (200) iterations. A non-positive curvature $p^{\mathsf T}Ap$ is
a typed error that says the SCF reference is not a stable minimum: that is
the stability analysis of section 8.4 surfacing inside the Hessian.

The coordinates run in lockstep blocks: each iteration builds $G$ for every
still-running coordinate in one four-centre quartet pass. The block length
is a function of the basis size alone
(`ferric-scf/src/hessian.rs::cphf_block_len`, a 32 MiB budget of stacked
response matrices), so the blocking is the same for every thread count. For UHF the $\alpha$ and $\beta$ rotations are one coupled
system, because the Coulomb term of $G_\sigma$ carries both spins; the solver
stacks the two blocks into one vector.

Step 7 assembles $R$ from the solutions, one row per perturbation $x$:

<!-- excerpt: ferric-scf/src/hessian.rs::assemble_response -->
```rust
for x in 0..n3 {
    let hs = (2.0 * frame.occ) * (&h1_mo[x] - &(&s1_mo[x] * &frame.eps_occ));
    let s1_oo = s1_mo[x].slice(s![..nocc, ..]);
    for (y, sol) in sols.iter().enumerate() {
        r[(x, y)] = (&hs * &sol.u).sum() - frame.occ * (&s1_oo * &sol.e1).sum();
    }
}
```

| Code | Equation |
|---|---|
| `hs` | $2\,\mathrm{occ}\,(F^x - S^x\varepsilon)_{pi}$ over all $p$ and occupied $i$ |
| `(&hs * &sol.u).sum()` | $\sum_{pi}2\,\mathrm{occ}\,(F^x - S^x\varepsilon)_{pi}U^y_{pi}$ |
| `frame.occ * (&s1_oo * &sol.e1).sum()` | $\mathrm{occ}\sum_{ij}S^x_{ij}\varepsilon^y_{ij}$ |

The loop computes $R_{xy}$ from $U^y$ and $R_{yx}$ from $U^x$ separately.
They are equal only in exact arithmetic with a converged CPHF, so their
largest difference is the analytic Hessian's internal consistency probe:
`RhfHessianParts::response_asymmetry`, which the frequency result reports
as `asymmetry`. The Hessian uses the symmetrized $R$.

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
