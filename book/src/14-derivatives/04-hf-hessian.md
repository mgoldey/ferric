# Hartree–Fock Hessian and coupled-perturbed equations {#hf-hessian}

<!-- status: ready; registry: hf-hessian -->

The Hessian is the curvature of the energy surface: how the force on one
nucleus changes when another moves. It is the derivative of the gradient of
section @sec:hf-gradient, and that gradient contains the converged density $D$ and the
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

Kohn–Sham references add exchange-correlation terms (section @sec:ks-hessian).
Frequencies from the Hessian are section @sec:frequencies.

## The equations {#hf-hessian-equations}

**The skeleton.** Differentiate the gradient of section @sec:hf-gradient once more along
$y$, holding $D$, $W$ and $\Gamma$ fixed. Every integral gets a second
derivative (section @sec:integrals-derivatives gives their layout):

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
the response large, as it makes the MP2 amplitudes large (section @sec:closed-shell-ri-mp2).

The operator on the left is the real singlet orbital Hessian of section
@sec:stability-stability-operators with $c_x = 1$. Expand $C_v^{\mathsf T}G(D_1(U))C_o$ in MO integrals: the
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
The sum over $\sigma$ has one term for RHF, with $\mathrm{occ} = 2$, and two
for UHF, with $\mathrm{occ} = 1$. For RHF this is
$4\sum (F^x - S^x\varepsilon) U^y - 2\sum S^x_{oo}\varepsilon^y$. The total is
$H_{xy} = H^{\mathrm{skel}}_{xy} + R_{xy}$, and a closed-shell UHF reproduces
the RHF Hessian term by term.

How much does the response matter? The anchors record it for ROHF: for the
test systems of section @sec:hf-hessian-numerical-anchors (NH2, CH2, HO2 /
cc-pVDZ), the Hessian without it misses PySCF's
finite differences by 0.35 to 0.83 Ha/Bohr², and its frequencies by
1.3e3 to 2.5e3 cm⁻¹. With the response the miss is $1.0\times10^{-6}$ or less.

**Two checks you can do in your head.** The Hessian is symmetric,
$H_{xy} = H_{yx}$, because mixed partial derivatives commute; the response
term is the only one the code computes in non-symmetric form, which is why
step 7 below symmetrizes it and the frequency result reports the asymmetry.
And it is translationally invariant: the gradient does not change when the
whole molecule moves, so for every row $(A,\alpha)$ and axis $\beta$,
$\sum_B H_{(A,\alpha),(B,\beta)} = 0$.

> **Verified equation {#eq:scf-hess-assembled}** · the assembled Hessian equals the difference of the gradient
>
> On the toy system of section @sec:hf-gradient (integrals quadratic in two parameters,
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

Both checks on the analytic RHF Hessian of water in cc-pVDZ at its
minimum, from the `3f612e0b3` build. The geometry file is the output of
`run_optimize` from `testdata/molecules/water.xyz` at RHF/cc-pVDZ, with
largest gradient component $5.6\times10^{-9}$ Ha/Bohr ($r_{\rm OH} =
0.9463$ Å, 104.61°). `asymmetry` is the largest
$\lvert H_{xy} - H_{yx}\rvert$ before symmetrization, the row sums are the
translational check, and the six projected translations and rotations sit
within $10^{-4}$ cm⁻¹ of zero:

```python
import numpy as np
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water_rhf_ccpvdz_opt.xyz")  # RHF/cc-pVDZ minimum
r = ferric.run_frequencies(mol, "cc-pvdz", reference="rhf", hessian="analytic")
H = np.asarray(r.cartesian_hessian)
n = H.shape[0] // 3
print("hessian_source:", r.hessian_source, " asymmetry:", r.asymmetry)
# sum_B H[(A,a),(B,b)] = 0 for every row (A,a) and column axis b
rowsum = np.array([[H[i, b::3].sum() for b in range(3)] for i in range(3 * n)])
print("max |sum over atoms B of H[(A,a),(B,b)]|:", np.abs(rowsum).max())
print("frequencies (cm^-1):", r.frequencies)
print("projected translation/rotation (cm^-1):", r.trans_rot_frequencies)
```

```text
hessian_source: analytic  asymmetry: 8.886169577948522e-12
max |sum over atoms B of H[(A,a),(B,b)]|: 4.039935053157251e-12
frequencies (cm^-1): [1775.6538704934494, 4113.408094041156, 4211.724258205592]
projected translation/rotation (cm^-1): [-2.046605246237348e-05, -1.6496913227920158e-05, -3.467573733374684e-09, 5.610652235453035e-09, 1.7164128291438074e-05, 5.148442298581013e-05]
```

> **Verified equation {#eq:scf-hess-occ-occ-rotation}** · the occupied-occupied rotation and the closed-shell limit
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
> | Negative control | $U_{oo} = -S_{oo}$ breaks H1 and H3 (see equation @eq:scf-hess-assembled) |
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

$$W^y = D^y_\alpha F_\alpha D_\alpha + D_\alpha \bar F^y_\alpha D_\alpha + D_\alpha F_\alpha D^y_\alpha
+ D^y_\alpha F_\beta D_\beta + D_\alpha \bar F^y_\beta D_\beta + D_\alpha F_\beta D^y_\beta,$$

the product-rule derivative of the ROHF $W$ of section @sec:hf-gradient. In
$W^y$ the Fock derivative is the total one,
$\bar F^y_\sigma = F^y_\sigma + G_\sigma(D^y)$, where $F^y_\sigma = h^y +
G^y_\sigma(D)$ is the fixed-density derivative used in $\Phi_\sigma$ and in
$R_{xy}$. The
skeleton also uses that ROHF $W$, so its $W$ term differs from UHF's at ROHF
orbitals.

### Fitted exchange in the open-shell response {#hf-hessian-factored-ri-k}

With RI-K, every CPHF iteration needs $K[D] = \sum_P B^PDB^P$ for each
coordinate's response density, and each $B^P$ is symmetric (section
@sec:ri-j-ri-k-split-ri-j). The response densities are indefinite, so the occupied route
does not apply, and the density route costs two $n\times n$ products per
auxiliary function. The open-shell kernel
(`ferric-scf/src/hessian.rs::uhf_fitted_exchanges`, which calls
`ferric-scf/src/df_k/response.rs::build_factored_many`) uses the form of
the density instead. Every density it sees is

$$ D = XC_o^{\mathsf T} + C_oX^{\mathsf T} $$

for some $n\times o$ matrix $X$ and occupied block $C_o$. Substitute the
first half into $B^PDB^P$ and regroup:
$B^PXC_o^{\mathsf T}B^P = (B^PX)(B^PC_o)^{\mathsf T}$, using
$B^{P\,\mathsf T} = B^P$. The second half is the transpose of the first.
Sum over $P$:

$$ K[D] = G + G^{\mathsf T}, \qquad G = \sum_P (B^PX)(B^PC_o)^{\mathsf T} . $$

$L^P = B^PC_o$ is built once per $C_o$ and reused for every $X$. Each
density then costs $B^PX$ and one product with $L^{P\,\mathsf T}$, about
$4\,n_{\rm aux}n^2o$ flops, against $4\,n_{\rm aux}n^3$ for the density
route. $D$ need not be definite or idempotent; only its form matters.

**UHF and UKS.** Each spin's CPHF density has this form with that spin's
occupied block, $C_o = C_{\sigma,o}$. The rotation response
$D_1(U_\sigma) = C_{\sigma,v}U_\sigma C_{\sigma,o}^{\mathsf T} + \text{h.c.}$
has $X_\sigma = C_{\sigma,v}U_\sigma$. The occupied block
$D^y_{\sigma,oo} = -C_{\sigma,o}S^y_{\sigma,oo}C_{\sigma,o}^{\mathsf T}$ has
$X_\sigma = -\tfrac12 C_{\sigma,o}S^y_{\sigma,oo}$, because $S_{oo}$ is
symmetric and the two halves add. The kernel
$v_\sigma = J(D_{1\alpha} + D_{1\beta}) - a\,K(D_{1\sigma})$, with $a$ the
exact-exchange fraction, never couples the spins in $K$. So the factored
pass runs once per spin, with $L^P_\sigma = B^PC_{\sigma,o}$, over every
coordinate of a lockstep block.

**ROHF.** There is one orbital set $C$ (closed | open | virtual) and an
occupation vector $n_\sigma$ per spin whose first $n_\sigma$ entries are one.
Every density change of the ROHF response is
$C(Un_\sigma + n_\sigma U^{\mathsf T})C^{\mathsf T}$ for a generator $U$:
the rotation part $(n_{\sigma q} - n_{\sigma p})\kappa_{pq}$ is $U = \kappa$
($\kappa$ antisymmetric), the overlap part
$-\tfrac12 S^y_{pq}(n_{\sigma p} + n_{\sigma q})$ is $U = -\tfrac12 S^y_{\rm MO}$
(symmetric), and the total $D^y_\sigma$ is $U = \kappa - \tfrac12 S^y_{\rm MO}$.
$(Un_\sigma)_{pq} = U_{pq}$ for $q < n_\sigma$ and zero otherwise, so

$$ C\,U n_\sigma\,C^{\mathsf T} = (CU)_{:,\,<n_\sigma}\,C_{:,\,<n_\sigma}^{\mathsf T}, $$

which is the required form with $C_{\sigma,o} = C_{:,\,<n_\sigma}$ and
$X_\sigma = (CU)_{:,\,<n_\sigma}$
(`ferric-scf/src/hessian/rohf.rs::spin_factors`). The ROHF response solves one
coordinate at a time, so each factored pass carries one density per spin.

**Closed-shell limit.** Set $C_\alpha = C_\beta$ and $U_\alpha = U_\beta$.
The RHF factor is $X_{\rm RHF} = 2X_\sigma$, so $K_{\rm RHF} = 2K(D_{1\sigma})$
and

$$ v_\sigma = J(2D_{1\sigma}) - K(D_{1\sigma}) = J(D_{\rm RHF}) - \tfrac12K(D_{\rm RHF}), $$

the RHF $G$ of the closed-shell factored route.

> **Verified equation {#eq:scf-hess-factored-k}** · the factored RI-K of the response densities
>
> For symmetric $B^P$, $K[XC_o^{\mathsf T} + C_oX^{\mathsf T}] = G + G^{\mathsf T}$ with
> $G = \sum_P(B^PX)(B^PC_o)^{\mathsf T}$ (claim F1, in symbols). The UHF and UKS
> rotation and occupied-block densities (claim F2), the ROHF generator form
> $C\,Un_\sigma C^{\mathsf T} = (CU)_{:,<n_\sigma}C_{:,<n_\sigma}^{\mathsf T}$
> (claims F3, F4) and the closed-shell limit (claim F5) give the exchange of
> the explicit four-index tensor.
>
> | | |
> |---|---|
> | Proof | `scripts/hessian_factored_k_sympy.py`, claims F1 to F5 |
> | Negative control | the runner drops the h.c. term (F1 fails), builds the $\beta$ exchange from the $\alpha$ occupied block (F2 fails) and the ROHF factor from $U^{\mathsf T}$ (F3 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/df_k/response.rs::build_factored_many`, `ferric-scf/src/hessian/rohf.rs::spin_factors` |
>
> <!-- verified: id=scf.hess.factored-k; script=scripts/hessian_factored_k_sympy.py; claim=F1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/df_k/response.rs::build_factored_many,ferric-scf/src/hessian/rohf.rs::spin_factors; registry=hf-hessian -->

`scripts/hessian_factored_k_sympy.py` proves these statements. F1 checks
$K[XC_o^{\mathsf T} + C_oX^{\mathsf T}] = G + G^{\mathsf T}$ in symbols
(symmetric $B^P$, $n = 3$, $o = 2$, two auxiliary functions). F3 checks the
ROHF identity above for a symbolic generator $U$, with $U = \kappa$ and
$U = -S/2$ as the two cases. F2 (UHF and UKS: rotation and occupied-block
densities, $n_\alpha \ne n_\beta$, non-orthonormal orbitals), F4 (ROHF:
$\kappa$, overlap and total densities for both spins) and F5 (the
closed-shell limit, occupation 2, $X = 2C_vU$) compare the factored $K$
with the exchange contracted from the explicit four-index tensor
$(\mu\lambda\vert\nu\sigma) = \sum_PB^P_{\mu\lambda}B^P_{\nu\sigma}$, which
never goes through $B^PDB^P$. The script's three mutations each turn it
red: dropping the h.c. term ($G$ in place of $G + G^{\mathsf T}$; F1, F2, F4,
F5 fail), building the $\beta$ exchange from the $\alpha$ occupied block (F2
fails), and the ROHF factor from $U^{\mathsf T}$ instead of $U$ (F3, F4
fail). It runs under `crates/ferric-python/tests/test_derivation_scripts.py`.

### Effective core potentials, fields and external charges {#hf-hessian-one-electron-operators}

Four extra pieces of physics can sit in the core Hamiltonian $h$ besides
$T$ and the nuclear attraction. An effective core potential replaces the
inner electrons of a heavy atom by a fixed operator centred on that atom.
A uniform electric field pulls the electrons one way and the nuclei the
other. Point charges and Gaussian-smeared charges stand in for an
environment held fixed in space (QM/MM). Each adds an operator $V$ to $h$
whose matrix elements move with the nuclei, and each except the ECP adds a
classical energy between the nuclei and the source. The Hessian needs the
same two things from all four: how $V$ curves, and how its slope drives
the orbital response.

**The general one-electron operator.** Write the energy with the extra
operator and its classical partner,

$$E(\mathbf R) = E_0(\mathbf R) + \sum_{\mu\nu} D_{\mu\nu} V_{\mu\nu}(\mathbf R) + E_{\mathrm{cl}}(\mathbf R),$$

where $E_0$ is the energy of section @sec:hf-hessian-equations, with $h$
there being $T$ plus the nuclear attraction. $V$ is part of $h$, so the
converged orbitals already include it. The gradient follows from
stationarity in the orbitals as in section @sec:hf-gradient: the integral
derivative at fixed $D$, plus the Pulay term from $W$, which already sees
$V$ through the orbital energies:

$$\frac{\mathrm dE}{\mathrm dx} = \frac{\mathrm dE_0}{\mathrm dx}\Big|_{h \to h + V}
+ \sum D_{\mu\nu} V^x_{\mu\nu} + E^x_{\mathrm{cl}} .$$

Differentiate along $y$. The product rule on $\sum D V^x$ gives two terms:

$$\frac{\mathrm d}{\mathrm dy}\sum D_{\mu\nu} V^x_{\mu\nu}
= \sum D_{\mu\nu} V^{xy}_{\mu\nu} + \sum D^y_{\mu\nu} V^x_{\mu\nu} .$$

The first is a skeleton term at fixed $D$, added to term 2. The second is
the change of $D$, and the response $R_{xy}$ of section
@sec:hf-hessian-equations is the derivative of $\sum D F^x - \sum W S^x$
with $D^y$ and $W^y$ written through $U^y$. Since $F^x = C^{\mathsf T}(h^x +
G^x(D))C_o$, adding $V^x$ to $h^x$ makes $\sum D^y F^x$ contain
$\sum D^y V^x$ exactly. The same $F^x$ is the right-hand side of the CPHF
equation, so $V^x$ drives the rotation $U^x$ as well. The full set of
changes is

$$H_{xy} \mathrel{+}= \sum_{\mu\nu} D_{\mu\nu}\,\frac{\partial^2 V_{\mu\nu}}{\partial x\,\partial y}
+ \frac{\partial^2 E_{\mathrm{cl}}}{\partial x\,\partial y},
\qquad F^x \to F^x + C^{\mathsf T}\,\frac{\partial V}{\partial x}\,C_o ,$$

and nothing else: the CPHF operator, $G$, $W$ and the overlap terms are the
ones of section @sec:hf-hessian-equations. The limit check is $V = 0$, $E_{\mathrm{cl}} = 0$:
every new term vanishes and the plain Hessian is left. On the open-shell
routes the same $\partial V/\partial x$ joins each spin's $F^x$, because
$V$ acts on both spins alike.

What does $V^x$ in $F^x$ do? Leave it out and the orbitals respond to the
moving basis but not to the moving operator. On the toy model of the proof
the Hessian then misses the finite difference by 0.39 (toy units),
against an agreement of $2\times10^{-11}$ with it in.

**The four operators.** The table lists what each $V$ depends on and how
its two derivatives are obtained.

| Source | $V_{\mu\nu}$ | moves with | $\sum D\,V^{xy}$ | $V^x$ in $F^x$ | $E_{\mathrm{cl}}$ |
|---|---|---|---|---|---|
| point charges $q_i$ at $\mathbf r_i$ | $-\sum_i q_i\langle\mu\vert\,\lvert\mathbf r-\mathbf r_i\rvert^{-1}\vert\nu\rangle$ | basis centres | libint2 second derivatives of the nuclear-attraction engine | libint2 first derivatives | $\sum_{A,i} Z_A q_i/R_{Ai}$ |
| smeared charges, width $w_i$ | $-\sum_i q_i\,(\mu\nu\vert g_i)/n_i$ | basis centres | stencil of three-centre first-derivative blocks | analytic three-centre first derivatives | $\sum_{A,i} Z_A q_i\,\mathrm{erf}(R_{Ai}/w_i)/R_{Ai}$ |
| uniform field $\mathbf E$ | $\mathbf E\cdot\langle\mu\vert\mathbf r\vert\nu\rangle$ | basis centres | stencil of the integrals themselves | stencil of the integrals | $-\mathbf E\cdot\sum_A Z_A\mathbf R_A$ |
| ECP $U_C$ on atom $C$ | $\sum_C\langle\mu\vert U_C\vert\nu\rangle$ | basis centres and $C$ | masked stencil of first-derivative ECP blocks | analytic first derivatives | none; $Z_A$ is the effective charge everywhere |

Here $R_{Ai} = \lvert\mathbf R_A - \mathbf r_i\rvert$, $g_i$ is the
s-Gaussian of exponent $1/w_i^2$ on site $i$ and $n_i$ its normalization
integral. The charges and the field are fixed in space: only QM nuclei
are Hessian coordinates (the QM-QM block). The skeleton terms are summed in
`ferric-scf/src/hessian.rs::extra_skeleton` (field, smeared charges, ECP)
and `ferric-scf/src/hessian.rs::contract_external_deriv2` (point charges);
the $V^x$ matrices join every spin's $F^x$ in
`ferric-scf/src/hessian.rs::add_extra_first_order`, with the point-charge
part built by `ferric-scf/src/hessian.rs::deriv1_1e_matrices` with the
charges loaded after the nuclei. In both point-charge routines libint2
also writes derivative blocks for the charge centres; the code drops them,
since the charges are not coordinates. The second-derivative engine is
handed the charges in chunks of `ferric-scf/src/hessian.rs::EXTERNAL_CHUNK`
(4), because its output grows with the square of the number of operator
centres.

**The classical terms.** The field term is linear in every $\mathbf R_A$:

$$\frac{\partial^2}{\partial R_{A\alpha}\,\partial R_{B\beta}}\Big(-\mathbf E\cdot\sum_C Z_C\mathbf R_C\Big) = 0 ,$$

so the field adds no classical Hessian; its gradient $-Z_A\mathbf E$ is a
constant force. The charge terms are radial functions $f(R)$ of
$\mathbf d = \mathbf R_A - \mathbf r_i$, $R = \lvert\mathbf d\rvert$.
Differentiate $f(R)$ once with $\partial R/\partial d_\alpha = d_\alpha/R$,

$$\frac{\partial f}{\partial d_\alpha} = f'\,\frac{d_\alpha}{R},$$

and again, with the product rule on $d_\alpha/R$:

$$\frac{\partial^2 f}{\partial d_\alpha\,\partial d_\beta}
= f''\,\frac{d_\alpha d_\beta}{R^2} + \frac{f'}{R}\Big(\delta_{\alpha\beta} - \frac{d_\alpha d_\beta}{R^2}\Big).$$

For a point charge, $f = 1/R$, $f' = -1/R^2$ and $f'' = 2/R^3$, so the
block of atom $A$ is

$$\frac{\partial^2 E_{\mathrm{cl}}}{\partial R_{A\alpha}\,\partial R_{A\beta}}
= Z_A q_i\Big(\frac{3\,d_\alpha d_\beta}{R^5} - \frac{\delta_{\alpha\beta}}{R^3}\Big).$$

Check the trace: $3R^2/R^5 - 3/R^3 = 0$, Laplace's equation for $1/R$
away from the charge. For a smeared charge,
$f = \mathrm{erf}(R/w)/R$. With $g = \mathrm{erf}(R/w)$ and
$g' = 2e^{-R^2/w^2}/(\sqrt\pi\,w)$, the radial derivatives are

$$f' = \frac{g'}{R} - \frac{g}{R^2},\qquad
f'' = -g'\Big(\frac{2}{w^2} + \frac{2}{R^2}\Big) + \frac{2g}{R^3}.$$

As $w \to 0$ at fixed $R$, $g \to 1$ and $g'$ vanishes like
$e^{-R^2/w^2}$, so the block becomes the point-charge block. The charges
do not move, so $E_{\mathrm{cl}}$ couples no two QM atoms: each charge adds
to its atom's own $3\times3$ block only. $E_{\mathrm{cl}}$ depends on
$\mathbf R_A - \mathbf r_i$ alone, so the QM rows of the Hessian sum to
minus the QM-charge cross block, which the code does not form. Both blocks
are summed in `ferric-core/src/external_potential.rs::charge_nuclear_hessian`
with $Z_A$ the effective charge; $f'$ and $f''$ are
`ferric-core/src/external_potential.rs::smeared_radial_derivs`.
`ferric-scf/src/hessian.rs::nuclear_term` adds them to the
nuclear-repulsion Hessian.

The ECP has no classical partner of its own.
`ferric-core/src/mol.rs::apply_ecp` records each ECP atom's core count, and
`ferric-core/src/mol.rs::effective_z` gives the charge $Z_A - n_{\mathrm{core}}$; the nuclear
repulsion, the nuclear-attraction operator and the charge terms above all
use that effective charge, so their Hessians need no change. A molecule
whose core counts do not match the basis's ECPs is refused
(`ferric-scf/src/hessian_ecp.rs::ecp_inconsistent`).

**Second derivatives from a stencil.** None of the three stencil terms has
an analytic second-derivative integral to call. The libint2 build has no
three-centre second derivatives and no derivatives of the dipole
integrals, and the ECP quadrature has first derivatives only. The code
builds the second derivative from analytic first derivatives instead. For
the skeleton $S_{xy} = \sum D\,V^{xy}$, let
$G_x(\mathbf R) = \sum D_{\mu\nu} V^x_{\mu\nu}(\mathbf R)$ at fixed $D$.
Displace one coordinate $y$ by $\pm h$ and $\pm 2h$ and take the
fourth-order central stencil

$$S_{xy} \approx \frac{G_x(\mathbf R - 2h\,\mathbf e_y) - 8\,G_x(\mathbf R - h\,\mathbf e_y)
+ 8\,G_x(\mathbf R + h\,\mathbf e_y) - G_x(\mathbf R + 2h\,\mathbf e_y)}{12\,h} .$$

Expand each $G_x$ in a Taylor series in $h$ about $\mathbf R$, with $G_x$
written as $\partial_x E_V$ along the $y$ line. The even powers cancel
between $\pm h$; the weights $(1,-8,8,-1)$ remove the $h^2$ term; what is
left is

$$S_{xy} = \partial_y G_x - \frac{h^4}{30}\,\partial_y^5 G_x + O(h^6).$$

At the step $h = 2\times10^{-3}$ Bohr the truncation coefficient is
$h^4/30 = 5.3\times10^{-13}$. A stencil of a first derivative divides by $h$
once, so its roundoff grows as $\epsilon/h$; a second difference of the
energy would divide by $h^2$. The code symmetrizes the ECP stencil result,
$(S + S^{\mathsf T})/2$: the exact skeleton is symmetric, so the
symmetrization removes only truncation error.

**Rigid per atom, and the mask.** A nuclear coordinate $y = (B,\beta)$
moves every function and every operator centre sitting on atom $B$. An
ECP integral $f(\mathbf P, \mathbf Q, \mathbf C)$ over a bra shell on
$\mathbf P$, a ket shell on $\mathbf Q$ and an ECP on $\mathbf C$ depends
on three centres, and by the chain rule

$$\frac{\partial f}{\partial R_{B\beta}} = \sum_{K \in \{P,Q,C\},\ K \text{ on } B} \frac{\partial f}{\partial K_\beta} .$$

So each stencil point must move all centres on $B$ together. Moving only
one shell of a pair that shares an atom computes a partial derivative, not
the atomic one.

The stencil then needs only some triples. Split $G_x$ into one term per
triple $(\mu\text{-shell}, \nu\text{-shell}, C)$ and look at the triples
whose partials cannot change when $B$ moves:

- No centre on $B$: the term does not depend on $\mathbf R_B$ at all, so
  every stencil point gives the same value. The weights sum to zero,
  $1 - 8 + 8 - 1 = 0$, so it contributes nothing.
- All three centres on $B$: the integral is unchanged by a common
  translation, $f(\mathbf P + \mathbf t, \mathbf Q + \mathbf t, \mathbf C + \mathbf t) = f(\mathbf P, \mathbf Q, \mathbf C)$.
  Differentiate that identity in $\mathbf P$, $\mathbf Q$ or $\mathbf C$:
  the partials are translation invariant too. Moving $B$ moves the whole
  triple, so again every stencil point gives the same value and the sum is
  zero.

Only triples with one or two centres on $B$ are evaluated.
`ferric-scf/src/hessian_ecp.rs::moving_triples` builds that mask:

<!-- excerpt: ferric-scf/src/hessian_ecp.rs::moving_triples -->
```rust
let on_b = [inp.shell_atom[i], inp.shell_atom[j], inp.ecp_atom[u]]
    .iter()
    .filter(|&&a| a == b)
    .count();
if on_b == 1 || on_b == 2 {
    mask[(i * ns + j) * ne + u] = 1;
    any = true;
}
```

| Code | Meaning |
|---|---|
| `[inp.shell_atom[i], inp.shell_atom[j], inp.ecp_atom[u]]` | the atoms of bra shell, ket shell and ECP centre |
| `on_b` | how many of the three sit on the displaced atom $B$ |
| `on_b == 1 \|\| on_b == 2` | the triple's partials change when $B$ moves; 0 and 3 contribute zero |

The loop runs over shell pairs $i \le j$ only, and
`ferric-scf/src/hessian_ecp.rs::folded_first_derivative` doubles the
density weight of the off-diagonal pairs, since $V$ and $D$ are symmetric.
It folds each partial onto its atom, as the ECP gradient does: the bra
partial to the bra shell's atom, the ket partial to the ket shell's atom,
the centre partial to the ECP's atom.
`ferric-scf/src/hessian_ecp.rs::ecp_skeleton_hessian` runs the stencil
over every atom $B$ and axis $\beta$.

The other two stencils use the same rigid displacement with a different
mask, because their operator centres do not sit on atoms:

- Smeared charges (`ferric-scf/src/hessian_smeared.rs::pair_stencil`). The
  sites are fixed, so only the two shells move. For each site and shell
  pair, the stencil runs over the atoms carrying a shell of the pair, and
  every shell of the pair on that atom moves. A pair with both shells on
  $B$ does contribute here: the site stays behind.
- The field (`ferric-scf/src/hessian_field.rs::field_skeleton_hessian`).
  The operator $\mathbf E\cdot\mathbf r$ has no centre, so $M_{\mu\nu}$
  depends only on the atoms of $\mu$ and $\nu$. For $\mu$, $\nu$ on the same
  atom $A$, write $\mathbf r = \mathbf R_A + (\mathbf r - \mathbf R_A)$:

  $$M_{\mu\nu} = \mathbf E\cdot\mathbf R_A\,S_{\mu\nu} + \mathbf E\cdot\langle\mu\vert\mathbf r - \mathbf R_A\vert\nu\rangle .$$

  Both $S_{\mu\nu}$ and $\langle\mu\vert\mathbf r-\mathbf R_A\vert\nu\rangle$
  move rigidly with $A$, so they do not depend on $\mathbf R_A$. The block
  is linear in $\mathbf R_A$: its second derivative is zero, and its first
  derivative is $E_\alpha S_{\mu\nu}$. The skeleton therefore skips the
  same-atom blocks. For each atom pair $a < c$ it builds the pair energy
  $g_{ac}(\mathbf R_a, \mathbf R_c) = 2\sum_{\mu\in a,\nu\in c} D_{\mu\nu}M_{\mu\nu}$
  on a two-atom sub-molecule, so a displaced evaluation is a small
  integral build, and differentiates its dipole values twice: the stencil
  $(-1,16,-30,16,-1)/(12h^2)$ for a repeated coordinate and the product of
  two first-derivative stencils, divided by $144h^2$, for two different
  ones. Both have $O(h^4)$ error and both vanish on a linear function, so
  skipping the linear same-atom blocks loses nothing. The first-derivative
  matrices for $F^x$ come from the first-derivative stencil of the integrals
  (`ferric-scf/src/hessian_field.rs::field_first_derivative_matrices`),
  which is exact on the linear same-atom block.

The constants are the same in all three files: the step
`ferric-scf/src/hessian_ecp.rs::STEP`, `ferric-scf/src/hessian_field.rs::STEP`
and `ferric-scf/src/hessian_smeared.rs::STEP` are $2\times10^{-3}$ Bohr; the
first-derivative stencil `ferric-scf/src/hessian_ecp.rs::D1` (and its
copies in the field and smeared files) is $(1,-8,8,-1)$ at offsets
$(-2,-1,1,2)$, divided by $12h$; the second-derivative stencil
`ferric-scf/src/hessian_field.rs::D2` is $(-1,16,-30,16,-1)$ at offsets
$-2$ to $2$, divided by $12h^2$. ECP shells are limited to
`ferric-scf/src/hessian_ecp.rs::MAX_L` ($l \le 3$, up to f); a higher shell
is a typed error, matching the four-centre second derivatives of the rest
of the Hessian.

A check you can run: on HI in def2-SVP (iodine carries a 28-electron ECP),
the fixed-density one-electron skeleton $\sum D\,\partial^2(T+V+V_{\mathrm{ECP}})$,
with the ECP part from the masked stencil, against a fourth-order
difference of $\sum D\,\partial h/\partial x$ taken on whole displaced
molecules, from the `3f612e0b3` build:

```python
import numpy as np
import ferric

bs = ferric.BasisSet.bundled("def2-svp")  # iodine carries a 28-electron ECP
BOHR = 0.529177210903  # Angstrom per Bohr


def hi(dz=0.0):  # move I along z by dz Angstrom
    return ferric.Molecule.from_xyz_string(f"2\n\nH 0 0.1 0\nI 0 0 {1.61 + dz:.10f}\n")


ints = ferric.AoIntegrals(hi(), bs)
rng = np.random.default_rng(0)
d = rng.standard_normal((ints.nao, ints.nao))
d = 0.05 * (d + d.T)  # any fixed symmetric density
H = np.asarray(ints.hess_hcore(d))  # sum D d2(T + V + V_ECP)/dx dy


def grad(dz):  # sum D dh/dx at the displaced geometry
    a = ferric.AoIntegrals(hi(dz), bs)
    return np.einsum("xmn,mn->x", np.asarray(a.hcore_deriv()), d)


h = 1e-3
g = {t: grad(t * h) for t in (-2, -1, 1, 2)}
fd = (8 * (g[1] - g[-1]) - (g[2] - g[-2])) / (12 * h / BOHR)
print("max |H[:, I_z] - FD|:", np.abs(H[:, 5] - fd).max())
```

```text
max |H[:, I_z] - FD|: 6.493472426427616e-11
```

The column's largest element is 2.16 Ha/Bohr², and a second difference of
$\sum D\,V_{\mathrm{ECP}}$ alone puts the ECP part of the iodine $zz$ element
near $-0.25$.

**Kohn–Sham.** $V$ is part of $h$ and does not enter $E_{xc}$, so on the
RKS and UKS routes the ECP adds exactly the two terms above (section
@sec:ks-hessian, "ECPs"); the code calls the same
`ferric-scf/src/hessian.rs::extra_skeleton` and `add_extra_first_order`.
Point charges, smeared charges and a field are refused on every KS route by
`ferric-scf/src/hessian/ks.rs::embedding_refusal`, which rejects any
external potential. On the Hartree–Fock routes the ECP is taken by RHF, UHF
and ROHF with exact or fitted J and K. The charges and the field are taken by
RHF and UHF with exact J and K: `ferric-scf/src/hessian.rs::config_refusal`
refuses them with fitted J or K, and
`ferric-scf/src/hessian.rs::reference_config_refusal` refuses them for ROHF.

**Design choices.** Why not difference the Hessian from gradients? A
gradient difference needs a new SCF per displaced geometry and carries the
SCF convergence error divided by $h$. The stencils here difference
integrals at a fixed density, which have no convergence error, and the
response stays analytic. Why a mask at all? Each ECP triple is a
numerical quadrature, and a stencil point re-evaluates every triple it
includes. The triples with no centre or all three centres on $B$ would be
evaluated four times per axis only to cancel, so the mask skips them. On `FERRIC_ECP_BACKEND=libecpint` the ECP
skeleton is refused (section @sec:hf-hessian-reproducibility-refusals):
only the default quadrature backend passes.

> **Verified equation {#eq:scf-hess-one-electron-operator}** · a one-electron operator adds $\sum D\,V^{xy}$, the classical Hessian, and $V^x$ in $F^x$
>
> On the toy system of section @sec:hf-gradient (integrals smooth in two
> parameters, non-orthogonal $S$, RHF re-solved at each point) with an
> added $V(\lambda)$ in $h$ and one nucleus beside a point charge, a smeared
> charge and a field: the Hessian with $\sum D\,V^{xy}$, the closed-form
> charge-nucleus blocks and $V^x$ in $F^x$ equals the fourth-order
> difference of the analytic gradient to $2\times10^{-11}$ (claim O1);
> dropping $V^x$ from $F^x$ misses by 0.39. $V = 0$ with no charges and no
> field gives the plain Hessian bit for bit (claim O0). The skeleton from
> the step-$2\times10^{-3}$ stencil of $\sum D\,V^x$ matches the exact one
> to $1\times10^{-12}$ (claim O2). In symbols: the field-nucleus energy has
> zero Hessian; the radial Hessian, the point-charge block and the smeared
> $f'$, $f''$ are the derivatives they claim to be (claim O3).
>
> | | |
> |---|---|
> | Proof | `scripts/hessian_one_electron_ops_sympy.py`, claims O0 to O3 |
> | Negative control | $V^x$ dropped from $F^x$ (O1 fails); the point-charge $3\,d_\alpha d_\beta$ changed to $2\,d_\alpha d_\beta$ (O1 and the small-width limit of O3 fail) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/hessian.rs::{extra_skeleton,add_extra_first_order,nuclear_term}`, `ferric-core/src/external_potential.rs::charge_nuclear_hessian` |
> | Code test | `embedded_hessian_matches_fd_of_embedded_gradient` in `crates/ferric-scf/tests/qmmm_hessian_fd.rs` |
>
> <!-- verified: id=scf.hess.one-electron-operator; script=scripts/hessian_one_electron_ops_sympy.py; claim=O1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/hessian.rs::{extra_skeleton,add_extra_first_order,nuclear_term},ferric-core/src/external_potential.rs::charge_nuclear_hessian; rust_test=ferric-scf/tests/qmmm_hessian_fd.rs::embedded_hessian_matches_fd_of_embedded_gradient; registry=hf-hessian -->

> **Verified equation {#eq:scf-hess-rigid-atom-stencil}** · the fourth-order stencils, the rigid-per-atom mask and the linear same-atom field block
>
> The stencil $(1,-8,8,-1)/(12h)$ of a first derivative leaves
> $-h^4 g^{(6)}/30$; $(-1,16,-30,16,-1)/(12h^2)$ and the product stencil over
> $144h^2$ are exact through $O(h^3)$; all weights sum to zero and the two
> second-derivative stencils vanish on linear functions (claim O4). On a
> one-dimensional model of three-centre Gaussian integrals, the stencil of
> the folded first derivatives over only the triples with one or two centres
> on the displaced atom equals the exact Hessian to $5\times10^{-12}$, and the
> triples with none or all centres on it add below $10^{-10}$ (claim O5).
> The same-atom dipole block is linear in $\mathbf R_A$ with slope
> $S_{\mu\nu}$, and the pair stencil that skips it reproduces the exact field
> skeleton to $2\times10^{-10}$ (claim O6).
>
> | | |
> |---|---|
> | Proof | `scripts/hessian_one_electron_ops_sympy.py`, claims O4 to O6 |
> | Negative control | the mask keeps only triples with one centre on the atom, or a displacement moves only the bra centre on the atom (O5 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/hessian_ecp.rs::{ecp_skeleton_hessian,moving_triples,folded_first_derivative}`, `ferric-scf/src/hessian_smeared.rs::pair_stencil`, `ferric-scf/src/hessian_field.rs::field_skeleton_hessian` |
> | Code test | `masked_stencil_matches_unmasked_stencil_of_the_gradient_matrices` in `crates/ferric-scf/tests/ecp_hessian.rs` |
>
> <!-- verified: id=scf.hess.rigid-atom-stencil; script=scripts/hessian_one_electron_ops_sympy.py; claim=O5; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/hessian_ecp.rs::{ecp_skeleton_hessian,moving_triples,folded_first_derivative},ferric-scf/src/hessian_smeared.rs::pair_stencil,ferric-scf/src/hessian_field.rs::field_skeleton_hessian; rust_test=ferric-scf/tests/ecp_hessian.rs::masked_stencil_matches_unmasked_stencil_of_the_gradient_matrices; registry=hf-hessian -->

What the proof does not cover: the integrals themselves. The toy $V$ is a
smooth matrix function, so the proof checks how a one-electron operator
enters the Hessian and how the stencils and the mask behave, not libint2's
point-charge derivatives, the three-centre engine or the ECP quadrature.
Those are covered by the Rust tests named in the boxes and by the
validation rows of section @sec:hf-hessian-numerical-anchors. The toy is
closed-shell RHF; the open-shell routes add the same $V^x$ to each spin's
$F^x$ and are not in the proof.

## The algorithm {#hf-hessian-algorithm}

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
the stability analysis of section @sec:stability surfacing inside the Hessian.

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

## Reproducibility and refusals {#hf-hessian-reproducibility-refusals}

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

## Options that change the equation {#hf-hessian-options-change-equation}

| Option | Effect | Where |
|---|---|---|
| density-fitted J, K | fitted skeletons, fitted $J^y$ and $K^y$ in $F^y$, the SCF's own `DfJ`/`DfK` in the CPHF kernel; nothing else changes | `ferric-scf/src/hessian_df.rs`, `ferric-scf/src/hessian_df_k.rs` |
| ECPs | skeleton $\sum D\,\partial^2V_{\mathrm{ECP}}$ by a masked fourth-order stencil of first-derivative ECP blocks, and $\partial V_{\mathrm{ECP}}/\partial x$ in $F^y$ (section @sec:hf-hessian-one-electron-operators) | `ferric-scf/src/hessian_ecp.rs` |
| external point charges | QM-QM block, charges fixed; analytic libint2 second derivatives and the charge–nucleus block; RHF and UHF with exact J and K | `ferric-scf/src/hessian.rs::{contract_external_deriv2,deriv1_1e_matrices}`, section @sec:hf-hessian-one-electron-operators |
| smeared charges | as point charges, with a stencil of three-centre first-derivative blocks | `ferric-scf/src/hessian_smeared.rs`, section @sec:hf-hessian-one-electron-operators |
| uniform field | the field-nuclear energy is linear in $\mathbf R$, so only the electronic term enters; dipole integrals differentiated by a stencil | `ferric-scf/src/hessian_field.rs`, section @sec:hf-hessian-one-electron-operators |
| Kohn–Sham | XC skeleton, XC first-order Fock matrix and $f_{xc}$ kernel in the CPHF | section @sec:ks-hessian |

## Numerical anchors {#hf-hessian-numerical-anchors}

| Check | Reference | Agreement | Test |
|---|---|---|---|
| RHF Hessian, skeleton and CPHF response: H2O, NH3, CH2O / cc-pVDZ; distorted H2O / def2-SVP | PySCF `hessian.rhf` analytic; PySCF finite differences of its gradient; ferric's gradient differenced | total at most 1.6e-7 Ha/Bohr² (at most 8.2e-8 at cc-pVDZ), bar 1e-6; skeleton at most 5.9e-10, bar 1e-8; frequencies at most 6.9e-4 cm⁻¹ | `crates/ferric-scf/tests/validation_rhf_hessian.rs`, `crates/ferric-scf/tests/rhf_hessian_fd.rs` |
| UHF Hessian: OH, NH2, CH2 / cc-pVDZ | PySCF `hessian.uhf` on the lowest internally stable UHF | total at most 3.8e-7 (OH; at most 7.7e-8 for NH2 and CH2), bar 2e-6; skeleton at most 1.4e-10; closed-shell water equals RHF term by term (response 2.8e-17) | `crates/ferric-scf/tests/validation_uhf_hessian.rs`, `crates/ferric-scf/tests/uhf_hessian_fd.rs` |
| ROHF Hessian: NH2, CH2, HO2 / cc-pVDZ; STO-3G, 6-31G | central FD of PySCF's ROHF gradient (PySCF has no ROHF Hessian); ferric's gradient differenced | vs ferric's differenced gradient at most 3.2e-10; vs PySCF FD 1.0e-6 or less, against PySCF's own h-vs-2h gap of 0.7e-6 to 2.6e-6; the Hessian without the response misses by 0.35 to 0.83 | `crates/ferric-scf/tests/validation_rohf_hessian.rs` |

<!-- figures: site/src/reference/validation.md "Analytic RHF Hessian (skeleton one- and two-electron, overlap/W and CPHF response terms)"; site/src/reference/validation.md "Analytic UHF Hessian (skeleton one- and two-electron"; site/src/reference/validation.md "Analytic ROHF Hessian (skeleton with the ROHF energy-weighted density" -->

Fitted-J/K, ECP, charge and field Hessians (the latter three in section
@sec:hf-hessian-one-electron-operators) have their own rows on the
validation page; their agreement with finite differences of the matching
gradient is at the $10^{-10}$ to $10^{-7}$ Ha/Bohr² level quoted there.

**Known limits.** RHF, UHF and ROHF references (Kohn–Sham: section
@sec:ks-hessian), with basis functions up to the angular momentum for which
the libint build carries second derivatives (f in the shipped build;
`ferric-scf/src/hessian.rs::require_hessian_classes`), and up to f for ECP
systems (`ferric-scf/src/hessian_ecp.rs::MAX_L`); no analytic ROKS Hessian (it refuses before the SCF, and ROKS
frequencies difference the analytic gradient); no solvent and no polarizable
embedding.

## Sources

| Source | Used for |
|---|---|
| `scripts/hf_hessian_sympy.py`, claims H1 to H4, S1, S2 | verified equations @eq:scf-hess-assembled and @eq:scf-hess-occ-occ-rotation |
| `scripts/hessian_factored_k_sympy.py`, claims F1 to F5 and its mutation list | the factored RI-K of section @sec:hf-hessian-factored-ri-k |
| `scripts/hessian_one_electron_ops_sympy.py`, claims O0 to O6 | section @sec:hf-hessian-one-electron-operators |
| `crates/ferric-scf/src/hessian.rs` module documentation | the term table, the solver, the refusals |
| `site/src/reference/validation.md` | the anchor rows and the known limits |
