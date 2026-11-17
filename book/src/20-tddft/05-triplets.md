# 20.5 Triplets

<!-- section: 20.5; status: ready; registry: tddft;
     theory: crates/ferric-tddft/theory.md#"9. Triplets" -->

Excite one electron of a closed shell and the two now-unpaired electrons
can pair their spins to a singlet or a triplet. The two states share an
orbital picture and differ in energy, and the triplet usually lies lower. In
response theory the difference is one sign: a singlet moves the $\alpha$
and $\beta$ densities together, a triplet moves them in opposite
directions. A closed-shell reference has the same spatial orbitals for both
spins, so the spin-orbital response problem splits exactly into a singlet
block and an $M_S = 0$ triplet block. The triplet block has no Coulomb
term and the opposite sign on the $\alpha\beta$ kernel.

| Symbol | Meaning | Code name |
|---|---|---|
| $X^S, X^T$ | $(X_\alpha \pm X_\beta)/\sqrt2$ | |
| $K^T_{ia,jb}$ | $(ia|f_{\alpha\alpha}-f_{\alpha\beta}|jb)$ | `fxc_ov_block` with `ResponseSpin::Triplet` |
| $s$ | $+1$ singlet, $-1$ triplet | the sign of $\delta D^\beta$ |

| Entry point | Location |
|---|---|
| Rust | `ferric-tddft/src/tddft.rs::{coulomb_factor,run_tddft}` with `TddftConfig::spin`; `ferric-dft/src/fxc.rs::spin_coefficients` |
| CLI | `[tddft] spin = "triplet"` |
| Related | `ferric-scf/src/rhf_newton.rs::triplet_hessian_matvec` (the RHF→UHF stability Hessian, the same spin channel at the orbital-rotation level) |

## 20.5.1 The spin blocks

Start in spin orbitals, with $\sigma, \tau \in \{\alpha, \beta\}$ and the
same spatial orbitals for both spins. Write $c(pq|rs)$ for the exchange
combination of section 20.4 (global, HF or range-separated). Then

$$ A_{ia\sigma,jb\tau} = \delta_{ij}\delta_{ab}\delta_{\sigma\tau}\,\Delta_i^a
+ (ia|jb) + (ia|f_{\sigma\tau}|jb) - \delta_{\sigma\tau}\,c\,(ij|ab), $$

$$ B_{ia\sigma,jb\tau} = (ia|jb) + (ia|f_{\sigma\tau}|jb) - \delta_{\sigma\tau}\,c\,(ib|ja) . $$

Each term carries its own spin rule. The orbital gap and exchange need
$\sigma = \tau$. The Coulomb term has no spin label, because the Coulomb
operator sees the total density. The kernel depends on the pair through
$f_{\sigma\tau}$.

The spatial orbitals are the same for both spins, so
$\mathbf A_{\alpha\alpha} = \mathbf A_{\beta\beta}$ and
$\mathbf A_{\alpha\beta} = \mathbf A_{\beta\alpha}$. A $2\times2$ block
matrix of that form is diagonalized by the sum and difference of its spin
components, $X^S = (X_\alpha + X_\beta)/\sqrt2$ (singlet) and
$X^T = (X_\alpha - X_\beta)/\sqrt2$ (the $M_S = 0$ triplet):

$$ \mathbf A^S = \mathbf A_{\alpha\alpha} + \mathbf A_{\alpha\beta},\qquad
\mathbf A^T = \mathbf A_{\alpha\alpha} - \mathbf A_{\alpha\beta}, $$

and the same for $\mathbf B$. Substitute the spin-orbital elements. In the
sum, the Coulomb term appears twice and gives the $2(ia|jb)$ of equation
20.1. In the difference it cancels, and so do the parts of the kernel that
do not depend on spin:

$$ \mathbf A^{S,T} = \mathbf A_{\alpha\alpha} \pm \mathbf A_{\alpha\beta},\qquad
A^T_{ia,jb} = \delta_{ij}\delta_{ab}\Delta_i^a - c\,(ij|ab) + K^T_{ia,jb},\qquad
B^T_{ia,jb} = -c\,(ib|ja) + K^T_{ia,jb} , \tag{20.9} $$

with $K^T_{ia,jb} = (ia|f_{\alpha\alpha}-f_{\alpha\beta}|jb)$.

Check the limits. For HF ($K = 0$, $c = 1$) the triplet matrix is the
orbital gap minus exchange. A single excitation $i\to a$ then has diagonal
elements that differ by $A^S_{ia,ia} - A^T_{ia,ia} = 2(ia|ia)$, the Coulomb
self-repulsion of the transition density, which is never negative: the
triplet pays none of it. For LDA exchange alone,
$f_{\alpha\beta} = 0$ (section 9.6.2), so singlet and triplet differ only
by the Coulomb term; with correlation the kernel differs as well.

$\mathbf A - \mathbf B$ is the same matrix for both spins. In
$\mathbf A^S - \mathbf B^S$ the $2(ia|jb)$ and the kernel cancel; in the
triplet they are absent or cancel. Both leave
$\Delta - c\,[(ij|ab) - (ib|ja)]$. Only $\mathbf A + \mathbf B$ tells the
spins apart.

The Coulomb coefficient of the triplet is therefore 0:

<!-- excerpt: ferric-tddft/src/tddft.rs::coulomb_factor -->
```rust
match spin {
    ResponseSpin::Singlet => 2.0,
    ResponseSpin::Triplet => 0.0,
}
```

`build_a_matrix` skips the Coulomb product when `c_j` is 0, so a triplet
run never forms $B_{ov}^{\mathsf T}B_{ov}$ for $\mathbf A$.

### The kernel block

`ferric_dft::lr_kernel` applies the spin-resolved AO kernel to
$(\delta D^\alpha, \delta D^\beta) = (\delta D, s\,\delta D)$, with $s = +1$
for the singlet and $s = -1$ for the triplet, and keeps
$\delta V_\alpha = \int(f_{\alpha\alpha} + s f_{\alpha\beta})\,\delta\rho$.
So $K^T$ comes from the same adapter as the singlet block with the sign of
the $\beta$ perturbation flipped (section 9.6.2), and it carries the
singlet block's normalization.

The matrix-free kernel needs the same sign one level down, in the
variables libxc differentiates. At a restricted point
$\rho_\alpha = \rho_\beta$ and $\nabla\rho_\alpha = \nabla\rho_\beta = \mathbf g$.
Take the triplet perturbation $\delta\rho_\beta = -\delta\rho_\alpha = -\rho_1$,
with $\nabla\delta\rho_\beta = -\nabla\delta\rho_\alpha$,
$\delta\tau_\beta = -\delta\tau_\alpha = -\tau_1$ and
$\gamma_1 = \mathbf g\cdot\nabla\delta\rho_\alpha$. Linearize each variable of
$x = (\rho_\alpha,\rho_\beta,\sigma_{\alpha\alpha},\sigma_{\alpha\beta},\sigma_{\beta\beta},\tau_\alpha,\tau_\beta)$.
For example
$\delta\sigma_{\alpha\beta} = \mathbf g\cdot\nabla\delta\rho_\beta + \mathbf g\cdot\nabla\delta\rho_\alpha = 0$,
and $\delta\sigma_{\alpha\alpha} = 2\gamma_1$. The result is

$$ \delta x = (\rho_1,\,-\rho_1,\,2\gamma_1,\,0,\,-2\gamma_1,\,\tau_1,\,-\tau_1) . $$

For the singlet it is
$(\rho_1, \rho_1, 2\gamma_1, 2\gamma_1, 2\gamma_1, \tau_1, \tau_1)$. Both are
$\delta x = (\rho_1,\,s\rho_1,\,2\gamma_1,\,(1+s)\gamma_1,\,2s\gamma_1,\,\tau_1,\,s\tau_1)$.
With $H$ the matrix of libxc second derivatives, row $m$ of
$\delta v = H\,\delta x$ is

$$ \delta v_m = (H_{m\rho_\alpha} + sH_{m\rho_\beta})\,\rho_1
+ \big(2H_{m\sigma_{\alpha\alpha}} + (1+s)H_{m\sigma_{\alpha\beta}} + 2sH_{m\sigma_{\beta\beta}}\big)\,\gamma_1
+ (H_{m\tau_\alpha} + sH_{m\tau_\beta})\,\tau_1 . $$

The $\alpha$ potential's gradient coefficient is
$\mathbf w^\alpha = (2\delta v_{\sigma_{\alpha\alpha}} + \delta v_{\sigma_{\alpha\beta}})\,\mathbf g
+ 2v_{\sigma_{\alpha\alpha}}\nabla\delta\rho_\alpha + v_{\sigma_{\alpha\beta}}\nabla\delta\rho_\beta$.
Substitute $\nabla\delta\rho_\beta = s\nabla\delta\rho_\alpha$:
$(2\delta v_{\sigma_{\alpha\alpha}} + \delta v_{\sigma_{\alpha\beta}})\,\mathbf g
+ (2v_{\sigma_{\alpha\alpha}} + s\,v_{\sigma_{\alpha\beta}})\nabla\delta\rho_\alpha$.
The density and $\tau$ coefficients are $u^\alpha = \delta v_{\rho_\alpha}$ and
$t^\alpha = \delta v_{\tau_\alpha}$, as for the singlet. The triplet branch of
`ferric-dft/src/fxc.rs::spin_coefficients` computes these coefficients:

<!-- excerpt: ferric-dft/src/fxc.rs::spin_coefficients -->
```rust
let row = |i: usize| {
    [
        wt * (h[i][RA] - h[i][RB]),
        wt * (2.0 * (h[i][SAA] - h[i][SBB])),
        wt * (h[i][TA] - h[i][TB]),
    ]
};
let (ra, saa, sab, ta) = (row(RA), row(SAA), row(SAB), row(TA));
let s = [
    2.0 * saa[0] + sab[0],
    2.0 * saa[1] + sab[1],
    2.0 * saa[2] + sab[2],
];
(ra, s, ta, wt * (2.0 * vs[0] - vs[1]))
```

| Code | Term, at $s = -1$ |
|---|---|
| `row(i)` | the coefficients of $\rho_1$, $\gamma_1$ and $\tau_1$ in $\delta v_i$, times the grid weight `wt`; the $\sigma_{\alpha\beta}$ coefficient $(1+s)$ is 0 |
| `ra`, `ta` | $u^\alpha = \delta v_{\rho_\alpha}$ and $t^\alpha = \delta v_{\tau_\alpha}$ |
| `s` | $2\delta v_{\sigma_{\alpha\alpha}} + \delta v_{\sigma_{\alpha\beta}}$, the coefficient of $\mathbf g$ |
| `2.0 * vs[0] - vs[1]` | $2v_{\sigma_{\alpha\alpha}} + s\,v_{\sigma_{\alpha\beta}}$, the coefficient of $\nabla\delta\rho_\alpha$ |

The singlet branch returns `restricted_coefficients`. Both branches are
pinned against the spin-resolved `linearize_point` in the unit test
`spin_coefficients_match_linearize_point`.

### What is not computed

The $M_S = \pm1$ components are degenerate with $M_S = 0$ and are not
computed. Triplet transition dipoles and oscillator strengths are exactly
zero: the dipole is a spin-free one-electron operator, so its matrix
element with $X^T$ is
$\sum_{ia}(X_\alpha - X_\beta)/\sqrt2\cdot\langle i|\mathbf r|a\rangle$, a sum
of equal and opposite spin amplitudes.

> **Verified equation 20.9** · the triplet block and its kernel sign
>
> Symbolically, the spin rotation $U$ block-diagonalizes the spin-orbital
> $\mathbf A$ and $\mathbf B$ with the triplet block at $c_J = 0$ and
> kernel $f_{\alpha\alpha} - f_{\alpha\beta}$; numerically, the code's
> triplet blocks equal the rotated spin-orbital matrices for HF, a global
> hybrid and an RSH, and the spin-orbital RPA roots are the union of the
> singlet and triplet roots.
>
> | | |
> |---|---|
> | Proof | `scripts/tddft_response_sympy.py`, claims A1, B1, B4 |
> | Negative control | the runner sets the triplet kernel sign $s$ to $+1$ (B1 fails); in the Rust validation, the singlet sign moves the triplet roots by 0.07 to 0.51 eV |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-tddft/src/tddft.rs::coulomb_factor`, `ferric-dft/src/lr_kernel.rs::fxc_ov_block` |
> | Code test | `crates/ferric-tddft/tests/validation_tddft_rsh.rs` (triplets against the PySCF `tdscf.uhf.get_ab` spin blocks) |
>
> <!-- verified: id=td.triplet.blocks; script=scripts/tddft_response_sympy.py; claim=A1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-tddft/src/tddft.rs::coulomb_factor; registry=tddft -->

## 20.5.2 Triplet instability

$\mathbf A^T + \mathbf B^T = \Delta - c(K_A + K_B) + 2K^T$, with $K_A$ and
$K_B$ the exchange integrals of $\mathbf A$ and $\mathbf B$. For an HF
reference this is
$(\varepsilon_a - \varepsilon_i)\delta - [(ab|ij) + (aj|bi)]$, the RHF
external (RHF → UHF) stability operator of section 8.4.1. Section 20.1.2
showed that, with $\mathbf A - \mathbf B$ positive definite, the Casida
problem has $\Omega^2 < 0$ exactly when $\mathbf A + \mathbf B$ has a
negative eigenvalue. So a triplet Casida run fails when the reference is
triplet-unstable: a lower, spin-broken solution exists.

Both solvers refuse a negative $\Omega^2$ in place of dropping the root.
RHF ethylene, the textbook case, and benzene are refused this way; the dense solver reports the reference's
$\Omega^2_{\min}$ ($-3.171\times10^{-3}$ Ha$^2$ for C2H4, from the
validation table's `tddft` row). TDA uses $\mathbf A^T$ alone and has no
such failure.

## 20.5.3 Numerical anchors

From `site/src/reference/validation.md`, "Range-separated hybrid and triplet
TDA and Casida excitation energies": singlets and triplets alike match
PySCF to $\le 4.4\times10^{-7}$ eV over the lowest five roots for HF, LDA,
PBE, B3LYP, r2SCAN, CAM-B3LYP and ωB97X on water, C2H4 and benzene / 6-31G
(`crates/ferric-tddft/tests/validation_tddft_rsh.rs`). The reference for
the triplets is the $\mathbf A_{\alpha\alpha} - \mathbf A_{\alpha\beta}$
combination of PySCF's `tdscf.uhf.get_ab` spin blocks on the same orbitals.

## 20.5.4 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-tddft/theory.md`, section 9 | the spin blocks, the matrix-free kernel, the instability |
