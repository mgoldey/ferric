# 20.5 Triplets

<!-- section: 20.5; status: ready; registry: tddft;
     theory: crates/ferric-tddft/theory.md#"9. Triplets" -->

A closed-shell reference has the same spatial orbitals for both spins, so the
spin-orbital response problem splits into a singlet and an $M_S = 0$
triplet block. The triplet block has no Coulomb term and the opposite sign on
the $\alpha\beta$ kernel.

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

<!-- include: crates/ferric-tddft/theory.md#"9. Triplets" -->

Spin symmetry gives $\mathbf A_{\alpha\alpha} = \mathbf A_{\beta\beta}$ and
$\mathbf A_{\alpha\beta} = \mathbf A_{\beta\alpha}$, so

$$ \mathbf A^{S,T} = \mathbf A_{\alpha\alpha} \pm \mathbf A_{\alpha\beta},\qquad
A^T_{ia,jb} = \delta_{ij}\delta_{ab}\Delta_i^a - c\,(ij|ab) + K^T_{ia,jb},\qquad
B^T_{ia,jb} = -c\,(ib|ja) + K^T_{ia,jb} , \tag{20.9} $$

with $c(pq|rs)$ the exchange combination of section 20.4. The Coulomb term
cancels between the spin blocks, so the triplet Coulomb coefficient is 0:

<!-- excerpt: ferric-tddft/src/tddft.rs::coulomb_factor -->
```rust
match spin {
    ResponseSpin::Singlet => 2.0,
    ResponseSpin::Triplet => 0.0,
}
```

The kernel block is built by the same adapter as the singlet one with
$\delta D^\beta = -\delta D^\alpha$ (section 9.6.2). At a restricted point
the perturbation of
$(\rho_\alpha,\rho_\beta,\sigma_{\alpha\alpha},\sigma_{\alpha\beta},\sigma_{\beta\beta},\tau_\alpha,\tau_\beta)$
is $(\rho_1,\,s\rho_1,\,2\gamma_1,\,(1+s)\gamma_1,\,2s\gamma_1,\,\tau_1,\,s\tau_1)$,
which the matrix-free kernel applies in `spin_coefficients`.
$\mathbf A - \mathbf B$ is the same matrix for both spins, since the
Coulomb and kernel terms cancel in it. The $M_S = \pm1$ components are
degenerate with $M_S = 0$ and are not computed, and triplet oscillator
strengths are exactly zero.

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

$\mathbf A^T + \mathbf B^T$ is not positive definite when the reference is
triplet-unstable, and the Casida problem then has $\Omega^2 < 0$. Both
solvers refuse it rather than drop the root. RHF ethylene and benzene are
refused this way; the dense solver reports the reference's $\Omega^2_{\min}$
($-3.171\times10^{-3}$ Ha$^2$ for C2H4, from the validation table's `tddft`
row). TDA uses $\mathbf A^T$ alone and has no such failure.

## 20.5.3 Numerical anchors

From `site/src/reference/validation.md`, "Range-separated hybrid and triplet
TDA and Casida excitation energies": singlets and triplets alike match
PySCF to $\le 4.4\times10^{-7}$ eV over the lowest five roots for HF, LDA,
PBE, B3LYP, r2SCAN, CAM-B3LYP and ωB97X on water, C2H4 and benzene / 6-31G
(`crates/ferric-tddft/tests/validation_tddft_rsh.rs`).

## 20.5.4 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-tddft/theory.md`, section 9 | the spin blocks, the matrix-free kernel, the instability |
