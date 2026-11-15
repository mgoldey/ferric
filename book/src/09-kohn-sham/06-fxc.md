# 9.6 $f_{xc}$ response kernels

<!-- section: 9.6; status: ready; registry: ks-hessian, tddft;
     theory: crates/ferric-dft/theory.md#"$f_{xc}$ response kernels" -->

Every second-order KS method needs the change of $V_{xc}$ under a density
perturbation: the Newton and stability orbital Hessians, the Z-vector of
section 15.4, the CPKS polarizability, the analytic Hessian and TDDFT. ferric
has one kernel object per family, and each returns $\delta V^\sigma$ for a
pair of AO perturbations $(\delta D^\alpha, \delta D^\beta)$ by the same
back-projection that builds $V^\sigma$, with the first-order coefficients in
place of $v_\rho$ and $\mathbf G^\sigma$.

| Symbol | Meaning | Code name |
|---|---|---|
| $\delta D^\sigma$ | AO density perturbation | `dd_a`, `dd_b` |
| $f_{x_ix_j}$ | libxc second derivatives (`v2rho2`, `v2rhosigma`, `v2sigma2`, and the $\tau$ blocks) | `PointHessian` |
| $u^\sigma$, $\mathbf w^\sigma$, $t^\sigma$ | first-order $v_\rho$, $\mathbf G^\sigma$, $v_\tau$ | the three outputs of `linearize_point` |
| $K_{ia,jb}$ | the $ov\times ov$ kernel block for TDDFT | `k` |

| Entry point | Location |
|---|---|
| Rust | `ferric-dft/src/fxc.rs::{LdaFxcKernel,GgaFxcKernel,MggaFxcKernel}`, `ferric-dft/src/lr_kernel.rs::{SingletKernel,fxc_ov_block,singlet_fxc_ov_block,SingletOvFxc}` |
| Consumers | `ferric-scf/src/rohf.rs::FxcKernelStore` (Newton, stability, Z-vector), `ferric-tddft/src/tddft.rs::run_tddft` (chapter 20) |

## 9.6.1 The linearized potential

<!-- include: crates/ferric-dft/theory.md#"$f_{xc}$ response kernels" -->

> **Verified equation 9.9** · the GGA and meta-GGA kernels are the linearization of $V_{xc}$
>
> For a generic cubic $e(\rho_\alpha, \rho_\beta, \sigma_{\alpha\alpha}, \sigma_{\alpha\beta}, \sigma_{\beta\beta})$
> whose first and second derivatives are independent symbols, the code's
> $\delta\sigma$, $u^\sigma$ and $\mathbf w^\sigma$ equal
> $\tfrac{d}{dt}$ of $v_{\rho_\sigma}$ and $\mathbf G^\sigma$ along the
> perturbation; the LDA kernel is the $\sigma$-free case; for a concrete
> seven-variable $e$ with the second derivatives in libxc's packed order,
> `linearize_point`'s $(u, \mathbf w, t)$ equals a central difference of
> $(v_\rho, \mathbf G, v_\tau)$ within its derived bar.
>
> | | |
> |---|---|
> | Proof | `scripts/dft_fxc_kernel_sympy.py`, claims dsigma, u, w, lda, mgga |
> | Negative control | the runner reads `v2sigmatau` in the wrong packing, drops the factor 2 on $v_{\sigma_{\alpha\alpha}}\nabla\delta\rho_\alpha$ in `linearize_point`, and drops the 2 on $\delta v_{\sigma_{\alpha\alpha}}$ in $\mathbf w^\alpha$; the script's own controls swap two `v2sigma2` slots and use $2\nabla\rho_\alpha\cdot\nabla\delta\rho_\beta$ for $\delta\sigma_{\alpha\beta}$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-dft/src/fxc.rs::{linearize_point,scatter_tau}` |
> | Code test | `gga_fxc_matches_finite_difference_of_vxc`, `mgga_fxc_matches_finite_difference_of_production_vxc_polarized` in `crates/ferric-dft/src/fxc.rs` |
>
> <!-- verified: id=ks.fxc.linearization; script=scripts/dft_fxc_kernel_sympy.py; claim=mgga; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-dft/src/fxc.rs::linearize_point; rust_test=ferric-dft/src/fxc.rs::gga_fxc_matches_finite_difference_of_vxc; registry=ks-hessian -->

`linearize_point` codes $\delta\sigma$ and $\mathbf w^\sigma$ term for
term as in the theory section. The meta-GGA second derivatives enter the
per-point Hessian through `scatter_tau`, in libxc's packed order:

<!-- excerpt: ferric-dft/src/fxc.rs::scatter_tau -->
```rust
for t in 0..2 {
    for s in 0..2 {
        add_sym(h, RA + s, TA + t, rt[2 * s + t]);
    }
    for c in 0..3 {
        add_sym(h, SAA + c, TA + t, st[2 * c + t]);
    }
}
```

| Symbol | Code |
|---|---|
| $f_{\rho_s\tau_t}$ (`v2rhotau`) | `rt[2 * s + t]` |
| $f_{\sigma_c\tau_t}$ (`v2sigmatau`) | `st[2 * c + t]` |
| the 7-variable symmetric Hessian | `h`, filled by `add_sym` |

## 9.6.2 The singlet and triplet $ov$ block

TDDFT needs the kernel contracted with occupied–virtual products. For one
column $jb$, `fxc_ov_block` forms the symmetric AO perturbation
$\delta D = \tfrac12(\mathbf c_j\mathbf c_b^{\mathsf T} + \mathbf c_b\mathbf c_j^{\mathsf T})$,
applies the kernel with $\delta D^\beta = +\delta D$ (singlet) or
$-\delta D$ (triplet), and projects the $\alpha$ response:

$$ K_{ia,jb} = \mathbf c_i^{\mathsf T}\,\delta V^\alpha[\delta D, \pm\delta D]\,\mathbf c_a = (ia\vert f_{\alpha\alpha} \pm f_{\alpha\beta}\vert jb) . \tag{9.10} $$

<!-- excerpt: ferric-dft/src/lr_kernel.rs::fxc_ov_block -->
```rust
let (dv_a, _dv_b) = match spin {
    ResponseSpin::Singlet => kernel.apply(&dd, &dd),
    ResponseSpin::Triplet => kernel.apply(&dd, &(-&dd)),
};
// K[:, jb] = C_occ^T δV_α C_vir, flattened as ia = i*nvir + a.
let block = orbo.t().dot(&dv_a).dot(&orbv); // (nocc, nvir)
```

The block is then checked for symmetry (relative asymmetry above
`FXC_BLOCK_ASYMMETRY_TOL` $= 10^{-8}$ is an error, since it means the
adapter, not the grid, is wrong) and symmetrized. `SingletOvFxc` applies the
same kernel matrix-free for the Davidson solver.

> **Verified equation 9.10** · singlet kernel normalization and spin adaptation
>
> For a spin-symmetric $e(\rho_\alpha, \rho_\beta)$,
> $2\,\partial^2e(\rho/2, \rho/2)/\partial\rho^2 = f_{\alpha\alpha} + f_{\alpha\beta}$:
> the singlet kernel is twice the unpolarized second derivative; the
> spin-orbital response matrices block-diagonalize into singlet
> ($f_{\alpha\alpha} + f_{\alpha\beta}$) and triplet
> ($f_{\alpha\alpha} - f_{\alpha\beta}$) blocks.
>
> | | |
> |---|---|
> | Proof | `scripts/tddft_response_sympy.py`, claims A1, A2, B1 |
> | Negative control | the runner sets the triplet sign of `fxc_ov_block` to $+1$ (B1 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-dft/src/lr_kernel.rs::fxc_ov_block` |
> | Code test | `crates/ferric-tddft/tests/validation_tddft_rsh.rs` (triplets against PySCF) |
>
> <!-- verified: id=ks.fxc.ov-block; script=scripts/tddft_response_sympy.py; claim=A2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-dft/src/lr_kernel.rs::fxc_ov_block; registry=tddft -->

## 9.6.3 Coverage

| Family | Kernel | Consumers | Refused |
|---|---|---|---|
| LDA | `LdaFxcKernel` | all | |
| GGA, global hybrid, RSH | `GgaFxcKernel` (exact exchange through $c_{\rm sr}, c_{\rm lr}$ in the caller) | all | |
| $\tau$-only meta-GGA | `MggaFxcKernel` | RKS/UKS Newton, RKS/UKS stability, RKS/UKS/ROKS Z-vector, TDA and Casida | ROKS Newton and augmented Hessian, TRAH, the analytic nuclear Hessian |
| VV10 | none | | every response method refuses a VV10 functional |

The kernels zero every term where the reference $\rho_\sigma$ is at or below
the functional's density floor, as $V^\sigma$ does, so the linearization is
of the matrix the SCF actually used.

## 9.6.4 Numerical anchors

From `site/src/reference/validation.md`, "Meta-GGA (τ-only) $f_{xc}$ response".

| Quantity | Value | Test |
|---|---|---|
| meta-GGA kernel vs FD of the production $V_{xc}$ | $5.4\times10^{-9}$ (r2SCAN), $3.6\times10^{-7}$ (SCAN) relative | `crates/ferric-dft/src/fxc.rs` |
| RKS/UKS stability $\lambda_{\min}$, r2SCAN, vs PySCF dense orbital Hessian | $\le 1.5\times10^{-10}$ Ha | `crates/ferric-scf/tests/mgga_response_vs_pyscf.rs` |
| Newton vs DIIS energy | $\le 1.4\times10^{-14}$ Ha | the same |
| Z-vector exact gradient vs FD (MBD@rsSCS) | $8.0\times10^{-10}$ (RKS), $1.1\times10^{-11}$ (UKS), $2.0\times10^{-11}$ (ROKS) Ha/Bohr | `crates/ferric-rpa/tests/mbd_scf_gradient.rs` and its UKS and ROKS siblings |

## 9.6.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-dft/theory.md`, "$f_{xc}$ response kernels" | the equations and the libxc packing |
| `crates/ferric-scf/theory.md`, "Kohn-Sham XC Hessian terms (fixed and moving grid)" | the nuclear-Hessian use of the same kernel |
| `crates/ferric-tddft/theory.md`, "1. Response matrices" | the TDDFT use |
