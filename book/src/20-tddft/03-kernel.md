# 20.3 The kernel and the functional gate

<!-- section: 20.3; status: ready; registry: tddft;
     theory: crates/ferric-tddft/theory.md#"5. Functional gate and kernel" -->

The reference functional fixes two things in $\mathbf A$ and $\mathbf B$:
the exact-exchange coefficients on the exchange terms and the semilocal
kernel block $K$. `run_tddft` resolves both before any integral work, and a
functional whose response would be incomplete is refused at that point.

| Symbol | Meaning | Code name |
|---|---|---|
| $(c_{\rm sr}, c_{\rm lr}, \omega)$ | exchange coefficients of the response | `ExchangeMix`, `ResponseXc` |
| $K_{ia,jb} = (ia|f_{\alpha\alpha}+f_{\alpha\beta}|jb)$ | singlet kernel block | `fxc_ov_block`, `SingletOvFxc` |
| `c_hf` | optional override of a global fraction | `TddftConfig::c_hf_override` |

| Entry point | Location |
|---|---|
| Rust | `ferric-tddft/src/tddft.rs::resolve_response_xc`, `ferric-dft/src/lr_kernel.rs::resolve_response_xc`, `ferric-dft/src/grid.rs::response_default_grid` |
| CLI | `[tddft] xc`, `c_hf` |

## 20.3.1 The gate

<!-- include: crates/ferric-tddft/theory.md#"5. Functional gate and kernel" -->

`ferric_dft::lr_kernel::resolve_response_xc` reads the functional at
`nspin = 2` (the kernel is spin-resolved) and returns its exchange
coefficients. A CAM functional with $\omega = 0$ collapses to one fraction,
and a global hybrid has $c_{\rm sr} = c_{\rm lr}$:

<!-- excerpt: ferric-dft/src/lr_kernel.rs::resolve_response_xc -->
```rust
let (c_sr, c_lr, omega) = match probe.cam {
    Some(cam) if cam.omega != 0.0 => (cam.c_sr, cam.c_lr, cam.omega),
    // ω = 0: erf(0) = 0, so only the short-range (= full) term remains.
    Some(cam) => (cam.c_sr, cam.c_sr, 0.0),
    None => {
        let mix = probe.b3lyp_mix.unwrap_or(0.0);
        (mix, mix, 0.0)
    }
};
```

| Reference | Exchange | Kernel | Status |
|---|---|---|---|
| HF (no functional) | $c = 1$ | none | CIS or TDHF; a `c_hf` other than 1 is refused |
| LDA, GGA, global hybrid | $c_{\rm HF}$ from libxc | `GgaFxcKernel` | accepted; `c_hf` replaces the fraction |
| $\tau$-only meta-GGA, hybrid meta-GGA | from libxc | `MggaFxcKernel` | accepted |
| range-separated hybrid | $c_{\rm sr}, c_{\rm lr}, \omega$ | `GgaFxcKernel` | accepted (section 20.4); any `c_hf` is refused |
| meta-GGA with no libxc second derivative | | | refused |
| VV10 | | | refused: its response is not in the kernel |

The kernel is evaluated on the reference SCF's grid. For a meta-GGA the
default response grid is 150 × 434 with NWChem radial pruning
(`AtomicGridConfig::meta_gga_response`), because the $\tau$ kernel is more
grid-sensitive than the LDA and GGA ones; an explicit grid key wins.

## 20.3.2 The kernel block

The block is section 9.6.2's $K_{ia,jb}$ for the singlet, added to both
$\mathbf A$ and $\mathbf B$. For a spin-symmetric functional it is twice the
unpolarized second derivative, which is the factor 2 on `frr` in PySCF's
`tdscf.rhf.get_ab`.

> **Verified equation 20.7** · the singlet kernel is twice the unpolarized second derivative
>
> For a spin-symmetric $e(\rho_\alpha, \rho_\beta)$,
> $2\,\partial^2 e(\rho/2, \rho/2)/\partial\rho^2 = f_{\alpha\alpha} + f_{\alpha\beta}$,
> symbolically.
>
> | | |
> |---|---|
> | Proof | `scripts/tddft_response_sympy.py`, claim A2 |
> | Negative control | the runner gives the triplet the singlet sign in the transcribed `fxc_ov_block` (B1 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-dft/src/lr_kernel.rs::{fxc_ov_block,singlet_fxc_ov_block}` |
> | Code test | `crates/ferric-tddft/tests/validation_tddft.rs` (LDA, PBE, B3LYP and r2SCAN against PySCF) |
>
> <!-- verified: id=td.kernel.singlet-normalization; script=scripts/tddft_response_sympy.py; claim=A2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-dft/src/lr_kernel.rs::singlet_fxc_ov_block; registry=tddft -->

## 20.3.3 Grid sensitivity

From `site/src/reference/validation.md`, rows `tda` and "TDA and Casida
TDDFT excitation energies".

| Quantity | Value | Test |
|---|---|---|
| B3LYP roots vs PySCF, default pruned grid | $\le 8.1\times10^{-4}$ eV | `crates/ferric-tddft/tests/validation_tddft.rs` |
| the same on the reference's flat 75 × 110 grid | $\le 2.8\times10^{-6}$ eV | the same |
| r2SCAN water roots, 150 × 434 default vs flat 150 × 590 | $9.2\times10^{-5}$ eV (6-31G), $5.5\times10^{-6}$ eV (aug-cc-pVDZ) | `water_631g_r2scan_default_grid_matches_fine_flat_grid` in the same file |
| r2SCAN on 75 × 110 vs flat 150 × 590 | $2.1\times10^{-3}$ eV | the same |

## 20.3.4 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-tddft/theory.md`, section 5 | the gate |
| `crates/ferric-dft/theory.md`, "$f_{xc}$ response kernels" | the kernel and its libxc packing |
