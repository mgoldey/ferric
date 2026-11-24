# The kernel and which functionals are accepted {#kernel}

<!-- status: ready; registry: tddft -->

When an electron moves from $i$ to $a$, the density changes, and so does
the exchange-correlation potential every other electron feels. That
feedback is the kernel block $K$ of equation @eq:td-matrices-spin-adaptation. The reference functional
fixes it, and it fixes the exact-exchange coefficients on the exchange
terms. If either piece is missing, the excitation energies are wrong by an
amount nothing in the output reveals. So `run_tddft` resolves both before
any integral work, and refuses a functional whose response would be
incomplete. A refused functional costs only the error message.

| Symbol | Meaning | Code name |
|---|---|---|
| $(c_{\rm sr}, c_{\rm lr}, \omega)$ | exchange coefficients of the response | `ExchangeMix`, `ResponseXc` |
| $K_{ia,jb} = (ia|f_{\alpha\alpha}+f_{\alpha\beta}|jb)$ | singlet kernel block | `fxc_ov_block`, `SingletOvFxc` |
| `c_hf` | optional override of a global fraction | `TddftConfig::c_hf_override` |

| Entry point | Location |
|---|---|
| Rust | `ferric-tddft/src/tddft.rs::resolve_response_xc`, `ferric-dft/src/lr_kernel.rs::resolve_response_xc`, `ferric-dft/src/grid.rs::response_default_grid` |
| CLI | `[tddft] xc`, `c_hf` |

## The gate {#kernel-gate}

`run_tddft` calls `ferric-tddft/src/tddft.rs::resolve_response_xc`. With
no functional the reference is HF and the coefficients are $c = 1$. For a
named functional it calls `ferric-dft/src/lr_kernel.rs::resolve_response_xc`,
which maps the functional to its exchange coefficients
$(c_{\rm sr}, c_{\rm lr}, \omega)$ and its kernel. A global hybrid has
$c_{\rm sr} = c_{\rm lr} = c_{\rm HF}$ and $\omega = 0$.

The library function reads the functional at `nspin = 2`, because the
kernel is spin-resolved: the singlet and triplet blocks come from feeding
it $\delta D^\beta = \pm\delta D^\alpha$ (section @sec:fxc-singlet-triplet-block). It refuses a
functional that carries VV10 and a meta-GGA with no libxc second
derivative, then reads the coefficients:

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

The three arms:

- **A CAM functional with $\omega \ne 0$** keeps both fractions and the
  range parameter (section @sec:rsh).
- **A CAM functional with $\omega = 0$** has $\mathrm{erf}(0) = 0$, so the
  long-range operator vanishes and the short-range one is the full
  Coulomb operator. One fraction, $c_{\rm sr}$, remains.
- **Anything else** has one global fraction, zero for a pure functional.

A second entry point, `ferric-dft/src/lr_kernel.rs::resolve_singlet_response_xc`,
also refuses $\omega \ne 0$. It serves drivers that build only
full-Coulomb exchange, not `run_tddft`.

| Reference | Exchange | Kernel | Status |
|---|---|---|---|
| HF (no functional) | $c = 1$ | none | CIS or TDHF; a `c_hf` other than 1 is refused |
| LDA, GGA, global hybrid | $c_{\rm HF}$ from libxc | `GgaFxcKernel` | accepted; `c_hf` replaces the fraction |
| $\tau$-only meta-GGA, hybrid meta-GGA (TPSSh, M06-2X) | from libxc | `MggaFxcKernel` | accepted |
| range-separated hybrid | $c_{\rm sr}, c_{\rm lr}, \omega$ | `GgaFxcKernel` | accepted (section @sec:rsh); any `c_hf` is refused |
| meta-GGA with no libxc second derivative | | | refused |
| VV10 | | | refused: its response is not in the kernel |
| any functional on an ROHF or ROKS reference | | | refused (section @sec:unrestricted-occupations-refusals) |

The $\tau$-only meta-GGAs are SCAN, r2SCAN and TPSS. The hybrid meta-GGAs
TPSSh and M06-2X are validated against PySCF on water / 6-31G
(`water_631g_tpssh`, `water_631g_m062x` in
`crates/ferric-tddft/tests/validation_tddft.rs`). The `c_hf` refusals
follow from what an override would mean. On an HF reference any value
other than 1 is neither CIS or TDHF nor any TDDFT, since there is no kernel
to pair it with. On a range-separated functional, at any value, it would
not say which range it replaces. On a global functional it replaces
$c_{\rm HF}$.

Singlet and $M_S = 0$ triplet excitations of a closed-shell reference are
both available (section @sec:triplets). UHF and UKS references take the
spin-conserving problem of section @sec:unrestricted through the same gate; ROHF and
ROKS references are refused there.

The kernel is evaluated on `TddftConfig.grid`, the grid of the reference
SCF. For a meta-GGA the default is 150 × 434 with NWChem radial pruning
(`ferric-dft/src/grid.rs::response_default_grid`, which returns
`AtomicGridConfig::meta_gga_response`), because the $\tau$ kernel is more
grid-sensitive than the LDA and GGA ones. The finer default keeps the
radial pruning and raises the shell count, so the thinned atoms still have
enough shells (section @sec:kernel-grid-sensitivity measures it). An
explicit grid key wins.

## The kernel block {#kernel-kernel-block}

The block is section @sec:fxc-singlet-triplet-block's $K_{ia,jb}$ for the singlet, added to both
$\mathbf A$ and $\mathbf B$. How does it relate to the closed-shell kernel
other codes use? Take a spin-symmetric energy density
$e(\rho_\alpha, \rho_\beta)$ and evaluate it at
$\rho_\alpha = \rho_\beta = \rho/2$. Differentiate twice by the chain rule:

$$ \frac{d^2}{d\rho^2}\,e\big(\tfrac\rho2, \tfrac\rho2\big)
= \tfrac14\big(f_{\alpha\alpha} + 2f_{\alpha\beta} + f_{\beta\beta}\big)
= \tfrac12\big(f_{\alpha\alpha} + f_{\alpha\beta}\big), $$

using $f_{\alpha\alpha} = f_{\beta\beta}$ at the symmetric point. So the
singlet kernel $f_{\alpha\alpha} + f_{\alpha\beta}$ is twice the
unpolarized second derivative. That factor 2 is the 2 on `frr` in PySCF's
`tdscf.rhf.get_ab`, and the reason ferric's block matches PySCF's `iajb`
without rescaling.

> **Verified equation {#eq:td-kernel-singlet-normalization}** · the singlet kernel is twice the unpolarized second derivative
>
> For a spin-symmetric $e(\rho_\alpha, \rho_\beta)$,
> $2\,\partial^2 e(\rho/2, \rho/2)/\partial\rho^2 = f_{\alpha\alpha} + f_{\alpha\beta}$,
> symbolically.
>
> | | |
> |---|---|
> | Proof | `scripts/tddft_response_sympy.py`, claim A2 |
> | Negative control | without the factor 2, $\partial^2 e(\rho/2,\rho/2)/\partial\rho^2 \ne f_{\alpha\alpha}+f_{\alpha\beta}$; the runner's mutation drops the 2 and A2 fails |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-dft/src/lr_kernel.rs::{fxc_ov_block,singlet_fxc_ov_block}` |
> | Code test | `check_case` in `crates/ferric-tddft/tests/validation_tddft.rs` (`water_631g_lda`, `water_631g_pbe`, `water_631g_b3lyp`, `water_631g_r2scan` against PySCF) |
>
> <!-- verified: id=td.kernel.singlet-normalization; script=scripts/tddft_response_sympy.py; claim=A2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-dft/src/lr_kernel.rs::singlet_fxc_ov_block; rust_test=ferric-tddft/tests/validation_tddft.rs::check_case; registry=tddft -->

`ferric-tddft/src/tddft.rs::add_fxc_block` checks the block before adding
it. Its relative asymmetry must stay below `FXC_BLOCK_ASYMMETRY_TOL`
$= 10^{-8}$; a
larger one means the AO-to-$(ia)$ adapter is inconsistent, and
symmetrizing would hide it, so the run stops.

## Grid sensitivity {#kernel-grid-sensitivity}

The kernel is evaluated point by point on the grid, so the roots inherit
the grid's quadrature error. For B3LYP, ferric's pruned default grid and
PySCF's flat grid give roots up to $8.1\times10^{-4}$ eV apart; on the same
flat grid they agree to $2.8\times10^{-6}$ eV.

Meta-GGAs are more sensitive. The production scheme's radial thinning
(H 50, O 66 shells) moves r2SCAN roots by up to $1.8\times10^{-3}$ eV,
against $1.6\times10^{-7}$ eV for PBE. That shift is grid error, not a
defect of the kernel. PySCF run on ferric's own pruned
coordinates and weights agrees with ferric to $6\times10^{-8}$ eV (the
comment on `kernel_grid` in `crates/ferric-tddft/tests/validation_tddft.rs`;
`site/src/methods/rpa-gw.md`, "Meta-GGA response needs a finer grid"). The
finer meta-GGA default of section @sec:kernel-gate is the fix, and the table below
measures it.

From `site/src/reference/validation.md`, rows `tda` and "TDA and Casida
TDDFT excitation energies".

| Quantity | Value | Test |
|---|---|---|
| B3LYP roots vs PySCF, default pruned grid (grid error) | $\le 8.1\times10^{-4}$ eV | `crates/ferric-tddft/tests/validation_tddft.rs` |
| the same on the reference's flat 75 × 110 grid (implementation error at a fixed grid) | $\le 2.8\times10^{-6}$ eV | the same |
| r2SCAN water roots, 150 × 434 default vs flat 150 × 590 | $9.2\times10^{-5}$ eV (6-31G), $5.5\times10^{-6}$ eV (aug-cc-pVDZ) | `water_631g_r2scan_default_grid_matches_fine_flat_grid` in the same file |
| r2SCAN on 75 × 110 vs flat 150 × 590 (grid error; against PySCF on one flat 75 × 110 grid the roots agree to $2.1\times10^{-6}$ eV) | $2.1\times10^{-3}$ eV | the same |

## Sources {#kernel-sources}

| Source | What it gives |
|---|---|
| section @sec:fxc-linearized-potential | the kernel and its libxc packing |
| R. Bauernschmitt and R. Ahlrichs, *Chem. Phys. Lett.* **256**, 454 (1996) | the adiabatic kernel with a hybrid's exact-exchange fraction |
| Q. Sun *et al.*, *J. Chem. Phys.* **153**, 024109 (2020) | PySCF, whose `tdscf` `get_ab` is the reference for the kernel normalization |
