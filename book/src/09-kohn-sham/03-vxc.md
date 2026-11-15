# 9.3 $V_{xc}$ matrix elements: LDA, GGA, meta-GGA

<!-- section: 9.3; status: ready; registry: ks-semilocal, ks-mgga;
     theory: crates/ferric-dft/theory.md#"Numerical Integration" "Semilocal $V_{xc}$ matrix elements" -->

The XC matrix is the derivative of the grid energy with respect to the
density matrix, so the SCF Fock matrix is exactly the gradient of the energy
it reports, whatever the grid. ferric builds it in the matrix-element form,
which needs the density and its first derivatives on the grid but no second
derivatives.

| Symbol | Meaning | Code name |
|---|---|---|
| $\chi_\mu(\mathbf r_g)$, $\partial_b\chi_\mu$ | AO values and gradients on the grid | `chi`, `dchi` |
| $\rho$, $\nabla\rho$, $\sigma$, $\tau$ | density, gradient, $\lvert\nabla\rho\rvert^2$, kinetic-energy density | `rho`, `grad`, `sigma`, `tau` |
| $\epsilon$ | XC energy per particle, from libxc | `exc` |
| $v_\rho, v_\sigma, v_\tau$ | $\partial(\rho\epsilon)/\partial(\rho, \sigma, \tau)$, from libxc | `vrho`, `vsigma`, `vtau` |
| $\mathbf G^\sigma$ | polarized gradient coefficient | |

| Entry point | Location |
|---|---|
| Rust, production | `ferric-dft/src/xc_batch.rs` behind `ferric-dft/src/ks.rs::KsXc` |
| Rust, dense reference | `ferric-dft/src/vxc.rs::{semilocal_vxc_closed,semilocal_vxc_polarized}`, `ferric-dft/src/density_on_grid.rs::{eval_density_closed,eval_tau_closed}` |
| Python | `AoIntegrals.veff`, the XC blocks tested in `crates/ferric-python/tests/test_xc_blocks.py` |

## 9.3.1 Grid quantities

<!-- include: crates/ferric-dft/theory.md#"Numerical Integration" -->

## 9.3.2 The matrix elements

<!-- include: crates/ferric-dft/theory.md#"Semilocal $V_{xc}$ matrix elements" -->

> **Verified equation 9.4** · $V_{xc} = \partial E_{xc}/\partial D$, closed shell and polarized
>
> For a generic $e(\rho, \sigma, \tau)$ (closed shell) and a generic
> seven-variable polarized $e$, the code's matrix elements, with the GGA
> factor 2, the cross coefficient 1 on $v_{\sigma_{\alpha\beta}}$ and the
> $\tfrac12$ on $v_\tau$, equal $\partial E/\partial D_{\mu\nu}$
> symbolically; on a 4-point, 3-AO model with a concrete meta-GGA-like $e$,
> a central difference of $E_{xc}(D)$ equals $\mathrm{Tr}[V X]$ within its
> derived truncation bar; the matrix-element and potential forms differ by a
> divergence.
>
> | | |
> |---|---|
> | Proof | `scripts/dft_vxc_matrix_elements_sympy.py`, claims closed, polarized, numeric, ibp |
> | Negative control | the runner drops the $\tfrac12$ on $v_\tau$ (numeric closed fails), sets the polarized cross coefficient to 2, and halves the GGA factor; the script's own controls fail the same way symbolically |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` (validation tier: about 12 minutes) |
> | Code | `ferric-dft/src/vxc.rs::{semilocal_vxc_closed_scratch,semilocal_vxc_polarized_scratch}` |
> | Code test | `tau_from_density_matrix_matches_explicit_mo_sum` in `crates/ferric-dft/src/density_on_grid.rs` |
>
> <!-- verified: id=ks.vxc.matrix-elements; script=scripts/dft_vxc_matrix_elements_sympy.py; claim=closed; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-dft/src/vxc.rs::{semilocal_vxc_closed_scratch,semilocal_vxc_polarized_scratch}; rust_test=ferric-dft/src/density_on_grid.rs::tau_from_density_matrix_matches_explicit_mo_sum; registry=ks-mgga -->

The production check of the same identity is end to end: a central
difference of ferric's own $E_{xc}$ along a random symmetric $X$ against
$\mathrm{Tr}[V_{xc}X]$ for PBE, SCAN and ωB97X-V, closed and polarized, to
$10^{-9}$ relative (`test_vxc_is_the_derivative_of_exc` in
`crates/ferric-python/tests/test_xc_blocks.py`).

## 9.3.3 The batched algorithm

The dense functions above hold $\chi$ and $\nabla\chi$ over the whole grid.
The SCF uses `xc_batch.rs`, which computes the same matrix in spatial
batches and calls the same libxc kernels, so the two cannot drift.

| Step | What | Cost |
|---|---|---|
| 1 | bisect the grid into compact batches of at most 128 points (pure function of the coordinates) | once per grid |
| 2 | per batch, keep a shell if any $\lvert\chi\rvert$ or $\lvert\partial\chi\rvert$ on the batch exceeds $10^{-10}$ (actual values, not a bound) | once per grid |
| 3 | per batch, $\Phi = D_{\rm sub}[\chi\,\vert\,\partial_x\chi\,\vert\,\partial_y\chi\,\vert\,\partial_z\chi]$; $\rho$, $\nabla\rho$, $\tau$ | one GEMM, $O(n_{\rm act}^2 n_b)$ |
| 4 | libxc once over the whole grid: $\epsilon$, $v_\rho$, $v_\sigma$, $v_\tau$ | $O(N_{\rm grid})$ |
| 5 | $E_{xc} = \sum_g w_g\rho_g\epsilon_g$, fixed grouping | $O(N_{\rm grid})$ |
| 6 | per batch, $V_b = A B^{\mathsf T}$, $B = [\chi\,\vert\,\partial\chi]$, $A = [\tfrac12 s\chi + \sum_a f_a\partial_a\chi\,\vert\,\tfrac12t\,\partial\chi]$; $V_{\rm sub} \mathrel{+}= V_b + V_b^{\mathsf T}$ | one GEMM |

with $s = wv_\rho$, $f_a = 2wv_\sigma\partial_a\rho$, $t = \tfrac12wv_\tau$
(below the density floor all three are zero). $V_b + V_b^{\mathsf T}$ is the
dense $\tfrac12(V + V^{\mathsf T})$ exactly: the LDA and $\tau$ pieces are
symmetric, and the GGA piece is $\sum_a(M_a + M_a^{\mathsf T})$.

<!-- excerpt: ferric-dft/src/vxc.rs::semilocal_vxc_closed_scratch -->
```rust
let e_xc: f64 = deterministic_point_sum(npts, |g| w[g] * dens.rho[g] * exc_total[g]);
```

<!-- excerpt: ferric-dft/src/vxc.rs::semilocal_vxc_closed_scratch -->
```rust
if above_density_floor(r, floor) {
    2.0 * w * v * gr
} else {
    0.0
}
```

| Symbol | Code |
|---|---|
| $w_g$, $\rho_g$, $\epsilon_g$ | `w`, `dens.rho`, `exc_total` |
| $v_\sigma$, $\partial_a\rho$ | `v`, `gr` |
| $f_a$ | the closure's value |

**Reproducibility.** Batch boundaries depend only on the coordinates;
inside a batch everything is serial; the batch contributions are summed in a
fixed grouping that depends on the problem shape, never on the thread
count. The result is bit-identical across thread counts. Setting the
screening threshold to zero keeps every shell and is the exactness anchor
of the screen.

## 9.3.4 Options

| Option | Effect |
|---|---|
| the functional family (section 9.1) | which of $v_\sigma$, $v_\tau$ are nonzero |
| `DensityFloor::for_xc` | $10^{-14}$, or $10^{-10}$ with a meta-GGA component; not user-set |
| resident or recompute AO batches | chosen from the memory budget; the same bits |

## 9.3.5 Numerical anchors

| Quantity | Value | Source |
|---|---|---|
| FD of $E_{xc}$ vs $\mathrm{Tr}[V_{xc}X]$, PBE, SCAN, ωB97X-V | $10^{-9}$ relative | `test_vxc_is_the_derivative_of_exc`, `crates/ferric-python/tests/test_xc_blocks.py` |
| polarized evaluation vs PySCF `eval_xc`, packing | bit-for-bit | `crates/ferric-dft/tests/mgga_polarized_packing.rs` |
| SCAN, r2SCAN energies vs PySCF | $\le 4.0\times10^{-12}$ Ha (RKS) | `crates/ferric-scf/tests/validation_mgga_energies.rs` |

## 9.3.6 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-dft/theory.md`, "Numerical Integration", "Semilocal $V_{xc}$ matrix elements" | the equations |
| `crates/ferric-dft/src/xc_batch.rs` module documentation | the batched design, its measurements and the determinism rules |
| Pople, Gill and Johnson, Chem. Phys. Lett. 199, 557 (1992) | the matrix-element form of the GGA potential |
