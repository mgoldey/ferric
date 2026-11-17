# $f_{xc}$ response kernels {#fxc}

<!-- status: ready; registry: ks-hessian, tddft -->

Perturb the density a little, by an electric field, by an orbital rotation
or by an excitation, and the exchange-correlation potential changes with
it. Every second-order KS method needs that change: the Newton and
stability orbital Hessians, the Z-vector of section @sec:ks-zvector, the CPKS
polarizability, the analytic Hessian and TDDFT. The change of $V_{xc}$ is
the second derivative of $E_{xc}$, and its pointwise ingredients are libxc's
second derivatives, collectively $f_{xc}$.

The plan: linearize the matrix of section @sec:vxc (@sec:fxc-linearized-potential), contract it with
occupied–virtual products for TDDFT (@sec:fxc-singlet-triplet-block), then list which family has a
kernel and which methods use it (@sec:fxc-coverage).

ferric has one kernel object per family. Each returns $\delta V^\sigma$ for a
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
| Consumers | `ferric-scf/src/rohf.rs::FxcKernelStore` (Newton, stability, Z-vector), `ferric-tddft/src/tddft.rs::run_tddft` (chapter @sec:tddft) |

## The linearized potential {#fxc-linearized-potential}

Start from the polarized matrix of section @sec:vxc-matrix-elements:

$$ V^\sigma_{\mu\nu} = \sum_g w_g\Bigl[v_{\rho_\sigma}\chi_\mu\chi_\nu + \mathbf G^\sigma\cdot\nabla(\chi_\mu\chi_\nu) + \tfrac12v_{\tau_\sigma}\nabla\chi_\mu\cdot\nabla\chi_\nu\Bigr] . $$

The AOs and the grid do not depend on the density. So a perturbation
$(\delta D^\alpha, \delta D^\beta)$ changes $V^\sigma$ only through the three
coefficients $v_{\rho_\sigma}$, $\mathbf G^\sigma$ and $v_{\tau_\sigma}$, and the
change is the same back-projection with their first-order changes:

$$ \delta V^\sigma_{\mu\nu} = \sum_g w_g\Big[u^\sigma\chi_\mu\chi_\nu + \mathbf w^\sigma\cdot\nabla(\chi_\mu\chi_\nu) + \tfrac12\, t^\sigma\,\nabla\chi_\mu\cdot\nabla\chi_\nu\Big], \qquad u^\sigma = \delta v_{\rho_\sigma},\quad \mathbf{w}^\sigma = \delta\mathbf{G}^\sigma,\quad t^\sigma = \delta v_{\tau_\sigma} . $$

Since $V = \partial E_{xc}/\partial D$, this is the second derivative of
$E_{xc}$ contracted with $\delta D$: the kernel is $f_{xc}$ in the AO basis.
What remains is to compute $u$, $\mathbf w$ and $t$ at each point.

**Step 1: the perturbed variables.** Take the seven variables
$x = (\rho_\alpha,\rho_\beta,\sigma_{\alpha\alpha},\sigma_{\alpha\beta},\sigma_{\beta\beta},\tau_\alpha,\tau_\beta)$.
$\rho_\sigma$ and $\tau_\sigma$ are linear in $D^\sigma$, so their changes are
the same formulas with $\delta D$:
$\delta\tau_\sigma = \tfrac12\sum_b\sum_{\mu\nu}\delta D^\sigma_{\mu\nu}\,\partial_b\chi_\mu\,\partial_b\chi_\nu$,
and likewise $\delta\rho_\sigma$ and $\nabla\delta\rho_\sigma$. The $\sigma$
variables are quadratic in the gradients; the product rule gives

$$ \delta\sigma_{\alpha\alpha} = 2\nabla\rho_\alpha\cdot\nabla\delta\rho_\alpha, \qquad \delta\sigma_{\alpha\beta} = \nabla\rho_\alpha\cdot\nabla\delta\rho_\beta + \nabla\rho_\beta\cdot\nabla\delta\rho_\alpha , $$

and $\delta\sigma_{\beta\beta}$ like $\delta\sigma_{\alpha\alpha}$.

**Step 2: the perturbed first derivatives.** Each first derivative
$v_{x_i}$ is a function of all seven variables, so to first order

$$ \delta v_{x_i} = \sum_j f_{x_i x_j}\,\delta x_j . $$

For a GGA, written out:

$$ \delta v_{\rho_\alpha} = f_{\rho_\alpha\rho_\alpha}\delta\rho_\alpha + f_{\rho_\alpha\rho_\beta}\delta\rho_\beta + \sum_c f_{\rho_\alpha\sigma_c}\delta\sigma_c, \qquad \delta v_{\sigma_c} = \sum_{\sigma'} f_{\sigma_c\rho_{\sigma'}}\delta\rho_{\sigma'} + \sum_{c'} f_{\sigma_c\sigma_{c'}}\delta\sigma_{c'} . $$

**Step 3: the perturbed gradient coefficient.**
$\mathbf G^\alpha = 2v_{\sigma_{\alpha\alpha}}\nabla\rho_\alpha + v_{\sigma_{\alpha\beta}}\nabla\rho_\beta$
is a product of a coefficient and a gradient, and both change. The product
rule gives four terms:

$$ \mathbf{w}^\alpha = 2\,\delta v_{\sigma_{\alpha\alpha}}\nabla\rho_\alpha + \delta v_{\sigma_{\alpha\beta}}\nabla\rho_\beta + 2 v_{\sigma_{\alpha\alpha}}\nabla\delta\rho_\alpha + v_{\sigma_{\alpha\beta}}\nabla\delta\rho_\beta . $$

The $\tau$ term has no analogue of the last two: the $\tau$ part of $V$ is
$\tfrac12v_\tau\nabla\chi_\mu\cdot\nabla\chi_\nu$, which depends on $D$ only
through $v_\tau$. So $t^\sigma = \delta v_{\tau_\sigma}$ and $v_\tau$ itself does
not appear in the response.

**Limits.** An LDA has no $\sigma$ or $\tau$: the kernel
(`LdaFxcKernel::apply_with_ref`) is
$u^\alpha = f_{\alpha\alpha}\delta\rho_\alpha + f_{\alpha\beta}\delta\rho_\beta$,
$\mathbf w = 0$. A GGA is the case $t = 0$. The tests
`gga_kernel_matches_lda_kernel_on_lda_functional` and
`mgga_kernel_on_gga_functional_equals_gga_kernel` check that each kernel
reduces to the one below it.

**In the code.** `ferric-dft/src/fxc.rs::linearize_point` is Steps 1 to 3 at
one point. After the destructuring of its arguments, the function starts
with Steps 1 and 2:

<!-- excerpt: ferric-dft/src/fxc.rs::linearize_point -->
```rust
let mut dx = [0.0; NVAR];
dx[RA] = p.drho[0];
dx[RB] = p.drho[1];
dx[SAA] = 2.0 * dot3(ga, da);
dx[SAB] = dot3(ga, db) + dot3(gb, da);
dx[SBB] = 2.0 * dot3(gb, db);
dx[TA] = p.dtau[0];
dx[TB] = p.dtau[1];
let mut dv = [0.0; NVAR];
for (i, row) in h.iter().enumerate() {
    dv[i] = row.iter().zip(&dx).map(|(a, b)| a * b).sum();
}
```

then, per axis, the four terms of $\mathbf w$:

<!-- excerpt: ferric-dft/src/fxc.rs::linearize_point -->
```rust
w[0][ax] =
    2.0 * dv[SAA] * ga[ax] + dv[SAB] * gb[ax] + 2.0 * vs[0] * da[ax] + vs[1] * db[ax];
```

| Symbol | Code |
|---|---|
| $\nabla\rho_\alpha$, $\nabla\rho_\beta$, $\nabla\delta\rho_\alpha$, $\nabla\delta\rho_\beta$ | `ga`, `gb`, `da`, `db` |
| $\delta x$ (Step 1) | `dx`, indexed by `RA, RB, SAA, SAB, SBB, TA, TB` $= 0..6$ |
| $f_{x_ix_j}$, $\delta v$ (Step 2) | `h`, `dv` |
| $v_{\sigma_{\alpha\alpha}}$, $v_{\sigma_{\alpha\beta}}$ | `vs[0]`, `vs[1]` |
| $u^\alpha$, $t^\alpha$ | `dv[RA]`, `dv[TA]` (returned) |

The function returns $([u^\alpha, u^\beta], \mathbf w, [t^\alpha, t^\beta])$.
The per-point Hessian `h` is a dense symmetric $7\times7$ array, so one loop
serves LDA, GGA and meta-GGA: the rows and columns a family does not have
are zero.

`ferric-dft/src/fxc.rs::MggaFxcKernel::backproject` then turns the
per-point $(u, \mathbf w, t)$ into the AO matrix
$\delta V^\sigma_{\mu\nu} = \sum_g w_g\big[u^\sigma\chi_\mu\chi_\nu
+ \mathbf w^\sigma\cdot\nabla(\chi_\mu\chi_\nu)
+ \tfrac12 t^\sigma\nabla\chi_\mu\cdot\nabla\chi_\nu\big]$ with one GEMM per
term: the $u$ term against $\chi$, per axis the $\mathbf w$ term (added with
its transpose) and the $\tfrac12 t$ term against $\partial_b\chi$, and a
final symmetrization $\tfrac12(V + V^T)$.

**libxc's packing.** libxc returns the second derivatives packed, and the
code reads them in this order:

- `v2rho2` $=[\alpha\alpha,\alpha\beta,\beta\beta]$;
- `v2rhosigma` $=[\alpha\cdot\alpha\alpha,\alpha\cdot\alpha\beta,\alpha\cdot\beta\beta,\beta\cdot\alpha\alpha,\beta\cdot\alpha\beta,\beta\cdot\beta\beta]$;
- `v2sigma2`, the upper triangle,
  $[\alpha\alpha\cdot\alpha\alpha,\alpha\alpha\cdot\alpha\beta,\alpha\alpha\cdot\beta\beta,\alpha\beta\cdot\alpha\beta,\alpha\beta\cdot\beta\beta,\beta\beta\cdot\beta\beta]$;
- for the $\tau$-only meta-GGA kernel (`MggaFxcKernel::apply_with_ref`,
  per-point Hessians from `point_hessians`): `v2rhotau`
  $=[\rho_\alpha\tau_\alpha,\rho_\alpha\tau_\beta,\rho_\beta\tau_\alpha,\rho_\beta\tau_\beta]$,
  `v2sigmatau`
  $=[\sigma_{\alpha\alpha}\tau_\alpha,\sigma_{\alpha\alpha}\tau_\beta,\sigma_{\alpha\beta}\tau_\alpha,\dots,\sigma_{\beta\beta}\tau_\beta]$
  and `v2tau2` $=[\alpha\alpha,\alpha\beta,\beta\beta]$.

`ferric-dft/src/fxc.rs::scatter_tau` writes the $\tau$ blocks into the
$7\times7$ Hessian:

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

The index arithmetic is the packing: $\tau$ runs fastest, so the entry for
$(\rho_s, \tau_t)$ sits at $2s + t$ and the entry for $(\sigma_c, \tau_t)$ at
$2c + t$. `add_sym` fills both $(i, j)$ and $(j, i)$. A wrong packing is the
first negative control of the box below.

> **Verified equation {#eq:ks-fxc-linearization}** · the GGA and meta-GGA kernels are the linearization of $V_{xc}$
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

The meta-GGA check's residual is $6\times10^{-18}$. In Rust,
`fxc::tests::gga_fxc_matches_finite_difference_of_vxc` checks the GGA kernel
against finite differences of $V_{xc}$ on the OH doublet in all spin
channels, and `mgga_fxc_matches_finite_difference_of_production_vxc_polarized`
and `_closed` do the same for the meta-GGA kernel.
`crates/ferric-scf/tests/uhf_newton_smoke.rs::uks_pbe_newton_engages_gga_fxc_and_matches_diis`
runs the kernel end to end: UKS/PBE Newton reaches the DIIS energy.

**The density floor.** The kernels zero every term at points where the
reference $\rho_\sigma$ is at or below the functional's `DensityFloor`, as
$V^\sigma$ does (section @sec:ks-equations-scf-iteration). The linearization is therefore of the
matrix the SCF used. None of these kernels contains the VV10 response; that
is `ferric-dft/src/vv10.rs::vv10_potential_tangent`.

## The singlet and triplet $ov$ block {#fxc-singlet-triplet-block}

TDDFT needs the kernel contracted with occupied–virtual products. For one
column $jb$, `ferric-dft/src/lr_kernel.rs::fxc_ov_block` forms the symmetric
AO perturbation
$\delta D = \tfrac12(\mathbf c_j\mathbf c_b^{\mathsf T} + \mathbf c_b\mathbf c_j^{\mathsf T})$,
applies the kernel with $\delta D^\beta = +\delta D$ (singlet) or
$-\delta D$ (triplet), and projects the $\alpha$ response:

$$ K_{ia,jb} = \mathbf c_i^{\mathsf T}\,\delta V^\alpha[\delta D, \pm\delta D]\,\mathbf c_a = (ia\vert f_{\alpha\alpha} \pm f_{\alpha\beta}\vert jb) . \label{eq:ks-fxc-ov-block} $$

The second form follows from Step 2 for an LDA. With
$\delta\rho_\alpha = \phi_j\phi_b$ and $\delta\rho_\beta = \pm\delta\rho_\alpha$,
$u^\alpha = (f_{\alpha\alpha} \pm f_{\alpha\beta})\phi_j\phi_b$, and projecting
onto $\phi_i\phi_a$ gives the integral. A singlet excitation moves both
spins together; a triplet moves them in opposite directions.

Check the spin limit with exchange alone. The LDA exchange energy density
separates by spin, $e_x(\rho_\alpha, \rho_\beta) = \tfrac12\bigl(e_x(2\rho_\alpha) + e_x(2\rho_\beta)\bigr)$,
so its $f_{\alpha\beta} = 0$ and the singlet and triplet exchange kernels are
equal. For an LDA, the difference $2f_{\alpha\beta}$ between the two kernels
comes from correlation alone.

<!-- excerpt: ferric-dft/src/lr_kernel.rs::fxc_ov_block -->
```rust
let (dv_a, _dv_b) = match spin {
    ResponseSpin::Singlet => kernel.apply(&dd, &dd),
    ResponseSpin::Triplet => kernel.apply(&dd, &(-&dd)),
};
// K[:, jb] = C_occ^T δV_α C_vir, flattened as ia = i*nvir + a.
let block = orbo.t().dot(&dv_a).dot(&orbv); // (nocc, nvir)
```

Two points about the function:

- **$\delta D$ is symmetrized explicitly.** The raw outer product
  $\mathbf c_j\mathbf c_b^{\mathsf T}$ is fine for an LDA and wrong for a GGA:
  the gradient terms see its antisymmetric part. Measured on water/STO-3G/PBE,
  the raw product gave a relative asymmetry of the block of
  $3.1\times10^{-2}$; symmetrizing brings PBE excitation energies to
  $1.0\times10^{-3}$ eV of PySCF `tddft.TDA`.
- **The block is checked before it is symmetrized.** A relative asymmetry
  above `FXC_BLOCK_ASYMMETRY_TOL` $= 10^{-8}$ is an error, since it means
  the adapter, not the grid, is wrong. `SingletOvFxc` applies the same
  kernel matrix-free for the Davidson solver.

> **Verified equation @eq:ks-fxc-ov-block** · singlet kernel normalization and spin adaptation
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

## Coverage {#fxc-coverage}

| Family | Kernel | Consumers | Refused |
|---|---|---|---|
| LDA | `LdaFxcKernel` | all | |
| GGA, global hybrid, RSH | `GgaFxcKernel` (exact exchange through $c_{\rm sr}, c_{\rm lr}$ in the caller) | all | |
| $\tau$-only meta-GGA | `MggaFxcKernel` | RKS/UKS Newton, RKS/UKS stability, RKS/UKS/ROKS Z-vector, TDA and Casida | ROKS Newton and augmented Hessian, TRAH, the analytic nuclear Hessian |
| VV10 | none | | every response method refuses a VV10 functional |

`ferric-scf/src/rohf.rs::FxcKernelStore` dispatches LDA, GGA or meta-GGA for
the orbital-Hessian users: RKS and UKS Newton steps, RKS/UKS internal
stability, and the RKS/UKS/ROKS Z-vector (`zvector_ks`).
`ferric-dft/src/lr_kernel.rs::SingletKernel` does the same for the singlet
$(ia|f_{\alpha\alpha}+f_{\alpha\beta}|jb)$ block of TDA and Casida TDDFT
(`ferric_tddft::run_tddft`, `ferric_gw::tddft::run_tda_dft`).

The meta-GGA refusals have two reasons. ROKS Newton, the augmented Hessian
and TRAH keep DIIS, because no reference anchors them. The analytic nuclear
Hessian is refused because the $\tau$-dependent nuclear derivatives of
$V_{xc}$ are not implemented.

The meta-GGA consumers are validated with r2SCAN in 6-31G on a matched
$(75, 110)$ grid, with references from `scripts/validation/gen_mgga_response.py`
and `gen_tddft_refs.py`:

- `crates/ferric-scf/tests/mgga_response_vs_pyscf.rs`: the dense
  orbital-Hessian spectra and stability $\lambda_{\min}$ of UKS NH2 and RKS
  water against PySCF's `gen_g_hop_uhf` and `gen_g_hop_rhf`, and the RKS/UKS
  Newton energy against DIIS;
- `crates/ferric-rpa/tests/mbd_scf_gradient{,_uks,_roks}.rs::*_r2scan`: the
  Z-vector, through the MBD@rsSCS gradient against finite differences of the
  full SCF pipeline;
- `crates/ferric-tddft/tests/validation_tddft.rs`, case `water_631g_r2scan`: TDA and
  Casida against PySCF's `get_ab`.

## Numerical anchors {#fxc-numerical-anchors}

From `site/src/reference/validation.md`, "Meta-GGA (τ-only) $f_{xc}$ response".

| Quantity | Value | Test |
|---|---|---|
| meta-GGA kernel vs FD of the production $V_{xc}$ | $5.4\times10^{-9}$ (r2SCAN), $3.6\times10^{-7}$ (SCAN) relative | `crates/ferric-dft/src/fxc.rs` |
| RKS/UKS stability $\lambda_{\min}$, r2SCAN, vs PySCF dense orbital Hessian | $\le 1.5\times10^{-10}$ Ha | `crates/ferric-scf/tests/mgga_response_vs_pyscf.rs` |
| Newton vs DIIS energy | $\le 1.4\times10^{-14}$ Ha | the same |
| Z-vector exact gradient vs FD (MBD@rsSCS) | $8.0\times10^{-10}$ (RKS), $1.1\times10^{-11}$ (UKS), $2.0\times10^{-11}$ (ROKS) Ha/Bohr | `crates/ferric-rpa/tests/mbd_scf_gradient.rs` and its UKS and ROKS siblings |

## Sources {#fxc-sources}

| Source | What it gives |
|---|---|
| section @sec:ks-hessian-xc-terms | the nuclear-Hessian use of the same kernel |
| section @sec:response-matrices | the TDDFT use |
