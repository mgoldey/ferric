# Range-separated hybrids {#rsh}

<!-- status: ready; registry: tddft -->

Consider an excitation that moves an electron from a donor at one end of a
molecule to an acceptor at the other. The transition density
$\phi_i\phi_a$ is close to zero everywhere, so the Coulomb term and the
local kernel of $A_{ia,ia}$ both vanish. What remains is the orbital gap and
the exchange term $-c\,(ii|aa)$, the attraction between the electron and
the hole it left. At long range that attraction is weighted by whatever
exact-exchange fraction the functional uses there. A range-separated hybrid
chooses that fraction separately: $c_{\rm sr}$ on
$\mathrm{erfc}(\omega r)/r$ and $c_{\rm lr}$ on $\mathrm{erf}(\omega r)/r$
(section @sec:hybrids). The response inherits the split. ferric writes it as a
full-Coulomb term plus a long-range correction, so only one extra set of
fitted integrals is needed.

| Symbol | Meaning | Code name |
|---|---|---|
| $(pq|rs)_\omega$ | erf-attenuated integral | from `ri.lr` |
| $\Delta c = c_{\rm lr} - c_{\rm sr}$ | long-range coefficient | `ExchangeMix::long_range` |
| $B^{\omega,P}_{pq}$ | erf-metric RI tensors | `build_lr_tensors` |

| Entry point | Location |
|---|---|
| Rust | `ferric-tddft/src/tddft.rs::{build_a_matrix,build_b_matrix,build_lr_tensors,ExchangeMix}`; the SCF-side analogue is `ferric-scf/src/rsh_response.rs::exchange_response` |
| CLI | `[tddft] xc = "cam-b3lyp"` (or any CAM functional) |

## The split {#rsh-split}

Split the Coulomb operator at $\omega$,
$1/r = \mathrm{erfc}(\omega r)/r + \mathrm{erf}(\omega r)/r$, and put exact
exchange fraction $c_{\rm sr}$ on the short-range part and $c_{\rm lr}$ on
the long-range part. In libxc these are `cam_alpha + cam_beta` and
`cam_alpha` (`ferric-dft/src/libxc.rs::CamCoeffs`). The exact-exchange
energy is

$$ E_x^{\rm HF} = -\tfrac12\sum_{\sigma}\sum_{ij\in\sigma}
\left[c_{\rm sr}(ij|{\rm erfc}|ji) + c_{\rm lr}(ij|{\rm erf}|ji)\right] . $$

The energy is a weighted sum of two exchange energies, one per operator,
so its second derivative is the same weighted sum of exchange terms: the
exchange of equation @eq:td-matrices-spin-adaptation with $c_{\rm HF}(pq|rs)$ replaced by
$c_{\rm sr}(pq|{\rm erfc}|rs) + c_{\rm lr}(pq|{\rm erf}|rs)$. Substitute
$\mathrm{erfc} = 1 - \mathrm{erf}$ and collect the erf terms:

$$ c_{\rm sr}(pq|\mathrm{erfc}|rs) + c_{\rm lr}(pq|\mathrm{erf}|rs)
= c_{\rm sr}(pq|rs) + \Delta c\,(pq|rs)_\omega , \label{eq:td-rsh-split} $$

with $(pq|rs)_\omega$ the erf-attenuated integral. So, for both spins,

$$ A_{ia,jb} \mathrel{-}= c_{\rm sr}(ij|ab) + \Delta c\,(ij|ab)_\omega,\qquad
B_{ia,jb} \mathrel{-}= c_{\rm sr}(ib|ja) + \Delta c\,(ib|ja)_\omega . $$

The Coulomb term $2(ia|jb)$ and the kernel do not change. The kernel is
libxc's for the same functional, whose semilocal exchange is already the
short-range-attenuated one.

Check the limits. With $c_{\rm sr} = c_{\rm lr}$, $\Delta c = 0$ and the
exchange is the global hybrid's. With $\omega \to 0$, $\mathrm{erf}(0) = 0$
and the long-range term vanishes, leaving the global hybrid with
$c_{\rm HF} = c_{\rm sr}$. For the charge-transfer case above, the long-range
electron–hole attraction is weighted by $c_{\rm sr} + \Delta c = c_{\rm lr}$.

`ferric-tddft/src/tddft.rs::build_a_matrix` applies the two exchange terms
in turn:

<!-- excerpt: ferric-tddft/src/tddft.rs::build_a_matrix -->
```rust
if mix.c_sr != 0.0 {
    subtract_exchange_a(&mut a, ri.tensors(), nocc, nvir, mix.c_sr);
}
if let (Some(lr), Some(dc)) = (ri.lr.as_deref(), mix.long_range()) {
    subtract_exchange_a(&mut a, lr.tensors(), nocc, nvir, dc);
}
```

The first call contracts the Coulomb-metric tensors with $c_{\rm sr}$, the
second the erf-metric tensors `lr` with $\Delta c$.
`ExchangeMix::long_range` returns $\Delta c$ only when $\omega \ne 0$, so a
global hybrid never enters the second branch.
`ferric-tddft/src/tddft.rs::build_b_matrix` does the same with
`subtract_exchange_b`, reading $(ib|ja)_\omega$ from
$B^{\omega\mathsf T}_{ov}B^\omega_{ov}$.

### Which integrals are fitted, and in which metric

The full-Coulomb exchange reuses the Coulomb-metric tensors
$B_{ov}, B_{oo}, B_{vv}$ of section @sec:response-matrices, which the Coulomb term needs
anyway. The long-range term takes a second set over the same auxiliary
basis,

$$ B^{\omega,P}_{pq} = \sum_Q[\mathbf M_\omega]_{PQ}(Q|\mathrm{erf}|pq),\qquad
\mathbf M_\omega^{\mathsf T}\mathbf M_\omega = \mathbf V_\omega^+,\qquad
[\mathbf V_\omega]_{PQ} = (P|\mathrm{erf}|Q), $$

with $\mathbf V_\omega^+$ the pseudo-inverse that drops eigenvalues below
$10^{-10}$ (`ferric-mp2/src/rimp2.rs::metric_inverse_sqrt`, by
eigendecomposition for the erf operator). Only the blocks the matrices use
are built: $oo$ and $vv$ for the exchange of $\mathbf A$, and $ov$ for the
exchange of $\mathbf B$ (Casida only).

Why the erf metric?

- It is the variational (Dunlap) fit for the erf operator. The error of
  $(pq|\mathrm{erf}|rs) \approx \sum_P B^{\omega,P}_{pq}B^{\omega,P}_{rs}$ is
  second order in the fitting residual, measured in the erf norm. The fitted
  block is a Gram matrix, so it stays symmetric positive semidefinite, as
  the exact one is.
- A Coulomb-metric fit of erf integrals has no symmetric $B^{\mathsf T}B$
  form. It needs
  $(pq|\mathrm{erf}|P)\,\mathbf V^{-1}(\ldots)\mathbf V^{-1}(Q|\mathrm{erf}|rs)$
  with the erf two-centre matrix in between, and neither the dense nor the
  matrix-free contraction is built for that.
- PySCF's `with_range_coulomb` density fitting and ferric's long-range SCF
  fitter (`build_rsh_dfk_pair`) use the same metric.

Why a pseudo-inverse in place of a Cholesky factor? $\mathbf V_\omega$ is
nearly singular for a Coulomb-fitted auxiliary basis. The erf kernel is
smooth, so tight auxiliary functions are nearly redundant under it.

The SCF and the response split the exchange differently. The SCF fits
$c_{\rm sr}K[\mathrm{erfc}] + c_{\rm lr}K[\mathrm{erf}]$ in the JK auxiliary
basis; the response fits $c_{\rm sr}K[1/r] + \Delta c\,K[\mathrm{erf}]$ in
the response auxiliary basis. For exact integrals the two are equal. They
differ by fitting error, of the same kind as the difference that already
exists between the JK-fitted SCF and the RI-fitted response of a global
hybrid.

### Limits that are exact in the code

- **$\omega = 0$.** The long-range term is identically zero. The driver
  builds no erf tensors, and the run is bitwise the global hybrid with
  $c_{\rm HF} = c_{\rm sr}$.
- **$c_{\rm sr} = c_{\rm lr} = c$ at any $\omega$.** Here $\Delta c = 0$.
  The erf tensors are still built and contracted with the coefficient 0, so
  the result is bitwise the global hybrid with $c_{\rm HF} = c$, unless the
  long-range term is weighted wrongly (by $c_{\rm lr}$ in place of
  $\Delta c$, for example). This limit is a test of the weighting.

### Matrix-free products and memory

Write $K^\omega_A(x)_{ia} = \sum_{jb}(ij|ab)_\omega x_{jb}$ and $K^\omega_B$
likewise. The products of `matvec.rs` gain $-\Delta c\,K^\omega_A$ in
$\mathbf Ax$, $-\Delta c\,(K^\omega_A + K^\omega_B)$ in
$(\mathbf A+\mathbf B)x$ and $-\Delta c\,(K^\omega_A - K^\omega_B)$ in
$(\mathbf A-\mathbf B)x$, contracted exactly as the Coulomb-metric exchange.
The second tensor set adds $N_{\rm aux}(n_o^2+n_v^2)$ resident doubles, and
$N_{\rm aux}n_on_v$ more for Casida; the memory plan declares them before
the erf tensors are built.

> **Verified equation @eq:td-rsh-split** · the range-separated exchange split
>
> $c_{\rm sr}\,\mathrm{erfc} + c_{\rm lr}\,\mathrm{erf} = c_{\rm sr} + (c_{\rm lr} - c_{\rm sr})\,\mathrm{erf}$,
> symbolically; numerically, the code's singlet and triplet blocks with RI
> tensors for Coulomb and erf equal the spin-adapted spin-orbital matrices
> built on the CAM kernel for a range-separated hybrid.
>
> | | |
> |---|---|
> | Proof | `scripts/tddft_response_sympy.py`, claims A4, B1 |
> | Negative control | the runner's B1 mutations; in the Rust validation, dropping the long-range term moves the roots by 2.9 to 6.0 eV |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-tddft/src/tddft.rs::{build_a_matrix,build_b_matrix}` |
> | Code test | `crates/ferric-tddft/tests/validation_tddft_rsh.rs` (CAM-B3LYP and ωB97X against PySCF `get_ab`) |
>
> <!-- verified: id=td.rsh.split; script=scripts/tddft_response_sympy.py; claim=A4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-tddft/src/tddft.rs::build_b_matrix; registry=tddft -->

## Numerical anchors {#rsh-numerical-anchors}

From `site/src/reference/validation.md`, "Range-separated hybrid and triplet
TDA and Casida excitation energies".

| Quantity | Value | Test |
|---|---|---|
| lowest five roots vs PySCF `get_ab`, water, C2H4, benzene / 6-31G, flat 75 × 110 grid | $\le 4.4\times10^{-7}$ eV over all functionals | `crates/ferric-tddft/tests/validation_tddft_rsh.rs` |
| CAM-B3LYP, worst molecule | $2.9\times10^{-7}$ eV | the same |
| ωB97X, worst molecule | $6.7\times10^{-8}$ eV | the same |

The negative control in the box puts these numbers in scale: dropping the
long-range term moves the roots by 2.9 to 6.0 eV.

## Sources {#rsh-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-tddft/theory.md`, section 8 | the split, the erf-metric fit, the exact limits and the matrix-free products |
| `crates/ferric-dft/theory.md`, "Exact exchange in hybrid and range-separated hybrid functionals" | the SCF exchange (section @sec:hybrids) |
