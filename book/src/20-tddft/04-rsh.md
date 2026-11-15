# 20.4 Range-separated hybrids

<!-- section: 20.4; status: ready; registry: tddft;
     theory: crates/ferric-tddft/theory.md#"8. Range-separated hybrids" -->

A range-separated hybrid puts exact-exchange fraction $c_{\rm sr}$ on
$\mathrm{erfc}(\omega r)/r$ and $c_{\rm lr}$ on $\mathrm{erf}(\omega r)/r$
(section 9.4). The response inherits that split. ferric writes it as a
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

## 20.4.1 The split

<!-- include: crates/ferric-tddft/theory.md#"8. Range-separated hybrids" -->

With $\mathrm{erfc} = 1 - \mathrm{erf}$,

$$ c_{\rm sr}(pq|\mathrm{erfc}|rs) + c_{\rm lr}(pq|\mathrm{erf}|rs)
= c_{\rm sr}(pq|rs) + \Delta c\,(pq|rs)_\omega , \tag{20.8} $$

so the exchange terms of equation 20.1 become
$c_{\rm sr}(ij|ab) + \Delta c\,(ij|ab)_\omega$ in $\mathbf A$ and
$c_{\rm sr}(ib|ja) + \Delta c\,(ib|ja)_\omega$ in $\mathbf B$. The Coulomb
and kernel terms do not change.

<!-- excerpt: ferric-tddft/src/tddft.rs::build_a_matrix -->
```rust
if mix.c_sr != 0.0 {
    subtract_exchange_a(&mut a, ri.tensors(), nocc, nvir, mix.c_sr);
}
if let (Some(lr), Some(dc)) = (ri.lr.as_deref(), mix.long_range()) {
    subtract_exchange_a(&mut a, lr.tensors(), nocc, nvir, dc);
}
```

The full-Coulomb exchange reuses the Coulomb-metric tensors. The
long-range term is fitted in the erf metric,
$B^{\omega,P}_{pq} = \sum_Q[\mathbf M_\omega]_{PQ}(Q|\mathrm{erf}|pq)$ with
$\mathbf M_\omega^{\mathsf T}\mathbf M_\omega = \mathbf V_\omega^+$, because
that is the variational fit for the erf operator and keeps the fitted block a
Gram matrix. $\mathbf V_\omega$ is close to singular for a Coulomb-fitted
auxiliary basis, so the pseudo-inverse drops eigenvalues below $10^{-10}$
rather than using a Cholesky factor.

Two limits are exact. At $\omega = 0$ no erf tensors are built and the run is
bitwise the global hybrid with $c_{\rm HF} = c_{\rm sr}$. At
$c_{\rm sr} = c_{\rm lr}$ the erf tensors are built but contracted with
$\Delta c = 0$, so the result is bitwise the global hybrid unless the long
range is weighted wrongly.

> **Verified equation 20.8** · the range-separated exchange split
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

## 20.4.2 Numerical anchors

From `site/src/reference/validation.md`, "Range-separated hybrid and triplet
TDA and Casida excitation energies".

| Quantity | Value | Test |
|---|---|---|
| lowest five roots vs PySCF `get_ab`, water, C2H4, benzene / 6-31G, flat 75 × 110 grid | $\le 4.4\times10^{-7}$ eV over all functionals | `crates/ferric-tddft/tests/validation_tddft_rsh.rs` |
| CAM-B3LYP, worst molecule | $2.9\times10^{-7}$ eV | the same |
| ωB97X, worst molecule | $6.7\times10^{-8}$ eV | the same |

## 20.4.3 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-tddft/theory.md`, section 8 | the split, the erf-metric fit, the exact limits and the matrix-free products |
| `crates/ferric-dft/theory.md`, "Exact exchange in hybrid and range-separated hybrid functionals" | the SCF exchange (section 9.4) |
