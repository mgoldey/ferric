# 19.3 The correlation self-energy

<!-- section: 19.3; status: ready; registry: gw;
     theory: crates/ferric-gw/theory.md#"4. Correlation self-energy on the imaginary axis" -->

The correlation self-energy of state $p$ is the frequency convolution of the
Green's function with the dynamic screened interaction. Evaluated at a complex
energy $z$ off the real axis, the convolution is a smooth integral over the
imaginary-frequency grid on which $W$ is stored; that is why GW here works on
the imaginary axis and continues to the real one afterwards (section 19.5).

| Symbol | Meaning | Code name |
|---|---|---|
| $z$ | complex energy, absolute (not measured from $\epsilon_F$) | `z` |
| $u_k$, $w_k$ | quadrature node and weight of the $W$ grid | `quad_freqs`, `quad_weights` |
| $W_{pq}(iu_k)$ | screened matrix element, section 19.2 | `w_nk` |
| $\epsilon_q$ | propagator energy of state $q$ | `eps_act` (a separate argument, so evGW can replace it) |
| $\Sigma^c_p(z)$ | correlation self-energy | the return of `sigma_c_at_z` |

| Entry point | Location |
|---|---|
| Rust | `ferric-gw/src/sigma.rs::{sigma_c_at_z,sigma_c_at_z_product,run_g0w0}` |
| CLI | `method.kind = "gw"`, `[gw] method = "g0w0"` |
| Python | `run_gw(method="g0w0")` |

## 19.3.1 The frequency integral

<!-- include: crates/ferric-gw/theory.md#"4. Correlation self-energy on the imaginary axis" -->

The kernel sum over $q$ is the whole active spectrum, with no occupied/virtual
distinction in the code: the time ordering sits in the sign of
$\mathrm{Re}(z-\epsilon_q)$, which flips the sign of the kernel
$a/(a^2+u'^2)$, $a = z-\epsilon_q$. The plain rule is one loop per state:

<!-- excerpt: ferric-gw/src/sigma.rs::sigma_c_at_z -->
```rust
    let mut sigma = Complex64::new(0.0, 0.0);
    for n_idx in 0..n_act {
        let eps_n = eps_act[n_idx];
        let diff = z - Complex64::new(eps_n, 0.0);
        let mut inner = Complex64::new(0.0, 0.0);
        for k in 0..n_quad {
            let omk = quad_freqs[k];
            let den = diff * diff + Complex64::new(omk * omk, 0.0);
            inner += Complex64::new(quad_weights[k] * w_nk[(n_idx, k)], 0.0) / den;
        }
        sigma += diff * inner;
    }
    -Complex64::new(1.0 / std::f64::consts::PI, 0.0) * sigma
```

| Symbol | Variable |
|---|---|
| $z - \epsilon_q$ | `diff` |
| $u_k$ | `omk` |
| $(z-\epsilon_q)^2 + u_k^2$ | `den` |
| $w_kW_{pq}(iu_k)$ | `quad_weights[k] * w_nk[(n_idx, k)]` |
| $-1/\pi$ | the last line |

> **Verified equation 19.4** · the frequency integral against the closed-form residue result
>
> For a single-pole $W(iu) = -2\Omega s/(u^2+\Omega^2)$,
> $\Sigma^c_p(z) = \sum_{q:\,\mathrm{Re}(z-\epsilon_q)>0}\frac{s_{pq}}{z-\epsilon_q+\Omega} + \sum_{q:\,\mathrm{Re}(z-\epsilon_q)<0}\frac{s_{pq}}{z-\epsilon_q-\Omega}$:
> occupied poles at $\epsilon_i - \Omega$, virtual poles at $\epsilon_a + \Omega$.
> Proven symbolically for real arguments and checked at complex $z$ by
> 40-digit quadrature of the definition.
>
> | | |
> |---|---|
> | Proof | `scripts/gw_self_energy_sympy.py`, claim 2 |
> | Negative control | the claim is an equality of two independent evaluations, the quadrature of the definition and the residue sum |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-gw/src/sigma.rs::sigma_c_at_z` |
> | Code test | `sigma_c_single_pole_w_matches_closed_form_residues` in `crates/ferric-gw/src/sigma.rs` |
>
> <!-- verified: id=gw.sigma-c.closed-form; script=scripts/gw_self_energy_sympy.py; claim=2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-gw/src/sigma.rs::sigma_c_at_z; rust_test=ferric-gw/src/sigma.rs::sigma_c_single_pole_w_matches_closed_form_residues; registry=gw -->

## 19.3.2 Product integration

The plain Gauss–Legendre sum cannot resolve the ridge of the kernel at
$u' = u$ when a state lies close to the continuation line, and its error does
not decrease with the number of nodes for states away from the frontier.
Every driver therefore uses `sigma_c_at_z_product` whenever the $W$ grid is a
scaled Gauss–Legendre rule: with $u' = u_0(1+x)/(1-x)$ the kernel becomes
exactly $\tfrac i2[1/(x-x_1) - 1/(x-x_2)]\,dx$, and each Cauchy integral is done
exactly for the degree-$(n-1)$ interpolant of $W_{pq}$ through the nodes. The
only error left is $W$'s interpolation error. The text of section 19.3.1
(the last paragraph of the included derivation) records the measured effect;
the rule is detected from the nodes (`ProductQuadrature::from_scaled_gauss_legendre`)
and a non-Gauss–Legendre grid keeps the plain sum.

> **Verified equation 19.5** · the kernel identity and the product rule
>
> The kernel identity $a/(a^2+u'^2)\,du' = \tfrac i2[1/(x-x_1)-1/(x-x_2)]\,dx$
> holds for $u' = u_0(1+x)/(1-x)$, and a port of the product rule with its
> Cauchy integral reproduces the closed form of box 19.4 for a single-pole $W$
> at 40 nodes, including a state 0.015 Ha from $\epsilon_F$.
>
> | | |
> |---|---|
> | Proof | `scripts/gw_self_energy_sympy.py`, claim 2b |
> | Negative control | the runner replaces the minus between the two Cauchy integrals by a plus |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-gw/src/sigma.rs::sigma_c_at_z_product` |
> | Code test | `product_rule_converges_where_the_plain_quadrature_cannot` in `crates/ferric-gw/src/sigma.rs` |
>
> <!-- verified: id=gw.sigma-c.product-rule; script=scripts/gw_self_energy_sympy.py; claim=2b; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-gw/src/sigma.rs::sigma_c_at_z_product; rust_test=ferric-gw/src/sigma.rs::product_rule_converges_where_the_plain_quadrature_cannot; registry=gw -->

## 19.3.3 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | $B$, $\Sigma^x$, $B^\xi$, $\tilde W_d(iu_k)$ | sections 19.1 and 19.2 | `run_gw` |
| 2 | the Fermi level, the mid-gap $\epsilon_F$ of the active spectrum | $O(1)$ | `fermi_level` |
| 3 | per state $p$ (independent, parallel over states): $W_{pq}(iu_k)$ for all $q$, $k$ | $n_{\rm quad}$ GEMMs | `screened_elements` |
| 4 | $\Sigma^c_p$ at each continuation node $z = \epsilon_F + iu$ (section 19.5) | $n_{\rm nodes}\,n_{\rm act}\,n_{\rm quad}$ | `sigma_c_at_z_product` |
| 5 | continuation and the quasiparticle root (section 19.5) | $O(n_{\rm nodes})$ per Newton step | `solve_qp_for_mo` |

The per-state sweep takes a soft scratch charge from the memory pool
(`charge_qp_sweep_scratch`); when the pool refuses it, the sweep runs
serially instead. GEMM and the Newton work are scalar-sized, so the sweep is
deterministic in the thread count up to the last bits of BLAS summation order.

## 19.3.4 Numerical anchors

From `site/src/reference/validation.md` (the `gw` row and the G0W0 rows of its
anchors table), at matched settings `n_quad = 100` and `trunc_thresh = 0`:

| Quantity | Reference | Error | Test |
|---|---|---|---|
| $\Sigma^c(\epsilon_F + iu)$ at the Padé nodes, plain quadrature | PySCF `gw_ac` | $\le 8.6\times10^{-11}$ Ha | `sigma_c_imag_axis_h2o_pbe_and_hf_vs_pyscf` in `crates/ferric-gw/tests/validation_gw.rs` |
| G0W0@HF QP energies | grid-free dRPA pole sum | $\le 1.2\times10^{-8}$ Ha | `crates/ferric-gw/tests/validation_gw.rs` |
| G0W0@PBE QP energies | grid-free dRPA pole sum | $\le 1.8\times10^{-8}$ Ha | `crates/ferric-gw/tests/validation_gw.rs` |
| G0W0@HF QP energies | PySCF `gw_ac` | $\le 1.6\times10^{-7}$ Ha | `crates/ferric-gw/tests/validation_gw.rs` |

The grade is Proven (narrow). Narrow means: states far from $\epsilon_F$ with
split spectral weight are limited by the analytic continuation (section
19.5), and the defaults (`n_quad = 40`, `trunc_thresh = 1e-4`) are coarser
than the matched settings, within 10 meV for HOMO and LUMO.

## 19.3.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-gw/theory.md`, "Correlation self-energy on the imaginary axis" | the integral, the closed form, the product rule |
| `scripts/gw_self_energy_sympy.py`, claims 2 and 2b | the proofs |
| Ren, Rinke, Blum *et al.*, *New J. Phys.* **14**, 053020 (2012) | RI GW on the imaginary axis |
