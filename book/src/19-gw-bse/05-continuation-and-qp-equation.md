# 19.5 Analytic continuation and the quasiparticle equation

<!-- section: 19.5; status: ready; registry: gw;
     theory: crates/ferric-gw/theory.md#"6. Analytic continuation" -->

Section 19.3 evaluates $\Sigma^c$ at complex energies, and the quasiparticle
energy is a root of an equation on the real axis. The bridge is analytic
continuation: sample $\Sigma^c$ on the imaginary line $z_k = \epsilon_F + iu_k$,
fit a rational function through the samples, and evaluate that function (and
its slope) at real energies. A Newton iteration on it then finds the root.

| Symbol | Meaning | Code name |
|---|---|---|
| $\epsilon_F$ | mid-gap of the active mean-field spectrum | `ef`, `fermi_level` |
| $z_k$ | continuation nodes, 18 by default | `pade_support_omegas` |
| $C_N(z)$ | the Thiele continued fraction | `PadeCF` |
| $\delta_p$ | static shift, $0$ (HF) or $\Sigma^x_p - v^{xc}_p$ (KS) | `static_shift` |
| $Z_p$ | renormalization factor | `z_renorm` |

| Entry point | Location |
|---|---|
| Rust | `ferric-gw/src/pade.rs::{fit,eval,deriv_real}`, `ferric-gw/src/sigma.rs::{solve_qp_for_mo,solve_qp_for_mo_with,qp_root_by_weight}` |
| Options | `[gw] pade_npts`, `qp_newton_damp`, `qp_mos` |

## 19.5.1 The continuation

<!-- include: crates/ferric-gw/theory.md#"6. Analytic continuation" -->

> **Verified equation 19.7** · the Thiele continued fraction, in exact arithmetic
>
> A line-for-line port of `PadeCF::fit` and `eval` in exact rationals
> interpolates its nodes; it reproduces a rational function of type $[L/M]$
> identically in $z$ once the node count reaches $L+M+1$ (a single-pole
> $\Sigma^c$ of two states, type $[1/2]$, from four nodes); one surplus node
> gives a zero coefficient and a second makes the next denominator exactly
> zero, which `fit` refuses with an error rather than a NaN.
>
> | | |
> |---|---|
> | Proof | `scripts/gw_self_energy_sympy.py`, claim 6 |
> | Negative control | three nodes, one short of the type, must not reproduce the function |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-gw/src/pade.rs::{fit,eval}` |
> | Code test | `interpolation_identity` in `crates/ferric-gw/src/pade.rs` |
>
> <!-- verified: id=gw.pade.thiele; script=scripts/gw_self_energy_sympy.py; claim=6; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-gw/src/pade.rs::{fit,eval}; rust_test=ferric-gw/src/pade.rs::interpolation_identity; registry=gw -->

> **Verified equation 19.8** · the four-point derivative stencil
>
> The central stencil of `deriv_real` with $h = 0.05$ Ha is exact through
> degree 4 and has error $-h^4f^{(5)}/30$.
>
> | | |
> |---|---|
> | Proof | `scripts/gw_self_energy_sympy.py`, claim 5 |
> | Negative control | the degree-5 monomial must show the stated error, not zero |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-gw/src/pade.rs::deriv_real` |
>
> <!-- verified: id=gw.pade.stencil; script=scripts/gw_self_energy_sympy.py; claim=5; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-gw/src/pade.rs::deriv_real; registry=gw -->

The nodes are chosen by `pade_node_indices`, a port of PySCF's `_get_ac_idx`,
from $\{0\}\cup\mathrm{GL}(100, u_0 = 0.5)$ with a decreasing step of ratio
$2/3$, and are independent of the grid on which $W$ was integrated. The
`Err` on a repeated support node is covered by
`fit_errors_on_repeated_support_node` in `crates/ferric-gw/src/pade.rs`.

## 19.5.2 The quasiparticle equation

<!-- include: crates/ferric-gw/theory.md#"7. Quasiparticle equation, Newton solve, renormalization" -->

The Newton loop, with the symbols of the equation above:

<!-- excerpt: ferric-gw/src/sigma.rs::solve_qp_for_mo_with -->
```rust
    for _ in 0..30 {
        let sc = pade.eval(Complex64::new(eps_curr, 0.0)).re;
        // QP residual: ε − ε_mf − static_shift − Σc(ε) = 0.
        let f = eps_curr - eps_m_mf - static_shift - sc;
        let dsc = pade.deriv_real(eps_curr, h).re;
        let fprime = 1.0 - dsc;
        if fprime.abs() < 1e-3 {
            break;
        }
        let step = -damp * f / fprime;
        eps_curr += step;
        if step.abs() < 1e-7 {
            newton_converged = true;
            break;
        }
    }
```

| Symbol | Variable |
|---|---|
| $E_k$ | `eps_curr` |
| $\epsilon_p$ | `eps_m_mf` |
| $\delta_p$ | `static_shift` |
| $\mathrm{Re}\,C_N(E_k)$ | `sc` |
| $f(E_k)$, $f'(E_k)$ | `f`, `fprime` |
| $d$ | `damp`, from `qp_newton_damp` clamped to $[0.1, 1]$ |

> **Verified equation 19.9** · the linearized start and the Newton iteration
>
> The linearized quasiparticle energy $\epsilon + Z(\Sigma(\epsilon)+\delta)$,
> with $Z = 1/(1-\Sigma'(\epsilon))$, is exactly one undamped Newton step of
> $f$ from $\epsilon$; $Z$ at a root is the residue of the quasiparticle pole
> of $G(E) = [E-\epsilon-\delta-\Sigma(E)]^{-1}$; and Newton on a
> single-pole $\Sigma$ converges quadratically to the exact root.
>
> | | |
> |---|---|
> | Proof | `scripts/gw_self_energy_sympy.py`, claim 4 |
> | Negative control | none separate: Newton is compared with the roots of an independent symbolic solve, and the quadratic-convergence ratio is bounded |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-gw/src/sigma.rs::solve_qp_for_mo_with` |
> | Code test | `qp_newton_converges_for_a_mild_self_energy` in `crates/ferric-gw/src/sigma.rs` |
>
> <!-- verified: id=gw.qp-equation; script=scripts/gw_self_energy_sympy.py; claim=4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-gw/src/sigma.rs::solve_qp_for_mo_with; rust_test=ferric-gw/src/sigma.rs::qp_newton_converges_for_a_mild_self_energy; registry=gw -->

## 19.5.3 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | $\epsilon_F$ and the 18 node frequencies | $O(1)$ | `fermi_level`, `pade_support_omegas` |
| 2 | $\Sigma^c(\epsilon_F + iu_k)$ at the nodes, product-integrated | section 19.3 | `sigma_c_at_z_product` |
| 3 | Thiele fit | $O(N^2)$, $N$ = nodes | `PadeCF::fit` |
| 4 | $Z_p$ at $\epsilon_p$, clamped to $[0, 1.5]$; linearized start | $O(N)$ | `solve_qp_for_mo_with` |
| 5 | Newton, at most 30 steps, stops on $\lvert\text{step}\rvert < 10^{-7}$ Ha or $\lvert f'\rvert < 10^{-3}$ | $O(N)$ per step | `solve_qp_for_mo_with` |
| 6 | if Newton did not converge: the root of largest weight $Z\in(0,1]$ in $\epsilon_p+\delta_p\pm1$ Ha (4001 samples) | $4001\,O(N)$ | `qp_root_by_weight` |

The static shift sits inside the equation, so $\Sigma^c$ is evaluated at the
shifted root. Adding it after an unshifted solve lands 0.38 to 0.96 eV away on
the open-shell @PBE cases, which `crates/ferric-gw/tests/validation_gw.rs`
requires ferric to miss. A state with no root of weight in $(0, 1]$ is left
unsolved: `GwResult.qp_converged` records it, and `run_gw` warns
(`warn_gw_unconverged`).

## 19.5.4 Options and limits

| Option | Effect |
|---|---|
| `pade_npts` | number of continuation nodes (default 18) |
| `qp_newton_damp` | damping of the Newton step, clamped to $[0.1, 1]$ |
| `qp_mos` | absolute MO range for which the QP energy is solved; default HOMO−2 … LUMO+2 (`default_qp_range`) |

**Known limits (2026-11-15).** The physical $\Sigma^c$ is a rational function
of degree about $n_{\rm act}\times n_{\rm quad}$, so 18 nodes give an
approximation, not a reconstruction. The error is largest for KS references,
whose $\epsilon_{\rm HOMO}$ lies far from $\epsilon_F$, and for states far from
$\epsilon_F$ with split spectral weight; this is what "narrow" means in the
grade. The unconverged-state warning is the signal to look at.

## 19.5.5 Numerical anchors

From `site/src/reference/validation.md` (`gw` row and G0W0 anchors), at
`n_quad = 100`, `trunc_thresh = 0`:

| Quantity | Reference | Error | Test |
|---|---|---|---|
| G0W0@HF QP energies, HOMO−2 … LUMO+2 | PySCF `gw_ac` | $\le 1.6\times10^{-7}$ Ha | `g0w0_hf_{h2o,nh3,n2}_vs_pyscf_gw_ac` in `crates/ferric-gw/tests/validation_gw.rs` |
| G0W0@HF with ECPs | grid-free dRPA pole sum | $\le 8.0\times10^{-10}$ Ha | `g0w0_hf_ecp_{i2,xe,ag2}_vs_pyscf_gw_ac` |
| G0W0@PBE | grid-free dRPA pole sum | $\le 1.8\times10^{-8}$ Ha | `g0w0_pbe_{h2o,n2}_vs_pyscf_gw_ac` |

## 19.5.6 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-gw/theory.md`, "Analytic continuation" and "Quasiparticle equation, Newton solve, renormalization" | the derivations |
| `scripts/gw_self_energy_sympy.py`, claims 4, 5, 6 | the proofs |
| Vidberg and Serene, *J. Low Temp. Phys.* **29**, 179 (1977) | Thiele–Padé continuation |
| Zhu and Chan, *J. Chem. Theory Comput.* **17**, 727 (2021) | the `gw_ac` conventions matched here |
