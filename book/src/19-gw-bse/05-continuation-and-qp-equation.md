# 19.5 Analytic continuation and the quasiparticle equation

<!-- section: 19.5; status: ready; registry: gw;
     theory: crates/ferric-gw/theory.md#"6. Analytic continuation" -->

A quasiparticle energy is a real number: the energy $E$ at which an added or
removed electron, dressed by the response of the others, propagates without
decaying. Section 19.3 evaluates $\Sigma^c$ only at complex energies. The
bridge is analytic continuation: sample $\Sigma^c$ on the imaginary line
$z_k = \epsilon_F + iu_k$, fit a rational function through the samples, and
evaluate that function and its slope at real energies. A Newton iteration on
it then finds the root of the quasiparticle equation.

The plan:

1. Choose the line and the nodes, and fit a Thiele continued fraction
   (19.5.1).
2. Write the quasiparticle equation, linearize it to get $Z$, and solve it by
   Newton with a fallback (19.5.2).
3. Look at the options and limits, of which the continuation is the largest
   (19.5.4).

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

**The line.** $\Sigma^c_p$ is sampled on the Fermi-shifted imaginary axis
$z_k = \epsilon_F + iu_k$, with
$\epsilon_F = \tfrac12(\epsilon_{\rm HOMO} + \epsilon_{\rm LUMO})$ of the
active mean-field spectrum (`ferric-gw/src/sigma.rs::fermi_level`). The line
has to cross the real axis inside the gap, because the sign of
$\mathrm{Re}(z - \epsilon_q)$ decides which states count as occupied
(section 19.3.1).

**The nodes.** The default is 18 nodes (`GwConfig.pade_npts`), chosen by
`ferric-gw/src/sigma.rs::pade_node_indices`, a port of PySCF's `_get_ac_idx`,
from the grid $\{0\}\cup\mathrm{GL}(100, u_0 = 0.5)$ with a decreasing step:
step ratio $2/3$, first index 1 (`pade_support_omegas`). The test
`sigma::tests::pade_node_indices_matches_pyscf_get_ac_idx` pins the port. The
nodes are independent of the grid on which $W$ was integrated: $\Sigma^c(z)$ is
a closed-form function of $z$ once $W$ is fixed, and the nodes are where it
is sampled.

**Why a rational function.** Section 19.3 shows that $\Sigma^c$ is a sum of
simple poles, $s/(z - \text{pole})$. A sum of poles is a rational function,
so a rational interpolant can reproduce it.

**The Thiele continued fraction.** Write the interpolant as

$$ C_N(z) = \cfrac{a_1}{1 + \cfrac{a_2(z - z_1)}{1 + \cfrac{a_3(z-z_2)}{1 + \cdots}}} . $$

To find the coefficients, define a chain of functions by
$g_1(z) = f(z)$ and $g_{\ell-1}(z) = a_{\ell-1}/\big(1 + (z - z_{\ell-1})\,g_\ell(z)\big)$
with $a_\ell = g_\ell(z_\ell)$. Solve that definition for $g_\ell$:

$$ g_1(z_k) = f(z_k), \qquad g_\ell(z_k) = \frac{g_{\ell-1}(z_{\ell-1}) - g_{\ell-1}(z_k)}{(z_k - z_{\ell-1})\,g_{\ell-1}(z_k)}, \qquad a_\ell = g_\ell(z_\ell) . $$

Each row of the table needs only the previous row, so
`ferric-gw/src/pade.rs::fit` keeps two rolling buffers. At $z = z_k$ every
factor $(z - z_k)$ deeper than level $k$ vanishes, so $C_N$ passes through
its nodes. The fraction is evaluated bottom-up, from the innermost level out,
in `ferric-gw/src/pade.rs::eval`:

<!-- excerpt: ferric-gw/src/pade.rs::eval -->
```rust
        let mut acc = Complex64::new(1.0, 0.0);
        for p in (1..n).rev() {
            // acc <- 1 + a_p (z - z_{p-1}) / acc
            acc = Complex64::new(1.0, 0.0) + self.coeffs[p] * (z - self.z[p - 1]) / acc;
        }
        self.coeffs[0] / acc
```

**What exact arithmetic says.** The sympy script runs a line-for-line port of
`fit` and `eval` in exact rationals. $C_N$ interpolates its $N$ nodes. It
reproduces a rational function of type $[L/M]$ identically in $z$ once
$N \ge L + M + 1$. So a single-pole $\Sigma^c$ of two states, type $[1/2]$, is
continued exactly from 4 imaginary-axis nodes to the real axis. One surplus
node gives $a_N = 0$. A second drives the next denominator to exactly zero,
which `fit` refuses with an `Err` instead of emitting NaN
(`pade::tests::fit_errors_on_repeated_support_node`). The other unit tests
are `pade::tests::interpolation_identity`, `shifted_nodes_far_eval_matches_exact`
and `pole_reproduction`.

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

**What real arithmetic gives.** The physical $\Sigma^c$ is a rational
function of degree about $n_{\rm act}\times n_{\rm quad}$, far above 18. So
18 nodes give an approximation, not a reconstruction. The measured
sensitivity, 0.97 meV on the H2O@PBE HOMO between 16 and 18 nodes, is
recorded in the module doc of `crates/ferric-gw/tests/validation_gw.rs`.

**The slope.** The quasiparticle equation also needs
$\partial_E\,\Sigma^c$ on the real axis. `ferric-gw/src/pade.rs::deriv_real`
takes a four-point central difference of $C_N$ with $h = 0.05$ Ha:

$$ C'(E) \approx \frac{-C(E+2h) + 8C(E+h) - 8C(E-h) + C(E-2h)}{12h} . $$

It is exact for polynomials through degree 4, and its error is
$-h^4C^{(5)}/30$.

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

## 19.5.2 The quasiparticle equation

**The equation.** The quasiparticle Green's function of state $p$ is
$G(E) = [E - \epsilon_p - \delta_p - \Sigma^c_p(E)]^{-1}$. Its poles are the
quasiparticle energies. On the real axis the code solves for the zero of the
inverse:

$$ f(E) = E - \epsilon_p - \delta_p - \mathrm{Re}\,C_N(E) = 0, \qquad \delta_p = \begin{cases} 0 & \text{HF reference} \\ \Sigma^x_p - v^{xc}_p & \text{KS reference} \end{cases} $$

with $v^{xc}_p$ from `ferric-gw/src/vxc_mo.rs::vxc_diagonal_mo`, the same
`add_xc` path as the SCF's KS Fock (section 19.1.3).

**The static shift belongs inside.** $\delta_p$ is part of $f$, so $\Sigma^c$
is evaluated at the shifted root. Adding $\delta_p$ after an unshifted solve
evaluates $\Sigma^c$ at the wrong energy. On the open-shell @PBE cases that
lands 0.38 to 0.96 eV away, and `crates/ferric-gw/tests/validation_gw.rs`
requires ferric to miss that number.

**Linearize.** Expand $\Sigma^c$ to first order about the mean-field energy,
$\Sigma^c(E) \approx \Sigma^c(\epsilon_p) + \Sigma^{c\prime}(\epsilon_p)\,(E - \epsilon_p)$,
put it into $f(E) = 0$ and solve for $E$:

$$ E_0 = \epsilon_p + Z_p\big(\Sigma^c_p(\epsilon_p) + \delta_p\big), \qquad Z_p = \left[1 - \partial_E \mathrm{Re}\,\Sigma^c_p(E)\right]^{-1}_{E=\epsilon_p} . $$

$f'(E) = 1 - \Sigma^{c\prime}(E)$, so $E_0$ is exactly one undamped Newton step
on $f$ from $\epsilon_p$.

**Check the limits.** With no correlation, $\Sigma^c = 0$, $Z = 1$ and
$E = \epsilon_p + \delta_p$: the Koopmans energy on a Hartree–Fock reference.
For the exact $\Sigma^c$, a sum of poles with nonnegative weights $s$
(section 19.3.1), each term has slope $-s/(E - \text{pole})^2 \le 0$ on the
real axis. So $1 - \Sigma^{c\prime} \ge 1$ and $0 < Z \le 1$. $Z$ is the
fraction of the spectral weight the quasiparticle peak keeps; the rest goes
to satellites. The code clamps $Z_p$ to $[0, 1.5]$. A value pinned at 0 or
1.5 means the unclamped $Z$ of the Padé model was outside the physical range,
a near-pole or a wrong-sign slope at $\epsilon_p$, and section 19.8 uses that
as a flag.

**$Z$ is evaluated at the mean-field energy.** `GwResult.z_factor` is
$Z_p$ at $\epsilon_p$, the linearization point, not at the converged root.

**Newton.** From $E_0$, iterate
$E_{k+1} = E_k - d\,f(E_k)/f'(E_k)$ with $f' = 1 - \partial_E\mathrm{Re}\,\Sigma^c$.
The damping $d$ is `qp_newton_damp` clamped to $[0.1, 1]$. The loop takes at
most 30 steps, converges when $|\text{step}| < 10^{-7}$ Ha, and stops
unconverged if $|f'| < 10^{-3}$, which happens at a root near a pole of
$C_N$. For a single-pole $\Sigma$ the quasiparticle equation has two roots,
the quasiparticle and a satellite, and undamped Newton converges
quadratically to the one nearest $\epsilon_p$. At a root $E^*$,
$Z(E^*) = 1/(1-\Sigma'(E^*))$ is the residue of the quasiparticle pole of
$G$.

The Newton loop, from `ferric-gw/src/sigma.rs::solve_qp_for_mo_with`:

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

**The fallback.** Newton from the linearized start can wander when the
quasiparticle equation has several roots close together. Only then does
`ferric-gw/src/sigma.rs::qp_root_by_weight` run. It samples $f$ at 4001
points in $\epsilon_p + \delta_p \pm 1$ Ha, takes every sign change with
$|f| < 0.5$ on both sides (a sign change with large $|f|$ is a pole of $C_N$,
not a root), refines each root by bisection, and returns the one of largest
weight $Z \in (0, 1]$: the quasiparticle, by definition. The example is the
$\alpha$ HOMO of NH2/cc-pVDZ in U-G0W0@UKS. Newton fails there at every
$n_{\rm quad}$. On the exact real axis the equation has roots at −11.80 eV
($Z$ 0.62) and −10.09 eV ($Z$ 0.24), and the fallback gives −11.802 eV at
every $n_{\rm quad} \ge 30$. A state with no root of weight in $(0, 1]$ is
left unsolved: `GwResult.qp_converged` records it, and `run_gw` warns
(`warn_gw_unconverged`).

## 19.5.3 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | $\epsilon_F$ and the 18 node frequencies | $O(1)$ | `fermi_level`, `pade_support_omegas` |
| 2 | $\Sigma^c(\epsilon_F + iu_k)$ at the nodes, product-integrated | section 19.3 | `sigma_c_at_z_product` |
| 3 | Thiele fit | $O(N^2)$, $N$ = nodes | `PadeCF::fit` |
| 4 | $Z_p$ at $\epsilon_p$, clamped to $[0, 1.5]$; linearized start | $O(N)$ | `solve_qp_for_mo_with` |
| 5 | Newton, at most 30 steps, stops on $\lvert\text{step}\rvert < 10^{-7}$ Ha or $\lvert f'\rvert < 10^{-3}$ | $O(N)$ per step | `solve_qp_for_mo_with` |
| 6 | if Newton did not converge: the root of largest weight $Z\in(0,1]$ in $\epsilon_p+\delta_p\pm1$ Ha (4001 samples) | $4001\,O(N)$ | `qp_root_by_weight` |

The tests of the solve are `sigma::tests::qp_newton_converges_for_a_mild_self_energy`,
`qp_newton_flags_nonconvergence_when_iteration_budget_exhausted`; in
`crates/ferric-gw/tests/validation_gw.rs`, `g0w0_hf_{h2o,nh3,n2}_vs_pyscf_gw_ac`,
`g0w0_pbe_{h2o,n2}_vs_pyscf_gw_ac`, `g0w0_hf_frozen_core_h2o_vs_pyscf_gw_ac`
and `g0w0_hf_ecp_{i2,xe,ag2}_vs_pyscf_gw_ac`; and
`crates/ferric-gw/tests/g0w0_pbe_h2o.rs`.

## 19.5.4 Options and limits

| Option | Effect |
|---|---|
| `pade_npts` | number of continuation nodes (default 18) |
| `qp_newton_damp` | damping of the Newton step, clamped to $[0.1, 1]$ |
| `qp_mos` | absolute MO range for which the QP energy is solved; default HOMO−2 … LUMO+2 (`default_qp_range`; open shell `default_u_qp_range`, around $\max(n_\alpha, n_\beta)$) |

**Known limits (2026-11-15).** The physical $\Sigma^c$ is a rational function
of degree about $n_{\rm act}\times n_{\rm quad}$, so 18 nodes give an
approximation, not a reconstruction. The error is largest for KS references,
whose $\epsilon_{\rm HOMO}$ lies far from $\epsilon_F$, so the continuation
extrapolates further, and for states far from $\epsilon_F$ with split
spectral weight. This is what "narrow" means in the grade. The
unconverged-state warning is the signal to look at.

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
