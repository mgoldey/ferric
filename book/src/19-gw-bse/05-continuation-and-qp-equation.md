# Analytic continuation and the quasiparticle equation {#continuation-and-qp-equation}

<!-- status: ready; registry: gw -->

A quasiparticle energy is a real number: the energy $E$ at which an added or
removed electron, dressed by the response of the others, propagates without
decaying. Section @sec:correlation-self-energy evaluates $\Sigma^c$ only at complex energies. The
bridge is analytic continuation: sample $\Sigma^c$ on the imaginary line
$z_k = \epsilon_F + iu_k$, fit a rational function through the samples, and
evaluate that function and its slope at real energies. A Newton iteration on
it then finds the root of the quasiparticle equation.

The plan:

1. Choose the line and the nodes, and fit a Thiele continued fraction
   (@sec:continuation-and-qp-equation-continuation).
2. Write the quasiparticle equation, linearize it to get $Z$, and solve it by
   Newton with a fallback (@sec:continuation-and-qp-equation-quasiparticle-equation).
3. Look at the options and limits, of which the continuation is the largest
   (@sec:continuation-and-qp-equation-options-limits).

| Symbol | Meaning | Code name |
|---|---|---|
| $\epsilon_F$ | mid-gap of the active mean-field spectrum | `ef`, `fermi_level` |
| $z_k$ | continuation nodes, 18 by default | `pade_support_omegas` |
| $C_N(z)$ | the Thiele continued fraction | `PadeCF` |
| $\delta_p$ | static shift, $0$ (HF) or $\Sigma^x_p - v^{xc}_p$ (KS) | `static_shift` |
| $c_{\rm sr}, c_{\rm lr}, \omega$ | exact-exchange fractions and range-separation parameter of a hybrid | `KMix { sr, lr, omega }` |
| $Z_p$ | renormalization factor | `z_renorm` |

| Entry point | Location |
|---|---|
| Rust | `ferric-gw/src/pade.rs::{fit,eval,deriv_real}`, `ferric-gw/src/sigma.rs::{solve_qp_for_mo,solve_qp_for_mo_with,qp_root_by_weight}` |
| Options | `[gw] pade_npts`, `qp_newton_damp`, `qp_mos` |

## The continuation {#continuation-and-qp-equation-continuation}

**The line.** $\Sigma^c_p$ is sampled on the Fermi-shifted imaginary axis
$z_k = \epsilon_F + iu_k$, with
$\epsilon_F = \tfrac12(\epsilon_{\rm HOMO} + \epsilon_{\rm LUMO})$ of the
active mean-field spectrum (`ferric-gw/src/sigma.rs::fermi_level`). The line
has to cross the real axis inside the gap, because the sign of
$\mathrm{Re}(z - \epsilon_q)$ decides which states count as occupied
(section @sec:correlation-self-energy-frequency-integral).

**The nodes.** The default is 18 nodes (`GwConfig.pade_npts`), chosen by
`ferric-gw/src/sigma.rs::pade_node_indices`, a port of PySCF's `_get_ac_idx`,
from the grid $\{0\}\cup\mathrm{GL}(100, u_0 = 0.5)$ with a decreasing step:
step ratio $2/3$, first index 1 (`pade_support_omegas`). The test
`sigma::tests::pade_node_indices_matches_pyscf_get_ac_idx` pins the port. The
nodes are independent of the grid on which $W$ was integrated: $\Sigma^c(z)$ is
a closed-form function of $z$ once $W$ is fixed, and the nodes are where it
is sampled.

**Why a rational function.** Section @sec:correlation-self-energy shows that $\Sigma^c$ is a sum of
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

> **Verified equation {#eq:gw-pade-thiele}** · the Thiele continued fraction, in exact arithmetic
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

> **Verified equation {#eq:gw-pade-stencil}** · the four-point derivative stencil
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

## The quasiparticle equation {#continuation-and-qp-equation-quasiparticle-equation}

**The equation.** The quasiparticle Green's function of state $p$ is
$G(E) = [E - \epsilon_p - \delta_p - \Sigma^c_p(E)]^{-1}$. Its poles are the
quasiparticle energies. On the real axis the code solves for the zero of the
inverse:

$$ f(E) = E - \epsilon_p - \delta_p - \mathrm{Re}\,C_N(E) = 0, \qquad \delta_p = \begin{cases} 0 & \text{HF reference} \\ \Sigma^x_p - v^{xc}_p & \text{KS reference} \end{cases} $$

with $v^{xc}_p$ from `ferric-gw/src/vxc_mo.rs::vxc_diagonal_mo`, the same
`add_xc` path as the SCF's KS Fock (section @sec:ri-factors-and-exchange-ks-reference-changes).
For a hybrid functional $v^{xc}_p$ also carries the exact exchange of that
Fock matrix (section @sec:continuation-and-qp-equation-hybrid-references).

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
(section @sec:correlation-self-energy-frequency-integral), each term has slope $-s/(E - \text{pole})^2 \le 0$ on the
real axis. So $1 - \Sigma^{c\prime} \ge 1$ and $0 < Z \le 1$. $Z$ is the
fraction of the spectral weight the quasiparticle peak keeps; the rest goes
to satellites. The code clamps $Z_p$ to $[0, 1.5]$. A value pinned at 0 or
1.5 means the unclamped $Z$ of the Padé model was outside the physical range,
a near-pole or a wrong-sign slope at $\epsilon_p$, and section @sec:bse-tda uses that
as a flag.

**$Z$ is evaluated at the mean-field energy.** `GwResult.z_factor` is
$Z_p$ at $\epsilon_p$, the linearization point, not at the converged root.

G0W0@HF for water / cc-pVDZ, from the `589b50832` build, at the matched
settings of the validation tables. Every $Z$ lies in $(0, 1]$, and the
$Z$ printed is the one at $\epsilon_p$.

```python
# Water / cc-pVDZ G0W0@HF: Koopmans energies against quasiparticle energies, with Sigma_x, Sigma_c and Z.
import ferric

HA_EV = 27.211386245988
mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

# Matched settings of the validation tables (n_quad = 100, trunc_thresh = 0).
r = ferric.run_gw(mol, bs, aux, method="g0w0", n_quad=100, trunc_thresh=0.0)
print(" MO   eps_HF(eV)  eps_QP(eV)  Sigma_x(eV) Sigma_c(eV)   Z    converged")
for k, mo in enumerate(r.mo_indices):
    print(f"{mo:3d} {r.eps_mf[k]*HA_EV:11.4f} {r.eps_qp[k]*HA_EV:11.4f} "
          f"{r.sigma_x[k]*HA_EV:11.4f} {r.sigma_c[k]*HA_EV:11.4f} "
          f"{r.z_factor[k]:6.3f}  {r.qp_converged[k]}")
```

```text
 MO   eps_HF(eV)  eps_QP(eV)  Sigma_x(eV) Sigma_c(eV)   Z    converged
  2    -18.9990    -18.5312    -24.6776      0.4678  0.953  True
  3    -15.4359    -14.4566    -26.3257      0.9793  0.950  True
  4    -13.4218    -12.1600    -27.0142      1.2618  0.949  True
  5      5.0444      4.7034     -2.6009     -0.3410  0.989  True
  6      6.9717      6.6555     -2.5337     -0.3162  0.988  True
  7     21.4290     20.3167     -5.5093     -1.1123  0.969  True
```

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

> **Verified equation {#eq:gw-qp-equation}** · the linearized start and the Newton iteration
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
$n_{\rm quad}$. On the exact real axis the equation has roots at −11.81 eV
($Z$ 0.62) and −10.09 eV ($Z$ 0.24), and the fallback gives −11.802 eV at
every $n_{\rm quad} \ge 30$. A state with no root of weight in $(0, 1]$ is
left unsolved: `GwResult.qp_converged` records it, and `run_gw` warns
(`warn_gw_unconverged`).

## Hybrid references {#continuation-and-qp-equation-hybrid-references}

What does $v^{xc}_p$ mean when the reference is PBE0 or CAM-B3LYP? The shift
$\delta_p$ removes from $\epsilon_p$ everything the Kohn–Sham Fock matrix holds
besides $h + J$, and puts $\Sigma^x_p$ in its place. A hybrid's Fock matrix
holds a fraction of exact exchange next to the semilocal potential. $\Sigma^x_p$
is the full exchange, so a $v^{xc}_p$ without that fraction counts it twice.

**The Fock matrix.** Closed shell, with $D = 2D^\alpha$, the SCF assembles
(`ferric-scf/src/rhf.rs`, "F = H + J − ½ K_total")

$$ F = h + J[D] + V^{\rm sl} - \tfrac12\big(c_{\rm sr} K^{\rm erfc}_\omega[D] + c_{\rm lr} K^{\rm erf}_\omega[D]\big), \qquad K[D]_{\mu\nu} = \sum_{\lambda\sigma} (\mu\lambda|\nu\sigma) D_{\lambda\sigma} . $$

$V^{\rm sl}$ is the semilocal potential with VV10, and
$(c_{\rm sr}, c_{\rm lr}, \omega)$ is the functional's `KMix`
(`ferric-dft/src/ks.rs::KsXc`, method `k_mix`). A global hybrid has
$\omega = 0$ and $c_{\rm sr} = c_{\rm lr} = c_x$ on one Coulomb $K$; a pure
functional has all three zero. $K$ is linear in $D$, so
$\tfrac12 K[D] = K[D^\alpha]$. Subtract $h + J$ and write the result per spin:

$$ v^{\rm mf}_\sigma = V^{\rm sl}_\sigma - c_{\rm sr} K^{\rm erfc}_\omega[D^\sigma] - c_{\rm lr} K^{\rm erf}_\omega[D^\sigma], \qquad \delta_p = \Sigma^x_p - \big(C^{\sigma T} v^{\rm mf}_\sigma C^\sigma\big)_{pp} . \label{eq:gw-hybrid-vmf} $$

The unrestricted Fock matrix has $-K[D^\sigma]$ in $F^\sigma$
(`ferric-scf/src/uhf.rs`), so (@eq:gw-hybrid-vmf) is also the UKS and ROKS
form.

**The global hybrid.** Transform $K[D^\sigma]$ to the MO basis. With
$D^\sigma = \sum_i C_{\mu i}C_{\nu i}$ over occupied $i$,
$(K[D^\sigma])_{pp} = \sum_i (pi|ip) = -\Sigma^x_p$. Substitute into
(@eq:gw-hybrid-vmf) with $\omega = 0$:

$$ \delta_p = (1 - c_x)\,\Sigma^x_p - V^{\rm sl}_{pp} . \label{eq:gw-hybrid-shift} $$

Check the limits. At $c_x = 0$ the shift reduces to the semilocal one of
section @sec:ri-factors-and-exchange-ks-reference-changes. At $c_x = 1$ and
$V^{\rm sl} = 0$, Hartree–Fock, it is zero, which is the shift an HF
reference gets. For a range-separated hybrid, $\mathrm{erf} + \mathrm{erfc} = 1$
splits $\Sigma^x$ into short- and long-range parts, and
$c_{\rm sr} = c_{\rm lr}$ collapses the result to (@eq:gw-hybrid-shift).

**The size of the double count.** Leave the $K$ term out and $\delta_p$ is
too large by $c_x \Sigma^x_p$. $\Sigma^x_p \le 0$, so every quasiparticle
energy moves down by $c_x\lvert\Sigma^x_p\rvert$. For an occupied valence
state $\lvert\Sigma^x_p\rvert$ is of order an hartree, so at $c_x = 1/4$ the
error is several eV.

> **Verified equation @eq:gw-hybrid-shift** · the static shift on a hybrid reference
>
> The RKS Fock matrix's exchange term $-\tfrac12 c K[D]$ is $-cK[D^\alpha]$,
> whose MO diagonal is $c\,\Sigma^x_p$; hence
> $\Sigma^x_p - v^{\rm mf}_p = (1-c)\Sigma^x_p - V^{\rm sl}_{pp}$, equal to the
> semilocal shift at $c = 0$ and to zero for Hartree–Fock; the erf/erfc split
> reduces to it when $c_{\rm sr} = c_{\rm lr}$; and PySCF's
> `get_veff − get_j` minus its semilocal potential equals $-\tfrac12 cK$
> built separately (H2O/cc-pVDZ, PBE0 and CAM-B3LYP).
>
> | | |
> |---|---|
> | Proof | `scripts/gw_hybrid_vmf_sympy.py`, claims H1 to H5 |
> | Negative control | the Fock exchange without the $\tfrac12$, and a shift that subtracts $V^{\rm sl}$ alone, must fail |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-gw/src/vxc_mo.rs::{vxc_diagonal_mo_on_grids,add_exact_exchange}` |
> | Code test | `g0w0_pbe0_h2o_vs_pyscf` in `crates/ferric-gw/tests/gw_hybrid_reference.rs` |
>
> <!-- verified: id=gw.hybrid-shift; script=scripts/gw_hybrid_vmf_sympy.py; claim=H2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-gw/src/vxc_mo.rs::{vxc_diagonal_mo_on_grids,add_exact_exchange}; rust_test=crates/ferric-gw/tests/gw_hybrid_reference.rs::g0w0_pbe0_h2o_vs_pyscf; registry=gw -->

**The code.** `ferric-gw/src/vxc_mo.rs::add_exact_exchange` adds the $K$ term
of (@eq:gw-hybrid-vmf) to the AO potential that `add_xc` built, before the MO
transform. The excerpt is the operator choice. Each term's $K[D^\sigma]$ comes
from `ferric-scf/src/rhf.rs::build_jk` and is subtracted with its coefficient:

<!-- excerpt: ferric-gw/src/vxc_mo.rs::add_exact_exchange -->
```rust
    let terms: Vec<(Operator, f64)> = if k_mix.omega > 0.0 {
        vec![
            (Operator::erfc(k_mix.omega), k_mix.sr), // c_sr K^erfc_ω
            (Operator::erf(k_mix.omega), k_mix.lr),  // c_lr K^erf_ω
        ]
    } else {
        vec![(Operator::coulomb(), k_mix.sr)] // c_x K
    };
```

- The closed-shell call passes $D^\alpha$ (`density_alpha`), which is
  $\tfrac12 D$; the factor $\tfrac12$ of the Fock matrix is in that choice.
- A term with a zero coefficient is dropped before any integral is computed.
  A pure functional therefore builds nothing, and its $v^{\rm mf}$ is the
  semilocal potential bit for bit (`pure_functional_has_no_k_term`).
- The $K$ term uses exact four-centre integrals on the operator the SCF's
  `k_mix` names. A reference SCF with RI-K, the default for a functional and
  the only route ferric has for $\omega > 0$, holds $K^{\rm RI}$ in its Fock
  matrix, so $\delta_p$ then carries $c\,(K^{\rm RI} - K)_{pp}$, the fitting
  error. $\Sigma^x$ is the GW auxiliary basis's RI exchange in every case,
  Hartree–Fock included.

## The algorithm {#continuation-and-qp-equation-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | $\epsilon_F$ and the 18 node frequencies | $O(1)$ | `fermi_level`, `pade_support_omegas` |
| 2 | $\Sigma^c(\epsilon_F + iu_k)$ at the nodes, product-integrated | section @sec:correlation-self-energy | `sigma_c_at_z_product` |
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

## Options and limits {#continuation-and-qp-equation-options-limits}

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

## Numerical anchors {#continuation-and-qp-equation-numerical-anchors}

From `site/src/reference/validation.md` (`gw` row and G0W0 anchors), at
`n_quad = 100`, `trunc_thresh = 0`:

| Quantity | Reference | Error | Test |
|---|---|---|---|
| G0W0@HF QP energies, HOMO−2 … LUMO+2 | PySCF `gw_ac` | $\le 1.6\times10^{-7}$ Ha | `g0w0_hf_{h2o,nh3,n2}_vs_pyscf_gw_ac` in `crates/ferric-gw/tests/validation_gw.rs` |
| G0W0@HF with ECPs | grid-free dRPA pole sum | $\le 8.0\times10^{-10}$ Ha | `g0w0_hf_ecp_{i2,xe,ag2}_vs_pyscf_gw_ac` |
| G0W0@PBE | grid-free dRPA pole sum | $\le 1.8\times10^{-8}$ Ha | `g0w0_pbe_{h2o,n2}_vs_pyscf_gw_ac` |
| G0W0@HF with ECPs | PySCF `gw_ac` | $\le 2.5\times10^{-6}$ Ha | the site's `gw` row |

A benchmark against experiment measures the method, not the
implementation, so it carries no pinned tolerance. For GW100, G0W0@HF gives
a mean absolute error of 0.353 eV at aug-cc-pVDZ and 0.491 eV at
aug-cc-pVTZ; these numbers are kept in the working notes, not in a test.
The algebra of this chapter is exact. The numerical error budget is the
$W$ quadrature, the 18-node Padé extrapolation (largest for KS references,
whose $\epsilon_{\rm HOMO}$ lies far from $\epsilon_F$), the RI basis, and
the PDEP truncation when it is enabled.

## Sources {#continuation-and-qp-equation-sources}

| Source | What it gives |
|---|---|
| `scripts/gw_self_energy_sympy.py`, claims 4, 5, 6 | the proofs |
| Vidberg and Serene, *J. Low Temp. Phys.* **29**, 179 (1977) | Thiele–Padé continuation |
| Zhu and Chan, *J. Chem. Theory Comput.* **17**, 727 (2021) | the `gw_ac` conventions matched here |
| J. Wilhelm, D. Golze, L. Talirz, J. Hutter, C. A. Pignedoli, *J. Phys. Chem. Lett.* **9**, 306 (2018) | analytic-continuation G0W0 for molecules |
| M. J. van Setten *et al.*, *J. Chem. Theory Comput.* **11**, 5665 (2015) | the GW100 benchmark set |
