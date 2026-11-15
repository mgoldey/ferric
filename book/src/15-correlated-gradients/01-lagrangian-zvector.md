# 15.1 Lagrangians and the Z-vector equation

<!-- section: 15.1; status: ready; registry: ri-mp2-gradient;
     theory: crates/ferric-mp2/theory.md#"9. Analytic RI-MP2 gradient and Z-vector" -->

A correlation energy built on Hartree–Fock orbitals is not stationary in
those orbitals, so its nuclear derivative seems to need the orbital response
to every one of the $3N_{\mathrm{at}}$ coordinates. The Lagrangian technique
removes that: the orbital conditions enter with multipliers, the multipliers
solve one linear equation that does not depend on the coordinate, and the
gradient becomes a contraction of densities with derivative integrals. The
equation for the multipliers is the Z-vector equation. Every correlated
gradient in this chapter (RI-MP2, SCS-MP2, RI-dRPA) and the MP2 relaxed
dipole solve the same one; the Kohn–Sham variant of section 15.4 differs in
the Hessian, not in the idea.

| Index | Range | Code name |
|---|---|---|
| $i, j$ | occupied, including frozen core in the response | `0..nocc_total` |
| $a, b$ | virtual | `0..nvir` |
| $p, q$ | all MOs | |
| $x$ | one nuclear coordinate | |

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/zvector.rs::{solve_zvector,build_relaxed_density_ao}` |
| Used by | `ferric-mp2/src/gradient.rs::{rimp2_gradient_analytical,relaxed_ri_gradient}`, `ferric-rpa/src/gradient_analytic.rs::rpa_analytic_gradient`, `ferric-mp2/src/cpks_polar.rs::static_relaxed_density_ao` |

## 15.1.1 The Lagrangian

Let $E(\mathbf C)$ be the total energy as a function of the orbitals. The
orbitals are fixed by the Brillouin condition $F_{ai} = 0$ and by
orthonormality $\mathbf C^{\mathsf T}\mathbf S\mathbf C = \mathbf 1$. Define

$$ \mathcal L(x, \mathbf C, z, W) = E(x, \mathbf C) + \sum_{ai} z_{ai}\,F_{ai}(x, \mathbf C) - \sum_{pq} W_{pq}\big[(\mathbf C^{\mathsf T}\mathbf S\mathbf C)_{pq} - \delta_{pq}\big] . \tag{15.1} $$

At the reference orbitals the constraints vanish, so $\mathcal L = E$. If $z$
and $W$ are chosen so that $\partial\mathcal L/\partial\mathbf C = 0$, the
total derivative is the explicit one:

$$ \frac{dE}{dx} = \frac{\partial\mathcal L}{\partial x}\bigg|_{\mathbf C, z, W} . \tag{15.2} $$

Stationarity in the occupied-virtual rotations gives the Z-vector equation;
stationarity in the remaining rotations and in orthonormality gives $W$, the
energy-weighted density, with no further solve.

## 15.1.2 The Z-vector equation in ferric

For a closed-shell RHF reference and the four-center $J$ and $K$, the
equation solved by `solve_zvector` is

$$ \Delta_i^a\,z_{ai} + \tfrac12\Big[\mathbf C_v^{\mathsf T}\big(4J - K - K^{\mathsf T}\big)[D^z]\,\mathbf C_o\Big]_{ai} = -X_{ai}, \qquad D^z = \sum_{ai} z_{ai}\big(\mathbf C_a\mathbf C_i^{\mathsf T} + \mathbf C_i\mathbf C_a^{\mathsf T}\big), \tag{15.3} $$

which is the orbital Hessian
$\Delta_i^a\delta_{ab}\delta_{ij} + 4(ai|bj) - (ab|ij) - (aj|bi)$ applied to
$z$, built from one $J$ and one $K$ of a density instead of from MO integrals.
$X_{ai}$ is the right-hand side of the method: section 15.2 gives it for
RI-MP2, and section 15.5 for RI-dRPA. It runs over every occupied orbital,
including frozen core.

> **Verified equation 15.3** · the Hessian product
>
> The vo block of $(4J - K - K^{\mathsf T})[D^z]$ is twice
> $\sum_{bj}[4(ai|bj) - (ab|ij) - (aj|bi)]z_{bj}$, so the code's factor
> $\tfrac12$ gives the closed-shell orbital Hessian.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_variant_identities.py`, claim 5 |
> | Negative control | the runner's mutation $(4J - K - K^{\mathsf T}) \to (4J - K)$ turns claim 5 red; its mutation of the coupling factor $\tfrac12 \to 1$ in `scripts/mp2_gradient_sympy.py` turns part G red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/zvector.rs::{compute_az_product,az_product_with_jk,solve_zvector_timed}` |
> | Code test | `test_zvector_converges` in `crates/ferric-mp2/src/zvector.rs` |
>
> <!-- verified: id=zvec.hessian-product; script=scripts/mp2_variant_identities.py; claim=5; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/zvector.rs::{compute_az_product,az_product_with_jk,solve_zvector_timed}; rust_test=ferric-mp2/src/zvector.rs::test_zvector_converges; registry=ri-mp2-gradient -->

**The solver.** The start is $z = -X/\Delta$. Each iteration forms the
Hessian product, the residual $r = -X - \Delta z - Az$, and a Jacobi update
$(−X − Az)/\Delta$ extrapolated by DIIS with eight vectors and the residual
as error. It stops when $\max|r| < 10^{-8}$; after 50 iterations without that,
the gradient path refuses the result (`ZvectorSolution::require_converged`).

<!-- excerpt: ferric-mp2/src/zvector.rs::solve_zvector_timed -->
```rust
let (az2, jz, kz) = az_product_with_jk(c, &z, jk, &zorb, pool.as_ref(), budget_bytes)?;
let az = 0.5 * &az2;
```

| Symbol | Code |
|---|---|
| $(4J - K - K^{\mathsf T})[D^z]_{vo}$ | `az2` |
| the Hessian coupling in (15.3) | `az` |
| $J[D^z], K[D^z]$ | `jz`, `kz` |

**Two details that change results.** The density $D^z$ is scaled by a power
of two before the $J$ and $K$ build so that the integral screening sees a
unit-scale density; a power of two is exact, so only screening changes. And
the $J$ and $K$ of the converged $z$ are returned with the solution
(`corr_jk`), so the gradient assembly reuses them and builds no further $J$ or
$K$ after the Z-vector.

## 15.1.3 The relaxed density

With $z$ known, the relaxed one-particle density is the unrelaxed correlation
density plus $z$ on both off-diagonal blocks. Its contraction with any
one-electron perturbation is the derivative of the energy with respect to
that perturbation, which makes it the density for properties (section 15.6)
as well as gradients.

> **Verified equation 15.4** · the relaxed density is the energy derivative
>
> For $h \to h + \lambda R$,
> $dE/d\lambda = \operatorname{Tr}[(D^{\mathrm{HF}} + D^{\mathrm{corr}}_{\mathrm{rel}})R]$,
> with $D^{\mathrm{corr}}_{\mathrm{rel}}$ built by the code's Z-vector, for MP2
> and SCS-MP2, with and without a frozen core.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_gradient_sympy.py`, part G (G1) |
> | Negative control | the runner's mutations of part G (Hessian factor, the sign of $I_{ai}$, the SCS weight, the $\zeta$ weight) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/zvector.rs::{solve_zvector,correlation_density_mo,build_relaxed_density_ao}` |
>
> <!-- verified: id=zvec.relaxed-density; script=scripts/mp2_gradient_sympy.py; claim=G; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/zvector.rs::{solve_zvector,correlation_density_mo,build_relaxed_density_ao}; rust_test=ferric-mp2/tests/cpks_polar.rs::cpks_static_relaxed_dipole_vs_pyscf; registry=ri-mp2-gradient -->

`build_relaxed_density_ao` assembles $2\delta_{ij} + P$ on the occupied block,
$P$ on the virtual block and $z$ on both off-diagonal blocks, in the AO basis,
for whatever $P$ the caller passes. The gradient itself does not call it; it
assembles the same terms inside its Lagrangian (section 15.2).

## 15.1.4 Options

| Option | Default | Effect |
|---|---|---|
| `ZvectorControls` (library) | tolerance $10^{-8}$, 50 iterations | the stop rule above |
| `FERRIC_ZVEC_TRACE` | off | prints the residual per iteration |

## 15.1.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-mp2/theory.md`, "9. Analytic RI-MP2 gradient and Z-vector" | (15.3), the right-hand side, the relaxed density |
| `scripts/mp2_variant_identities.py`, claim 5 | the Hessian factor |
| `scripts/mp2_gradient_sympy.py`, part G | the relaxed density and the assembled gradient |

**Known limits (2026-11-15).** Closed-shell reference only. The Z-vector of
an unrestricted correlated method is not implemented; the unrestricted
OO-RI-MP2 needs none (section 15.3).
