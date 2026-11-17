# Lagrangians and the Z-vector equation {#lagrangian-zvector}

<!-- status: ready; registry: ri-mp2-gradient -->

Move a nucleus and the Hartree–Fock orbitals change shape: they follow the
basis functions and re-mix to stay self-consistent. The Hartree–Fock energy
does not notice the re-mixing at first order, because the orbitals already
minimize it. The MP2 energy does notice. It is computed from those orbitals
but was never minimized with respect to them, so a small orbital rotation
changes it at first order. A correlated gradient therefore needs the orbital
response, and the direct route costs one coupled-perturbed solve per nuclear
coordinate, $3N_{\mathrm{at}}$ in all (section @sec:hf-hessian solves exactly those for
the Hartree–Fock Hessian).

The Z-vector replaces those $3N_{\mathrm{at}}$ solves by one. The orbital
conditions enter a Lagrangian with multipliers, the multipliers solve one
linear equation that does not depend on the coordinate, and the gradient
becomes a contraction of densities with derivative integrals. Every
correlated gradient in this chapter (RI-MP2, SCS-MP2, RI-dRPA) and the MP2
relaxed dipole solve the same equation. The Kohn–Sham variant of section
@sec:ks-zvector differs in the Hessian, not in the idea.

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

## The Lagrangian {#lagrangian-zvector-lagrangian}

Start with the cost of the direct route. Write the orbitals as the
reference orbitals rotated by $\kappa_{ai}$. The energy depends on the
coordinate $x$ explicitly (through the integrals) and through the rotation
$\kappa(x)$ that keeps the Brillouin condition $F_{ai} = 0$ true at every
geometry. The chain rule gives

$$ \frac{dE}{dx} = \frac{\partial E}{\partial x} + \sum_{ai}\frac{\partial E}{\partial\kappa_{ai}}\frac{d\kappa_{ai}}{dx} . $$

For Hartree–Fock the second term is zero, since $\partial E/\partial\kappa = 0$
at the SCF solution. For MP2 it is not. To find $d\kappa/dx$, differentiate
$F_{ai}(x, \kappa(x)) = 0$ with respect to $x$. With the orbital Hessian
$H_{ai,bj} = \partial F_{ai}/\partial\kappa_{bj}$,

$$ H\,\frac{d\kappa}{dx} = -\frac{\partial F}{\partial x} . $$

That is one linear solve per coordinate. Now substitute
$d\kappa/dx = -H^{-1}\,\partial F/\partial x$ into the chain rule and move
$H^{-1}$ onto the other factor:

$$ \sum_{ai}\frac{\partial E}{\partial\kappa_{ai}}\frac{d\kappa_{ai}}{dx}
 = -\Big(\frac{\partial E}{\partial\kappa}\Big)^{\mathsf T}H^{-1}\frac{\partial F}{\partial x}
 = z^{\mathsf T}\frac{\partial F}{\partial x}, \qquad H^{\mathsf T}z = -\frac{\partial E}{\partial\kappa} . $$

The vector $z$ depends on $\partial E/\partial\kappa$ and $H$, and on no
coordinate. One solve for $z$ serves all $3N_{\mathrm{at}}$ coordinates. This
is the Z-vector trick of Handy and Schaefer.

The Lagrangian packages the same argument and adds orthonormality. Let
$E(\mathbf C)$ be the total energy as a function of the orbitals. The
orbitals are fixed by the Brillouin condition $F_{ai} = 0$ and by
orthonormality $\mathbf C^{\mathsf T}\mathbf S\mathbf C = \mathbf 1$. Define

$$ \mathcal L(x, \mathbf C, z, W) = E(x, \mathbf C) + \sum_{ai} z_{ai}\,F_{ai}(x, \mathbf C) - \sum_{pq} W_{pq}\big[(\mathbf C^{\mathsf T}\mathbf S\mathbf C)_{pq} - \delta_{pq}\big] . \label{eq:lagrangian-zvector-1} $$

At the reference orbitals both constraints vanish, so $\mathcal L = E$. If
$z$ and $W$ are chosen so that $\partial\mathcal L/\partial\mathbf C = 0$,
the implicit dependence of $\mathbf C$ on $x$ drops out of $d\mathcal L/dx$,
and the total derivative is the explicit one:

$$ \frac{dE}{dx} = \frac{\partial\mathcal L}{\partial x}\bigg|_{\mathbf C, z, W} . \label{eq:lagrangian-zvector-2} $$

The conditions $\partial\mathcal L/\partial\mathbf C = 0$ split by rotation
type. Stationarity in the occupied–virtual rotations is the Z-vector equation
(section @sec:lagrangian-zvector-z-vector-equation-ferric). Stationarity in the remaining rotations and in
orthonormality gives $W$, the energy-weighted density, with no further solve.
The coordinate enters (@eq:lagrangian-zvector-2) only through derivative integrals: $h'$, $S'$,
the two-electron integrals and, for RI methods, the fitting integrals.

Check the limit where the method adds nothing. If $E$ is the Hartree–Fock
energy itself, $\partial E/\partial\kappa_{ai}$ is a multiple of $F_{ai}$,
which is zero, so $z = 0$ and (@eq:lagrangian-zvector-2) is the Hartree–Fock gradient of
section @sec:hf-gradient. Section @sec:oo-rimp2-gradient uses the same argument for OO-MP2, whose
orbitals make the correlated energy stationary.

## The Z-vector equation in ferric {#lagrangian-zvector-z-vector-equation-ferric}

For a closed-shell RHF reference and the four-center $J$ and $K$, the
equation solved by `ferric-mp2/src/zvector.rs::solve_zvector` is

$$ \Delta_i^a\,z_{ai} + \tfrac12\Big[\mathbf C_v^{\mathsf T}\big(4J - K - K^{\mathsf T}\big)[D^z]\,\mathbf C_o\Big]_{ai} = -X_{ai}, \qquad D^z = \sum_{ai} z_{ai}\big(\mathbf C_a\mathbf C_i^{\mathsf T} + \mathbf C_i\mathbf C_a^{\mathsf T}\big), \label{eq:zvec-hessian-product} $$

with $\Delta_i^a = \varepsilon_a - \varepsilon_i$. Read it term by term:

- $\Delta_i^a z_{ai}$ is the diagonal of the orbital Hessian, the cost of
  promoting an electron from $i$ to $a$ in canonical orbitals.
- The bracket is the Coulomb and exchange response of the Fock matrix to the
  density change $D^z$ that the rotation $z$ would cause. It is the
  orbital Hessian $\Delta_i^a\delta_{ab}\delta_{ij} + 4(ai|bj) - (ab|ij) - (aj|bi)$
  applied to $z$, built from one $J$ and one $K$ of a density instead of from
  MO integrals.
- $X_{ai}$ is $\partial E/\partial\kappa_{ai}$ of the correlated method, the
  right-hand side of @sec:lagrangian-zvector-lagrangian. Section @sec:rimp2-gradient gives it for RI-MP2, and section
  @sec:rpa-gradient for RI-dRPA. It runs over every occupied orbital, including frozen
  core.

The Hessian is the closed-shell $A + B$ of real orbital rotations (section
@sec:second-order-orbital-gradient-hessian derives the same matrix for the second-order SCF). It is symmetric, so
$H^{\mathsf T}z = H z$ and the transpose in @sec:lagrangian-zvector-lagrangian does not change the
equation.

> **Verified equation @eq:zvec-hessian-product** · the Hessian product
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

### The solver

The start is $z = -X/\Delta$, the solution with the two-electron coupling
switched off. Each iteration forms the Hessian product, the residual
$r = -X - \Delta z - Az$, and a Jacobi update $(-X - Az)/\Delta$, extrapolated
by DIIS with eight vectors and the residual as error. It stops when
$\max|r| < 10^{-8}$. After 50 iterations without that, the gradient path
refuses the result (`ZvectorSolution::require_converged`).

The iteration body of `ferric-mp2/src/zvector.rs::solve_zvector_with_imat`
starts with the Hessian product:

<!-- excerpt: ferric-mp2/src/zvector.rs::solve_zvector_with_imat -->
```rust
let (az2, jz, kz) = az_product_with_jk(c, &z, jk, &zorb, pool.as_ref(), budget_bytes)?;
let az = 0.5 * &az2;
```

| Symbol | Code |
|---|---|
| $(4J - K - K^{\mathsf T})[D^z]_{vo}$ | `az2` |
| the Hessian coupling in (@eq:zvec-hessian-product) | `az` |
| $J[D^z], K[D^z]$ | `jz`, `kz` |

The rest of the loop body forms the residual and the Jacobi update. It is
shown here with the loops over $a$ and $i$, the trace output, the
converged-return branch and the guard against a gap below $10^{-12}$
removed:

```rust
let denom = eps[nocc_total + a] - eps[i];                  // Δ_i^a
residual[(a, i)] = rhs[(a, i)] - denom * z[(a, i)] - az[(a, i)];   // r = −X − Δz − Az
// ...
z_new[(a, i)] = (rhs[(a, i)] - az[(a, i)]) / denom;        // Jacobi: (−X − Az)/Δ
// ...
z = diis.step(&z_new, &residual);                          // DIIS, error vector r
```

Here `rhs` already holds $-X$. The code builds it with the minus sign so that
the loop solves $\Delta z + Az = \mathrm{rhs}$ and returns the same $z$ as
PySCF's CPHF driver, element for element.

**Two details that change results.**

- The density $D^z$ is scaled by a power of two before the $J$ and $K$
  build, so that the integral screening sees a unit-scale density. A power
  of two is exact in floating point, so only the screening changes. Without
  it, the screening threshold is absolute and discards a growing share of the
  coupling as $z$ shrinks.
- The $J$ and $K$ of the converged $z$ are returned with the solution
  (`corr_jk`). The gradient assembly needs $J$ and $K$ of the relaxed
  correlation density, and by linearity those are the right-hand side's
  $J$, $K$ of the unrelaxed density plus these. No further $J$ or $K$ is
  built after the Z-vector.

## The relaxed density {#lagrangian-zvector-relaxed-density}

With $z$ known, the relaxed one-particle density is the unrelaxed
correlation density plus $z$ on both off-diagonal blocks. Why is that the
right density for properties? Take a one-electron perturbation
$h \to h + \lambda R$. By (@eq:lagrangian-zvector-2), $dE/d\lambda$ is the explicit derivative of
$\mathcal L$, and $\lambda$ enters $\mathcal L$ only through $h$: in the
Hartree–Fock energy, in the Fock matrix that sets the orbital energies of the
MP2 denominators, and in the constraint term $z_{ai}F_{ai}$. In each place $h$ enters through its
matrix elements, so each derivative is a sum of elements of $R$ with fixed
weights, and $dE/d\lambda$ is a trace of $R$ with one density that collects
all three. That density is the relaxed density, and its contraction
with any one-electron operator is the first derivative of the energy with
respect to that operator. It is the density for properties (section @sec:mp2-polarizability)
as well as gradients.

The $z$ block is what distinguishes it from the unrelaxed density. The
unrelaxed density answers "how does the energy change if the orbitals are
held fixed?" The $z$ block adds the change from the orbitals relaxing in
response to the perturbation.

> **Verified equation {#eq:zvec-relaxed-density}** · the relaxed density is the energy derivative
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

`ferric-mp2/src/zvector.rs::build_relaxed_density_ao` assembles
$2\delta_{ij} + P$ on the active occupied block, $P$ on the virtual block and
$z$ on both off-diagonal blocks, in the AO basis, for whatever $P$ the caller
passes. The gradient itself does not call it. It assembles the same terms
inside its Lagrangian (section @sec:rimp2-gradient), through
`ferric-mp2/src/zvector.rs::correlation_density_mo`, which also carries the
frozen core–active block.

## Options {#lagrangian-zvector-options}

| Option | Default | Effect |
|---|---|---|
| `ZvectorControls` (library) | tolerance $10^{-8}$, 50 iterations | the stop rule above |
| `FERRIC_ZVEC_TRACE` | off | prints the residual per iteration |

## Sources {#lagrangian-zvector-sources}

| Source | What it gives |
|---|---|
| `scripts/mp2_variant_identities.py`, claim 5 | the Hessian factor |
| `scripts/mp2_gradient_sympy.py`, part G | the relaxed density and the assembled gradient |

**Known limits (2026-11-15).** `solve_zvector` takes a closed-shell
reference only. The unrestricted RI-MP2 gradients solve their own coupled
two-spin Z-vector (section @sec:u-rimp2-gradient); the unrestricted OO-RI-MP2 needs none
(section @sec:oo-rimp2-gradient).
