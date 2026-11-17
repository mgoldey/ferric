# The correlation self-energy {#correlation-self-energy}

<!-- status: ready; registry: gw -->

An electron added to state $p$ polarizes the molecule, and the polarization
acts back on it. Because the polarization takes time to build, the energy
of the added electron depends on how fast the response is compared with the
energy available: the correction is a function of energy, $\Sigma^c_p(E)$.
In GW it is the frequency convolution of the propagator $G$ with the dynamic
part of $W$ from section @sec:screened-interaction.

On the real axis that convolution runs over the poles of $G$ and $W$ and is
hard to integrate. Evaluated at a complex energy $z$ off the real axis, it
becomes a smooth integral over the imaginary-frequency grid on which $W$ is
stored. That is why GW here works on the imaginary axis and continues to the
real one afterwards (section @sec:continuation-and-qp-equation).

| Symbol | Meaning | Code name |
|---|---|---|
| $z$ | complex energy, absolute (not measured from $\epsilon_F$) | `z` |
| $u_k$, $w_k$ | quadrature node and weight of the $W$ grid | `quad_freqs`, `quad_weights` |
| $W_{pq}(iu_k)$ | screened matrix element, section @sec:screened-interaction | `w_nk` |
| $\epsilon_q$ | propagator energy of state $q$ | `eps_act` (a separate argument, so evGW can replace it) |
| $\Sigma^c_p(z)$ | correlation self-energy | the return of `sigma_c_at_z` |

| Entry point | Location |
|---|---|
| Rust | `ferric-gw/src/sigma.rs::{sigma_c_at_z,sigma_c_at_z_product,run_g0w0}` |
| CLI | `method.kind = "gw"`, `[gw] method = "g0w0"` |
| Python | `run_gw(method="g0w0")` |

## The frequency integral {#correlation-self-energy-frequency-integral}

**The integral.** For any complex $z$ off the real axis the code evaluates

$$ \Sigma^c_p(z) = -\frac{1}{\pi}\sum_q \int_0^\infty du'\; W_{pq}(iu')\,\frac{z - \epsilon_q}{(z - \epsilon_q)^2 + u'^2} . $$

The convolution is an integral over the whole imaginary axis with prefactor
$\frac{1}{2\pi}$. Its integrand is even in $u'$, so the integral over
$(-\infty, \infty)$ is twice the integral over $(0, \infty)$, and
$\frac{1}{2\pi}\cdot 2 = \frac1\pi$. The integral uses the PDEP quadrature
(`pdep.quad_freqs`, `pdep.quad_weights`): scaled Gauss–Legendre nodes
$u = u_0(1+x)/(1-x)$ from `ferric-rpa/src/quadrature.rs::gauss_legendre_nodes`.

**Where the occupied and virtual states separate.** The sum over $q$ runs
over every active orbital, with no occupied/virtual distinction in the code.
The time ordering sits in the kernel. Write $a = z - \epsilon_q$. For real
$a \neq 0$,

$$ \int_0^\infty \frac{a}{a^2+u'^2}\,du' = \frac{\pi}{2}\,\mathrm{sgn}(a) , $$

so the kernel changes sign with $\mathrm{Re}(z - \epsilon_q)$. A state below
$z$ (occupied, on the continuation line of section @sec:continuation-and-qp-equation) enters with one sign
and a state above it with the other. $z$ is an absolute energy, and the
propagator energies `eps_act` are a separate argument so that evGW can
replace them (section @sec:eigenvalue-self-consistency).

**Check the single-pole model.** Feed in the $W$ of one particle–hole pair,
$W_{pq}(iu) = -2\Omega s_{pq}/(u^2+\Omega^2)$ (section @sec:screened-interaction-dielectric-matrix-single-pole). The integral
can be done by residues, and gives

$$ \Sigma^c_p(z) = \sum_{q:\,\mathrm{Re}(z-\epsilon_q)>0} \frac{s_{pq}}{z - \epsilon_q + \Omega} \;+\; \sum_{q:\,\mathrm{Re}(z-\epsilon_q)<0} \frac{s_{pq}}{z - \epsilon_q - \Omega} . $$

This is the textbook result: occupied poles at $\epsilon_i - \Omega$, virtual
poles at $\epsilon_a + \Omega$. A removed electron's energy is shifted by
coupling to states where the hole is accompanied by an excitation of energy
$\Omega$, and likewise for an added electron. On the continuation line
$z = \epsilon_F + iu$ the occupied states are exactly those with
$\mathrm{Re}(z-\epsilon_q) > 0$. The script proves the closed form
symbolically for real arguments and checks it at complex $z$ to $10^{-30}$.
The Rust test `sigma::tests::sigma_c_single_pole_w_matches_closed_form_residues`
feeds this $W$ through `sigma_c_at_z` on the 100-point grid and asserts the
closed form to $10^{-8}$ Ha at $z = \epsilon_F + i\{0, 0.1, 0.6, 2\}$. The
measured error, $2.3\times10^{-9}$ Ha, is the quadrature error of that grid.

> **Verified equation {#eq:gw-sigma-c-closed-form}** · the frequency integral against the closed-form residue result
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

**The plain rule.** Replace the integral by the quadrature sum. This is the
loop of `ferric-gw/src/sigma.rs::sigma_c_at_z`, one per state $p$, after the
screened elements have been formed by `screened_elements` (section @sec:screened-interaction-pdep-basis):

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
| $-1/\pi$ | the last line, applied after the sum over $q$ |

The factor $z - \epsilon_q$ does not depend on $k$, so the code multiplies it
in once per $q$, outside the frequency loop. A unit test
(`sigma::tests::sigma_c_blas3_matches_scalar_reference`) checks the GEMM
form of the screened elements against a scalar loop.

## Product integration {#correlation-self-energy-product-integration}

**What goes wrong with the plain rule.** On the continuation line
$z = \epsilon_F + iu$ the kernel $a/(a^2+u'^2)$, $a = z - \epsilon_q$, peaks at
$u' = u$ with width $|\epsilon_F - \epsilon_q|$. When a state sits close to
$\epsilon_F$ the ridge is narrow, and a Gauss–Legendre sum with a few dozen
nodes cannot resolve it. The per-spin mid-gap $\epsilon_{F,\sigma}$ of a UKS
radical is the extreme case: in OH/cc-pVDZ at PBE the $\beta$ HOMO and LUMO
lie 0.015 Ha either side of it. Against the grid-free U-dRPA pole sum the
plain rule misses $\Sigma^c$ at the Padé nodes by 0.11 Ha at 100 points and
0.02 Ha at 400, and the QP energies wander by 0.1 to 2 eV with $n_{\rm quad}$.
States away from the frontier suffer too: on H2O/cc-pVDZ G0W0@PBE the plain
rule puts HOMO−2 +136, −102 and −9.7 meV from the exact-$\Sigma^c$ root at
40, 100 and 200 points.

**The fix: integrate the kernel exactly.** $W_{pq}(iu')$ is smooth. The kernel
is the part that is hard to sample, and it is known in closed form. So
interpolate $W$ and integrate the kernel against the interpolant exactly.

Step 1, partial fractions in $u'$:

$$ \frac{a}{a^2+u'^2} = \frac12\left[\frac{1}{a+iu'} + \frac{1}{a-iu'}\right] . $$

Step 2, change variable to the Gauss–Legendre coordinate,
$u' = u_0(1+x)/(1-x)$ with $du' = 2u_0\,dx/(1-x)^2$. Each fraction becomes a
rational function of $x$. Expand both in partial fractions in $x$: the
$1/(1-x)$ terms cancel between the two, and what remains is exactly

$$ \frac{a}{a^2+u'^2}\,du' = \frac{i}{2}\left[\frac{1}{x-x_1} - \frac{1}{x-x_2}\right]dx, \qquad x_1 = \frac{a - iu_0}{a + iu_0}, \quad x_2 = \frac{1}{x_1} . $$

Step 3, interpolate. Let $\mathcal P$ be the degree-$(n-1)$ polynomial through
the values $W_{pq}(iu_k)$ at the $n$ Gauss–Legendre nodes $x_k$. Each piece of
the kernel is then a Cauchy-type integral. Add and subtract $\mathcal P(x_0)$:

$$ \int_{-1}^{1} \frac{\mathcal P(x)}{x-x_0}\,dx = \mathcal P(x_0)\int_{-1}^{1}\frac{dx}{x-x_0} + \int_{-1}^{1}\frac{\mathcal P(x) - \mathcal P(x_0)}{x-x_0}\,dx = \mathcal P(x_0)\ln\frac{1-x_0}{-1-x_0} + \sum_k w_k\frac{\mathcal P_k - \mathcal P(x_0)}{x_k - x_0} . $$

The first integral is a logarithm. The second integrand is a polynomial of
degree $n-2$, which the Gauss–Legendre rule integrates exactly. So the only
error left is $W$'s interpolation error.

**Check the static limit.** A frequency-independent $W$ is its own
interpolant, so the rule returns $\tfrac{\pi}{2}\,\mathrm{sgn}(\mathrm{Re}\,a)$
times $W$ exactly, at any grid. That is the COHSEX limit of section @sec:cohsex.

**The far field.** When $x_0$ is far from $[-1, 1]$, the barycentric
interpolant loses accuracy to cancellation, while the plain Gauss–Legendre sum
there has error $\rho^{-2n}$, with $\rho(x_0)$ the Bernstein-ellipse parameter
of $x_0$. The code switches to the plain sum where $n\ln\rho(x_0) > \ln 10^6$.

The loop over states in `ferric-gw/src/sigma.rs::sigma_c_at_z_product`:

<!-- excerpt: ferric-gw/src/sigma.rs::sigma_c_at_z_product -->
```rust
    for (n_idx, &eps_n) in eps_act.iter().enumerate() {
        for (k, pk) in p.iter_mut().enumerate() {
            *pk = w_nk[(n_idx, k)];
        }
        let a = z - Complex64::new(eps_n, 0.0);
        let x1 = (a - iu0) / (a + iu0);
        let x2 = Complex64::new(1.0, 0.0) / x1;
        sigma += half_i * (pq.cauchy(&p, x1) - pq.cauchy(&p, x2));
    }
    -Complex64::new(1.0 / std::f64::consts::PI, 0.0) * sigma
```

| Code | Equation |
|---|---|
| `p` | the node values $\mathcal P_k = W_{pq}(iu_k)$ for one $q$ |
| `a` | $a = z - \epsilon_q$ |
| `x1`, `x2` | $x_1$, $x_2 = 1/x_1$ |
| `pq.cauchy(&p, x0)` | $\int_{-1}^1\mathcal P/(x-x_0)\,dx$, step 3 |
| `half_i * (...)` | $\tfrac i2[\cdots]$ of step 2 |

**Measured effect.** On OH and NH2 / cc-pVDZ at PBE the node error against the
pole sum is $2\times10^{-8}$ Ha at 40 points and $2\times10^{-13}$ at 100, and
the frontier QP energies are converged to 0.005 meV at 40. On the H2O HOMO−2
case above, the product rule is within 0.004 meV from 40 points on.

**Which drivers use it.** Every driver uses `sigma_c_at_z_product` whenever
the $W$ grid is a scaled Gauss–Legendre rule: the closed-shell G0W0, evGW$_0$,
evGW and the MPI sweep through `solve_qp_for_mo`, and the open-shell drivers,
which build the rule once. The rule recovers $u_0$ from the nodes
(`ProductQuadrature::from_scaled_gauss_legendre`) and checks the rule's
moments. A non-Gauss–Legendre grid (the Chebyshev tan-map) keeps the plain
sum. The tests are `sigma::tests::product_rule_*`, `u_g0w0_uks_pbe_*` in
`crates/ferric-gw/tests/validation_gw.rs`, and `u_g0w0_uks_*` in
`crates/ferric-gw/tests/validation_gw_defaults.rs`.

> **Verified equation {#eq:gw-sigma-c-product-rule}** · the kernel identity and the product rule
>
> The kernel identity $a/(a^2+u'^2)\,du' = \tfrac i2[1/(x-x_1)-1/(x-x_2)]\,dx$
> holds for $u' = u_0(1+x)/(1-x)$, and a port of the product rule with its
> Cauchy integral reproduces the closed form of box @eq:gw-sigma-c-closed-form for a single-pole $W$
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

## The algorithm {#correlation-self-energy-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | $B$, $\Sigma^x$, $B^\xi$, $\tilde W_d(iu_k)$ | sections @sec:ri-factors-and-exchange and @sec:screened-interaction | `run_gw` |
| 2 | the Fermi level, the mid-gap $\epsilon_F$ of the active spectrum | $O(1)$ | `fermi_level` |
| 3 | per state $p$ (independent, parallel over states): $W_{pq}(iu_k)$ for all $q$, $k$ | $n_{\rm quad}$ GEMMs | `screened_elements` |
| 4 | $\Sigma^c_p$ at each continuation node $z = \epsilon_F + iu$ (section @sec:continuation-and-qp-equation) | $n_{\rm nodes}\,n_{\rm act}\,n_{\rm quad}$ | `sigma_c_at_z_product` |
| 5 | continuation and the quasiparticle root (section @sec:continuation-and-qp-equation) | $O(n_{\rm nodes})$ per Newton step | `solve_qp_for_mo` |

The per-state sweep takes a soft scratch charge from the memory pool
(`charge_qp_sweep_scratch`). When the pool refuses it, the sweep runs
serially instead. GEMM and the Newton work are scalar-sized, so the sweep is
deterministic in the thread count up to the last bits of BLAS summation order.

## Numerical anchors {#correlation-self-energy-numerical-anchors}

From `site/src/reference/validation.md` (the `gw` row and the G0W0 rows of its
anchors table), at matched settings `n_quad = 100` and `trunc_thresh = 0`:

| Quantity | Reference | Error | Test |
|---|---|---|---|
| $\Sigma^c(\epsilon_F + iu)$ at the Padé nodes, plain quadrature | PySCF `gw_ac` | $\le 8.6\times10^{-11}$ Ha | `sigma_c_imag_axis_h2o_pbe_and_hf_vs_pyscf`, `sigma_c_imag_axis_n2_pbe_vs_pyscf` in `crates/ferric-gw/tests/validation_gw.rs` |
| G0W0@HF QP energies | grid-free dRPA pole sum | $\le 1.2\times10^{-8}$ Ha | `crates/ferric-gw/tests/validation_gw.rs` |
| G0W0@PBE QP energies | grid-free dRPA pole sum | $\le 1.8\times10^{-8}$ Ha | `crates/ferric-gw/tests/validation_gw.rs` |
| G0W0@HF QP energies | PySCF `gw_ac` | $\le 1.6\times10^{-7}$ Ha | `crates/ferric-gw/tests/validation_gw.rs` |

The grade is Proven (narrow). Narrow means: states far from $\epsilon_F$ with
split spectral weight are limited by the analytic continuation (section
@sec:continuation-and-qp-equation), and the defaults (`n_quad = 40`, `trunc_thresh = 1e-4`) are coarser
than the matched settings, within 10 meV for HOMO and LUMO.

## Sources {#correlation-self-energy-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-gw/theory.md`, "Correlation self-energy on the imaginary axis" | the integral, the closed form, the product rule |
| `scripts/gw_self_energy_sympy.py`, claims 2 and 2b | the proofs |
| Ren, Rinke, Blum *et al.*, *New J. Phys.* **14**, 053020 (2012) | RI GW on the imaginary axis |
