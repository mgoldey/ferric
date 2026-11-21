# The omega-RI fit: an attenuated metric {#omega-ri-fit}

<!-- status: verify; registry: omega-ri -->

The Coulomb-metric fit of section @sec:factorization-and-metric couples every
auxiliary function to every other one: $1/r$ has no length scale, so the metric
$V$ and the three-index integrals $(P|ia)$ are dense however far apart the
functions sit. Glasbrenner, Graf and Ochsenfeld (JCTC 16, 6856 (2020)) and,
for RPA, Lünser, Schurkus and Ochsenfeld (JCTC 13, 1647 (2017)) fit instead
under the short-range kernel $\operatorname{erfc}(\omega r)/r$. Its integrals
decay like a Gaussian with distance, which is what a sparse implementation
needs. The price is a fit that is no longer optimal for the Coulomb energy, so
its error is first order in the fitting residual.

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/omega_ri.rs::{fit_dressing,omega_ri_dressing_from_metrics,RiFit}` |
| MP2 | `ferric-mp2/src/rimp2.rs::ri_mp2_fitted`, `ferric-mp2/src/omega_ri_cdd.rs::omega_ri_cdd_mp2` |
| RPA | `ferric-rpa/src/lib.rs::{run_pdep_rpa_fitted,run_u_pdep_rpa_fitted}` |
| Config | `[ri] metric = "erfc"`, `omega_bohr` (Bohr⁻¹); Python `run_omega_ri` |

| Symbol | Definition |
|---|---|
| $V_{PQ}$ | $(P\vert r^{-1}\vert Q)$, the Coulomb metric, $V = LL^{\mathsf T}$ |
| $\tilde V_{PQ}$ | $(P\vert \operatorname{erfc}(\omega r)/r\vert Q)$ |
| $A_{P,ia}$ | $(P\vert \operatorname{erfc}(\omega r)/r\vert ia)$ |
| $c_{ia}$ | fit coefficients, $\tilde\rho_{ia} = \sum_P c_{P,ia}\chi_P$ |
| $\delta_{ia}$ | the residual $\rho_{ia} - \tilde\rho_{ia}$ |

## The fit and its dressing {#omega-ri-fit-dressing}

Minimize the residual in the erfc norm. The normal equations are
$(\delta_{ia}|\operatorname{erfc}(\omega r)/r|P) = 0$ for every $P$, so

$$
c_{ia} = \tilde V^{-1} A_{ia}.
$$

Evaluate the Coulomb interaction of two fitted densities:

$$
(ia|jb) \approx (\tilde\rho_{ia}|\tilde\rho_{jb})
= A_{ia}^{\mathsf T}\,\tilde V^{-1} V \tilde V^{-1}\,A_{jb}.
\label{eq:omega-ri-four-index}
$$

The metric product matches the paper's eq 15-18 and Lünser's eq 29. Write
$V = LL^{\mathsf T}$ and define the dressing $D = L^{\mathsf T}\tilde V^{-1}$.
Then $D^{\mathsf T} D = \tilde V^{-1} L L^{\mathsf T} \tilde V^{-1}$, which is the
metric product, so with $B_{ia} = D A_{ia}$

$$
(ia|jb) \approx B_{ia}^{\mathsf T} B_{jb}.
\label{eq:omega-ri-gram}
$$

The approximate four-index matrix is a Gram matrix, so it is positive
semidefinite like the exact one. Every method downstream (canonical MP2, the
Laplace pseudo-orbital MP2 of section @sec:omega-ri-cdd-mp2, the PDEP-RPA of
section @sec:pdep) reads only $B$, and that is how the code shares one fit.

Check the limit $\omega \to 0$. Then $\operatorname{erfc}(\omega r) \to 1$,
$\tilde V \to V$ and $A_{ia} \to (P|ia)$, so $D \to L^{\mathsf T} V^{-1} =
L^{-1}$: the Coulomb dressing of section @sec:factorization-and-metric.

Check $\omega \to \infty$. The kernel integrates to
$\int \operatorname{erfc}(\omega r)/r\, d^3 r = \pi/\omega^2$ and its next
moment is $O(\omega^{-4})$, so it tends to $(\pi/\omega^2)\,\delta(\mathbf r)$.
The scalar $\pi/\omega^2$ appears twice in $\tilde V^{-1}$ and twice in $A$,
and cancels in (@eq:omega-ri-four-index). The limit is the overlap-metric fit.

The metric exists for every $\omega > 0$: the Fourier transform of the kernel is

$$
\frac{4\pi}{k^2}\left(1 - e^{-k^2/4\omega^2}\right) > 0,
$$

so $\tilde V$ is positive definite on any linearly independent auxiliary set.

## The error is first order {#omega-ri-fit-error-order}

Write $\rho = \tilde\rho + \delta$ in the exact integral and expand:

$$
(ia|jb) - (\tilde\rho_{ia}|\tilde\rho_{jb})
= (\delta_{ia}|\tilde\rho_{jb}) + (\tilde\rho_{ia}|\delta_{jb}) + (\delta_{ia}|\delta_{jb}).
$$

The Coulomb fit satisfies $(\delta_{ia}|P) = 0$ for every $P$ (Galerkin), so
the two cross terms vanish and the error is the residual-residual interaction,
second order. The erfc fit satisfies $(\delta_{ia}|\operatorname{erfc}(\omega
r)/r|P) = 0$ instead. The Coulomb cross terms survive, and the error is first
order in $\delta$. Section @sec:robust-fit derives the robust (Dunlap) form that
restores second order for any fitting kernel; the omega-RI method of the papers
uses the plain form (@eq:omega-ri-four-index), and so does this code.

The fitted charge is a symptom of the same fact. Neither fit conserves the
charge of a pair density: for water/cc-pVDZ with cc-pVDZ-RI the largest fitted
occupied-virtual pair charge (true value 0) is $4.96\times10^{-3}$ with the
Coulomb metric and $7.74\times10^{-3}$ with $\omega = 0.1$ Bohr⁻¹. The Coulomb
fit's leaked monopole sits inside $\delta$, which is Coulomb-orthogonal to the
auxiliary set, so it reaches the energy only through $(\delta|\delta)$. The
erfc fit's residual is not: $|(\delta|P)|/|(ia|P)|$ is $1.5\times10^{-15}$ for
the Coulomb fit and $8.9\times10^{-3}$ for the erfc fit at $\omega = 0.1$.

> **Verified equation {#eq:omega-ri-error-order}** · the omega-RI fit error order
>
> The Coulomb-fit error is second order in the residual scale, the erfc-fit
> error first order, the robust form second order, on a symbolic expansion and
> on a random Gram model (log-log slopes 2, 1, 2).
>
> | | |
> |---|---|
> | Proof | `scripts/omega_ri_sympy.py`, claim 4 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/omega_ri.rs::fit_dressing` |
>
> <!-- verified: id=df.omega-ri.order; script=scripts/omega_ri_sympy.py; claim=4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/omega_ri.rs::fit_dressing -->

What does this cost in energy? For water/cc-pVDZ (`examples/water-omega-ri-mp2.toml`)
the RI-MP2 correlation energy is $-0.2040252384$ Eh with the Coulomb fit and
$-0.2040233778$ Eh at $\omega = 0.1$ Bohr⁻¹, a shift of $+1.86\times10^{-6}$ Eh.
The papers recommend $\omega = 0.1$ Bohr⁻¹.

## The code {#omega-ri-fit-code}

`ferric-mp2/src/omega_ri.rs::omega_ri_dressing_from_metrics` builds $D$ from the
two metrics (error mapping trimmed):

```rust
let l = with_blas_threads(opt_in_blas_threads(), || v.cholesky(UPLO::Lower)) // V = L L^T
// W = V~^-1 L  (naux x naux), then D = Wᵀ.
let w = crate::rimp2::solve_metric_system(v_w, &l)?;                          // W = V~^-1 L
Ok(w.t().as_standard_layout().into_owned())                                   // D = W^T = L^T V~^-1
```

- $D$ multiplies the erfc three-index tensor from the left, exactly where the
  Coulomb path applies $L^{-1}$; `fit_dressing` returns the operator the
  three-index source must use (`Operator::erfc(omega)`) with $D$, so the two can
  never be mixed.
- $\omega = 0$ goes through the same code with Coulomb integrals and returns
  $L^{\mathsf T} V^{-1} = L^{-1}$ up to rounding.
- The physical operator must be the bare Coulomb operator. An attenuated
  physical kernel combined with the omega-RI fit is refused.

## Design choices {#omega-ri-fit-design}

The fit is a parameter of the existing builders
(`ferric-mp2/src/rimp2.rs::compute_rpa_intermediates_fitted`), not a new
pipeline: canonical MP2, the Laplace path and PDEP-RPA receive the dressed $B$
and run unchanged. The algebra after $B$ stays dense, so in this code the fit
changes the energy and leaves the cost where it is; the sparsity the erfc
integrals offer is not exploited.

## What is proven and what is not {#omega-ri-fit-proven}

> **Verified equation {#eq:omega-ri-dressing}** · the coded dressing
>
> $D^{\mathsf T}D = \tilde V^{-1} V \tilde V^{-1}$, $D(\omega = 0) = L^{-1}$, and
> the fitted four-index matrix is positive semidefinite.
>
> | | |
> |---|---|
> | Proof | `scripts/omega_ri_sympy.py`, claim 5 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/omega_ri.rs::omega_ri_dressing_from_metrics` |
>
> <!-- verified: id=df.omega-ri.dressing; script=scripts/omega_ri_sympy.py; claim=5; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/omega_ri.rs::omega_ri_dressing_from_metrics -->

> **Verified equation {#eq:omega-ri-kernel-positive}** · the erfc kernel is positive definite
>
> $\mathcal F[\operatorname{erfc}(\omega r)/r] = 4\pi k^{-2}(1 - e^{-k^2/4\omega^2}) > 0$,
> symbolically and by quadrature.
>
> | | |
> |---|---|
> | Proof | `scripts/omega_ri_sympy.py`, claim 1 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/omega_ri.rs::fit_dressing` |
>
> <!-- verified: id=df.omega-ri.kernel; script=scripts/omega_ri_sympy.py; claim=1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/omega_ri.rs::fit_dressing -->

The script also checks both limits on PySCF integrals for water (at
$\omega = 10^{-5}$ the energy equals the Coulomb fit to $2.6\times10^{-11}$
Eh; the error against the overlap-metric fit falls as $\omega^{-2}$). The
Rust anchors (`crates/ferric-mp2/tests/omega_ri_anchors.rs`,
`crates/ferric-rpa/tests/omega_ri_rpa.rs`) compare the shift
$E(\omega) - E(\text{Coulomb})$ for MP2 and RPA against a numpy assembly of
(@eq:omega-ri-four-index) by explicit linear solves on PySCF integrals
(`scripts/validation/gen_omega_ri.py`), for water and ammonia at
$\omega = 0.05$ to $0.5$ Bohr⁻¹. No energy gradient exists for the omega-RI
fit.
