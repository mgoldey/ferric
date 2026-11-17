# Mixed metrics and robust (Dunlap) fitting {#robust-fit}

<!-- status: ready; registry: none;
     theory: crates/ferric-integrals/theory.md#"Density fitting: the fit and its error", "Robust fitting with a mismatched metric" -->

Section @sec:factorization-and-metric fits each product density under the same kernel that the
integrals use. The fit then minimizes exactly the energy error we care about,
and that error comes out second order in the part of the density the
auxiliary basis misses. Attenuated MP2 and its relatives sometimes want to fit
under one kernel and evaluate under another: fit with a short-range erfc
kernel, whose metric couples only nearby auxiliary functions, and evaluate the
full Coulomb interaction. The obvious substitution then has a first-order
error, and for an attenuated fitting kernel that error grows without bound as
the kernels separate. Dunlap's robust form evaluates the interaction so that
the error is again the interaction of two residuals, whatever kernel produced
the fit.

The plan:

1. Derive the error of the same-kernel fit and show it is stationary (@sec:robust-fit-fit-error).
2. Derive the robust form for a mismatched kernel and show why the obvious
   substitution fails (@sec:robust-fit-robust-form).
3. Read the code that evaluates the robust form (@sec:robust-fit-algorithm).

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/rimp2.rs::{ri_mp2_robust_attenuated_metric,ri_mp2_robust_attenuated_metric_with,robust_fit_coulomb_parts,solve_metric_system,spin_components_from_g}` |
| Config | `RiMp2Config::metric_op` (the fitting kernel $\hat w$) |

Notation for this section: $\hat C$ is the physical kernel (Coulomb in the
code), $\hat w$ the fitting kernel, and $x, y$ product densities (orbital or
AO pairs).

| Symbol | Definition |
|---|---|
| $A_{Px}$ | $(P\vert\hat w\vert x)$ |
| $J_{Px}$ | $(P\vert\hat C\vert x)$ |
| $(V_w)_{PQ}$, $(V_C)_{PQ}$ | $(P\vert\hat w\vert Q)$, $(P\vert\hat C\vert Q)$ |
| $c_x$ | fit coefficients, $\tilde\rho_x = \sum_P c_{Px}\chi_P$ |
| $\delta_x$ | the residual density $\rho_x - \tilde\rho_x$ |

## The fit and its error {#robust-fit-fit-error}

The fit of section @sec:factorization-and-metric-factorization minimizes
$(\rho_x - \tilde\rho_x|\hat w|\rho_x - \tilde\rho_x)$ over the coefficients,
which gives

$$ c_x = V_w^{-1} A_x . $$

The residual $\delta_x$ is $\hat w$-orthogonal to every $\chi_P$:
$(P|\hat w|\delta_x) = A_{Px} - (V_wc_x)_P = 0$.

Suppose first that the physical kernel is the fitting kernel,
$\hat C = \hat w$. How far is the fitted interaction
$(\tilde\rho_x|\hat w|\tilde\rho_y) = A_x^{\mathsf T}V_w^{-1}A_y$ from the exact
one? Write $\rho = \tilde\rho + \delta$ in both slots of the exact integral
and expand:

$$ (x|\hat w|y) = (\tilde\rho_x|\hat w|\tilde\rho_y) + (\tilde\rho_x|\hat w|\delta_y) + (\delta_x|\hat w|\tilde\rho_y) + (\delta_x|\hat w|\delta_y) . $$

The two middle terms are sums of $(P|\hat w|\delta)$ weighted by coefficients,
and each $(P|\hat w|\delta)$ is zero by the orthogonality above. What is left
is

$$ (x|\hat w|y) - A_x^{\mathsf T}V_w^{-1}A_y = (\delta_x|\hat w|\delta_y) . $$

The error is second order in the residuals. A residual of size $\delta$
gives an integral error of size $\delta^2$.

The diagonal error is also stationary in the coefficients (Dunlap). Write it
as a function of arbitrary coefficients,
$D_x(c) = (x|\hat w|x) - 2c^{\mathsf T}A_x + c^{\mathsf T}V_wc$. This is a
quadratic with its minimum at $c_x$. Expand about the minimum,
$c = c_x + d$; the linear term $2d^{\mathsf T}(V_wc_x - A_x)$ vanishes, and

$$ D_x(c_x + d) = D_x(c_x) + d^{\mathsf T}V_w\,d . $$

So an error $d$ in the coefficients, from rounding or from an iterative
solve, moves the energy error only at second order in $d$.

Check the complete-basis limit: if the auxiliary basis spans $\rho_x$, then
$\delta_x = 0$ and the fit is exact.

**Standard RI is this formula factored.** Standard RI writes
$A_x^{\mathsf T}V^{-1}A_y$ as $B^{\mathsf T}B$ with $B = MA$ and any $M$
satisfying $M^{\mathsf T}M = V^{-1}$ (section @sec:factorization-and-metric-any-factor-gives-same). ferric's $M = L^{-1}$
from $V = LL^{\mathsf T}$ is not symmetric, so $MM \ne V^{-1}$. A formula that
is not a Gram product $B^{\mathsf T}B$ must therefore solve with $V$ instead
of applying $M$ twice. The robust form below is such a formula.

> **Verified equation {#eq:df-robust-collapse}** · with $\hat w = \hat C$ the robust form is standard RI
>
> $G = J^{\mathsf T}V_C^{-1}J = B^{\mathsf T}B$ with $B = L^{-1}J$, and the
> Cholesky factor satisfies $M^{\mathsf T}M = V^{-1}$ but $MM \ne V^{-1}$.
>
> | | |
> |---|---|
> | Proof | `scripts/robust_fit_sympy.py`, claims collapse and cholesky |
> | Negative control | $V^{-1} \to MM$ must fail (recorded mutation) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/rimp2.rs::solve_metric_system` |
>
> <!-- verified: id=df.robust.collapse; script=scripts/robust_fit_sympy.py; claim=collapse; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/rimp2.rs::solve_metric_system -->

> **Verified equation {#eq:df-robust-stationary}** · the same-kernel fitting error is quadratic and stationary
>
> With $\hat w = \hat C$ the diagonal error $(\delta_x|\hat C|\delta_x)$ has zero
> first derivative in the coefficients; with $\hat w \ne \hat C$ it does not.
>
> | | |
> |---|---|
> | Proof | `scripts/robust_fit_sympy.py`, claim stationary |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/rimp2.rs::solve_metric_system` |
>
> <!-- verified: id=df.robust.stationary; script=scripts/robust_fit_sympy.py; claim=stationary; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/rimp2.rs::solve_metric_system -->

Section @sec:closed-shell-ri-mp2 states the error identity as the unverified box
`df.quadratic-error`. Verified equation @eq:df-robust-stationary is that statement: the error
identity of @sec:robust-fit-robust-form with $\hat w = \hat C$ is the same equation, and claim
residual (Verified equation @eq:df-robust-residual) proves it for any coefficients.

## The robust form {#robust-fit-robust-form}

Now let the fitting kernel $\hat w$ differ from the physical kernel $\hat C$.
The coefficients are still $c_x = V_w^{-1}A_x$, and the residual is
$\hat w$-orthogonal to the auxiliary functions. It is not $\hat C$-orthogonal,
so the cross terms of the expansion in @sec:robust-fit-fit-error no longer vanish.

**The naive substitution.** Replace $V^{-1}$ in the standard formula by
$V_w^{-1}$ and keep the physical three-index integrals:
$J_x^{\mathsf T}V_w^{-1}J_y$. Nothing in this expression cancels the cross
terms. For an attenuated $\hat w$ it is worse: an erfc kernel weaker than
Coulomb gives a small $V_w$, so $V_w^{-1}$ is large and multiplies the full
Coulomb $J$ on both sides.

**The robust form.** Fit with $\hat w$, and evaluate the physical
interaction symmetrically: each fitted density interacts with the exact other
one, and the fit-fit term is subtracted once so that it is not counted twice.

$$ (x|\hat C|y) \approx G_{xy} = (\tilde\rho_x|\hat C|\rho_y) + (\rho_x|\hat C|\tilde\rho_y) - (\tilde\rho_x|\hat C|\tilde\rho_y) = c_x^{\mathsf T}J_y + J_x^{\mathsf T}c_y - c_x^{\mathsf T}V_C\,c_y . $$

Why this combination? Expand the residual interaction under the physical
kernel, with $\delta = \rho - \sum_P c_P\chi_P$ for any coefficients:

$$ (\delta_x|\hat C|\delta_y) = (x|\hat C|y) - c_x^{\mathsf T}J_y - J_x^{\mathsf T}c_y + c_x^{\mathsf T}V_C\,c_y . $$

The last three terms are $-G_{xy}$. So, for any coefficients $c$,

$$ (x|\hat C|y) - G_{xy} = (\delta_x|\hat C|\delta_y) . $$

The error is bilinear in the residuals, whatever kernel produced them. The
diagonal error $(\delta_x|\hat C|\delta_x)$ is $\ge 0$ because $\hat C$ is
positive definite, so the robust form never overbinds a self-interaction.

Check the limits.

- **Same kernel.** With $\hat w = \hat C$, $c_x = V_C^{-1}J_x$. Substitute:
  $G = J^{\mathsf T}V_C^{-1}J + J^{\mathsf T}V_C^{-1}J - J^{\mathsf T}V_C^{-1}J = J^{\mathsf T}V_C^{-1}J$,
  which is standard RI (Verified equation @eq:df-robust-collapse).
- **Complete auxiliary basis.** The residuals are zero for any $\hat w$, and
  $G$ is exact.
- **Coefficient error.** Apply the quadratic expansion of @sec:robust-fit-fit-error to the
  physical kernel. The diagonal error is $D_x(c)$ with $\hat C$ in place of
  $\hat w$, minimized at the $\hat C$-optimal coefficients $c^C_x$. At the
  $\hat w$ fit,
  $D_x(c^w_x) = D_x(c^C_x) + (c^w_x - c^C_x)^{\mathsf T}V_C\,(c^w_x - c^C_x)$.
  With $\hat w \ne \hat C$ the coefficients are no longer the $\hat C$-optimal
  ones, and the error is not stationary in $c$: it grows quadratically with
  the coefficient mismatch. So each diagonal element of the robust error is
  at least the standard-RI one, and equal to it only when $\hat w = \hat C$.

What does this buy on a model? `scripts/robust_fit_sympy.py` builds $s$
Gaussians on three centres with twelve auxiliary functions, fits under
$\operatorname{erfc}(\omega_m r)/r$ and evaluates under Coulomb. Its output
(largest error relative to the largest exact integral; standard RI gives
$2.19\times10^{-3}$):

| $\omega_m$ | robust | naive |
|---|---|---|
| 0.05 | $2.19\times10^{-3}$ | $1.12\times10^{-1}$ |
| 0.50 | $2.21\times10^{-3}$ | $6.47$ |
| 1.50 | $2.25\times10^{-3}$ | $5.61\times10^{1}$ |
| 3.00 | $2.25\times10^{-3}$ | $2.23\times10^{2}$ |
| 6.00 | $2.24\times10^{-3}$ | $8.88\times10^{2}$ |

The naive error passes 100% of the integral by $\omega_m = 0.5$. The robust
error stays within 3% of the standard-RI error over the whole range.

> **Verified equation {#eq:df-robust-residual}** · the robust error is the residual self-interaction
>
> $(x|\hat C|y) - G_{xy} = (\delta_x|\hat C|\delta_y)$ for any coefficients:
> symbolically for a generic bilinear form, and on a model of $s$ Gaussians on
> three centres (twelve auxiliary functions, every integral closed form)
> against residual integrals built independently, to $10^{-12}$ relative.
>
> | | |
> |---|---|
> | Proof | `scripts/robust_fit_sympy.py`, claims residual and model |
> | Negative control | dropping $-c^{\mathsf T}V_Cc$, using $V_w$ in its place, and $2c^{\mathsf T}J$ for the symmetric pair must each fail (recorded mutations) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/rimp2.rs::ri_mp2_robust_attenuated_metric` |
>
> <!-- verified: id=df.robust.residual; script=scripts/robust_fit_sympy.py; claim=model; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/rimp2.rs::ri_mp2_robust_attenuated_metric -->

> **Verified equation {#eq:df-robust-diverges}** · the naive form diverges, the robust one does not
>
> On the model, as the erfc fitting kernel's $\omega_m$ grows, the naive
> $J^{\mathsf T}V_w^{-1}J$ error grows without bound while the robust error
> stays below the exact integral and below the naive error, tends to the
> standard-RI error as $\omega_m \to 0$, and its diagonal is never negative.
>
> | | |
> |---|---|
> | Proof | `scripts/robust_fit_sympy.py`, claims diverges and psd-diag |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/rimp2.rs::ri_mp2_robust_attenuated_metric` |
>
> <!-- verified: id=df.robust.diverges; script=scripts/robust_fit_sympy.py; claim=diverges; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/rimp2.rs::ri_mp2_robust_attenuated_metric -->

The script's full claim list is residual, collapse, cholesky, stationary,
model, diverges and psd-diag.

## The algorithm {#robust-fit-algorithm}

| Step | Quantity | Kernel | Function |
|---|---|---|---|
| 1 | $J_{P,ia} = (P\vert\hat C\vert ia)$, $(V_C)_{PQ}$ | physical (`op`) | inline in `ri_mp2_robust_attenuated_metric`; `robust_fit_coulomb_parts` for a sweep |
| 2 | $A_{P,ia} = (P\vert\hat w\vert ia)$, $(V_w)_{PQ}$ | fitting (`metric_op`) | `ri_mp2_robust_attenuated_metric`, `ri_mp2_robust_attenuated_metric_with` |
| 3 | $c = V_w^{-1}A$ by Cholesky | — | `solve_metric_system` |
| 4 | $G = c^{\mathsf T}J + J^{\mathsf T}c - c^{\mathsf T}V_Cc$, an $(n_on_v)^2$ matrix | — | same two functions |
| 5 | $E_{\mathrm{OS}}$, $E_{\mathrm{SS}}$ from $G$ | — | `spin_components_from_g` |

Here are steps 2 to 5 in
`ferric-mp2/src/rimp2.rs::ri_mp2_robust_attenuated_metric`. The comment block
before the solve is removed and the line comments are added:

```rust
let mut src_a = ThreeIndexSource::build(metric_op, obs, dfbs, budget_bytes)?;
let a_ov = stream_dressed_mo_band(&mut src_a, &identity, &c_occ, &c_vir, None)?; // A = (P|w|ia), raw

let v_w = threeindex::coulomb_metric_2c(metric_op, dfbs)?;  // V_w
let coef = solve_metric_system(&v_w, &a_ov)?;                // c = V_w^{-1} A, (naux, nov)

let cj = coef.t().dot(&j_ov);                                // c^T J
let cvc = coef.t().dot(&v_c.dot(&coef));                     // c^T V_C c
let mut g = &cj + &cj.t();                                   // c^T J + J^T c
g -= &cvc;                                                   // G

Ok(spin_components_from_g(
    &g, eps, nocc, nvir, first_occ, nocc_total,
))
```

Points about it:

- **Raw tensors.** Both three-index tensors are streamed with the identity
  as the dressing matrix, so `j_ov` and `a_ov` are the raw $J$ and $A$ in the
  $ov$ block. No $M$ is applied anywhere on this path.
- **$J^{\mathsf T}c$ is the transpose of $c^{\mathsf T}J$.** The code forms
  one product and adds its transpose, so $G$ is symmetric by construction.
  The recorded mutation $2c^{\mathsf T}J$ breaks exactly this.
- **$G$ is an explicit matrix.** It has three terms and is not a Gram product,
  so the energy cannot come from a $B$ tensor.
  `ferric-mp2/src/rimp2.rs::spin_components_from_g` reads $(ia|jb)$ and
  $(ib|ja)$ straight from $G$, with the same energy expressions and the same
  $i \le j$ pair weighting as the standard path (equation @eq:mp2-cs-spin-adaptation).

**Why solve, and not apply the metric factor.**
`ferric-mp2/src/rimp2.rs::solve_metric_system` factors $V_w = LL^{\mathsf T}$
and does two triangular solves:

```rust
let l = v.cholesky(UPLO::Lower).map_err(|e| {      // V_w = L L^T
    // ...
})?;
let y = l
    .solve_triangular(UPLO::Lower, Diag::NonUnit, rhs)   // L y = A
    .map_err(|e| FerricError::Lapack(format!("metric solve L y = R failed: {e}")))?;
l.t()
    .solve_triangular(UPLO::Upper, Diag::NonUnit, &y)    // L^T c = y
    .map_err(|e| FerricError::Lapack(format!("metric solve Lᵀ c = y failed: {e}")))
```

The BLAS-thread wrapper and the body of the first error message are removed.
The code solves instead of applying a factor for two reasons. First,
`metric_inverse_sqrt` returns $L^{-1}$ for positive definite kernels, and $G$ needs $V_w^{-1}$ itself; applying $L^{-1}$ twice gives
$MM \ne V^{-1}$ (Verified equation @eq:df-robust-collapse). Second, an explicit inverse is
inaccurate here. RI and JK-fit metrics are ill-conditioned (condition number
$3\times10^{7}$ to $7\times10^{8}$ on benzene/def2-universal-jkfit for
Coulomb and erfc). `inv()` followed by a matrix product has a forward error of
about $\epsilon\cdot\mathrm{cond}$ that is not a smooth function of the right-hand
side, and a quadratic form built from it jitters by $10^{-10}$ to $10^{-8}$
relative, where the Cholesky solve stays at $10^{-14}$
(`ferric-mp2/src/rimp2.rs::metric_solve_quadratic_form_is_smooth`).

**A failed factorization stops the run.** A Cholesky failure of $V_w$ means
the attenuated metric has lost rank in a Coulomb-fitted auxiliary basis, so
that fitting kernel is unusable, and the code reports that as an error rather
than regularizing it.

**Sweeps.** The step-1 parts do not depend on $\hat w$, so a sweep over
fitting kernels builds them once with
`ferric-mp2/src/rimp2.rs::robust_fit_coulomb_parts` and passes them to
`ri_mp2_robust_attenuated_metric_with` for each $\hat w$.

**Cost.** $G$ is dense in $(n_on_v)^2$ and the path builds both the Coulomb
and the attenuated three-index tensors, so it is for measurement on small
systems, not production.

## Sources

- B. I. Dunlap, *J. Mol. Struct. THEOCHEM* **529**, 37 (2000).
- B. I. Dunlap, J. W. D. Connolly and J. R. Sabin, *J. Chem. Phys.* **71**,
  3396 (1979).
