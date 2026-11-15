# 6.3 Mixed metrics and robust (Dunlap) fitting

<!-- section: 6.3; status: ready; registry: none;
     theory: crates/ferric-integrals/theory.md#"Density fitting: the fit and its error", "Robust fitting with a mismatched metric" -->

Density fitting is exact to second order only when the fitting kernel is the
physical one. Attenuated MP2 and its relatives sometimes want to fit under one
kernel and evaluate under another. The naive substitution then has a
first-order error that grows without bound as the kernels separate; Dunlap's
robust form keeps the error bilinear in the residuals. This section derives
both and shows the code evaluates the robust form.

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/rimp2.rs::{ri_mp2_robust_attenuated_metric,ri_mp2_robust_attenuated_metric_with,robust_fit_coulomb_parts,solve_metric_system,spin_components_from_g}` |
| Config | `RiMp2Config::metric_op` (the fitting kernel $\hat w$) |

## 6.3.1 The fit and its error

<!-- include: crates/ferric-integrals/theory.md#"Density fitting: the fit and its error" -->

> **Verified equation 6.1** · with $\hat w = \hat C$ the robust form is standard RI
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

> **Verified equation 6.2** · the same-kernel fitting error is quadratic and stationary
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

Verified equation 6.2 is the statement the sample section 10.1 leaves as the
unverified box `df.quadratic-error`: the error identity of 6.3.2 with
$\hat w = \hat C$ is that equation, and claim residual below proves it for any
coefficients.

## 6.3.2 The robust form

<!-- include: crates/ferric-integrals/theory.md#"Robust fitting with a mismatched metric" -->

> **Verified equation 6.3** · the robust error is the residual self-interaction
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

> **Verified equation 6.4** · the naive form diverges, the robust one does not
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

## 6.3.3 The algorithm

| Step | Quantity | Kernel | Function |
|---|---|---|---|
| 1 | $J_{P,ia} = (P\vert\hat C\vert ia)$, $(V_C)_{PQ}$ | physical (`op`) | inline in `ri_mp2_robust_attenuated_metric`; `robust_fit_coulomb_parts` for a sweep |
| 2 | $A_{P,ia} = (P\vert\hat w\vert ia)$, $(V_w)_{PQ}$ | fitting (`metric_op`) | `ri_mp2_robust_attenuated_metric`, `ri_mp2_robust_attenuated_metric_with` |
| 3 | $c = V_w^{-1}A$ by Cholesky | — | `solve_metric_system` |
| 4 | $G = c^{\mathsf T}J + J^{\mathsf T}c - c^{\mathsf T}V_Cc$, an $(n_on_v)^2$ matrix | — | same two functions |
| 5 | $E_{\mathrm{OS}}$, $E_{\mathrm{SS}}$ from $G$ | — | `spin_components_from_g` |

The step-1 parts do not depend on $\hat w$, so a sweep over fitting kernels
builds them once (`ri_mp2_robust_attenuated_metric_with`). $G$ is dense in
$(n_on_v)^2$, so the path is for measurement on small systems, not production.
A Cholesky failure of $V_w$ stops the run.

## Sources

- B. I. Dunlap, *J. Mol. Struct. THEOCHEM* **529**, 37 (2000).
- B. I. Dunlap, J. W. D. Connolly and J. R. Sabin, *J. Chem. Phys.* **71**,
  3396 (1979).
