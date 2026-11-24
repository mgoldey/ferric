# The mask certificate {#lmp2-mask-certificate}

<!-- status: edit; registry: local-mp2 -->

*Notation follows `book/notation.md` and section @sec:amplitude-lmp2.
$J$ is the integral vector $(ia|jb)$, $\mathcal A$ the Fock superoperator,
$\mathcal S$ the $a\leftrightarrow b$ swap within a pair and
$\mathcal W = 2-\mathcal S$. $K$ is the set of kept amplitudes and $D$ the
dropped set.*

A local MP2 run drops amplitudes it judges negligible and reports the
energy of the rest. How much correlation energy did the mask lose? The
exact answer is the difference to an unmasked solve, which is the
calculation the mask was meant to avoid. The measured records of section
@sec:amplitude-lmp2-fock-screen-measured answer the question on average,
for the molecules they were measured on. This section computes an answer
for the run in hand: two rigorous bounds and an estimate, from quantities
that exist after the masked solve, at a cost of the order of the solve
itself (section @sec:lmp2-mask-certificate-measured).

The plan:

1. Write the mask error as one quadratic form in the residual of the masked
   solution, valid for any amplitude vector.
2. Bound that form from above with one step of Gauss–Radau quadrature, and
   from below with a line search.
3. Make both cheap: evaluate the residual on a near set of dropped
   amplitudes, and bound the rest by norms alone.

## The error is a quadratic form in the residual {#lmp2-mask-certificate-identity}

The masked solve minimizes the Hylleraas functional of equation
@eq:lmp2-hylleraas over amplitudes supported on $K$. Write $t$ for the
amplitudes it returns, $E_K = E_H(t)$ for their energy and $E^*$ for the
minimum over all amplitudes. Let

$$ R = J + \mathcal A t $$

be the full residual, on $K$ and on $D$. On $K$ it is the PCG residual of
section @sec:amplitude-lmp2-accurate-solve. On $D$ it is the drive the
dropped amplitudes would feel: their integrals plus the Fock coupling of
the kept amplitudes into them.

The minimizer is $t^* = -\mathcal A^{-1}J$. Put $t = t^* + d$, so that
$R = \mathcal A d$ and $d = \mathcal A^{-1}R$. Expand $E_H$ around $t^*$ as
in section @sec:amplitude-lmp2-accurate-solve: the linear term vanishes at
the stationary point, and

$$ E_K - E^* = \langle \mathcal W\, d,\ \mathcal A\, d\rangle = \langle \mathcal W R,\ \mathcal A^{-1} R\rangle . \label{eq:lmp2-mask-certificate-identity} $$

The second form uses $[\mathcal W,\mathcal A] = 0$: $\mathcal A$ acts on
each index separately, so it commutes with the swap. No property of $t$
was used, so the identity holds for an inexact solve too. Both errors a
local run makes are special cases:

- With $R_K = 0$ (an exact masked solve), the identity reduces to the
  truncation error (@eq:lmp2-combined-truncation-error), since
  $(\mathcal A^{-1})_{DD}$ is the inverse of the Schur complement.
- With $D$ empty, it is the solver error of (@eq:lmp2-cg-bound).

A bound on (@eq:lmp2-mask-certificate-identity) therefore bounds the mask
error and the solver error together. Both are non-negative, and
$\mathcal W$ has eigenvalues 1 and 3, so $\langle\mathcal W v, v\rangle$
is a norm, written $\lVert v\rVert_{\mathcal W}^2$ below.

**What $E^*$ is.** $E^*$ is the minimum for the same $J$ and the same
$\mathcal A$. On the in-core path $J = \tilde B^{\mathsf T}\tilde B$ is
the canonical RI integral in the localized basis, so $E^*$ is the
canonical RI-MP2 correlation energy. The certificate does not see the RI
fit error against exact four-index integrals. It cannot see the
integral-direct path's map errors either, because those change $J$
itself; that path refuses the certificate.

> **Verified equation @eq:lmp2-mask-certificate-identity** · the mask error is a quadratic form in the full residual, for any amplitudes
>
> $E_K - E^* = R_D^{\mathsf T}\Sigma^{-1}R_D$ with $\Sigma$ the Schur
> complement, in symbols on a 2+1 block model (C1);
> $(\mathcal A^{-1})_{DD} = \Sigma^{-1}$, so the full $R$ may be used (C2);
> $[\mathcal W,\mathcal A] = 0$ and $E_H(t) - E^* = \langle\mathcal W R(t),\mathcal A^{-1}R(t)\rangle$
> for a symbolic $t$, and numerically on random swap-closed masks with
> the LMP2 block structure (C3).
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_mask_certificate_sympy.py`, claims C1, C2, C3 |
> | Negative control | the runner drops $\mathcal A_{DK}t_K$ from $R_D$ ("C1"), uses $\mathcal A_{DD}$ in place of the Schur complement ("C2") and sets $\mathcal W = 1$ ("C3"); each turns the script red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/lmp2_mask_certificate.rs::mask_certificate` |
> | Code test | `bounds_enclose_the_true_mask_error_butane` in `crates/ferric-mp2/tests/lmp2_mask_certificate.rs` |
>
> <!-- verified: id=lmp2.mask-certificate.identity; script=scripts/lmp2_mask_certificate_sympy.py; claim=C3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/lmp2_mask_certificate.rs::mask_certificate; rust_test=ferric-mp2/tests/lmp2_mask_certificate.rs::bounds_enclose_the_true_mask_error_butane; registry=local-mp2 -->

## An upper bound from one quadrature step {#lmp2-mask-certificate-radau}

For a vector $v$, the form $\langle\mathcal W v,\mathcal A^{-1}v\rangle$ is
an integral of $1/\lambda$ against the spectral measure of $\mathcal A$
weighted by $v$. Its first moments are cheap:

$$ \mu_0 = \lVert v\rVert_{\mathcal W}^2, \qquad
\mu_0\,\alpha = \langle \mathcal W v,\ \mathcal A v\rangle, \qquad
\nu = \lVert \mathcal A v\rVert_{\mathcal W}^2 . $$

Gauss–Radau quadrature with one free node and one node fixed at
$a \le \lambda_{\min}(\mathcal A)$ overestimates the integral of
$1/\lambda$. Its error is the third derivative of $1/\lambda$, which is
negative, times an integral of $(\lambda - a)$ times a square, which is
non-negative. One Lanczos step from $v$ gives
$\beta^2 = \nu/\mu_0 - \alpha^2$, and the Radau-modified Jacobi matrix
gives

$$ \omega = a + \frac{\beta^2}{\alpha - a}, \qquad
\langle\mathcal W v,\mathcal A^{-1}v\rangle \;\le\; U_1(v) = \frac{\mu_0\,\omega}{\alpha\,\omega - \beta^2} \;\le\; \frac{\mu_0}{a} . \label{eq:lmp2-mask-certificate-radau} $$

Check the limit: if $v$ is an eigenvector, $\beta = 0$, and
$U_1 = \mu_0/\alpha$ is exact.

**The node.** $\lambda_{\min}(\mathcal A) = 2(\epsilon_{\mathrm{LUMO}} -
\epsilon_{\mathrm{HOMO}})$, from the eigenvalues of the localized `f_oo`
and `f_vv` blocks (section @sec:amplitude-lmp2-accurate-solve). The
localized basis is a rotation of the canonical one, so these are the
canonical orbital energies. The gap of the local Fock *diagonal*,
$2(\min_a F_{aa} - \max_i f_{ii})$, is larger: a diagonal element lies
inside the spectrum. Using it would place the node above $\lambda_{\min}$,
and the quadrature would no longer bound from above. On butane and benzene
in 6-31G it is 1.9 to 2.0 times the canonical gap, and 1.6 times on
water.

The code evaluates (@eq:lmp2-mask-certificate-radau) as follows. The
excerpt is copied unchanged:

<!-- excerpt: ferric-mp2/src/lmp2_mask_certificate.rs::radau_one_step -->
```rust
let crude = mu0 / a;
let alpha = alpha_raw / mu0;
let beta2 = (nu / mu0 - alpha * alpha).max(0.0);
if alpha - a <= 1e-14 * alpha.abs() {
    return crude;
}
let omega = a + beta2 / (alpha - a);
let det = alpha * omega - beta2;
if det <= 0.0 {
    return crude;
}
(mu0 * omega / det).min(crude)
```

| Code | Equation |
|---|---|
| `mu0`, `alpha_raw`, `nu` | $\mu_0$, $\mu_0\alpha$, $\nu$ |
| `a` | the node $a = \lambda_{\min}$ |
| `crude` | the crude bound $\mu_0/a$ |
| `beta2`, `omega`, `det` | $\beta^2$, $\omega$, $\alpha\omega-\beta^2$ |

Rounding can push $\beta^2$ slightly negative or the determinant to zero;
both guards fall back to the crude bound, which is always valid.

> **Verified equation @eq:lmp2-mask-certificate-radau** · one-step Gauss–Radau bounds the quadratic form from above
>
> The Lanczos form the code evaluates equals the moment form
> $(\mu_0,\mu_1,\mu_2)$ in symbols; on random SPD operators the bound holds
> for every node $a \le \lambda_{\min}$ (C5).
> $\lambda_{\min}(\mathcal A) = 2(\epsilon_{\mathrm{LUMO}}-\epsilon_{\mathrm{HOMO}})$ in
> every rotated basis, and the local Fock diagonal gap exceeds it (C4).
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_mask_certificate_sympy.py`, claims C4, C5 |
> | Negative control | the runner uses the local Fock diagonal gap as $\lambda_{\min}$ ("C4"), puts the node at $1.5\,\lambda_{\min}$ ("C5") and drops $\beta^2$ from the Lanczos form ("C5: lanczos") |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/lmp2_mask_certificate.rs::radau_one_step`, `ferric-mp2/src/lmp2_amplitude.rs::operator_lambda_min_lower_bound` |
> | Code test | `radau_one_step_is_exact_on_two_levels_and_an_upper_bound_below` (unit test); the node equals the canonical RHF gap in `crates/ferric-mp2/tests/lmp2_mask_certificate.rs` |
>
> <!-- verified: id=lmp2.mask-certificate.radau; script=scripts/lmp2_mask_certificate_sympy.py; claim=C5; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/lmp2_mask_certificate.rs::radau_one_step,ferric-mp2/src/lmp2_amplitude.rs::operator_lambda_min_lower_bound; rust_test=ferric-mp2/tests/lmp2_mask_certificate.rs::eps_zero_certificate_is_exactly_zero; registry=local-mp2 -->

## A lower bound from a line search {#lmp2-mask-certificate-line-search}

Any amplitude vector has a Hylleraas energy at or above $E^*$. Step from
$t$ along a direction $d$ supported on $D$, and minimize along the line:

$$ E_H(t + s\,d) = E_K + 2s\,\langle\mathcal W d, R\rangle + s^2\,\langle\mathcal W d,\mathcal A d\rangle , \qquad
\min_s E_H(t+s\,d) = E_K - \frac{x^2}{y} , $$

with $x = \langle\mathcal W d, R\rangle$ and $y = \langle\mathcal W d,\mathcal A d\rangle$.
The minimum is still at or above $E^*$, so

$$ L = \frac{x^2}{y} \;\le\; E_K - E^* . \label{eq:lmp2-mask-certificate-lower} $$

The code takes the Jacobi direction $d = \Delta^{-1}R_N$ on the near set
$N \subseteq D$ of the next subsection, with $\Delta$ the local Fock
diagonal. Then

$$ x = \sum_{e\in N} \frac{R_e\,(\mathcal W R)_e}{\Delta_e} , $$

the Kapuy-form estimate of the dropped energy with the actual residual in
place of the dressed integral of section @sec:amplitude-lmp2-combined-p2.
The code reports $x$ as the point estimate. No bound is claimed for it,
and $2x$, which looks like a cheaper lower bound, is not one: the negative
control below turns the proof red.

> **Verified equation @eq:lmp2-mask-certificate-lower** · the line-search decrease is a rigorous lower bound
>
> $E_H(t) - \min_s E_H(t+s\,d) = \langle\mathcal W d,R\rangle^2/\langle\mathcal W d,\mathcal A d\rangle$
> in symbols, and $L \le E_K - E^*$ for random directions on random
> models (C6).
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_mask_certificate_sympy.py`, claim C6 |
> | Negative control | the runner uses $2x$ as the lower bound, which exceeds the error ("C6") |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/lmp2_mask_certificate.rs::mask_certificate` |
> | Code test | `bounds_enclose_the_true_mask_error_benzene` in `crates/ferric-mp2/tests/lmp2_mask_certificate.rs` |
>
> <!-- verified: id=lmp2.mask-certificate.lower; script=scripts/lmp2_mask_certificate_sympy.py; claim=C6; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/lmp2_mask_certificate.rs::mask_certificate; rust_test=ferric-mp2/tests/lmp2_mask_certificate.rs::bounds_enclose_the_true_mask_error_benzene; registry=local-mp2 -->

## Making it cheap: the near and far sets {#lmp2-mask-certificate-near-far}

$R_D$ has an entry for every dropped amplitude, and its integral part
$J_D$ is exactly the set of integrals the mask avoided. Evaluating all of
it would cost as much as a canonical calculation. The bounds do not need
all of it.

Split $R_D = J_D + H_D$, with $H = \mathcal A t$ the Fock product of the
kept amplitudes. $H$ is cheap: the code forms it once per pair. Define the
near set from $H$ with a threshold $\tau$ ($= \varepsilon$ by default),

$$ N = \{ e \in D : \lvert H_e\rvert > \tau \ \text{or}\ \lvert H_{\mathcal S e}\rvert > \tau \}, \qquad F = D \setminus N . $$

$N$ is closed under the swap, so the line search of
(@eq:lmp2-mask-certificate-lower) stays inside a swap-closed space. On
$K \cup N$ the code evaluates $R$ in full, with $J_N$ from a sub-GEMM of
the whitened three-index tensor. Call that vector $v$. On $F$ it uses only
norms:

- $\lVert H_F\rVert_{\mathcal W}$ directly.
- $\lVert J_F\rVert$ from the Gram identity
  $\lVert J\rVert^2 = \lVert\tilde B^{\mathsf T}\tilde B\rVert_F^2 = \lVert\tilde B\tilde B^{\mathsf T}\rVert_F^2$,
  an $n_{\mathrm{aux}}\times n_{\mathrm{aux}}$ product:
  $\lVert J_F\rVert^2 = \lVert\tilde B\tilde B^{\mathsf T}\rVert_F^2 - \lVert J_K\rVert^2 - \lVert J_N\rVert^2$,
  plus a relative rounding allowance of $10^{-13}\lVert J\rVert^2$.

Since $R = v + R_F$ and the form of (@eq:lmp2-mask-certificate-identity)
is an inner product, Cauchy–Schwarz gives

$$ E_K - E^* \;\le\; U = \Big(\sqrt{U_1(v)} + \sqrt{U_F}\Big)^2, \qquad
U_F = \frac{\big(\sqrt3\,\lVert J_F\rVert + \lVert H_F\rVert_{\mathcal W}\big)^2}{\lambda_{\min}} . \label{eq:lmp2-mask-certificate-split} $$

$U_F$ uses $\lVert J_F\rVert_{\mathcal W} \le \sqrt3\,\lVert J_F\rVert$
and $\lVert\mathcal A^{-1}\rVert = 1/\lambda_{\min}$.

Check the limits:

- $\varepsilon = 0$: $D$ is empty, so $N$ and $F$ are empty, $x = L = 0$
  exactly, $U_F = 0$, and $U = U_1(R_K)$ bounds the solver error alone.
- $\tau = \infty$: $N$ is empty, $v = R_K$, and the far term covers all
  of $D$. The bound stays rigorous for every $\tau$; $\tau$ trades cost
  against tightness.

The assembly in the code, copied unchanged:

<!-- excerpt: ferric-mp2/src/lmp2_mask_certificate.rs::mask_certificate -->
```rust
let near_upper = radau_one_step(s.mu0, s2.alpha, s2.nu, lmin);
let (g_far_sq, far_upper) = if s.n_far == 0 {
    (0.0, 0.0)
} else {
    let gram = inp.btilde.dot(&inp.btilde.t());
    let g_tot: f64 = gram.iter().map(|x| x * x).sum();
    let g_far_sq = (g_tot - s.g_p2 - s.g_n2).max(0.0) + GRAM_ROUNDING_REL * g_tot;
    let h_far = s.h_far_m2.max(0.0).sqrt();
    let far = (3f64.sqrt() * g_far_sq.sqrt() + h_far).powi(2) / lmin;
    (g_far_sq, far)
};
let upper = (near_upper.sqrt() + far_upper.sqrt()).powi(2);
```

| Code | Equation |
|---|---|
| `near_upper` | $U_1(v)$ of (@eq:lmp2-mask-certificate-radau) |
| `gram`, `g_tot` | $\tilde B\tilde B^{\mathsf T}$ and $\lVert J\rVert^2$ |
| `s.g_p2`, `s.g_n2` | $\lVert J_K\rVert^2$, $\lVert J_N\rVert^2$ |
| `h_far` | $\lVert H_F\rVert_{\mathcal W}$ |
| `far_upper`, `upper` | $U_F$ and $U$ of (@eq:lmp2-mask-certificate-split) |

Points about the construction:

- **The unit of work is an output pair $(i,j)$.** For each pair the code
  forms the dense $n_v\times n_v$ block of $H$, classifies every element
  as kept, near or far, and accumulates $\mu_0$, $\lVert J_K\rVert^2$,
  $\lVert J_N\rVert^2$ and $\lVert H_F\rVert_{\mathcal W}^2$. A second pass
  applies $\mathcal A$ to $v$ and to $d$ for $\alpha$, $\nu$ and $y$. Pairs
  run in parallel; nothing of size $(n_o n_v)^2$ is stored.
- **$v$ includes $R_K$.** The PCG residual on the kept set enters the
  Radau vector, so $U$ covers the solver error and the mask error
  together, as (@eq:lmp2-mask-certificate-identity) requires.
- **Why the Gram identity.** $\lVert J_F\rVert$ is a sum over the far
  integrals, which the run never forms. The identity gives it from one
  $n_{\mathrm{aux}}^2$ matrix.

> **Verified equation @eq:lmp2-mask-certificate-split** · the near/far route encloses the error
>
> The Gram identity and the split bound $(\sqrt{e_1}+\sqrt{e_2})^2$, with a
> case where the cross term is needed (C7). The cheap route, transcribed
> from the code, gives $L \le E_K - E^* \le U$ on random masked models with
> an inexact $t_K$, and exactly zero estimate, lower bound and far term
> when $D$ is empty (C8).
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_mask_certificate_sympy.py`, claims C7, C8 |
> | Negative control | the runner drops the cross term ("C7"), drops $\lVert J_F\rVert$ from $U_F$ and drops $H$ from $R_D$ ("C8: upper") |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/lmp2_mask_certificate.rs::mask_certificate` |
> | Code test | `far_term_alone_encloses_the_true_mask_error_butane` in `crates/ferric-mp2/tests/lmp2_mask_certificate.rs` |
>
> <!-- verified: id=lmp2.mask-certificate.split; script=scripts/lmp2_mask_certificate_sympy.py; claim=C8: upper; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/lmp2_mask_certificate.rs::mask_certificate; rust_test=ferric-mp2/tests/lmp2_mask_certificate.rs::far_term_alone_encloses_the_true_mask_error_butane; registry=local-mp2 -->

## Using it {#lmp2-mask-certificate-usage}

The certificate is opt-in. In the CLI, under `method.kind = "rimp2"` (or
`att-rimp2`) with `[local] scheme = "amplitude-threshold"`:

```toml
[local]
scheme = "amplitude-threshold"
eps = 1e-5
mask_certificate = true
```

In Python, `run_rimp2(..., local="amplitude-threshold", eps=1e-5,
mask_certificate=True)`, and the result is `result.local["mask_certificate"]`.
The library flag is `AmplitudeLmp2Config::mask_certificate`, and
`mask_certificate_tau` overrides $\tau$ (library only). The run reports
`lower_bound` $L$, `upper_bound` $U$, `error_estimate` $x$, the interval
$[E_K - U,\ E_K - L]$ on $E^*$, `e_corr_corrected` $= E_K - x$, the counts
of $K$, $N$ and $F$, and its own wall time `t_s`.

It is refused in three configurations, each because $J$ would differ from
the canonical RI integrals: `integral_direct = true`, `fit_radius_bohr`
and `aux_tail_frac`.

## Measured {#lmp2-mask-certificate-measured}

Measured with `measure_mask_certificate_octane_naphthalene` in
`crates/ferric-mp2/tests/lmp2_mask_certificate.rs` (an ignored test that
prints these rows), at commit `0b59822ce`, 2026-11-19. Octane
(`alkane_8.xyz`, 8 frozen core orbitals) and naphthalene
(`scaling/naphthalene.xyz`, 10 frozen), 6-31G with cc-pVDZ-RI, RHF
converged to $10^{-10}$ Ha. Each row is one run of the in-core path. The
error is $E_K$ minus the $\varepsilon = 0$ solve on the same localized
spaces; "Eq 8" is the plain integral mask, "Fock" the default Fock screen
($\kappa = 1$); $\lambda$ is the λ-Boys weight (0 is plain Boys, 10 the
default). $\tau = \varepsilon$. N/K is the near-set size over the kept
count. Times are wall seconds from the test profile (opt-level 2),
`OPENBLAS_NUM_THREADS=1`, the default rayon pool, one run each under
`box-lease run`, so they are ratios on one machine and nothing more.

The $\varepsilon = 0$ anchor of each series:

| system | λ | $E_K - E_{\mathrm{RI}}$ at ε = 0 (Ha) | U at ε = 0 (Ha) | PCG bound (Ha) |
|---|---|---|---|---|
| octane | 0 | 6.27e-13 | 5.44e-13 | 2.71e-12 |
| octane | 10 | 7.30e-13 | 5.58e-13 | 2.77e-12 |
| naphthalene | 0 | 3.73e-13 | 6.06e-13 | 3.45e-12 |
| naphthalene | 10 | 3.27e-13 | 2.72e-13 | 1.62e-12 |

Octane:

| λ | mask | ε | keep | error (Ha) | x/err | L/err | U/err | N/K | near bound | far bound | t assembly (s) | t solve (s) | t cert (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Eq 8 | 1e-3 | 0.013 | 2.753e-2 | 0.052 | 0.052 | 5.57 | 0.036 | 1.60e-3 | 1.24e-1 | 0.75 | 0.10 | 0.15 |
| 0 | Eq 8 | 1e-4 | 0.112 | 1.912e-3 | 0.339 | 0.340 | 5.14 | 0.083 | 7.32e-4 | 5.20e-3 | 0.71 | 0.73 | 0.59 |
| 0 | Eq 8 | 1e-5 | 0.390 | 8.635e-5 | 0.705 | 0.710 | 3.62 | 0.093 | 6.86e-5 | 8.85e-5 | 0.71 | 2.38 | 1.50 |
| 0 | Eq 8 | 1e-6 | 0.746 | 2.136e-6 | 0.891 | 0.894 | 2.23 | 0.068 | 2.14e-6 | 5.19e-7 | 0.79 | 2.96 | 2.32 |
| 0 | Eq 8 | 1e-7 | 0.931 | 6.744e-8 | 0.951 | 0.951 | 1.35 | 0.024 | 7.15e-8 | 1.16e-9 | 0.74 | 2.99 | 2.48 |
| 0 | Fock | 1e-3 | 0.016 | 2.213e-2 | 0.005 | 0.005 | 5.26 | 0.008 | 1.20e-4 | 1.09e-1 | 0.77 | 0.13 | 0.15 |
| 0 | Fock | 1e-4 | 0.133 | 9.995e-4 | 0.100 | 0.100 | 5.61 | 0.028 | 1.12e-4 | 4.14e-3 | 0.71 | 0.87 | 0.66 |
| 0 | Fock | 1e-5 | 0.440 | 2.092e-5 | 0.258 | 0.259 | 5.49 | 0.033 | 6.09e-6 | 6.81e-5 | 0.77 | 2.49 | 1.56 |
| 0 | Fock | 1e-6 | 0.792 | 1.953e-7 | 0.542 | 0.543 | 4.82 | 0.026 | 1.19e-7 | 3.91e-7 | 0.75 | 2.93 | 2.18 |
| 0 | Fock | 1e-7 | 0.949 | 1.062e-9 | 0.790 | 0.789 | 3.40 | 0.009 | 9.39e-10 | 8.68e-10 | 0.74 | 2.92 | 2.54 |
| 10 | Eq 8 | 1e-3 | 0.013 | 2.773e-2 | 0.051 | 0.051 | 5.55 | 0.035 | 1.57e-3 | 1.24e-1 | 0.80 | 0.10 | 0.15 |
| 10 | Eq 8 | 1e-4 | 0.113 | 1.913e-3 | 0.337 | 0.337 | 5.14 | 0.083 | 7.27e-4 | 5.20e-3 | 0.73 | 0.74 | 0.55 |
| 10 | Eq 8 | 1e-5 | 0.391 | 7.788e-5 | 0.677 | 0.681 | 3.77 | 0.093 | 5.95e-5 | 8.90e-5 | 0.74 | 2.62 | 1.54 |
| 10 | Eq 8 | 1e-6 | 0.746 | 3.017e-6 | 0.912 | 0.914 | 2.04 | 0.067 | 3.11e-6 | 5.18e-7 | 0.82 | 2.98 | 2.26 |
| 10 | Eq 8 | 1e-7 | 0.932 | 1.332e-7 | 0.943 | 0.943 | 1.25 | 0.024 | 1.40e-7 | 1.16e-9 | 0.74 | 2.92 | 2.62 |
| 10 | Fock | 1e-3 | 0.016 | 2.218e-2 | 0.005 | 0.005 | 5.24 | 0.008 | 1.17e-4 | 1.09e-1 | 0.75 | 0.11 | 0.17 |
| 10 | Fock | 1e-4 | 0.134 | 1.001e-3 | 0.099 | 0.098 | 5.59 | 0.028 | 1.10e-4 | 4.14e-3 | 0.76 | 0.84 | 0.60 |
| 10 | Fock | 1e-5 | 0.441 | 2.082e-5 | 0.253 | 0.254 | 5.50 | 0.033 | 5.93e-6 | 6.84e-5 | 0.75 | 2.54 | 1.50 |
| 10 | Fock | 1e-6 | 0.793 | 1.911e-7 | 0.537 | 0.539 | 4.86 | 0.026 | 1.15e-7 | 3.89e-7 | 0.84 | 2.93 | 2.17 |
| 10 | Fock | 1e-7 | 0.949 | 9.539e-10 | 0.773 | 0.773 | 3.54 | 0.009 | 8.25e-10 | 8.63e-10 | 0.80 | 2.92 | 2.33 |

Naphthalene:

| λ | mask | ε | keep | error (Ha) | x/err | L/err | U/err | N/K | near bound | far bound | t assembly (s) | t solve (s) | t cert (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Eq 8 | 1e-3 | 0.025 | 6.042e-2 | 0.090 | 0.093 | 7.94 | 0.046 | 7.44e-3 | 3.67e-1 | 0.67 | 0.23 | 0.19 |
| 0 | Eq 8 | 1e-4 | 0.220 | 3.734e-3 | 0.460 | 0.474 | 7.13 | 0.091 | 2.32e-3 | 1.32e-2 | 0.63 | 1.56 | 0.87 |
| 0 | Eq 8 | 1e-5 | 0.629 | 1.435e-4 | 0.820 | 0.830 | 3.76 | 0.085 | 1.52e-4 | 1.18e-4 | 0.65 | 2.90 | 1.65 |
| 0 | Eq 8 | 1e-6 | 0.915 | 2.258e-6 | 0.948 | 0.941 | 2.04 | 0.029 | 2.54e-6 | 3.06e-7 | 0.65 | 2.86 | 1.93 |
| 0 | Eq 8 | 1e-7 | 0.991 | 6.680e-8 | 0.951 | 0.950 | 1.32 | 0.004 | 7.68e-8 | 3.97e-10 | 0.65 | 2.84 | 1.88 |
| 0 | Fock | 1e-3 | 0.032 | 4.602e-2 | 0.044 | 0.047 | 8.07 | 0.022 | 2.85e-3 | 3.09e-1 | 0.63 | 0.29 | 0.23 |
| 0 | Fock | 1e-4 | 0.259 | 1.630e-3 | 0.213 | 0.217 | 9.19 | 0.037 | 4.53e-4 | 1.02e-2 | 0.72 | 1.86 | 0.84 |
| 0 | Fock | 1e-5 | 0.682 | 2.071e-5 | 0.473 | 0.478 | 8.20 | 0.037 | 1.24e-5 | 9.04e-5 | 0.70 | 2.88 | 1.62 |
| 0 | Fock | 1e-6 | 0.938 | 1.333e-7 | 0.787 | 0.783 | 5.08 | 0.011 | 1.27e-7 | 2.17e-7 | 0.66 | 2.88 | 1.94 |
| 0 | Fock | 1e-7 | 0.995 | 1.952e-10 | 0.803 | 0.800 | 4.77 | 0.002 | 1.88e-10 | 2.82e-10 | 0.67 | 2.89 | 1.91 |
| 10 | Eq 8 | 1e-3 | 0.023 | 4.925e-2 | 0.096 | 0.101 | 8.06 | 0.035 | 6.54e-3 | 3.02e-1 | 0.66 | 0.17 | 0.19 |
| 10 | Eq 8 | 1e-4 | 0.175 | 2.137e-3 | 0.343 | 0.348 | 8.02 | 0.077 | 9.43e-4 | 1.00e-2 | 0.65 | 1.28 | 0.68 |
| 10 | Eq 8 | 1e-5 | 0.496 | 6.707e-5 | 0.754 | 0.762 | 5.06 | 0.081 | 6.35e-5 | 1.09e-4 | 0.68 | 2.70 | 1.50 |
| 10 | Eq 8 | 1e-6 | 0.796 | 2.739e-6 | 0.936 | 0.935 | 2.35 | 0.034 | 3.02e-6 | 6.38e-7 | 0.70 | 2.71 | 1.93 |
| 10 | Eq 8 | 1e-7 | 0.953 | 3.317e-8 | 0.937 | 0.938 | 1.71 | 0.012 | 3.92e-8 | 1.64e-9 | 0.65 | 2.76 | 1.99 |
| 10 | Fock | 1e-3 | 0.030 | 3.356e-2 | 0.027 | 0.030 | 8.35 | 0.009 | 1.27e-3 | 2.44e-1 | 0.63 | 0.22 | 0.18 |
| 10 | Fock | 1e-4 | 0.204 | 1.111e-3 | 0.121 | 0.124 | 9.32 | 0.029 | 1.74e-4 | 7.84e-3 | 0.72 | 1.52 | 0.71 |
| 10 | Fock | 1e-5 | 0.540 | 1.654e-5 | 0.413 | 0.414 | 8.83 | 0.035 | 8.46e-6 | 8.42e-5 | 0.80 | 2.87 | 1.42 |
| 10 | Fock | 1e-6 | 0.826 | 1.206e-7 | 0.548 | 0.549 | 7.85 | 0.013 | 8.05e-8 | 4.75e-7 | 0.86 | 2.72 | 1.78 |
| 10 | Fock | 1e-7 | 0.966 | 3.750e-10 | 0.638 | 0.639 | 6.85 | 0.005 | 2.84e-10 | 1.14e-9 | 0.71 | 2.94 | 1.86 |

What the rows show:

- **Every row encloses the error** (the test asserts $L \le$ error $\le U$
  on all 40). $U$/err runs from 1.25 to 9.32; it is tightest on the plain
  mask at small $\varepsilon$ (1.25 to 2.35 at $10^{-6}$ and $10^{-7}$)
  and loosest with the Fock screen (3.40 to 9.32), where the far bound is
  as large as the near one or larger.
- **The estimate tracks the error only on the plain mask at small
  $\varepsilon$** (x/err 0.89 to 0.95 at $10^{-6}$ and $10^{-7}$). At
  $\varepsilon = 10^{-3}$ it is 0.005 to 0.10 of the error, and with the
  Fock screen it stays below 0.81 everywhere. The estimate sees only the
  near set. $L$/err is within 0.014 of $x$/err in every row.
- **The cost is of the order of the solve.** The certificate takes 0.15
  to 2.62 s against 0.10 to 2.99 s for the solve and 0.63 to 0.86 s for
  the assembly: 0.45 to 1.5 times the solve. The code forms the dense
  $n_v\times n_v$ Fock product for every pair, $O(n_o^2 n_v^3)$ work
  whatever the mask keeps; how the time divides among the parts was not
  profiled.

## Mutations {#lmp2-mask-certificate-mutations}

Each test in `crates/ferric-mp2/tests/lmp2_mask_certificate.rs` was run
against three deliberate defects in the production code (2026-11-19):

| Mutation | Result |
|---|---|
| $H$ dropped from $R_D$ (only $J_D$ on the dropped set) | the upper bound falls below the true error in all three enclosure tests (butane at $\varepsilon = 10^{-6}$: $U = 8.0\times10^{-10}$ against an error of $6.1\times10^{-7}$ Ha) |
| the local Fock diagonal gap as the node | every enclosure still holds: the node is about twice the canonical gap (butane 2.60 against 1.34 Ha), and the bounds have that much slack here. The node assertion fails in three tests |
| $\lVert J_F\rVert = 0$ | the default-$\tau$ tests pass, because there $N$ carries the error. The $\tau = \infty$ test fails: butane at $\varepsilon = 10^{-4}$ with the Fock screen gives $U = 1.64\times10^{-4}$ against $1.71\times10^{-4}$ Ha, and at $\varepsilon = 10^{-3}$ the $H$-only far term is 0.38–0.41 of the error |

The second row is why the node is tested directly against the canonical
RHF gap, an independent construction: an enclosure test can pass with a
wrong node.

## What it does not cover {#lmp2-mask-certificate-scope}

- **The RI fit error.** $E^*$ is canonical RI-MP2 in the run's auxiliary
  basis. The certificate says nothing about the distance from there to
  MP2 with exact integrals.
- **The integral-direct path.** Its locality maps and pair-local fits
  change the integrals; the error they make is not a mask error. The
  certificate is refused there, and with the in-core domain-fit options.
- **The estimate.** $x$ has no sign guarantee. Quote $[L, U]$ when you
  need a bound.
- **The SCF and localization.** The certificate bounds the correlation
  energy for the orbitals and Fock blocks the run used.
