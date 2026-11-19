# The terf table engine and its far-field fallback {#terf-table-engine}

<!-- status: edit; registry: terfc-rimp2-gradient -->

Every Gaussian integral code reduces a two-electron integral over the
Coulomb kernel to one special function, the Boys function $F_m(T)$. erf and
erfc fit the same mould: their integrals are Coulomb integrals with a
modified exponent, so libint2 evaluates them with the same machinery. terf
does not fit. Its switch sits at a finite distance $r_0$, and the integral
depends on two dimensionless numbers, how far apart the charge
distributions are and how large $r_0$ is, both measured in units of the
same width. One variable becomes two.

libint2 has no terf kernel, so ferric evaluates terf and terfc integrals with
its own McMurchie–Davidson driver. The driver is the Coulomb one with the Boys
function replaced by a two-variable auxiliary function $G_m(S, s)$, read from
tables, a series or a tail form depending on where $(S, s)$ falls. This
section derives $G_m$ from the kernel, shows that the code's series is that
function, and states when each evaluation route is taken. Section @sec:erf-terf defines
the kernels and the curvature constraint.

The plan:

1. Transform the kernels to Fourier space, where terf becomes erf times
   $\cos(kr_0)$ (@sec:terf-table-engine-kernels-fourier-space).
2. Do the $s$-type two-center integral in Fourier space and recognize the
   result as the kernel at a new sharpness $\phi$; define $G_0$ (@sec:terf-table-engine-center-integral-auxiliary-function).
3. Differentiate to get $G_m$, show that the Poisson series in the shim is
   $G_m$, and split it into $F_m$ minus a short tail (@sec:terf-table-engine-center-integral-auxiliary-function).
4. Route each $(S, s)$ to a table, the tail form or the far-field shortcut
   (@sec:terf-table-engine-evaluation-routes).

| Entry point | Location |
|---|---|
| Rust | `Operator::terf`, `Operator::terfc`, `Operator::terfc_with_omega` in `ferric-integrals/src/operator.rs::Operator`; every `Engine` constructor dispatches them to the table engine (`ferric-integrals/src/engine.rs::is_table_engine`) |
| C++ | `crates/ferric-integrals/shim/shim.cc`: `terf_aux`, `terf_G_series`, `terf_G_tail`, `terf_delta`, `terf_far_field` |
| Tables | `terf-tables/*.bin`, located by `FERRIC_TERF_TABLE_DIR` |
| Optional | a libint2-native terf route behind the `libint2_terf` feature (`ferric-integrals/src/engine.rs::new_terf_libint2`) |

## The kernels in Fourier space {#terf-table-engine-kernels-fourier-space}

For a radial kernel $K(r)$ the three-dimensional transform is a sine
transform of $rK(r)$. Do the angular integral of
$e^{-i\mathbf k\cdot\mathbf r}$, which gives $4\pi\sin(kr)/(kr)$:

$$ \hat K(k) = \int K(r)\, e^{-i\mathbf k\cdot\mathbf r}\, d\mathbf r = \frac{4\pi}{k}\int_0^\infty r K(r)\sin(kr)\, dr . $$

The erf transform is standard. For terf, use its structure. In one
dimension, $\operatorname{terf}(r)$ is the odd function
$\operatorname{erf}(\omega r)$ convolved with
$\tfrac12[\delta(r - r_0) + \delta(r + r_0)]$: the convolution shifts erf by
$\pm r_0$ and averages, which is (@eq:erf-terf-2). The sine transform of a convolution
with that pair of deltas multiplies by $\cos(kr_0)$. So every terf
transform is the erf transform times $\cos(kr_0)$, and terfc follows from
terf + terfc = 1:

| Kernel | $\hat K(k)$ |
|---|---|
| $1/r$ | $4\pi/k^2$ |
| $\operatorname{erf}(\omega r)/r$ | $(4\pi/k^2)\, e^{-k^2/4\omega^2}$ |
| $\operatorname{erfc}(\omega r)/r$ | $(4\pi/k^2)\,\big(1 - e^{-k^2/4\omega^2}\big)$ |
| $\operatorname{terf}(r)/r$ | $(4\pi/k^2)\cos(kr_0)\, e^{-k^2/4\omega^2}$ |
| $\operatorname{terfc}(r)/r$ | $(4\pi/k^2)\,\big(1 - \cos(kr_0)\, e^{-k^2/4\omega^2}\big)$ |

Check the limits. $r_0 = 0$ sets $\cos(kr_0) = 1$ and the terf row becomes
the erf row. $\omega \to \infty$ at $r_0 = 0$ sends the Gaussian factor to 1
and the erf row to $4\pi/k^2$, the Coulomb transform.

Chapter @sec:screening uses two consequences.

- $\hat K_{\mathrm{terf}}$ is negative for $kr_0 \in (\pi/2, 3\pi/2)$ and
  periodically beyond; its first zero is at $kr_0 = \pi/2$. By Bochner's
  theorem the terf kernel is not positive definite.
- $\cos(kr_0) \le 1$, so $\hat K_{\mathrm{terfc}} \ge (4\pi/k^2)(1 -
  e^{-k^2/4\omega^2}) > 0$: terfc is positive definite for every
  $(r_0, \omega)$. Where $\cos(kr_0) < 0$ it exceeds $4\pi/k^2$, so
  $\operatorname{terfc}(r) \le 1$ pointwise does not make
  $(\rho|\operatorname{terfc}|\rho) \le (\rho|1/r|\rho)$.

The second point matters for screening. A pointwise bound in $r$ does not
give a bound on the integrals; the bound has to hold in $k$, where the
integrals are sums of $|\hat\rho(k)|^2 \hat K(k)$. The transforms carry the
factor $1/(2\pi)^3$ of the inverse transform in the code comments of
`schwarz.rs`, which write $\hat K/(2\pi)^3 =
\cos(kr_0)e^{-k^2/4\omega^2}/(2\pi^2k^2)$. The refusal itself is one match
arm in `ferric-integrals/src/schwarz.rs::check_operator_supported`, shown in
section @sec:schwarz-bound-admitted-kernels.

![The four kernels in r (left) and in k relative to the Coulomb transform (right); terf's transform crosses zero at k r0 = pi/2. Script: book/src/figures/04-operators/kernels_rk.py](../figures/04-operators/kernels_rk.svg)

> **Verified equation {#eq:operators-terf-fourier}** · Fourier transforms of the short-range kernels
>
> $\hat K_{\mathrm{terfc}}(k) = (4\pi/k^2)\,(1 - \cos(kr_0)e^{-k^2/4\omega^2})$, and
> the erfc row of the table, agree with a direct radial quadrature of
> $(4\pi/k)\int_0^\infty rK(r)\sin(kr)\,dr$.
>
> | | |
> |---|---|
> | Proof | `scripts/screening_bounds_sympy.py`, claim fourier |
> | Negative control | the transform with $+\cos$ must fail (recorded mutation) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-integrals/src/schwarz.rs::check_operator_supported` (the transforms are the reason terfc is admitted and terf refused) |
>
> <!-- verified: id=operators.terf.fourier; script=scripts/screening_bounds_sympy.py; claim=fourier; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-integrals/src/schwarz.rs::check_operator_supported -->

The same script also proves the sign change of the terf transform and the indefiniteness of its metric (claims sign and indefinite of `scripts/screening_bounds_sympy.py`).

## The two-center integral and the auxiliary function {#terf-table-engine-center-integral-auxiliary-function}

Take $s$ Gaussians $e^{-p|\mathbf r - \mathbf A|^2}$ and
$e^{-q|\mathbf r - \mathbf B|^2}$ at distance $R$. We want
$(a|K|b) = \iint a(\mathbf r_1) K(r_{12}) b(\mathbf r_2)$ for the terf kernel.

**Step 1: reduce to one Gaussian.** The integral depends only on the
relative coordinate, so it is the kernel averaged over the convolution of
the two Gaussians. That convolution is a Gaussian of exponent
$\theta^2 = pq/(p+q)$ centred at distance $R$:

$$ (a|K|b) = \Big(\frac{\pi^2}{pq}\Big)^{3/2} \int \Big(\frac{\theta^2}{\pi}\Big)^{3/2} e^{-\theta^2|\mathbf x - \mathbf R|^2}\, K(x)\, d\mathbf x . $$

**Step 2: go to Fourier space.** The normalized Gaussian transforms to
$e^{-k^2/4\theta^2}$, and the integral of a product becomes an integral of
transforms:

$$ (a|K|b) = \Big(\frac{\pi^2}{pq}\Big)^{3/2} \int \frac{d\mathbf k}{(2\pi)^3}\, e^{-k^2/4\theta^2}\, \hat K(k)\, e^{i\mathbf k\cdot\mathbf R} . $$

**Step 3: merge the Gaussians.** Insert
$\hat K = (4\pi/k^2)\cos(kr_0)e^{-k^2/4\omega^2}$. The two Gaussian factors
combine into $e^{-k^2/4\phi^2}$ with

$$ \frac{1}{\phi^2} = \frac{1}{\theta^2} + \frac{1}{\omega^2} = \frac1p + \frac1q + \frac{1}{\omega^2} . $$

Attenuation enters here: the operator's width $1/\omega$ adds in
quadrature to the widths of the two basis functions.

**Step 4: do the integrals.** The angular integral of
$e^{i\mathbf k\cdot\mathbf R}$ gives $4\pi\sin(kR)/(kR)$. Use
$\sin(kR)\cos(kr_0) = \tfrac12[\sin k(R + r_0) + \sin k(R - r_0)]$ to
turn the product into two sines, and for each one use

$$ \frac{2}{\pi}\int_0^\infty e^{-k^2/4\phi^2}\,\frac{\sin(kx)}{k}\,dk = \operatorname{erf}(\phi x) . $$

The result is the kernel itself at sharpness $\phi$:

$$ (a|\operatorname{terf}|b) = \Big(\frac{\pi^2}{pq}\Big)^{3/2} \frac{\operatorname{terf}(R; r_0, \phi)}{R} = \frac{2\pi^{5/2}}{pq\sqrt{p+q}}\, \frac{\phi}{\theta}\, G_0(S, s), \qquad S = \phi^2R^2, \quad s = \phi^2 r_0^2, $$

$$ G_0(S, s) = \frac{\sqrt\pi}{4\sqrt S}\Big[\operatorname{erf}\big(\sqrt S + \sqrt s\big) + \operatorname{erf}\big(\sqrt S - \sqrt s\big)\Big], \qquad G_0(0, s) = e^{-s} . $$

The second equality is a rewriting. Put $\sqrt S = \phi R$ in $G_0$ and use
$pq\sqrt{p+q}\,\theta = (pq)^{3/2}$; both sides become
$\pi^3\operatorname{terf}(R; r_0, \phi)/\big((pq)^{3/2}R\big)$.

Check the limits:

- $S \to 0$ (the two Gaussians on one centre): expand
  $\operatorname{erf}(\sqrt s \pm \sqrt S)$ to first order in $\sqrt S$.
  The zeroth-order terms cancel, the first-order terms add to
  $(4/\sqrt\pi)\sqrt S\,e^{-s}$, and $G_0(0, s) = e^{-s}$.
- $s = 0$: $G_0 = \sqrt\pi\operatorname{erf}(\sqrt S)/(2\sqrt S) = F_0(S)$,
  the erf-kernel integral.
- $\phi \to \theta$ ($\omega \to \infty$) at $s = 0$: the Coulomb integral
  $2\pi^{5/2}F_0(\theta^2R^2)/(pq\sqrt{p+q})$.

The Coulomb prefactor is common to every kernel; the kernel enters through
$\phi/\theta$ and through $G_0$ in place of $F_0$. Claim [two-center-steps]
of `scripts/terf_identities_sympy.py` checks the steps: the $k$ integral of
step 4 against $\operatorname{erf}(\phi x)$ by quadrature, the prefactor
identity, and the limits $S \to 0$ and $s = 0$.

> **Verified equation {#eq:operators-terf-two-center}** · the $s$-$s$ terf integral is the kernel at sharpness $\phi$
>
> For $s$ Gaussians of exponents $p$, $q$ at distance $R$,
> $(a|\operatorname{terf}|b) = \frac{2\pi^{5/2}}{pq\sqrt{p+q}}\frac{\phi}{\theta}G_0(S, s)$
> with the closed form of $G_0$ above.
>
> | | |
> |---|---|
> | Proof | `scripts/terf_identities_sympy.py`, claim two-center (closed form against a 30-digit quadrature of the defining six-dimensional integral, reduced to one radial integral) |
> | Negative control | $\operatorname{erf}(\sqrt S - \sqrt s) \to \operatorname{erf}(\sqrt s - \sqrt S)$ and dropping $\phi/\theta$ must fail (recorded mutations) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-integrals/src/engine.rs::is_table_engine`; `terf_aux` in `crates/ferric-integrals/shim/shim.cc` |
> | Code test | `crates/ferric-integrals/tests/validation_terfc_integrals.rs` (against an independent Fourier-space construction) |
>
> <!-- verified: id=operators.terf.two-center; script=scripts/terf_identities_sympy.py; claim=two-center; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-integrals/src/engine.rs::is_table_engine; rust_test=ferric-integrals/tests/validation_terfc_integrals.rs::terfc_integrals_h2o_cc_pvdz_vs_kspace -->

### Higher angular momentum

McMurchie-Davidson builds every other integral from derivatives of the $s$
integral with respect to $\mathbf R$ through the $R^n_{tuv}$ recursion of section @sec:md-three-center, seeded
with
$R^n_{000} = (-2\alpha)^n F_n(\alpha R^2)$. Each derivative in $\mathbf R$
brings down a factor through the chain rule. The dependence on $R$ is
through $S$ alone at fixed $s$, and $\partial S/\partial R_x = 2\phi^2 R_x$,
so the terf integrals use the same recursion with reduced exponent
$\phi^2$ and the seed

$$ R^n_{000} = (-2\phi^2)^n\, \frac{\phi}{\theta}\, G_n(S, s), \qquad G_n(S, s) = \Big(-\frac{\partial}{\partial S}\Big)^n G_0(S, s) . $$

The MD superscript $n$ is the $S$-derivative order; the series below, the
boxes and `terf_G_series` call the same order $m$. The shim calls
$A_n = (\phi/\theta)G_n(S, s)$ the Boys-replacement vector
(`terf_aux`). terfc integrals are Coulomb minus terf, built by the same
McMurchie-Davidson pass so the two share ordering and normalization. The
$R^n_{tuv}$ recursion is linear in its seeds and does not contain the exponent,
so the 3-centre values and derivatives form the difference on the seeds,

$$ R^n_{000} = (-2\theta^2)^n F_n(\theta^2 R^2) - (-2\phi^2)^n \frac{\phi}{\theta} G_n(S, s) , $$

and run one recursion and one contraction (`md_terms` in the shim). The
2-centre values subtract two passes.

The seed loop of `md_terms::kernel_R` in
`crates/ferric-integrals/shim/shim.cc` follows, with the table lookup above it
reduced to its arguments:

```cpp
phi2 = theta2 * omega2 / (theta2 + omega2);      // 1/phi^2 = 1/theta^2 + 1/omega^2
const double S_arg = phi2 * PQ2;                 // S = phi^2 R^2
const double s_arg = phi2 * r02;                 // s = phi^2 r0^2
// ... terf_aux fills G[n] = (phi/theta) G_n(S, s) ...
double facC = 1.0, facT = 1.0;
for (int n = 0; n <= L; ++n) {
    double seed;
    if (kernel == MdKernel::Coulomb) {
        seed = facC * F[n];                      // (-2 theta^2)^n F_n
    } else if (kernel == MdKernel::Terf) {
        seed = facT * G[n];                      // (-2 phi^2)^n (phi/theta) G_n
    } else {
        seed = facC * F[n] - facT * G[n];        // terfc: Coulomb - terf
    }
    Rn[RI(n, 0, 0, 0)] = seed;
    facC *= (-2.0 * theta2);
    facT *= (-2.0 * phi2);
}
```

Three points about this loop:

- **`theta2` here is $\theta^2$ of the pair being contracted**, the reduced
  exponent of the bra and ket in the 3-centre pass. The formula
  $\phi^2 = \theta^2\omega^2/(\theta^2 + \omega^2)$ is step 3 solved for
  $\phi^2$.
- **`G` already carries $\phi/\theta$.** `terf_aux` multiplies the
  auxiliary by that constant before returning, so the seed only needs the
  power of $-2\phi^2$.
- **terfc costs one recursion.** Forming the difference on the seeds keeps
  the $E$ coefficients, the recursion and the contraction shared between
  the Coulomb and terf parts.

### The defining series

The code evaluates $G_m$ from a Poisson series. With
$\pi_S(i) = e^{-S}S^i/i!$, $c_s(i) = \sum_{j \le i} e^{-s}s^j/j!$,
$\nabla f(i) = f(i) - f(i-1)$ and $(2i)!!/(2i+1)!!$,

$$ G_m(S, s) = \sum_{i \ge 0} \frac{(2i)!!}{(2i+1)!!}\, [\nabla^{m}\pi_S]{}(i)\, c_s(i) . $$

For $m = 0$ the series is an average over the Poisson distribution
$\pi_S(i)$, the probability of $i$ events at mean $S$, set by the distance
between the centres. It averages the product of two weights: the
double-factorial ratio, which is the weight of the Boys series, and
$c_s(i)$, the probability of at most $i$ events at mean $s$, set by
$r_0$.

Each $\nabla$ is one $S$-derivative. To see it, differentiate
$\pi_S(i) = e^{-S}S^i/i!$:
$\partial_S\pi_S(i) = -\pi_S(i) + \pi_S(i-1)$. So
$-\partial_S \pi_S(i) = \pi_S(i) - \pi_S(i-1) = \nabla\pi_S(i)$, with
$\pi_S(-1) = 0$.

Check $s = 0$. Then $c_0(i) = 1$ for every $i$, and with
$(2i)!! = 2^i\,i!$ the term $(2i)!!/(2i+1)!!\cdot S^i/i!$ becomes
$(2S)^i/(2i+1)!!$. For $m = 0$ the series is
$F_0(S) = e^{-S}\sum_i (2S)^i/(2i+1)!!$, and $m$ applications of $\nabla$
give $F_m = (-\partial_S)^mF_0$. Writing
$c_s = 1 - \tau_s$ with the Poisson tail $\tau_s(i) = \sum_{j > i} e^{-s}s^j/j!$
splits it exactly:

$$ G_m(S, s) = F_m(S) - \Delta_m(S, s), \qquad \Delta_m(S, s) = \sum_{i \ge 0} \frac{(2i)!!}{(2i+1)!!}\, [\nabla^{m}\pi_S]{}(i)\, \tau_s(i) . $$

$\Delta_m$ is the terfc auxiliary itself, and its sum length is set by $s$, not
$S$, because $\tau_s(i)$ decays super-exponentially once $i > s$
(`terf_delta`, `terf_G_tail`; the truncation index is `terf_tail_index`).

The reference implementation is a direct transcription. This is
`terf_G_series` with the far-field early return and the buffer
declarations removed:

```cpp
const int N = (int)(S + 12.0 * std::sqrt(S) + 60.0);   // covers the support of pi_S
pmf[0] = std::exp(-S);
for (int i = 1; i < N; ++i) pmf[i] = pmf[i - 1] * S / i;   // pi_S(i)
double t = std::exp(-s), acc = t;
cdf[0] = acc;
for (int i = 1; i < N; ++i) {
    t *= s / i;
    acc += t;
    cdf[i] = acc < 1.0 ? acc : 1.0;                          // c_s(i), clamped at 1
}
df[0] = 1.0;
for (int i = 1; i < N; ++i) df[i] = df[i - 1] * (2.0 * i) / (2.0 * i + 1.0);  // (2i)!!/(2i+1)!!
std::vector<double> row = pmf;
for (int m = 0; m <= mmax; ++m) {
    if (m > 0) {                                             // row <- nabla row
        next[0] = row[0];
        for (int i = 1; i < N; ++i) next[i] = row[i] - row[i - 1];
        row.swap(next);
    }
    double total = 0.0;
    for (int i = 0; i < N; ++i) total += df[i] * row[i] * cdf[i];
    G[m] = total;                                            // G_m(S, s)
}
```

The length $N = S + 12\sqrt S + 60$ is the cost of this form: it grows with
the separation. The tail form replaces it with a sum whose length depends
on $s$ only. `terf_delta` builds $\tau_s(i)$ downward from the top, adding
the smallest terms first, because $1 - c_s(i)$ cancels catastrophically:
for $s \le 1/2$, $c_s(i)$ is within about $10^{-17}$ of 1 by $i \approx 14$.
The sum still needs more terms than that. The bar is on $\Delta_m$, not on
$G_m$ (an error on $G_m$ is divided by $F_m(S) = O(1)$), and
$[\nabla^m\pi_S](i)$ grows with $m$, so terms with a negligible $\tau_s(i)$
still matter at $10^{-14}$ relative. The truncation index is

```cpp
inline int terf_tail_index(double s) {
    const int I = (int)std::ceil(1.80 * s + 40.0);
    if (I < 40) return 40;
    if (I > TERF_TAIL_IMAX) return TERF_TAIL_IMAX;   // 260
    return I;
}
```

The shim comment above it records the measurement behind the rule: an
mpmath reference needs $I \ge 32$ at $s = 0.5$ and $I \ge 172$ at $s = 80$
to reach $10^{-14}$, and $1.80s + 40$ sits above those values over
$s \in [0.01, 80]$. A Chernoff bound on the Poisson tail,
$I \approx s + 8\sqrt s + 16$, was rejected: it gives 23 at $s = 0.5$, short
of 32, and is also short at $s = 2$ and $s = 80$.

> **Verified equation {#eq:operators-terf-series}** · the code's Poisson series is $(-\partial_S)^m G_0$
>
> The series of `terf_G_series` equals the $m$-th $S$-derivative of the
> closed form, its derivative in $R$ has the Boys shape, it reduces to $F_m$
> at $s = 0$, and it splits exactly as $F_m - \Delta_m$.
>
> | | |
> |---|---|
> | Proof | `scripts/terf_identities_sympy.py`, claims series, boys-shape, s-zero, tail-form |
> | Negative control | the double-factorial ratio $(2i)/(2i+1) \to (2i+1)/(2i+2)$ must fail (recorded mutation) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-integrals/src/engine.rs::is_table_engine`; `terf_G_series`, `terf_G_tail` in `crates/ferric-integrals/shim/shim.cc` |
> | Code test | `crates/ferric-integrals/tests/terfc_series_anchors.rs` ($G_m(S, 0) = F_m$; mpmath anchors) |
>
> <!-- verified: id=operators.terf.series; script=scripts/terf_identities_sympy.py; claim=series; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-integrals/src/engine.rs::is_table_engine; rust_test=ferric-integrals/tests/terfc_series_anchors.rs::series_g_at_s_zero_equals_boys_s25 -->

### Far field and the range of $s$

**Far field.** For $\sqrt S - \sqrt s = \phi(R - r_0) \gg 1$ both error
functions in $G_0$ are $1$ to within $\operatorname{erfc}(\sqrt S - \sqrt s)$
and $G_m \to F_m(S)$. Physically, the two charge distributions are farther
apart than $r_0$ by many widths $1/\phi$, so the terf kernel is $1/r$ over
all of them and the integral is the Coulomb one. The shim takes
$G_m = F_m(S)$ when $S > 50$ and $\sqrt S - \sqrt s > 6$
(`terf_far_field`). The second test matters when $\omega$ is free: $S > 50$
alone admits pairs inside the $r_0$ shell.

```cpp
inline bool terf_far_field(double S, double s) {
    return S > TERF_ASYMPTOTIC_S && std::sqrt(S) - std::sqrt(s) > 6.0;   // S > 50, phi (R - r0) > 6
}
```

**The range of $s$.** From step 3, $1/\phi^2 \ge 1/\omega^2$, so
$\phi \le \omega$ and $s \le (\omega r_0)^2$; under the curvature
constraint $s < 1/2$ for every pair.

> **Verified equation {#eq:operators-terf-far-field}** · far field and the range of $s$
>
> $G_m = F_m(S)$ to $10^{-12}$ relative where the shim takes the far-field
> branch ($S > 50$ and $\sqrt S - \sqrt s > 6$), and not at points that pass
> only the first test; $s \le (\omega r_0)^2$, so $s < 1/2$ under the
> curvature constraint.
>
> | | |
> |---|---|
> | Proof | `scripts/terf_identities_sympy.py`, claims far-field, s-bound |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-integrals/src/engine.rs::is_table_engine`; `terf_far_field` in `crates/ferric-integrals/shim/shim.cc` |
> | Code test | `crates/ferric-integrals/tests/terf_asymptotic_crossover.rs` |
>
> <!-- verified: id=operators.terf.far-field; script=scripts/terf_identities_sympy.py; claim=far-field; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-integrals/src/engine.rs::is_table_engine; rust_test=ferric-integrals/tests/terf_asymptotic_crossover.rs::asymptotic_matches_the_exact_series_at_the_crossover -->

### Tables

$G_{m,n}(S, s)$ is tabulated on four grids, finest first, covering
$S \le 20$ and $s \le 80$ (files `16_4_2.bin`, `8_10_5.bin`,
`4_20_20.bin`, `2_20_80.bin`, named by points per unit, $S_{\max}$ and
$s_{\max}$), with $m < 24$. A lookup takes the finest covering table and
interpolates with a $10 \times 10$ Lagrange stencil on the integer nodes.
The second index $n$ is an $s$-derivative index; energy integrals use
$n = 0$. Outside every table the tail form above is summed directly.
The top table limits a free $\omega$: $s \le 80$ means
$r_0\omega \le$ about 8.9 (the `Operator::terfc_with_omega` doc comment).

The inner loop of `terf_aux` runs after the table and the Lagrange
weights `wS`, `ws` have been chosen (the hit counter is removed):

```cpp
const int dimn = tbl->dimn;
for (int m = 0; m <= mmax; ++m) {
    double acc = 0.0;
    for (int a = 0; a < K_S; ++a) {            // 10 nodes in S
        const int iS = iS0 + a;
        double rowacc = 0.0;
        for (int c = 0; c < K_s; ++c) {        // 10 nodes in s
            rowacc += ws[c] * tbl->data[((((size_t)iS * ns) + (is0 + c)) * tbl->dimm + m) * dimn];
        }                                      //   table value G_{m,0} at node (iS, is0+c)
        acc += wS[a] * rowacc;
    }
    A[m] = phi_over_theta * acc;               // A_m = (phi/theta) G_m(S, s)
}
```

Two points:

- **The weights do not depend on $m$.** Every order shares the same
  $(S, s)$, so the table, the node window and the Lagrange weights are
  computed once and only the tabulated values change inside the $m$ loop.
- **The $n$ index is fixed at 0.** The trailing `* dimn` steps over the
  $s$-derivative slots. The shim comment states that $n$ is 0 at every
  call site.

## Evaluation routes {#terf-table-engine-evaluation-routes}

| Region | Route | Shim function |
|---|---|---|
| $S > 50$ and $\sqrt S - \sqrt s > 6$ | $G_m = F_m(S)$ | `terf_far_field` |
| inside a table ($S \le 20$, $s \le 80$) | $10 \times 10$ Lagrange interpolation in the finest covering table | `terf_aux` |
| elsewhere | $F_m(S) - \Delta_m(S, s)$, summed to `terf_tail_index(s)` | `terf_G_tail`, `terf_delta` |
| reference only | the defining series | `terf_G_series` |

In the code the order of tests is: `terf_aux` looks for a covering table
first, and only outside every table calls `terf_G_tail`, which applies the
far-field test before it sums $\Delta_m$. The table region ($S \le 20$) and
the far-field region ($S > 50$) do not overlap, so the order does not
change the result.

The second far-field test matters only when $\omega$ is chosen apart from
$r_0$ (`Operator::terfc_with_omega`). Under the curvature constraint every
pair with $S > 50$ already has $\sqrt S - \sqrt s > 6.36$, and the branch is
the same as with the first test alone. The 6.36 is
$\sqrt{50} - \sqrt{1/2}$, the smallest gap the constraint allows.

## Derivative integrals {#terf-table-engine-derivative-integrals}

A terfc-attenuated RI-MP2 gradient (section
@sec:rimp2-gradient-attenuated-erfc-ri-mp2) needs the first nuclear
derivatives of $(P|\mu\nu)$ and $(P|Q)$ under terf and terfc. libint2 has no
terf kernel, so the shim computes them with its own McMurchie–Davidson pass.
`ferric-integrals/src/engine.rs::Engine::new_3center_deriv` and
`new_2center_deriv` with `Operator::terf` or `Operator::terfc` return them in
libint2's block order, 9 blocks for three centres and 6 for two (section
@sec:integrals-derivatives). The shim functions are `md_terms::eri3_derivs`
(three centres) and `compute_cart_eri2_deriv` (two).

**The Hermite derivative.** A Cartesian Gaussian on $\mathbf A$
differentiates as

$$ \frac{\partial}{\partial A_x}\big(x_A^{i} e^{-a x_A^2}\big) = 2a\, x_A^{i+1} e^{-a x_A^2} - i\, x_A^{i-1} e^{-a x_A^2} . $$

The product $x_A^ix_B^je^{-ax_A^2 - bx_B^2}$ expands in Hermite Gaussians
$\Lambda_t(x; P)$ with coefficients $E^{ij}_t$. Apply the rule above to the
$x_A^i$ factor and expand each of the two products again: the derivative is
the same expansion over the same $\Lambda_t$, with

$$ \frac{\partial E^{ij}_t}{\partial A_x} \to 2a\,E^{i+1,j}_t - i\,E^{i-1,j}_t, \qquad t \le i + j + 1 . $$

The ket side is unchanged, so the contraction with $R_{tuv}$ needs $R$ one
order higher: total order $L = l_P + l_A + l_B + 1$. An auxiliary function on
$\mathbf P$ alone has no partner, and its derivative uses $E^{i\pm1,0}$ the
same way. In the shim this is `hermite_deriv_first`, which reads an $E$ table
built one step deeper in $i$:

```cpp
// d/d(first centre) of an E table built to (li+1, lj): 2a E^{i+1,j} - i E^{i-1,j}.
void hermite_deriv_first(const HermiteE &H, double a, int li, int lj, HermiteTab &T) {
    T.init(li, lj, 1);
    for (int i = 0; i <= li; ++i)
        for (int j = 0; j <= lj; ++j)
            for (int t = 0; t <= i + j + 1; ++t) {      // one more Hermite order
                double x = 2.0 * a * H.at(i + 1, j, t);  // 2a E^{i+1,j}_t
                if (i > 0) x -= i * H.at(i - 1, j, t);   // - i E^{i-1,j}_t
                T.set(i, j, t, x);
            }
}
```

**The third centre.** A rigid translation leaves every integral unchanged,
so the derivatives with respect to all centres sum to zero. The shim
computes $\partial_P$ and $\partial_A$ and sets
$\partial_B = -(\partial_P + \partial_A)$. In the two-centre case
$\partial_Q = -\partial_P$.

**The kernel.** The kernel enters only through $R^n_{000}$. For terf it is
$(-2\phi^2)^n(\phi/\theta)G_n(S, s)$, the seed of
@sec:terf-table-engine-center-integral-auxiliary-function, read from the
tables at $n \le L$. For terfc the shim forms
$R_{\mathrm{terfc}} = R_{\mathrm{Coulomb}} - R_{\mathrm{terf}}$ before the
contraction, from the combined seeds in three centres and from two $R$ tables
in two (`md_kernel_R`), so both parts share the $E$ coefficients and the
prefactor. The derivative reads $G$ one order higher than the value. The
tables hold $m < 24$ and the seeds run to $m = L$, so the engine refuses
$L \ge 24$; with s..g auxiliaries and s..f orbitals, $L \le 11$.

**Term lists.** The three-centre kernels, for values and derivatives, write
each Cartesian component as a list of Hermite terms $(c, t, u, v)$ and store
$(t, u, v)$ as its flat index $(t(L+1) + u)(L+1) + v$ in an $R$ cube of side
$L + 1$. The index is linear, so $R_{t'+t,\,u'+u,\,v'+v}$ is the cube
entry at the sum of the two indices, and an integral is a double sum over the
auxiliary list and the orbital-pair list. Pair lists are built once per
primitive pair, auxiliary lists once per shell triple. The kernels they
replaced, `compute_cart_eri3` and `compute_cart_eri3_deriv`, stay in the shim
as the reference the tests compare against.

**Screening.** The derivative loop of the RI-MP2 gradient screens nothing:
every $(P|\mu\nu)$ shell triple is evaluated. The terfc energy is also
unscreened (it has no `eri3_screen_thresh`), so the gradient differentiates
the same sum. Schwarz bounds are not used for terfc derivatives.

### What is proven and what is not {#terf-table-engine-proven}

`crates/ferric-integrals/tests/terfc_deriv_integrals.rs`:

| Check | Systems | Measured | Bar |
|---|---|---|---|
| Coulomb kernel through the same MD derivative vs libint2 | water, cc-pVTZ + cc-pVTZ-RI, every class to $(g\vert ff)$ and $(g\vert g)$ | $3.8\times10^{-14}$ | $10^{-10}$ |
| terf and terfc derivatives vs five-point FD of the engine's values ($\omega r_0 = 1/\sqrt2$ and 4) | water, cc-pVDZ + cc-pVDZ-RI, every class | $2.6\times10^{-11}$ | 10 times the Coulomb FD floor ($3.9\times10^{-11}$) |
| terfc derivative = libint2 Coulomb − table terf | the same | $2.1\times10^{-14}$ | $10^{-10}$ |

Dropping the $-i\,E^{i-1,j}$ term fails all three checks. Writing
$R_{\mathrm{Coulomb}} + R_{\mathrm{terf}}$ fails the second and third.

`crates/ferric-integrals/tests/terfc_md_terms_vs_reference.rs` compares the
term-list kernels with the reference kernels on every class of water /
cc-pVTZ + cc-pVTZ-RI. terf values are bit-identical; terfc values and all
derivatives agree to $5.1\times10^{-15}$, against a bar of $10^{-13}$ times
the largest element. Flipping the sign of the terf seed, swapping $u$ and $v$
in the pair index, or dropping $-i\,E^{i-1,j}$ each fails it.

`scripts/terfc_rimp2_gradient_sympy.py` proves the $s$-type closed forms and
their derivatives against a $k$-space quadrature (part K) and the Hermite
identity for $i, j \le 3$ (part H). The general-$l$ shim code is covered only
by the Rust tests above.

## Refusals {#terf-table-engine-refusals}

| Condition | Result |
|---|---|
| tables not found | engine construction errors; nothing falls back to Coulomb |
| total angular momentum reaches the table depth (24) | the `_checked` calls return the refusal |
| a derivative integral with $L \ge 24$ ($L = l_P + l_A + l_B + 1$) | refused (@sec:terf-table-engine-derivative-integrals) |
| terf in a Schwarz or CSB table | refused (section @sec:schwarz) |

Why no fallback to Coulomb? A Coulomb integral in place of a terfc one is
wrong by the whole terf part, the long range that the operator exists to
remove. Construction fails instead, before any integral is computed.

## Numerical anchors {#terf-table-engine-numerical-anchors}

| What | Test | Bar |
|---|---|---|
| terf and terfc $(P\vert\mu\nu)$, $(P\vert Q)$ against a Fourier-space construction sharing nothing with the MD driver (H2O, NH3 / cc-pVDZ; H2O / aug-cc-pVDZ; $r_0$ = 0.75 and 1.05 Å) | `crates/ferric-integrals/tests/validation_terfc_integrals.rs` | $2\times10^{-12}$ (3-index), $3\times10^{-11}$ (2-index) |
| tail form against the series | `crates/ferric-integrals/tests/terf_tail_form.rs` | $10^{-9}$ relative on $\Delta_m$ |
| far field against the series at the crossover | `crates/ferric-integrals/tests/terf_asymptotic_crossover.rs` | $10^{-4}$ over a sweep that includes $S = 25$, below the crossover, where the far-field form is off by $2.5\times10^{-5}$; on the rows the code takes ($S \ge 50$) the test comment records about $10^{-13}$, which the bar does not pin |
| $G_m(S, 0) = F_m(S)$ | `crates/ferric-integrals/tests/terfc_series_anchors.rs` | per test |

## Sources

- A. D. Dutoi and M. Head-Gordon, *J. Phys. Chem. A* **112**, 2110 (2008).
- M. Goldey, PhD dissertation, University of California, Berkeley (2014),
  chapter 4.
- L. E. McMurchie and E. R. Davidson, *J. Comput. Phys.* **26**, 218 (1978).
