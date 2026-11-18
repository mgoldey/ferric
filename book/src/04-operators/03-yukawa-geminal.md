# Yukawa, Slater geminal and fitted geminal kernels {#yukawa-geminal}

<!-- status: gap; registry: none -->

The kernels of 4.1 split $1/r$ with an error function. The kernels here cut
it off with an exponential. Two physical situations call for that. A charge
inside a polarizable medium is screened: its potential falls off as
$e^{-\zeta r}/r$, faster than $1/r$, and $1/\zeta$ is the screening length.
And the correlated wavefunction of two electrons has a cusp, a linear term
in $r_{12}$ at coalescence, which a product of smooth orbitals cannot
represent. Explicitly correlated (F12) methods add a factor that has that
linear term and decays at large $r_{12}$, and the usual choice is the Slater
geminal $e^{-\zeta r}$.

Three kernel families decay exponentially in $r_{12}$. The Yukawa kernel is a
screened Coulomb interaction; the Slater geminal $e^{-\zeta r}$ and its
Gaussian fit are the correlation factors of explicitly correlated (F12)
methods. All three come from libint2.

The plan: list the three kernels, show why the exact ones need a
two-variable auxiliary function, look at the fit and what it cannot do,
then state the parameter limits a caller has to respect.

| Entry point | Location |
|---|---|
| Rust | `ferric-integrals/src/operator.rs::{yukawa,slater_geminal,stg}`, `ferric-integrals/src/engine.rs::new_2e_geminal` |
| Consumer | `crates/ferric-mp2/src/f12.rs` (CABS construction; section @sec:cabs) |

## The three kernels

| Kernel | `Operator` | libint2 class | Parameter |
|---|---|---|---|
| $e^{-\zeta r}/r$ | `Operator::yukawa` | `stg_x_coulomb` | $\zeta > 0$ |
| $e^{-\zeta r}$ | `Operator::slater_geminal` | `stg` | $\zeta > 0$ |
| $-\gamma^{-1}e^{-\gamma r} \approx \sum_i c_i e^{-\alpha_i r^2}$ | `Operator::stg` | `cgtg`, `cgtg_x_coulomb`, `delcgtg2` | $\gamma$ |

The first two are exact kernels. The third is a fit. The Rust side maps the
first two straight onto libint2 operator codes, in
`operator_kind_to_ffi` of `crates/ferric-integrals/src/engine.rs` (shown
whole):

```rust
fn operator_kind_to_ffi(kind: OperatorKind) -> Result<c_int, FerricError> {
    match kind {
        OperatorKind::Coulomb => Ok(ffi::OP_COULOMB),
        OperatorKind::ErfCoulomb => Ok(ffi::OP_ERF_COULOMB),
        OperatorKind::ErfcCoulomb => Ok(ffi::OP_ERFC_COULOMB),
        OperatorKind::Yukawa => Ok(ffi::OP_YUKAWA),               // libint2 stg_x_coulomb
        OperatorKind::SlaterGeminal => Ok(ffi::OP_SLATER_GEMINAL), // libint2 stg
        _ => Err(FerricError::Libint(format!(
            "operator {:?} not supported for libint2 engines",
            kind
        ))),
    }
}
```

The operator's `omega` field carries $\zeta$ for both. The fitted geminal
does not pass through this function; it has its own constructor, below.

### Why the exact kernels need two variables

libint2 evaluates the two exact kernels through its `TennoGmEval` core,
which replaces the Boys function by a two-variable function of
$T = \rho R^2$ and $U = \zeta^2/4\rho$, with $\rho$ the reduced exponent of
the bra and ket. The two variables come out of the Fourier-space form of
the integral, as in @sec:terf-table-engine-center-integral-auxiliary-function. The Yukawa kernel transforms to

$$ \widehat{\Big(\frac{e^{-\zeta r}}{r}\Big)}(k) = \frac{4\pi}{k^2 + \zeta^2}, $$

and the Gaussian charge distributions contribute $e^{-k^2/4\rho}$ and the
phase $e^{i\mathbf k\cdot\mathbf R}$. Substitute $k = 2\sqrt\rho\,\kappa$.
The Gaussian becomes $e^{-\kappa^2}$, the denominator becomes
$k^2 + \zeta^2 = 4\rho(\kappa^2 + U)$, and the phase depends on
$kR = 2\kappa\sqrt T$. After the substitution only $T$ and $U$ remain. At
$\zeta = 0$, $U = 0$ and the integral is the Coulomb one, a function of $T$
alone: the Boys function. $\zeta \to 0$ turns the Yukawa kernel into
Coulomb, approached from below because $e^{-\zeta r} < 1$. The substitution
was checked with sympy, and the transform by direct integration of
$(4\pi/k)\int_0^\infty e^{-\zeta r}\sin(kr)\,dr$.

Its interpolation covers $U \in [10^{-7}, 10^{3}]$; outside that range the
result depends on the libint2 build, which `Operator::yukawa` documents
case by case. @sec:yukawa-geminal-limits-caller-must-respect lists the cases.

### The fitted geminal

The third is a fit: the six-term Tew-Klopper least-squares fit of
$e^{-r}$, scaled $r \to \gamma r$, so $\alpha_i \to \gamma^2\alpha_i$, with
the $-1/\gamma$ of the F12 geminal folded into the coefficients. The same
fit feeds three libint2 classes, chosen by `kind`: the geminal $f_{12}$
(`cgtg`), the geminal times $1/r_{12}$ (`cgtg_x_coulomb`), and
$|\nabla f_{12}|^2$ (`delcgtg2`). Each is a sum of Gaussian-kernel
integrals, one per term of the fit. This is the construction in
`Operator::stg`, with the doc comment and the `FIT` table removed:

```rust
let mut op = Self::primitive(kind, gamma, 0.0);
op.is_composite = true;
op.num_components = FIT.len();                 // 6 Gaussians
let g2 = gamma * gamma;
let pref = -1.0 / gamma;                       // f12 = -(1/gamma) exp(-gamma r12)
for (i, (alpha, c)) in FIT.iter().enumerate() {
    op.c_omegas[i] = alpha * g2;               // exp(-alpha (gamma r)^2): exponent alpha gamma^2
    op.c_coeffs[i] = pref * c;                 // -c_i / gamma
    op.c_kinds[i] = kind;                      // the same fit feeds cgtg / cgtg_x_coulomb / delcgtg2
}
op
```

One fitted geminal is one libint2 engine (`Engine::new_2e_geminal`), which
reads the six exponents and coefficients out of the composite arrays and
refuses an operator that has none.

The fit cannot reproduce the slope at coalescence. Differentiate the exact
geminal:
$\partial_r\big(-e^{-\gamma r}/\gamma\big) = e^{-\gamma r}$, which is 1 at
$r = 0$ for every $\gamma$. That unit slope is the linear term the cusp
needs. Every Gaussian has zero slope at $r = 0$, so the fitted geminal has
zero slope there. Evaluated from the coefficients in `operator.rs` (at
$\gamma = 1$, before the $-1/\gamma$ prefactor), the fit gives
$\sum_i c_i = 1.011522$ at $r = 0$ against $e^0 = 1$, and its largest
deviation from $e^{-r}$ on $[0, 10]$ is 0.0194, near $r = 0.42$. These two
numbers come from evaluating the six terms with numpy; no committed test
pins them.

None of the three is admitted by the Schwarz table builder.

## Limits a caller must respect {#yukawa-geminal-limits-caller-must-respect}

| Region | Stock libint2 | With ferric's `boys.h` overrides |
|---|---|---|
| $U = \zeta^2/4\rho > 10^3$ | out-of-range table read (2.7.2) or a Gaussian-fit fallback (2.13.1) | rejected with an error status (2.7.2) |
| $T > 1024$ and $U + \zeta R \gtrsim 709$ | NaN, silently | the underflowed true value, below $10^{-200}$ |
| Yukawa, small $T$ with $U < 10^{-7}$ | the upward recursion loses all digits at high $m$ | a Gauss–Legendre quadrature of the defining integral |
| $\zeta \le 0$ | invalid | invalid (`debug_assert`) |

Read the rows in terms of the two variables. $U$ is large when the
operator decays faster than the basis functions are wide: a tight kernel
on diffuse functions. $T$ is large when the two charge distributions are
far apart compared with their width. The 709 in the second row is about
$\ln$ of the largest double ($e^{709} \approx 8\times10^{307}$), and
$U + \zeta R = U + 2\sqrt{UT}$ because $\sqrt{UT} = \zeta R/2$. The third row is the near-Coulomb corner:
$\zeta$ so small that $U$ drops below the table, where the stock code
switches to a recursion in $m$ that loses digits.

The overrides live in `crates/ferric-integrals/shim/libint2_overrides/`; a
libint2 build that does not receive them keeps the stock behaviour, and the
build prints a warning. Ferric builds no Schwarz table for these kernels
(`ferric-integrals/src/schwarz.rs::check_operator_supported` errors), so no
screen hides a bad value.

## Numerical anchors {#yukawa-geminal-numerical-anchors}

| What | Test | Bar |
|---|---|---|
| Yukawa and Slater-geminal auxiliary functions at the edges of libint2's table, against a quadrature reference that is itself anchored | `crates/ferric-integrals/tests/tenno_table_edge.rs` | per point, derived from the reference value |
| small-$T$ behaviour of the Yukawa recursion | `crates/ferric-integrals/tests/yukawa_urr_small_t.rs` | per test |

No proof script covers these kernels. The Fourier transform
$\widehat{e^{-\zeta r}/r} = 4\pi/(k^2 + \zeta^2)$ shows the Yukawa kernel is
positive definite, so a Schwarz bound would be valid; the code does not build
one. A proof script and a committed test of the fitted geminal against the
exact one would move this section out of `gap`.

## Sources

- D. P. Tew and W. Klopper, *J. Chem. Phys.* **123**, 074101 (2005).
- S. Ten-no, *Chem. Phys. Lett.* **398**, 56 (2004).
- E. F. Valeev, Libint, <https://github.com/evaleev/libint>.
