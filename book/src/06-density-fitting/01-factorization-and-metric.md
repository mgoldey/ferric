# 6.1 The factorization, the metric, Cholesky and eigendecomposition factors

<!-- section: 6.1; status: edit; registry: none;
     theory: crates/ferric-mp2/theory.md#"1. Density fitting" -->

Every correlated method in ferric and most of its SCF builders replace
four-index integrals by products of three-index factors. This section fixes
the factorization once: the metric, the two ways the code takes its inverse
square root, and why the choice does not change an energy but does change
what can go wrong. Section 10.1 uses the result for RI-MP2; chapter 18 uses
it for RPA.

| Index | Range | Code name |
|---|---|---|
| $p, q, r, s$ | molecular orbitals | |
| $\mu, \nu$ | AO functions | `0..nbas` |
| $P, Q$ | auxiliary functions | `0..naux` |

| Entry point | Location |
|---|---|
| Metric | `ferric-integrals/src/threeindex.rs::coulomb_metric_2c` |
| Factors | `ferric-mp2/src/rimp2.rs::{cholesky_inverse_sqrt,eigh_inverse_sqrt,metric_inverse_sqrt}` |
| CLI, Python | the auxiliary basis: `[mp2] auxbasis`, `[rpa] auxbasis`; the `auxbasis` argument of the `run_*` functions |

## 6.1.1 The factorization

$$ V_{PQ} = (P|\hat w|Q), \qquad B^P_{pq} = \sum_Q M_{PQ}\,(Q|\hat w|pq), \qquad M^{\mathsf T}M = V^{-1}, \qquad (pq|\hat w|rs) \approx \sum_P B^P_{pq}B^P_{rs} . \tag{6.1} $$

$V$ is the metric, $(Q|pq)$ the raw three-index integral and $B$ the dressed
factor (chapter 0, section 7). The metric and the three-index integrals use
the same operator $\hat w$. With that choice the fitted density is the
$\hat w$-projection of the orbital product onto the auxiliary space, and the
error in (6.1) is second order in the fitting residual (section 10.1,
equation 10.5).

`coulomb_metric_2c(op, dfbs)` builds $V$ for any operator, not only Coulomb:
it loops over unique shell pairs $P \ge Q$ with a two-center engine for `op`
and fills both triangles.

## 6.1.2 Any factor gives the same energy

Only $B^{\mathsf T}B = (pq|P)\,V^{-1}\,(Q|rs)$ enters the four-index
approximation, so any $M$ with $M^{\mathsf T}M = V^{-1}$ gives the same
integrals. The same holds for every energy built from them, including those
that are not a polynomial in the integrals: the dRPA energy depends on $B$
only through $B^{\mathsf T}B$, because $\det(1 + B G B^{\mathsf T}) =
\det(1 + G B^{\mathsf T}B)$ for any diagonal $G$ (Sylvester's identity).

> **Verified equation 6.2** · the energy depends only on $B^{\mathsf T}B$
>
> The closed-shell dRPA energy computed from $B = L^{-1}(Q|ia)$ and from
> $B = V^{-1/2}(Q|ia)$ is the same, and the log-determinant is invariant under
> $B \to OB$ for orthogonal $O$.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_drpa_pdep_sympy.py`, checks `ri-factor-freedom` and `sylvester` |
> | Negative control | none; the claim is an identity with no tunable part |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/rimp2.rs::metric_inverse_sqrt` |
>
> <!-- verified: id=df.factor-freedom; script=scripts/rpa_drpa_pdep_sympy.py; claim=ri-factor-freedom; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/rimp2.rs::{metric_inverse_sqrt,cholesky_inverse_sqrt,eigh_inverse_sqrt} -->

The dissertation and most of the literature write $M = V^{-1/2}$, the
symmetric inverse square root (dissertation chapter 1, section 1.4.1, and
chapter 4, Eq. 4.5). The code uses two factors.

## 6.1.3 The two factors

| Factor | $M$ | $M^{\mathsf T}M$ | Cost | Function |
|---|---|---|---|---|
| Cholesky | $L^{-1}$, $V = LL^{\mathsf T}$ | $V^{-1}$ | one factorization and a triangular inverse | `cholesky_inverse_sqrt` |
| Eigendecomposition | $U_{\mathrm{k}}\lambda_{\mathrm{k}}^{-1/2}U_{\mathrm{k}}^{\mathsf T}$ over eigenvalues $\lambda \ge 10^{-10}$ | $V^{+}$, the pseudo-inverse on the kept modes | one symmetric eigendecomposition | `eigh_inverse_sqrt` |

`metric_inverse_sqrt(v, op)` chooses between them by operator:

<!-- excerpt: ferric-mp2/src/rimp2.rs::metric_inverse_sqrt -->
```rust
if matches!(op.kind, OperatorKind::ErfCoulomb | OperatorKind::Terf) {
    eigh_inverse_sqrt(v)
} else {
    cholesky_inverse_sqrt(v)
}
```

**Cholesky is the default.** Coulomb, erfc, terfc and Yukawa have positive
definite kernels (section 4.1), so their metric is positive definite and the
factorization exists. A Cholesky failure on one of them is an error, and the
run stops: it means an integral or auxiliary-basis defect, not
near-dependence to be regularized away.

**The eigendecomposition is for erf and terf.** The erf kernel is positive
definite but smooth, so tight auxiliary functions in a Coulomb-fitted basis
give near-zero eigenvalues that round to either sign. The terf kernel is not
positive definite at all. In both cases `eigh_inverse_sqrt` drops every mode
with $\lambda < 10^{-10}$, whatever its sign, and scales the rest by
$\lambda^{-1/2}$. The product $M^{\mathsf T}M$ is then the pseudo-inverse
$V^{+}$ restricted to the kept modes, not $V^{-1}$; the fit is exact on the
span of the kept modes and blind to the dropped ones. A negative eigenvalue
larger than $10^{-10}$ in magnitude is printed as a NOTE and dropped like the
others.

**Orientation.** $L^{-1}$ is lower triangular and not symmetric, so a factor
used on the other side of a product has to be transposed. Where a code path
maps a vector $u$ in the dressed space back to auxiliary-function
coefficients, it forms $c = M^{\mathsf T}u$. The PDEP eigenpotentials are the
example (section 18.3): with the symmetric factor the transpose is invisible,
and with $L^{-1}$ it is the difference between a right and a wrong
eigenpotential.

## 6.1.4 Algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | $V_{PQ} = (P\vert\hat w\vert Q)$ | $N_{\mathrm{aux}}^2$ integrals | `coulomb_metric_2c` |
| 2 | $M$: Cholesky or eigendecomposition by operator | $N_{\mathrm{aux}}^3$ | `metric_inverse_sqrt` |
| 3 | $B = M\,(Q\vert pq)$ for the orbital blocks a method needs | $N_{\mathrm{aux}}^2\,n_pn_q$ | per method (section 10.1 for $B_{ov}$) |

Step 2 runs once per intermediates build, outside any parallel region. Both
factorizations may use threaded BLAS there.

## 6.1.5 Options

| Option | Effect |
|---|---|
| Auxiliary basis | the fitting space; the RI sets bundled with ferric are listed in `crates/ferric-core/src/basis/bundled` |
| Operator | chooses the metric operator and, through it, the factor |
| `RiMp2Config::metric_op` (library) | a metric operator different from the integral operator; a measurement option that makes the fitting error first order (section 6.3) |

## 6.1.6 Numerical anchors

The factor choice has no anchor of its own; it is exercised by every RI
anchor in the book. The RI-MP2 energy anchors (section 10.1) use the Cholesky
factor. The attenuated PDEP anchors with an erf operator use the
eigendecomposition: $\le 5.8\times10^{-11}$ Ha (erf) and $3.5\times10^{-11}$ Ha
(erfc) against an independent reference (`site/src/reference/validation.md`).

## 6.1.7 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-mp2/theory.md`, "1. Density fitting" | (6.1), the Cholesky default |
| `scripts/rpa_drpa_pdep_sympy.py` | factor freedom (checks `ri-factor-freedom`, `sylvester`) |
| Dissertation, chapter 1 (section 1.4.1) and chapter 4 (Eq. 4.5) | the RI approximation with $V^{-1/2}$ |

**Known limits (2026-11-15).** No Rust test compares the two factors on one
system; the equivalence is proved by the script, and the eigendecomposition
path is covered only through the erf and terf anchors.
