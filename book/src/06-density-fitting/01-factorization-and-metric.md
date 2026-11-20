# The factorization, the metric, Cholesky and eigendecomposition factors {#factorization-and-metric}

<!-- status: edit; registry: rs-mp2-rpa, u-rs-mp2-rpa -->

A two-electron integral $(pq|rs)$ is the Coulomb energy between two charge
distributions, the orbital products $\rho_{pq} = \phi_p\phi_q$ and
$\rho_{rs} = \phi_r\phi_s$. There are $N^2$ such products and $N^4$ integrals
between them. Most of the products look alike: each is a smooth blob centred
somewhere in the molecule, and a few atom-centred functions can describe it
well. Density fitting expands every product in one auxiliary basis
$\{\chi_P\}$ of $N_{\mathrm{aux}}$ functions. The four-index
integral then becomes a product of two three-index objects, and nothing of
size $N^4$ is ever stored.

Every correlated method in ferric and most of its SCF builders use this
factorization. This section fixes it once: how the fit is defined, why the
metric carries the same operator as the integrals, the two ways the code
takes the metric's inverse square root, and why that choice does not change
an energy but does change what can go wrong. Section @sec:closed-shell-ri-mp2 uses the result
for RI-MP2; chapter @sec:rpa uses it for RPA.

The plan:

1. Define the fit and derive the factorized integral (@sec:factorization-and-metric-factorization).
2. Show that any factor of $V^{-1}$ gives the same energy (@sec:factorization-and-metric-any-factor-gives-same).
3. Read the two factors the code uses and the rule that picks one (@sec:factorization-and-metric-factors).
4. Fit a kernel that is not positive definite with a signed factor (@sec:factorization-and-metric-signed).

| Index | Range | Code name |
|---|---|---|
| $p, q, r, s$ | molecular orbitals | |
| $\mu, \nu$ | AO functions | `0..nbas` |
| $P, Q$ | auxiliary functions | `0..naux` |

| Entry point | Location |
|---|---|
| Metric | `ferric-integrals/src/threeindex.rs::coulomb_metric_2c` |
| Factors | `ferric-mp2/src/rimp2.rs::{cholesky_inverse_sqrt,eigh_inverse_sqrt,metric_inverse_sqrt}` |
| Signed factor | `ferric-mp2/src/rimp2.rs::{signed_pair_block,spin_components_from_signed_b_ov}`, `ferric-rpa/src/energy.rs::eval_signed_trace_log_summands` |
| CLI, Python | the auxiliary basis: `[mp2] auxbasis`, `[rpa] auxbasis`; the `auxbasis` argument of the `run_*` functions |

## The factorization {#factorization-and-metric-factorization}

Take one product density $\rho_x$, where $x$ stands for an orbital pair $pq$
or an AO pair $\mu\nu$. We want coefficients $c_{Px}$ so that
$\tilde\rho_x = \sum_P c_{Px}\chi_P$ is close to $\rho_x$. "Close" needs a
measure. The integrals we are after are energies under a kernel $\hat w$, so
measure the error by the self-energy of the error density under that same
kernel:

$$ \mathcal E_x(c) = (\rho_x - \tilde\rho_x|\hat w|\rho_x - \tilde\rho_x) = (x|\hat w|x) - 2\sum_P c_{Px}A_{Px} + \sum_{PQ} c_{Px}V_{PQ}c_{Qx}, \qquad A_{Px} = (P|\hat w|x), \quad V_{PQ} = (P|\hat w|Q) . $$

Set the derivative with respect to $c_{Px}$ to zero. That gives
$-2A_x + 2Vc_x = 0$, so

$$ c_x = V^{-1}A_x . $$

Two facts follow at once. First, the residual
$\delta_x = \rho_x - \tilde\rho_x$ has zero $\hat w$-overlap with every
auxiliary function, since $(P|\hat w|\delta_x) = A_{Px} - (Vc_x)_P = 0$. The
fitted density is the $\hat w$-projection of $\rho_x$ onto the auxiliary
space. Second, for a Coulomb kernel $\mathcal E_x \ge 0$, and $\mathcal E_x$ is the electrostatic
self-repulsion of the part of $\rho_x$ the basis misses.

Now replace both densities in $(x|\hat w|y)$ by their fits. Substitute
$c = V^{-1}A$:

$$ (\tilde\rho_x|\hat w|\tilde\rho_y) = c_x^{\mathsf T}Vc_y = A_x^{\mathsf T}V^{-1}A_y . $$

Write $V^{-1} = M^{\mathsf T}M$ for some square matrix $M$ and group each
$M$ with its $A$:

$$ A_x^{\mathsf T}M^{\mathsf T}MA_y = (MA_x)^{\mathsf T}(MA_y) = \sum_P B^P_xB^P_y, \qquad B_x = MA_x . $$

With $x = pq$ and $y = rs$ that is the factorization used everywhere in the
book:

$$ V_{PQ} = (P|\hat w|Q), \qquad B^P_{pq} = \sum_Q M_{PQ}\,(Q|\hat w|pq), \qquad M^{\mathsf T}M = V^{-1}, \qquad (pq|\hat w|rs) \approx \sum_P B^P_{pq}B^P_{rs} . \label{eq:factorization-and-metric-1} $$

$V$ is the metric, $(Q|pq)$ the raw three-index integral and $B$ the dressed
factor (chapter @sec:notation, section @sec:notation-density-fitting).

Check the complete-basis limit. If $\rho_x$ lies in the span of the
$\chi_P$, say $\rho_x = \sum_P d_P\chi_P$, then $A_x = Vd$, the fit returns
$c_x = d$, the residual is zero, and (@eq:factorization-and-metric-1) is exact.

**Why the metric carries $\hat w$.** The metric and the three-index integrals
use the same operator. With that choice the residual is $\hat w$-orthogonal
to the auxiliary space, and the error in (@eq:factorization-and-metric-1) is the interaction of two
residuals, second order in the fitting residual (section @sec:closed-shell-ri-mp2, equation @eq:df-quadratic-error;
section @sec:robust-fit derives it and shows what happens when the kernels differ).
A fit under another kernel loses the orthogonality. If the fitted densities
are then evaluated under the physical kernel, the cross terms
$(\tilde\rho|\hat w|\delta)$ survive and the error is first order in the
residual. If only the metric is swapped, $J^{\mathsf T}V_w^{-1}J$, the result
is wrong even for a density the auxiliary basis spans
(section @sec:robust-fit-robust-form). In the code the default is
`RiMp2Config::metric_op = None`, which means the metric uses the integral
operator.

`ferric-integrals/src/threeindex.rs::coulomb_metric_2c(op, dfbs)` builds $V$
for any operator, Coulomb or not. Here is its serial branch, with line
comments added (the parallel branch over shells is removed):

```rust
let mut eng = Engine::new_2center(op, dfbs, 1e-14)?;   // (P|w|Q) engine for op
for sp in 0..nsh {
    for sq in 0..=sp {                                 // unique shell pairs P >= Q
        let block = eng.compute_eri2(dfbs, sp, sq);
        let np = dims[sp];
        let nq = dims[sq];
        for p in 0..np {
            for q in 0..nq {
                let val = block[p * nq + q];
                v[(offs[sp] + p, offs[sq] + q)] = val;   // V_PQ
                v[(offs[sq] + q, offs[sp] + p)] = val;   // V_QP = V_PQ
            }
        }
    }
}
```

The operator lives in the engine, so the same loop builds the Coulomb, erf,
erfc, terf, terfc and Yukawa metrics. Each unique shell pair $P \ge Q$ is
computed once and written into both triangles.

## Any factor gives the same energy {#factorization-and-metric-any-factor-gives-same}

$M$ appears in the derivation above only through $M^{\mathsf T}M$. So
$B^{\mathsf T}B = (pq|P)\,V^{-1}\,(Q|rs)$, the four-index approximation, is
the same for every $M$ with $M^{\mathsf T}M = V^{-1}$. Two such factors differ
by an orthogonal matrix on the left, $B \to OB$, and
$(OB)^{\mathsf T}(OB) = B^{\mathsf T}O^{\mathsf T}OB = B^{\mathsf T}B$.

The same holds for every energy built from the integrals, including those
that are not a polynomial in them. The dRPA energy depends on $B$ only
through $B^{\mathsf T}B$, because $\det(1 + B G B^{\mathsf T}) =
\det(1 + G B^{\mathsf T}B)$ for any diagonal $G$ (Sylvester's identity).

> **Verified equation {#eq:df-factor-freedom}** · the energy depends only on $B^{\mathsf T}B$
>
> The closed-shell dRPA energy computed from $B = L^{-1}(Q|ia)$ and from
> $B = V^{-1/2}(Q|ia)$ is the same, and the log-determinant is invariant under
> $B \to OB$ for orthogonal $O$.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_drpa_pdep_sympy.py`, checks `ri-factor-freedom` and `sylvester` |
> | Negative control | the factor applied from the wrong side, $M = L^{-\mathsf T}$ (so $M^{\mathsf T}M = L^{-1}L^{-\mathsf T} \ne V^{-1}$), turns `ri-factor-freedom` red (recorded mutation) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/rimp2.rs::metric_inverse_sqrt` |
>
> <!-- verified: id=df.factor-freedom; script=scripts/rpa_drpa_pdep_sympy.py; claim=ri-factor-freedom; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/rimp2.rs::{metric_inverse_sqrt,cholesky_inverse_sqrt,eigh_inverse_sqrt} -->

The dissertation and most of the literature write $M = V^{-1/2}$, the
symmetric inverse square root (dissertation chapter 1, section 1.4.1, and
chapter 4, Eq. 4.5). The code uses two factors and picks one by operator.

## The two factors {#factorization-and-metric-factors}

| Factor | $M$ | $M^{\mathsf T}M$ | Cost | Function |
|---|---|---|---|---|
| Cholesky | $L^{-1}$, $V = LL^{\mathsf T}$ | $V^{-1}$ | one factorization and a triangular inverse | `cholesky_inverse_sqrt` |
| Eigendecomposition | $U_{\mathrm{k}}\lambda_{\mathrm{k}}^{-1/2}U_{\mathrm{k}}^{\mathsf T}$ over eigenvalues $\lambda \ge 10^{-10}$ | $V^{+}$, the pseudo-inverse on the kept modes | one symmetric eigendecomposition | `eigh_inverse_sqrt` |

`ferric-mp2/src/rimp2.rs::metric_inverse_sqrt(v, op)` chooses between them by
operator, and refuses terf, whose metric neither factor can represent
(section @sec:factorization-and-metric-signed):

<!-- excerpt: ferric-mp2/src/rimp2.rs::metric_inverse_sqrt -->
```rust
match op.kind {
    OperatorKind::Terf => Err(FerricError::General(
        "metric_inverse_sqrt: the terf kernel is not positive definite, so its RI \
         metric is indefinite and has no inverse square root; form the terf \
         interaction as Coulomb RI minus terfc RI (a signed factor) instead"
            .into(),
    )),
    OperatorKind::ErfCoulomb => eigh_inverse_sqrt(v),
    _ => cholesky_inverse_sqrt(v),
}
```

**Cholesky is the default.** $M = L^{-1}$ satisfies the condition of
(@eq:factorization-and-metric-1):
$(L^{-1})^{\mathsf T}L^{-1} = (LL^{\mathsf T})^{-1} = V^{-1}$. Section
@sec:closed-shell-ri-mp2-density-fitting-making-affordable gives the cost
argument for preferring it to $V^{-1/2}$. Coulomb, erfc and terfc
(section @sec:erf-terf) and Yukawa (section @sec:yukawa-geminal) have
positive definite kernels, so their
metric is positive definite and the factorization exists. A Cholesky failure
on one of them is an error, and the run stops: it means an integral or
auxiliary-basis defect, not near-dependence to be regularized away.

After LAPACK returns $L$, `ferric-mp2/src/rimp2.rs::cholesky_inverse_sqrt`
inverts it by forward substitution. Row $i$ of $L L^{-1} = I$ reads
$\sum_{k \le i} L_{ik}X_{kj} = \delta_{ij}$, and $X_{kj} = 0$ for $k < j$
because $X = L^{-1}$ is lower triangular. Solve for $X_{ij}$:

$$ X_{ii} = \frac{1}{L_{ii}}, \qquad X_{ij} = -\frac{1}{L_{ii}}\sum_{k=j}^{i-1} L_{ik}X_{kj} \quad (j < i) . $$

The loop is that formula, with the BLAS-thread wrapper and the error mapping
on the `cholesky` call removed:

```rust
let l = v.cholesky(UPLO::Lower)?;                // V = L L^T
let mut l_inv = Array2::zeros((n, n));
for i in 0..n {
    l_inv[(i, i)] = 1.0 / l[(i, i)];            // X_ii = 1 / L_ii
    for j in (0..i).rev() {
        let mut sum = 0.0;
        for k in j..i {
            sum += l[(i, k)] * l_inv[(k, j)];    // sum_{k=j}^{i-1} L_ik X_kj
        }
        l_inv[(i, j)] = -sum / l[(i, i)];        // X_ij
    }
}
```

The triangular solve is scalar and serial on purpose; only the factorization
may use threaded BLAS.

**The eigendecomposition is for erf.** The erf kernel is positive
definite but smooth, so tight auxiliary functions in a Coulomb-fitted basis
give near-zero eigenvalues that round to either sign.
`ferric-mp2/src/rimp2.rs::eigh_inverse_sqrt` diagonalizes
$V = U\lambda U^{\mathsf T}$, drops every mode with $\lambda < 10^{-10}$
(`LINDEP_THRESH`), whatever its sign, and scales the rest by
$\lambda^{-1/2}$. Here is the loop, unchanged apart from the threading
wrapper on the last product:

```rust
let mut u_scaled = evecs.clone();
for k in 0..n {
    if evals[k] < LINDEP_THRESH {
        for r in 0..n {
            u_scaled[(r, k)] = 0.0;               // drop mode k
        }
    } else {
        let s = 1.0 / evals[k].sqrt();
        for r in 0..n {
            u_scaled[(r, k)] *= s;                // U_k lambda_k^{-1/2}
        }
    }
}
Ok(u_scaled.dot(&evecs.t()))                      // U_k lambda_k^{-1/2} U_k^T
```

The result is symmetric. $M^{\mathsf T}M = U_{\mathrm{k}}\lambda_{\mathrm{k}}^{-1}U_{\mathrm{k}}^{\mathsf T}$
is the pseudo-inverse $V^{+}$ restricted to the kept modes, not $V^{-1}$. The
fit is exact on the span of the kept modes and blind to the dropped ones.
For a positive definite kernel the dropped modes carry rounding only. A
negative eigenvalue larger than $10^{-10}$ in magnitude cannot come from a
positive definite kernel; it is printed as a NOTE (an integral defect is the
usual cause) and dropped like the others.

**Orientation.** $L^{-1}$ is lower triangular and not symmetric, so a factor
used on the other side of a product has to be transposed. Where a code path
maps a vector $u$ in the dressed space back to auxiliary-function
coefficients, it forms $c = M^{\mathsf T}u$. The PDEP eigenpotentials are the
example (section @sec:pdep): with the symmetric factor the transpose is invisible,
and with $L^{-1}$ it is the difference between a right and a wrong
eigenpotential.

## A kernel that is not positive definite {#factorization-and-metric-signed}

The fit of section @sec:factorization-and-metric-factorization needs $V$ invertible, and the
Cholesky factor needs more: $V$ positive definite. The terf kernel has
neither guarantee. Its transform $\cos(kr_0)\,e^{-k^2/4\omega^2}$ is negative
near $kr_0 = \pi$ (section @sec:erf-terf-properties-decide-integrals-used), so the metric
$V_{\mathrm t} = (P|\operatorname{terf}|Q)$ has negative eigenvalues in any
auxiliary basis that can describe a density modulated at that wavelength,
and eigenvalues that cross zero between the two signs. The error functional
$\mathcal E_x(c)$ is then unbounded below, and $c_x = V^{-1}A_x$ is a saddle point of
it, not a minimum.

Neither factor of section @sec:factorization-and-metric-factors applies. Cholesky fails. The
eigendecomposition factor drops the negative modes together with the
near-null ones, and $M^{\mathsf T}M$ becomes the pseudo-inverse of the
positive part $V_{\mathrm t,+}$. Take a density in the auxiliary span,
$\rho_x = \sum_P d_{Px}\chi_P$. Then $A_x = V_{\mathrm t}d_x$, and the fitted
interaction is $d_x^{\mathsf T}V_{\mathrm t,+}d_y$ where the exact one is
$d_x^{\mathsf T}V_{\mathrm t}d_y$: it is the interaction under a different,
positive definite kernel. A signed pseudo-inverse, which keeps the negative
modes with their sign, is exact in that limit, but it inverts the modes that
cross zero, and its value depends on where the threshold cuts them.

ferric never fits terf in its own metric. It applies the split
$\operatorname{terf} = 1 - \operatorname{terfc}$ of (@eq:erf-terf-2) to the fitted
integrals:

$$ (ia|\operatorname{terf}|jb) \approx \sum_P B^P_{\mathrm C,ia}B^P_{\mathrm C,jb} - \sum_P B^P_{\mathrm s,ia}B^P_{\mathrm s,jb} = \big(\mathbf B^{\mathsf T}\boldsymbol\Sigma\,\mathbf B\big)_{ia,jb}, \qquad \mathbf B = \begin{pmatrix}\mathbf B_{\mathrm C}\\ \mathbf B_{\mathrm s}\end{pmatrix}, \quad \boldsymbol\Sigma = \begin{pmatrix}\mathbf 1 & 0\\ 0 & -\mathbf 1\end{pmatrix} . \label{eq:factorization-and-metric-signed} $$

$\mathbf B_{\mathrm C}$ and $\mathbf B_{\mathrm s}$ are the Cholesky-dressed Coulomb
and terfc factors of (@eq:factorization-and-metric-1). Both metrics are positive definite, each
fit is the stationary fit of its own kernel, and each error is second order
in its own residual. For a density in the auxiliary span the right side is
$d_x^{\mathsf T}(V_{\mathrm C} - V_{\mathrm s})d_y = d_x^{\mathsf T}V_{\mathrm t}d_y$,
exact. $\mathbf B$ has $2N_{\mathrm{aux}}$ rows and $\boldsymbol\Sigma$ is its
signature. No real factor with $\boldsymbol\Sigma = \mathbf 1$ can represent an
indefinite matrix, so the sign is part of the factor.

What does each energy need from it? MP2 needs only the integrals. The pair
block of occupied orbital $i$ is
$G_i = \mathbf B_{\mathrm C,i}^{\mathsf T}\mathbf B_{\mathrm C} - \mathbf B_{\mathrm s,i}^{\mathsf T}\mathbf B_{\mathrm s}$
(`signed_pair_block`), two GEMMs where the one-factor kernel has one, and the
pair loop that consumes $G_i$ is unchanged. The dRPA energy goes through
Sylvester's identity, which holds with $\boldsymbol\Sigma$ inside. For a
diagonal $G \ge 0$, move $G^{1/2}\mathbf B^{\mathsf T}$ from the left of the
product to the right:

$$ \det\big(\mathbf 1_{ov} + G^{1/2}\mathbf B^{\mathsf T}\boldsymbol\Sigma\,\mathbf B\,G^{1/2}\big) = \det\big(\mathbf 1_{2N_{\mathrm{aux}}} + \boldsymbol\Sigma\,\mathbf B G\mathbf B^{\mathsf T}\big), \qquad \operatorname{tr}\big(G\,\mathbf B^{\mathsf T}\boldsymbol\Sigma\,\mathbf B\big) = \operatorname{tr}\big(\boldsymbol\Sigma\,\mathbf B G\mathbf B^{\mathsf T}\big) . \label{eq:factorization-and-metric-signed-sylvester} $$

With $G = -\chi_0(iu)$, the diagonal of $4\Delta_{ia}/(u^2 + \Delta_{ia}^2)$
for a closed shell (the book's $\chi_0$ is negative), the left sides are the
dRPA integrand in the excitation space, so the right sides give it in a space of $2N_{\mathrm{aux}}$
dimensions. $\boldsymbol\Sigma\mathbf B G\mathbf B^{\mathsf T}$ is not symmetric, so
`eval_signed_trace_log_summands` takes the log-determinant by LU, as the
energy-only dielectric route of chapter @sec:rpa already does, and
needs no eigenvalues. The frequency integral of $\ln\det - \operatorname{tr}$
equals the plasmon formula for an indefinite kernel too, as long as the
determinant stays positive. For one excitation of energy $\Delta$ coupled by
$k$ (so $A = \Delta + 2k$, $B = 2k$, plasmon frequency
$\Omega = \sqrt{\Delta(\Delta + 4k)}$), the integral is
$\tfrac12(\Omega - \Delta - 2k)$ for every $k > -\Delta/4$. Section
@sec:two-formulations-terf-sign proves this one-mode case with the coupling
written as $2k$. The LU
routine refuses a nonpositive determinant. The static-eigenbasis route of
chapter @sec:rpa diagonalizes a symmetric $\epsilon(0)$ and keeps its
modes by $|\lambda - 1|$; it has no meaning for a signed factor, and the code
refuses truncation, the Laplace $\chi_0$ backend and the Boys-screened
$\chi_0$ for it.

> **Verified equation {#eq:df-signed-factor}** · the signed factor is the terf interaction
>
> $\mathbf B^{\mathsf T}\boldsymbol\Sigma\mathbf B = \mathbf B_{\mathrm C}^{\mathsf T}\mathbf B_{\mathrm C} - \mathbf B_{\mathrm s}^{\mathsf T}\mathbf B_{\mathrm s}$
> reproduces $d^{\mathsf T}V_{\mathrm t}d$ exactly for densities in the
> auxiliary span on s-Gaussian Gram matrices with an indefinite
> $V_{\mathrm t}$, where the positive-subspace pseudo-inverse misses it; the
> signed LU log-determinant integrated over frequency equals the plasmon
> dRPA energy of the indefinite kernel, closed and open shell.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_terf_signed_fit_sympy.py`, checks `signed-factor`, `signed-is-exact`, `drop-misses`, `logdet-vs-plasmon`, `u-logdet-vs-plasmon` |
> | Negative control | the runner gives the terfc block signature $+1$, keeps only positive modes in the signed pseudo-inverse, and halves the closed-shell $\chi_0$ prefactor; each turns its check red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/rimp2.rs::signed_pair_block`, `ferric-rpa/src/energy.rs::eval_signed_trace_log_summands` |
> | Code test | `terf_lr_matches_independent_constructions_ne2` in `crates/ferric-rpa/tests/terf_signed_fit.rs` (needs the terf tables; skips without them) |
>
> <!-- verified: id=df.signed-factor; script=scripts/rpa_terf_signed_fit_sympy.py; claim=signed-is-exact; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/rimp2.rs::{signed_pair_block,spin_components_from_signed_b_ov}; rust_test=ferric-rpa/tests/terf_signed_fit.rs::terf_lr_matches_independent_constructions_ne2 -->

The Rust test builds four fits of the terf interaction on Ne$_2$ from the raw
integrals: Coulomb RI minus terfc RI, the signed pseudo-inverse of
$V_{\mathrm t}$, Coulomb-fitted densities with $V_{\mathrm t}$ between them,
and the positive-subspace fit. The first three agree on the counterpoise
interaction to within their fitting differences, and the production code
equals the first to rounding. The positive-subspace fit misses by more than
the interaction itself. `scripts/gen_terf_signed_fit_refs.py` records the
numbers in `testdata/reference/ne2_terf_signed_fit.json`.

What does it cost? The terfc factor and the Coulomb factor are both built
anyway by RS-MP2 + LR-RPA (chapter @sec:rs-mp2-rpa), so the terf three-index
build disappears. The Coulomb factor stays resident across the scan of
$\omega$ points, one more $N_{\mathrm{aux}}\times n_on_v$ plane, and the
log-determinant is $2N_{\mathrm{aux}}$ wide, eight times the LU work of one
fit.

## Algorithm {#factorization-and-metric-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | $V_{PQ} = (P\vert\hat w\vert Q)$ | $N_{\mathrm{aux}}^2$ integrals | `coulomb_metric_2c` |
| 2 | $M$: Cholesky or eigendecomposition by operator; terf refused | $N_{\mathrm{aux}}^3$ | `metric_inverse_sqrt` |
| 3 | $B = M\,(Q\vert pq)$ for the orbital blocks a method needs | $N_{\mathrm{aux}}^2\,n_pn_q$ | per method (section @sec:closed-shell-ri-mp2 for $B_{ov}$) |

Step 2 runs once per intermediates build, outside any parallel region. Both
factorizations may use threaded BLAS there.

## Options {#factorization-and-metric-options}

| Option | Effect |
|---|---|
| Auxiliary basis | the fitting space; the RI sets bundled with ferric are listed in `crates/ferric-core/src/basis/bundled` |
| Operator | chooses the metric operator and, through it, the factor; a terf metric is an error |
| `RiMp2Config::metric_op` (library) | a metric operator different from the integral operator; a measurement option. It evaluates $J^{\mathsf T}V_w^{-1}J$, whose error does not vanish with the fitting residual (section @sec:robust-fit) |

## Numerical anchors {#factorization-and-metric-numerical-anchors}

The factor choice has no anchor of its own; every RI anchor in the book
exercises it. The RI-MP2 energy anchors (section @sec:closed-shell-ri-mp2) use the Cholesky
factor. The attenuated PDEP anchors cover both factors: erf through the
eigendecomposition ($\le 5.8\times10^{-11}$ Ha) and erfc through Cholesky
($3.5\times10^{-11}$ Ha), each against an independent reference
(`site/src/reference/validation.md`).

## Sources {#factorization-and-metric-sources}

| Source | What it gives |
|---|---|
| section @sec:closed-shell-ri-mp2-density-fitting-making-affordable | the cost argument for the Cholesky default |
| `scripts/rpa_drpa_pdep_sympy.py` | factor freedom (checks `ri-factor-freedom`, `sylvester`) |
| Dissertation, chapter 1 (section 1.4.1) and chapter 4 (Eq. 4.5) | the RI approximation with $V^{-1/2}$ |

**Known limits (2026-11-15).** No Rust test compares the two factors on one
system. Each factor is anchored on its own kernel (erf and erfc above), and
the script proves the equivalence.
