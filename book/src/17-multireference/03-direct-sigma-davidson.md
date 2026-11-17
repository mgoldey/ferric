# Direct sigma and the Davidson solver {#direct-sigma-davidson}

<!-- status: ready; registry: casci;
     theory: crates/ferric-ci/theory.md#"Direct sigma (`src/sigma.rs`)" -->

A CAS(14,14) singlet has about $1.2\times10^7$ determinants, and its
Hamiltonian stored as a dense matrix would take $8N_\text{det}^2 \approx
1.1\times10^{15}$ bytes. Davidson iteration finds the lowest root without
it: all it needs from $H$ is the product $\sigma = HC$ and the diagonal.
The direct sigma builds that product from single-excitation tables over α
and β strings, so it never forms $H$ and never loops over determinant
pairs. Its cost is set by the number of single excitations per string, not
by $N_\text{det}^2$.

| Symbol | Meaning | Code name |
|---|---|---|
| $C(I_\alpha, I_\beta)$ | CI vector as an α-string × β-string matrix | `c` (row-major, α rows) |
| $E^\sigma_{pq}$ | $a^\dagger_{p\sigma}a_{q\sigma}$ | |
| $g_{pq}$ | $h_{pq} - \tfrac12\sum_r(pr|rq)$ | `one_body_g` |
| $\sigma_1, \sigma_2, \sigma_3$ | ββ, αα and αβ parts of $HC$ | `DirectSigma::apply` |

| Entry point | Location |
|---|---|
| Rust | `ferric-ci/src/sigma.rs::DirectSigma`, `ferric-ci/src/davidson.rs::davidson_lowest`, `ferric-ci/src/lib.rs::{run_cas_ci,run_cas_ci_on_integrals,dense_lowest}` |
| Python | `run_casci` |

## The split of $H$ {#direct-sigma-davidson-split}

### Rewriting $H$ with one-body operators

The plan is to write $H$ entirely with the one-spin excitation operators
$E^\sigma_{pq} = a^\dagger_{p\sigma}a_{q\sigma}$, because each of them maps
one string to one string with a single phase. Start from the product of two
of them and move $a_{q\sigma}$ to the right with the anticommutator
$a_{q\sigma}a^\dagger_{r\tau} = \delta_{qr}\delta_{\sigma\tau} - a^\dagger_{r\tau}a_{q\sigma}$:

$$
E^\sigma_{pq}E^\tau_{rs}
= a^\dagger_{p\sigma}\big(\delta_{qr}\delta_{\sigma\tau} - a^\dagger_{r\tau}a_{q\sigma}\big)a_{s\tau}
= \delta_{qr}\delta_{\sigma\tau}E^\sigma_{ps} + a^\dagger_{p\sigma}a^\dagger_{r\tau}a_{s\tau}a_{q\sigma} ,
$$

where the last step swapped $a_{q\sigma}$ and $a_{s\tau}$ (one sign).
Solve for the two-electron string and substitute it into $H$ of section
@sec:determinants-hamiltonian. The $\delta$ term gives $-\tfrac12\sum_{pqs,\sigma}(pq|qs)E^\sigma_{ps}$,
a one-electron operator. Rename its indices and add it to $h$:

$$
H - E_\text{core} = \sum_{pq,\sigma} g_{pq} E^\sigma_{pq}
  + \tfrac12 \sum_{pqrs,\sigma\tau} (pq|rs)\, E^\sigma_{pq} E^\tau_{rs},
\qquad
g_{pq} = h_{pq} - \tfrac12 \sum_r (pr|rq).
$$

Now split the spin sum. The $\sigma = \tau$ terms stay as they are. The
$\alpha\beta$ and $\beta\alpha$ terms are equal: $E^\alpha$ and $E^\beta$
commute, and $(pq|rs) = (rs|pq)$. They merge into one term with
coefficient 1:

$$
H - E_\text{core} = \sum_{pq,\sigma} g_{pq} E^\sigma_{pq}
  + \tfrac12 \sum_{pqrs,\sigma} (pq|rs)\, E^\sigma_{pq} E^\sigma_{rs}
  + \sum_{pqrs} (pq|rs)\, E^\alpha_{pq} E^\beta_{rs}.
$$

Check one orbital with two electrons. $E^\alpha_{00}$ and $E^\beta_{00}$
are both 1, so the energy is
$2g_{00} + \tfrac12(00|00)\cdot2 + (00|00) = 2h_{00} - (00|00) + 2(00|00)
= 2h_{00} + (00|00)$, the closed-shell energy of one doubly occupied
orbital.

### Three string contractions

Write the CI vector as the matrix $C(I_\alpha, I_\beta)$. Then
$\sigma = \sigma_1 + \sigma_2 + \sigma_3 + E_\text{core} C$ with

$$
\begin{aligned}
\sigma_1(I_\alpha,I_\beta) &= \sum_{J_\beta} \langle I_\beta|H_\beta|J_\beta\rangle\, C(I_\alpha,J_\beta), \\
\sigma_2(I_\alpha,I_\beta) &= \sum_{J_\alpha} \langle I_\alpha|H_\alpha|J_\alpha\rangle\, C(J_\alpha,I_\beta), \\
\sigma_3(I_\alpha,I_\beta) &= \sum_{pqrs} (pq|rs) \sum_{J_\alpha J_\beta}
  \langle I_\alpha|E_{pq}|J_\alpha\rangle \langle I_\beta|E_{rs}|J_\beta\rangle\, C(J_\alpha,J_\beta),
\end{aligned}
$$

where $H_\sigma$ is the first two terms of $H - E_\text{core}$ restricted
to spin σ. $\sigma_1$ and $\sigma_2$ act on one index of $C$ each, like a
matrix multiplied from one side. $\sigma_3$ needs the factorization
$\langle I_\alpha I_\beta|E^\alpha_{pq}E^\beta_{rs}|J_\alpha J_\beta\rangle
= \langle I_\alpha|E_{pq}|J_\alpha\rangle\langle I_\beta|E_{rs}|J_\beta\rangle$.
It holds in the α-before-β order because $E^\beta_{rs}$ carries
$(-1)^{N_\alpha}$ twice (section @sec:determinants-convention).

> **Verified equation {#eq:ci-sigma-split}** · the three-part split of $H$
>
> As $64\times64$ Jordan–Wigner matrices on three active orbitals with
> random symmetric $h$ and 8-fold-symmetric $(pq|rs)$,
> $H - E_\text{core} = \sum g_{pq}E_{pq} + \tfrac12\sum_\sigma(pq|rs)E^\sigma_{pq}E^\sigma_{rs} + \sum(pq|rs)E^\alpha_{pq}E^\beta_{rs}$
> with $g_{pq} = h_{pq} - \tfrac12\sum_r(pr|rq)$.
>
> | | |
> |---|---|
> | Proof | `scripts/ci_rdm_casscf_gradient_sympy.py`, check `sigma-split` |
> | Negative control | the runner's mutation: the $\tfrac12$ in $g$ set to 1 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-ci/src/sigma.rs::{one_body_g,DirectSigma}` |
> | Code test | `direct_sigma_matches_naive_on_random_integrals` in `crates/ferric-ci/tests/direct_sigma_vs_naive.rs` |
>
> <!-- verified: id=ci.sigma.split; script=scripts/ci_rdm_casscf_gradient_sympy.py; claim=sigma-split; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-ci/src/sigma.rs::{one_body_g,DirectSigma}; rust_test=ferric-ci/tests/direct_sigma_vs_naive.rs::direct_sigma_matches_naive_on_random_integrals; registry=casci -->

The box proves the operator identity. The phases of the excitation tables
are those of section @sec:slater-condon, and the Rust test compares the whole product
element-wise with the `general_element` pair loop.

### The excitation tables

Every factor in $\sigma_1, \sigma_2, \sigma_3$ is a single excitation in one
channel, and comes from one table per channel. For each string $S$, each
occupied $q$ and each $p$ that is empty in $S$ or equal to $q$:

$$
E_{pq}|S\rangle = \epsilon\,|T\rangle,\qquad
\epsilon = \text{pb}(S,q)\;\text{pb}(S\setminus q,\,p),\qquad
T = (S\setminus q)\cup p .
$$

The phase is $\epsilon_1$ of section @sec:slater-condon. A string with $k$ electrons
has $k(n-k)$ entries with $p \ne q$ and $k$ with $p = q$, so $k(n-k)+k$ in
all: 12 per string for $k = 3$, $n = 6$. `DirectSigma::new` builds the
tables once per CAS-CI solve with this loop of
`ferric-ci/src/sigma.rs::excitation_lists` (the string index map elided):

```rust
for q in crate::strings::occupied_orbitals(s) {
    let s_minus = s & !(1u64 << q);
    let pb_q = phase_below(s, q);                   // pb(S, q)
    for p in 0..n {
        if p != q && s & (1u64 << p) != 0 {
            continue;                               // p occupied, p ≠ q: E_pq|S⟩ = 0
        }
        let t = s_minus | (1u64 << p);              // T = (S \ q) ∪ p
        out.push(Excitation {
            pq: (p * n + q) as u32,
            target: index[&t],
            sign: pb_q * phase_below(s_minus, p),   // ε = pb(S, q) pb(S \ q, p)
        });
    }
}
```

### Contracting over the output string

The table lists excitations out of a string, $E_{pq}|I\rangle$, while
$\sigma(I)$ needs $\langle I|\cdots|J\rangle$ for all $J$. With real
symmetric integrals $\langle J|E_{pq}|I\rangle = \langle I|E_{qp}|J\rangle$,
and $H$ is symmetric. So the excitations OF the output string $I$ give
everything $\sigma(I)$ needs:

* $\sigma_2$, per row $I_\alpha$: for each $E_{tu}|I_\alpha\rangle=\epsilon_1|K\rangle$
  add $\epsilon_1 g_{tu}$ to $F(K)$, and for each
  $E_{vw}|K\rangle=\epsilon_2|J\rangle$ add $\tfrac12\epsilon_1\epsilon_2 (vw|tu)$
  to $F(J)$. Then $F(J) = \langle J|H_\alpha|I_\alpha\rangle$, and
  $\sigma(I_\alpha,:) \mathrel{+}= \sum_J F(J)\, C(J,:)$.
* $\sigma_1$: the same on $C^\mathsf{T}$ with the β table.
* $\sigma_3$, per β pair $rs$: gather $C'(J_\alpha,m) = \epsilon_m C(J_\alpha, J_{\beta,m})$
  over the β strings with $E_{rs}|J_{\beta,m}\rangle = \epsilon_m|I_{\beta,m}\rangle$.
  Per row $I_\alpha$, form $V(m) = \sum \epsilon\,(pq|rs)\,C'(J_\alpha,m)$ over
  $E_{pq}|I_\alpha\rangle = \epsilon|J_\alpha\rangle$. That entry is
  $\langle I_\alpha|E_{qp}|J_\alpha\rangle = \epsilon$, and $(qp|rs)=(pq|rs)$.
  Then $\sigma(I_\alpha, I_{\beta,m}) \mathrel{+}= V(m)$.

The $\sigma_2$ step is `ferric-ci/src/sigma.rs::same_spin`. Its inner loop, with the bookkeeping of
the `touched` list removed:

```rust
for e1 in &lists[i] {                                // E_tu|I⟩ = ε₁|K⟩
    let k = e1.target as usize;
    let kl = e1.pq as usize;
    // ...
    f[k] += e1.sign * g[kl];                         // ε₁ g_tu
    for e2 in &lists[k] {                            // E_vw|K⟩ = ε₂|J⟩
        let j = e2.target as usize;
        // ...
        f[j] += 0.5 * e1.sign * e2.sign * eri[e2.pq as usize * n2 + kl]; // ½ ε₁ε₂ (vw|tu)
    }
}
```

Two points about it:

- **$F$ is one row of $H_\alpha$.** For a fixed output string $I$, `f` holds
  $\langle J|H_\alpha|I\rangle$ for every $J$ reachable by one or two
  excitations. The row is then applied to all β columns of $C$ at once, a
  row-times-matrix product.
- **The second excitation starts from $K$, not $I$.** That is the order of
  $E_{vw}E_{tu}$ acting on $|I\rangle$, and it is why the coefficient is
  $\tfrac12\epsilon_1\epsilon_2$ with the integral index $(vw|tu)$.

**Cost.** One call costs $O(N_\text{det}\, n_\text{exc}^\alpha n_\text{exc}^\beta)$
for $\sigma_3$, with $n_\text{exc} = k(n-k)+k$, and no determinant pair is
visited. $\sigma_1$ and $\sigma_2$ cost $O(N_\text{det}\,n_\text{exc})$
plus the same-spin double fan-out. Rows of $\sigma$ are owned by one rayon
task each, so the result does not depend on the thread count. Scratch
beyond the Davidson vectors is four $N_\text{det}$ vectors (the output,
$C^\mathsf{T}$, $\sigma_1^\mathsf{T}$ and $C'$) and the tables,
$16(n_\text{str}^\alpha n^\alpha_\text{exc} + 2 n_\text{str}^\beta n^\beta_\text{exc})$
bytes. Each table entry is 16 bytes, and the β table is stored twice: per
string and grouped by pair $rs$ for $\sigma_3$.

## The Davidson solver {#direct-sigma-davidson-davidson-solver}

`davidson_lowest` is a block-size-1 Davidson for the lowest root.

**Where the correction comes from.** Suppose the subspace gives a Ritz pair
$(\theta, x)$ with residual $r = Hx - \theta x$. We want a correction $t$
with $(H - \theta)(x + t) \approx 0$, that is $(H - \theta)t \approx -r$.
Replace $H$ by its diagonal, the only part of $H$ that can be inverted for
free:

$$
t_i = -\frac{r_i}{H_{ii} - \theta} .
$$

The code computes $t_i = r_i/(H_{ii} - \theta)$, without the minus. The
subspace spanned by $t$ is the same either way, and Rayleigh–Ritz picks the
coefficient. A denominator near zero (a determinant whose diagonal energy is
close to $\theta$) is held at magnitude $10^{-8}$:

```rust
let denom = diag[i] - theta;                  // H_ii − θ
let denom = if denom.abs() < 1e-8 {
    if denom < 0.0 {
        -1e-8
    } else {
        1e-8
    }
} else {
    denom
};
t[i] = resid[i] / denom;                      // t_i = r_i / (H_ii − θ)
```

The steps:

| Step | Operation |
|---|---|
| 1 | guess: the unit vector on the determinant with the lowest diagonal |
| 2 | Rayleigh–Ritz on the subspace, Ritz pair $(\theta, x)$, residual $r = Hx - \theta x$ |
| 3 | stop when $\lVert r\rVert <$ `conv_thresh` |
| 4 | correction $t_i = r_i/(H_{ii} - \theta)$, denominators kept at least $10^{-8}$ in magnitude |
| 5 | two Gram–Schmidt passes against the subspace; a correction that shrinks below $10^{-10}$ of its norm is already spanned, and the solve stops unconverged |
| 6 | at `max_subspace`, collapse to the Ritz vector and the new correction |

Step 5 measures the shrinkage relative to the correction's own norm before
orthogonalization. The correction scales with the residual, so an absolute
cut would stop the solve as soon as the residual reached it.

$E_\text{core}$ is folded into the diagonal, so the eigenvalue is the total
CAS-CI energy.

**Dense fallback** (`ferric-ci/src/lib.rs::dense_lowest`). Davidson's
residual floor is set by its Gram–Schmidt round-off and can sit above a
tight threshold. When Davidson stops without meeting `conv_thresh` and the
space has at most `DENSE_SOLVE_MAX_DETS` = 1024 determinants, the dense $H$
is diagonalized (LAPACK `eigh`) and its root kept if its residual is
smaller. Thresholds below Davidson's reach are then met on small spaces.
The result's `residual_norm` is the recomputed $\lVert Hc - Ec\rVert$.
`converged` is Davidson's own test, $\lVert r\rVert <$ `conv_thresh` on its
Ritz residual, or, when the dense root replaces the Davidson one, the
recomputed residual below `conv_thresh`.

**Memory.** The Davidson subspace is the peak:
$2\cdot\texttt{max\_subspace}\cdot N_\text{det}\cdot 8$ bytes (the basis and its
products). For N2 in a CAS(6,6) with the default 50 vectors that is 320 kB;
for a CAS(14,14) singlet it is 9.4 GB. The CAS-CI memory plan charges it,
and the RDM scratch when `compute_rdms` is set, before anything is
allocated.

## Options {#direct-sigma-davidson-options}

| Option | Values | Default (Python) | Effect |
|---|---|---|---|
| `n_active`, `n_elec_active` | integer, $(n_\alpha, n_\beta)$ | required | the active space and its $M_s$ sector |
| `active_start` | integer | $(N - n_\alpha - n_\beta)/2$ | first active MO; the orbitals below it are the core |
| `conv_thresh` | residual norm | $10^{-9}$ | Davidson and dense-solve threshold |
| `max_subspace` | integer | 50 | subspace size before a collapse |
| `max_iter` | integer | 200 | Davidson iteration cap |
| `memory_budget_gb` | GB | auto | the CAS-CI memory plan |

## Numerical anchors {#direct-sigma-davidson-numerical-anchors}

| Quantity | Systems | Error | Test |
|---|---|---|---|
| $E_\text{CASCI}$ vs PySCF `mcscf.CASCI` | N2 CAS(6,6) / cc-pVDZ at 1.10 and 2.20 Å; H2O CAS(4,4) / 6-31G; each also with the window moved down one orbital | $\le 1.4\times10^{-11}$ Ha | `crates/ferric-ci/tests/validation_casci.rs` |
| $E_\text{core}$ and the active-space energy | same | $\le 2.6\times10^{-10}$ Ha | `crates/ferric-ci/tests/validation_casci.rs` |
| direct `sigma` vs `sigma_naive` (the `general_element` pair loop) on random vectors | random 8-fold-symmetric integrals in ten sectors with $N_\alpha \ne N_\beta$, including empty and full channels; real H2O/STO-3G integrals for full CI (441 determinants), 8 electrons in 6 orbitals above a frozen O 1s (225 determinants) and four unequal-spin sectors | element-wise to $10^{-12}$ | `crates/ferric-ci/tests/direct_sigma_vs_naive.rs` |
| `hamiltonian_diagonal` vs the diagonal of the dense $H$; $\sigma$ of unit vectors vs the dense columns | same | element-wise to $10^{-12}$ | `crates/ferric-ci/tests/direct_sigma_vs_naive.rs` |

## Sources {#direct-sigma-davidson-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-ci/theory.md`, "Direct sigma" | the split, the tables, the three contractions, cost and memory |
| Knowles and Handy, CPL 111, 315 (1984); Olsen, Roos, Jørgensen and Jensen, JCP 89, 2185 (1988) | the string-driven direct CI |
| `scripts/ci_rdm_casscf_gradient_sympy.py` | `sigma-split` |

**Known limits (2026-11-15).** Lowest root of one $(n_\alpha, n_\beta)$
sector only: no multiple roots, no state averaging and no $S^2$ handling.
Rows of $\sigma$ are owned by one rayon task each, so the product does not
depend on the thread count.
