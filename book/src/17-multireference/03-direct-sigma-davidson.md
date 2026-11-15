# 17.3 Direct sigma and the Davidson solver

<!-- section: 17.3; status: ready; registry: casci;
     theory: crates/ferric-ci/theory.md#"Direct sigma (`src/sigma.rs`)" -->

The lowest CAS-CI root is found by Davidson iteration, and the only thing
Davidson needs from $H$ is the product $\sigma = HC$ and the diagonal. The
direct sigma builds that product from single-excitation tables over α and
β strings, so it never forms $H$ and never loops over determinant pairs.
Its cost is set by the number of single excitations per string, not by
$N_\text{det}^2$.

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

## 17.3.1 The split of $H$

<!-- include: crates/ferric-ci/theory.md#"Direct sigma (`src/sigma.rs`)" -->

> **Verified equation 17.3** · the three-part split of $H$
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
are those of section 17.2, and the Rust test compares the whole product
element-wise with the `general_element` pair loop.

## 17.3.2 The Davidson solver

`davidson_lowest` is a block-size-1 Davidson for the lowest root:

| Step | Operation |
|---|---|
| 1 | guess: the unit vector on the determinant with the lowest diagonal |
| 2 | Rayleigh–Ritz on the subspace, Ritz pair $(\theta, x)$, residual $r = Hx - \theta x$ |
| 3 | stop when $\lVert r\rVert <$ `conv_thresh` |
| 4 | correction $t_i = r_i/(H_{ii} - \theta)$, denominators kept at least $10^{-8}$ in magnitude |
| 5 | two Gram–Schmidt passes against the subspace; a correction that shrinks below $10^{-10}$ of its norm is already spanned, and the solve stops unconverged |
| 6 | at `max_subspace`, collapse to the Ritz vector and the new correction |

$E_\text{core}$ is folded into the diagonal, so the eigenvalue is the total
CAS-CI energy.

**Dense fallback** (`dense_lowest`). When Davidson stops above
`conv_thresh` and the space has at most `DENSE_SOLVE_MAX_DETS` = 1024
determinants, the dense $H$ is diagonalized and its root kept if its
residual is smaller. Thresholds below Davidson's reach are then met on small
spaces. The result's `converged` flag is $\lVert r\rVert <$ `conv_thresh`
either way, and `residual_norm` is reported.

**Memory.** The Davidson subspace is the peak:
$2\cdot\texttt{max\_subspace}\cdot N_\text{det}\cdot 8$ bytes (the basis and its
products). The CAS-CI memory plan charges it, and the RDM scratch when
`compute_rdms` is set, before anything is allocated.

## 17.3.3 Options

| Option | Values | Default (Python) | Effect |
|---|---|---|---|
| `n_active`, `n_elec_active` | integer, $(n_\alpha, n_\beta)$ | required | the active space and its $M_s$ sector |
| `active_start` | integer | $(N - n_\alpha - n_\beta)/2$ | first active MO; the orbitals below it are the core |
| `conv_thresh` | residual norm | $10^{-9}$ | Davidson and dense-solve threshold |
| `max_subspace` | integer | 50 | subspace size before a collapse |
| `max_iter` | integer | 200 | Davidson iteration cap |
| `memory_budget_gb` | GB | auto | the CAS-CI memory plan |

## 17.3.4 Numerical anchors

| Quantity | Systems | Error | Test |
|---|---|---|---|
| $E_\text{CASCI}$ vs PySCF `mcscf.CASCI` | N2 CAS(6,6) / cc-pVDZ at 1.10 and 2.20 Å; H2O CAS(4,4) / 6-31G; each also with the window moved down one orbital | $\le 1.4\times10^{-11}$ Ha | `crates/ferric-ci/tests/validation_casci.rs` |
| $E_\text{core}$ and the active-space energy | same | $\le 2.6\times10^{-10}$ Ha | `crates/ferric-ci/tests/validation_casci.rs` |
| direct sigma vs the pair loop, random and real integrals | ten sectors, H2O / STO-3G | element-wise to $10^{-12}$ | `crates/ferric-ci/tests/direct_sigma_vs_naive.rs` |

## 17.3.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-ci/theory.md`, "Direct sigma" | the split, the tables, the three contractions, cost and memory |
| Knowles and Handy, CPL 111, 315 (1984); Olsen, Roos, Jørgensen and Jensen, JCP 89, 2185 (1988) | the string-driven direct CI |
| `scripts/ci_rdm_casscf_gradient_sympy.py` | `sigma-split` |

**Known limits (2026-11-15).** Lowest root of one $(n_\alpha, n_\beta)$
sector only: no multiple roots, no state averaging and no $S^2$ handling.
Rows of $\sigma$ are owned by one rayon task each, so the product does not
depend on the thread count.
