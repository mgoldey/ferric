# omega-RI-CDD-MP2: Laplace MP2 in Cholesky pseudo-orbitals {#omega-ri-cdd-mp2}

<!-- status: verify; registry: omega-ri-cdd-mp2 -->

Canonical orbitals spread over the whole molecule, so every occupied-virtual
pair couples to every other one. The Laplace form of section @sec:laplace-mp2
replaces the energy denominator by a sum of exponentials, and at each
quadrature point the occupied and virtual sums become pseudo-densities. A
pivoted Cholesky decomposition of those densities gives localized
pseudo-orbitals with no iterative localization. Glasbrenner, Graf and Ochsenfeld
(JCTC 16, 6856 (2020)) combine these Cholesky-decomposed densities (CDD) with
the omega-RI fit of section @sec:omega-ri-fit and screen the exchange term with
Schwarz-type estimates. This section derives that energy as the code computes it.

The plan:

1. Write the Laplace energy in Fermi-shifted pseudo-densities (@sec:omega-ri-cdd-mp2-pseudo-densities).
2. Factor them by pivoted Cholesky and transform the fitted tensor (@sec:omega-ri-cdd-mp2-factors).
3. Screen the exchange term (@sec:omega-ri-cdd-mp2-screening).

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/omega_ri_cdd.rs::{omega_ri_cdd_mp2,cdd_factors,cdd_transform,pivoted_cholesky_factored,lowdin_coefficients,principal_axis_ao_keys,screened_exchange}` |
| CLI | `laplace-mp2` / `laplace-sos-mp2` with `[ri] metric = "erfc"` or any `[mp2] cdd_*` key |
| Python | `run_omega_ri(method="cdd-mp2" \| "cdd-sos-mp2")` |

| Symbol | Definition |
|---|---|
| $\Delta_{ia,jb}$ | $\varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j$ |
| $t_\ell, w_\ell$ | the minimax Laplace nodes and weights of section @sec:laplace-mp2-quadrature |
| $\varepsilon_F$ | the midgap energy $(\varepsilon_{\mathrm{HOMO}} + \varepsilon_{\mathrm{LUMO}})/2$ |
| $B_{P,ia}$ | the dressed fitted tensor of (@eq:omega-ri-gram) |
| $S$ | the AO overlap matrix |
| $X^{o}, X^{v}$ | Cholesky pseudo-orbital coefficients in the MO basis |

## Pseudo-densities {#omega-ri-cdd-mp2-pseudo-densities}

Start from $1/\Delta = \int_0^\infty e^{-\Delta t}\,dt \approx \sum_\ell w_\ell
e^{-\Delta t_\ell}$. Split the exponential by orbital and insert the Fermi level,
which cancels because $\Delta$ has two occupied and two virtual energies:

$$
e^{-\Delta_{ia,jb} t}
= e^{t(\varepsilon_i - \varepsilon_F)}\, e^{-t(\varepsilon_a - \varepsilon_F)}\,
  e^{t(\varepsilon_j - \varepsilon_F)}\, e^{-t(\varepsilon_b - \varepsilon_F)}.
$$

Every factor is now at most one: the occupied energies lie below $\varepsilon_F$
and the virtual ones above. Collect them into the pseudo-densities

$$
\underline P(t) = C_o\, \operatorname{diag}\!\big(e^{t(\varepsilon_i - \varepsilon_F)}\big) C_o^{\mathsf T},
\qquad
\overline P(t) = C_v\, \operatorname{diag}\!\big(e^{-t(\varepsilon_a - \varepsilon_F)}\big) C_v^{\mathsf T}.
$$

At $t = 0$ they are the occupied and virtual density matrices. As $t$ grows,
the deep core and the high virtuals are damped, so the numerical rank falls.

## Cholesky factors and the energy {#omega-ri-cdd-mp2-factors}

The AO basis is not orthogonal, so the paper (eqs 22-27) factors the
Löwdin-orthogonalized density. Factor
$S^{1/2}\underline P(t) S^{1/2} = \underline L'\,\underline L'^{\mathsf T}$ by
pivoted Cholesky and carry the factor back, $\underline L = S^{-1/2}\underline L'$.
Then $\underline L\,\underline L^{\mathsf T} = \underline P(t)$, and the pivot of
each column is an orthogonalized AO, so each pseudo-orbital sits near the atom
of its pivot. Write $\underline L = C_o X^{o}$; likewise $\overline L = C_v X^{v}$.
Because $C_o^{\mathsf T} S C_o = 1$, $X^{o} = C_o^{\mathsf T} S \underline L$ and
$X^{o} X^{o\mathsf T} = \operatorname{diag}(e^{t(\varepsilon_i - \varepsilon_F)})$.
Transforming the fitted tensor

$$
\breve B_{P,kl} = \sum_{ia} B_{P,ia}\, X^{o}_{ik} X^{v}_{al}
$$

gives, with $G_{kl}[m,n] = \sum_P \breve B_{P,km}\breve B_{P,ln}$ the
pseudo-orbital integrals,

$$
E_C(t) = -2\,\big\lVert \breve B \breve B^{\mathsf T}\big\rVert_F^2
= -2 \sum_{ia,jb} (ia|jb)^2 e^{-\Delta_{ia,jb} t},
\qquad
E_X(t) = \sum_{kl} \operatorname{tr}(G_{kl} G_{kl})
= \sum_{ia,jb} (ia|jb)(ib|ja)\, e^{-\Delta_{ia,jb} t}.
\label{eq:omega-ri-cdd-per-point}
$$

Both right-hand sides follow by substituting $\breve B$ and using
$X^{o}X^{o\mathsf T}$ and $X^{v}X^{v\mathsf T}$ diagonal. Sum over the quadrature:

$$
E_{\mathrm{OS}} = \tfrac12 \sum_\ell w_\ell E_C(t_\ell), \qquad
E_{\mathrm{SS}} = \tfrac12 \sum_\ell w_\ell E_C(t_\ell) + \sum_\ell w_\ell E_X(t_\ell).
$$

Any column order of $\underline L$ is a factor of $\underline P(t)$. The code
sorts the pseudo-orbitals by the position of their pivot along the principal
axis of the molecule, so that orbitals close in space are close in index
(section @sec:omega-ri-cdd-mp2-screening uses this).

Check the trivial limit. With no Cholesky truncation the identity
(@eq:omega-ri-cdd-per-point) is exact at every $t$, so the energy equals the MO
Laplace energy of section @sec:laplace-mp2 at the same nodes; at $\omega = 0$ it
is that energy with the Coulomb fit. SOS-MP2 is $c_{\mathrm{OS}} E_{\mathrm{OS}}$
and skips $E_X$.

> **Verified equation {#eq:omega-ri-cdd-identity}** · the CDD energy per Laplace point
>
> With untruncated factored pivoted Cholesky of the Löwdin density in a
> non-orthogonal AO basis, carried back by $S^{-1/2}$ and reordered by the
> pivot key, $E_C(t)$ and $E_X(t)$ of (@eq:omega-ri-cdd-per-point) equal the
> explicit four-index sums at every $t$, and an exact quadrature of $1/\Delta$
> reproduces $E_{\mathrm{OS}}$ and $E_{\mathrm{SS}}$. The Löwdin pivots equal
> those of a textbook pivoted Cholesky of the explicit $S^{1/2}\underline P S^{1/2}$.
>
> | | |
> |---|---|
> | Proof | `scripts/omega_ri_sympy.py`, claim 6 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/omega_ri_cdd.rs::cdd_factors` |
>
> <!-- verified: id=laplace.omega-ri-cdd.identity; script=scripts/omega_ri_sympy.py; claim=6; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/omega_ri_cdd.rs::cdd_factors -->

## The code {#omega-ri-cdd-mp2-code}

`ferric-mp2/src/omega_ri_cdd.rs::cdd_factors` forms the Fermi-shifted weights
(the matching virtual line has $-t$):

```rust
let so: Vec<f64> = eps_occ
    .iter()
    .map(|&e| (0.5 * t * (e - eps_f)).exp())   // e^{t(eps_i - eps_F)/2}
    .collect();
```

- The half exponent is the square root of the pseudo-density weight: the
  factor $G = C\,\operatorname{diag}(e^{t(\varepsilon - \varepsilon_F)/2})$ satisfies
  $GG^{\mathsf T} = \underline P(t)$, and
  `ferric-mp2/src/omega_ri_cdd.rs::pivoted_cholesky_factored` runs the pivoted
  Cholesky without forming the $n_{\mathrm{AO}}^2$ matrix. It returns $M$ with
  factor $GM$.
- The Löwdin step changes only the matrix the pivots are chosen on. The
  caller passes $S^{1/2}C$ (`lowdin_coefficients`), so the pivoted Cholesky of
  $S^{1/2}G$ gives $L' = S^{1/2} G M$ and $L = S^{-1/2}L' = GM$: the same
  coordinates, $X = \operatorname{diag}(e^{t(\varepsilon-\varepsilon_F)/2})\,M$.
- `principal_axis_ao_keys` projects each AO centre on the largest-variance
  axis of the atomic coordinates; `cdd_factors` sorts the columns of $M$ by
  the key of their pivot (stable).
- The pivots stop when the largest remaining diagonal falls below
  `cdd_tol`, or below $m\,\epsilon_{\mathrm{mach}}$ times the first pivot
  when `cdd_tol = 0`; the default keeps every pseudo-orbital above rounding.
- $B$ is built once (section @sec:omega-ri-fit) and transformed per point by
  `ferric-mp2/src/omega_ri_cdd.rs::cdd_transform`.

## Screening {#omega-ri-cdd-mp2-screening}

With $Q_{kl} = \lVert \breve B_{\cdot,kl}\rVert$ the Schwarz factor of a
pseudo-orbital pair, `ferric-mp2/src/omega_ri_cdd.rs::screened_exchange` drops
an occupied pair $(k,l)$ when $(\sum_m Q_{km}Q_{lm})^2 < \vartheta_{ij}$ and,
inside a kept pair, a virtual $m$ when
$Q_{km}Q_{lm}\max_n(Q_{kn}Q_{ln}) < \vartheta_{\mathrm{schwarz}}$. The paper uses
$\vartheta_{ij} = 10^{-6}$ and $\vartheta_{\mathrm{schwarz}} = 10^{-9}$
(`[mp2] cdd_pair_thresh`, `cdd_schwarz_thresh`). Both default to zero, which
keeps every term. Only the exchange term is screened; $E_C$ is one dense
contraction.

For a kept pair the code forms $G_{kl}$ on the index range from the first to
the last kept virtual, as one GEMM on a strided view of $\breve B$ (no copy),
and sums $G_{kl}[m,n]G_{kl}[n,m]$ over the kept $m, n$ only. The sum is over
exactly the kept set whatever the order. The order decides how much of the
range is wasted: with the pseudo-orbitals sorted along the molecule, the
virtuals kept for a pair are those near it, and the range is close to the
kept set.

Measured on n-alkanes (cc-pVDZ / cc-pVDZ-RI, frozen core, $\omega = 0.1$
Bohr⁻¹, 7 points, the paper's thresholds, RI-JK SCF, 4 threads, 2 runs that
agree to 2%, 2026-11-21): the GEMM work of exactly the kept entries,
$\sum k_{kl}^2/(n_{\mathrm{pairs}} r_v^2)$, is 0.206 at C16 and 0.117 at
C20; the work on the ranges is 0.374 and 0.228, about twice the kept set. The
screened exchange takes 0.41 (C16) and 0.26 (C20) of the unscreened exchange
time, which follows the range work. Tiling the range to skip its unkept
blocks would close the rest of that factor of two; the code does not do it.

## Design choices {#omega-ri-cdd-mp2-design}

The energy is built from the MO-basis tensor $B$ transformed per point. This
is the paper's AO-basis contraction (eqs 22-27) rearranged: the same
pseudo-orbitals, the same pair screens. Every matrix stays dense. The
Cholesky truncation reduces the pseudo-orbital count, and the screens skip
exchange work, but no step uses the locality of the erfc integrals, so the
cost has the scaling of the MO Laplace path.

The exchange is most of the cost. Measured with the settings above
(seconds, one of two runs; the other agrees to 2%):

| | C16 | C20 |
|---|---|---|
| unscreened CDD, total / exchange | 22.3 / 16.4 | 59.1 / 46.8 |
| paper screens, total / exchange | 12.5 / 6.6 | 24.2 / 12.1 |
| non-exchange stages ($B_{ov}$, transform, Coulomb, dressing, Cholesky) | 5.8 | 12.2 |
| canonical RI-MP2 (Coulomb fit), total | 4.8 | 11.3 |

The screened CDD energy is 2.6 times the RI-MP2 time at C16 and 2.1 times at
C20. There is no crossover at these sizes. The non-exchange stages alone
already cost as much as RI-MP2 and grow like the MO Laplace path, so a
crossover needs the sparse AO and pseudo-orbital pipeline of the paper, which
the code does not have.

The Löwdin pseudo-orbitals are more local than Cholesky factors of the
AO density, and the same thresholds drop more. At C20 they keep 0.625 of the
pairs and 0.242 of the virtual entries (AO pivots: 0.731 and 0.355). The
screening error against the unscreened energy is $-3.20\times10^{-5}$ Eh
($-3.14\times10^{-5}$ with AO pivots), and $-2.28\times10^{-5}$ at C16
($-2.20\times10^{-5}$).

## What is proven and what is not {#omega-ri-cdd-mp2-proven}

Claim 6 of `scripts/omega_ri_sympy.py` proves (@eq:omega-ri-cdd-per-point) for
untruncated factors. The Rust anchors
(`crates/ferric-mp2/tests/omega_ri_anchors.rs`) show that untruncated CDD
equals `ferric-mp2/src/laplace.rs::LaplaceMp2` at $\omega = 0$ to
$10^{-10}$ Eh, and the canonical-pseudo-orbital evaluation at
$\omega = 0.1$ Bohr⁻¹ to $10^{-12}$ Eh (measured $6\times10^{-17}$ on water).
That anchor cannot tell the Löwdin pivots from AO pivots, because any
full-rank factor is a rotation. The unit test
`lowdin_pivots_match_the_explicit_orthogonalized_density` checks the pivots
against a textbook pivoted Cholesky of $S^{1/2}\underline P S^{1/2}$, and
`range_exchange_equals_the_kept_set_oracle_under_any_order` checks the range
exchange against a brute-force sum over the kept set and under a permutation
of the orbitals. The errors of a positive `cdd_tol` and of the screens are
measured, not bounded.
