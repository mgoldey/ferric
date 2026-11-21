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
| Rust | `ferric-mp2/src/omega_ri_cdd.rs::{omega_ri_cdd_mp2,cdd_factors,cdd_transform,pivoted_cholesky_factored,screened_exchange}` |
| CLI | `laplace-mp2` / `laplace-sos-mp2` with `[ri] metric = "erfc"` or any `[mp2] cdd_*` key |
| Python | `run_omega_ri(method="cdd-mp2" \| "cdd-sos-mp2")` |

| Symbol | Definition |
|---|---|
| $\Delta_{ia,jb}$ | $\varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j$ |
| $t_\ell, w_\ell$ | the minimax Laplace nodes and weights of section @sec:laplace-mp2-quadrature |
| $\varepsilon_F$ | the midgap energy $(\varepsilon_{\mathrm{HOMO}} + \varepsilon_{\mathrm{LUMO}})/2$ |
| $B_{P,ia}$ | the dressed fitted tensor of (@eq:omega-ri-gram) |
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

Factor $\underline P(t) = \underline L\,\underline L^{\mathsf T}$ by pivoted
Cholesky and write $\underline L = C_o X^{o}$; likewise $\overline L = C_v X^{v}$.
Then $X^{o} X^{o\mathsf T} = \operatorname{diag}(e^{t(\varepsilon_i - \varepsilon_F)})$,
and transforming the fitted tensor

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

Check the trivial limit. With no Cholesky truncation the identity
(@eq:omega-ri-cdd-per-point) is exact at every $t$, so the energy equals the MO
Laplace energy of section @sec:laplace-mp2 at the same nodes; at $\omega = 0$ it
is that energy with the Coulomb fit. SOS-MP2 is $c_{\mathrm{OS}} E_{\mathrm{OS}}$
and skips $E_X$.

> **Verified equation {#eq:omega-ri-cdd-identity}** · the CDD energy per Laplace point
>
> With untruncated factored pivoted Cholesky, $E_C(t)$ and $E_X(t)$ of
> (@eq:omega-ri-cdd-per-point) equal the explicit four-index sums at every $t$,
> and an exact quadrature of $1/\Delta$ reproduces $E_{\mathrm{OS}}$ and $E_{\mathrm{SS}}$.
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
  Cholesky on $G$ without forming the $n_{\mathrm{AO}}^2$ matrix.
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

## Design choices {#omega-ri-cdd-mp2-design}

The energy is built from the MO-basis tensor $B$ transformed per point. This
is the paper's AO-basis contraction (eqs 22-27) rearranged: the same
pseudo-orbitals, the same pair screens. Every matrix stays dense. The
Cholesky truncation reduces the pseudo-orbital count, and the screens skip
exchange pairs, but no step uses the locality of the erfc integrals, so the
cost has the scaling of the MO Laplace path.

## What is proven and what is not {#omega-ri-cdd-mp2-proven}

Claim 6 of `scripts/omega_ri_sympy.py` proves (@eq:omega-ri-cdd-per-point) for
untruncated factors. The Rust anchors
(`crates/ferric-mp2/tests/omega_ri_anchors.rs`) show that untruncated CDD
equals `ferric-mp2/src/laplace.rs::LaplaceMp2` at $\omega = 0$ and the
canonical-pseudo-orbital evaluation at $\omega = 0.1$ Bohr⁻¹ to $10^{-10}$ Eh.
The errors of a positive `cdd_tol` and of the screens are measured, not bounded.
