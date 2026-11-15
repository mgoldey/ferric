# 12.3 Laplace RI-MP2 and Laplace SOS-MP2

<!-- section: 12.3; status: ready; registry: laplace-mp2, laplace-sos-mp2;
     theory: crates/ferric-mp2/theory.md#"8. Laplace-transform MP2"; crates/ferric-quadrature/theory.md#"1. Minimax Laplace Quadrature" -->

*Notation follows `book/notation.md`. $t$ is the Laplace variable (section 12
introduction).*

The denominator $\Delta_{ij}^{ab}$ couples the two orbital pairs of an MP2
term. Writing $1/\Delta$ as a sum of exponentials splits every term into a
product of an $(i,a)$ factor and a $(j,b)$ factor. The energy then becomes a
short sum over quadrature points. Each point has the form of an RI-MP2
energy with weighted factors.

The code uses this rewrite two ways:

- **Exactly**, in the MO and AO bases (this section). It is the graded
  method.
- **Truncated**, with localized orbitals (section 12.4).

Both kinds are closed shell (RHF) and energy only.

| Index | Range | Code name |
|---|---|---|
| $i, j$; $a, b$ | active occupied; virtual | `0..nocc` offset `frozen_core`; `0..nvir` offset `nocc_total` |
| $\mu, \nu$; $P, Q$ | AO; auxiliary | `0..nbas`; `0..naux` |
| $k$ | quadrature point | `points`, `weights` of `LaplaceMp2` |

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/laplace.rs::laplace_ri_mp2` (full MP2); `ferric-mp2/src/laplace.rs::laplace_sos_mp2`, `laplace_sos_mp2_with_basis` (SOS) |
| CLI | `method.kind = "laplace-mp2"` (`examples/water-laplace-rimp2.toml`); `method.kind = "laplace-sos-mp2"` (`examples/water-laplace-sos-mp2.toml`); keys `n_quad`, `frozen_core`, `c_os`, `sos_formulation`, `domain_cutoff_bohr` in `[mp2]`, `[memory] budget_gb` |
| Python | `run_laplace_mp2`, `run_laplace_sos_mp2` (`crates/ferric-python/ferric.pyi`) |

## 12.3.1 The quadrature

For $x > 0$,

$$ \frac1x = \int_0^\infty e^{-tx}\,dt \;\approx\; \sum_{k=1}^{n_q} w_k\,e^{-t_kx}, \qquad x \in [y_{\min}, y_{\max}] . \tag{12.8} $$

$t_k$ and $w_k$ come from the minimax tables of Takatsuka, Ten-no and
Hackbusch for $1/y$ on $[1, R]$ (`ferric-quadrature/src/minimax.rs::select_minimax_points`).
Those tables give exponents $\tau_k$ and weights $\omega_k$.
`LaplaceQuadrature::new` rescales them to the actual range:

$$ R = \frac{y_{\max}}{y_{\min}}, \qquad t_k = \frac{\tau_k}{y_{\min}}, \qquad w_k = \frac{\omega_k}{y_{\min}} . \tag{12.9} $$

The range is set in every driver as

$$ y_{\min} = 2(\varepsilon_{\mathrm{LUMO}} - \varepsilon_{\mathrm{HOMO}}), \qquad y_{\max} = 2(\varepsilon_{n_{\mathrm{mo}}} - \varepsilon_1) . $$

$\varepsilon_1$ is the lowest orbital energy, a frozen-core orbital when
`frozen_core > 0`. The range is therefore wider than the active
$\Delta_{ij}^{ab}$ need, which is safe.

The rules for the table:

- `n_quad` must be 3, 5 or 7. Any other value is an error, not a rounding.
- The table entry used is the first with
  $R_{\mathrm{tab}} \ge 0.99\,R$. An entry up to 1% short of the true range
  is accepted, to absorb round-off.
- $R$ beyond the largest entry is an error. The largest entry is 100 for
  `n_quad` = 3 and 1000 for 5 and 7.
- The CLI default is `n_quad` = 7.

> **Verified equation 12.8** · Laplace identity
>
> $\int_0^\infty e^{-tx}dt = 1/x$ for $x > 0$. The energy assembled from
> (12.10), integrated over $t$ by an adaptive quadrature, equals the
> brute-force spin-orbital MP2 energy (to about $2\times10^{-15}$ on the
> script's model).
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_family_sympy.py`, part F |
> | Negative control | the half-exponent of $B(t)$ changed from $\tfrac12$ to 1 must fail ("FAIL F") |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/laplace.rs::weighted_b_mo` |
>
> <!-- verified: id=laplace.identity; script=scripts/mp2_family_sympy.py; claim=F; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/laplace.rs::weighted_b_mo -->

The proof covers the exact integral. The minimax weights are tabulated data
and nothing proves them. Their error is measured instead (section 12.3.6).

## 12.3.2 The factorized energy

Since $x = (\varepsilon_a - \varepsilon_i) + (\varepsilon_b - \varepsilon_j)$,
the exponential splits. With the dressed factors $B^P_{ia}$ of (10.4):

$$ B^P_{ia}(t) = B^P_{ia}\,e^{-t(\varepsilon_a - \varepsilon_i)/2}, \qquad J_{PQ}(t) = \sum_{ia} B^P_{ia}(t)B^Q_{ia}(t), \qquad G_{PQ}[i,j](t) = \sum_a B^P_{ia}(t)\,B^Q_{ja}(t), $$

$$ E^{(2)} \approx -\sum_k w_k\Big[\,2\sum_{PQ}J_{PQ}(t_k)^2 - \sum_{PQ}\sum_{ij} G_{PQ}[i,j](t_k)\,G_{PQ}[j,i](t_k)\Big], \qquad E_{\mathrm{OS}} \approx -\sum_k w_k\sum_{PQ}J_{PQ}(t_k)^2 . \tag{12.10} $$

The first sum is the Coulomb-like term $\sum (ia|jb)^2 e^{-tx}$. The second
is the exchange term $\sum (ia|jb)(ib|ja)\,e^{-tx}$. In the split of (10.2),
$E_{\mathrm{OS}}$ is the $J^2$ term and $E_{\mathrm{SS}}$ is $J^2$ minus the
$G\cdot G$ term.

> **Verified equation 12.10** · factorization at fixed $t$
>
> At any fixed $t$ and for any coefficients,
> $\sum_{PQ}J_{PQ}(t)^2 = \sum_{ijab}(ia|jb)^2e^{-t\Delta_{ij}^{ab}}$ and
> $\sum_{PQ}\sum_{ij}G_{PQ}[i,j]G_{PQ}[j,i] = \sum_{ijab}(ia|jb)(ib|ja)e^{-t\Delta_{ij}^{ab}}$.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_variant_identities.py`, claim 3 |
> | Negative control | the half-exponent $-\tfrac12t \to -t$ must fail |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/laplace.rs::sos_j_square_trace`, `ferric-mp2/src/laplace.rs::laplace_exchange_energy` |
>
> <!-- verified: id=laplace.factorization; script=scripts/mp2_variant_identities.py; claim=3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/laplace.rs::laplace_exchange_energy -->

## 12.3.3 The AO form of the Coulomb term

Define the occupied and virtual pseudo-densities

$$ \underline P(t) = \sum_i C_{\cdot i}\,e^{+t\varepsilon_i}\,C_{\cdot i}^{\mathsf T}, \qquad \underline Q(t) = \sum_a C_{\cdot a}\,e^{-t\varepsilon_a}\,C_{\cdot a}^{\mathsf T}, \qquad J_{PQ}(t) = \operatorname{Tr}\big[B^P\,\underline P(t)\,B^Q\,\underline Q(t)\big], \tag{12.11} $$

with $B^P_{\mu\nu}$ the dressed AO factor. They are underlined here to keep
them apart from the auxiliary index $P$ and the MP2 density.
`build_pseudo_density_occ` and `build_pseudo_density_vir` form them, and
`laplace_ao_coulomb_energy` forms the trace in panels of $\mu$ rows.

The exchange term has no AO trace of this shape. The script's negative
control shows that the natural candidate
$\sum_{PQ}\operatorname{Tr}(B^P\underline P B^Q\underline Q B^Q\underline P B^P\underline Q)$
is not it. The AO path therefore still forms the exchange term from the MO
factors $B(t)$.

> **Verified equation 12.11** · AO trace of the Coulomb term
>
> $\operatorname{Tr}[B^P\underline P(t)B^Q\underline Q(t)] = J_{PQ}(t)$
> elementwise.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_variant_identities.py`, claim 3 |
> | Negative control | the four-factor AO trace used as the exchange term must differ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/laplace.rs::laplace_ao_coulomb_energy`, `build_pseudo_density_occ`, `build_pseudo_density_vir` |
>
> <!-- verified: id=laplace.ao-trace; script=scripts/mp2_variant_identities.py; claim=3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/laplace.rs::laplace_ao_coulomb_energy -->

## 12.3.4 SOS-MP2

Laplace SOS-MP2 keeps only the opposite-spin part:

$$ E_{\mathrm{SOS}} = c_{\mathrm{OS}}\,E_{\mathrm{OS}}, \qquad c_{\mathrm{OS}} = 1.3 \text{ by default (Jung, Lochan, Dutoi, Head-Gordon)}. \tag{12.12} $$

There is no $c_{\mathrm{SS}}$, by design. Dropping the same-spin term is what
lets the denominator factorize.

- In the opposite-spin numerator $(ia|jb)^2$, the $(i,a)$ and $(j,b)$
  pairings match the split of $e^{-tx}$.
- In the same-spin exchange product $(ia|jb)(ib|ja)$ the pairings cross, and
  no change to the denominator undoes that.
- The CLI warns if `[mp2] c_ss` is set.
- A non-finite $c_{\mathrm{OS}}$ is an error.

## 12.3.5 The algorithm

| Step | Quantity | Function |
|---|---|---|
| 1 | $y_{\min}$, $y_{\max}$, the grid (12.9) | `LaplaceMp2::init_quadrature`, `LaplaceQuadrature::new` |
| 2 | $V$, $M = L^{-1}$ | `coulomb_metric_2c`, `cholesky_inverse_sqrt` |
| 3 | MO path: $B^P_{ia}$, blocked transform then dressing | `eri3_mo_ov_blocked`, `dress_with_metric` |
| 3′ | AO path: dressed $B^P_{\mu\nu}$, stored sparse (threshold $10^{-12}$) | `compute_ao`, `compute_sos_ao` |
| 4 | per point, in parallel over $k$: $B(t_k)$, then $\sum J^2$ and (full MP2) $\sum G\cdot G$ | `weighted_b_mo`, `sos_j_square_trace`, `laplace_exchange_energy` |
| 4′ | AO path, per point: $\underline P(t_k)$, $\underline Q(t_k)$, the trace (12.11) | `laplace_ao_coulomb_energy` |
| 5 | $-\sum_k w_k(\ldots)$ | the `par_iter` sum |

Which path each entry point takes:

| Entry | Coulomb term | Exchange term |
|---|---|---|
| `laplace-mp2` (`laplace_ri_mp2` → `compute_ao(.., None)`) | AO trace (12.11) | MO $G\cdot G$ |
| `laplace-sos-mp2`, `mo` (default) | MO $J = B(t)B(t)^{\mathsf T}$ (`compute_sos_mo`) | none |
| `laplace-sos-mp2`, `ao` | AO trace (`compute_sos_ao(.., None)`) | none |
| `LaplaceMp2::compute_mo` | MO | MO; library and tests only, the MO side of the MO ≡ AO checks |

## 12.3.6 The kernel

The per-point MO energy in `ferric-mp2/src/laplace.rs::LaplaceMp2::compute_mo`:

Abridged from `ferric-mp2/src/laplace.rs::LaplaceMp2::compute_mo` (not a verbatim excerpt):
```rust
let b_t = weighted_b_mo(&b_flat, &occ_scale, &vir_scale, nocc, nvir, t);
let e_coul = sos_j_square_trace(&b_t);
let e_exch = laplace_exchange_energy(&b_t, naux, nocc, nvir, k_budget);
-w * (2.0 * e_coul - e_exch)
```

| Code | Equation |
|---|---|
| `t`, `w` | $t_k$, $w_k$ of (12.9) |
| `b_t` | $B^P_{ia}(t_k)$, scaled by $e^{t\varepsilon_i/2}e^{-t\varepsilon_a/2}$ |
| `e_coul` | $\sum_{PQ}J_{PQ}(t_k)^2$ |
| `e_exch` | $\sum_{PQ}\sum_{ij}G_{PQ}[i,j]G_{PQ}[j,i]$ |

`laplace_exchange_energy` reads `b_t` as an $(N_{\mathrm{aux}}n_o)\times n_v$
matrix $X$. It forms $Y = X_{\mathrm{blk}}X^{\mathsf T}$ in blocks of whole
auxiliary indices $P$, so that each $n_o\times n_o$ block $G_{PQ}$ lies
inside one $Y$ block.

## 12.3.7 Reproducibility and memory

- **Fixed blocking.** The $J$ lower triangle is built in fixed blocks of
  `SOS_J_COL_BLOCK` = 128 columns, and the block sums are folded in order.
  The per-task memory share is the budget divided by the compile-time
  constant `NOMINAL_TASK_WORKERS` = 8, not by the live pool size. Block
  widths, and so every bit of the energy, do not depend on
  `RAYON_NUM_THREADS`.
  `crates/ferric-mp2/tests/laplace_energy_is_thread_independent.rs` checks
  this.
- **Wide pools.** A pool wider than 8 can exceed the budget by
  $n_{\mathrm{workers}}/8$. The code warns when that happens
  (`warn_if_task_workers_exceed_nominal`). The per-task share has a
  64 MiB floor (`MIN_PER_TASK`), which depends on nothing ambient.
- **MO path.** It holds $B_{ov}$ ($N_{\mathrm{aux}}n_on_v$) and one weighted
  copy per point in flight.
- **AO path.** It holds two $N_{\mathrm{aux}}N^2$ tensors at once, plus the
  sparsified copy. It fails fast when `laplace_dressing_peak_bytes` exceeds
  the budget.
- **Choosing a path.** The AO peak grows as $N_{\mathrm{aux}}N^2$ against
  $N_{\mathrm{aux}}n_on_v$ for `mo`. The two give the same energy
  (section 12.3.8), so `mo` is the path to use when memory is short.

## 12.3.8 Anchors

| Check | Reference | Agreement | Test |
|---|---|---|---|
| `laplace-mp2`, five systems, $R$ = 19.0 to 45.7, `n_quad` = 7 | exact DF-MP2 (PySCF `DFMP2`) | the minimax quadrature error, at most $2.2\times10^{-7}$ Ha (H2O/aug-cc-pVDZ), bar $2\times10^{-6}$ | `laplace_mp2_*_vs_pyscf_dfmp2` in `crates/ferric-mp2/tests/validation_laplace_mp2.rs` |
| same five systems | quadrature-isolated reference: a dense numpy $(i,a,j,b)$ sum on PySCF fitted integrals with $1/\Delta$ replaced by the same minimax nodes, rescaled as in (12.9) | at most $3.1\times10^{-11}$ Ha | same tests |
| MO against AO, same grid | — | at most $3.4\times10^{-13}$ Ha | same tests |
| `laplace-sos-mp2` $E_{\mathrm{OS}}$, five closed-shell systems, all electrons | PySCF `DFMP2` `e_corr_os`, same auxiliary basis | the quadrature error, at most $1.4\times10^{-7}$ Ha at `n_quad` = 7 | `sos_mp2_*_vs_pyscf_dfmp2` in `crates/ferric-mp2/tests/validation_sos_mp2.rs` |
| SOS, MO against AO | — | $1.6\times10^{-14}$ measured, bar $10^{-10}$ | same tests |

<!-- figures: site/src/reference/validation.md rows "laplace-mp2", "laplace-sos-mp2"; crates/ferric-mp2/tests/validation_laplace_mp2.rs module doc table (MEASURED 2026-10-03); crates/ferric-mp2/tests/validation_sos_mp2.rs module doc -->

`validation_laplace_mp2.rs` asserts the second row so that a disagreement
with DF-MP2 can be attributed. Its "vs exact DF-MP2" column reproduces each
system's independently computed quadrature error to the last printed digit.

The test's negative controls are recorded mutations:

| Mutation | Error produced | Bar |
|---|---|---|
| drop one weight | $1.30\times10^{-2}$ | $10^{-9}$ |
| take $y_{\max}$ from $\varepsilon_{\mathrm{HOMO}}$ | $1.31\times10^{-3}$ | $10^{-9}$ |
| frozen core off by one | $2.46\times10^{-3}$ | $10^{-9}$ |

Each produces an error far above its bar against the quadrature-isolated
reference.

Both kinds are graded Proven. For SOS the grade covers the `mo` and `ao`
formulations only. Frozen core is not covered by the PySCF reference.

**Known limits (2026-11-15).**

- Only `n_quad` = 3, 5, 7 are tabulated. There is no error-controlled choice
  of the number of points.
- $y_{\max}$ includes frozen-core energies, which widens $R$.
- The AO path is dense; no reduced scaling has been measured for it.
- `build_pseudo_density_occ` and `build_pseudo_density_vir` are scalar
  triple loops, not GEMMs.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/theory.md`, section 8 | equations 12.8 to 12.12 |
| `crates/ferric-quadrature/theory.md`, section 1; `crates/ferric-quadrature/src/minimax.rs` | the grid and its rescaling (12.9) |
| `scripts/mp2_family_sympy.py`, part F; `scripts/mp2_variant_identities.py`, claim 3 | verified equations 12.8, 12.10, 12.11 |
| M. Häser and J. Almlöf, J. Chem. Phys. 96, 489 (1992) (as listed in `site/src/reference/references.md`) | the Laplace-transformed denominator |
| A. Takatsuka, S. Ten-no and W. Hackbusch, J. Chem. Phys. 129, 044112 (2008) | the minimax tables |
| Y. Jung, R. C. Lochan, A. D. Dutoi and M. Head-Gordon, J. Chem. Phys. 121, 9793 (2004) | SOS-MP2 and $c_{\mathrm{OS}} = 1.3$ |
| `site/src/reference/validation.md`; the two validation test files | the anchor figures |

Changes from the sources:

- The pseudo-densities are underlined.
- The quadrature range is stated with the frozen-core caveat and the 1%
  table slack, as in `theory.md` section 8.
- `theory.md` section 8's formulation table lists `local`; its full
  treatment is section 12.4.
