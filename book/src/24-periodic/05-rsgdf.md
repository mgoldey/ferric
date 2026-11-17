# Range-separated Gaussian density fitting for J and K {#rsgdf}

<!-- status: ready; registry: pbc-hf -->

The dense builder of section @sec:exchange-divergence stores $n_\text{ao}^4$ Coulomb integrals and
sums a $\mathbf G$ sphere that grows with the cell volume and the steepest
exponent, so it is a correctness oracle for toy cells. Production cells use
`jk = "rsgdf"`: the density fitting of chapter @sec:density-fitting, with every pair density
expanded in an auxiliary Gaussian basis, and the Ewald split of section @sec:ewald
applied to the two- and three-index integrals. The fit keeps the convention
that drops $\mathbf G = 0$, so it fits the same kernel the energy uses.

Section @sec:rsgdf-fitted-integrals derives the integrals, the fit and the two builders;
@sec:rsgdf-algorithm gives the algorithm and the code; @sec:rsgdf-reproducibility-memory to @sec:rsgdf-sources cover
reproducibility, options, anchors and sources.

| Symbol | Meaning | Code name |
|---|---|---|
| $\mu\nu$, $\lambda\sigma$ | orbital AO pairs, flattened `m*nao + n` | |
| $P, Q$ | auxiliary functions (pure shells only) | `naux` |
| $X_P(\mathbf G)$ | Fourier transform of $\chi_P$ | `aux_ft` |
| $J2$, $J3$ | metric and three-index integrals in the $\mathbf G = 0$-dropped kernel | `j2`, `j3` |
| $q_P$ | charge $\int\chi_P$ | `q` |
| $\eta$ | the fit's Ewald split | `RsGdfConfig::omega`, `gdf_omega` |
| $B_{k,\mu\nu}$ | fitted factor over the kept metric eigenvectors | `RsGdf::b` |

| Entry point | Location |
|---|---|
| Rust, Gamma | `ferric-pbc/src/rsgdf.rs::{subtract_g0,fit_with_metric,j_builder,k_builder}` |
| Rust, $k$-points | `crates/ferric-pbc/src/rsgdf/kpoint.rs` (`KRsGdf`) |
| Split chooser | `crates/ferric-pbc/src/rsgdf/auto_omega.rs`; Python `auto_gdf_omega` |
| CLI | `[cell] jk = "rsgdf"` with `auxbasis` |
| Python | `jk="rsgdf", auxbasis=...` on every periodic entry point |

## The fitted integrals {#rsgdf-fitted-integrals}

**What is fitted.** The goal is
$I_{\mu\nu,\lambda\sigma} \approx \sum_kB_{k,\mu\nu}B_{k,\lambda\sigma}$, in
the PySCF RSGDF conventions (`crates/ferric-pbc/src/rsgdf.rs`). Every fitted
quantity uses the energy's kernel, $v(\mathbf G) = 4\pi/G^2$ with
$\mathbf G = 0$ dropped. With $X_P$ the transform of $\chi_P$ and $P_{\mu\nu}$
the pair transform of section @sec:pair-ft, the metric and the three-index integrals
are

$$
J2_{PQ} = \frac{1}{\Omega}\sum_{\mathbf G\neq0}v(\mathbf G)\,X_P^*(\mathbf G)X_Q(\mathbf G),\qquad
J3_{\mu\nu,P} = \frac{1}{\Omega}\sum_{\mathbf G\neq0}v(\mathbf G)\,P_{\mu\nu}^*(\mathbf G)X_P(\mathbf G).
$$

**Evaluation by an Ewald split.** Neither sum is evaluated as written: a
compact auxiliary function or a tight orbital pair would need a huge
$\mathbf G$ sphere. Split $1/r = \operatorname{erfc}(\eta r)/r + \operatorname{erf}(\eta r)/r$
(the code's `omega`, PySCF's `omega` of the fit), evaluate the erfc part in
real space and the erf part in reciprocal space:

$$
\begin{aligned}
J2_{PQ} &= \sum_{\mathbf T}(P_0|Q_{\mathbf T})_{\operatorname{erfc}}
 + \frac{1}{\Omega}\sum_{\mathbf G\neq0}\frac{4\pi}{G^2}e^{-G^2/4\eta^2}X_P^*X_Q
 - c_0\,q_Pq_Q, \\
J3_{\mu\nu,P} &= \sum_{\mathbf L,\mathbf T}(\mu_0\nu_{\mathbf L}|P_{\mathbf T})_{\operatorname{erfc}}
 + \frac{1}{\Omega}\sum_{\mathbf G\neq0}\frac{4\pi}{G^2}e^{-G^2/4\eta^2}P_{\mu\nu}^*X_P
 - c_0\,S_{\mu\nu}\,q_P,
\end{aligned}
$$

with $c_0 = \pi/(\eta^2\Omega)$, $q_P = \int\chi_P = X_P(0)$ and
$S_{\mu\nu} = P_{\mu\nu}(0)$ the lattice overlap. The last terms come from
the same argument as $V_\text{G0}$ in section @sec:nuclear-attraction. The real-space erfc sums
(libint2 erfc integrals on translated shells,
`Engine::compute_eri2_shifted` and `compute_eri3_shifted`) are full lattice
sums, so they contain the $\mathbf G = 0$ term of the erfc kernel,
$\int\operatorname{erfc}(\eta r)/r\,d^3r = \pi/\eta^2$, times the two charges.
For $J2$ the charges are $q_P$ and $q_Q$. For $J3$ they are $q_P$ and the
charge of the lattice-summed pair, $S_{\mu\nu}$. `subtract_g0` removes both,
and it is the one place the convention lives. Per $\mathbf G \neq 0$ the erfc
and erf pieces add to the bare kernel (verified equation @eq:pbc-hcore-split), so $J2$ and
$J3$ do not depend on $\eta$ up to truncation.

**The erfc integral of two Gaussian charges.** The screen of the real-space
walk needs the size of one erfc term. For two unit Gaussian charges of
exponents $a$ and $b$ at distance $d$, the Coulomb interaction is
$\operatorname{erf}(\gamma d)/d$ with $1/\gamma^2 = 1/a + 1/b$: the widths add.
The erf operator is Coulomb with one more Gaussian smearing, so it adds
$1/\eta^2$. The erfc interaction is the difference:

$$
\frac{\operatorname{erf}(\gamma d) - \operatorname{erf}(\gamma_\eta d)}{d},\qquad
\frac{1}{\gamma^2} = \frac1a + \frac1b,\quad
\frac{1}{\gamma_\eta^2} = \frac1a + \frac1b + \frac{1}{\eta^2} .
$$

Check two limits. At $d = 0$ it is $2(\gamma - \gamma_\eta)/\sqrt\pi$, finite.
At $\eta \to \infty$, $\gamma_\eta \to \gamma$ and it vanishes: all of the
interaction has moved to reciprocal space. The per-triplet screen of
`rsgdf.rs` is built on a bound of this Gaussian form.

> **Verified equation {#eq:pbc-rsgdf-split}** · the Ewald-split fit integrals
>
> On s Gaussians in a triclinic cell, $J2$ and $J3$ assembled as erfc
> real-space sums plus the long-range reciprocal sums minus $c_0q_Pq_Q$ and
> $c_0S_{\mu\nu}q_P$ equal the pure $\mathbf G \neq 0$ reciprocal sums at
> $\eta = 1.0$ and $1.6$, relative to $10^{-10}$. The erfc pair interaction,
> its $d \to 0$ limit and $\int\operatorname{erfc}(\eta r)/r = \pi/\eta^2$ are
> checked symbolically and by quadrature.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_rsgdf_sympy.py`, claims `erfc-pair`, `split-j2`, `split-j3` |
> | Negative control | the runner's mutation: the three-index subtraction $c_0S_{\mu\nu}q_P$ dropped (the code's `G0Handling::MetricOnlyMutant`) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/rsgdf.rs::subtract_g0` |
> | Code test | `fitted_eri_is_omega_independent` in `crates/ferric-pbc/tests/pbc_rsgdf.rs` |
>
> <!-- verified: id=pbc.rsgdf.split; script=scripts/pbc_rsgdf_sympy.py; claim=split-j3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/rsgdf.rs::subtract_g0; rust_test=ferric-pbc/tests/pbc_rsgdf.rs::fitted_eri_is_omega_independent; registry=pbc-hf -->

**The fit.** Diagonalize the metric, $J2 = UsU^{\mathsf T}$. Drop the
eigenvalues $s_k \le$ `lindep` (default $10^{-10}$, PySCF
`LINEAR_DEP_THR`) and count them; never invert them
(`fit_with_metric`):

$$
W = U_\text{kept}\,s_\text{kept}^{-1/2},\qquad B = (J3\,W)^{\mathsf T} .
$$

Multiply out: $B^{\mathsf T}B = J3\,WW^{\mathsf T}J3^{\mathsf T}$, and
$WW^{\mathsf T} = U_\text{kept}s_\text{kept}^{-1}U_\text{kept}^{\mathsf T}$ is
the pseudo-inverse $J2^{+}$ over the kept space. So

$$ B^{\mathsf T}B = J3\,J2^{+}J3^{\mathsf T} , $$

the same-kernel fit of section @sec:factorization-and-metric, here in the periodic kernel.

**Why the fit is variational.** $I$, $J2$ and $J3$ are Gram matrices of one
positive kernel (pair densities with pair densities, auxiliaries with
auxiliaries, pairs with auxiliaries). So the block matrix
$\begin{pmatrix}I & J3\\ J3^{\mathsf T} & J2\end{pmatrix}$ is positive
semidefinite, and the fitting error $I - B^{\mathsf T}B$ is its Schur
complement: positive semidefinite too. For any density $d$,
$d^{\mathsf T}B^{\mathsf T}Bd \le d^{\mathsf T}Id$: the fitted Coulomb energy
never exceeds the exact one in the kernel the energy uses. A charged
auxiliary function needs no compensating charge, because the only divergent
term, $\mathbf G = 0$, is absent from $I$, $J2$ and $J3$ alike.

**Check the trivial limit.** When the auxiliary functions span the pair
densities, $J3$ and $J2$ describe the same space as $I$ and the fit is exact,
$B^{\mathsf T}B = I$. The code test compares the fitted ERI with the dense
tensor of section @sec:exchange-divergence in that limit.

> **Verified equation {#eq:pbc-rsgdf-fit}** · the robust fit is variational and exact in the trivial limit
>
> $B = (J3\,U_\text{kept}s_\text{kept}^{-1/2})^{\mathsf T}$ gives
> $B^{\mathsf T}B = J3\,J2^{+}J3^{\mathsf T}$; the fitting error
> $I - B^{\mathsf T}B$ is positive semidefinite with charged auxiliary
> functions; when the auxiliary set is the pair densities,
> $B^{\mathsf T}B = I$.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_rsgdf_sympy.py`, claims `fit-psd`, `fit-trivial` |
> | Negative control | the runner's mutation: eigenvectors scaled by $1/s$ instead of $1/\sqrt s$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/rsgdf.rs::fit_with_metric` |
> | Code test | `rsgdf_matches_dense_aft_in_the_trivial_limit` in `crates/ferric-pbc/tests/pbc_rsgdf.rs` |
>
> <!-- verified: id=pbc.rsgdf.fit; script=scripts/pbc_rsgdf_sympy.py; claim=fit-trivial; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/rsgdf.rs::fit_with_metric; rust_test=ferric-pbc/tests/pbc_rsgdf.rs::rsgdf_matches_dense_aft_in_the_trivial_limit; registry=pbc-hf -->

**J and K from one factor.** Insert the fitted ERI into $J$ and $K$ and
group the sums. With $B_k$ the $(n_\text{ao}, n_\text{ao})$ slice of row $k$:

$$
J_{\mu\nu} = \sum_k B_{k,\mu\nu}\sum_{\lambda\sigma}B_{k,\lambda\sigma}D_{\lambda\sigma},\qquad
K = \sum_k B_k\,D\,B_k^{\mathsf T} + v_M\,SDS ,
$$

with the Madelung term of section @sec:exchange-divergence for `exxdiv = ewald`. $J$ is two
matrix–vector products, $B^{\mathsf T}(Bd)$. $K$ is a sum over $k$ of
$n_\text{ao}$-square matrix products. One $B$ serves both builders (`RsGdfJ`,
`RsGdfK`), so the fitted ERI stays a single positive semidefinite
$B^{\mathsf T}B$. The exchange sum runs over 24 fixed auxiliary groups
(`EXCHANGE_AUX_GROUPS`), so every bit of $K$ depends on $n_\text{aux}$ alone,
not on the thread count.

> **Verified equation {#eq:pbc-rsgdf-jk}** · J and K from one factor
>
> $J = B^{\mathsf T}(Bd)$ and $K = \sum_kB_kDB_k^{\mathsf T}$ are the
> Coulomb and exchange contractions of the fitted ERI $\sum_kB_kB_k^{\mathsf T}$.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_rsgdf_sympy.py`, claim `fit-jk` |
> | Negative control | none in the script |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/rsgdf.rs::{j_builder,k_builder}` |
> | Code test | `rhf_energy_is_rsgdf_omega_independent_and_the_build_follows_omega` in `crates/ferric-pbc/tests/pbc_rsgdf.rs` |
>
> <!-- verified: id=pbc.rsgdf.jk; script=scripts/pbc_rsgdf_sympy.py; claim=fit-jk; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/rsgdf.rs::{j_builder,k_builder}; rust_test=ferric-pbc/tests/pbc_rsgdf.rs::rhf_energy_is_rsgdf_omega_independent_and_the_build_follows_omega; registry=pbc-hf -->

**At a $k$-point mesh.** `rsgdf/kpoint.rs` fits each momentum-transfer class
$q = k' - k$ separately. The Bloch pair density $P^{k'}_{\mu\nu}(\mathbf K)$ at
$\mathbf K = \mathbf G + q$ (section @sec:kpoint-scf) is fitted in the $q$-Bloch
auxiliary set $X^q_P = \sum_{\mathbf T}e^{iq\cdot\mathbf T}\chi_P(\mathbf r - \mathbf T)$,
with a Hermitian metric $J2(q) = UsU^{\mathsf H}$ cut per $q$. The
real-space erfc integrals do not depend on $q$ or $k$: the Gamma walk is
computed once, binned by $(\mathbf L \bmod N, \mathbf T \bmod N)$, and each
$(q, k')$ is a phase contraction of the bins. Only $q = 0$ contains
$\mathbf K = 0$, so only $q = 0$ subtracts $c_0$. Time reversal gives
$J2(-q) = J2(q)^*$, and only one of each $(q, -q)$ is built. The mesh fit is
the Gamma fit of the $\operatorname{diag}(N)$ supercell, block-diagonal in
$q$; `crates/ferric-pbc/tests/pbc_krsgdf.rs` tests it.

## The algorithm {#rsgdf-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | SR metric: $\sum_{\mathbf T}(P_0|Q_{\mathbf T})_{\operatorname{erfc}}$ | $N_\text{aux}^2N_{\mathbf T}$ | `Engine::compute_eri2_shifted` |
| 2 | SR three-index: $\sum_{\mathbf L,\mathbf T}(\mu_0\nu_{\mathbf L}|P_{\mathbf T})_{\operatorname{erfc}}$, screened per triplet, each unordered shell pair once | $N_\text{pair}N_{\mathbf L}N_\text{aux}N_{\mathbf T}$ | the SR walk in `rsgdf.rs` |
| 3 | LR on the half $\mathbf G$ sphere, chunked: aux and pair transforms, GEMMs | $n_\text{ao}^2N_\text{aux}N_{\mathbf G}$ | `aux_ft`, `pair_ft_chunked` |
| 4 | $\mathbf G = 0$: $J2 \mathrel{-}= c_0qq^{\mathsf T}$, $J3 \mathrel{-}= c_0Sq^{\mathsf T}$ | $n_\text{ao}^2N_\text{aux}$ | `subtract_g0` |
| 5 | $J2 = UsU^{\mathsf T}$, keep $s >$ `lindep`, $B = (J3\,W)^{\mathsf T}$ | $N_\text{aux}^3 + n_\text{ao}^2N_\text{aux}^2$ | `fit_with_metric` |
| 6 | per SCF iteration: $J$, $K$ from $B$, plus $v_MSDS$ | $n_\text{ao}^3N_\text{aux}$ | `RsGdfJ`, `RsGdfK` |

The $\mathbf G = 0$ bookkeeping is one function:

<!-- excerpt: ferric-pbc/src/rsgdf.rs::subtract_g0 -->
```rust
    if do_metric {
        for (p, qp) in q.iter().enumerate() {
            for (r, qr) in q.iter().enumerate() {
                j2[(p, r)] -= c0 * qp * qr;
            }
        }
    }
    if do_three {
        for (mn, smn) in s_flat.iter().enumerate() {
            for (p, qp) in q.iter().enumerate() {
                j3[(mn, p)] -= c0 * smn * qp;
            }
        }
    }
```

| Symbol | Code |
|---|---|
| $c_0 = \pi/(\eta^2\Omega)$ | `c0` |
| $q_P$ | `q[p]` |
| $S_{\mu\nu}$ | `s_flat[mn]` |
| production / mutants | `mode` (`G0Handling::Consistent` in production) |

`do_metric` and `do_three` are both true in production.
`G0Handling::MetricOnlyMutant` skips the three-index subtraction and
`G0Handling::NeitherMutant` skips both; the anchors of @sec:rsgdf-numerical-anchors detect each.

The fit is this part of `ferric-pbc/src/rsgdf.rs::fit_with_metric`:

<!-- excerpt: ferric-pbc/src/rsgdf.rs::fit_with_metric -->
```rust
    let keep: Vec<usize> = (0..naux).filter(|&k| evals[k] > lindep).collect();
    if keep.is_empty() {
        return Err(FerricError::General(format!(
            "RsGdf: every metric eigenvalue is <= lindep {lindep:e} (max {:e}); the aux basis \
             has no fitting power in the G = 0-dropped kernel",
            evals[naux - 1]
        )));
    }
    let mut w = Array2::<f64>::zeros((naux, keep.len()));
    for (c, &k) in keep.iter().enumerate() {
        let sc = 1.0 / evals[k].sqrt();
        for r in 0..naux {
            w[(r, c)] = evecs[(r, k)] * sc;
        }
    }
    let bt = j3.dot(&w); // (n², nkeep)
```

| Code | Equation |
|---|---|
| `evals`, `evecs` | $s$, $U$ of $J2 = UsU^{\mathsf T}$ |
| `keep` | the eigenvalues $s_k >$ `lindep` |
| `w` | $W = U_\text{kept}s_\text{kept}^{-1/2}$ |
| `bt` | $J3\,W$; its transpose is $B$ |

An auxiliary basis with no eigenvalue above `lindep` has no fitting power in
the $\mathbf G = 0$-dropped kernel, and the build stops with that message.

## Reproducibility and memory {#rsgdf-reproducibility-memory}

- The SR three-index walk evaluates each unordered shell pair once and copies
  the block into both orderings, so $J3_\text{SR}$ is exactly symmetric and
  every element is owned by one task: the build is bitwise identical at any
  thread count. The symmetry is a lattice translation,
  $(\nu_0\mu_{\mathbf L}|P_{\mathbf T}) = (\mu_0\nu_{-\mathbf L}|P_{\mathbf T-\mathbf L})$.
- Exchange is summed over 24 fixed auxiliary groups (`EXCHANGE_AUX_GROUPS`),
  so $K$ depends on $N_\text{aux}$ only.
- Every buffer that grows with the lattice, the $\mathbf G$ sphere or
  $N_\text{aux}n_\text{ao}^2$ is reserved on the memory ledger before it is
  allocated. The resident $B$ is $8N_\text{aux}n_\text{ao}^2$ bytes, in core;
  there is no spill path.
- The metric is never Cholesky- or LU-factored: diffuse auxiliary functions in
  a small cell make it near-singular, and the eigenvalue cut is the stable
  solve. The number of dropped eigenvalues is reported. This is also why the
  molecular `DfJ`/`DfK` builders are not reused: both recompute a molecular
  metric, and `DfJ` inverts it with LU.
- The auxiliary basis must have pure (solid-harmonic) shells above $l = 1$:
  libint2's two- and three-centre integrals abort on a Cartesian one, so the
  build refuses such a basis with a typed error first.

## Options {#rsgdf-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `[cell] jk`, `jk=` | `"dense"`, `"rsgdf"` | `"dense"` | the J/K builder |
| `[cell] auxbasis`, `auxbasis=` | basis name, e.g. `"cc-pvdz-ri"` | none (required for `rsgdf`) | the fitting basis |
| `[cell] gdf_omega`, `gdf_omega=` | positive, in `unit`$^{-1}$, or `"auto"` | 1 Bohr$^{-1}$ | the fit's $\eta$; any value gives the same energy to the fitting precision, and moves work between the real-space and reciprocal sums |
| `[cell] range_split` | `false`, `true` ($\lambda = 1$), or $\lambda > 0$ | `false` | moves the short-range blocks whose transform converges in the long-range sphere into $\mathbf G$ space; Gamma point or `kmesh`, `task = "energy"` only; refused with a range-separated hybrid at Gamma and with `task = "optimize"` |
| `[cell] sr_column_rotation` | `true`, `false` | on with `rsgdf` | evaluates the SR walk on column-rotated general contractions, back-transformed exactly |
| `[memory] budget_gb`, `memory_budget_gb=` | GB | auto | the ledger |

## Numerical anchors {#rsgdf-numerical-anchors}

| Quantity | Bar | Test |
|---|---|---|
| fitted ERI against the dense AFT tensor when the auxiliary set spans the pairs | $10^{-10}$, energies $10^{-10}$ Ha | `crates/ferric-pbc/tests/pbc_rsgdf.rs` `rsgdf_matches_dense_aft_in_the_trivial_limit` |
| the two $\mathbf G = 0$ mutants are detected | above $10^{-2}$ and $10^{-3}$ | `g0_mutants_and_incomplete_aux_fail_the_anchor` |
| fitted ERI at different $\eta$ | $10^{-9}$ | `fitted_eri_is_omega_independent` |

## Sources {#rsgdf-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-pbc/theory.md`, §7 | the fit, the split, the $k$-point form |
| `scripts/pbc_rsgdf_sympy.py` | the proof |
| H. Ye and T. C. Berkelbach, *J. Chem. Phys.* **154**, 131104 (2021) | range-separated GDF |
| B. I. Dunlap, *J. Mol. Struct. THEOCHEM* **529**, 37 (2000) | the robust fit |
