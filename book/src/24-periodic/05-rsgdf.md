# 24.5 Range-separated Gaussian density fitting for J and K

<!-- section: 24.5; status: ready; registry: pbc-hf;
     theory: crates/ferric-pbc/theory.md#"7. Range-separated Gaussian density fitting (RS-GDF)" -->

The dense builder of section 24.4 stores $n_\text{ao}^4$ Coulomb integrals and
sums a $\mathbf G$ sphere that grows with the cell volume and the steepest
exponent. It is a correctness oracle for toy cells. Production cells use
`jk = "rsgdf"`: a density fit in an auxiliary Gaussian basis, with the same
Ewald split as section 24.1 applied to the two- and three-index integrals.
The fit keeps the convention that drops $\mathbf G = 0$, so it fits the same
kernel the energy uses.

| Symbol | Meaning | Code name |
|---|---|---|
| $\mu\nu$, $\lambda\sigma$ | orbital AO pairs, flattened `m*nao + n` | |
| $P, Q$ | auxiliary functions (pure shells only) | `naux` |
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

<!-- include: crates/ferric-pbc/theory.md#"7. Range-separated Gaussian density fitting (RS-GDF)" -->

## 24.5.1 Verified equations

> **Verified equation 24.8** · the Ewald-split fit integrals
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

> **Verified equation 24.9** · the robust fit is variational and exact in the trivial limit
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

> **Verified equation 24.10** · J and K from one factor
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

## 24.5.2 The algorithm

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

## 24.5.3 Reproducibility and memory

- The SR three-index walk evaluates each unordered shell pair once and copies
  the block into both orderings, so $J3_\text{SR}$ is exactly symmetric and
  every element is owned by one task: the build is bitwise identical at any
  thread count.
- Exchange is summed over 24 fixed auxiliary groups (`EXCHANGE_AUX_GROUPS`),
  so $K$ depends on $N_\text{aux}$ only.
- Every buffer that grows with the lattice, the $\mathbf G$ sphere or
  $N_\text{aux}n_\text{ao}^2$ is reserved on the memory ledger before it is
  allocated. The resident $B$ is $8N_\text{aux}n_\text{ao}^2$ bytes, in core;
  there is no spill path.
- The metric is never Cholesky- or LU-factored: diffuse auxiliary functions in
  a small cell make it near-singular, and the eigenvalue cut is the stable
  solve. The number of dropped eigenvalues is reported.

## 24.5.4 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `[cell] jk`, `jk=` | `"dense"`, `"rsgdf"` | `"dense"` | the J/K builder |
| `[cell] auxbasis`, `auxbasis=` | basis name, e.g. `"cc-pvdz-ri"` | none (required for `rsgdf`) | the fitting basis |
| `[cell] gdf_omega`, `gdf_omega=` | positive, in `unit`$^{-1}$, or `"auto"` | 1 Bohr$^{-1}$ | the fit's $\eta$; any value gives the same energy to the fitting precision, and moves work between the real-space and reciprocal sums |
| `[cell] range_split` | `false`, `true` ($\lambda = 1$), or $\lambda > 0$ | `false` | moves the short-range blocks whose transform converges in the long-range sphere into $\mathbf G$ space; Gamma point or `kmesh`, `task = "energy"` only; refused with a range-separated hybrid at Gamma and with `task = "optimize"` |
| `[cell] sr_column_rotation` | `true`, `false` | on with `rsgdf` | evaluates the SR walk on column-rotated general contractions, back-transformed exactly |
| `[memory] budget_gb`, `memory_budget_gb=` | GB | auto | the ledger |

## 24.5.5 Numerical anchors

| Quantity | Bar | Test |
|---|---|---|
| fitted ERI against the dense AFT tensor when the auxiliary set spans the pairs | $10^{-10}$, energies $10^{-10}$ Ha | `crates/ferric-pbc/tests/pbc_rsgdf.rs` `rsgdf_matches_dense_aft_in_the_trivial_limit` |
| the two $\mathbf G = 0$ mutants are detected | above $10^{-2}$ and $10^{-3}$ | `g0_mutants_and_incomplete_aux_fail_the_anchor` |
| fitted ERI at different $\eta$ | $10^{-9}$ | `fitted_eri_is_omega_independent` |

## 24.5.6 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-pbc/theory.md`, §7 | the fit, the split, the $k$-point form |
| `scripts/pbc_rsgdf_sympy.py` | the proof |
| H. Ye and T. C. Berkelbach, *J. Chem. Phys.* **154**, 131104 (2021) | range-separated GDF |
| B. I. Dunlap, *J. Mol. Struct. THEOCHEM* **529**, 37 (2000) | the robust fit |
