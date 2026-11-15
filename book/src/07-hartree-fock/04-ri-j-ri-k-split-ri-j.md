# 7.4 RI-J, RI-K and Split-RI-J

<!-- section: 7.4; status: ready; registry: rhf, uhf, rohf;
     theory: crates/ferric-scf/theory.md#"Density-fitted Coulomb and exchange (RI-J, RI-K, Split-RI-J)" -->

Density fitting replaces each orbital product by its expansion in an
auxiliary basis, so the four-centre integrals factor into three-centre ones.
It is the default for Kohn–Sham runs and opt-in for Hartree–Fock. This
section states the fit, the three ways the code contracts exchange, and how
Split-RI-J computes the same Coulomb matrix without storing the tensor.

| Symbol | Meaning | Code name |
|---|---|---|
| $(P\vert\mu\nu)$ | raw three-index integrals, packed $\mu \ge \nu$ | the stored tensor of `DfJ` |
| $V_{PQ}$ | Coulomb metric $(P\vert Q)$ | the metric |
| $B^P_{\mu\nu}$ | dressed factor $\sum_Q M_{PQ}(Q\vert\mu\nu)$, $M^{\mathsf T}M = V^{-1}$ | the dressed tensor of `DfK` |
| $w_{\mu\nu}$ | packed density weights | `packed_density_weights` |
| $X_\pm$ | signed factors of an indefinite $D$ | `x_pos`, `x_neg` |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/df_j.rs::DfJ`, `ferric-scf/src/df_k.rs::DfK`, `ferric-scf/src/split_rij.rs::SplitRiJ` |
| CLI | `[scf] df_j_aux`, `df_k_aux`, `jk_storage`, `ri_j_precision`; KS defaults to `def2-universal-jkfit` |
| Python | `df_j_aux=`, `df_k_aux=` on `run_rhf`, `run_uhf`, `run_rohf`, `run_dft` |

Which route runs for which functional and spin is the table in
`site/src/methods/scf.md`, "Choosing how exchange is built"; this section is
about what each route computes.

## 7.4.1 The equations

<!-- include: crates/ferric-scf/theory.md#"Density-fitted Coulomb and exchange (RI-J, RI-K, Split-RI-J)" -->

> **Verified equation 7.4** · the Coulomb fit and its error
>
> The fit $Vc = \gamma$ is the Coulomb-metric projection of $\rho$, and
> $(\rho|\rho) - \gamma^{\mathsf T}V^{-1}\gamma = (\rho - \tilde\rho|\rho - \tilde\rho) \ge 0$:
> the fitted Coulomb energy errs low, at second order in the residual. In a
> complete auxiliary space RI-J and RI-K are exact.
>
> | | |
> |---|---|
> | Proof | `scripts/scf_jk_builders_sympy.py`, claims S1, S2, S3, N1 and N6 |
> | Negative control | $c = \gamma$ in place of $V^{-1}\gamma$ must fail N1 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/df_j.rs::DfJ` |
>
> <!-- verified: id=scf.rij-fit; script=scripts/scf_jk_builders_sympy.py; claim=N1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/df_j.rs::DfJ; registry=rhf -->

> **Verified equation 7.5** · three contractions of one RI-K
>
> With any $M$ satisfying $M^{\mathsf T}M = V^{-1}$, the density route
> $\sum_P B^PDB^P$, the occupied route $2\sum_P(B^PC_o)(B^PC_o)^{\mathsf T}$
> for $D = 2C_oC_o^{\mathsf T}$, and the signed route
> $K_{\rm occ}(X_+) - K_{\rm occ}(X_-)$ give the same $K$. Cholesky and
> symmetric dressings agree; a truncated $V^{-1/2}$ fits the kept span.
>
> | | |
> |---|---|
> | Proof | `scripts/scf_jk_builders_sympy.py`, claims N2, N3, N4, N5 and N8 |
> | Negative control | the RHF factor 2 → 1 must fail N4; the signed `-` → `+` must fail N5 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/df_k.rs::{DfK,build_from_density_factor}` |
>
> <!-- verified: id=scf.rik-routes; script=scripts/scf_jk_builders_sympy.py; claim=N5; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/df_k.rs::{DfK,build_from_density_factor}; registry=rhf -->

## 7.4.2 The algorithm

| Step | RI-J | RI-K | Split-RI-J |
|---|---|---|---|
| once per geometry | $(P\vert\mu\nu)$ packed, Cholesky of $V$ | $B = M(P\vert\mu\nu)$, $M$ the truncated $V^{-1/2}$ | Cholesky of $V$ only |
| per build | $\gamma = (P\vert\mu\nu)\,w$, solve $Vc = \gamma$, $J = (\mu\nu\vert P)c$ | one of the three routes | recompute the integrals, contract $D$ and $c$ in Hermite form |
| storage | $N_{\rm aux}N(N+1)/2$, f32 by default | $N_{\rm aux}N^2$ f64, in core or spilled | none |

The packed weights carry the factor 2 of the off-diagonal pairs:

<!-- excerpt: ferric-scf/src/df_j.rs::packed_density_weights -->
```rust
for mu in 0..n {
        for nu in 0..mu {
            w[k] = d[(mu, nu)] + d[(nu, mu)];
            k += 1;
        }
        w[k] = d[(mu, mu)];
        k += 1;
    }
```

| Code | Equation |
|---|---|
| `w[k]`, `mu > nu` | $w_{\mu\nu} = D_{\mu\nu} + D_{\nu\mu}$ |
| `w[k]`, `mu == nu` | $w_{\mu\mu} = D_{\mu\mu}$ |
| `k` | the packed pair index, row-major lower triangle |

> **Verified equation 7.6** · packed weights
>
> $\sum_{\mu\nu}(P|\mu\nu)D_{\mu\nu} = \sum_{\mu\ge\nu}(P|\mu\nu)w_{\mu\nu}$ for
> any $D$.
>
> | | |
> |---|---|
> | Proof | `scripts/scf_jk_builders_sympy.py`, claim N7 |
> | Negative control | off-diagonal weight 2 → 1 must fail N7 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/df_j.rs::packed_density_weights` |
>
> <!-- verified: id=scf.rij-packed; script=scripts/scf_jk_builders_sympy.py; claim=N7; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/df_j.rs::packed_density_weights; registry=rhf -->

The signed route for an indefinite density is two occupied builds:

<!-- excerpt: ferric-scf/src/df_k.rs::build_from_density_factor -->
```rust
Some((x_pos, x_neg)) => {
                self.build_from_occ_impl(&x_pos, k)?;
                if x_neg.ncols() > 0 {
                    let mut k_neg = Array2::<f64>::zeros(k.raw_dim());
                    self.build_from_occ_impl(&x_neg, &mut k_neg)?;
                    *k -= &k_neg;
                }
                Ok(0)
            }
```

| Code | Equation |
|---|---|
| `x_pos`, `x_neg` | $X_\pm = W_\pm\lvert\Lambda_\pm\rvert^{1/2}$ |
| `build_from_occ_impl(&x, k)` | $K_{\rm occ}(X) = \sum_P(B^PX)(B^PX)^{\mathsf T}$ |
| `*k -= &k_neg` | $K = K_{\rm occ}(X_+) - K_{\rm occ}(X_-)$ |

## 7.4.3 Precision

RI-J stores the tensor in f32 for single points by default. Every other
quantity is f64. The SCF then converges on the operator of the rounded tensor,
so the energy error is first order in the rounding, which
`site/src/methods/scf.md` states. Geometry optimizations and frequencies
default to f64.

**Open item.** `DfJ::f64_exit_coulomb_energy` would make that error second
order by recomputing $E_J$ once at the converged density in f64. It is
implemented and tested (`crates/ferric-scf/tests/dfj_f32_storage.rs`) but no
production path calls it, while the module doc of `crates/ferric-scf/src/df_j.rs`
describes the f64 exit as if the SCF energy used it. Either the exit is wired
in or the module doc says it is unused.

## 7.4.4 Split-RI-J

Split-RI-J solves the same $Vc = \gamma$ with the same Cholesky factor. The
difference is that $(P|\mu\nu)$ is recomputed each build in a
McMurchie–Davidson kernel that never forms the Cartesian block. Two
approximations sit on top of RI-J: the large-argument Boys asymptote where its
error is at most $10^{-14}$ per term (`FAR_EPS`), and a Schwarz-type screen on
(orbital pair, auxiliary shell) blocks. It is RI-J to those tolerances, not to
rounding. It is chosen by `jk_storage = "split"`, or by `"auto"` when the
three-index tensor does not fit the budget
(`crates/ferric-scf/tests/auto_split_rij.rs`).

## 7.4.5 Numerical anchors

The fitting error is a property of the auxiliary basis and grows with system
size; `site/src/methods/scf.md` records that RI-JK has not been measured
against exact exchange on the systems of its exchange table. The equations of
this section are checked by the proofs above and by
`crates/ferric-scf/tests/split_rij.rs` (Split-RI-J against a reference
tensor) and `crates/ferric-scf/tests/dfj_f32_storage.rs` (f32 storage).

## 7.4.6 Options that change the equation

| Option | Effect |
|---|---|
| `df_j_aux`, `df_k_aux` | the auxiliary basis; `""` turns fitting off |
| `ri_j_precision = "f32"` / `"f64"` | RI-J tensor storage |
| `jk_storage = "auto"`, `"memory"`, `"disk"`, `"direct"`, `"split"` | where the tensor lives, or Split-RI-J |
| a range-separated hybrid | two RI-K fitters, $\mathrm{erfc}(\omega r)/r$ and $\mathrm{erf}(\omega r)/r$, on one auxiliary basis |

## 7.4.7 Doc fixes needed

| Text | Code |
|---|---|
| `df_j.rs` module doc: "the inverse Coulomb metric $V^{-1}$ [is] precomputed" | a Cholesky factor is stored and solved against; no inverse is formed |
| `df_j.rs` module doc: f64 exit makes the error second order | the exit is not called (open item above) |
| orphaned doc comments in `df_k.rs` and `fock_assembly.rs` | attach or delete |
| `split_rij.rs` describes the build as exact | it carries `FAR_EPS` and the block screen |

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/theory.md`, "Density-fitted Coulomb and exchange (RI-J, RI-K, Split-RI-J)" | the equations |
| `scripts/scf_jk_builders_sympy.py` | verified equations 7.4 to 7.6 |
| `site/src/methods/scf.md`, "Choosing how exchange is built" | routes and the f32 statement |
