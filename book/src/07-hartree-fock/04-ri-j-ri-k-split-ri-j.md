# RI-J, RI-K and Split-RI-J {#ri-j-ri-k-split-ri-j}

<!-- status: ready; registry: rhf, uhf, rohf -->

Density fitting (chapter @sec:density-fitting) expands each orbital product
$\chi_\mu\chi_\nu$ in an auxiliary basis, so the four-centre integrals of
section @sec:direct-and-incremental factor into three-centre ones that are computed once per geometry
and stored. It is the default for Kohn–Sham runs and opt-in for
Hartree–Fock. Coulomb and exchange use the fit differently. The Coulomb matrix
is the potential of one charge cloud, the density, so one fit per iteration
is enough. Exchange has no single cloud, because the density connects the two
electrons of each integral, so every exchange product has to be fitted.

This section derives RI-J and its error, the three ways the code contracts
exchange, and how Split-RI-J computes the same Coulomb matrix without storing
the tensor; then precision, anchors and options.

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

## The equations {#ri-j-ri-k-split-ri-j-equations}

### RI-J

Take the density $\rho = \sum_{\lambda\sigma}D_{\lambda\sigma}\chi_\lambda\chi_\sigma$
and fit it, $\tilde\rho = \sum_Pc_P\chi_P$, by minimizing the Coulomb
self-energy of the residual $\rho - \tilde\rho$. Section @sec:factorization-and-metric-factorization does this
minimization for one product; for the density it is the same with $D$
contracted in. Setting the derivative with respect to $c$ to zero gives

$$ \gamma_P = \sum_{\mu\nu}(P|\mu\nu)D_{\mu\nu}, \qquad Vc = \gamma, \qquad J_{\mu\nu} = \sum_P(\mu\nu|P)\,c_P . $$

The first step projects the density onto each auxiliary function, the second
removes the double counting from their overlap, and the third evaluates the
potential of $\tilde\rho$ between $\chi_\mu$ and $\chi_\nu$. Each step is a
matrix–vector product with the stored tensor, or a solve with the
$N_{\rm aux}\times N_{\rm aux}$ metric, so a build costs
$O(N_{\rm aux}N^2)$ instead of the $O(N^4)$ of the direct contraction.

Check the complete-basis limit. If $\rho$ lies in the span of the $\chi_P$,
the fit reproduces it and $J$ is exact (section @sec:factorization-and-metric-factorization).

**Packed storage.** The code stores $(P|\mu\nu)$ undressed, as the packed
lower triangle, since $(P|\mu\nu) = (P|\nu\mu)$. Split the sum for $\gamma$
into the diagonal and the two triangles and use that symmetry:

$$ \gamma_P = \sum_\mu(P|\mu\mu)D_{\mu\mu} + \sum_{\mu>\nu}(P|\mu\nu)\,(D_{\mu\nu} + D_{\nu\mu}) = \sum_{\mu\ge\nu}(P|\mu\nu)\,w_{\mu\nu}, $$

with $w_{\mu\mu} = D_{\mu\mu}$ and $w_{\mu\nu} = D_{\mu\nu} + D_{\nu\mu}$
($\nu < \mu$) (`packed_density_weights`). Only the symmetric part of $D$
enters.

> **Verified equation {#eq:scf-rij-packed}** · packed weights
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

**The solve.** $Vc = \gamma$ is solved with the Cholesky factor
$V = LL^{\mathsf T}$ computed once: no explicit inverse, no eigenvalue cut. A
metric that is not positive definite is an error. Section @sec:factorization-and-metric-factors explains why
a failed Cholesky factorization is treated as a defect of the input.

**The error.** How wrong is the fitted Coulomb energy? Expand the residual
self-energy and substitute $c = V^{-1}\gamma$:

$$ (\rho - \tilde\rho|\rho - \tilde\rho) = (\rho|\rho) - 2c^{\mathsf T}\gamma + c^{\mathsf T}Vc = (\rho|\rho) - \gamma^{\mathsf T}V^{-1}\gamma . $$

The fitted Coulomb energy is $\tfrac12\gamma^{\mathsf T}V^{-1}\gamma$, so its
error is the self-interaction of the residual,

$$ \tfrac12(\rho|\rho) - \tfrac12\gamma^{\mathsf T}V^{-1}\gamma = \tfrac12(\rho - \tilde\rho\,|\,\rho - \tilde\rho) \ \ge 0, $$

second order in the residual. The Coulomb self-energy of any charge
distribution is non-negative, so the fitted energy errs low.

The robust (Dunlap) form $2\gamma^{\mathsf T}c' - c'^{\mathsf T}Vc'$
(section @sec:robust-fit) has the same value at $c' = c$, namely
$2\gamma^{\mathsf T}V^{-1}\gamma - \gamma^{\mathsf T}V^{-1}\gamma$, and its
derivative $2\gamma - 2Vc'$ vanishes there. That is why the plain and robust
energies coincide for RI-J with one density.

> **Verified equation {#eq:scf-rij-fit}** · the Coulomb fit and its error
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

**f32 storage** (`ri_j_precision = "f32"`, the single-point default). Each
$(P|\mu\nu)$ is rounded to f32 once; $w$, $\gamma$, $c$, the solve and $J$
are f64. This halves the stored tensor, the largest object of an RI-J run
(section @sec:three-index-sources). The SCF converges on the operator of the rounded tensor, so the
energy carries an error first order in the rounding. Geometry and frequency
tasks default to f64.

Water in cc-pVDZ with def2-universal-jkfit for both fits, from the `589b50832`
build, shows the two errors side by side: the fitting error of RI-JK against
exact $J$ and $K$, and the f32 rounding on top of it, about fifty times smaller.

```python
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = "def2-universal-jkfit"

e_direct = ferric.run_rhf(mol, bs).energy                      # exact J and K
e_rijk64 = ferric.run_rhf(mol, bs, df_j_aux=aux, df_k_aux=aux,
                          ri_j_precision="f64").energy         # fitted, f64 tensor
e_rijk32 = ferric.run_rhf(mol, bs, df_j_aux=aux, df_k_aux=aux).energy  # f32 RI-J default
print(f"direct              {e_direct:.10f}")
print(f"RI-JK (f64) - direct {e_rijk64 - e_direct:.2e}")
print(f"RI-JK f32 - f64      {e_rijk32 - e_rijk64:.2e}")
```

```text
direct              -76.0267679974
RI-JK (f64) - direct 3.26e-05
RI-JK f32 - f64      -6.28e-07
```

### RI-K

Exchange pairs the two electrons differently:
$K_{\mu\nu} = \sum_{\lambda\sigma}(\mu\lambda|\nu\sigma)D_{\lambda\sigma}$, so
the products to fit are $\chi_\mu\chi_\lambda$ and $\chi_\nu\chi_\sigma$, and
$D$ sits between them. Fit each product with the factorization (@eq:factorization-and-metric-1),
$(\mu\lambda|\nu\sigma) \approx \sum_PB^P_{\mu\lambda}B^P_{\nu\sigma}$, and
use $B^P_{\nu\sigma} = B^P_{\sigma\nu}$:

$$ K_{\mu\nu} = \sum_P\sum_{\lambda\sigma}B^P_{\mu\lambda}D_{\lambda\sigma}B^P_{\sigma\nu} = \sum_P\big(B^PDB^P\big)_{\mu\nu} . $$

Each $B^P$ is a symmetric $N\times N$ matrix, and $K$ is a sum of
$N_{\rm aux}$ matrix triple products. The inverse metric is folded into $B$
once per geometry, so RI-K stores the dressed tensor and needs no solve per
build. RI-J stores the raw tensor and solves with $V$ each build, which costs
one $N_{\rm aux}$-vector solve.

The dressing for K is the symmetric $M = U_s\lambda_s^{-1/2}U_s^{\mathsf T}$
over the eigenpairs of $V$ with $\lambda$ above a cut of about $10^{-10}$
(`v_inv_sqrt_lindep`, `metric_lindep_cut`): $M^{\mathsf T}M$ is the
pseudo-inverse of $V$ on the kept span. RI-J uses the full Cholesky inverse,
so the two fits differ only when modes are cut.

**Three contractions.** They give the same $K$ and differ in cost:

- Density route, $\sum_PB^PDB^P$, for any $D$ (`density_chunk_k`). Two
  $N\times N$ products per auxiliary function: $O(N_{\rm aux}N^3)$.
- Occupied route, $K_{\rm occ}(X) = \sum_P(B^PX)(B^PX)^{\mathsf T}$, with no
  occupation factor (`occ_chunk_k`, `occ_half_panel`). Substitute
  $D = 2C_oC_o^{\mathsf T}$ into the density route:
  $B^PDB^P = 2(B^PC_o)(B^PC_o)^{\mathsf T}$. So RHF multiplies by 2, and UHF
  passes $C^\sigma_o$ with factor 1. With $n_o$ occupied orbitals the cost is
  $O(N_{\rm aux}N^2n_o)$, smaller by $n_o/N$.
- Signed factor route, for a symmetric $D$ without orbitals
  (`build_from_density_factor`). Diagonalize $D = W\Lambda W^{\mathsf T}$,
  keep the eigenvalues above $+\epsilon$ and below $-\epsilon$
  ($\epsilon = 10^{-11}\max|\Lambda|$), and set $X_\pm = W_\pm|\Lambda_\pm|^{1/2}$.
  Then $D \approx X_+X_+^{\mathsf T} - X_-X_-^{\mathsf T}$, and inserting this
  in the density route gives $K = K_{\rm occ}(X_+) - K_{\rm occ}(X_-)$. It
  falls back to the density route when $D$ is not symmetric to $10^{-12}$ or
  the kept columns exceed $3N/4$; the occupied-route cost grows with the
  number of columns.

The incremental and response densities, which are indefinite, use the density
route or the factored response form. For $D = XC_o^{\mathsf T} + C_oX^{\mathsf T}$,
substituting into $B^PDB^P$ gives two terms that are transposes of each
other:

$$ K = G + G^{\mathsf T}, \qquad G = \sum_P(B^PX)(B^PC_o)^{\mathsf T} . $$

The CPHF kernel of the Hessian uses this form for every spin and every
coordinate; section @sec:hf-hessian-factored-ri-k gives $X$ and $C_o$ for UHF, UKS and ROHF
and the cost against the density route.

A range-separated hybrid contracts two fitters,
$K = c_{\rm SR}K[\mathrm{erfc}(\omega)] + c_{\rm LR}K[\mathrm{erf}(\omega)]$,
each with its own metric and dressing on one auxiliary basis.

> **Verified equation {#eq:scf-rik-routes}** · three contractions of one RI-K
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

### Split-RI-J

Split-RI-J (`SplitRiJ`, `crates/ferric-scf/src/split_rij.rs`) uses the same
three equations, $\gamma = (P|\mu\nu)D$, $Vc = \gamma$ by the same Cholesky
factor, $J = (\mu\nu|P)c$, with $(P|\mu\nu)$ never stored. Each build
recomputes the integrals in a McMurchie–Davidson kernel. It contracts $D$
into Hermite densities of each primitive pair before the auxiliary side, and
$c$ into auxiliary Hermite vectors before the orbital side, so the
$n_{\rm cart}^3$ integral block is never formed. The memory cost is that of
the vectors, and the time cost is one integral evaluation per build.

Two approximations are added on top of RI-J: the large-argument asymptote of
the Boys function where its relative error is at most $10^{-14}$ per term
(`FAR_EPS`), and a Schwarz-type screen on (orbital pair, auxiliary shell)
blocks at `integral_thresh`. It applies to the Coulomb operator with orbital
$l \le 4$ and auxiliary $l \le 6$.

`scripts/scf_jk_builders_sympy.py` (claims S1–S3, N1–N8) proves that the fit
is a Coulomb-metric projection, the energy error is the residual self-energy,
RI-J and RI-K are exact for a complete auxiliary set, $M = L^{-1}$ and
$M = V^{-1/2}$ agree, the occupied and signed-factor routes equal the density
route, the cut metric fits the kept span, and the packed weights are exact.
The robust-form stationarity was checked numerically for this section. The
response form $G + G^{\mathsf T}$ is claim F1 of
`scripts/hessian_factored_k_sympy.py` (section @sec:hf-hessian-factored-ri-k).

## The algorithm {#ri-j-ri-k-split-ri-j-algorithm}

| Step | RI-J | RI-K | Split-RI-J |
|---|---|---|---|
| once per geometry | $(P\vert\mu\nu)$ packed, Cholesky of $V$ | $B = M(P\vert\mu\nu)$, $M$ the truncated $V^{-1/2}$ | Cholesky of $V$ only |
| per build | $\gamma = (P\vert\mu\nu)\,w$, solve $Vc = \gamma$, $J = (\mu\nu\vert P)c$ | one of the three routes | recompute the integrals, contract $D$ and $c$ in Hermite form |
| storage | $N_{\rm aux}N(N+1)/2$, f32 by default | $N_{\rm aux}N^2$ f64, in core or spilled | none |

Where the tensor lives (in core, packed, spilled to disk) is the subject of
section @sec:three-index-sources.

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

Between the two passes, the solve is one call on the stored factor:

<!-- excerpt: ferric-scf/src/df_j.rs::solve_metric -->
```rust
self.v_chol
            .solvec(d_p)
```

| Code | Equation |
|---|---|
| `v_chol` | $L$, with $V = LL^{\mathsf T}$, factored once |
| `d_p` | $\gamma$ |
| `solvec` | $c = V^{-1}\gamma$ by a forward and a backward triangular solve |

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

A density with no negative eigenvalues skips the second build. When
`density_factor` returns `None` (a non-symmetric $D$, or too many columns),
the `None` arm runs the density route.

## Precision {#ri-j-ri-k-split-ri-j-precision}

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

## Split-RI-J {#ri-j-ri-k-split-ri-j-split-ri-j}

Split-RI-J solves the same $Vc = \gamma$ with the same Cholesky factor. The
difference is that $(P|\mu\nu)$ is recomputed each build in a
McMurchie–Davidson kernel that never forms the Cartesian block. Two
approximations sit on top of RI-J: the large-argument Boys asymptote where its
error is at most $10^{-14}$ per term (`FAR_EPS`), and a Schwarz-type screen on
(orbital pair, auxiliary shell) blocks. It is RI-J to those tolerances, not to
rounding. It is chosen by `jk_storage = "split"`, or by `"auto"` when the
three-index tensor does not fit the budget
(`crates/ferric-scf/tests/auto_split_rij.rs`). It trades memory for time: it
stores nothing of size $N_{\rm aux}N^2$, and pays one three-index integral
evaluation per build.

## Numerical anchors {#ri-j-ri-k-split-ri-j-numerical-anchors}

The fitting error is a property of the auxiliary basis and grows with system
size; `site/src/methods/scf.md` records that RI-JK has not been measured
against exact exchange on the systems of its exchange table. The equations of
this section are checked by the proofs above and by
`crates/ferric-scf/tests/split_rij.rs` (Split-RI-J against a reference
tensor) and `crates/ferric-scf/tests/dfj_f32_storage.rs` (f32 storage).

## Options that change the equation {#ri-j-ri-k-split-ri-j-options-change-equation}

| Option | Effect |
|---|---|
| `df_j_aux`, `df_k_aux` | the auxiliary basis; `""` turns fitting off |
| `ri_j_precision = "f32"` / `"f64"` | RI-J tensor storage |
| `jk_storage = "auto"`, `"memory"`, `"disk"`, `"direct"`, `"split"` | where the tensor lives, or Split-RI-J |
| a range-separated hybrid | two RI-K fitters, $\mathrm{erfc}(\omega r)/r$ and $\mathrm{erf}(\omega r)/r$, on one auxiliary basis |

## Doc fixes needed {#ri-j-ri-k-split-ri-j-doc-fixes-needed}

| Text | Code |
|---|---|
| `df_j.rs` module doc: "the inverse Coulomb metric $V^{-1}$ [is] precomputed" | a Cholesky factor is stored and solved against; no inverse is formed |
| `df_j.rs` module doc: f64 exit makes the error second order | the exit is not called (open item above) |
| orphaned doc comments in `df_k.rs` and `fock_assembly.rs` | attach or delete |
| `split_rij.rs` describes the build as exact | it carries `FAR_EPS` and the block screen |

## Sources

| Source | Used for |
|---|---|
| `scripts/scf_jk_builders_sympy.py` | verified equations @eq:scf-rij-packed to @eq:scf-rik-routes |
| `site/src/methods/scf.md`, "Choosing how exchange is built" | routes and the f32 statement |
