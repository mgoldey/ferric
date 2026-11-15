# 13.2 The integral-direct path

<!-- section: 13.2; status: gap; registry: none;
     theory: crates/ferric-mp2/theory.md#"2. Closed-shell RI-MP2" -->

The amplitude-threshold method of section 13.1 builds the global
three-index tensor of shape $(N_{\mathrm{aux}}, n_on_v)$ before it truncates
anything. That tensor is $O(N^3)$ in memory and the whitening product behind it is
$O(N^5)$, so the truncation cannot reduce what has already been paid. The
integral-direct path never forms either. It evaluates only the three-index
integrals that the retained pairs need, half-transforms them into one small
strip per occupied orbital, and fits each pair in a domain-local auxiliary
basis. Solve and energy are the unchanged ragged machinery of 13.1.

This section is marked `gap`: the construction is documented in the module
header of `crates/ferric-mp2/src/lmp2_direct.rs` and the user guide, and
`crates/ferric-mp2/theory.md` section 12 is a pointer with no derivation. The
fit identities are machine-checked below. The locality maps are definitions, not
equations, and are checked by the trivial-limit tests.

| Symbol | Meaning | Code name |
|---|---|---|
| $D_i$ | auxiliary fit domain of occupied $i$: aux shells within `aux_radius` of its Boys centroid | `aux_dom` |
| $V_i$ | virtual domain of $i$: virtuals within `virt_radius` of its centroid | `virt_dom` |
| $\tilde D_i,\ \tilde V_i$ | unions over $i$'s surviving partners | strip rows and columns |
| $D_{ij} = D_i\cup D_j$ | pair fit domain | `pair_dom` |
| $C_{ij} = V_i\cup V_j$ | candidate virtuals of a pair | `pair_cands` |
| $A_i$ | strip of occupied $i$: raw $(P\vert ia)$ for $P\in D_{ij}$, $a\in C_{ij}$, unwhitened | `Strip` |
| $V_{DD}$ | the metric restricted to $D_{ij}$ | `vdd` |
| $W_k = L^{-1}A_k$ | whitened strip, $V_{DD} = LL^{\mathsf T}$ | `ws` |

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/lmp2_direct.rs::amplitude_lmp2_direct`, `ferric-mp2/src/lmp2_direct.rs::assemble_ragged_direct_local` |
| CLI | `[local] scheme = "amplitude-threshold"`, `eps`, `integral_direct = true`, with `aux_radius`, `virt_radius`, `ao_tail`, `schwarz_skip`, `batch_merge`, `gate_cal`, `virt_schwarz_kappa` (`examples/alkane8-rimp2-local-direct.toml`) |
| Python | `run_rimp2(..., local="amplitude-threshold", eps=..., integral_direct=True)` |

## 13.2.1 The domain-local fit

For a pair $(i,j)$ the fitted integral over its domain is the RI product of
section 10.1 restricted to $D_{ij}$:

$$ J_{ia,jb} = \sum_{P,Q\in D_{ij}} (P\vert ia)\,[V_{DD}^{-1}]_{PQ}\,(Q\vert jb) = (W_i^{\mathsf T}W_j)_{ab}, \qquad V_{DD} = LL^{\mathsf T},\quad W_k = L^{-1}A_k . \tag{13.5} $$

The right-hand form is how the code evaluates it: one Cholesky factor per pair
domain, one triangular solve per distinct occupied in the group, and one
product per pair, with no explicit inverse. The same value follows from the
Dunlap robust form when one domain and one kernel serve the whole pair. The
metric and the three-index integrals use the same operator, which is why the
two forms coincide and why the fit error is quadratic, as in section 10.1.

> **Verified equation 13.5** · the whitened Gram product and the robust form
>
> With $V = LL^{\mathsf T}$ for a generic lower-triangular $3\times3$ factor,
> $W_i^{\mathsf T}W_j = A_i^{\mathsf T}V^{-1}A_j$ in symbols (claim L1), and
> the robust form $A_i^{\mathsf T}c_j + c_i^{\mathsf T}A_j - c_i^{\mathsf T}Vc_j$ with $c = V^{-1}A$ equals
> the Gram form (claim L2), so the Dunlap correction cancels identically in one
> shared domain.
>
> | | |
> |---|---|
> | Proof | `scripts/local_mp2_fit_pno_sympy.py`, claims L1 and L2 |
> | Negative control | the runner replaces $W_i^{\mathsf T}W_j$ by $W_i^{\mathsf T}W_i$ (claim L1 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/lmp2_direct.rs::{fit_domain_group,whiten,chol_lower}` |
> | Code test | `trivial_limit_matches_global_domain_fit_and_canonical` in `crates/ferric-mp2/tests/lmp2_direct.rs` |
>
> <!-- verified: id=lmp2.domain-fit; script=scripts/local_mp2_fit_pno_sympy.py; claim=L1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/lmp2_direct.rs::{fit_domain_group,whiten,chol_lower}; rust_test=ferric-mp2/tests/lmp2_direct.rs::trivial_limit_matches_global_domain_fit_and_canonical -->

The product is the whole of the pair step.

<!-- excerpt: ferric-mp2/src/lmp2_direct.rs::fit_domain_group -->
```rust
let g = if i == j {
    w_i.t().dot(&w_i)
} else {
    w_i.t().dot(&ws[&j].select(cij))
};
```

| Code | Equation |
|---|---|
| `w_i` | $W_i$ restricted to the candidate columns $C_{ij}$ |
| `ws[&j].select(cij)` | $W_j$ on the same columns |
| `g` | $J_{ia,jb}$ over $a,b\in C_{ij}$, passed to the $\varepsilon$ mask of section 13.1 (`mask_pair`) |

## 13.2.2 The algorithm

| Stage | Work | Function |
|---|---|---|
| 1 | the pair gate of 13.1, giving surviving pairs and per-occupied partner lists | `ferric-mp2/src/lmp2_amplitude.rs::pair_gate_keep` |
| 2 | the maps: $D_i$, $V_i$ and each orbital's AO-shell support (`ao_tail`, on the per-shell maximum $\lvert C\rvert$; truncation is exactly zeroing small coefficients) | `ferric-mp2/src/lmp2_direct.rs::build_locality_maps` |
| 3 | occupieds grouped by nearest atom to their Boys centroid; per batch, evaluate only the $(\mu\nu\vert P)$ shell triples needed and half-transform into per-occupied strips $A_i$ over $\tilde D_i\times\tilde V_i$ | `ferric-mp2/src/lmp2_direct.rs::collect_strips`, `ferric-mp2/src/lmp2_direct.rs::eval_batch` |
| 4 | the metric, only for $(P\vert Q)$ shell pairs inside some $\tilde D_i^2$. This is exact for the domain-local fit, which never reads a metric element outside a pair domain | `ferric-mp2/src/lmp2_direct.rs::strips_and_metric` |
| 5 | per pair-domain group: Cholesky once, whiten each distinct occupied once, $J_{ij} = W_i^{\mathsf T}W_j$, the Eq-8 mask, the `PairBlock` | `ferric-mp2/src/lmp2_direct.rs::fit_pairs` |
| 6 | PCG, tolerance policy and Hylleraas energy, unchanged | section 13.1 |

Every pair $(i,j)$ reads its $D_i\cup D_j$ rows and $V_i\cup V_j$ columns from
the two strips, so a strip is shared by all of an orbital's partners. The strips
are unwhitened: they hold the same raw values as the global tensor, only for
the rows and columns that survive.

**Variants of stage 3.** Per-atom batches re-evaluate shared shell
triples. `batch_merge` merges several nearest-atom batches along the long axis,
which evaluates fewer triples at the cost of larger scratch slabs, and agrees with
$k=1$ to the float-reassociation floor, not bitwise. An alternative pass
(`Eri3Pass::AuxMajor`) takes one task per auxiliary shell over the union of the
supports that need it, so each needed triple is evaluated once. Both build the
same strips to round-off.

## 13.2.3 The exactness anchor and the artifact hypothesis

These were written before any sweep was run, following the experimental protocol
of the repository.

**Trivial limit.** With every map a no-op (`virt_radius = None`, `ao_tail = 0`,
`aux_radius` at least $10^5$ Bohr, no gate, `schwarz_skip = 0`, $\varepsilon = 0$)
the energy must reproduce the global-tensor domain-fit path of the same radius
and canonical RI-MP2, to the CG tolerance. The test has one mutation arm per
map.

**Artifact hypothesis.** If the locality is real, then at fixed radii and
thresholds the extra error against the $\varepsilon$-only path stays flat as an
alkane grows, and strip sizes saturate past the onset. If a map is mis-constructed,
by mixing two orbital index sets, exactness will instead demand radii that track the
molecular diameter, as a domain mask that mixed two index sets did in an earlier
AO-sparse Laplace implementation (section 12.4). The two
predictions differ, so a size sweep can tell them apart.

## 13.2.4 Memory and determinism

- The peak three-index working set is the strips, reported as
  `DirectStats::strip_bytes_total`, with mean and maximum strip rows and columns
  (`strip_rows_mean`, `strip_cols_max`). The strip sizes are the locality claim,
  so they are reported rather than inferred.
- The stage-3 integral-slab scratch is capped globally
  (`DirectConfig::scratch_budget_bytes`, 2 GiB by default), shared across the batch
  workers: each worker slabs under budget over the number of workers, so the
  co-resident scratch never exceeds the cap. A per-worker budget would be a budget
  on nothing.
- The opt-in exact reference still forms the global tensor, so only the
  reference makes a run not reduced-cost.

## 13.2.5 Options that change the equation

| Option | Trivial limit | CLI default | Effect |
|---|---|---|---|
| `aux_radius` | at least $10^5$ Bohr (global fit) | 10.0 Bohr | auxiliary fit domain |
| `virt_radius` | all virtuals | 12.0 Bohr | virtual domain on dipole centroids |
| `ao_tail` | 0.0 keeps every shell | 1e-3 | AO-shell support threshold |
| `schwarz_skip` | 0.0 evaluates every triple (bit-identical to a run without the cut) | 1e-5 | Cauchy–Schwarz cut on batch triples, $\sqrt{\max(P\vert P)}\,Q(\mu\nu) <$ the threshold, with both factors from the run's operator |
| `gate_cal` | none (gate off) | none | the pair gate of equation 13.4 |
| `virt_schwarz_kappa` | none (off) | none | $\varepsilon$-linked virtual-candidate screen: keep $a$ if $q_{ia}q^{\max}_j \ge \kappa\varepsilon$ or the mirror |
| `batch_merge` | 1 | 4 | batches merged per integral pass |

The Rust `DirectConfig` defaults are the trivial ones where the table lists a
trivial limit, and differ from the CLI defaults. The CLI defaults are in
`site/src/reference/input.md`, section `[local]`.

## 13.2.6 What is measured

| Check | Reference | Agreement | Test |
|---|---|---|---|
| $\varepsilon = 0$, trivial maps, integral-direct: H2O, n-butane / cc-pVDZ, 6-31G (cc-pvdz-ri) | PySCF `DFMP2`, same auxiliary basis | at most $1.38\times10^{-11}$ Ha over the 24 comparisons of the validation page, bar $10^{-10}$ Ha | `crates/ferric-mp2/tests/validation_lmp2_amplitude.rs`, `crates/ferric-mp2/tests/lmp2_direct.rs` |
| finite $\varepsilon$, n-butane / cc-pVDZ | PySCF `DFMP2` | recovers 99.98%, 99.65%, 95.0% at $\varepsilon = 10^{-5}$, $10^{-4}$, $10^{-3}$ with production maps, against 99.98%, 99.67%, 95.0% in core | same |

<!-- figures: site/src/reference/validation.md "Local RI-MP2 (`rimp2` with `[local] scheme = \"amplitude-threshold\"`)" -->

**Cost and scaling (provisional).** On n-alkanes C20 to C48 (6-31G with
cc-pVDZ-RI, $\varepsilon = 10^{-4}$, frozen carbon cores, a calibrated pair gate),
the correlation stage grew as about $N^{1.24}$ with the erfc kernel and $N^{1.4}$
with the Coulomb kernel, fitted to the last three sizes. The integral-direct path
was about at parity with canonical RI-MP2 at C20 (about 1.1 times slower) and about
6 times faster at C32 (5.7 to 6.3 times). This is a fit to **three points** on **one
family of molecules in one basis**. It is not shown to be linear, and it is not
measured on three-dimensional or diffuse systems. It also excludes the dense
$O(N^3)$ VV-HV virtual-space build (`t_spaces_s`), whose cost the fitted exponents
do not cover. Local MP2 without `integral_direct` still builds the global
tensor and makes no scaling claim. The measurement is `bench_direct_alkane_series`
in `crates/ferric-mp2/tests/lmp2_direct.rs`.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/src/lmp2_direct.rs` module documentation, `fit_domain_group`, `DirectConfig` | the pipeline, the maps, the exactness anchor and the artifact hypothesis |
| `scripts/local_mp2_fit_pno_sympy.py`, claims L1, L2 | verified equation 13.5 |
| `site/src/methods/mp2.md`, "Exact and local MP2" | the user-facing description and the measured scaling |
| `site/src/reference/input.md`, `[local]` | the CLI defaults |
| `site/src/reference/validation.md` | the anchor row and the known limits |
