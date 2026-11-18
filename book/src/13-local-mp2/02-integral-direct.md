# The integral-direct path {#integral-direct}

<!-- status: ready; registry: local-mp2 -->

The method of section @sec:amplitude-lmp2 builds the global three-index tensor, of shape
$(N_{\mathrm{aux}}, n_on_v)$, before it truncates anything. That tensor is
$O(N^3)$ in memory, and the whitening product behind it is $O(N^5)$ in
time. Dropping amplitudes afterwards cannot reduce a cost that has already
been paid. Yet a charge distribution $\phi_i\phi_a$ from a localized pair
lives near $i$ and $a$, and only auxiliary functions nearby can describe
it. The integral-direct path uses this. It evaluates only the three-index
integrals that the retained pairs need, half-transforms them into one small
strip per occupied orbital, and fits each pair in a local auxiliary basis.
The solve and the energy are the unchanged ragged machinery of 13.1.

The section derives the local fit (@sec:integral-direct-domain-local-fit), walks through the stages and
the truncation each locality map makes (@sec:integral-direct-algorithm-each-map-truncates), and then states the
exactness anchor, the cost of the stages and what is measured.

The construction is
also documented in the module header of `crates/ferric-mp2/src/lmp2_direct.rs`
and in the user guide. The fit identities are machine-checked below. The
locality maps are definitions, not equations, and the trivial-limit tests
check them.

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
| CLI | `[local] scheme = "amplitude-threshold"`, `eps`, `integral_direct = true`, with `aux_radius`, `virt_radius`, `ao_tail`, `schwarz_skip`, `batch_merge`, `gate_cal`, `virt_schwarz_kappa`, `fock_screen` (`examples/alkane8-rimp2-local-direct.toml`) |
| Python | `run_rimp2(..., local="amplitude-threshold", eps=..., integral_direct=True)` |

## The domain-local fit {#integral-direct-domain-local-fit}

Density fitting (section @sec:closed-shell-ri-mp2) expands $\phi_i\phi_a$ in the auxiliary
functions. The best coefficients in the Coulomb metric are
$c = V^{-1}A$, where $A_{P,a} = (P\vert ia)$ and $V_{PQ} = (P\vert Q)$.
For pair $(i,j)$, restrict that fit to the functions in $D_{ij}$, the
auxiliary shells near either orbital. Substitute the restricted
coefficients into the Coulomb product of the two fitted densities:

$$ J_{ia,jb} = \sum_{P,Q\in D_{ij}} (P\vert ia)\,[V_{DD}^{-1}]_{PQ}\,(Q\vert jb) = (W_i^{\mathsf T}W_j)_{ab}, \qquad V_{DD} = LL^{\mathsf T},\quad W_k = L^{-1}A_k . \label{eq:lmp2-domain-fit} $$

The right-hand form follows in one line. Factor $V_{DD} = LL^{\mathsf T}$,
so $V_{DD}^{-1} = L^{-\mathsf T}L^{-1}$, and group each $L^{-1}$ with its
strip: $A_i^{\mathsf T}L^{-\mathsf T}L^{-1}A_j = (L^{-1}A_i)^{\mathsf T}(L^{-1}A_j)$.
That is the same Cholesky whitening as section @sec:closed-shell-ri-mp2, applied to one domain.

Check the limit. When $D_{ij}$ holds every auxiliary function, $V_{DD} = V$
and (@eq:lmp2-domain-fit) is the global RI integral of equation (@eq:closed-shell-ri-mp2-4).

Why one domain for the whole pair, rather than one per orbital? Dunlap's
robust form, $A_i^{\mathsf T}c_j + c_i^{\mathsf T}A_j - c_i^{\mathsf T}V_{DD}c_j$
with $c = V_{DD}^{-1}A$, is the fit whose error is quadratic in the error of
the fitted densities. When one domain and one kernel serve both orbitals,
substitute $c$: each of the three terms equals $A_i^{\mathsf T}V_{DD}^{-1}A_j$,
so the sum is the Gram form (@eq:lmp2-domain-fit). The metric and the three-index
integrals use the same operator, which is why the two forms coincide and
why the fit error is quadratic, as in section @sec:closed-shell-ri-mp2.

> **Verified equation @eq:lmp2-domain-fit** · the whitened Gram product and the robust form
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

**What the restriction does to the values.** A fit in fewer functions
describes each density less completely. For a diagonal element this has a
sign. Write $\bar D$ for the auxiliary functions outside $D$, $S$ for the
Schur complement $V_{\bar D\bar D} - V_{\bar DD}V_{DD}^{-1}V_{D\bar D}$ and
$y = A_{\bar D} - V_{\bar DD}V_{DD}^{-1}A_D$. Block elimination of $V$ gives

$$ A^{\mathsf T}V^{-1}A - A_D^{\mathsf T}V_{DD}^{-1}A_D = y^{\mathsf T}S^{-1}y \ \ge\ 0 , $$

because $S$ is positive definite. So each fitted $(ia\vert ia)$ can only
fall when the domain shrinks. Off-diagonal elements have no such sign, and
neither has the energy error. The fit radius changes $J$ values, so the
one-sided argument of @sec:amplitude-lmp2-amplitude-equations-energy does not cover it.

**The code.** `ferric-mp2/src/lmp2_direct.rs::fit_domain_group` handles all
pairs that share one domain $D_{ij}$. It factors $V_{DD}$ once, whitens
each distinct occupied of the group once with a triangular solve, and then
forms one product per pair, with no explicit inverse. The product is the
whole of the pair step. The excerpt is copied unchanged:

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
| `g` | $J_{ia,jb}$ over $a,b\in C_{ij}$, passed to the $\varepsilon$ mask of section @sec:amplitude-lmp2 (`mask_pair`) |

Grouping by domain pays because pairs repeat domains: on the C20–C48 maps
there are 6–9 times more surviving pairs than distinct pair domains, <!-- pending: re-measure at virt_radius=20/ao_tail=1e-4 --> so one
Cholesky factor serves several pairs (`fit_pairs`).

## The algorithm and what each map truncates {#integral-direct-algorithm-each-map-truncates}

| Stage | Work | Function |
|---|---|---|
| 1 | the pair gate of 13.1, giving surviving pairs and per-occupied partner lists | `ferric-mp2/src/lmp2_amplitude.rs::pair_gate_keep` |
| 2 | the maps: $D_i$, $V_i$ and each orbital's AO-shell support (`ao_tail`, on the per-shell maximum $\lvert C\rvert$; truncation is exactly zeroing small coefficients) | `ferric-mp2/src/lmp2_direct.rs::build_locality_maps` |
| 3 | occupieds grouped by nearest atom to their Boys centroid; per batch, evaluate only the $(\mu\nu\vert P)$ shell triples needed and half-transform into per-occupied strips $A_i$ over $\tilde D_i\times\tilde V_i$ | `ferric-mp2/src/lmp2_direct.rs::collect_strips`, `ferric-mp2/src/lmp2_direct.rs::eval_batch` |
| 4 | the metric, only for $(P\vert Q)$ shell pairs inside some $\tilde D_i^2$. This is exact for the domain-local fit, which never reads a metric element outside a pair domain | `ferric-mp2/src/lmp2_direct.rs::strips_and_metric` |
| 5 | per pair-domain group: Cholesky once, whiten each distinct occupied once, $J_{ij} = W_i^{\mathsf T}W_j$, the Eq-8 mask, the `PairBlock` | `ferric-mp2/src/lmp2_direct.rs::fit_pairs` |
| 6 | PCG, tolerance policy and Hylleraas energy, unchanged | section @sec:amplitude-lmp2 |

**Strips.** A pair $(i,j)$ reads rows $D_i\cup D_j$ and columns
$V_i\cup V_j$ from the two strips $A_i$ and $A_j$. So a strip must cover
the union over all of an orbital's partners, $\tilde D_i\times\tilde V_i$,
and then one strip serves every pair that $i$ belongs to. The strips are
unwhitened: they hold the same raw values as the global tensor, only for
the rows and columns that survive. Whitening happens per group in stage 5,
because $L$ depends on the pair domain.

**Two kinds of map.** As in @sec:amplitude-lmp2-options-change-equation, a map either removes amplitudes and
leaves every kept $J$ value alone, or it changes the $J$ values.

- *Pattern maps* can only raise the energy relative to the same run without
  them, up to the solver error of @sec:amplitude-lmp2-accurate-solve, because $E_H$ is then minimized
  over fewer amplitudes on a swap-closed pattern.
  - The pair gate (`gate_cal`), as in 13.1.
  - The virtual domains (`virt_radius`). The candidate set
    $C_{ij} = V_i\cup V_j$ restricts rows and columns of pair $(i,j)$ alike,
    so the pattern stays swap-closed, and selecting columns of $W_i$ and
    $W_j$ leaves each kept product unchanged.
  - The candidate screen (`virt_schwarz_kappa` $=\kappa$). With $\varepsilon > 0$
    it removes $a$ from $C_{ij}$ unless $q_{ia}q^{\max}_j \ge \kappa\varepsilon$
    or $q_{ja}q^{\max}_i \ge \kappa\varepsilon$, where
    $q_{ia} = \sqrt{A_{ia}^{\mathsf T}V_{D_iD_i}^{-1}A_{ia}}$ is the fitted
    diagonal in $i$'s own domain. On a global Gram fit this is the exact
    Cauchy–Schwarz screen of @sec:amplitude-lmp2-algorithm for $\kappa \le 1$. On the domain-local fit
    it is not a proven bound; the `DirectConfig` documentation records it as
    measured escape-free at $\kappa \le 1$, both operators, on C8. A
    $\kappa$ above 1 drops elements that would pass the Eq-8 test.
- *Value maps* change the integrals, and their energy error has no proven
  sign.
  - The auxiliary domains (`aux_radius`), @sec:integral-direct-domain-local-fit.
  - The AO support (`ao_tail`). An AO shell enters an orbital's support
    when its largest $\lvert C\rvert$ is at least the threshold. Leaving a
    shell out is the same as setting those coefficients to zero in the half
    transform, applied alike to both strips of a pair. The Fock blocks are
    untouched.
  - The triple cut (`schwarz_skip`). A batch triple is skipped when its
    Cauchy–Schwarz bound $\sqrt{\max(P\vert P)}\,Q(\mu\nu)$ falls below the
    threshold, with $Q(\mu\nu) = \sqrt{(\mu\nu\vert\mu\nu)}$ and both
    factors from the run's operator. The bound is rigorous for each skipped
    raw integral $(\mu\nu\vert P)$; the aux factor is the maximum over the
    shell, so the cut stays conservative at shell granularity.

The pair candidates of stage 5 come from `ferric-mp2/src/lmp2_direct.rs::pair_candidates`.
This is the body of its per-pair closure, unchanged except for the
comments:

```rust
let cij = sorted_union([virt_dom[i].iter().copied(), virt_dom[j].iter().copied()]); // C_ij = V_i ∪ V_j
match (qvirt, kappa_eps) {
    (Some((q, qmax)), Some(th)) => cij                         // κ screen on, th = κ ε
        .into_iter()
        .filter(|&a| {
            let qia = q[i][strips[i].col_pos[a]];               // q_ia
            let qja = q[j][strips[j].col_pos[a]];               // q_ja
            qia * qmax[j] >= th || qja * qmax[i] >= th          // keep a if either bound reaches κ ε
        })
        .collect(),
    _ => cij,                                                   // screen off or ε = 0
}
```

The screen only ever shrinks $V_i\cup V_j$. So every candidate still lies
in both strips' columns, and a missing column position is a construction
error that panics.

**Variants of stage 3.** Per-atom batches re-evaluate shell triples that
neighbouring batches also need. `batch_merge` merges several nearest-atom
batches along the molecule's long axis, which evaluates fewer triples at
the cost of larger scratch slabs. It agrees with $k=1$ to the
float-reassociation floor, not bitwise, because the per-occupied transform
runs over a larger batch union. An alternative pass
(`Eri3Pass::AuxMajor`) takes one task per auxiliary shell over the union of
the supports that need it, so each needed triple is evaluated once. Both
build the same strips to round-off.

## How the two kinds of map combine at ε = 0 {#integral-direct-map-errors-eps0}

At $\varepsilon = 0$ the Eq-8 mask keeps every amplitude. Any difference
from canonical RI-MP2 then comes from the maps. A pattern map and a value map
leave different fingerprints on the energy, and with both on, the two errors
do not add.

Write the Hylleraas functional as $E(t) = 2\langle t, J\rangle + \langle t, \mathcal A t\rangle$,
with $\mathcal A$ the positive definite Fock superoperator of @sec:amplitude-lmp2
and $J$ the fitted pair integrals. Its minimum is $E^\star = -\langle J, \mathcal A^{-1}J\rangle$
at $t^\star = -\mathcal A^{-1}J$.

**A pattern map.** Split the amplitudes into a kept block $K$ and a dropped
block $D$, solve on $K$ only, $t_K = -\mathcal A_{KK}^{-1}J_K$, and eliminate
the $D$ block of $\mathcal A$. With
$S = \mathcal A_{DD} - \mathcal A_{DK}\mathcal A_{KK}^{-1}\mathcal A_{KD}$,

$$ E_K - E^\star = t^{\star\mathsf T}_D\, S\, t^\star_D \ \ge\ 0 . \label{eq:lmp2-map-pattern} $$

$S$ is positive definite because $\mathcal A$ is, so the error of
`virt_radius` is never negative. It is second order in the dropped exact
amplitudes. Check the limit: with nothing dropped, $D$ is empty and the error
is zero.

**A value map.** `aux_radius`, `ao_tail` and `schwarz_skip` replace $J$ by
$J + \delta J$ and leave the pattern and $\mathcal A$ alone. `ao_tail` zeroes
coefficients only inside the strip transform; the Fock blocks use the full
orbitals. Expand $E^\star(J+\delta J) = -\langle J+\delta J, \mathcal A^{-1}(J+\delta J)\rangle$
and use $t^\star = -\mathcal A^{-1}J$:

$$ E^\star(J + \delta J) - E^\star(J) = 2\langle t^\star, \delta J\rangle - \langle \delta J, \mathcal A^{-1}\delta J\rangle . \label{eq:lmp2-map-value} $$

The leading term is linear in $\delta J$ and has the sign of
$\langle t^\star, \delta J\rangle$, which no map controls. A negative
$\varepsilon = 0$ error therefore comes from a value map. Truncating an
orbital's AO expansion also leaves the occupied and virtual spaces of the
strips no longer orthogonal.

**Both on.** Solve the restricted problem with the perturbed integrals and
subtract $E^\star$. Adding and subtracting $E_K(J)$ splits the result into
(@eq:lmp2-map-pattern) and (@eq:lmp2-map-value) restricted to $K$:

$$ E_K(J + \delta J) - E^\star = t^{\star\mathsf T}_D S\, t^\star_D + 2\langle t_K, \delta J_K\rangle - \langle \delta J_K, \mathcal A_{KK}^{-1}\delta J_K\rangle . \label{eq:lmp2-map-both} $$

The value map is charged only on the kept block, and against $t_K$, not
$t^\star_K$. So the error with both maps on is not the sum of the two errors
measured one map at a time. A small combined error can be a positive pattern
error cancelling a value error of the other sign. Such a cancellation depends
on the geometry, so it does not carry over from one conformer to the next.
To attribute an error, switch one map on at a time from the trivial limit,
and also switch one map off at a time from the production setting. The
ignored probe `probe_map_ablation` in `crates/ferric-mp2/tests/lmp2_map_ablation.rs`
does both, and `probe_tail_mass` reports the coefficient mass `ao_tail` drops
per orbital.

> **Verified equations @eq:lmp2-map-pattern, @eq:lmp2-map-value and @eq:lmp2-map-both** · how pattern and value maps enter the energy
>
> On a generic symmetric $2+1$ block, $E_K - E^\star$ equals the Schur form
> (claim 1). On a generic $3\times3$ model the value-map expansion is exact
> through both orders (claim 2). On the $2+1$ block, both maps together give
> the kept-block form (claim 3). A random rational positive definite model
> checks all three by brute-force solves. It finds value-map errors of both
> signs and a combined error that differs from the one-at-a-time sum.
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_direct_maps_sympy.py`, claims 1, 2 and 3 |
> | Negative control | the runner flips the sign of the Schur coupling term (claim 1 fails), drops the quadratic term (claim 2 fails), and charges the value map on $t^\star_K$ (claim 3 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/lmp2_direct.rs::{build_locality_maps,pair_candidates}` |
> | Code test | `trivial_maps_match_canonical_aconf20_sto3g` in `crates/ferric-mp2/tests/lmp2_map_ablation.rs` |
>
> <!-- verified: id=lmp2.direct-maps; script=scripts/lmp2_direct_maps_sympy.py; claim=1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/lmp2_direct.rs::{build_locality_maps,pair_candidates}; rust_test=ferric-mp2/tests/lmp2_map_ablation.rs::trivial_maps_match_canonical_aconf20_sto3g -->

The proof covers the algebra of a fixed $\mathcal A$ and $J$. It does not
bound the size of $\delta J$ for any map. The sizes come from measurement.

## The exactness anchor and the artifact hypothesis {#integral-direct-exactness-anchor-artifact-hypothesis}

The repository's experimental protocol requires both of these to be stated
before any sweep is run, and the module documentation of `lmp2_direct.rs`
records them that way.

**Trivial limit.** Set every map to a no-op: `virt_radius = None`,
`ao_tail = 0`, `aux_radius` at least $10^5$ Bohr, no gate,
`schwarz_skip = 0`, and $\varepsilon = 0$. Then the energy must reproduce
the global-tensor domain-fit path of the same radius, and canonical RI-MP2,
to the CG tolerance. The test has one mutation arm per map, so each map is
shown to change the energy when it is broken.

n-Butane in cc-pVDZ shows the trivial limit and then the default maps
(the `589b50832` build). Python has no way to pass `virt_radius = None`, so
the first run keeps the 20 Bohr default; butane is 10.5 Bohr between its
farthest atoms, so that radius drops nothing. The trivial run matches exact
RI-MP2 to $4.93\times10^{-13}$ Ha. With the default maps (`virt_radius` 20,
`ao_tail` $10^{-4}$) the direct recovery agrees with the in-core one to
$10^{-4}$ percentage points at each $\varepsilon$:

```python
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/alkane_4.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

exact = ferric.run_rimp2(mol, bs, aux, frozen_core=0).mp2_corr
trivial = ferric.run_rimp2(mol, bs, aux, frozen_core=0, local="amplitude-threshold", eps=0.0,
                           integral_direct=True, aux_radius=1e5, ao_tail=0.0, schwarz_skip=0.0,
                           batch_merge=1).mp2_corr
print(f"exact {exact:.12f}  direct trivial maps {trivial:.12f}  diff {trivial - exact:+.2e} Ha")
for eps in (1e-5, 1e-4, 1e-3):
    incore = ferric.run_rimp2(mol, bs, aux, frozen_core=0, local="amplitude-threshold", eps=eps).mp2_corr
    direct = ferric.run_rimp2(mol, bs, aux, frozen_core=0, local="amplitude-threshold", eps=eps,
                              integral_direct=True).mp2_corr
    print(f"eps={eps:7.0e}  in-core {100 * incore / exact:8.4f}%  direct (default maps) {100 * direct / exact:8.4f}%")
```

```text
exact -0.599645225098  direct trivial maps -0.599645225098  diff +4.93e-13 Ha
eps=  1e-05  in-core  99.9804%  direct (default maps)  99.9805%
eps=  1e-04  in-core  99.6708%  direct (default maps)  99.6708%
eps=  1e-03  in-core  94.9942%  direct (default maps)  94.9943%
```

**Artifact hypothesis.** The two outcomes of a size sweep are predicted in
advance:

- If the locality is real, then at fixed radii and thresholds the extra
  error against the $\varepsilon$-only path stays flat as an alkane grows,
  and the strip sizes saturate past the onset.
- If a map is mis-constructed by mixing two orbital index sets, exactness
  will instead demand radii that track the molecular diameter. A domain
  mask that mixed two index sets did this in an AO-sparse Laplace
  implementation (section @sec:local-laplace-sos).

The two predictions differ, so the sweep can tell them apart.

## Memory and determinism {#integral-direct-memory-determinism}

- The peak three-index working set is the strips. It is reported as
  `DirectStats::strip_bytes_total`, with mean and maximum strip rows and
  columns (`strip_rows_mean`, `strip_cols_max`). The strip sizes are the
  locality claim, so the code reports them rather than leaving them to be
  inferred.
- The stage-3 integral-slab scratch has one global cap
  (`DirectConfig::scratch_budget_bytes`, 2 GiB by default), shared by the
  batch workers. Each worker slabs under the budget divided by the number
  of workers, so the co-resident scratch never exceeds the cap. A
  per-worker budget would not bound the total.
- The opt-in exact reference still forms the global tensor. Only the
  reference makes a run not reduced-cost.

## Options that change the equation {#integral-direct-options-change-equation}

| Option | Trivial limit | CLI default | Effect |
|---|---|---|---|
| `aux_radius` | at least $10^5$ Bohr (global fit) | 10.0 Bohr | auxiliary fit domain |
| `virt_radius` | all virtuals | 20.0 Bohr | virtual domain on dipole centroids |
| `ao_tail` | 0.0 keeps every shell | 1e-4 | AO-shell support threshold |
| `schwarz_skip` | 0.0 evaluates every triple (bit-identical to a run without the cut) | 1e-5 | Cauchy–Schwarz cut on batch triples, $\sqrt{\max(P\vert P)}\,Q(\mu\nu) <$ the threshold, with both factors from the run's operator |
| `gate_cal` | none (gate off) | none | the pair gate of equation @eq:amplitude-lmp2-4 |
| `virt_schwarz_kappa` | none (off) | none | $\varepsilon$-linked virtual-candidate screen: keep $a$ if $q_{ia}q^{\max}_j \ge \kappa\varepsilon$ or the mirror |
| `fock_screen` | `false` (plain Eq-8 mask) | `true` (κ = 1) | the Fock-propagated screen of section @sec:amplitude-lmp2-fock-screen; inert at $\varepsilon = 0$ |
| `batch_merge` | 1 | 4 | batches merged per integral pass |

The Rust `DirectConfig` defaults are the trivial limits for every option in
the table except `aux_radius_bohr`, which defaults to 10.0 Bohr as in the
CLI. They differ from the CLI defaults for `virt_radius`, `ao_tail`,
`schwarz_skip` and `batch_merge`. The CLI defaults are in
`site/src/reference/input.md`, section `[local]`.

## What is measured {#integral-direct-measured}

| Check | Reference | Agreement | Test |
|---|---|---|---|
| $\varepsilon = 0$, trivial maps, integral-direct: H2O, n-butane / cc-pVDZ, 6-31G (cc-pvdz-ri) | PySCF `DFMP2`, same auxiliary basis | at most $1.38\times10^{-11}$ Ha over the 24 comparisons of the validation page, bar $10^{-10}$ Ha | `crates/ferric-mp2/tests/validation_lmp2_amplitude.rs`, `crates/ferric-mp2/tests/lmp2_direct.rs` |
| finite $\varepsilon$, n-butane / cc-pVDZ | PySCF `DFMP2` | recovers 99.98%, 99.65%, 95.0% at $\varepsilon = 10^{-5}$, $10^{-4}$, $10^{-3}$ with production maps, against 99.98%, 99.67%, 95.0% in core <!-- pending: re-measure at virt_radius=20/ao_tail=1e-4 --> | same |

<!-- figures: site/src/reference/validation.md "Local RI-MP2 (`rimp2` with `[local] scheme = \"amplitude-threshold\"`)" -->

<!-- pending: re-measure at virt_radius=20/ao_tail=1e-4 --> On this molecule the integral-direct maps change the recovery by 0.02
percentage points at $\varepsilon = 10^{-4}$, and not at the reported
precision at the other two thresholds.

**Cost and scaling (provisional).** <!-- pending: re-measure at virt_radius=20/ao_tail=1e-4 --> On n-alkanes C20 to C48 (6-31G with
cc-pVDZ-RI, $\varepsilon = 10^{-4}$, frozen carbon cores, a calibrated pair
gate), the correlation stage grew as about $N^{1.24}$ with the erfc kernel
and $N^{1.4}$ with the Coulomb kernel, fitted to the last three sizes. The
integral-direct path was about at parity with canonical RI-MP2 at C20
(about 1.1 times slower) and about 6 times faster at C32 (5.7 to 6.3
times).

The limits of that measurement:

- It is a fit to **three points** on **one family of molecules in one
  basis**. It is not shown to be linear.
- It is not measured on three-dimensional or diffuse systems.
- It excludes the dense $O(N^3)$ VV-HV virtual-space build (`t_spaces_s`),
  whose cost the fitted exponents do not cover.

Local MP2 without `integral_direct` still builds the global tensor and
makes no scaling claim. The measurement is `bench_direct_alkane_series` in
`crates/ferric-mp2/tests/lmp2_direct.rs`.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/src/lmp2_direct.rs` module documentation, `fit_domain_group`, `DirectConfig` | the pipeline, the maps, the exactness anchor and the artifact hypothesis |
| `scripts/local_mp2_fit_pno_sympy.py`, claims L1, L2 | verified equation @eq:lmp2-domain-fit |
| `scripts/lmp2_direct_maps_sympy.py`, claims 1 to 3 | verified equations @eq:lmp2-map-pattern, @eq:lmp2-map-value, @eq:lmp2-map-both |
| `crates/ferric-mp2/tests/lmp2_map_ablation.rs` | the $\varepsilon = 0$ anchors at octane and C$_{20}$H$_{42}$ size, and the ablation probes |
| `site/src/methods/mp2.md`, "Exact and local MP2" | the user-facing description and the measured scaling |
| `site/src/reference/input.md`, `[local]` | the CLI defaults |
| `site/src/reference/validation.md` | the anchor row and the known limits |
